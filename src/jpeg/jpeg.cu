#define CUB_DISABLE_BF16_SUPPORT

// === required by GLM ===
#define GLM_FORCE_CUDA
#define CUDA_VERSION 12000
namespace std {
	using size_t = ::size_t;
};
// =======================

#include <curand_kernel.h>
#include <cooperative_groups.h>

#include "./libs/glm/glm/glm.hpp"
#include "./libs/glm/glm/gtc/matrix_transform.hpp"
#include "./libs/glm/glm/gtc/matrix_access.hpp"
#include "./libs/glm/glm/gtx/transform.hpp"
#include "./libs/glm/glm/gtc/quaternion.hpp"

// #include "./utils.cuh"
#include "./BitReaderGPU.cuh"
#include "../kernels/HostDeviceInterface.h"
#include "./HashMap.cuh"
#include "./dct.cuh"
#include "./JptInterface.cuh"

namespace cg = cooperative_groups;

using glm::ivec2;
using glm::i8vec4;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::ivec3;
using glm::ivec4;
using glm::mat4;

constexpr uint32_t UV_BITS = 12;
constexpr float UV_FACTOR = 1 << UV_BITS;

template<typename T>
__device__
T clamp(T value, T min, T max){

	if(value < min) return min;
	if(value > max) return max;

	return value;
}

struct Decoded {
	vec2 uv;
	int id;
};


#define RGBA(r, g, b) ((uint32_t(255) << 24) | (uint32_t(r) << 16) | (uint32_t(g) << 8) | uint32_t(b))

void idct8x8_optimized(float* block, int thread) {
	auto cuda_block = cg::this_thread_block();

	cuda_block.sync();

	// Perform IDCT on rows
	CUDAsubroutineInplaceIDCTvector(&block[thread * 8], 1);

	cuda_block.sync();

	// Perform IDCT on columns
	CUDAsubroutineInplaceIDCTvector(&block[thread], 8);
}



int decodeHuffman_warpwide(BitReaderGPU& bit_reader, const HuffmanTable& huffman_table) {

	auto block = cg::this_thread_block();
	auto warp = cg::tiled_partition<32>(block);

	uint32_t code_peek = bit_reader.peek16Bit2();

	// Each thread loads one packed entry: high 16 bits = code length, low 16 bits = key.
	// One cache line (128 bytes) covers all 32 lanes — half the memory traffic vs two
	// separate codelengths[]+huffman_keys[] loads. Break early when all lengths are 0.
	for (int i = 0; i < 6; i++) {
		int codeIndex = i * 32 + warp.thread_rank();
		uint32_t p = __ldg(&huffman_table.packed[codeIndex]);
		uint32_t bit_length = p >> 16;
		uint32_t key = p & 0xFFFF;
		uint32_t code = code_peek >> (16 - bit_length);
		bool isValidCode = (bit_length > 0) && (key == code);

		uint32_t mask = warp.ballot(isValidCode);

		if (mask > 0) {
			int winningLane = __ffs(mask) - 1;

			bit_length = warp.shfl(bit_length, winningLane);
			uint32_t huffmanValue = warp.shfl(__ldg(&huffman_table.huffman_values[codeIndex]), winningLane);

			bit_reader.advance(bit_length);

			return huffmanValue;
		}

		// No valid codes remain in this chunk — stop scanning
		if (warp.ballot(bit_length > 0) == 0) break;
	}

	return -1;
}

int DecodeNumber(int code, int bits) {
	int l = 1 << (code - 1);
	if (bits >= l) {
		return bits;
	}
	else {
		return bits - (2 * l - 1);
	}
}


void decodeCoefficients_warpwide(
	BitReaderGPU& bit_reader,
	const HuffmanTable& huffman_table,
	float* sh_coefficients
) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	auto warp = cg::tiled_partition<32>(block);
	int i = 1;
#pragma unroll
	while (i < 64) {
		int ac_code = decodeHuffman_warpwide(bit_reader, huffman_table);

		if (ac_code == 0) {
			break;
		}
		int size = ac_code;
		if (ac_code > 15) {
			int run_length = (ac_code >> 4) & 0xF;
			size = ac_code & 0xF;
			i += run_length;
		}

		if (i >= 64) break;

		int ac_value = bit_reader.read_bits(size);
		ac_value = DecodeNumber(size, ac_value);

		sh_coefficients[i++] = ac_value;

		// warp.sync();
	}
}


__constant__ int dezigzag_order[64] = {
	0, 1, 8, 16, 9, 2, 3, 10,
 17, 24, 32, 25, 18, 11, 4, 5,
 12, 19, 26, 33, 40, 48, 41, 34,
 27, 20, 13, 6, 7, 14, 21, 28,
 35, 42, 49, 56, 57, 50, 43, 36,
 29, 22, 15, 23, 30, 37, 44, 51,
 58, 59, 52, 45, 38, 31, 39, 46,
 53, 60, 61, 54, 47, 55, 62, 63
};

// fetch bit-offset to the mcu from the indexing table
int calculate_datastart(int mcu, const uint32_t* mcu_index) {
	int packed_index = (mcu / 9) * 5;
	int offset_within_packed = mcu % 9;
	int absolute_offset = mcu_index[packed_index];
	if (offset_within_packed == 0) {
		return absolute_offset;
	}

	int rel_index = offset_within_packed - 1; 
	int word = mcu_index[packed_index + 1 + (rel_index / 2)];
	int shift = (rel_index % 2 == 0) ? 16 : 0;
	int relative_offset = (word >> shift) & 0xFFFF;
	return absolute_offset + relative_offset;
}

uint16_t get12bit(const uint8_t* buf, int idx) {
	int group = idx / 2;
	int byte_idx = group * 3;

	if ((idx % 2) == 0) {
		return (buf[byte_idx]) | ((buf[byte_idx + 1] & 0x0F) << 8);
	}
	else {
		return ((buf[byte_idx + 1] >> 4) & 0x0F) | (buf[byte_idx + 2] << 4);
	}
}


extern "C" __global__
void kernel_decode_420(
	uint32_t* toDecode,
	uint32_t numToDecode,
	uint32_t* decoded,
	Texture* texturesData,
	// TextureData* texturesData,
	HashMap decodedMcuMap,
	uint32_t* TBSlots,
	uint32_t firstAvailableTBSlotsIndex
) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	auto warp = cg::tiled_partition<32>(block);

	// 64 threads per block, 2 MCUs per block.
	// Warp 0 (threads  0-31) → MCU 0
	// Warp 1 (threads 32-63) → MCU 1
	// Both warps decode their MCU's Huffman stream in parallel.
	int mcu_in_block = warp.meta_group_rank();  // 0 or 1
	int thread = warp.thread_rank();       // 0..31

	uint32_t global_mcu_index = grid.block_rank() * 2 + mcu_in_block;
	bool is_valid = global_mcu_index < numToDecode;

	__shared__ float   sh_coefficients[384 * 2];
	__shared__ float   sh_dezigzagged[384 * 2];
	__shared__ uint8_t sh_data[512 * 2];
	__shared__ int     sh_tbslot[2];

	int coeff_off = mcu_in_block * 384;
	int dezzig_off = mcu_in_block * 384;
	int data_off = mcu_in_block * 512;

	uint32_t toDecode_packed = is_valid ? toDecode[global_mcu_index] : 0;
	uint32_t mcu;
	uint32_t texture_id;
	uint32_t mipLevel;
	unpack_mcuidx_textureidx_miplevel(toDecode_packed, &mcu, &texture_id, &mipLevel);
	texture_id = texture_id + mipLevel;
	const Texture& textureData = texturesData[texture_id];

	int datastart = is_valid ? calculate_datastart(mcu, textureData.mcuPositions) : 0;
	int datastartbit = datastart % 8;

	for (int i = 0; i < 12; i++) {
		sh_data[data_off + thread + 32 * i] = is_valid
			? __ldg(((const uint8_t*)textureData.data) + datastart / 8 + thread + 32 * i)
			: 0;
		sh_coefficients[coeff_off + thread + 32 * i] = 0;
	}

	block.sync();

	BitReaderGPU bit_reader(&sh_data[data_off], datastartbit);

	// Each warp independently decodes its MCU's DC coefficients
	if (is_valid) {
		int previousDC = (bit_reader.read_bits(12) & 0x0fff) - 2048;
		sh_coefficients[coeff_off + 0] = previousDC;
		for (int i = 1; i < 4; i++) {
			const HuffmanTable& huffmanTable = textureData.huffmanTables[0];
			int dc_value = decodeHuffman_warpwide(bit_reader, huffmanTable);
			if (dc_value > 0) {
				int dc_difference = bit_reader.read_bits(dc_value);
				dc_value = DecodeNumber(dc_value, dc_difference);
			}
			sh_coefficients[coeff_off + i * 64] = dc_value + previousDC;
			previousDC = dc_value + previousDC;
		}
		sh_coefficients[coeff_off + 64 * 4] = (bit_reader.read_bits(12) & 0x0fff) - 2048;
		sh_coefficients[coeff_off + 64 * 5] = (bit_reader.read_bits(12) & 0x0fff) - 2048;
	}

	// Each warp independently decodes its MCU's AC coefficients
	if (is_valid) {
		for (int i = 0; i < 6; i++) {
			int huff_index = (i <= 3) ? 0 : (i - 3);
			const HuffmanTable& huffmanTable = textureData.huffmanTables[3 + huff_index];
			decodeCoefficients_warpwide(bit_reader, huffmanTable, &sh_coefficients[coeff_off + i * 64]);
		}
	}

	block.sync();
	if (is_valid) {
		float q1_lo = __ldg(&textureData.quanttables[0].values[thread]);
		float q1_hi = __ldg(&textureData.quanttables[0].values[thread + 32]);
		float q2_lo = __ldg(&textureData.quanttables[1].values[thread]);
		float q2_hi = __ldg(&textureData.quanttables[1].values[thread + 32]);

		for (int i = 0; i < 4; i++) {
			sh_dezigzagged[dezzig_off + dezigzag_order[thread] + i * 64] = sh_coefficients[coeff_off + thread + 64 * i] * q1_lo;
			sh_dezigzagged[dezzig_off + dezigzag_order[thread + 32] + i * 64] = sh_coefficients[coeff_off + thread + 32 + 64 * i] * q1_hi;
		}
		for (int i = 0; i < 2; i++) {
			sh_dezigzagged[dezzig_off + dezigzag_order[thread] + i * 64 + 256] = sh_coefficients[coeff_off + thread + 256 + 64 * i] * q2_lo;
			sh_dezigzagged[dezzig_off + dezigzag_order[thread + 32] + i * 64 + 256] = sh_coefficients[coeff_off + thread + 32 + 256 + 64 * i] * q2_hi;
		}
	}
	block.sync();

	for (int pass = 0; pass < 2; pass++) {
		int t = thread + pass * 32;
		if (t / 8 < 6)
			CUDAsubroutineInplaceIDCTvector(&sh_dezigzagged[dezzig_off + (t / 8) * 64 + (t % 8) * 8], 1);
	}

	block.sync();
	for (int pass = 0; pass < 2; pass++) {
		int t = thread + pass * 32;
		if (t / 8 < 6)
			CUDAsubroutineInplaceIDCTvector(&sh_dezigzagged[dezzig_off + (t / 8) * 64 + t % 8], 8);
	}

	block.sync();

	// Acquire a texture block cache slot (thread 0 of each warp)
	if (thread == 0 && is_valid) {
		uint32_t slotIndex = firstAvailableTBSlotsIndex + global_mcu_index;
		uint32_t tbslot = TBSlots[slotIndex];
		uint32_t visFlag = 0b0000'0001; // mark as visible & newly cached.
		uint32_t value = (visFlag << 24) | tbslot;
		uint64_t entry = (uint64_t(toDecode_packed) << 32) | uint64_t(value);

		bool alreadyExists = false;
		int location = 0;
		decodedMcuMap.set(toDecode_packed, 0, &location, &alreadyExists);
		atomicExch(&decodedMcuMap.entries[location], entry);

		sh_tbslot[mcu_in_block] = tbslot;
	}

	block.sync();

	// Write decoded texels. 2 passes × 4 blocks = 256 texels per MCU.
	if (is_valid) {
		for (int pass = 0; pass < 2; pass++) {
			int t = thread + pass * 32;
			for (int i = 0; i < 4; i++) {
				uint8_t* rgba = (uint8_t*)&decoded[sh_tbslot[mcu_in_block] * 256 + t + 64 * i];
				float y = sh_dezigzagged[dezzig_off + t + 64 * i] + 128.0f;

				int chroma_x = (t % 8) / 2;
				int chroma_y = (t / 8) / 2;
				int chroma_index = chroma_y * 8 + chroma_x + i / 2 * 4 * 8 + i % 2 * 4;

				float cb = sh_dezigzagged[dezzig_off + chroma_index + 64 * 4];
				float cr = sh_dezigzagged[dezzig_off + chroma_index + 64 * 5];

				rgba[0] = clamp(y + 1.402f * cr, 0.0f, 255.0f);
				rgba[1] = clamp(y - 0.344136f * cb - 0.714136f * cr, 0.0f, 255.0f);
				rgba[2] = clamp(y + 1.772f * cb, 0.0f, 255.0f);
				rgba[3] = 255;
			}
		}
	}
	
}


// This kernel is used to indirectly launch kernel_decode_420 from the GPU, 
// so that we don't have to memcpy <toDecodeCounter> to host before launching it. 
extern "C" __global__
void kernel_launch_decode(
	uint32_t* toDecodeCounter,
	uint32_t* TBSlots,
	uint32_t* TBSlotsCounter,
	uint32_t* toDecode,
	uint32_t* decoded,
	Texture* texturesData,
	HashMap decodedMcuMap
) {
	auto grid = cg::this_grid();

	if (grid.thread_rank() == 0) {

		uint32_t numToDecode = *toDecodeCounter;
		uint32_t numBlocks = (numToDecode + 1) / 2;  // 2 MCUs per block
		uint32_t firstAvailableTBSlotsIndex = *TBSlotsCounter;
		kernel_decode_420 << <numBlocks, 64 >> > (
			toDecode,
			numToDecode,
			decoded,
			texturesData,
			decodedMcuMap,
			TBSlots,
			*TBSlotsCounter
			);

		*TBSlotsCounter = (*toDecodeCounter) + (*TBSlotsCounter);
	}
}



extern "C" __global__
void kernel_init_availableMcuSlots(
	uint32_t* TBSlots,
	uint32_t* TBSlotsCounter,
	uint32_t numDecodedMcuCapacity
) {

	auto grid = cg::this_grid();

	if(grid.thread_rank() >= numDecodedMcuCapacity) return;

	TBSlots[grid.thread_rank()] = grid.thread_rank();
}

extern "C" __global__
void kernel_update_cache(
	HashMap decodedMcuMap_source,
	HashMap decodedMcuMap_target,
	uint32_t* TBSlots,
	uint32_t* TBSlotsCounter,
	bool freezeCache
) {

	auto grid = cg::this_grid();

	if(grid.thread_rank() >= decodedMcuMap_source.capacity) return;

	uint64_t entry = decodedMcuMap_source.entries[grid.thread_rank()];
	uint32_t key = entry >> 32;
	uint32_t value = entry & 0xffffffff;
	uint32_t visFlag  = (value >> 24) & 0xff;
	uint32_t slot = (value >>  0)  & 0xffffff;

	bool isMcuVisible = visFlag != 0;
	bool isNewlyDecoded = (visFlag == 0b00000001);

	// Note: If cache is frozen: Only remove newly decoded entries 
	// but preserve previously cached entries, including those that are currently invisible. 

	if(entry == HashMap::EMPTYENTRY) return;

	// Put slot to decoded texture block back in pool of slots
	auto remove = [&](){
		uint32_t old = atomicSub(TBSlotsCounter, 1);
		uint32_t slotIndex = old - 1;

		if(slotIndex > JPEG_NUM_DECODED_MCU_CAPACITY){
			printf("ERROR: slotIndex > JPEG_NUM_DECODED_MCU_CAPACITY \n");
			return;
		}

		TBSlots[slotIndex] = slot;
	};

	// Replicate entry in new hash map, which will be used in the next frame. 
	auto preserve = [&](){
		int location;
		bool alreadyExists;
		uint32_t newVal = (0x00 << 24) | slot;
		decodedMcuMap_target.set(key, newVal, &location, &alreadyExists);
	};

	if(freezeCache){
		if(isNewlyDecoded){
			remove();
		}else{
			preserve();
		}
	}else{
		if(isMcuVisible == 0){
			remove();
		}else{
			preserve();
		}
	}
}


