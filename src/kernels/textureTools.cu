#define CUB_DISABLE_BF16_SUPPORT

// === required by GLM ===
// #define GLM_FORCE_CUDA
// #define GLM_FORCE_NO_CTOR_INIT
// #define CUDA_VERSION 12000
// namespace std {
// 	using size_t = ::size_t;
// };
// =======================

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
#include "../cuda_to_hip.h"
#else
#include <cooperative_groups.h>
#endif

namespace cg = cooperative_groups;

// calls function <f> <size> times
// calls are distributed over all available threads
template<typename Function>
__device__
void processRange(int size, Function&& f){

	uint32_t totalThreadCount = blockDim.x * gridDim.x;
	
	int itemsPerThread = size / totalThreadCount + 1;

	for(int i = 0; i < itemsPerThread; i++){
		int block_offset  = itemsPerThread * blockIdx.x * blockDim.x;
		int thread_offset = itemsPerThread * threadIdx.x;
		int index = block_offset + thread_offset + i;

		if(index >= size){
			break;
		}

		f(index);
	}
}


// Computes one mip level from the previous one. The grid-wide ordering that
// the cooperative kernel below provides via grid.sync() between levels is
// supplied by the host instead: each level is a separate ordinary launch, and
// the kernel boundary is the grid-wide barrier. See computeMipMap().
#if defined(USE_HIP)
extern "C" __global__
void kernel_computeMipMapLevel(
	uint32_t* data, int level,
	int64_t source_width, int64_t source_height, int64_t source_pixelOffset
){
	int64_t target_width = (source_width + 2 - 1) / 2;
	int64_t target_height = (source_height + 2 - 1) / 2;
	int64_t target_numPixels = target_width * target_height;
	int64_t target_pixelOffset = source_pixelOffset + source_width * source_height;

	processRange(target_numPixels, [&](int64_t threadID){
		int64_t target_x = threadID % target_width;
		int64_t target_y = threadID / target_width;

		if(target_x >= target_width) return;
		if(target_y >= target_height) return;

		uint32_t color = 0xff0000ff;
		uint8_t* rgba = (uint8_t*)&color;
		if(level == 1) color = 0x4f3ed5;
		if(level == 2) color = 0x436df4;
		if(level == 3) color = 0x61aefd;
		if(level == 4) color = 0x8be0fe;
		if(level == 5) color = 0xbfffff;
		if(level == 6) color = 0x98f5e6;
		if(level == 7) color = 0xa4ddab;
		if(level == 8) color = 0xa5c266;
		if(level == 9) color = 0xbd8832;

		int32_t sR = 0;
		int32_t sG = 0;
		int32_t sB = 0;
		int32_t sA = 0;

		auto fetchAdd = [&](int source_x, int source_y){
			if(source_x >= source_width) return;
			if(source_y >= source_height) return;

			int64_t source_pixelID = source_x + source_width * source_y;

			uint32_t sample = data[source_pixelOffset + source_pixelID];
			uint8_t* rgba = (uint8_t*)&sample;

			sR += rgba[0];
			sG += rgba[1];
			sB += rgba[2];
			sA += rgba[3];
		};

		fetchAdd(2 * target_x + 0, 2 * target_y + 0);
		fetchAdd(2 * target_x + 0, 2 * target_y + 1);
		fetchAdd(2 * target_x + 1, 2 * target_y + 0);
		fetchAdd(2 * target_x + 1, 2 * target_y + 1);

		rgba[0] = sR / 4;
		rgba[1] = sG / 4;
		rgba[2] = sB / 4;
		rgba[3] = sA / 4;

		int64_t target_pixelID = target_x + target_width * target_y;

		data[target_pixelOffset + target_pixelID] = color;
	});
}
#else
extern "C" __global__
void kernel_computeMipMap(uint32_t* data, int width, int height){

	auto grid = cg::this_grid();

	// if(grid.thread_rank() > 1) return;

	// printf("========================= kernel_computeMipMap ======================\n");

	int64_t levels = log2(float(max(width, height)));

	// if(grid.thread_rank() == 0){
	// 	printf("===============================\n");
	// 	printf("# MIP MAPS\n");
	// 	printf("===============================\n");
	// 	printf("levels: %llu \n", levels);
	// }

	int64_t source_width = width;
	int64_t source_height = height;
	int64_t source_pixelOffset = 0;

	for(int level = 1; level < levels; level++){

		int64_t target_width = (source_width + 2 - 1) / 2;
		int64_t target_height = (source_height + 2 - 1) / 2;
		int64_t target_numPixels = target_width * target_height;
		int64_t target_pixelOffset = source_pixelOffset + source_width * source_height;

		processRange(target_numPixels, [&](int64_t threadID){
			int64_t target_x = threadID % target_width;
			int64_t target_y = threadID / target_width;

			if(target_x >= target_width) return;
			if(target_y >= target_height) return;

			uint32_t color = 0xff0000ff;
			uint8_t* rgba = (uint8_t*)&color;
			if(level == 1) color = 0x4f3ed5;
			if(level == 2) color = 0x436df4;
			if(level == 3) color = 0x61aefd;
			if(level == 4) color = 0x8be0fe;
			if(level == 5) color = 0xbfffff;
			if(level == 6) color = 0x98f5e6;
			if(level == 7) color = 0xa4ddab;
			if(level == 8) color = 0xa5c266;
			if(level == 9) color = 0xbd8832;

			int32_t sR = 0;
			int32_t sG = 0;
			int32_t sB = 0;
			int32_t sA = 0;

			auto fetchAdd = [&](int source_x, int source_y){
				if(source_x >= source_width) return;
				if(source_y >= source_height) return;

				int64_t source_pixelID = source_x + source_width * source_y;

				uint32_t sample = data[source_pixelOffset + source_pixelID];
				uint8_t* rgba = (uint8_t*)&sample;

				sR += rgba[0];
				sG += rgba[1];
				sB += rgba[2];
				sA += rgba[3];
			};

			fetchAdd(2 * target_x + 0, 2 * target_y + 0);
			fetchAdd(2 * target_x + 0, 2 * target_y + 1);
			fetchAdd(2 * target_x + 1, 2 * target_y + 0);
			fetchAdd(2 * target_x + 1, 2 * target_y + 1);

			rgba[0] = sR / 4;
			rgba[1] = sG / 4;
			rgba[2] = sB / 4;
			rgba[3] = sA / 4;

			int64_t target_pixelID = target_x + target_width * target_y;

			data[target_pixelOffset + target_pixelID] = color;

			// if(target_x < 2 && target_y < 2){
			// 	printf("level %d, pixel[%llu, %llu] = %llu. pixelID: %llu, target_pixelOffset %llu \n",
			// 		level, target_x, target_y, color, target_pixelID, target_pixelOffset
			// 		);
			// }

		});

		grid.sync();

		source_width = target_width;
		source_height = target_height;
		source_pixelOffset = target_pixelOffset;
	}

}
#endif


__host__
void computeMipMap(uint32_t* data, int width, int height){

	int threads_per_block = 128;

	int device = 0;
#if defined(USE_HIP)
	hipGetDevice(&device);

	hipDeviceProp_t deviceProp;
	hipGetDeviceProperties(&deviceProp, device);

	int numBlocksPerSm;
	hipOccupancyMaxActiveBlocksPerMultiprocessor(
		&numBlocksPerSm,
		kernel_computeMipMapLevel,
		threads_per_block,
		0
	);
	uint32_t max_supported_blocks = deviceProp.multiProcessorCount * numBlocksPerSm;

	dim3 dimGrid(max_supported_blocks);
	dim3 dimBlock(threads_per_block);

	// Build the mip pyramid one level at a time with ordinary (non-cooperative)
	// launches. Each launch reads the previous level and writes the next; the
	// kernel boundary provides the grid-wide synchronization that the
	// cooperative kernel's grid.sync() provides on CUDA. This avoids
	// hipLaunchCooperativeKernel, which fails on gfx1201/ROCm 7.14/Windows and
	// would otherwise leave the mip levels ungenerated.
	int64_t levels = (int64_t)log2(float(width > height ? width : height));
	int64_t source_width = width;
	int64_t source_height = height;
	int64_t source_pixelOffset = 0;

	for(int level = 1; level < levels; level++){
		hipLaunchKernelGGL(
			kernel_computeMipMapLevel,
			dimGrid, dimBlock, 0, 0,
			data, level, source_width, source_height, source_pixelOffset
		);

		int64_t target_width = (source_width + 2 - 1) / 2;
		int64_t target_height = (source_height + 2 - 1) / 2;
		int64_t target_pixelOffset = source_pixelOffset + source_width * source_height;

		source_width = target_width;
		source_height = target_height;
		source_pixelOffset = target_pixelOffset;
	}

	hipDeviceSynchronize();
#else
	cudaGetDevice(&device);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);

	int numBlocksPerSm;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(
		&numBlocksPerSm,
		kernel_computeMipMap,
		threads_per_block,
		0
	);
	uint32_t max_supported_blocks = deviceProp.multiProcessorCount * numBlocksPerSm;

	dim3 dimGrid(max_supported_blocks);
	dim3 dimBlock(threads_per_block);

	void* kernel_args[] = {
		&data,
		&width,
		&height,
	};

	cudaError_t err = cudaLaunchCooperativeKernel(
		(const void*)kernel_computeMipMap,
		dimGrid,
		dimBlock,
		kernel_args,
		0, // Shared memory (in bytes)
		0  // Stream (0 for default stream)
	);

	if (err != cudaSuccess) {
		fprintf(stderr, "Cooperative launch failed: %s\n", cudaGetErrorString(err));
	}
#endif
}