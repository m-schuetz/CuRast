#pragma once

constexpr int64_t JPEG_NUM_DECODED_MCU_CAPACITY = 700'072;
constexpr int64_t JPEG_BYTES_PER_DECODED_MCU = 16 * 16 * 4;
constexpr int64_t JPEG_MAX_HEIGHT = 4096;
constexpr int64_t JPEG_MAX_WIDTH = 4096;

uint32_t uvToMCUIndex(int width, int height, float u, float v) {
	int tx = (int(u * width) % width);
	int ty = (int(v * height) % height);
	return tx / 16 + ty / 16 * width / 16;
}

uint32_t pack_mcuidx_textureidx_miplevel(uint32_t mcu, uint32_t texID, uint32_t mipLevel){
	
	// mcu: 20 bit
	// texID: 8 bit
	// mipLevel: 4 bit
	uint32_t packed = (mcu << 12) | texID << 4 | mipLevel;

	return packed;
}

void unpack_mcuidx_textureidx_miplevel(uint32_t packed, uint32_t* mcu, uint32_t* texID, uint32_t* mipLevel){

	*mcu      = (packed >> 12);
	*texID    = (packed >>  4) & 0xff;
	*mipLevel = (packed >>  0) & 0b1111;
}
