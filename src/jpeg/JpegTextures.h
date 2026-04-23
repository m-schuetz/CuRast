#pragma once

#include "cuda.h"

#include "kernels/HostDeviceInterface.h"
#include "CudaModularProgram.h"
#include "MemoryManager.h"
#include "JptInterface.cuh"

struct JpegTextures{

	HashMap* decodedMcuMap           = nullptr;
	HashMap* decodedMcuMap_tmp       = nullptr;
	CudaModularProgram* prog    = nullptr;
	// CUdeviceptr cptr_texture_pointer = 0;
	CUdeviceptr cptr_toDecode        = 0;
	CUdeviceptr cptr_toDecodeCounter = 0;
	CUdeviceptr cptr_decoded         = 0;
	CUdeviceptr cptr_TBSlots         = 0;
	CUdeviceptr cptr_TBSlotsCounter  = 0;

	JpegTextures(){
		prog = new CudaModularProgram({"./src/jpeg/jpeg.cu"});

		decodedMcuMap = new HashMap();
		decodedMcuMap->capacity = 1'000'003;
		decodedMcuMap->entries = (uint64_t*)MemoryManager::alloc(8 * decodedMcuMap->capacity, "decodedMcuMap->entries");

		decodedMcuMap_tmp = new HashMap();
		decodedMcuMap_tmp->capacity = decodedMcuMap->capacity;
		decodedMcuMap_tmp->entries = (uint64_t*)MemoryManager::alloc(8 * decodedMcuMap->capacity, "decodedMcuMap->capacity->entries");

		cuMemsetD8((CUdeviceptr)decodedMcuMap->entries, 0xff, 8 * decodedMcuMap->capacity);
		cuMemsetD8((CUdeviceptr)decodedMcuMap_tmp->entries, 0xff, 8 * decodedMcuMap->capacity);

		// cptr_texture_pointer = MemoryManager::alloc(sizeof(TextureData) * texturesData.size(), "cptr_texture_pointer");
		cptr_toDecode        = MemoryManager::alloc(sizeof(uint32_t) * JPEG_NUM_DECODED_MCU_CAPACITY, "cptr_toDecode");
		cptr_toDecodeCounter = MemoryManager::alloc(sizeof(uint32_t), "cptr_toDecodeCounter");
		cptr_decoded         = MemoryManager::alloc(JPEG_NUM_DECODED_MCU_CAPACITY * JPEG_BYTES_PER_DECODED_MCU, "cptr_decoded");
		cptr_TBSlots         = MemoryManager::alloc(sizeof(uint32_t) * JPEG_NUM_DECODED_MCU_CAPACITY, "cptr_TBSlots");
		cptr_TBSlotsCounter  = MemoryManager::alloc(sizeof(uint32_t), "cptr_TBSlotsCounter");

		{
			uint32_t capacity = JPEG_NUM_DECODED_MCU_CAPACITY;
			prog->launch("kernel_init_availableMcuSlots", {&cptr_TBSlots, &cptr_TBSlotsCounter, &capacity}, capacity,  0);

			cuMemsetD32(cptr_TBSlotsCounter, 0, 1);
		}
	}

	void updateCache(){
		cuMemsetD8((CUdeviceptr)decodedMcuMap_tmp->entries, 0xff, decodedMcuMap_tmp->capacity * 8);
		//bool freezeCache = editor->settings.freezeCache;
		bool freezeCache = false;
		prog->launch("kernel_update_cache", {
			decodedMcuMap, 
			decodedMcuMap_tmp, 
			&cptr_TBSlots,
			&cptr_TBSlotsCounter,
			&freezeCache
		}, decodedMcuMap->capacity);
		cuMemcpy((CUdeviceptr)decodedMcuMap->entries, (CUdeviceptr)decodedMcuMap_tmp->entries, decodedMcuMap_tmp->capacity * 8);
	}

};