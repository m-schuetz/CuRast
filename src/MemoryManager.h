#pragma once 

#include <mutex>
#include <vector>
#include <string>

#include "cuda.h"
#include "CURuntime.h"
#include "CudaVirtualMemory.h"
#include "VKRenderer.h"

struct MemoryManager{

	struct Allocation {
		string label;
		CUdeviceptr cptr;
		int64_t size;
	};

	inline static mutex mtx;
	inline static vector<Allocation> allocations;
	inline static vector<CudaVirtualMemory*> cudaVirtual;

	inline static CudaVirtualMemory* allocVirtualCuda(uint64_t virtualCapacity, string label = "none"){

		CudaVirtualMemory* memory = CudaVirtualMemory::create(virtualCapacity, label);
		cudaVirtual.push_back(memory);

		return memory;
	}

	inline static CUdeviceptr alloc(int64_t size, string label){
		CUdeviceptr cptr;

		auto result = cuMemAlloc(&cptr, size);
		CURuntime::assertCudaSuccess(result);

		lock_guard<mutex> lock(mtx);
		Allocation entry = { label, cptr, size};
		allocations.push_back(entry);

		return cptr;
	}

	

	static void free(CUdeviceptr cptr) {
		if (cptr == 0) {
			println("WARNING: attempted to CURuntime::free a null ptr. Already freed?");
			return;
		}

		lock_guard<mutex> lock(mtx);

		int index = -1;
		for(int i = 0; i < allocations.size(); i++){
			if(allocations[i].cptr == cptr){
				index = i;
				cuMemFree(cptr);
			}
		}

		if(index != -1){
			allocations.erase(allocations.begin() + index);
		}
	}

	static int64_t getByteSize(CUdeviceptr cptr){
		for(int i = 0; i < allocations.size(); i++){
			if(allocations[i].cptr == cptr){
				return allocations[i].size;
			}
		}

		return 0;
	}

	static int64_t getTotalAllocatedMemory(){

		int64_t bytes = 0;

		for(int i = 0; i < allocations.size(); i++){
			bytes += allocations[i].size;
		}

		for(auto memory : cudaVirtual){
			bytes += memory->comitted;
		}

		return bytes;
	}

};