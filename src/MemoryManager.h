#pragma once

#include <mutex>
#include <vector>
#include <string>
#include <print>

#include <cuda_runtime.h>

#include "unsuck.hpp"
#include "CURuntime.h"

using std::println;
using std::mutex;
using std::lock_guard;

// Device buffer that grows on demand, e.g. for framebuffers that are resized with the window.
// Growing reallocates the buffer, i.e., the pointer changes and the previous content is discarded.
struct CudaBuffer{

	string label;
	void* ptr = nullptr;
	uint64_t size = 0;

	// Makes sure that the buffer can hold at least <requestedSize> bytes.
	void resize(uint64_t requestedSize){
		if(requestedSize <= size) return;

		if(ptr != nullptr){
			CURuntime::assertCudaSuccess(cudaFree(ptr));
		}

		CURuntime::assertCudaSuccess(cudaMalloc(&ptr, requestedSize));
		size = requestedSize;
	}

};

struct MemoryManager{

	struct Allocation {
		string label;
		void* ptr;
		int64_t size;
	};

	inline static mutex mtx;
	inline static vector<Allocation> allocations;
	inline static vector<CudaBuffer*> buffers;

	inline static CudaBuffer* allocBuffer(uint64_t size, string label = "none"){

		CudaBuffer* buffer = new CudaBuffer();
		buffer->label = label;
		buffer->resize(size);
		buffers.push_back(buffer);

		return buffer;
	}

	inline static void* alloc(int64_t size, string label){
		void* ptr = nullptr;

		CURuntime::assertCudaSuccess(cudaMalloc(&ptr, size));

		lock_guard<mutex> lock(mtx);
		Allocation entry = { label, ptr, size};
		allocations.push_back(entry);

		return ptr;
	}

	static void free(void* ptr) {
		if (ptr == nullptr) {
			println("WARNING: attempted to MemoryManager::free a null ptr. Already freed?");
			return;
		}

		lock_guard<mutex> lock(mtx);

		int index = -1;
		for(int i = 0; i < allocations.size(); i++){
			if(allocations[i].ptr == ptr){
				index = i;
				cudaFree(ptr);
			}
		}

		if(index != -1){
			allocations.erase(allocations.begin() + index);
		}
	}

	static int64_t getByteSize(void* ptr){
		for(int i = 0; i < allocations.size(); i++){
			if(allocations[i].ptr == ptr){
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

		for(auto buffer : buffers){
			bytes += buffer->size;
		}

		return bytes;
	}

};
