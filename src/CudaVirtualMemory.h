#pragma once

#include <print>
#include <mutex>
#include <stacktrace>

#include "cuda.h"
#include "unsuck.hpp"
#include "CURuntime.h"

using std::println;
using std::mutex;
using std::lock_guard;
using std::stacktrace;

// see https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/
struct CudaVirtualMemory{

	string label;
	uint64_t size = 0;
	uint64_t comitted = 0;
	uint64_t granularity = 0;
	CUdeviceptr cptr = 0;
	static inline bool compress = false;

	// Keeping track of allocated physical memory, so we can remap or free
	std::vector<CUmemGenericAllocationHandle> allocHandles;
	std::vector<uint64_t> allocHandleSizes;

	CudaVirtualMemory(){
		
	}

	~CudaVirtualMemory(){
		destroy();
	}

	void destroy(){

		// cuMemCreate          ->  cuMemRelease
		// cuMemMap             ->  cuMemUnmap
		// cuMemAddressReserve  ->  cuMemAddressFree 

		// TODO: cuMemUnmap ? 

		println("TODO: CudaVirtualMemory::destroy");

		// if(cptr == 0){
		// 	println("WARNING: tried to destroy virtual memory that was already destroyed.");
		// 	return;
		// }
		
		// for(auto handle : allocHandles){
		// 	cuMemRelease(handle); 
		// }
		// allocHandles.clear();

		// cuMemAddressFree(cptr, size);

		// cptr = 0;
	}

	// allocate potentially large amounts of virtual memory
	static CudaVirtualMemory* create(uint64_t virtualSize = 2'000'000'000, string label = "none") {

		CUdevice cuDevice;
		cuDeviceGet(&cuDevice, 0);
		
		CUmemAllocationProp prop = {};
		prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
		prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		prop.location.id   = cuDevice;
		if(CudaVirtualMemory::compress){
			prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
		}

		uint64_t granularity = 0;
		cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

		uint64_t padded_size = roundUp(virtualSize, granularity);

		// reserve lots of virtual memory
		CUdeviceptr cptr = 0;
		auto result = cuMemAddressReserve(&cptr, padded_size, 0, 0, 0);

		if(result != CUDA_SUCCESS){
			println("error {} while trying to reserve virtual memory.", int(result));
			exit(52457);
		}
		
		CudaVirtualMemory* memory = new CudaVirtualMemory();
		memory->size = padded_size;
		memory->granularity = granularity;
		memory->cptr = cptr;
		memory->comitted = 0;
		memory->label = label;

		return memory;
	}

	// commits <size> physical memory. 
	void commit(uint64_t requested_size){

		static mutex mtx;
		lock_guard<mutex> lock(mtx);

		int64_t padded_requested_size = roundUp(requested_size, granularity);
		int64_t required_additional_size = padded_requested_size - comitted;

		// Do we already have enough comitted memory?
		if(required_additional_size <= 0) return;
		if(size < comitted + required_additional_size){
			// TODO: reserve new virtual range and remap
			println("physically comitting beyond initial virtual range not yet implemented.");
			println("TODO: reserve new virtual range and remap");
			println("{}", stacktrace::current());
			exit(6235266);
		}

		CUdevice cuDevice;
		cuDeviceGet(&cuDevice, 0);

		CUmemAllocationProp prop = {};
		prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
		prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		prop.location.id   = cuDevice;
		if(CudaVirtualMemory::compress){
			prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
		}

		// create a little bit of physical memory
		CUmemGenericAllocationHandle allocHandle;
		auto result = cuMemCreate(&allocHandle, required_additional_size, &prop, 0);
		CURuntime::assertCudaSuccess(result);

		// and map the physical memory
		result = cuMemMap(cptr + comitted, required_additional_size, 0, allocHandle, 0); 
		CURuntime::assertCudaSuccess(result);

		// make the new memory accessible
		CUmemAccessDesc accessDesc = {};
		accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		accessDesc.location.id = cuDevice;
		accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
		result = cuMemSetAccess(cptr + comitted, required_additional_size, &accessDesc, 1);
		CURuntime::assertCudaSuccess(result);

		comitted += required_additional_size;
		allocHandles.push_back(allocHandle);
		allocHandleSizes.push_back(required_additional_size);
	}

	void memcopyHtoD(uint64_t offset, void* source, uint64_t size, stacktrace trace = stacktrace::current()){

		bool validRange = true;
		validRange = validRange && offset >= 0 && offset < comitted;
		validRange = validRange && (offset + size) < comitted;

		if(!validRange){
			println("ERROR: Attempted to memcpy to unallocated or uncomitted range.");
			println("    cptr:          {:15L}", cptr);
			println("    comitted:      {:15L}", comitted);
			println("    target offset: {:15L}", offset);
			println("    source size:   {:15L}", size);

			println("{}", trace);
			__debugbreak();
			exit(652345345);
		}

		CUresult result = cuMemcpyHtoD(cptr + offset, source, size);
		CURuntime::assertCudaSuccess(result, trace);

		// if(result != CUDA_SUCCESS){
		// 	println("cuMemcpyHtoD failed with error code {}", int(result));
		// 	println("{}", trace);
		// 	__debugbreak();
		// 	exit(6125234);
		// }
	}

};

