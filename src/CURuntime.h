
#pragma once

#include <string>
#include <unordered_map>
#include <map>
#include <vector>
#include <stacktrace>

#include "OrbitControls.h"
#include "unsuck.hpp"

#include "glm/common.hpp"

#include "cuda.h"
#include "cuda_runtime.h"

using namespace std;

struct CURuntime{

	inline static CUdevice device;
	
	CURuntime(){
		
	}

	static int getNumSMs(){
		CUdevice device;
		int numSMs;
		cuCtxGetDevice(&device);
		cuDeviceGetAttribute(&numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);

		return numSMs;
	}


	static void assertCudaSuccess(CUresult result, std::stacktrace trace = std::stacktrace::current()){

		if(result == CUDA_SUCCESS) return;

		println("ERROR: CUDA result != CUDA_SUCCESS.");

		const char* name = nullptr;
		const char* desc = nullptr;
		cuGetErrorName(result, &name);
		cuGetErrorString(result, &desc);

		println(stderr, "CUDA error {} ({}): {}\n ",
			int(result),
			name ? name : "unknown",
			desc ? desc : "unknown");

		println("{}", trace);

		__debugbreak();

		exit(6123453456);
	}

};