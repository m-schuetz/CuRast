
#pragma once

#include <string>
#include <unordered_map>
#include <map>
#include <vector>
#include <stacktrace>
#include <iostream>

#include "OrbitControls.h"
#include "unsuck.hpp"

#include "glm/common.hpp"

#include "cuda_to_hip.h"

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


	// Kept guarded: the HIP runtime hipGetErrorName/String return the string
	// directly, whereas the CUDA driver cuGetErrorName/String use an out-param
	// (a signature the compat header does not bridge), and the messages differ.
#if defined(USE_HIP)
	static void assertCudaSuccess(hipError_t result, std::stacktrace trace = std::stacktrace::current()){

		if(result == hipSuccess) return;

		println("ERROR: HIP result != hipSuccess.");

		const char* name = hipGetErrorName(result);
		const char* desc = hipGetErrorString(result);

		std::cerr << "HIP error " << int(result) << " ("
			<< (name ? name : "unknown") << "): "
			<< (desc ? desc : "unknown") << "\n";

		std::cerr << to_string(trace) << std::endl;

		__builtin_trap();

		exit(6123453456);
	}
#else
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

		std::cerr << to_string(trace) << std::endl;

		__debugbreak();

		exit(6123453456);
	}
#endif

};