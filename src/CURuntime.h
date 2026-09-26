#pragma once

#include <stacktrace>

#include <cuda_runtime.h>

#include "unsuck.hpp"

struct CURuntime{

	inline static int device = 0;

	static void assertCudaSuccess(cudaError_t result, std::stacktrace trace = std::stacktrace::current()){

		if(result == cudaSuccess) return;

		println("ERROR: CUDA result != cudaSuccess.");

		println(stderr, "CUDA error {} ({}): {}\n ",
			int(result),
			cudaGetErrorName(result),
			cudaGetErrorString(result));

		println("{}", trace);

		__debugbreak();

		exit(6123453456);
	}

};
