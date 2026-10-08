// High-quality shading of point clouds (CuRastSettings::highQualityShading). Instead of the color of the closest point,
// each pixel shows the average color of the points close to the closest one, which blends overlapping points:
// - kernel_clearPointBuffers resets the render target's point depth and accumulation buffers.
// - The point cloud kernels draw all points twice, see points.cuh: POINT_PASS_DEPTH writes the depth of the closest point
//   of each pixel into pointDepthbuffer, with 32 bit atomicMin. POINT_PASS_ACCUMULATE then sums up the colors and the
//   number of points up to HQ_POINT_DEPTH_RANGE behind it in pointAccumbuffer, with atomicAdd.
// - kernel_normalizePoints divides the sums of colors by the number of points, and writes the colors with the closest
//   depth into the colorbuffer, i.e., into the visibility buffer that meshes are drawn into afterwards.

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

// GLM detects CUDA via CUDA_VERSION from the driver API's cuda.h. The runtime API defines CUDART_VERSION.
#ifndef CUDA_VERSION
	#define CUDA_VERSION CUDART_VERSION
#endif
#define GLM_FORCE_CUDA
#define GLM_FORCE_NO_CTOR_INIT

#include "./glm/glm/glm.hpp"

#include "./HostDeviceInterface.h"
#include "../types.h"
#include "./kernels.h"
#include "../Timer.h"

extern "C" __global__
void kernel_clearPointBuffers(RenderTarget target) {
	u32 pixelID = blockIdx.x * blockDim.x + threadIdx.x;
	if(pixelID >= target.width * target.height) return;

	target.pointDepthbuffer[pixelID] = __float_as_uint(Infinity);
	target.pointAccumbuffer[2 * pixelID + 0] = 0;
	target.pointAccumbuffer[2 * pixelID + 1] = 0;
}

extern "C" __global__
void kernel_normalizePoints(RenderTarget target) {
	u32 pixelID = blockIdx.x * blockDim.x + threadIdx.x;
	if(pixelID >= target.width * target.height) return;

	u64 redGreen  = target.pointAccumbuffer[2 * pixelID + 0];
	u64 blueCount = target.pointAccumbuffer[2 * pixelID + 1];
	u64 count = blueCount >> 32;

	if(count == 0) return;

	// rounded to the nearest integer
	u32 r = ((redGreen  & 0xffffffff) + count / 2) / count;
	u32 g = ((redGreen  >> 32)        + count / 2) / count;
	u32 b = ((blueCount & 0xffffffff) + count / 2) / count;
	u32 color = r | (g << 8) | (b << 16) | (0xffu << 24);

	u64 udepth = target.pointDepthbuffer[pixelID];
	u64 fragment = udepth << 32 | color;

	// One thread per pixel, but the pixel may already contain something closer, e.g. from a mesh
	if(fragment < target.colorbuffer[pixelID]){
		target.colorbuffer[pixelID] = fragment;
	}
}

// ------------------------------------------------------------------------------------------------
// Host
// ------------------------------------------------------------------------------------------------

static bool registered =
	registerKernel("pointsHighQuality.cu", "kernel_clearPointBuffers", (const void*)kernel_clearPointBuffers) &&
	registerKernel("pointsHighQuality.cu", "kernel_normalizePoints", (const void*)kernel_normalizePoints);

void launch_clearPointBuffers(const RenderTarget& target){
	u32 numPixels = target.width * target.height;
	u32 blockSize = 256;
	u32 gridSize = (numPixels + blockSize - 1) / blockSize;

	auto start = Timer::recordCudaTimestamp();
	kernel_clearPointBuffers<<<gridSize, blockSize>>>(target);
	checkKernelLaunch("kernel_clearPointBuffers");
	Timer::recordDuration("kernel_clearPointBuffers", start, Timer::recordCudaTimestamp());
}

void launch_normalizePoints(const RenderTarget& target){
	u32 numPixels = target.width * target.height;
	u32 blockSize = 256;
	u32 gridSize = (numPixels + blockSize - 1) / blockSize;

	auto start = Timer::recordCudaTimestamp();
	kernel_normalizePoints<<<gridSize, blockSize>>>(target);
	checkKernelLaunch("kernel_normalizePoints");
	Timer::recordDuration("kernel_normalizePoints", start, Timer::recordCudaTimestamp());
}
