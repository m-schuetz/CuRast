#pragma once

// Host functions that launch the CUDA kernels. They are implemented at the end of the .cu files in this directory.
//
// glm types and RenderTarget are passed by reference: With GLM_FORCE_CUDA, nvcc and the host compiler
// disagree on whether glm types are trivially copyable, which changes how they are passed by value.

#include <string>
#include <vector>
#include <cstdint>

#include <cuda_runtime.h>

#include "glm/glm.hpp"
#include "HostDeviceInterface.h"

// resolve.cu
void launch_dummy(uint32_t* data);
void launch_clearFramebuffer(uint64_t* framebuffer, uint32_t numPixels, uint32_t clearColor, float clearDepth);
void launch_resolveColorbufferToSurface(
	const RenderTarget& target, cudaSurfaceObject_t surface,
	int width, int height, int mouseX, int mouseY,
	bool enableEDL, bool showInset, uint32_t backgroundColor);

// laspoints.cu
void launch_drawLasPoints(
	const RenderTarget& target, uint8_t* points, uint64_t numPoints,
	uint32_t pointRecordSize, int32_t offset_rgb,
	const glm::vec3& scale, const glm::mat4& worldView);

// potreeFileRenderer.cu
void launch_drawPotreeFileNodes(const RenderTarget& target, PotreeNode* nodes, uint64_t numNodes);

// List of all kernels, e.g. to inspect their register and shared memory usage.
// Each .cu file registers its kernels during static initialization.
struct KernelInfo{
	std::string module;
	std::string name;
	const void* function;
};

inline std::vector<KernelInfo>& getKernelInfos(){
	static std::vector<KernelInfo> infos;
	return infos;
}

inline bool registerKernel(std::string module, std::string name, const void* function){
	getKernelInfos().push_back({module, name, function});
	return true;
}

// Reports errors of the most recent kernel launch
inline void checkKernelLaunch(const char* name){
	cudaError_t result = cudaGetLastError();
	if(result != cudaSuccess){
		fprintf(stderr, "ERROR: failed to launch kernel %s: %s\n", name, cudaGetErrorString(result));
	}
}
