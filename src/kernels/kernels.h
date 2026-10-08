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
// image: RGBA8, target.width x target.height, rows from top to bottom. 
// transparentBackground: empty pixels are (0, 0, 0, 0) instead of backgroundColor.
void launch_resolveColorbufferToImage(const RenderTarget& target, uint32_t* image, bool enableEDL, uint32_t backgroundColor, bool transparentBackground);

// The point cloud kernels draw in one of the PointPasses, see points.cuh

// laspoints.cu
void launch_drawLasPoints(
	const RenderTarget& target, PointPass pass, uint8_t* points, uint64_t numPoints,
	uint32_t pointRecordSize, int32_t offset_rgb,
	const glm::vec3& scale, const glm::mat4& worldView);

// potreeFileRenderer.cu
void launch_drawPotreeFileNodes(const RenderTarget& target, PointPass pass, PotreeNode* nodes, uint64_t numNodes);

// potreeDirectStorageRenderer.cu
void launch_drawPotreeDirectStorageNodes(const RenderTarget& target, PointPass pass, PotreeNode* nodes, uint64_t numNodes);

// pointsHighQuality.cu: high-quality shading of point clouds, i.e., the passes POINT_PASS_DEPTH and POINT_PASS_ACCUMULATE.
// Clear before the depth pass, and normalize after the accumulation pass.
void launch_clearPointBuffers(const RenderTarget& target);
void launch_normalizePoints(const RenderTarget& target);

// trianglesClustered.cu
// The counters must be zero before selection.
// visibleOffset: number of visible clusters of previously drawn meshes, to make the triangle IDs in the framebuffer unique.
// uvVertexMask: optional, see ClusteredMeshNode::getUvVertexMask()
void launch_selectClusters(const RenderTarget& target, const ClusteredMesh& mesh, uint32_t* visibleClusters, ClusterCounters* counters);
void launch_drawClusters(
	const RenderTarget& target, const ClusteredMesh& mesh, uint32_t* visibleClusters, uint32_t numVisibleClusters, uint32_t visibleOffset);
void launch_shadeClusters(
	const RenderTarget& target, const ClusteredMesh& mesh, uint32_t* visibleClusters, uint32_t numVisibleClusters,
	uint32_t visibleOffset, ClusterCounters* counters, uint32_t* uvVertexMask);
void launch_countDistinctUvs(const uint32_t* uvVertexMask, uint32_t numWords, ClusterCounters* counters);

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
