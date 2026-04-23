
#include "VulkanCudaSharedMemory.h"

struct Mesh{
	string name = "undefined";
	uint32_t numTriangles = 0;
	uint32_t numVertices = 0;
	bool isLoaded = false;
	uint32_t index_min = 0;
	uint32_t index_max = 0;
	Box3 aabb;

	bool compressed = false;

	CUdeviceptr cptr_position = 0;
	CUdeviceptr cptr_normal = 0;
	CUdeviceptr cptr_color = 0;
	CUdeviceptr cptr_uv = 0;
	CUdeviceptr cptr_indices = 0;

	VulkanCudaSharedMemory vkc_position;
	VulkanCudaSharedMemory vkc_normal;
	VulkanCudaSharedMemory vkc_color;
	VulkanCudaSharedMemory vkc_uv;
	VulkanCudaSharedMemory vkc_indices;
};