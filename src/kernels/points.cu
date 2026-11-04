#define CUB_DISABLE_BF16_SUPPORT

// === required by GLM ===
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
#include "../cuda_to_hip.h"
// Make GLM detect "CUDA" compiler to add __device__ __host__ qualifiers
#ifndef __CUDACC__
#define __CUDACC__
#endif
#define GLM_FORCE_PURE  // Disable x86 SIMD intrinsics
#else
#define GLM_FORCE_CUDA
#include <cooperative_groups.h>
#endif
#define GLM_FORCE_NO_CTOR_INIT
#ifndef CUDA_VERSION
#define CUDA_VERSION 12000
#endif
namespace std {
	using size_t = ::size_t;
};
// =======================

#include "./glm/glm/glm.hpp"
#include "./glm/glm/gtc/matrix_transform.hpp"
#include "./glm/glm/gtc/matrix_access.hpp"
#include "./glm/glm/gtx/transform.hpp"
#include "./glm/glm/gtc/quaternion.hpp"

#include "./utils.cuh"
#include "./HostDeviceInterface.h"
#include "../BitEdit.h"
#include "../types.h"

using glm::ivec2;
using glm::i8vec4;
using glm::vec4;


extern "C" __global__
void kernel_drawPoints(
	RenderTarget target,
	vec3* positions,
	u32* colors,
	u64 numPoints,
	mat4 worldView
) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	mat4 transform = target.proj * worldView;
	
	for(
		int i = grid.thread_rank(); 
		i < numPoints;
		i += grid.num_threads()
	){
		
		vec3 pos = positions[i];
		
		vec4 ndc = transform * vec4(pos, 1.0f);
		float depth = ndc.w;
		ndc.x = ndc.x / depth;
		ndc.y = ndc.y / depth;
		
		if(depth <= 0.0f) continue;
		if(ndc.x < -1.0f || ndc.x > 1.0f) continue;
		if(ndc.y < -1.0f || ndc.y > 1.0f) continue;
		
		vec2 pixelPos = {
			target.width * (ndc.x * 0.5f + 0.5f),
			target.height * (ndc.y * 0.5f + 0.5f)
		};
		
		
		i32 pixelID = int(pixelPos.x) + target.width * int(pixelPos.y);
		u64 udepth = __float_as_uint(depth);
		u64 color = colors[i];
		// color = 0xff00ffff;
		
		u32 batchIndex = (i / 50'000);
		// color = u32(batchIndex * 12345678);
		
		// only draw every 11th batch
		// if(batchIndex % 11 != 0) continue;
		
		// only draw first 1000 points of each batch
		// if(i % 50'000 > 500) continue;
		
		u64 fragment = udepth << 32 | color;
		
		// safety guard, but proper culling should already prevent this from happening
		if(pixelID < 0 || pixelID >= target.width * target.height){
			continue;
		}
		
		if(fragment < target.colorbuffer[pixelID]){
			atomicMin(&target.colorbuffer[pixelID], fragment);
		}
	}
	
}