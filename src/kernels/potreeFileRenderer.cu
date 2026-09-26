// Renders octree nodes of a PotreeFileNode directly from the memory-mapped octree.bin.
// Requires that the GPU can access pageable host memory (e.g. HMM on linux).
//
// The host is responsible for determining which nodes are visible and passes only those.
// Each block draws all points of one node.

#define CUB_DISABLE_BF16_SUPPORT

// === required by GLM ===
#define GLM_FORCE_CUDA
#define GLM_FORCE_NO_CTOR_INIT
#define CUDA_VERSION 12000
namespace std {
	using size_t = ::size_t;
};
// =======================

#include <cooperative_groups.h>

#include "./glm/glm/glm.hpp"
#include "./glm/glm/gtc/matrix_transform.hpp"
#include "./glm/glm/gtc/matrix_access.hpp"
#include "./glm/glm/gtx/transform.hpp"
#include "./glm/glm/gtc/quaternion.hpp"

#include "./utils.cuh"
#include "./HostDeviceInterface.h"
#include "../types.h"

using glm::ivec2;
using glm::vec4;

// Point records are packed and not aligned (e.g. 21 bytes per point),
// so values are assembled from individual bytes.
inline u32 readU16(const u8* p){
	return u32(p[0]) | (u32(p[1]) << 8);
}

inline i32 readI32(const u8* p){
	return i32(u32(p[0]) | (u32(p[1]) << 8) | (u32(p[2]) << 16) | (u32(p[3]) << 24));
}

extern "C" __global__
void kernel_drawPotreeFileNodes(
	RenderTarget target,
	PotreeNode* nodes,
	u64 numNodes
) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	// Current block rank draws all points of nodes[blockRank].
	// Loops in case fewer blocks than nodes were launched.
	for(
		u64 nodeIndex = grid.block_rank();
		nodeIndex < numNodes;
		nodeIndex += grid.num_blocks()
	){
		PotreeNode node = nodes[nodeIndex];

		mat4 transform = target.proj * node.worldView;
		bool hasColor = node.offset_color < node.bytesPerPoint;

		for(
			u64 i = block.thread_rank();
			i < node.numPoints;
			i += block.num_threads()
		){
			const u8* point = node.data + i * node.bytesPerPoint;
			const u8* xyz = point + node.offset_position;

			vec3 pos = vec3(
				float(readI32(xyz + 0)),
				float(readI32(xyz + 4)),
				float(readI32(xyz + 8))
			) * node.scale + node.offset;

			vec4 ndc = transform * vec4(pos, 1.0f);
			float depth = ndc.w;
			ndc.x = ndc.x / depth;
			ndc.y = ndc.y / depth;

			if(depth <= 0.0f) continue;
			if(ndc.x < -1.0f || ndc.x > 1.0f) continue;
			if(ndc.y < -1.0f || ndc.y > 1.0f) continue;

			ivec2 pixelCoords = {
				int(target.width  * (ndc.x * 0.5f + 0.5f)),
				int(target.height * (ndc.y * 0.5f + 0.5f))
			};

			if(pixelCoords.x < 0 || pixelCoords.x >= target.width) continue;
			if(pixelCoords.y < 0 || pixelCoords.y >= target.height) continue;

			i32 pixelID = pixelCoords.x + target.width * pixelCoords.y;

			u32 color = 0xff888888;
			if(hasColor){
				// rgb is stored as 16 bit, but some files only use the 8 bit range
				const u8* rgb = point + node.offset_color;
				u32 R = readU16(rgb + 0);
				u32 G = readU16(rgb + 2);
				u32 B = readU16(rgb + 4);

				u32 r = R <= 255 ? R : R / 256;
				u32 g = G <= 255 ? G : G / 256;
				u32 b = B <= 255 ? B : B / 256;

				color = r | (g << 8) | (b << 16) | (0xffu << 24);
			}

			u64 udepth = __float_as_uint(depth);
			u64 fragment = udepth << 32 | color;

			if(fragment < target.colorbuffer[pixelID]){
				atomicMin(&target.colorbuffer[pixelID], fragment);
			}
		}
	}

}
