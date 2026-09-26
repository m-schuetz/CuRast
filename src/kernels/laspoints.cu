// Renders points of a LasfileNode directly from the memory-mapped las file.
// Requires that the GPU can access pageable host memory (e.g. HMM on linux).
//
// - positions are decoded as X * scale, without adding the offset, so they stay centered around the origin
// - only the first MAX_LAS_POINTS points are rendered

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

// Point records are packed and not aligned (e.g. 34 bytes per point, first point at an odd byte offset),
// so values are assembled from individual bytes.
inline u32 readU16(const u8* p){
	return u32(p[0]) | (u32(p[1]) << 8);
}

inline i32 readI32(const u8* p){
	return i32(u32(p[0]) | (u32(p[1]) << 8) | (u32(p[2]) << 16) | (u32(p[3]) << 24));
}

extern "C" __global__
void kernel_drawLasPoints(
	RenderTarget target,
	u8* points,             // memory-mapped las file, pointing to the first point record
	u64 numPoints,
	u32 pointRecordSize,
	i32 offset_rgb,         // byte offset of rgb within a point record, -1 if the format has no rgb
	vec3 scale,
	mat4 worldView
) {
	auto grid = cg::this_grid();

	mat4 transform = target.proj * worldView;

	u64 numRenderedPoints = min(numPoints, MAX_LAS_POINTS + 10'000'000);

	for(
		u64 i = grid.thread_rank();
		i < numRenderedPoints;
		i += grid.num_threads()
	){
		const u8* record = points + i * pointRecordSize;

		// X, Y, Z are the first 12 bytes of every point format
		vec3 pos = {
			float(readI32(record + 0)) * scale.x,
			float(readI32(record + 4)) * scale.y,
			float(readI32(record + 8)) * scale.z,
		};

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
		if(offset_rgb >= 0){
			// rgb is stored as 16 bit, but some files only use the 8 bit range
			u32 R = readU16(record + offset_rgb + 0);
			u32 G = readU16(record + offset_rgb + 2);
			u32 B = readU16(record + offset_rgb + 4);

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
