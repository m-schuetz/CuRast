
#pragma once

#include <cmath>
#include <bit>


#ifdef __CUDA_ARCH__
	#include <math_constants.h>
	constexpr float Infinity = __builtin_bit_cast(float, 0x7f800000);

	// === required by GLM ===
	#define GLM_FORCE_CUDA
	#define CUDA_VERSION 12000
	namespace std {
		using size_t = ::size_t;
	};
	// =======================
	#include "./glm/glm/glm.hpp"

#else
	#include <cstdint>
	constexpr float Infinity = __builtin_bit_cast(float, 0x7f800000);
#endif

#include "../types.h"


 using glm::vec2;
 using glm::vec3;
 using glm::vec4;
 using glm::ivec2;
 using glm::ivec3;
 using glm::ivec4;
 using glm::mat4;


// Rendering las files directly from the memory-mapped file is limited to the first N points
constexpr uint64_t MAX_LAS_POINTS = 2'000'000;

struct Box3 {
	vec3 min = { Infinity, Infinity, Infinity };
	vec3 max = { -Infinity, -Infinity, -Infinity };
};

struct RenderTarget{
	uint64_t* colorbuffer;
	int width;
	int height;
	mat4 proj;
};


struct PotreeNode{
	u8* data; // Pointer to the memory-mapped location of this octree node
	u64 numPoints;
	u64 offset_color;      // byte offset of uint16 rgb within a point. Set to >= bytesPerPoint if there is no rgb.
	mat4 worldView;

	// Points are stored interleaved with bytesPerPoint stride, positions as int32 xyz.
	// Decoded position: vec3(xyz) * scale + offset
	// - offset should be relative to a nearby origin (e.g. metadata offset minus bounding box center), 
	//   computed in double on host and only then converted to float, to keep precision reasonable. 
	u64 bytesPerPoint;
	u64 offset_position;   // byte offset of int32 xyz within a point
	vec3 scale;
	vec3 offset;
};

extern __constant__ RenderTarget c_target;

