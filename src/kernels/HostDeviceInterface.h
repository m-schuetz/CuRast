
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

// constexpr uint32_t BACKGROUND_COLOR = 0xff887766;
constexpr uint32_t BACKGROUND_COLOR = 0xffffffff;
constexpr uint64_t DEFAULT_PIXEL = (uint64_t(0x7f800000) << 32) | BACKGROUND_COLOR;

// Rendering las files directly from the memory-mapped file is limited to the first N points
constexpr uint64_t MAX_LAS_POINTS = 2'000'000;

struct Box3 {
	vec3 min = { Infinity, Infinity, Infinity };
	vec3 max = { -Infinity, -Infinity, -Infinity };

	bool isDefault() {
		return min.x == Infinity && min.y == Infinity && min.z == Infinity && max.x == -Infinity && max.y == -Infinity && max.z == -Infinity;
	}

	bool isEqual(Box3 box, float epsilon) {
		float diff_min = length(box.min - min);
		float diff_max = length(box.max - max);

		if (diff_min >= epsilon) return false;
		if (diff_max >= epsilon) return false;

		return true;
	}

	void extend(vec3 v){
		this->min.x = ::min(this->min.x, v.x);
		this->min.y = ::min(this->min.y, v.y);
		this->min.z = ::min(this->min.z, v.z);
		this->max.x = ::max(this->max.x, v.x);
		this->max.y = ::max(this->max.y, v.y);
		this->max.z = ::max(this->max.z, v.z);
	}

	Box3 transform(mat4 matrix){

		Box3 result;

		vec3 corners[8] = {
			{min.x, min.y, min.z},
			{max.x, min.y, min.z},
			{min.x, max.y, min.z},
			{max.x, max.y, min.z},
			{min.x, min.y, max.z},
			{max.x, min.y, max.z},
			{min.x, max.y, max.z},
			{max.x, max.y, max.z},
		};

		for(auto& c : corners){
			result.extend(vec3(matrix * vec4(c, 1.0f)));
		}

		return result;
	}
};

struct DeviceState{
	int counter;
	uint64_t dbg_fragcount;
};

struct RenderTarget{
	uint64_t* colorbuffer;
	int width;
	int height;
	mat4 view;
	mat4 viewI;
	mat4 proj;
	vec3 cameraPos;
	float f;
	float aspect;
	bool debug;
};

struct Uniforms{
	mat4 world;
	mat4 camWorld;
	mat4 transform;
	float time;
	float pad;
	uint32_t frameCount;

	struct {
		bool show;
		ivec2 start;
		ivec2 size;
	} inset;

};

struct CommonLaunchArgs{
	Uniforms uniforms;
	DeviceState* state;
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

struct CPointcloud{
	mat4 world;
	vec3* positions;
	uint32_t* colors;
	uint32_t numPoints;
};

extern __constant__ RenderTarget c_target;

