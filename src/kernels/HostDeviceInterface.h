
#pragma once

#include <cmath>
#include <bit>
#include <cstdint>

#include "glm/glm.hpp"

constexpr float Infinity = __builtin_bit_cast(float, 0x7f800000);

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
	u8* data; // Pointer to this node's points: in the memory-mapped octree.bin, or in VRAM (direct storage)
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

// A cluster of a clustered LOD mesh, as stored in clusters.bin by tools/clodbuilder.
// See tools/clodbuilder/README.md for the meaning of the bounds and errors.
struct Cluster{
	vec3 aabbMin;
	u32 vertexOffset;      // first vertex in positions/uvs
	vec3 aabbMax;
	u32 triangleOffset;    // first triangle in triangles
	vec4 cullSphere;       // xyz: center, w: radius
	vec4 lodSphere;
	vec4 parentSphere;
	float lodError;
	float parentError;     // FLT_MAX if the cluster's group was not simplified any further
	u32 vertexCount;       // <= 128
	u32 triangleCount;     // <= 128
	u32 level;             // 0 = original mesh
	u32 group;
	i32 refinedGroup;
	u32 padding;
};
static_assert(sizeof(Cluster) == 112);

enum ClusterColorMode : int {
	CLUSTER_COLOR_TEXTURE = 0,
	CLUSTER_COLOR_LEVEL   = 1, // color by LOD level
	CLUSTER_COLOR_CLUSTER = 2, // random color per cluster
};

// BC7-compressed texture with mip levels, laid out as in a dds file:
// the blocks of each level row by row, starting with the largest level. Decoded in the kernel, see bc7.cuh.
struct BC7Texture{
	u8* data;                // first block of the largest level, nullptr if there is no texture. 4 byte aligned.
	u32 width;
	u32 height;
	u32 numLevels;
};

// A ClusteredMeshNode's data in VRAM or memory-mapped files, plus the per-frame parameters for drawing it
struct ClusteredMesh{
	Cluster* clusters;
	vec3* positions;
	vec2* uvs;
	u8* triangles;           // 3 cluster-local vertex indices per triangle
	BC7Texture texture;
	u32 numClusters;

	mat4 worldView;
	vec3 cameraPosition;     // in the mesh's coordinate system
	vec4 frustumPlanes[5];   // in the mesh's coordinate system: left, right, bottom, top, near. xyz: normal, w: distance
	bool frustumCulling;
	float znear;
	float lodErrorThreshold; // in pixels
	int colorMode;           // ClusterColorMode
};
