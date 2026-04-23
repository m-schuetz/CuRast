
#pragma once

#include <cmath>
#include <bit>


#ifdef __CUDA_ARCH__
	#include <math_constants.h>
	constexpr float Infinity = __builtin_bit_cast(float, 0x7f800000);
	#include "../jpeg/HashMap.cuh"
	#include "../jpeg/JptInterface.cuh"

	// === required by GLM ===
	#define GLM_FORCE_CUDA
	#define CUDA_VERSION 12000
	namespace std {
		using size_t = ::size_t;
	};
	// =======================
	#include "./glm/glm/glm.hpp"

#else
	constexpr float Infinity = __builtin_bit_cast(float, 0x7f800000);
	#include "./jpeg/HashMap.cuh"
#endif


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

constexpr uint32_t RASTERIZER_BASIC = 0;
constexpr uint32_t RASTERIZER_VISBUFFER = 1;
constexpr uint32_t RASTERIZER_OPENGL = 2;
constexpr uint32_t RASTERIZER_VISBUFFER2 = 3;
constexpr uint32_t RASTERIZER_VISBUFFER_SERIALIZED = 4;
constexpr uint32_t RASTERIZER_VISBUFFER_SCANLINE = 5;
constexpr uint32_t RASTERIZER_VISBUFFER_16BYTE_ALIGNED = 6;
constexpr uint32_t RASTERIZER_FORWARD = 7;
constexpr uint32_t RASTERIZER_VISBUFFER_INDEXED = 8;
//constexpr uint32_t RASTERIZER_VISBUFFER_CLUSTERS = 9;
constexpr uint32_t RASTERIZER_VISBUFFER_INSTANCED = 10;
constexpr uint32_t RASTERIZER_VULKAN = 11;
constexpr uint32_t RASTERIZER_VULKAN_INDEXPULLING = 12;
constexpr uint32_t RASTERIZER_VULKAN_INDEXED_DRAW = 13;
constexpr uint32_t RASTERIZER_VULKAN_INDEXPULLING_VISBUFFER = 14;
constexpr uint32_t RASTERIZER_VULKAN_INDEXPULLING_INSTANCED = 15;

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
	uint32_t numSmall;
	uint32_t numLarge;
	uint32_t numMassive;
	// uint32_t numNontrivial;
	uint64_t nanotime_start;
	uint64_t nanotime_stage_1;
	uint64_t nanotime_stage_2;
	uint64_t nanotime_stage_3;

	int32_t hovered_meshId;
	int32_t hovered_triangleIndex;

	uint32_t dbg_hovered_textureHandle;
	uint32_t dbg_hovered_mipLevel;
	uint32_t dbg_hovered_tx;
	uint32_t dbg_hovered_ty;
	uint32_t dbg_hovered_mcu_x;
	uint32_t dbg_hovered_mcu_y;
	uint32_t dbg_hovered_mcu;
	uint32_t dbg_hovered_decoded_color;
};

struct RenderTarget{
	uint64_t* framebuffer;
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

struct HuffmanTable {
	int num_codes_per_bit_length[16];
	int huffman_values[256];
	// packed[i] = (codelength << 16) | huffman_key � one load covers both per ballot lane
	uint32_t packed[256];
};

struct QuantizationTable {
	int values[64];
};

struct Texture{
	int width;
	int height;
	uint32_t* data;
	HuffmanTable* huffmanTables;
	QuantizationTable* quanttables;
	uint32_t* mcuPositions;
	uint32_t handle;
};

struct JpegPipeline{
	uint32_t* toDecode;
	uint32_t* toDecodeCounter;
	uint32_t* decoded;
	uint32_t* TBSlots;
	uint32_t* TBSlotsCounter;
	HashMap decodedMcuMap;
};

struct CMesh{
	uint32_t numTriangles;
	uint32_t* indices;
	vec3* positions;
	uint64_t cummulativeTriangleCount; // sum of all triangles in prior CMesh instances
	Box3 aabb;
	vec3 compressionFactor;
	uint32_t index_min;
	uint32_t bitsPerIndex;

	struct{
		int offset;
		int count;
	} instances;

	mat4 world;
	int id;
	vec2* uvs;
	vec3* normals;
	uint32_t* colors;
	uint32_t firstTriangle;
	uint32_t numVertices;
	uint32_t index_max;
	uint64_t address;


	Texture texture;

	bool isLoaded;
	bool flipTriangles;
	bool compressed;


	struct{
		vec3* positions;
		uint32_t* colors;
		uint32_t numPoints;
	} impostor;
};

// struct InstanceData{
// 	mat4 transform;
// 	bool flip;
// };

struct CPointcloud{
	mat4 world;
	vec3* positions;
	uint32_t* colors;
	uint32_t numPoints;
};

// struct Instances{
// 	uint32_t meshIndex;
// 	uint32_t numInstances;
// 	mat4* transforms;
// };

struct HugeTriangle{
	int meshIndex;
	int triangleIndex;
	int tile_x;
	int tile_y;
};

constexpr int TILE_SIZE = 64;
constexpr uint32_t TRIANGLES_PER_SWEEP = 256;
constexpr uint32_t MAX_HUGE_TRIANGLES = 5'000'000;
constexpr uint32_t MAX_NONTRIVIAL_TRIANGLES = 5'000'000;
constexpr uint32_t THRESHOLD_SMALL = 128;
constexpr uint32_t THRESHOLD_LARGE = 4096;
constexpr uint32_t TRIANGLES_PER_CHUNK = 128;

//constexpr float NEAR = 0.01f;
constexpr uint64_t PACKMASK_MESHINDEX = 0b111'1111'1111'1111; // 15 bit
constexpr uint64_t PACKMASK_TRIANGLEINDEX = 0b1'1111'1111'1111'1111'1111'1111; // 25 bit
// constexpr float INFINITY_F32 = INFINITY; // 1.0f / 0.0f;
// constexpr float INFINITY_F32 = 0x7F800000u; // 1.0f / 0.0f;
// constexpr float NEAR = 0.01f;
// constexpr float INFINITY_F32 = __builtin_huge_valf();

enum class DisplayAttribute : int{
	NONE,
	TEXTURE,
	UV,
	NORMAL,
	VERTEX_COLORS,
	TRIANGLE_ID,
	MESH_ID,
	STAGE,
};

struct RasterizationSettings{
	bool showWireframe;
	bool enableDiffuseLighting;
	bool enableObjectPicking;
	DisplayAttribute displayAttribute;
};

struct IndexbufferCompressInfo{
	uint32_t* uncompressedIndices;
	void* compressedIndices;
	uint32_t numIndices;
	uint32_t minIndex;
	uint32_t maxIndex;
};

enum class IndexFetch{DIRECT, INDEXBUFFER};
enum class Compression{UNCOMPRESSED, IX_PU16};
enum class Instancing{NO, YES};



struct RasterArgs{
	CMesh* meshes;
	uint32_t numMeshes;
	CMesh* instances;
	uint32_t numInstances;
	mat4* transforms;
	uint32_t* numProcessedBatches;
	uint32_t* numProcessedBatches_nontrivial;
	HugeTriangle* hugeTriangles;
	uint32_t* hugeTrianglesCounter;
	uint32_t* numProcessedHugeTriangles;
	uint32_t* nontrivialTrianglesCounter;
	uint64_t* nontrivialTrianglesList;
	RenderTarget target;
};

extern __constant__ RenderTarget c_target;