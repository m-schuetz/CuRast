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
// #include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#endif
#define GLM_FORCE_NO_CTOR_INIT
#ifndef CUDA_VERSION
#define CUDA_VERSION 12000
#endif
namespace std {
	using size_t = ::size_t;
};
// =======================

#include "./libs/glm/glm/glm.hpp"
#include "./libs/glm/glm/gtc/matrix_transform.hpp"
#include "./libs/glm/glm/gtc/matrix_access.hpp"
#include "./libs/glm/glm/gtx/transform.hpp"
#include "./libs/glm/glm/gtc/quaternion.hpp"

#include "./../types.h"

#include "./utils.cuh"
#include "./HostDeviceInterface.h"
#include "../BitEdit.h"
#include "./rasterization_helpers.cuh"

using glm::ivec2;
using glm::i8vec4;
using glm::vec4;

#if defined(__HIPCC_RTC__)
#pragma clang attribute push (__attribute__((device)), apply_to=function)
#endif

vec4 getVertex(const CMesh& mesh, u32 vertexIndex){

	u32 resolvedIndex;
	if(mesh.indices){
		resolvedIndex = mesh.indices[vertexIndex];
	}else{
		resolvedIndex = vertexIndex;
	}

	vec4 pos = vec4(mesh.positions[resolvedIndex], 1.0f);
	
	return pos;
}

vec2 getUV(const CMesh& mesh, u32 vertexIndex){

	u32 resolvedIndex;
	if(mesh.indices){
		resolvedIndex = mesh.indices[vertexIndex];
	}else{
		resolvedIndex = vertexIndex;
	}

	vec2 uv = mesh.uvs[resolvedIndex];
	
	return uv;
}

__device__
uint32_t sampleColor_nearest(
	uint32_t* textureData,
	int width,
	int height,
	vec2 uv
){

	if(textureData == nullptr) return 0;
	// return 0xff660066;
	uv.x = uv.x - floor(uv.x);
	uv.y = uv.y - floor(uv.y);
	int tx = int(uv.x * float(width) + 0.5f) % width;
	int ty = int(uv.y * float(height) + 0.5f) % height;
	int texelID = tx + ty * width;
	texelID = clamp(texelID, 0, width * height - 1);

	uint32_t color = 0;
	color = textureData[texelID];
	//uint8_t *rgb = (uint8_t *)&color;

	return color;
}

__device__
uint32_t sampleColor_linear(
	uint32_t* textureData,
	int width,
	int height,
	vec2 uv
){

	if(textureData == nullptr) return 0;

	uint32_t color = 0xff000000;
	uint8_t* rgba = (uint8_t*)&color;

	// Only for ply with textures
	// if(uv.x > 1.0f) return 0;
	// if(uv.y > 1.0f) return 0;
	// uv.y = 1.0f - uv.y;

	float ftx = (uv.x - floor(uv.x)) * float(width);
	float fty = (uv.y - floor(uv.y)) * float(height);

	auto getTexel = [&](float ftx, float fty) -> vec4 {
		int tx = fmodf(ftx, float(width));
		int ty = fmodf(fty, float(height));
		int texelID = tx + ty * width;
		texelID = clamp(texelID, 0, width * height - 1);

		uint32_t texel = textureData[texelID];
		uint8_t* rgba = (uint8_t*)&texel;

		return vec4{rgba[0], rgba[1], rgba[2], rgba[3]};
	};

	vec4 t00 = getTexel(ftx - 0.5f, fty - 0.5f);
	vec4 t01 = getTexel(ftx - 0.5f, fty + 0.5f);
	vec4 t10 = getTexel(ftx + 0.5f, fty - 0.5f);
	vec4 t11 = getTexel(ftx + 0.5f, fty + 0.5f);

	float wx = fmodf(ftx + 0.5f, 1.0f);
	float wy = fmodf(fty + 0.5f, 1.0f);

	vec4 interpolated =
		(1.0f - wx) * (1.0f - wy) * t00 +
		wx * (1.0f - wy) * t10 +
		(1.0f - wx) * wy * t01 +
		wx * wy * t11;

	rgba[0] = interpolated.r;
	rgba[1] = interpolated.g;
	rgba[2] = interpolated.b;
	rgba[3] = interpolated.a;


	return color;
}

// Used to sort queued translucent triangles by tile and depth.
// attributes: <tile><depth><queueIndex>
// tile_x: 8 bit
// tile_y: 8 bit
// depth: 24 bit
// triangle part index: 24 bit
u64 encodeTranslucentKeyValue(int tile_x, int tile_y, float depth, int queueIndex){
	
	u32 udepth = __float_as_uint(depth) & 0xffffff00;
	
	u64 keyValue = 
			u64(tile_y) << 56 |
			u64(tile_x) << 48 |
			u64(udepth) << 16 |
			u64(queueIndex);
			
	return keyValue;
}

void decodeTranslucentKeyValue(
	u64 keyValue, 
	int& tile_x, 
	int& tile_y, 
	float& depth,
	int& queueIndex
){
	tile_y = (keyValue >> 56) & 0xff;
	tile_x = (keyValue >> 48) & 0xff;
	u32 udepth = (keyValue >> 24) & 0xffffff00;
	depth = __uint_as_float(udepth);
	queueIndex = (keyValue >> 0) & 0xffffff;
}

void binning(
	const RasterArgs args,
	const CMesh& sh_mesh,
	int meshIndex,
	const mat4& worldView,
	int triangleIndex,
	int instanceIndex,
	vec4 a_object,
	vec4 b_object,
	vec4 c_object,
	TranslucentTriangle* queueTriangles,
	u64* queueKeyValues,
	u32* queueSize
){
	
	if(triangleIndex >= sh_mesh.numTriangles) return;
	
	// if(meshIndex != 0) return;
	// if(triangleIndex > 1) return;
	
	float f = args.target.proj[1][1];
	float aspect = float(args.target.width) / float(args.target.height);

	vec3 a_view = worldView * a_object;
	vec3 b_view = worldView * b_object;
	vec3 c_view = worldView * c_object;
	vec3 a_ndc = viewToNDC(a_view, f, aspect);
	vec3 b_ndc = viewToNDC(b_view, f, aspect);
	vec3 c_ndc = viewToNDC(c_view, f, aspect);
	
	// vec2 a_screen = ndcToScreen(a_ndc, args.target.width, args.target.height);
	// vec2 b_screen = ndcToScreen(b_ndc, args.target.width, args.target.height);
	// vec2 c_screen = ndcToScreen(c_ndc, args.target.width, args.target.height);
	
	float min_x, min_y, max_x, max_y;
	computeScreenSpaceBoundingBox(
		a_ndc, b_ndc, c_ndc, a_view, b_view, c_view,
		worldView, f, aspect, args.target.width, args.target.height,
		&min_x, &max_x, &min_y, &max_y
	);
	
	if(a_ndc.z < 0.0f && b_ndc.z < 0.0f && c_ndc.z < 0.0f) return;
	// if(a_ndc.z < 0.0f || b_ndc.z < 0.0f || c_ndc.z < 0.0f) return;
	if(min_x >= max_x) return;
	if(min_y >= max_y) return;
	
	// auto drawLine = [&](vec2 start, vec2 end){
	// 	for(float i = 0; i <= 200; i++){
	// 		float u = float(i) / 200.0f;
			
	// 		vec2 pos = (1.0f - u) * start + u * end;
			
	// 		int px = pos.x;
	// 		int py = pos.y;
			
	// 		if(px < 0 || px >= args.target.width) continue;
	// 		if(py < 0 || py >= args.target.height) continue;
			
	// 		int pixelID = px + args.target.width * py;
			
	// 		u64 pixel = 0x00000000'ff0000ff;
	// 		atomicMin(&args.target.colorbuffer[pixelID], pixel);
	// 	}
	// };
	
	// drawLine({min_x, min_y}, {max_x, min_y});
	// drawLine({min_x, max_y}, {max_x, max_y});
	// drawLine({min_x, min_y}, {min_x, max_y});
	// drawLine({max_x, min_y}, {max_x, max_y});
	
	// printf("%.3f, %.3f \n", min_x, max_x);
	
	
	// int size_x = max_x - min_x;
	// int size_y = max_y - min_y;
	int tiles_x = int(max_x / 16) - int(min_x / 16) + 1;
	int tiles_y = int(max_y / 16) - int(min_y / 16) + 1;
	// int tiles_x = (size_x + TILE_SIZE_TRANSLUCENT - 1) / TILE_SIZE_TRANSLUCENT;
	// int tiles_y = (size_y + TILE_SIZE_TRANSLUCENT - 1) / TILE_SIZE_TRANSLUCENT;
	int numTiles = tiles_x * tiles_y;
	
	// printf("%d %d\n", min_x, max_x);
	
	if(numTiles == 0) return;
	
	u32 queueIndexStart = atomicAdd(queueSize, numTiles);
	
	for(int i = 0; i < numTiles; i++){
		int lx = i % tiles_x;
		int ly = i / tiles_x;
		
		TranslucentTriangle t;
		t.meshIndex = meshIndex;
		t.tile_x = int(min_x) / TILE_SIZE_TRANSLUCENT + lx;
		t.tile_y = int(min_y) / TILE_SIZE_TRANSLUCENT + ly;
		t.triangleIndex = triangleIndex;
		
		float cx = (t.tile_x + 0.5f) * TILE_SIZE_TRANSLUCENT;
		float cy = (t.tile_y + 0.5f) * TILE_SIZE_TRANSLUCENT;
		float ndc_x = 2.0f * cx / float(args.target.width) - 1.0f;
		float ndc_y = 2.0f * cy / float(args.target.height) - 1.0f;
		vec3 ray_dir = {ndc_x * aspect / f, ndc_y / f, -1.0f};

		vec3 plane_normal = cross(b_view - a_view, c_view - a_view);
		float denom = dot(plane_normal, ray_dir);
		float depth = (abs(denom) > 1e-6f) ? dot(plane_normal, a_view) / denom : a_ndc.z;
		u64 keyValue = encodeTranslucentKeyValue(t.tile_x, t.tile_y, depth, queueIndexStart + i);
			
		queueTriangles[queueIndexStart + i] = t;
		queueKeyValues[queueIndexStart + i] = keyValue;
	}
	
	
	
}


#if defined(__HIPCC_RTC__)
#pragma clang attribute pop
#endif

extern "C" __global__
void kernel_stage1_binning(
	RasterArgs args,
	TranslucentTriangle* queueTriangles,
	u64* queueKeyValues,
	u32* queueSize
){
#if defined(USE_HIP)
	auto block = cg::this_thread_block();

	// The counters, the queue size and dbg_fragcount are zeroed by the CPU
	// before this kernel is launched (see drawTrianglesTranslucent), so no
	// grid.sync() is needed and the kernel can be launched non-cooperatively.

	// Initialize block state
	__shared__ int sh_blockBatchIndex;
	__shared__ int sh_blockLocalBatchIndex;
	__shared__ int sh_meshIndex;
	// HIP doesn't support __shared__ vars with constructors; use raw storage
	__shared__ alignas(CMesh) char sh_mesh_storage[sizeof(CMesh)];
	CMesh& sh_mesh = *reinterpret_cast<CMesh*>(sh_mesh_storage);

	if (block.thread_rank() == 0){
		sh_blockBatchIndex = 0;
		sh_blockLocalBatchIndex = 0;
		sh_meshIndex = 0;
		sh_mesh = args.meshes[0];
	}

	block.sync();
#else
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	
	// Initialize gridwide state
	if(grid.thread_rank() == 0){
		*args.numProcessedBatches = 0;
		*args.numProcessedBatches_nontrivial = 0;
		*args.hugeTrianglesCounter = 0;
		*args.nontrivialTrianglesCounter = 0;
		*args.numProcessedHugeTriangles = 0;
		*queueSize = 0;
	}
	

	// Initialize block state
	__shared__ int sh_blockBatchIndex;
	__shared__ int sh_blockLocalBatchIndex;
	__shared__ int sh_meshIndex;
	__shared__ CMesh sh_mesh;

	if (block.thread_rank() == 0){
		sh_blockBatchIndex = 0;
		sh_blockLocalBatchIndex = 0;
		sh_meshIndex = 0;
		sh_mesh = args.meshes[0];
		args.state->dbg_fragcount = 0;
	}

	grid.sync();
#endif
	
	

	// LOOP THROUGH TRIANGLES
	while (true){

		// Claim Work: Check which batch of triangles this block should render next.
		block.sync();
		if (block.thread_rank() == 0){

			u32 next = atomicAdd(args.numProcessedBatches, 1);
			u32 diff = next - sh_blockBatchIndex;
			
			sh_blockBatchIndex = next;
			sh_blockLocalBatchIndex += diff;
			
			// Next batch is outside of current mesh.
			// Advance through meshes to the one containing the next batch
			int numBatchesInMesh = (sh_mesh.numTriangles + TRIANGLES_PER_SWEEP - 1) / TRIANGLES_PER_SWEEP;
			while (sh_blockLocalBatchIndex >= numBatchesInMesh){
				sh_meshIndex++;

				if (sh_meshIndex >= args.numMeshes) break;
				
				sh_blockLocalBatchIndex -= numBatchesInMesh;
				sh_mesh = args.meshes[sh_meshIndex];
				numBatchesInMesh = (sh_mesh.numTriangles + TRIANGLES_PER_SWEEP - 1) / TRIANGLES_PER_SWEEP;
			}
		}
		block.sync();

		if (sh_meshIndex >= args.numMeshes) return;
		
		u32 firstTriangleInBatch = sh_blockLocalBatchIndex * TRIANGLES_PER_SWEEP;
		int triangleIndex = firstTriangleInBatch + block.thread_rank();

		if(triangleIndex >= sh_mesh.numTriangles) continue;

		vec4 a_object = getVertex(sh_mesh, 3 * triangleIndex + 0);
		vec4 b_object = getVertex(sh_mesh, 3 * triangleIndex + 1);
		vec4 c_object = getVertex(sh_mesh, 3 * triangleIndex + 2);
		
		for(int instanceIndex = 0; instanceIndex < sh_mesh.instances.count; instanceIndex++){
			mat4 worldView = args.transforms[sh_mesh.instances.offset + instanceIndex];
			binning(
				args, sh_mesh, sh_meshIndex, worldView, triangleIndex, instanceIndex, 
				a_object, b_object, c_object,
				queueTriangles, queueKeyValues, queueSize
			);
		}
	}

}


extern "C" __global__
void kernel_stage3_computeRanges(
	RasterArgs args,
	TranslucentTriangle* queueTriangles,
	u64* queueKeyValues,
	u32* queueSize,
	ivec2* tileRanges,
	u32 numTiles
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	
	u32 trianglePieceIndex = grid.thread_rank();
	
	if(trianglePieceIndex >= *queueSize) return;

	
	auto getTileID = [&](int trianglePieceIndex){
		u64 triangleKeyValue = queueKeyValues[trianglePieceIndex];
		u32 tiles_x = (args.target.width + TILE_SIZE_TRANSLUCENT - 1) / TILE_SIZE_TRANSLUCENT;
		
		int tile_x, tile_y;
		float depth;
		int queueIndex;
		decodeTranslucentKeyValue(triangleKeyValue, tile_x, tile_y, depth, queueIndex);
		
		if(trianglePieceIndex == 0){
			// printf("dec: %d, %d \n", tile_x, tile_y);
		}
		
		return tile_x + tiles_x * tile_y;
	};
	
	if(grid.thread_rank() == 0){
		
		// printf("==== \n");
		for(int i = 0; i < *queueSize; i++){
			
			u64 triangleKeyValue = queueKeyValues[i];
			u32 tiles_x = (args.target.width + TILE_SIZE_TRANSLUCENT - 1) / TILE_SIZE_TRANSLUCENT;
			
			int tile_x, tile_y;
			float depth;
			int queueIndex;
			decodeTranslucentKeyValue(triangleKeyValue, tile_x, tile_y, depth, queueIndex);
			
			// printf("%d, %d \n", tile_x, tile_y);
		}
	}
	
	u32 tileID = getTileID(trianglePieceIndex);
	
	if(trianglePieceIndex == 0){
		ivec2 range = tileRanges[0];
		
		u64 triangleKeyValue = queueKeyValues[2];
	}
	
	if(trianglePieceIndex == 0){
		tileRanges[tileID].x = 0; // unnecessary?
	}else{
		u32 prevTileID = getTileID(trianglePieceIndex - 1);
		
		if(tileID != prevTileID){
			// printf("    tileRanges[%d].y = %d\n", prevTileID, trianglePieceIndex - 1);
			// printf("    tileRanges[%d].x = %d\n", tileID, trianglePieceIndex);
			tileRanges[prevTileID].y = trianglePieceIndex;
			tileRanges[tileID].x = trianglePieceIndex;
		}
	}
	
	if(trianglePieceIndex == *queueSize - 1){
		tileRanges[tileID].y = *queueSize;
	}
}

extern "C" __global__
void kernel_stage4_blend(
	RasterArgs args,
	TranslucentTriangle* queueTriangles,
	u64* queueKeyValues,
	u32* queueSize,
	ivec2* tileRanges,
	u32 numTiles
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	
#if defined(USE_HIP)
	// HIP's grid_group has no block_rank(); the launch grid is one-dimensional.
	u32 tileID = blockIdx.x;
#else
	u32 tileID = grid.block_rank();
#endif
	
	u32 tiles_x = (args.target.width + TILE_SIZE_TRANSLUCENT - 1) / TILE_SIZE_TRANSLUCENT;
	u32 tiles_y = (args.target.height + TILE_SIZE_TRANSLUCENT - 1) / TILE_SIZE_TRANSLUCENT;
	u32 tile_x = tileID % tiles_x;
	u32 tile_y = tileID / tiles_x;
	int lx = block.thread_rank() % TILE_SIZE_TRANSLUCENT; // local coordinate within tile
	int ly = block.thread_rank() / TILE_SIZE_TRANSLUCENT;
	int x = tile_x * TILE_SIZE_TRANSLUCENT + lx; // global framebuffer coordinate
	int y = tile_y * TILE_SIZE_TRANSLUCENT + ly; // global framebuffer coordinate
	int pixelID = toFramebufferIndex(x, y, args.target.width);
	
	if(tileID >= numTiles) return;
	
	float depth = Infinity;
	vec4 rgba = {0.0f, 0.0f, 0.0f, 0.0f};
	float remainingTranslucency = 1.0f;
	
	if(x > 0 && x < args.target.width)
	if(y > 0 && y < args.target.height)
	{
		u64 currentPixel = args.target.colorbuffer[pixelID];
		
		depth = __uint_as_float(currentPixel >> 32);
		u32 currentColor = currentPixel & 0xffffffff;
		u8* currentRGBA = (u8*)&currentPixel;
		
		// rgba.x = float(currentRGBA[0]) / 256.0f;
		// rgba.y = float(currentRGBA[1]) / 256.0f;
		// rgba.z = float(currentRGBA[2]) / 256.0f;
		// rgba.w = 0.0f;
		
	}
	
	float f = args.target.proj[1][1];
	float aspect = float(args.target.width) / float(args.target.height);
	float faI = 1.0f / (f / aspect);
	float fI = 1.0f / f;
	vec3 origin = vec4(0.0f, 0.0f, 0.0f, 1.0f);
	vec3 viewDir = vec4(0.0f, 0.0f, -1.0f, 0.0f);
	
	float u = 2.0f * (float(x) + 0.5f) / float(args.target.width) - 1.0f;
	float v = 2.0f * (float(y) + 0.5f) / float(args.target.height) - 1.0f;

	vec3 rayDir = normalize(vec3{
		(1.0f / (f / aspect)) * u,
		(1.0f / f) * v,
		-1.0f
	});
	
	ivec2 range = tileRanges[tileID];
	u32 numPiecesInTile = range.y - range.x;
	
	auto drawLine = [&](vec2 start, vec2 end){
		for(float i = 0; i <= 200; i++){
			float u = float(i) / 200.0f;
			
			vec2 pos = (1.0f - u) * start + u * end;
			
			int px = pos.x;
			int py = pos.y;
			
			if(px < 0 || px >= args.target.width) continue;
			if(py < 0 || py >= args.target.height) continue;
			
			int pixelID = px + args.target.width * py;
			
			u64 pixel = 0x00000000'ff0000ff;
			atomicMin(&args.target.colorbuffer[pixelID], pixel);
		}
	};
	
	if(numPiecesInTile == 0) return;
	
	// if(block.thread_rank() == 0){
	// 	// printf("%d \n", tileID);
		
	// 	int min_x = TILE_SIZE_TRANSLUCENT * tile_x;
	// 	int min_y = TILE_SIZE_TRANSLUCENT * tile_y;
		
	// 	drawLine({min_x, min_y}, {min_x + 16.0f, min_y + 0.0f});
	// 	drawLine({min_x, min_y + 16.0f}, {min_x + 16.0f, min_y + 16.0f});
	// 	drawLine({min_x, min_y}, {min_x, min_y + 16.0f});
	// 	drawLine({min_x + 16.0f, min_y}, {min_x + 16.0f, min_y + 16.0f});
	// }
	
	constexpr int PREFETCH_COUNT = 256;
	int iterations = (numPiecesInTile + PREFETCH_COUNT - 1) / PREFETCH_COUNT;
	
	__shared__ TranslucentTriangle sh_triangles[PREFETCH_COUNT];
	
	for(int iteration = 0; iteration < iterations; iteration++){
		
		int index = PREFETCH_COUNT * iteration + block.thread_rank();
		
		// fetch triangles
		if(index < numPiecesInTile && block.thread_rank() < PREFETCH_COUNT){
			u64 keyValue = queueKeyValues[range.x + index];
			
			i32 tmp_tile_x, tmp_tile_y;
			float tmp_depth;
			int queueIndex;
			decodeTranslucentKeyValue(keyValue, tmp_tile_x, tmp_tile_y, tmp_depth, queueIndex);
			
			sh_triangles[block.thread_rank()] = queueTriangles[queueIndex];
		}
		
		block.sync();
		
		// blend triangles
		int numTriangles = min(numPiecesInTile - PREFETCH_COUNT * iteration, PREFETCH_COUNT);
		
		// Stash some fragments. We'll blend the closest of multiple stashed ones in each iteration.
		vec4 stashedColor[4] = {};
		float stashedDepth[4] = {Infinity, Infinity, Infinity, Infinity};

		// Process the closest stashed fragment
		auto blendMin = [&]() {
			int minIdx = 0;
			for(int k = 1; k < 4; k++)
				if(stashedDepth[k] < stashedDepth[minIdx]) minIdx = k;
			if(stashedDepth[minIdx] == Infinity) return;
			vec4 c = stashedColor[minIdx];
			rgba.r += c.r * c.a * remainingTranslucency;
			rgba.g += c.g * c.a * remainingTranslucency;
			rgba.b += c.b * c.a * remainingTranslucency;
			rgba.a += c.a * remainingTranslucency;
			remainingTranslucency *= (1.0f - c.a);
			stashedDepth[minIdx] = Infinity;
		};
		
		for(int i = 0; i < numTriangles; i++){
			TranslucentTriangle triangle = sh_triangles[i];
			CMesh& mesh = args.meshes[triangle.meshIndex];
			
			mat4 worldView = args.transforms[mesh.instances.offset];
			
			vec4 a_object = getVertex(mesh, 3 * triangle.triangleIndex + 0);
			vec4 b_object = getVertex(mesh, 3 * triangle.triangleIndex + 1);
			vec4 c_object = getVertex(mesh, 3 * triangle.triangleIndex + 2);
			vec3 a_view = worldView * a_object;
			vec3 b_view = worldView * b_object;
			vec3 c_view = worldView * c_object;
			
			vec2 a_uv = getUV(mesh, 3 * triangle.triangleIndex + 0);
			vec2 b_uv = getUV(mesh, 3 * triangle.triangleIndex + 1);
			vec2 c_uv = getUV(mesh, 3 * triangle.triangleIndex + 2);
			
			float bary_u, bary_v;
			float t = intersectTriangle_mt(
				origin, rayDir,
				a_view, b_view, c_view,
				false,
				bary_u, bary_v
			);

			float bary_w = 1.0f - bary_u - bary_v;
			vec2 uv = bary_w * a_uv + bary_u * b_uv + bary_v * c_uv;
			
			if(t == Infinity || pixelID >= args.target.width * args.target.height) {
				continue;
			}
			
			float t_view = t * (-rayDir.z);
			if(t_view > depth) continue;
			
			
			uint32_t triangleColor = sampleColor_linear(mesh.texture.data, mesh.texture.width, mesh.texture.height, uv);
			u8* triangleRgba = (u8*)&triangleColor;

			float src_r = triangleRgba[0] / 255.0f;
			float src_g = triangleRgba[1] / 255.0f;
			float src_b = triangleRgba[2] / 255.0f;
			float src_a = (triangleRgba[3] / 255.0f);
			
			// Find a slot to insert the new fragment
			int slot = -1;
			for(int k = 0; k < 4; k++) {
				if(stashedDepth[k] == Infinity) { slot = k; break; }
			}
			
			// If no slot available, process the closest one, then look again.
			if(slot == -1) {
				blendMin();
				for(int k = 0; k < 4; k++) {
					if(stashedDepth[k] == Infinity) { slot = k; break; }
				}
			}
			
			// Add fragment to empty stash slot.
			stashedColor[slot] = {src_r, src_g, src_b, src_a};
			stashedDepth[slot] = t_view;
		}
		
		// Blend remaing stashed fragments
		for(int k = 0; k < 4; k++) blendMin();
	}
	
	if(x > 0 && x < args.target.width)
	if(y > 0 && y < args.target.height)
	{
		
		u64 currentPixel = args.target.colorbuffer[pixelID];
		u32 currentColor = currentPixel & 0xffffffff;
		u8* currentRGBA = (u8*)&currentPixel;
		
		// rgba.x = float(currentRGBA[0]) / 256.0f;
		// rgba.y = float(currentRGBA[1]) / 256.0f;
		// rgba.z = float(currentRGBA[2]) / 256.0f;
		// rgba.w = 0.0f;
		
	
		
		u32 C = 0;
		u8* RGBA = (u8*)&C;
		
		RGBA[0] = clamp((1.0f - rgba.a) * currentRGBA[0] + rgba.r * 255.0f, 0.0f, 255.0f);
		RGBA[1] = clamp((1.0f - rgba.a) * currentRGBA[1] + rgba.g * 255.0f, 0.0f, 255.0f);
		RGBA[2] = clamp((1.0f - rgba.a) * currentRGBA[2] + rgba.b * 255.0f, 0.0f, 255.0f);
		RGBA[3] = 255.0f;
		
		
		// if(rgba.a > 0){
			// RGBA[0] = clamp(rgba.r * 255.0f, 0.0f, 255.0f);
			// RGBA[1] = clamp(rgba.g * 255.0f, 0.0f, 255.0f);
			// RGBA[2] = clamp(rgba.b * 255.0f, 0.0f, 255.0f);
			// RGBA[3] = clamp(rgba.a * 255.0f, 0.0f, 255.0f);
		// }
		
		u64 udepth = __float_as_uint(depth);
		u64 pixel = udepth << 32 | C;
		
		args.target.colorbuffer[pixelID] = pixel;
		
	}
	
	
	
	
	
}