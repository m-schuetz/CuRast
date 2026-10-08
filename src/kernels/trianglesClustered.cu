// Renders clustered LOD meshes (see ClusteredMeshNode and tools/clodbuilder/README.md) in two passes:
// 1. kernel_selectClusters: One thread per cluster. Selects the clusters of the LOD cut for the current view,
//    i.e., clusters whose own error is small enough on screen while the error of the coarser clusters that
//    replace them is not. Culls them against the view frustum and appends the remaining ones to a list.
// 2. kernel_drawClusters: One block per visible cluster. Transforms the cluster's vertices into shared memory,
//    then rasterizes its triangles. Small triangles are rasterized by one thread each. Larger ones are stashed in 
//    shared memory. Once its threads are done with their small triangles, each warp repeatedly takes the next 
//    stashed triangle, and rasterizes it with all 32 threads.
//
// The texture is BC7-compressed and decoded here (see bc7.cuh), so it can be read from VRAM or from a memory-mapped file.
//
// Limitations: Triangles that cross the near plane are discarded instead of clipped.
// The texture's mip level is selected once per triangle, and sampled bilinearly without blending between levels.

#include <cstdint>
#include <cstdio>
#include <cfloat>
#include <cuda_runtime.h>

// GLM detects CUDA via CUDA_VERSION from the driver API's cuda.h. The runtime API defines CUDART_VERSION.
#ifndef CUDA_VERSION
	#define CUDA_VERSION CUDART_VERSION
#endif
#define GLM_FORCE_CUDA
#define GLM_FORCE_NO_CTOR_INIT

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "./glm/glm/glm.hpp"
#include "./glm/glm/gtc/matrix_transform.hpp"
#include "./glm/glm/gtc/matrix_access.hpp"
#include "./glm/glm/gtx/transform.hpp"
#include "./glm/glm/gtc/quaternion.hpp"

namespace cg = cooperative_groups;

#include "./HostDeviceInterface.h"
#include "../types.h"
#include "./kernels.h"
#include "../Timer.h"
#include "./bc7.cuh"

using glm::ivec2;
using glm::vec4;

// One thread per vertex and per triangle of a cluster
constexpr int CLUSTER_BLOCK_SIZE = 128;

// Triangles with larger screen-space bounding boxes are rasterized by all threads of a warp. 
// 16 was the fastest of 4 to 128 pixels on odm_wietrznia (RTX 4090), both for distant and close views.
constexpr int MAX_THREAD_TRIANGLE_PIXELS = 16;

__constant__ u32 LEVEL_COLORS[] = {
	0xff4b19e6, 0xff4bb43c, 0xff19e1ff, 0xffc88200, 0xff3082f5, 0xffb41e91, 0xfff0f046, 0xffe632f0,
	0xff3cf5d2, 0xffd4befa, 0xff808000, 0xffffbedc, 0xff286eaa, 0xffc8faff, 0xff000080, 0xffc3ffaa,
	0xff008080, 0xffb4d7ff, 0xff800000, 0xff808080,
};

// Approximate screen-space size of an error, in pixels. See tools/clodbuilder/README.md
__device__ float projectedError(vec4 sphere, float error, const ClusteredMesh& mesh, const RenderTarget& target){
	float distance = length(vec3(sphere) - mesh.cameraPosition) - sphere.w;
	distance = max(distance, mesh.znear);

	return error / distance * target.proj[1][1] * 0.5f * float(target.height);
}

extern "C" __global__
void kernel_selectClusters(
	RenderTarget target,
	ClusteredMesh mesh,
	u32* visibleClusters,
	ClusterCounters* counters
){
	auto grid = cg::this_grid();

	u32 clusterIndex = grid.thread_rank();
	if(clusterIndex >= mesh.numClusters) return;

	Cluster cluster = mesh.clusters[clusterIndex];

	// LOD cut: the cluster's own error must be small enough, but not the error of the coarser clusters that replace it
	float threshold = mesh.lodErrorThreshold;
	bool fineEnough = projectedError(cluster.lodSphere, cluster.lodError, mesh, target) <= threshold;
	bool parentTooCoarse = cluster.parentError == FLT_MAX || projectedError(cluster.parentSphere, cluster.parentError, mesh, target) > threshold;

	if(!fineEnough || !parentTooCoarse) return;

	if(mesh.frustumCulling){
		vec3 center = vec3(cluster.cullSphere);
		float radius = cluster.cullSphere.w;

		for(int i = 0; i < 5; i++){
			vec4 plane = mesh.frustumPlanes[i];
			if(dot(vec3(plane), center) + plane.w < -radius) return;
		}
	}

	u32 index = atomicAdd(&counters->numVisibleClusters, 1);
	visibleClusters[index] = clusterIndex;
	atomicAdd(&counters->numVisibleTriangles, cluster.triangleCount);
	atomicAdd(&counters->numVisibleVertices, cluster.vertexCount);
}

// Vertex of the current cluster in shared memory. Plain struct, because __shared__ variables can't have constructors.
struct ScreenVertex{
	float x, y;   // in pixels
	float w;      // view-space depth
};

struct Triangle{
	vec2 p0, p1, p2;               // screen space, in pixels
	float invW0, invW1, invW2;     // 1 / view-space depth
	vec2 uv0, uv1, uv2;
	float invArea;
	u32 textureLevel;              // mip level, from the size of a pixel in texels
	float texelsPerPixel;          // area of a pixel in texels of that level, to estimate how much texture data is read
	int minX, minY, maxX, maxY;    // pixel bounding box, clamped to the render target
};

// RGBA8 texel of a mip level, clamp to edge
__device__ u32 fetchTexel(const u8* levelData, u32 levelWidth, u32 levelHeight, int x, int y){
	x = glm::clamp(x, 0, int(levelWidth) - 1);
	y = glm::clamp(y, 0, int(levelHeight) - 1);

	u32 blocksPerRow = (levelWidth + 3) / 4;
	const u8* block = levelData + 16 * ((y / 4) * blocksPerRow + x / 4);

	return decodeBC7Texel(block, (y % 4) * 4 + (x % 4));
}

__device__ vec4 unpackColor(u32 color){
	return vec4(color & 0xff, (color >> 8) & 0xff, (color >> 16) & 0xff, color >> 24);
}

// Bilinear sample of a mip level. Returns RGBA in 0 to 255.
__device__ vec4 sampleTexture(const BC7Texture& texture, vec2 uv, u32 levelIndex){

	// levels are stored one after another, starting with the largest
	const u8* levelData = texture.data;
	u32 levelWidth = texture.width;
	u32 levelHeight = texture.height;
	for(u32 i = 0; i < levelIndex; i++){
		levelData += 16ull * ((levelWidth + 3) / 4) * ((levelHeight + 3) / 4);
		levelWidth = max(levelWidth / 2, 1u);
		levelHeight = max(levelHeight / 2, 1u);
	}

	// clamp before converting to int, uvs of thin triangles can be far outside of [0, 1]
	float x = glm::clamp(uv.x * float(levelWidth) - 0.5f, -1.0f, float(levelWidth));
	float y = glm::clamp(uv.y * float(levelHeight) - 0.5f, -1.0f, float(levelHeight));
	float x0 = floorf(x);
	float y0 = floorf(y);
	float fx = x - x0;
	float fy = y - y0;
	int ix = int(x0);
	int iy = int(y0);

	vec4 c00 = unpackColor(fetchTexel(levelData, levelWidth, levelHeight, ix + 0, iy + 0));
	vec4 c10 = unpackColor(fetchTexel(levelData, levelWidth, levelHeight, ix + 1, iy + 0));
	vec4 c01 = unpackColor(fetchTexel(levelData, levelWidth, levelHeight, ix + 0, iy + 1));
	vec4 c11 = unpackColor(fetchTexel(levelData, levelWidth, levelHeight, ix + 1, iy + 1));

	return glm::mix(glm::mix(c00, c10, fx), glm::mix(c01, c11, fx), fy);
}

__device__ float edgeFunction(vec2 a, vec2 b, vec2 p){
	return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

// Sets up the triangle with the given cluster-local index. Returns false if it is not visible.
__device__ bool setupTriangle(
	Triangle& t, u32 localIndex, const Cluster& cluster,
	const ScreenVertex* vertices, const float2* uvs,
	const ClusteredMesh& mesh, const RenderTarget& target
){
	const u8* indices = mesh.triangles + 3 * (cluster.triangleOffset + localIndex);
	ScreenVertex v0 = vertices[indices[0]];
	ScreenVertex v1 = vertices[indices[1]];
	ScreenVertex v2 = vertices[indices[2]];

	if(v0.w < mesh.znear || v1.w < mesh.znear || v2.w < mesh.znear) return false;

	t.p0 = {v0.x, v0.y};
	t.p1 = {v1.x, v1.y};
	t.p2 = {v2.x, v2.y};

	float area = edgeFunction(t.p0, t.p1, t.p2);
	if(area == 0.0f) return false;

	// no backface culling, both sides are rendered
	t.invArea = 1.0f / area;

	float minX = min(min(t.p0.x, t.p1.x), t.p2.x);
	float minY = min(min(t.p0.y, t.p1.y), t.p2.y);
	float maxX = max(max(t.p0.x, t.p1.x), t.p2.x);
	float maxY = max(max(t.p0.y, t.p1.y), t.p2.y);

	// clamp in float first, the coordinates of vertices close to the camera can exceed the int range
	t.minX = int(max(floorf(minX), 0.0f));
	t.minY = int(max(floorf(minY), 0.0f));
	t.maxX = int(min(ceilf(maxX), float(target.width - 1)));
	t.maxY = int(min(ceilf(maxY), float(target.height - 1)));

	if(t.minX > t.maxX || t.minY > t.maxY) return false;

	t.invW0 = 1.0f / v0.w;
	t.invW1 = 1.0f / v1.w;
	t.invW2 = 1.0f / v2.w;

	t.uv0 = {uvs[indices[0]].x, uvs[indices[0]].y};
	t.uv1 = {uvs[indices[1]].x, uvs[indices[1]].y};
	t.uv2 = {uvs[indices[2]].x, uvs[indices[2]].y};

	t.textureLevel = 0;
	t.texelsPerPixel = 0.0f;

	if(mesh.texture.data != nullptr){
		// mip level closest to the size of a pixel in texels, from the screen-space derivatives of the (affinely interpolated) uvs
		vec2 textureSize = {float(mesh.texture.width), float(mesh.texture.height)};
		vec2 dw0 = vec2(-(t.p2.y - t.p1.y), t.p2.x - t.p1.x) * t.invArea;
		vec2 dw1 = vec2(-(t.p0.y - t.p2.y), t.p0.x - t.p2.x) * t.invArea;
		vec2 dw2 = vec2(-(t.p1.y - t.p0.y), t.p1.x - t.p0.x) * t.invArea;
		vec2 dTexelsdx = (t.uv0 * dw0.x + t.uv1 * dw1.x + t.uv2 * dw2.x) * textureSize;
		vec2 dTexelsdy = (t.uv0 * dw0.y + t.uv1 * dw1.y + t.uv2 * dw2.y) * textureSize;
		float level = max(log2f(max(length(dTexelsdx), length(dTexelsdy))), 0.0f);
		t.textureLevel = min(u32(level + 0.5f), mesh.texture.numLevels - 1);

		// the pixel's footprint is a parallelogram spanned by the derivatives. Each level has a quarter of the texels.
		float footprint = abs(dTexelsdx.x * dTexelsdy.y - dTexelsdx.y * dTexelsdy.x);
		t.texelsPerPixel = footprint / float(1u << (2 * t.textureLevel));
	}

	return true;
}

// Returns true if the texture was sampled
__device__ bool drawPixel(const Triangle& t, int x, int y, u32 flatColor, const ClusteredMesh& mesh, const RenderTarget& target){

	// All three edge functions are evaluated explicitly, so that pixels on an edge shared by two triangles
	// are covered by at least one of them.
	vec2 p = {float(x) + 0.5f, float(y) + 0.5f};
	float w0 = edgeFunction(t.p1, t.p2, p) * t.invArea;
	float w1 = edgeFunction(t.p2, t.p0, p) * t.invArea;
	float w2 = edgeFunction(t.p0, t.p1, p) * t.invArea;

	if(w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) return false;

	float invW = w0 * t.invW0 + w1 * t.invW1 + w2 * t.invW2;
	float depth = 1.0f / invW;

	i32 pixelID = x + target.width * y;
	u64 udepth = __float_as_uint(depth);

	// skip the texture fetch if the pixel already holds something closer
	if(udepth > (target.colorbuffer[pixelID] >> 32)) return false;

	u32 color = flatColor;
	bool textured = mesh.colorMode == CLUSTER_COLOR_TEXTURE && mesh.texture.data != nullptr;
	if(textured){
		vec2 uv = (w0 * t.invW0 * t.uv0 + w1 * t.invW1 * t.uv1 + w2 * t.invW2 * t.uv2) / invW;
		vec4 texel = sampleTexture(mesh.texture, uv, t.textureLevel);

		color = u32(texel.r + 0.5f)
			| (u32(texel.g + 0.5f) << 8)
			| (u32(texel.b + 0.5f) << 16)
			| (0xffu << 24);
	}

	u64 fragment = udepth << 32 | color;

	if(fragment < target.colorbuffer[pixelID]){
		atomicMin((unsigned long long*)&target.colorbuffer[pixelID], (unsigned long long)fragment);
	}

	return textured;
}

extern "C" __global__ __launch_bounds__(CLUSTER_BLOCK_SIZE)
void kernel_drawClusters(
	RenderTarget target,
	ClusteredMesh mesh,
	u32* visibleClusters,
	ClusterCounters* counters
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	auto warp = cg::tiled_partition<32>(block);

	__shared__ ScreenVertex sh_vertices[CLUSTER_BLOCK_SIZE];
	__shared__ float2 sh_uvs[CLUSTER_BLOCK_SIZE];

	// Larger triangles of the current cluster. Raw storage, because __shared__ variables can't have constructors.
	__shared__ alignas(16) u8 sh_largeTriangleStorage[CLUSTER_BLOCK_SIZE * sizeof(Triangle)];
	__shared__ u32 sh_numLargeTriangles;
	__shared__ u32 sh_nextLargeTriangle;
	Triangle* sh_largeTriangles = reinterpret_cast<Triangle*>(sh_largeTriangleStorage);

	mat4 transform = target.proj * mesh.worldView;
	u32 numVisibleClusters = counters->numVisibleClusters;
	u32 tid = block.thread_rank();
	u32 lane = warp.thread_rank();

	// estimated number of distinct texels sampled by this thread, see Triangle::texelsPerPixel
	float sampledTexels = 0.0f;

	// Current block draws visibleClusters[blockRank].
	// Loops because only as many blocks are launched as can be resident on the GPU.
	for(
		u32 i = grid.block_rank();
		i < numVisibleClusters;
		i += grid.num_blocks()
	){
		u32 clusterIndex = visibleClusters[i];
		Cluster cluster = mesh.clusters[clusterIndex];

		if(tid < cluster.vertexCount){
			vec3 position = mesh.positions[cluster.vertexOffset + tid];
			vec4 clip = transform * vec4(position, 1.0f);

			sh_vertices[tid] = {
				(clip.x / clip.w * 0.5f + 0.5f) * float(target.width),
				(clip.y / clip.w * 0.5f + 0.5f) * float(target.height),
				clip.w
			};
			vec2 uv = mesh.uvs[cluster.vertexOffset + tid];
			sh_uvs[tid] = make_float2(uv.x, uv.y);
		}

		if(tid == 0){
			sh_numLargeTriangles = 0;
			sh_nextLargeTriangle = 0;
		}

		block.sync();

		u32 flatColor = 0xff888888;
		if(mesh.colorMode == CLUSTER_COLOR_LEVEL){
			flatColor = LEVEL_COLORS[cluster.level % 20];
		}else if(mesh.colorMode == CLUSTER_COLOR_CLUSTER){
			flatColor = (clusterIndex * 2654435761u) | 0xff000000;
		}

		Triangle t;
		bool visible = tid < cluster.triangleCount && setupTriangle(t, tid, cluster, sh_vertices, sh_uvs, mesh, target);
		bool large = visible && (t.maxX - t.minX + 1) * (t.maxY - t.minY + 1) > MAX_THREAD_TRIANGLE_PIXELS;

		// stash larger triangles, with one atomic per warp
		u32 largeMask = warp.ballot(large);
		u32 warpOffset = 0;
		if(lane == 0 && largeMask != 0) warpOffset = atomicAdd(&sh_numLargeTriangles, __popc(largeMask));
		warpOffset = warp.shfl(warpOffset, 0);
		if(large){
			sh_largeTriangles[warpOffset + __popc(largeMask & ((1u << lane) - 1))] = t;
		}

		// all triangles are set up and stashed
		block.sync();

		// small triangles: one thread each
		if(visible && !large){
			for(int y = t.minY; y <= t.maxY; y++)
			for(int x = t.minX; x <= t.maxX; x++){
				if(drawPixel(t, x, y, flatColor, mesh, target)) sampledTexels += t.texelsPerPixel;
			}
		}

		// larger triangles: each warp repeatedly takes the next stashed triangle, and processes its pixels with all threads
		u32 numLarge = sh_numLargeTriangles;
		while(true){
			u32 j = 0;
			if(lane == 0) j = atomicAdd(&sh_nextLargeTriangle, 1);
			j = warp.shfl(j, 0);
			if(j >= numLarge) break;

			Triangle large = sh_largeTriangles[j];

			int width = large.maxX - large.minX + 1;
			int numPixels = width * (large.maxY - large.minY + 1);

			for(int pixel = lane; pixel < numPixels; pixel += 32){
				if(drawPixel(large, large.minX + pixel % width, large.minY + pixel / width, flatColor, mesh, target)) sampledTexels += large.texelsPerPixel;
			}
		}

		// shared memory is reused for the next cluster
		block.sync();
	}

	float warpTexels = cg::reduce(warp, sampledTexels, cg::plus<float>());
	if(warp.thread_rank() == 0 && warpTexels > 0.0f){
		atomicAdd(&counters->textureTexels, warpTexels);
	}
}

// ------------------------------------------------------------------------------------------------
// Host
// ------------------------------------------------------------------------------------------------

static bool registered =
	registerKernel("trianglesClustered.cu", "kernel_selectClusters", (const void*)kernel_selectClusters) &&
	registerKernel("trianglesClustered.cu", "kernel_drawClusters", (const void*)kernel_drawClusters);

void launch_selectClusters(const RenderTarget& target, const ClusteredMesh& mesh, uint32_t* visibleClusters, ClusterCounters* counters){
	if(mesh.numClusters == 0) return;

	// one thread per cluster
	uint32_t blockSize = 256;
	uint32_t gridSize = (mesh.numClusters + blockSize - 1) / blockSize;

	auto start = Timer::recordCudaTimestamp();
	kernel_selectClusters<<<gridSize, blockSize>>>(target, mesh, visibleClusters, counters);
	checkKernelLaunch("kernel_selectClusters");
	Timer::recordDuration("kernel_selectClusters", start, Timer::recordCudaTimestamp());
}

void launch_drawClusters(const RenderTarget& target, const ClusteredMesh& mesh, uint32_t* visibleClusters, ClusterCounters* counters){
	if(mesh.numClusters == 0) return;

	// The number of visible clusters is only known on the GPU.
	// We launch as many blocks as can be resident, and each loops over the visible clusters.
	static int numBlocks = [&](){
		int device, numSMs, blocksPerSM;
		cudaGetDevice(&device);
		cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_drawClusters, CLUSTER_BLOCK_SIZE, 0);
		return std::max(numSMs * blocksPerSM, 10);
	}();

	auto start = Timer::recordCudaTimestamp();
	kernel_drawClusters<<<numBlocks, CLUSTER_BLOCK_SIZE>>>(target, mesh, visibleClusters, counters);
	checkKernelLaunch("kernel_drawClusters");
	Timer::recordDuration("kernel_drawClusters", start, Timer::recordCudaTimestamp());
}
