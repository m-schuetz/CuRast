#define CUB_DISABLE_BF16_SUPPORT

// === required by GLM ===
#define GLM_FORCE_CUDA
#define GLM_FORCE_NO_CTOR_INIT
#define CUDA_VERSION 12000
namespace std {
	using size_t = ::size_t;
};
// =======================

// #include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include "./glm/glm/glm.hpp"
#include "./glm/glm/gtc/matrix_transform.hpp"
#include "./glm/glm/gtc/matrix_access.hpp"
#include "./glm/glm/gtx/transform.hpp"
#include "./glm/glm/gtc/quaternion.hpp"

#include "./utils.cuh"
#include "./HostDeviceInterface.h"

using glm::ivec2;
using glm::i8vec4;
using glm::vec4;

__constant__ RenderTarget c_target;

__device__
vec3 worldToNDC(vec3 v, mat4 worldView, float f, float aspect){
	vec4 viewSpace = worldView * vec4(v.x, v.y, v.z, 1.0f);
	float depth = -viewSpace.z;
	float x_ndc = (f / aspect) * viewSpace.x / depth;
	float y_ndc = f * viewSpace.y / depth;

	return vec3{x_ndc, y_ndc, depth};
}

__device__
vec3 viewToNDC(vec3 viewSpace, float f, float aspect){
	float depth = -viewSpace.z;
	float x_ndc = (f / aspect) * viewSpace.x / depth;
	float y_ndc = f * viewSpace.y/ depth;

	return vec3{x_ndc, y_ndc, depth};
}

__device__
vec2 ndcToScreen(vec3 ndc, float width, float height){
	return vec2{
		(ndc.x * 0.5f + 0.5f) * width,
		(ndc.y * 0.5f + 0.5f) * height,
	};
}

__device__
void drawLine(int x0, int y0, int x1, int y1, RenderTarget target){

	if(x0 < 0 || x0 >= target.width) return;
	if(x1 < 0 || x1 >= target.width) return;
	if(y0 < 0 || y0 >= target.height) return;
	if(y1 < 0 || y1 >= target.height) return;

	int dx = abs(x1 - x0);
	int dy = abs(y1 - y0);

	int sx = (x0 < x1) ? 1 : -1;
	int sy = (y0 < y1) ? 1 : -1;

	int err = dx - dy;

	while (true) {
			
		int pixelID = x0 + y0 * target.width;
		if(pixelID >= 0 && pixelID < target.width * target.height)
		{ // draw


			float depth = 0.1f;
			uint64_t udepth = __float_as_uint(depth);
			uint64_t pixel = (udepth << 32ull) | 0xffff00ff;
			atomicMin(&target.colorbuffer[pixelID], pixel);
		}

		if (x0 == x1 && y0 == y1) break;

		int e2 = 2 * err;

		if (e2 > -dy) {
			err -= dy;
			x0 += sx;
		}
		if (e2 < dx) {
			err += dx;
			y0 += sy;
		}
	}
}

__device__
inline float cross(vec2 a, vec2 b){ 
	return a.x * b.y - a.y * b.x; 
}

__device__
inline vec4 toScreenCoord(vec3 p, mat4 &transform, int width, int height)
{
	vec4 pos = transform * vec4{p.x, p.y, p.z, 1.0f};

	pos.x = pos.x / pos.w;
	pos.y = pos.y / pos.w;

	vec4 imgPos = {
		(pos.x * 0.5f + 0.5f) * width,
		(pos.y * 0.5f + 0.5f) * height,
		pos.z,
		pos.w};

	return imgPos;
}

__device__
void drawLine(vec3 start, vec3 end, uint32_t color = 0xff0000ff){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int i = block.thread_rank();
	int iterations = 3;
	int max_samples = iterations * block.size();
	for(int j = 0; j < iterations; j++)
	{
		float w = float(i + j * block.size()) / float(max_samples);

		vec3 worldPos = (1.0f - w) * start + w * end;
		
		float f = c_target.proj[1][1];
		float aspect = float(c_target.width) / float(c_target.height);

		vec3 pos_ndc = worldToNDC(worldPos, c_target.view, f, aspect);
		vec2 pos_screen = ndcToScreen(pos_ndc, c_target.width, c_target.height);

		if(pos_ndc.x < -1.0f) continue;
		if(pos_ndc.x >  1.0f) continue;
		if(pos_ndc.y < -1.0f) continue;
		if(pos_ndc.y >  1.0f) continue;
		if(pos_ndc.z <  0.0f) continue;

		int2 pixelCoords = make_int2(pos_screen.x, pos_screen.y);
		int pixelID = pixelCoords.x + pixelCoords.y * c_target.width;
		pixelID = clamp(pixelID, 0, int(c_target.width * c_target.height) - 1);

		float depth = pos_ndc.z;

		if(depth > 0.0f){
			// depth = 0.1f;
			// int meshIndex = PACKMASK_MESHINDEX; 
			// int triangleIndex = PACKMASK_TRIANGLEINDEX;
			// uint64_t pixel = pack_pixel(depth, meshIndex, triangleIndex);
			uint64_t udepth = __float_as_uint(depth);
			uint64_t pixel = (udepth << 32) | color;
			atomicMin(&c_target.framebuffer[pixelID], pixel);
		}

		block.sync();
	}
}


extern "C" __global__
void kernel_drawBoundingBoxes(
	CMesh* meshes,
	uint32_t numMeshes,
	uint32_t* numProcessedBatches
) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	__shared__ int sh_meshIndex;
	__shared__ CMesh sh_mesh;

	if (block.thread_rank() == 0){
		sh_meshIndex = 0;
		sh_mesh = meshes[0];
	}

	grid.sync();

	// return;

	while (true){

		// Check which batch of triangles this block should render next.
		block.sync();
		if (block.thread_rank() == 0){
			sh_meshIndex = atomicAdd(numProcessedBatches, 1);
		}
		block.sync();

		sh_mesh = meshes[sh_meshIndex];

		// TODO: Consider return, labeled break, or goto instead to leave both loops
		if (sh_meshIndex >= numMeshes) break;

		// if(sh_mesh.isLoaded) continue;

		Box3 aabb = sh_mesh.aabb;
		vec3 worldMin = {Infinity, Infinity, Infinity};
		vec3 worldMax = {-Infinity, -Infinity, -Infinity};

		auto sample = [&](vec3 pos){
			vec3 worldPos = sh_mesh.world * vec4(pos, 1.0f);
			worldMin.x = min(worldMin.x, worldPos.x);
			worldMin.y = min(worldMin.y, worldPos.y);
			worldMin.z = min(worldMin.z, worldPos.z);
			worldMax.x = max(worldMax.x, worldPos.x);
			worldMax.y = max(worldMax.y, worldPos.y);
			worldMax.z = max(worldMax.z, worldPos.z);
		};

		sample({aabb.min.x, aabb.min.y, aabb.min.z});
		sample({aabb.min.x, aabb.min.y, aabb.max.z});
		sample({aabb.min.x, aabb.max.y, aabb.min.z});
		sample({aabb.min.x, aabb.max.y, aabb.max.z});
		sample({aabb.max.x, aabb.min.y, aabb.min.z});
		sample({aabb.max.x, aabb.min.y, aabb.max.z});
		sample({aabb.max.x, aabb.max.y, aabb.min.z});
		sample({aabb.max.x, aabb.max.y, aabb.max.z});

		block.sync();

		// if(sh_meshIndex == 1)
		{

			// if(block.thread_rank() == 0) printf("%.1f, %.1f, %.1f \n", aabb.min.x, aabb.min.y, aabb.min.z);

			// BOTTOM
			drawLine({worldMin.x, worldMin.y, worldMin.z}, {worldMax.x, worldMin.y, worldMin.z});
			drawLine({worldMin.x, worldMax.y, worldMin.z}, {worldMax.x, worldMax.y, worldMin.z});
			drawLine({worldMin.x, worldMin.y, worldMin.z}, {worldMin.x, worldMax.y, worldMin.z});
			drawLine({worldMax.x, worldMin.y, worldMin.z}, {worldMax.x, worldMax.y, worldMin.z});
			// BOTTOM to TOP
			drawLine({worldMin.x, worldMin.y, worldMin.z}, {worldMin.x, worldMin.y, worldMax.z});
			drawLine({worldMin.x, worldMax.y, worldMin.z}, {worldMin.x, worldMax.y, worldMax.z});
			drawLine({worldMax.x, worldMin.y, worldMin.z}, {worldMax.x, worldMin.y, worldMax.z});
			drawLine({worldMax.x, worldMax.y, worldMin.z}, {worldMax.x, worldMax.y, worldMax.z});
			// TOP
			drawLine({worldMin.x, worldMin.y, worldMax.z}, {worldMax.x, worldMin.y, worldMax.z});
			drawLine({worldMin.x, worldMax.y, worldMax.z}, {worldMax.x, worldMax.y, worldMax.z});
			drawLine({worldMin.x, worldMin.y, worldMax.z}, {worldMin.x, worldMax.y, worldMax.z});
			drawLine({worldMax.x, worldMin.y, worldMax.z}, {worldMax.x, worldMax.y, worldMax.z});
		}

	}
}

#ifndef __CUDACC_RTC__

__host__
void launch_drawBoundingBoxes(
	RenderTarget target,
	CMesh* meshes,
	uint32_t numMeshes,
	uint32_t* numProcessedBatches
) {
	
	int threads_per_block = 128;

	int device = 0;
	cudaGetDevice(&device);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);

	int numBlocksPerSm;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(
		&numBlocksPerSm,
		kernel_drawBoundingBoxes,
		threads_per_block,
		0
	);
	uint32_t max_supported_blocks = deviceProp.multiProcessorCount * numBlocksPerSm;
	
	dim3 dimGrid(max_supported_blocks);
	dim3 dimBlock(threads_per_block);

	cudaMemcpyToSymbol(c_target, &target, sizeof(target));

	void* kernel_args[] = {
		&meshes,
		&numMeshes,
		&numProcessedBatches,
	};

	cudaError_t err = cudaLaunchCooperativeKernel(
		(const void*)kernel_drawBoundingBoxes,
		dimGrid,
		dimBlock,
		kernel_args,
		0, // Shared memory (in bytes)
		0  // Stream (0 for default stream)
	);

	if (err != cudaSuccess) {
		fprintf(stderr, "Cooperative launch failed: %s\n", cudaGetErrorString(err));
	}
}

#endif


extern "C" __global__
void kernel_drawLine(
	vec3 start, 
	vec3 end,
	uint32_t color
) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	drawLine(start, end);
}

extern "C" __global__
void kernel_drawFrustum(mat4 view, mat4 proj, uint32_t color) {

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	float f = c_target.proj[1][1];
	float aspect = float(c_target.width) / float(c_target.height);

	float depth = 10.0f;

	mat4 viewInverse = glm::inverse(view);
	vec3 start = viewInverse * vec4{0.0f, 0.0f, 0.0f, 1.0f};
	vec3 end = viewInverse * vec4{0.0f, 0.0f, -depth, 1.0f};
	drawLine(start, end, 0xffff5555);


	vec3 r = viewInverse * vec4{depth / (f / aspect), 0.0f, -depth, 1.0f};
	vec3 l = viewInverse * vec4{-depth / (f / aspect), 0.0f, -depth, 1.0f};
	vec3 t = viewInverse * vec4{0.0f, depth / f, -depth, 1.0f};
	vec3 b = viewInverse * vec4{0.0f, -depth / f, -depth, 1.0f};
	drawLine(start, r, 0xff000000);
	drawLine(start, l, 0xff000000);
	drawLine(start, t, 0xff000000);
	drawLine(start, b, 0xff000000);

	vec3 rt = viewInverse * vec4{+depth / (f / aspect), +depth / f, -depth, 1.0f};
	vec3 lt = viewInverse * vec4{-depth / (f / aspect), +depth / f, -depth, 1.0f};
	vec3 rb = viewInverse * vec4{+depth / (f / aspect), -depth / f, -depth, 1.0f};
	vec3 lb = viewInverse * vec4{-depth / (f / aspect), -depth / f, -depth, 1.0f};
	drawLine(start, lt, 0xff000000);
	drawLine(start, rt, 0xff000000);
	drawLine(start, lb, 0xff000000);
	drawLine(start, rb, 0xff000000);

	drawLine(lt, rt, 0xff0000ff);
	drawLine(lb, rb, 0xff0000ff);
	drawLine(rb, rt, 0xff00ff00);
	drawLine(lb, lt, 0xff00ff00);

	// { // Test frustum plane extraction
	// 	mat4 viewProj = proj * view;

	// 	struct Plane {
	// 		glm::vec3 n; // normal
	// 		float d;
	// 	};

	// 	auto normalizePlane = [](const glm::vec4& p){
	// 		float len = glm::length(glm::vec3(p));
	// 		return Plane{ glm::vec3(p) / len, p.w / len };
	// 	};

	// 	glm::vec4 row0 = glm::row(viewProj,0);
	// 	glm::vec4 row1 = glm::row(viewProj,1);
	// 	glm::vec4 row2 = glm::row(viewProj,2);
	// 	glm::vec4 row3 = glm::row(viewProj,3);

	// 	Plane planes[6];
	// 	planes[0] = normalizePlane(row3 + row0); // Left
	// 	planes[1] = normalizePlane(row3 - row0); // Right
	// 	planes[2] = normalizePlane(row3 + row1); // Bottom
	// 	planes[3] = normalizePlane(row3 - row1); // Top
	// 	planes[4] = normalizePlane(row3 + row2); // Near
	// 	planes[5] = normalizePlane(row3 - row2); // Far

	// 	drawLine(start, start + 5.0f * planes[5].n, 0xffff00ff);

	// 	// for(int i = 0; i < 6; i++){
	// 	// 	Plane plane = planes[i];
	// 	// 	drawLine(start, start + 5.0f * plane.n, 0xffff00ff);
	// 	// }

	// }

}
