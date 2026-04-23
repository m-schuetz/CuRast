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
// #include <cooperative_groups/memcpy_async.h>

#include "./glm/glm/glm.hpp"
#include "./glm/glm/gtc/matrix_transform.hpp"
#include "./glm/glm/gtc/matrix_access.hpp"
#include "./glm/glm/gtx/transform.hpp"
#include "./glm/glm/gtc/quaternion.hpp"

#include "./utils.cuh"
#include "./HostDeviceInterface.h"
#include "../BitEdit.h"
#include "./rasterization_helpers.cuh"
// #include "../jpeg/JptInterface.cuh"

using glm::ivec2;
using glm::i8vec4;
using glm::vec4;

// uint32_t uvToMCUIndex(int width, int height, float u, float v) {
// 	int tx = (int(u * width) % width);
// 	int ty = (int(v * height) % height);
// 	return tx / 16 + ty / 16 * width / 16;
// }

__constant__ RenderTarget c_target;

__device__
vec4 getVertex(CMesh& mesh, uint32_t vertexIndex){
	vec4 position;
	if(!mesh.compressed){
		uint32_t resolvedIndex = mesh.indices[vertexIndex];
		position = vec4(mesh.positions[resolvedIndex], 1.0f);
	}else{
		uint32_t indexRange = mesh.index_max - mesh.index_min;
		uint64_t bitsPerIndex = ceil(log2f(float(indexRange + 1)));
		uint32_t resolvedIndex = BitEdit::readU32(mesh.indices, bitsPerIndex * vertexIndex, bitsPerIndex) + mesh.index_min;

		uint16_t* positions = (uint16_t*)mesh.positions;
		uint16_t X = positions[3 * resolvedIndex + 0];
		uint16_t Y = positions[3 * resolvedIndex + 1];
		uint16_t Z = positions[3 * resolvedIndex + 2];

		vec3 aabbSize = mesh.aabb.max - mesh.aabb.min;

		
		position.x = (float(X) / 65536.0f) * aabbSize.x + mesh.aabb.min.x;
		position.y = (float(Y) / 65536.0f) * aabbSize.y + mesh.aabb.min.y;
		position.z = (float(Z) / 65536.0f) * aabbSize.z + mesh.aabb.min.z;
		position.w = 1.0f;
	}

	return position;
};

__device__
vec4 getVertex_resolved(CMesh& mesh, uint32_t resolvedIndex){
	vec4 position;
	if(!mesh.compressed){
		position = vec4(mesh.positions[resolvedIndex], 1.0f);
	}else{
		uint16_t* positions = (uint16_t*)mesh.positions;
		uint16_t X = positions[3 * resolvedIndex + 0];
		uint16_t Y = positions[3 * resolvedIndex + 1];
		uint16_t Z = positions[3 * resolvedIndex + 2];

		vec3 aabbSize = mesh.aabb.max - mesh.aabb.min;

		
		position.x = (float(X) / 65536.0f) * aabbSize.x + mesh.aabb.min.x;
		position.y = (float(Y) / 65536.0f) * aabbSize.y + mesh.aabb.min.y;
		position.z = (float(Z) / 65536.0f) * aabbSize.z + mesh.aabb.min.z;
		position.w = 1.0f;
	}

	return position;
};

__device__
vec2 getUV(CMesh& mesh, uint32_t vertexIndex){

	bool isCompressed = mesh.positions == 0;

	if(mesh.uvs == nullptr){
		return vec2{0.0f, 0.0f};
	}else if(isCompressed){
		uint32_t indexRange = mesh.index_max - mesh.index_min;
		uint64_t bitsPerIndex = ceil(log2f(float(indexRange + 1)));
		uint32_t resolvedIndex = BitEdit::readU32(mesh.indices, bitsPerIndex * vertexIndex, bitsPerIndex) + mesh.index_min;
		return mesh.uvs[resolvedIndex];
	}else{
		uint32_t resolvedIndex = mesh.indices[vertexIndex];
		return mesh.uvs[resolvedIndex];
	}
};

__device__
vec2 getUV_resolved(CMesh& mesh, uint32_t resolvedIndex){

	bool isCompressed = mesh.positions == 0;

	if(mesh.uvs == nullptr){
		return vec2{0.0f, 0.0f};
	}else if(isCompressed){
		return mesh.uvs[resolvedIndex];
	}else{
		return mesh.uvs[resolvedIndex];
	}
};

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
	rgba[3] = 255;


	return color;
}

extern "C" __global__
void kernel_dummy(
	uint32_t* data
) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	if(grid.thread_rank() == 0) *data = 123;
}

extern "C" __global__
void kernel_clearFramebuffer(
	uint64_t* framebuffer,
	uint32_t numPixels,
	uint32_t clearColor,
	float clearDepth
) {
	auto grid = cg::this_grid();

	int pixelID = grid.thread_rank();
	if (pixelID >= numPixels) return;

	// uint64_t udepth = __float_as_uint(clearDepth);
	// uint64_t udepth = 0x00ffffff;
	// uint64_t pixel = udepth << 40;
	uint64_t pixel = 0xFFFFFFF0'00000000ULL;
	framebuffer[pixelID] = pixel;

	framebuffer[pixelID] = pixel;
}


__device__
float getEdlShadingFactor(uint64_t* colorbuffer, float depth, int x, int y, int distance){
	auto getNeighborDepth = [&](int x, int y) -> float{

		if(x < 0 || x >= c_target.width) return Infinity;
		if(y < 0 || y >= c_target.height) return Infinity;

		int pixelID = toFramebufferIndex(x, y, c_target.width);
		uint64_t pixel = colorbuffer[pixelID];

		float d = __uint_as_float(pixel >> 32);

		return d;
	};

	float sum = 0.0f;
	int numSamples = 8;
	for(int i = 0; i < numSamples; i++){
		float u = 2.0f * 3.1415f * float(i) / float(numSamples);
		float dx = float(distance) * cos(u);
		float dy = float(distance) * sin(u);
		
		sum += max(log2f(depth) - log2f(getNeighborDepth(x + dx, y + dy)), 0.0f);
	}

	// float response = sum / 4.0f;
	float response = sum / float(numSamples);
	float edlStrength = 0.9f;
	float shade = exp(-response * 300.0f * edlStrength);
	shade = clamp(shade, 0.3f, 1.0f);

	shade = shade * 0.8f + 0.2f;

	return shade;
}


__device__ __forceinline__ float fractf(float x) {
	return x - floorf(x);
}

// ── SSAO Helpers ─────────────────────────────────────────────────────────────

// Reconstruct view-space position from screen pixel + linear depth.
// Depth is the positive Z distance from the camera; proj contains focal lengths.
__device__ vec3 ssao_viewPos(float px, float py, float depth, int width, int height) {
	float ndc_x = (2.0f * (px + 0.5f) / float(width))  - 1.0f;
	float ndc_y = (2.0f * (py + 0.5f) / float(height)) - 1.0f;
	return vec3(
		ndc_x * depth / c_target.proj[0][0],
		ndc_y * depth / c_target.proj[1][1],
		depth
	);
}

// Project a view-space position back to screen pixel coordinates.
__device__ vec2 ssao_screenPos(vec3 P, int width, int height) {
	float inv_z = 1.0f / P.z;
	return vec2(
		((c_target.proj[0][0] * P.x * inv_z) + 1.0f) * 0.5f * float(width)  - 0.5f,
		((c_target.proj[1][1] * P.y * inv_z) + 1.0f) * 0.5f * float(height) - 0.5f
	);
}

// Reconstruct view-space normal from the depth buffer via edge-aware cross products.
// The returned normal points toward the camera (N.z <= 0 with depth = +Z into scene).
__device__ vec3 ssao_normal(int x, int y, float d0, uint64_t* cb, int w, int h) {
	auto getDepth = [&](int nx, int ny) -> float {
		nx = clamp(nx, 0, w - 1);
		ny = clamp(ny, 0, h - 1);
		return __uint_as_float(cb[ny * w + nx] >> 32);
	};

	float dr = getDepth(x + 1, y),  dl = getDepth(x - 1, y);
	float dd = getDepth(x, y + 1),  du = getDepth(x, y - 1);

	// Edge-aware: pick the neighbor with the smaller depth discontinuity
	bool use_right = fabsf(dr - d0) < fabsf(d0 - dl);
	bool use_down  = fabsf(dd - d0) < fabsf(d0 - du);
	int  sx = use_right ? +1 : -1;
	int  sy = use_down  ? +1 : -1;
	float dh = use_right ? dr : dl;
	float dv = use_down  ? dd : du;

	// Fallback if a neighbor is missing (silhouette against sky)
	if (isinf(dh) || isinf(dv)) return vec3(0.0f, 0.0f, -1.0f);

	vec3 P  = ssao_viewPos(float(x),       float(y),       d0, w, h);
	vec3 Ph = ssao_viewPos(float(x + sx),  float(y),       dh, w, h);
	vec3 Pv = ssao_viewPos(float(x),       float(y + sy),  dv, w, h);

	// Consistent forward-differences regardless of which neighbor was chosen
	vec3 dPh = (sx > 0) ? (Ph - P) : (P - Ph);
	vec3 dPv = (sy > 0) ? (Pv - P) : (P - Pv);

	// Cross product; negate so the normal points toward the camera
	vec3 N = normalize(cross(dPh, dPv));
	return (N.z > 0.0f) ? -N : N;
}

// ─────────────────────────────────────────────────────────────────────────────

// Scale-agnostic hemisphere SSAO with view-space normal reconstruction.
//
// The AO radius is world_radius = depth * RADIUS_FRACTION, which makes the
// projected screen-space footprint constant regardless of scene scale:
//   pixel_radius = RADIUS_FRACTION * proj[1][1] * height   (depth-independent)
// Doubling the scene (objects + distances) leaves the shading identical.
//
// Occlusion test uses a tangent-plane reference depth to prevent self-occlusion
// on steep triangles.  For a steep surface, depth changes rapidly across pixels,
// so S.z can easily be "behind" the surface at the reprojected pixel even though
// S is above the surface in 3D.  The tangent-plane reference (expected_z) gives
// the depth the *current* surface should have at each sample location; only
// geometry that protrudes above that plane counts as a real occluder.
__device__ float getSSAOShadingFactor(
	uint64_t* colorbuffer,
	float     center_depth,
	int x, int y,
	int width, int height,
	float /* focal_length — kept for API compatibility; use c_target.proj directly */
) {
	if (isinf(center_depth) || center_depth <= 0.0f) return 1.0f;

	// ── Tuning ──────────────────────────────────────────────────────────────
	const int   NUM_SAMPLES     = 32;
	const float RADIUS_FRACTION = 0.2025f; // world radius = 2.5 % of depth
	const float INTENSITY       = 1.1f;
	const float RANGE_MUL       = 2.5f;  // reject occluders farther than RANGE_MUL * radius
	const float BIAS_FRACTION   = 0.05f; // bias = BIAS_FRACTION * world_radius
	// ────────────────────────────────────────────────────────────────────────

	float world_radius = center_depth * RADIUS_FRACTION;
	float bias         = BIAS_FRACTION * world_radius;

	// Reconstruct view-space geometry at the center pixel
	vec3 P = ssao_viewPos(float(x), float(y), center_depth, width, height);
	vec3 N = ssao_normal(x, y, center_depth, colorbuffer, width, height);

	// Screen-space depth gradients for the tangent-plane reference.
	// Edge-aware: pick the side with the smaller depth jump to avoid warping
	// across silhouettes.
	auto getDepth = [&](int nx, int ny) -> float {
		nx = clamp(nx, 0, width  - 1);
		ny = clamp(ny, 0, height - 1);
		return __uint_as_float(colorbuffer[ny * width + nx] >> 32);
	};
	float dz_left  = center_depth - getDepth(x - 1, y);
	float dz_right = getDepth(x + 1, y) - center_depth;
	float dz_up    = center_depth - getDepth(x, y - 1);
	float dz_down  = getDepth(x, y + 1) - center_depth;
	float dz_dx = (fabsf(dz_left) < fabsf(dz_right)) ? dz_left : dz_right;
	float dz_dy = (fabsf(dz_up)   < fabsf(dz_down))  ? dz_up   : dz_down;

	// Build orthonormal tangent frame (T, B, N) around the surface normal
	vec3 up = (fabsf(N.z) < 0.95f) ? vec3(0.0f, 0.0f, 1.0f) : vec3(1.0f, 0.0f, 0.0f);
	vec3 T  = normalize(cross(up, N));
	vec3 B  = cross(N, T);

	// Per-pixel decorrelation — integer hash avoids the banding of smooth spatial functions
	uint32_t h = uint32_t(x) * 2246822519u ^ uint32_t(y) * 3266489917u;
	h ^= h >> 13; h *= 0xbf58476du; h ^= h >> 31;
	float rand_angle = float(h >> 8) * (6.28318530f / float(1 << 24));

	float occlusion          = 0.0f;
	const float INV_N        = 1.0f / float(NUM_SAMPLES);
	const float GOLDEN_ANGLE = 2.39996323f;

	for (int i = 0; i < NUM_SAMPLES; ++i) {
		// Cosine-weighted hemisphere sample via golden spiral.
		// Mapping fi → sin²(θ) gives cosine-weighted elevation distribution.
		float fi        = (float(i) + 0.5f) * INV_N;
		float sin_theta = sqrtf(fi);
		float cos_theta = sqrtf(1.0f - fi);
		float phi       = float(i) * GOLDEN_ANGLE + rand_angle;

		// Sample direction in view space (hemisphere oriented around N)
		vec3 dir = T * (sin_theta * cosf(phi))
		         + B * (sin_theta * sinf(phi))
		         + N *  cos_theta;

		// Distribute samples at increasing radii (sqrt → uniform area coverage)
		float r = world_radius * sqrtf(float(i + 1) * INV_N);
		// r = pow(r, 1.7f);
		vec3 S  = P + dir * r;   // sample point in view space

		if (S.z <= 0.001f) continue;   // behind camera

		// Reproject the sample and read the actual scene depth at that pixel
		vec2 sp = ssao_screenPos(S, width, height);
		int  sx = clamp(int(sp.x), 0, width  - 1);
		int  sy = clamp(int(sp.y), 0, height - 1);

		float actual_depth = __uint_as_float(colorbuffer[sy * width + sx] >> 32);
		if (isinf(actual_depth)) continue;   // sky / background

		// Tangent-plane reference: the depth the current surface is expected to
		// have at (sx, sy) if it were smooth.  Comparing actual_depth against
		// this — rather than against S.z — prevents the surface from occluding
		// itself on steep triangles where depth changes rapidly across pixels.
		float dx = sp.x - float(x);
		float dy = sp.y - float(y);
		float expected_z = center_depth + dz_dx * dx + dz_dy * dy;

		// Positive dz means geometry protrudes above the tangent plane → real occluder
		float dz = expected_z - actual_depth;

		if (dz > bias && dz < world_radius * RANGE_MUL) {
			float range_falloff = 1.0f - dz / (world_radius * RANGE_MUL);
			occlusion += range_falloff * cos_theta;   // cosine-weighted contribution
		}
	}

	return clamp(1.0f - occlusion * INV_N * INTENSITY, 0.0f, 1.0f);
}

extern "C" __global__
void kernel_enlarge(
	cudaSurfaceObject_t gl_desktop,
	float* ssaoShadeBuffer,
	uint64_t* fbo_enlarge,
	int width, 
	int height,
	int mouseX,
	int mouseY,
	DeviceState* state,
	bool enableEDL,
	bool enableSSAO
) {
	auto grid = cg::this_grid();

	int numPixels = width * height;

	int n = 20;
	uint64_t DEFAULT = uint64_t(__float_as_uint(INFINITY)) << 32 | 0xff0000ff;

	// Enlarge horizontally, write result to temp buffer
	process(numPixels, [&](int pixelID){
		int x = pixelID % c_target.width;
		int y = pixelID / c_target.width;

		uint64_t closest = DEFAULT;
		for(int dx = -n; dx <= n; dx++){
			
			int sx = x + dx;
			int sy = y;

			if(sx < 0 || sx >= c_target.width) continue;

			int sourcePixelID = sx + sy * c_target.width;
			uint64_t pixel = c_target.colorbuffer[sourcePixelID];
			
			// add offsets to the depth of points, based on how far they are from the center
			float depth = __uint_as_float(pixel >> 32);
			if(!isinf(depth)){
				uint64_t color = pixel & 0xffffffff;
				float f = 0.01f * abs(dx * dx) + 1.0f;
				depth = depth * f;
				pixel = (uint64_t(__float_as_uint(depth)) << 32) | color;
			}

			closest = min(closest, pixel);
		}

		if(closest != DEFAULT){
			fbo_enlarge[pixelID] = closest;
		}else{
			fbo_enlarge[pixelID] = c_target.colorbuffer[pixelID];
		}
	});

	grid.sync();

	// Enlarge vertically, write result back in main color buffer
	process(numPixels, [&](int pixelID){
		int x = pixelID % c_target.width;
		int y = pixelID / c_target.width;

		uint64_t closest = DEFAULT;
		for(int dy = -n; dy <= n; dy++){
			
			int sx = x;
			int sy = y + dy;

			if(sy < 0 || sy >= c_target.height) continue;

			int sourcePixelID = sx + sy * c_target.width;
			uint64_t pixel = fbo_enlarge[sourcePixelID];

			// add offsets to the depth of points, based on how far they are from the center
			float depth = __uint_as_float(pixel >> 32);
			if(!isinf(depth)){
				uint64_t color = pixel & 0xffffffff;
				float f = 0.01f * abs(dy * dy) + 1.0f;
				depth = depth * f;
				pixel = (uint64_t(__float_as_uint(depth)) << 32) | color;
			}

			closest = min(closest, pixel);
		}

		if(closest != DEFAULT){
			c_target.colorbuffer[pixelID] = closest;
		}else{
			c_target.colorbuffer[pixelID] = fbo_enlarge[pixelID];
		}
	});

}




extern "C" __global__
void kernel_ssaoOcclusion(
	uint64_t* occlusionBuffer,
	float* ssaoShadeBuffer
){
	auto grid = cg::this_grid();
	int x = grid.thread_index().x;
	int y = grid.thread_index().y;

	if(x >= c_target.width || y >= c_target.height) return;

	int pixelID = toFramebufferIndex(x, y, c_target.width);
	uint64_t pixel = c_target.colorbuffer[pixelID];
	float depth = __uint_as_float(pixel >> 32);
	float focal_length = c_target.proj[1][1];
	float ssao = getSSAOShadingFactor(c_target.colorbuffer, depth, x, y, c_target.width, c_target.height, focal_length);

	uint64_t occ = uint64_t(__float_as_uint(depth)) << 32 | uint64_t(__float_as_uint(ssao));

	occlusionBuffer[pixelID] = occ;
}

extern "C" __global__
void kernel_ssaoBlur(
	uint64_t* occlusionBuffer,
	float* ssaoShadeBuffer
){
	auto grid = cg::this_grid();
	int x = grid.thread_index().x;
	int y = grid.thread_index().y;

	int width = c_target.width;
	int height = c_target.height;
	
	if(x >= width || y >= height) return;

	int centerIdx = y * width + x;

	// Fetch center depth for bilateral weighting
	float center_depth = __uint_as_float(occlusionBuffer[centerIdx] >> 32);

	// Background / sky pixel — no occlusion
	if(isinf(center_depth)){
		ssaoShadeBuffer[centerIdx] = 1.0f;
		return;
	}

	// Separable 1D Gaussian weights for offsets {-3, -2, -1, 0, +1, +2, +3} (sigma ≈ 1)
	const int   RADIUS = 3;
	const float gaussian[7] = { 0.015625f, 0.09375f, 0.234375f, 0.3125f, 0.234375f, 0.09375f, 0.015625f };

	// Depth-relative bilateral sigma: keeps edges sharp regardless of viewing distance.
	// 0.002 was too tight (rejects all neighbors), 0.02 gives smooth blending.
	const float DEPTH_SIGMA = center_depth * 0.2f;
	const float inv2sigma2  = 1.0f / (2.0f * DEPTH_SIGMA * DEPTH_SIGMA);

	float sum         = 0.0f;
	float totalWeight = 0.0f;

	for(int dy = -RADIUS; dy <= RADIUS; dy++)
	for(int dx = -RADIUS; dx <= RADIUS; dx++)
	{
		int nx = clamp(x + dx, 0, width  - 1);
		int ny = clamp(y + dy, 0, height - 1);
		int samplePixelID = ny * width + nx;

		uint64_t occ = occlusionBuffer[samplePixelID];
		float sample_depth     = __uint_as_float(occ >> 32);
		float sample_occlusion = __uint_as_float(occ & 0xffffffff);

		float depth_diff   = sample_depth - center_depth;
		float depth_weight = expf(-depth_diff * depth_diff * inv2sigma2);
		depth_weight = 1.0f;

		float w = gaussian[dx + RADIUS] * gaussian[dy + RADIUS] * depth_weight;

		sum         += w * sample_occlusion;
		totalWeight += w;
	}

	float shade = 1.0f;
	if(totalWeight > 0.0f){
		shade = sum / totalWeight;
	}

	// shade = shade / 2.0f + 0.5f;

	ssaoShadeBuffer[centerIdx] = shade;
	// ssaoShadeBuffer[centerIdx] = 1.0f;
}


extern "C" __global__
void kernel_resolve_visbuffer_to_colorbuffer2D(
	CMesh* meshes,
	uint32_t numMeshes,
	uint64_t* triangleCountPrefixSum,
	int mouseX,
	int mouseY,
	DeviceState* state,
	RasterizationSettings rasterizationSettings,
	JpegPipeline jpp
) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int x = grid.thread_index().x;
	int y = grid.thread_index().y;
	// int pixelID = x + c_target.width * y;
	int pixelID = toFramebufferIndex(x, y, c_target.width);

	if(x >= c_target.width) return;
	if(y >= c_target.height) return;

	auto computeRayDir = [](float x, float y, float width, float height, mat4 view, mat4 proj){
		float u = 2.0f * x / width - 1.0f;
		float v = 2.0f * y / height - 1.0f;

		mat4 viewI = inverse(view);
		vec3 origin = viewI * vec4(0.0f, 0.0f, 0.0f, 1.0f);
		vec3 rayDir_view = normalize(vec3{
			1.0f / proj[0][0] * u,
			1.0f / proj[1][1] * v,
			-1.0f
		});
		vec3 rayDir_world = normalize(vec3(viewI * vec4(rayDir_view, 1.0f)) - origin);

		return rayDir_world;
	};

	mat4 viewI = inverse(c_target.view);
	vec3 origin = viewI * vec4(0.0f, 0.0f, 0.0f, 1.0f);
	vec3 rayDir    = computeRayDir(float(x) + 0.5f, float(y) + 0.5f, c_target.width, c_target.height, c_target.view, c_target.proj);
	vec3 rayDir_10 = computeRayDir(float(x) + 1.5f, float(y) + 0.5f, c_target.width, c_target.height, c_target.view, c_target.proj);
	vec3 rayDir_01 = computeRayDir(float(x) + 0.5f, float(y) + 1.5f, c_target.width, c_target.height, c_target.view, c_target.proj);
	vec3 rayDir_11 = computeRayDir(float(x) + 1.5f, float(y) + 1.5f, c_target.width, c_target.height, c_target.view, c_target.proj);

	uint64_t pixel = c_target.framebuffer[pixelID];
	uint64_t pixel_colorbuffer = c_target.colorbuffer[pixelID];
	
	float depth;
	uint32_t meshIndex;
	uint64_t totalTriangleIndex = 0;
	unpack_pixel(pixel, &depth, &totalTriangleIndex);

	float depth_colorbuffer = __uint_as_float(pixel_colorbuffer >> 32);

	if(depth_colorbuffer < depth) return;

	{// Find mesh corresponding to totalTriangleIndex via binary search
		int left = 0;
		int right = numMeshes - 1;
		int searchResult = 0;

		while(left <= right){
			int mid = left + (right - left) / 2;
			
			int64_t before = triangleCountPrefixSum[mid];
			int64_t after = (mid == (numMeshes - 1)) ? 0xffffffffff : triangleCountPrefixSum[mid + 1];

			if(totalTriangleIndex >= before && totalTriangleIndex < after){
				searchResult = mid;
				break;
			}else if(after <= totalTriangleIndex){
				left = mid + 1;
			}else{
				right = mid - 1;
			}

			if(mid == left) break;
		}

		meshIndex = searchResult;
	}
	
	uint32_t color = 0;
	uint8_t *rgb = (uint8_t *)&color;

	if(!isinf(depth)){
		CMesh mesh = meshes[meshIndex];
		uint32_t triangleIndex = totalTriangleIndex - mesh.cummulativeTriangleCount;

		if(x == mouseX && y == mouseY){
			state->hovered_meshId = mesh.id;
			state->hovered_triangleIndex = triangleIndex;
		}

		triangleIndex = min(triangleIndex, mesh.numTriangles - 1);

		if(!mesh.isLoaded){
			color = 0xffff00ff;
			goto shade_finished;
		}

		// resolve indices first
		uint32_t i0, i1, i2;
		if(!mesh.compressed){
			if(mesh.indices){
				i0 = mesh.indices[3 * triangleIndex + 0];
				i1 = mesh.indices[3 * triangleIndex + 1];
				i2 = mesh.indices[3 * triangleIndex + 2];
			}else{
				i0 = 3 * triangleIndex + 0;
				i1 = 3 * triangleIndex + 1;
				i2 = 3 * triangleIndex + 2;
			}
		}else{
			uint32_t indexRange = mesh.index_max - mesh.index_min;
			uint64_t bitsPerIndex = ceil(log2f(float(indexRange + 1)));
			i0 = BitEdit::readU32(mesh.indices, bitsPerIndex * (3 * triangleIndex + 0), bitsPerIndex) + mesh.index_min;
			i1 = BitEdit::readU32(mesh.indices, bitsPerIndex * (3 * triangleIndex + 1), bitsPerIndex) + mesh.index_min;
			i2 = BitEdit::readU32(mesh.indices, bitsPerIndex * (3 * triangleIndex + 2), bitsPerIndex) + mesh.index_min;
		}

		// Then load geometry data
		vec3 a_object = getVertex_resolved(mesh, i0);
		vec3 b_object = getVertex_resolved(mesh, i1);
		vec3 c_object = getVertex_resolved(mesh, i2);
		vec2 uv_a = getUV_resolved(mesh, i0);
		vec2 uv_b = getUV_resolved(mesh, i1);
		vec2 uv_c = getUV_resolved(mesh, i2);
		// ---------------------------------------------------


		// mat4 worldView = c_target.view * mesh.world;
		vec3 a_world = mesh.world * vec4(a_object, 1.0f);
		vec3 b_world = mesh.world * vec4(b_object, 1.0f);
		vec3 c_world = mesh.world * vec4(c_object, 1.0f);

		vec3 a_view = c_target.view * vec4(a_world, 1.0f);
		vec3 b_view = c_target.view * vec4(b_world, 1.0f);
		vec3 c_view = c_target.view * vec4(c_world, 1.0f);

		vec3 N = normalize(cross(b_view - a_view, c_view - a_view));
		
		// For mip map level:
		// - Find triangle intersection in current pixel for current uv
		// - Also find triangle intersection for pixels to the right and the top to 
		//   compute the change of uv coordinates.
		// - But actually do plane intersection because triangle may not extend to adjacent pixels
		// float t    = intersectTriangle(origin, rayDir, a, b, c, false);
		float t, t_10, t_01, t_11;
		{
			vec3 edge1 = b_world - a_world;
			vec3 edge2 = c_world - a_world;
			vec3 normal = normalize(cross(edge1, edge2));

			float d = dot(a_world, normal);

			t    = intersectPlane(origin, rayDir,    normal, -d);
			t_10 = intersectPlane(origin, rayDir_10, normal, -d);
			t_01 = intersectPlane(origin, rayDir_01, normal, -d);
			t_11 = intersectPlane(origin, rayDir_11, normal, -d);
		}

		// Store 2-component barycentric coordinates because the 3rd component is deducted from the other two.
		vec2 stv, stv_10, stv_01, stv_11;
		{

			vec3 v0 = b_world - a_world;
			vec3 v1 = c_world - a_world;

			float d00 = dot(v0, v0);
			float d01 = dot(v0, v1);
			float d11 = dot(v1, v1);
			float denom = d00 * d11 - d01 * d01;
			float denomI = 1.0f / denom;
			
			auto computeSTV = [&](float t, vec3 rayDir){
				vec3 p = origin + t * rayDir;
				vec3 v2 = p - a_world;

				float d20 = dot(v2, v0);
				float d21 = dot(v2, v1);

				vec2 stv;
				stv.x = (d11 * d20 - d01 * d21) * denomI;
				stv.y = (d00 * d21 - d01 * d20) * denomI;

				return stv;
			};
			
			stv    = computeSTV(t, rayDir);
			stv_10 = computeSTV(t_10, rayDir_10);
			stv_01 = computeSTV(t_01, rayDir_01);
			stv_11 = computeSTV(t_11, rayDir_11);
		}
		
		vec2 uv    = uv_a * (1.0f - stv.x    - stv.y   ) + uv_b * stv.x    + uv_c * stv.y;
		vec2 uv_10 = uv_a * (1.0f - stv_10.x - stv_10.y) + uv_b * stv_10.x + uv_c * stv_10.y;
		vec2 uv_01 = uv_a * (1.0f - stv_01.x - stv_01.y) + uv_b * stv_01.x + uv_c * stv_01.y;
		vec2 uv_11 = uv_a * (1.0f - stv_11.x - stv_11.y) + uv_b * stv_11.x + uv_c * stv_11.y;

		vec2 uvmax = {
			max(max(uv.x, uv_10.x), max(uv_01.x, uv_11.x)),
			max(max(uv.y, uv_10.y), max(uv_01.y, uv_11.y)),
		};
		vec2 uvmin = {
			min(min(uv.x, uv_10.x), min(uv_01.x, uv_11.x)),
			min(min(uv.y, uv_10.y), min(uv_01.y, uv_11.y)),
		};

		// Compute Mip Map level by delta of texcoords
		uv.x = uv.x - floor(uv.x);
		uv.y = uv.y - floor(uv.y);
		vec2 dx = (uvmax.x - uvmin.x) * vec2{mesh.texture.width, mesh.texture.height};
		vec2 dy = (uvmax.y - uvmin.y) * vec2{mesh.texture.width, mesh.texture.height};
		float mipLevel = 0.5f * log2(max(dot(dx, dx), dot(dy, dy)));
		mipLevel = int(mipLevel);
		mipLevel = min(mipLevel, 7.0f);

		// pick a mip map level
		int target_level = clamp(mipLevel, 0.0f, 7.0f);
		// target_level = 0;
		uint32_t* data = mesh.texture.data;
		uint32_t width = mesh.texture.width;
		uint32_t height = mesh.texture.height;
		for(int i = 1; i <= target_level; i++){
			data = data + width * height;
			width = (width + 2 - 1) / 2;
			height = (height + 2 - 1) / 2;
		}
		color = 0xffff00ff;
		
		// TODO: These blocks here are a major performance bottleneck. 
		// Removing them cuts registers from 88 to 56 and improves perf from 1.3ms to 0.88ms
		if(rasterizationSettings.displayAttribute == DisplayAttribute::NONE){
			color = 0xffffffff;
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::TRIANGLE_ID){
			color = triangleIndex * 12345678;
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::MESH_ID){
			color = (mesh.id + 1) * 12345678;
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::TEXTURE && mesh.uvs){
			if(!mesh.texture.huffmanTables){
				color = sampleColor_linear(data, width, height, uv);
			}else{
				// uint32_t mcu = uvToMCUIndex(width, height, uv.x, uv.y);
				int tx = (int(uv.x * width) % width);
				int ty = (int(uv.y * height) % height);
				uint32_t mcu_x = tx / 16;
				uint32_t mcu_y = ty / 16;
				uint32_t mcus_x = width / 16;
				uint32_t mcu = mcu_x + mcus_x * mcu_y;
				uint32_t key = pack_mcuidx_textureidx_miplevel(mcu, mesh.texture.handle, uint32_t(mipLevel));

				// to avoid contention, make sure that for any MCU, only one thread per warp continues.
				{ 
					// mask of warp threads with same key
					// auto block = cg::this_thread_block();
					// auto warp = cg::tiled_partition<32>(block);
					auto warp = cg::coalesced_threads(); 
					uint32_t mask = warp.match_any(key);

					// find the lowest lane between threads with the same key
					int winningLane = __ffs(mask) - 1;

					// return early because another thread handles this MCU
					if(warp.thread_rank() != winningLane) goto shade_finished;
				}

				// Reserve a spot in the hash map. The value, the TB-Slot, will be acquired and set by the decode kernel
				bool alreadyExists = false;
				int location = 0;
				jpp.decodedMcuMap.set(key, 0, &location, &alreadyExists);

				if(!alreadyExists){
					// Add MCU to decoder queue
					uint32_t decodeIndex = atomicAdd(jpp.toDecodeCounter, 1);
					jpp.toDecode[decodeIndex] = key;
				}else{
					// MCU is in cache - flag as visible
					atomicOr(&jpp.decodedMcuMap.entries[location], 0x00000000'ff000000);
				}

			}
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::UV && mesh.uvs){
			rgb[0] = fmodf(uv.x - floor(uv.x), 1.0f) * 256.0f;
			rgb[1] = fmodf(uv.y - floor(uv.y), 1.0f) * 256.0f;
			rgb[2] = 0;
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::NORMAL && mesh.normals){
			
			vec3 n0 = mesh.normals[i0];
			vec3 n1 = mesh.normals[i1];
			vec3 n2 = mesh.normals[i2];

			n0 = mesh.world * vec4(n0, 0.0f);
			n1 = mesh.world * vec4(n1, 0.0f);
			n2 = mesh.world * vec4(n2, 0.0f);

			vec3 N = n0 * (1.0f - stv.x - stv.y) + n1 * stv.x + n2 * stv.y;
			N = normalize(N);

			rgb[0] = clamp(N.x * 255.0f, 0.0f, 255.0f);
			rgb[1] = clamp(N.y * 255.0f, 0.0f, 255.0f);
			rgb[2] = clamp(N.z * 255.0f, 0.0f, 255.0f);
			
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::NORMAL && !mesh.normals){
			vec3 edge1 = b_world - a_world;
			vec3 edge2 = c_world - a_world;
			vec3 N = normalize(cross(edge1, edge2));

			rgb[0] = clamp(N.x * 255.0f, 0.0f, 255.0f);
			rgb[1] = clamp(N.y * 255.0f, 0.0f, 255.0f);
			rgb[2] = clamp(N.z * 255.0f, 0.0f, 255.0f);
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::VERTEX_COLORS && mesh.colors){
			uint32_t C_a = mesh.colors[i0];
			uint32_t C_b = mesh.colors[i1];
			uint32_t C_c = mesh.colors[i2];
			color = C_a;

			auto toVec3 = [](uint32_t C){
				return vec3{
					(C >>  0) & 0xff,
					(C >>  8) & 0xff,
					(C >> 16) & 0xff,
				};
			};

			vec3 c_a = toVec3(C_a);
			vec3 c_b = toVec3(C_b);
			vec3 c_c = toVec3(C_c);

			vec3 c = c_a * (1.0f - stv.x - stv.y) + c_b * stv.x + c_c * stv.y;
			rgb[0] = clamp(c.x, 0.0f, 255.0f);
			rgb[1] = clamp(c.y, 0.0f, 255.0f);
			rgb[2] = clamp(c.z, 0.0f, 255.0f);
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::STAGE){
			// We don't actually know which stage a pixel was rasterized by, 
			// but we can assume based on the triangle's size and whether it intersects the near plane.
			float f = c_target.proj[1][1];
			float aspect = float(c_target.width) / float(c_target.height);

			vec3 a_ndc = viewToNDC(a_view, f, aspect);
			vec3 b_ndc = viewToNDC(b_view, f, aspect);
			vec3 c_ndc = viewToNDC(c_view, f, aspect);

			bool isNontrivial = (a_ndc.z <= 0.0f || b_ndc.z <= 0.0f || c_ndc.z <= 0.0f);

			vec2 a_screen = ndcToScreen(a_ndc, c_target.width, c_target.height);
			vec2 b_screen = ndcToScreen(b_ndc, c_target.width, c_target.height);
			vec2 c_screen = ndcToScreen(c_ndc, c_target.width, c_target.height);

			// screen-space bounding box of triangle
			float min_x = min(a_screen.x, min(b_screen.x, c_screen.x));
			float max_x = max(a_screen.x, max(b_screen.x, c_screen.x));
			float min_y = min(a_screen.y, min(b_screen.y, c_screen.y));
			float max_y = max(a_screen.y, max(b_screen.y, c_screen.y));

			// clip to screen
			min_x = max(min_x, 0.0f);
			max_x = min(max_x, float(c_target.width - 1));
			min_y = max(min_y, 0.0f);
			max_y = min(max_y, float(c_target.height - 1));

			int size_x = ceil(max_x) - floor(min_x);
			int size_y = ceil(max_y) - floor(min_y);
			int numFragments = size_x * size_y;

			int stage = 0;

			if(numFragments <= THRESHOLD_SMALL)      stage = 0;
			else if(numFragments <= THRESHOLD_LARGE) stage = 1;
			else                                     stage = 2;
			
			if(isNontrivial) stage = 2;

			color = 0xffffaaff;
			if(stage == 0) color = SCHEME_SPECTRAL[9];
			if(stage == 1) color = SCHEME_SPECTRAL[7];
			if(stage == 2) color = SCHEME_SPECTRAL[3];

			// if(numFragments <= 1)           color = SCHEME_SPECTRAL[9];
			// else if(numFragments <= 4)      color = SCHEME_SPECTRAL[8];
			// else if(numFragments <= 16)      color = SCHEME_SPECTRAL[7];
			// else if(numFragments <= 64)     color = SCHEME_SPECTRAL[6];
			// else if(numFragments <= 256)    color = SCHEME_SPECTRAL[5];
			// else if(numFragments <= 1024)    color = SCHEME_SPECTRAL[4];
			// else if(numFragments <= 4096)   color = SCHEME_SPECTRAL[3];
			// else if(numFragments <= 16384)   color = SCHEME_SPECTRAL[2];
			// else if(numFragments <= 262144)  color = SCHEME_SPECTRAL[1];
			// else                            color = SCHEME_SPECTRAL[0];



		}else{
			color = 0xff0000ff;
		}
		color = color | 0xff000000;

		if(rasterizationSettings.enableDiffuseLighting && mesh.normals){

			vec3 n0 = mesh.normals[i0];
			vec3 n1 = mesh.normals[i1];
			vec3 n2 = mesh.normals[i2];

			n0 = mesh.world * vec4(n0, 0.0f);
			n1 = mesh.world * vec4(n1, 0.0f);
			n2 = mesh.world * vec4(n2, 0.0f);

			vec3 N = n0 * (1.0f - stv.x - stv.y) + n1 * stv.x + n2 * stv.y;
			vec3 L = normalize(vec3{1.0f, 1.0f, 1.0f});
			float d = max(dot(N, L), 0.0f);

			vec3 ambient = {0.5, 0.5f, 0.5f};
			vec3 diffuse = {0.7f, 0.7f, 0.7f};

			rgb[0] = clamp((d * diffuse.x + ambient.x) * float(rgb[0]), 0.0f, 255.0f);
			rgb[1] = clamp((d * diffuse.y + ambient.y) * float(rgb[1]), 0.0f, 255.0f);
			rgb[2] = clamp((d * diffuse.z + ambient.z) * float(rgb[2]), 0.0f, 255.0f);
		}

		// Highlight hovered mesh: draw borders
		if(rasterizationSettings.enableObjectPicking && mesh.id == state->hovered_meshId){

			bool isInside = true;

			for(int dx : {-1, 0, 1})
			for(int dy : {-1, 0, 1})
			{

				int nx = clamp(x + dx, 0, c_target.width - 1);
				int ny = clamp(y + dy, 0, c_target.height - 1);
				int neighbor_pixelID = toFramebufferIndex(nx, ny, c_target.width);
				uint64_t neighbor_pixel = c_target.framebuffer[neighbor_pixelID];

				float neighbor_depth;
				uint32_t neighbor_meshIndex;
				uint64_t neighbor_totalTriangleIndex = 0;
				unpack_pixel(neighbor_pixel, &neighbor_depth, &neighbor_totalTriangleIndex);

				// Check if neighbor's total triangle index is within this pixel's mesh
				if(neighbor_totalTriangleIndex < mesh.cummulativeTriangleCount) isInside = false;
				if(neighbor_totalTriangleIndex >= mesh.cummulativeTriangleCount + mesh.numTriangles) isInside = false;
			}

			if(!isInside){
				color = 0xff0000ff;
			}
		}
	}else{
		color = 0;
		// rgb[0] = 0.1f * 256.0f;
		// rgb[1] = 0.2f * 256.0f;
		// rgb[2] = 0.3f * 256.0f;

		if(x == mouseX && y == mouseY){
			state->hovered_meshId = 0xffffffff;
			state->hovered_triangleIndex = 0xffffffff;
		}
	}

	shade_finished:

	// WIREFRAME
	if(rasterizationSettings.showWireframe){
		uint32_t numPixels = c_target.width * c_target.height;
		int pid_00 = clamp(toFramebufferIndex(x + 0, y + 0, c_target.width), 0u, numPixels - 1);
		int pid_01 = clamp(toFramebufferIndex(x + 0, y + 1, c_target.width), 0u, numPixels - 1);
		int pid_10 = clamp(toFramebufferIndex(x + 1, y + 0, c_target.width), 0u, numPixels - 1);

		uint32_t p_00 = c_target.framebuffer[pid_00] & 0xffffffff;
		uint32_t p_01 = c_target.framebuffer[pid_01] & 0xffffffff;
		uint32_t p_10 = c_target.framebuffer[pid_10] & 0xffffffff;

		if(p_00 != p_10) color = 0xffff00ff;
		if(p_00 != p_01) color = 0xffff00ff;

		// for(int dx : {0, 1})
		// for(int dy : {0, 1})
		for(int dx : {-1, 0, 1})
		for(int dy : {-1, 0, 1})
		{
			int pid_neighbor = clamp(toFramebufferIndex(x + dx, y + dy, c_target.width), 0u, numPixels - 1);
			uint32_t p_neighbor = c_target.framebuffer[pid_neighbor] & 0xffffffff;

			if(p_00 != p_neighbor) color = 0xffff00ff;
		}
	}

	// 64x64 tile grid
	// if(x % 64 == 0 || y % 64 == 0){
	// 	color = 0xffff00ff;
	// }

	

	if(depth != Infinity){
		uint64_t udepth = __float_as_uint(depth);
		uint64_t pixel = (udepth << 32) | color;
		c_target.colorbuffer[pixelID] = pixel;
	}else{
		c_target.colorbuffer[pixelID] = uint64_t(__float_as_uint(INFINITY)) << 32 | 0;
	}
}

extern "C" __global__
void kernel_resolve_colorbuffer_to_opengl_2D(
	cudaSurfaceObject_t gl_desktop,
	float* ssaoShadeBuffer,
	int width, 
	int height,
	int mouseX,
	int mouseY,
	DeviceState* state,
	bool enableEDL,
	bool enableSSAO,
	uint32_t backgroundColor
) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	RenderTarget& source = c_target;

	if(width == source.width && height == source.height){
		int x = grid.thread_index().x;
		int y = grid.thread_index().y;
		int pixelID = toFramebufferIndex(x, y, source.width);

		if(x >= source.width) return;
		if(y >= source.height) return;

		uint64_t pixel = c_target.colorbuffer[pixelID];
		float depth = __uint_as_float(pixel >> 32);
		uint32_t color = pixel & 0xffffffff;

		float edl = 1.0f;
		float ssao = 1.0f;

		if(enableEDL){
			edl = getEdlShadingFactor(c_target.colorbuffer, depth, x, y, 1);
		}

		if(enableSSAO){
			ssao = ssaoShadeBuffer[pixelID] * 0.4f + 0.6f;
		}

		if(isinf(depth)) color = backgroundColor;

		float shade = edl * ssao;
		uint8_t* rgba = (uint8_t*)&color;
		rgba[0] = shade * float(rgba[0]);
		rgba[1] = shade * float(rgba[1]);
		rgba[2] = shade * float(rgba[2]);

		surf2Dwrite(color, gl_desktop, x * 4, y);
	}else{
		int target_x = grid.thread_index().x;
		int target_y = grid.thread_index().y;
		int pixelID = toFramebufferIndex(target_x, target_y, width);

		if(target_x >= width) return;
		if(target_y >= height) return;

		vec4 color = {0.0f, 0.0f, 0.0f, 0.0f};
		float edl = 0.0f;
		float ssao = 0.0f;

		int supersamplingFactor = source.width / width;

		int numSamples = 0;
		for(int dx = 0; dx < supersamplingFactor; dx++)
		for(int dy = 0; dy < supersamplingFactor; dy++)
		{
			int source_x = supersamplingFactor * target_x + dx;
			int source_y = supersamplingFactor * target_y + dy;
			int sourcePixelID = toFramebufferIndex(source_x, source_y, source.width);

			uint64_t pixel = c_target.colorbuffer[sourcePixelID];
			float depth = __uint_as_float(pixel >> 32);
			uint32_t C = pixel & 0xffffffff;
			color.r += (C >>  0) & 0xff;
			color.g += (C >>  8) & 0xff;
			color.b += (C >> 16) & 0xff;

			if(isinf(depth)){
				color.r += (BACKGROUND_COLOR >>  0) & 0xff;
				color.g += (BACKGROUND_COLOR >>  8) & 0xff;
				color.b += (BACKGROUND_COLOR >> 16) & 0xff;
			}

			if(enableEDL){
				edl += getEdlShadingFactor(c_target.colorbuffer, depth, source_x, source_y, supersamplingFactor);
			}
			if(enableSSAO){
				ssao += ssaoShadeBuffer[sourcePixelID];
			}
			numSamples++;
		}

		if(enableEDL){
			edl = edl / float(numSamples);
		}else{
			edl = 1.0f;
		}

		if(enableSSAO){
			ssao = ssao / float(numSamples);
		}else{
			ssao = 1.0f;
		}
		

		float shade = edl * ssao;
		color = shade * color / float(numSamples);
		uint32_t C;
		uint8_t* rgba = (uint8_t*)&C;
		rgba[0] = clamp(color.r, 0.0f, 255.0f);
		rgba[1] = clamp(color.g, 0.0f, 255.0f);
		rgba[2] = clamp(color.b, 0.0f, 255.0f);
		rgba[3] = 255;

		surf2Dwrite(C, gl_desktop, target_x * 4, target_y);
	}
}


extern "C" __global__
void kernel_resolve_colorbuffer_to_screenshot(
	uint32_t* screenshot,
	float* ssaoShadeBuffer,
	bool enableEDL,
	bool enableSSAO,
	int windowWidth,
	int windowHeight,
	uint32_t backgroundColor
) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	RenderTarget& source = c_target;

	int x = grid.thread_index().x;
	int y = grid.thread_index().y;
	int pixelID = toFramebufferIndex(x, y, source.width);

	if(x >= source.width) return;
	if(y >= source.height) return;

	uint64_t pixel = c_target.colorbuffer[pixelID];
	float depth = __uint_as_float(pixel >> 32);
	uint32_t color = pixel & 0xffffffff;

	float edl = 1.0f;
	float ssao = 1.0f;

	if(enableEDL){
		int supersamplingFactor = source.width / windowWidth;
		edl = getEdlShadingFactor(c_target.colorbuffer, depth, x, y, supersamplingFactor);
	}

	if(enableSSAO){
		ssao = ssaoShadeBuffer[pixelID] * 0.4f + 0.6f;
	}

	if(isinf(depth)) color = backgroundColor;

	float shade = edl * ssao;
	uint8_t* rgba = (uint8_t*)&color;
	rgba[0] = shade * float(rgba[0]);
	rgba[1] = shade * float(rgba[1]);
	rgba[2] = shade * float(rgba[2]);

	// surf2Dwrite(color, gl_desktop, x * 4, y);
	screenshot[pixelID] = color;
	
}


uint32_t sampleJpeg_nearest(
	uint32_t texID,
	vec2 uv, 
	Texture* textures, 
	uint32_t* decoded,
	HashMap& decodedMcuMap,
	uint32_t mipLevel
){

	uint32_t color = 0;
	auto tex = &textures[texID + mipLevel];

	int tx = (int(uv.x * tex->width) % tex->width);
	int ty = (int(uv.y * tex->height) % tex->height);
	uint32_t mcu_x = tx / 16;
	uint32_t mcu_y = ty / 16;
	uint32_t mcus_x = tex->width / 16;
	uint32_t mcu = mcu_x + mcus_x * mcu_y;

	// int mcu = tx / 16 + ty / 16 * tex->width / 16;
	// uint32_t key = ((mcu & 0xffff) << 16) | (texID & 0xffff);
	uint32_t key = pack_mcuidx_textureidx_miplevel(mcu, texID, mipLevel);


	uint32_t value;
	if(decodedMcuMap.get(key, &value)){
		uint32_t decodedMcuIndex = value & 0x00ffffff;
		bool isNewlyDecoded = (value >> 31) != 0;

		int tx = (int(uv.x * tex->width) % tex->width);
		int ty = (int(uv.y * tex->height) % tex->height);

		tx %= 16;
		ty %= 16;
		int offset = tx % 8 + (tx / 8) * 64 + (ty % 8) * 8 + (ty / 8) * 128;
		color = decoded[decodedMcuIndex * 256 + offset];

	}else{
		color = 0x00000000;
	}

	return color;
}

// retrieve the 4 texels around the given uv coordinate
void getTexels(
	uint32_t texID,
	vec2 uv, 
	Texture* tex, 
	uint32_t mipLevel,
	uint32_t* decoded,
	HashMap& decodedMcuMap,
	vec4* t00,
	vec4* t01,
	vec4* t10,
	vec4* t11
){
	float ftx = fmodf(uv.x, 1.0f) * float(tex->width);
	float fty = fmodf(uv.y, 1.0f) * float(tex->height);

	float ftlx = fmodf(ftx, 16.0f);
	float ftly = fmodf(fty, 16.0f);

	*t00 = {0.0f, 0.0f, 0.0f, 255.0f};
	*t01 = {0.0f, 0.0f, 0.0f, 255.0f};
	*t10 = {0.0f, 0.0f, 0.0f, 255.0f};
	*t11 = {0.0f, 0.0f, 0.0f, 0.0f};

	auto toVec4 = [](uint32_t color){
		return vec4{
			(color >>  0) & 0xff,
			(color >>  8) & 0xff,
			(color >> 16) & 0xff,
			(color >> 24) & 0xff,
		}; 
	};

	if(ftlx > 0.5f && ftlx < 15.5f && ftly > 0.5f && ftly < 15.5f){
		// Easy and fast case: All texels in same MCU

		int tx = ftx - 0.5f;
		int ty = fty - 0.5f;

		uint32_t mcu_x = tx / 16;
		uint32_t mcu_y = ty / 16;
		uint32_t mcus_x = tex->width / 16;
		uint32_t mcu = mcu_x + mcus_x * mcu_y;
		// uint32_t key = ((mcu & 0xffff) << 16) | (texID & 0xffff);
		uint32_t key = pack_mcuidx_textureidx_miplevel(mcu, tex->handle - mipLevel, mipLevel);

		uint32_t value;
		if(decodedMcuMap.get(key, &value)){
			uint32_t decodedMcuIndex = value & 0x00ffffff;
			
			tx %= 16;
			ty %= 16;
			int offset_00 = (tx + 0) % 8 + ((tx + 0) / 8) * 64 + ((ty + 0) % 8) * 8 + ((ty + 0) / 8) * 128;
			int offset_01 = (tx + 0) % 8 + ((tx + 0) / 8) * 64 + ((ty + 1) % 8) * 8 + ((ty + 1) / 8) * 128;
			int offset_10 = (tx + 1) % 8 + ((tx + 1) / 8) * 64 + ((ty + 0) % 8) * 8 + ((ty + 0) / 8) * 128;
			int offset_11 = (tx + 1) % 8 + ((tx + 1) / 8) * 64 + ((ty + 1) % 8) * 8 + ((ty + 1) / 8) * 128;

			*t00 = toVec4(decoded[decodedMcuIndex * 256 + offset_00]);
			*t01 = toVec4(decoded[decodedMcuIndex * 256 + offset_01]);
			*t10 = toVec4(decoded[decodedMcuIndex * 256 + offset_10]);
			*t11 = toVec4(decoded[decodedMcuIndex * 256 + offset_11]);

			// *t00 = {0.0f, 1.0f, 0.0f, 255.0f};
		}
	}else{

		// Trickier case: texels reside in adjacent MCUs, which may or may not be available.
		uint32_t v_00, v_01, v_10, v_11 = 0;

		// return;

		auto texelCoordToKey = [&](int tx, int ty){
			uint32_t mcu_x = tx / 16;
			uint32_t mcu_y = ty / 16;
			uint32_t mcus_x = tex->width / 16;
			uint32_t mcu = mcu_x + mcus_x * mcu_y;
			// uint32_t key = ((mcu & 0xffff) << 16) | (texID & 0xffff);
			uint32_t key = pack_mcuidx_textureidx_miplevel(mcu, tex->handle - mipLevel, mipLevel);
			return key;
		};

		bool v00Exists = decodedMcuMap.get(texelCoordToKey(ftx - 0.5f, fty - 0.5f), &v_00);
		bool v01Exists = decodedMcuMap.get(texelCoordToKey(ftx - 0.5f, fty + 0.5f), &v_01);
		bool v10Exists = decodedMcuMap.get(texelCoordToKey(ftx + 0.5f, fty - 0.5f), &v_10);
		bool v11Exists = decodedMcuMap.get(texelCoordToKey(ftx + 0.5f, fty + 0.5f), &v_11);

		auto toTexel = [&](uint32_t value, int tx, int ty){
			uint32_t decodedMcuIndex = value & 0x00ffffff;

			tx %= 16;
			ty %= 16;
			int offset = tx % 8 + (tx / 8) * 64 + (ty% 8) * 8 + (ty / 8) * 128;

			vec4 color = toVec4(decoded[decodedMcuIndex * 256 + offset]);

			return color;
		};

		// If a texel's MCU is not decoded, clamp to one of the decoded MCUs
		
		// texel 00
		if (v00Exists)        *t00 = toTexel(v_00, ftx - 0.5f, fty - 0.5f);
		else *t00 = vec4(0,0,0,0);
		
		// texel 01
		if(v01Exists)        *t01 = toTexel(v_01, ftx - 0.5f, fty + 0.5f);
		else *t01 = vec4(0, 0, 0, 0);
		// texel 10
		if(v10Exists)        *t10 = toTexel(v_10, ftx + 0.5f, fty - 0.5f);
		else *t10 = vec4(0, 0, 0, 0);
		// texel 11
		if(v11Exists)        *t11 = toTexel(v_11, ftx + 0.5f, fty + 0.5f);
		else *t11 = vec4(0, 0, 0, 0);
	}
}

uint32_t sampleJpeg_linear(
	uint32_t texID,
	vec2 uv,
	Texture* textures,
	uint32_t* decoded,
	HashMap& decodedMcuMap,
	uint32_t mipLevel
) {
	uint32_t color = 0xff000000;
	uint8_t* rgba = (uint8_t*)&color;
	
	auto tex = &textures[texID + mipLevel];

	float ftx = fmodf(uv.x - 0.5f / float(tex->width), 1.0f) * float(tex->width);
	float fty = fmodf(uv.y - 0.5f / float(tex->height), 1.0f) * float(tex->height);

	float wx = fmodf(ftx, 1.0f);
	float wy = fmodf(fty, 1.0f);

	// --- read bilinear samples from mip i0 ---
	vec4 t00, t01, t10, t11;
	getTexels(texID + mipLevel, uv, tex, mipLevel, decoded, decodedMcuMap, &t00, &t01, &t10, &t11);
	vec4 result =
		(1.0f - wx) * (1.0f - wy) * t00 +
		wx * (1.0f - wy) * t10 +
		(1.0f - wx) * wy * t01 +
		wx * wy * t11;

	rgba[0] = result.r;
	rgba[1] = result.g;
	rgba[2] = result.b;
	rgba[3] = 255;

	return color;
}

extern "C" __global__
void kernel_resolve_jpeg(
	CMesh* meshes,
	uint32_t numMeshes,
	uint64_t* triangleCountPrefixSum,
	int mouseX,
	int mouseY,
	DeviceState* state,
	RasterizationSettings rasterizationSettings,
	JpegPipeline jpp,
	Texture* textures
) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int x = grid.thread_index().x;
	int y = grid.thread_index().y;
	int pixelID = toFramebufferIndex(x, y, c_target.width);

	if(x >= c_target.width) return;
	if(y >= c_target.height) return;

	auto computeRayDir = [](float x, float y, float width, float height, mat4 view, mat4 proj){
		float u = 2.0f * x / width - 1.0f;
		float v = 2.0f * y / height - 1.0f;

		mat4 viewI = inverse(view);
		vec3 origin = viewI * vec4(0.0f, 0.0f, 0.0f, 1.0f);
		vec3 rayDir_view = normalize(vec3{
			1.0f / proj[0][0] * u,
			1.0f / proj[1][1] * v,
			-1.0f
		});
		vec3 rayDir_world = normalize(vec3(viewI * vec4(rayDir_view, 1.0f)) - origin);

		return rayDir_world;
	};

	mat4 viewI = inverse(c_target.view);
	vec3 origin = viewI * vec4(0.0f, 0.0f, 0.0f, 1.0f);
	vec3 rayDir    = computeRayDir(float(x) + 0.5f, float(y) + 0.5f, c_target.width, c_target.height, c_target.view, c_target.proj);
	vec3 rayDir_10 = computeRayDir(float(x) + 1.5f, float(y) + 0.5f, c_target.width, c_target.height, c_target.view, c_target.proj);
	vec3 rayDir_01 = computeRayDir(float(x) + 0.5f, float(y) + 1.5f, c_target.width, c_target.height, c_target.view, c_target.proj);
	vec3 rayDir_11 = computeRayDir(float(x) + 1.5f, float(y) + 1.5f, c_target.width, c_target.height, c_target.view, c_target.proj);

	uint64_t pixel = c_target.framebuffer[pixelID];
	uint64_t pixel_colorbuffer = c_target.colorbuffer[pixelID];
	
	float depth;
	uint32_t meshIndex;
	uint64_t totalTriangleIndex = 0;
	unpack_pixel(pixel, &depth, &totalTriangleIndex);

	float depth_colorbuffer = __uint_as_float(pixel_colorbuffer >> 32);

	if(depth_colorbuffer < depth) return;

	{// Find mesh corresponding to totalTriangleIndex via binary search
		int left = 0;
		int right = numMeshes - 1;
		int searchResult = 0;

		while(left <= right){
			int mid = left + (right - left) / 2;
			
			int64_t before = triangleCountPrefixSum[mid];
			int64_t after = (mid == (numMeshes - 1)) ? 0xffffffffff : triangleCountPrefixSum[mid + 1];

			if(totalTriangleIndex >= before && totalTriangleIndex < after){
				searchResult = mid;
				break;
			}else if(after <= totalTriangleIndex){
				left = mid + 1;
			}else{
				right = mid - 1;
			}

			if(mid == left) break;
		}

		meshIndex = searchResult;
	}

	uint32_t color = 0;
	uint8_t *rgb = (uint8_t *)&color;

	if(!isinf(depth)){
		CMesh mesh = meshes[meshIndex];
		uint32_t triangleIndex = totalTriangleIndex - mesh.cummulativeTriangleCount;

		if(x == mouseX && y == mouseY){
			state->hovered_meshId = mesh.id;
			state->hovered_triangleIndex = triangleIndex;
		}

		triangleIndex = min(triangleIndex, mesh.numTriangles - 1);

		if(!mesh.isLoaded){
			color = 0xffff00ff;
			goto shade_finished;
		}

		// resolve indices first
		uint32_t i0, i1, i2;
		if(!mesh.compressed){
			if(mesh.indices){
				i0 = mesh.indices[3 * triangleIndex + 0];
				i1 = mesh.indices[3 * triangleIndex + 1];
				i2 = mesh.indices[3 * triangleIndex + 2];
			}else{
				i0 = 3 * triangleIndex + 0;
				i1 = 3 * triangleIndex + 1;
				i2 = 3 * triangleIndex + 2;
			}
		}else{
			uint32_t indexRange = mesh.index_max - mesh.index_min;
			uint64_t bitsPerIndex = ceil(log2f(float(indexRange + 1)));
			i0 = BitEdit::readU32(mesh.indices, bitsPerIndex * (3 * triangleIndex + 0), bitsPerIndex) + mesh.index_min;
			i1 = BitEdit::readU32(mesh.indices, bitsPerIndex * (3 * triangleIndex + 1), bitsPerIndex) + mesh.index_min;
			i2 = BitEdit::readU32(mesh.indices, bitsPerIndex * (3 * triangleIndex + 2), bitsPerIndex) + mesh.index_min;
		}

		// Then load geometry data
		vec3 a_object = getVertex_resolved(mesh, i0);
		vec3 b_object = getVertex_resolved(mesh, i1);
		vec3 c_object = getVertex_resolved(mesh, i2);
		vec2 uv_a = getUV_resolved(mesh, i0);
		vec2 uv_b = getUV_resolved(mesh, i1);
		vec2 uv_c = getUV_resolved(mesh, i2);
		// ---------------------------------------------------


		// mat4 worldView = c_target.view * mesh.world;
		vec3 a_world = mesh.world * vec4(a_object, 1.0f);
		vec3 b_world = mesh.world * vec4(b_object, 1.0f);
		vec3 c_world = mesh.world * vec4(c_object, 1.0f);

		vec3 a_view = c_target.view * vec4(a_world, 1.0f);
		vec3 b_view = c_target.view * vec4(b_world, 1.0f);
		vec3 c_view = c_target.view * vec4(c_world, 1.0f);

		vec3 N = normalize(cross(b_view - a_view, c_view - a_view));
		
		// For mip map level:
		// - Find triangle intersection in current pixel for current uv
		// - Also find triangle intersection for pixels to the right and the top to 
		//   compute the change of uv coordinates.
		// - But actually do plane intersection because triangle may not extend to adjacent pixels
		// float t    = intersectTriangle(origin, rayDir, a, b, c, false);
		float t, t_10, t_01, t_11;
		{
			vec3 edge1 = b_world - a_world;
			vec3 edge2 = c_world - a_world;
			vec3 normal = normalize(cross(edge1, edge2));

			float d = dot(a_world, normal);

			t    = intersectPlane(origin, rayDir,    normal, -d);
			t_10 = intersectPlane(origin, rayDir_10, normal, -d);
			t_01 = intersectPlane(origin, rayDir_01, normal, -d);
			t_11 = intersectPlane(origin, rayDir_11, normal, -d);
		}

		// Store 2-component barycentric coordinates because the 3rd component is deducted from the other two.
		vec2 stv, stv_10, stv_01, stv_11;
		{

			vec3 v0 = b_world - a_world;
			vec3 v1 = c_world - a_world;

			float d00 = dot(v0, v0);
			float d01 = dot(v0, v1);
			float d11 = dot(v1, v1);
			float denom = d00 * d11 - d01 * d01;
			float denomI = 1.0f / denom;
			
			auto computeSTV = [&](float t, vec3 rayDir){
				vec3 p = origin + t * rayDir;
				vec3 v2 = p - a_world;

				float d20 = dot(v2, v0);
				float d21 = dot(v2, v1);

				vec2 stv;
				stv.x = (d11 * d20 - d01 * d21) * denomI;
				stv.y = (d00 * d21 - d01 * d20) * denomI;

				return stv;
			};
			
			stv    = computeSTV(t, rayDir);
			stv_10 = computeSTV(t_10, rayDir_10);
			stv_01 = computeSTV(t_01, rayDir_01);
			stv_11 = computeSTV(t_11, rayDir_11);
		}
		
		vec2 uv    = uv_a * (1.0f - stv.x    - stv.y   ) + uv_b * stv.x    + uv_c * stv.y;
		vec2 uv_10 = uv_a * (1.0f - stv_10.x - stv_10.y) + uv_b * stv_10.x + uv_c * stv_10.y;
		vec2 uv_01 = uv_a * (1.0f - stv_01.x - stv_01.y) + uv_b * stv_01.x + uv_c * stv_01.y;
		vec2 uv_11 = uv_a * (1.0f - stv_11.x - stv_11.y) + uv_b * stv_11.x + uv_c * stv_11.y;

		vec2 uvmax = {
			max(max(uv.x, uv_10.x), max(uv_01.x, uv_11.x)),
			max(max(uv.y, uv_10.y), max(uv_01.y, uv_11.y)),
		};
		vec2 uvmin = {
			min(min(uv.x, uv_10.x), min(uv_01.x, uv_11.x)),
			min(min(uv.y, uv_10.y), min(uv_01.y, uv_11.y)),
		};

		// Compute Mip Map level by delta of texcoords
		vec2 dx = (uvmax.x - uvmin.x) * vec2{mesh.texture.width, mesh.texture.height};
		vec2 dy = (uvmax.y - uvmin.y) * vec2{mesh.texture.width, mesh.texture.height};
		float mipLevel = 0.5f * log2(max(dot(dx, dx), dot(dy, dy)));
		mipLevel = int(mipLevel);
		mipLevel = min(mipLevel, 7.0f);

		// pick a mip map level
		int target_level = clamp(mipLevel, 0.0f, 7.0f);
		// target_level = 0;
		uint32_t* data = mesh.texture.data;
		uint32_t width = mesh.texture.width;
		uint32_t height = mesh.texture.height;
		for(int i = 1; i <= target_level; i++){
			data = data + width * height;
			width = (width + 2 - 1) / 2;
			height = (height + 2 - 1) / 2;
		}
		color = 0xffff00ff;
		
		// TODO: These blocks here are a major performance bottleneck. 
		// Removing them cuts registers from 88 to 56 and improves perf from 1.3ms to 0.88ms
		if(rasterizationSettings.displayAttribute == DisplayAttribute::NONE){
			color = 0xffffffff;
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::TRIANGLE_ID){
			color = triangleIndex * 12345678;
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::MESH_ID){
			color = (mesh.id + 1) * 12345678;
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::TEXTURE && mesh.uvs){
			if(!mesh.texture.huffmanTables){
				color = sampleColor_linear(data, width, height, uv);
			}else{
				// uint32_t mcu = uvToMCUIndex(width, height, uv.x, uv.y);
				uv.x = uv.x - floor(uv.x);
				uv.y = uv.y - floor(uv.y);
				int tx = (int(uv.x * width) % width);
				int ty = (int(uv.y * height) % height);
				uint32_t mcu_x = tx / 16;
				uint32_t mcu_y = ty / 16;
				uint32_t mcus_x = width / 16;
				uint32_t mcu = mcu_x + mcus_x * mcu_y;
				uint32_t key = pack_mcuidx_textureidx_miplevel(mcu, mesh.texture.handle, uint32_t(mipLevel));

				uint32_t texID = mesh.texture.handle;

				// color = sampleJpeg_nearest(texID, uv, textures, jpp.decoded, jpp.decodedMcuMap, mipLevel);
				color = sampleJpeg_linear(texID, uv, textures, jpp.decoded, jpp.decodedMcuMap, mipLevel);

				// if(x == mouseX && y == mouseY){
				// 	state->dbg_hovered_textureHandle = mesh.texture.handle;
				// 	state->dbg_hovered_mipLevel      = mipLevel;
				// 	state->dbg_hovered_tx            = tx;
				// 	state->dbg_hovered_ty            = ty;
				// 	state->dbg_hovered_mcu_x         = mcu_x;
				// 	state->dbg_hovered_mcu_y         = mcu_y;
				// 	state->dbg_hovered_mcu           = mcu;
				// 	state->dbg_hovered_decoded_color = color;
				// }
			}

		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::UV && mesh.uvs){
			rgb[0] = fmodf(uv.x - floor(uv.x), 1.0f) * 256.0f;
			rgb[1] = fmodf(uv.y - floor(uv.y), 1.0f) * 256.0f;
			rgb[2] = 0;
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::NORMAL && mesh.normals){
			
			vec3 n0 = mesh.normals[i0];
			vec3 n1 = mesh.normals[i1];
			vec3 n2 = mesh.normals[i2];

			n0 = mesh.world * vec4(n0, 0.0f);
			n1 = mesh.world * vec4(n1, 0.0f);
			n2 = mesh.world * vec4(n2, 0.0f);

			vec3 N = n0 * (1.0f - stv.x - stv.y) + n1 * stv.x + n2 * stv.y;
			N = normalize(N);

			rgb[0] = clamp(N.x * 255.0f, 0.0f, 255.0f);
			rgb[1] = clamp(N.y * 255.0f, 0.0f, 255.0f);
			rgb[2] = clamp(N.z * 255.0f, 0.0f, 255.0f);
			
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::NORMAL && !mesh.normals){
			vec3 edge1 = b_world - a_world;
			vec3 edge2 = c_world - a_world;
			vec3 N = normalize(cross(edge1, edge2));

			rgb[0] = clamp(N.x * 255.0f, 0.0f, 255.0f);
			rgb[1] = clamp(N.y * 255.0f, 0.0f, 255.0f);
			rgb[2] = clamp(N.z * 255.0f, 0.0f, 255.0f);
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::VERTEX_COLORS && mesh.colors){
			uint32_t C_a = mesh.colors[i0];
			uint32_t C_b = mesh.colors[i1];
			uint32_t C_c = mesh.colors[i2];
			color = C_a;

			auto toVec3 = [](uint32_t C){
				return vec3{
					(C >>  0) & 0xff,
					(C >>  8) & 0xff,
					(C >> 16) & 0xff,
				};
			};

			vec3 c_a = toVec3(C_a);
			vec3 c_b = toVec3(C_b);
			vec3 c_c = toVec3(C_c);

			vec3 c = c_a * (1.0f - stv.x - stv.y) + c_b * stv.x + c_c * stv.y;
			rgb[0] = clamp(c.x, 0.0f, 255.0f);
			rgb[1] = clamp(c.y, 0.0f, 255.0f);
			rgb[2] = clamp(c.z, 0.0f, 255.0f);
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::STAGE){
			// We don't actually know which stage a pixel was rasterized by, 
			// but we can assume based on the triangle's size and whether it intersects the near plane.
			float f = c_target.proj[1][1];
			float aspect = float(c_target.width) / float(c_target.height);

			vec3 a_ndc = viewToNDC(a_view, f, aspect);
			vec3 b_ndc = viewToNDC(b_view, f, aspect);
			vec3 c_ndc = viewToNDC(c_view, f, aspect);

			bool isNontrivial = (a_ndc.z <= 0.0f || b_ndc.z <= 0.0f || c_ndc.z <= 0.0f);

			vec2 a_screen = ndcToScreen(a_ndc, c_target.width, c_target.height);
			vec2 b_screen = ndcToScreen(b_ndc, c_target.width, c_target.height);
			vec2 c_screen = ndcToScreen(c_ndc, c_target.width, c_target.height);

			// screen-space bounding box of triangle
			float min_x = min(a_screen.x, min(b_screen.x, c_screen.x));
			float max_x = max(a_screen.x, max(b_screen.x, c_screen.x));
			float min_y = min(a_screen.y, min(b_screen.y, c_screen.y));
			float max_y = max(a_screen.y, max(b_screen.y, c_screen.y));

			// clip to screen
			min_x = max(min_x, 0.0f);
			max_x = min(max_x, float(c_target.width - 1));
			min_y = max(min_y, 0.0f);
			max_y = min(max_y, float(c_target.height - 1));

			int size_x = ceil(max_x) - floor(min_x);
			int size_y = ceil(max_y) - floor(min_y);
			int numFragments = size_x * size_y;

			int stage = 0;

			if(numFragments <= THRESHOLD_SMALL)      stage = 0;
			else if(numFragments <= THRESHOLD_LARGE) stage = 1;
			else                                     stage = 2;
			
			if(isNontrivial) stage = 2;

			color = 0xffffaaff;
			if(stage == 0) color = 0xffbd8832;
			if(stage == 1) color = 0xffa4ddab;
			if(stage == 2) color = 0xff61aefd;
		}else{
			color = 0xff0000ff;
		}
		color = color | 0xff000000;

		if(rasterizationSettings.enableDiffuseLighting && mesh.normals){

			vec3 n0 = mesh.normals[i0];
			vec3 n1 = mesh.normals[i1];
			vec3 n2 = mesh.normals[i2];

			n0 = mesh.world * vec4(n0, 0.0f);
			n1 = mesh.world * vec4(n1, 0.0f);
			n2 = mesh.world * vec4(n2, 0.0f);

			vec3 N = n0 * (1.0f - stv.x - stv.y) + n1 * stv.x + n2 * stv.y;
			vec3 L = normalize(vec3{1.0f, 1.0f, 1.0f});
			float d = max(dot(N, L), 0.0f);

			vec3 ambient = {0.5, 0.5f, 0.5f};
			vec3 diffuse = {0.7f, 0.7f, 0.7f};

			rgb[0] = clamp((d * diffuse.x + ambient.x) * float(rgb[0]), 0.0f, 255.0f);
			rgb[1] = clamp((d * diffuse.y + ambient.y) * float(rgb[1]), 0.0f, 255.0f);
			rgb[2] = clamp((d * diffuse.z + ambient.z) * float(rgb[2]), 0.0f, 255.0f);
		}

		// Highlight hovered mesh: draw borders
		if(rasterizationSettings.enableObjectPicking && mesh.id == state->hovered_meshId){

			bool isInside = true;

			for(int dx : {-1, 0, 1})
			for(int dy : {-1, 0, 1})
			{

				int nx = clamp(x + dx, 0, c_target.width - 1);
				int ny = clamp(y + dy, 0, c_target.height - 1);
				int neighbor_pixelID = toFramebufferIndex(nx, ny, c_target.width);
				uint64_t neighbor_pixel = c_target.framebuffer[neighbor_pixelID];

				float neighbor_depth;
				uint32_t neighbor_meshIndex;
				uint64_t neighbor_totalTriangleIndex = 0;
				unpack_pixel(neighbor_pixel, &neighbor_depth, &neighbor_totalTriangleIndex);

				// Check if neighbor's total triangle index is within this pixel's mesh
				if(neighbor_totalTriangleIndex < mesh.cummulativeTriangleCount) isInside = false;
				if(neighbor_totalTriangleIndex >= mesh.cummulativeTriangleCount + mesh.numTriangles) isInside = false;
			}

			if(!isInside){
				color = 0xff0000ff;
			}
		}
		

	}else{
		color = 0;

		if(x == mouseX && y == mouseY){
			state->hovered_meshId = 0xffffffff;
			state->hovered_triangleIndex = 0xffffffff;
		}
	}

	shade_finished:

	// WIREFRAME
	if(rasterizationSettings.showWireframe){
		uint32_t numPixels = c_target.width * c_target.height;
		int pid_00 = clamp(toFramebufferIndex(x + 0, y + 0, c_target.width), 0u, numPixels - 1);
		int pid_01 = clamp(toFramebufferIndex(x + 0, y + 1, c_target.width), 0u, numPixels - 1);
		int pid_10 = clamp(toFramebufferIndex(x + 1, y + 0, c_target.width), 0u, numPixels - 1);

		uint32_t p_00 = c_target.framebuffer[pid_00] & 0xffffffff;
		uint32_t p_01 = c_target.framebuffer[pid_01] & 0xffffffff;
		uint32_t p_10 = c_target.framebuffer[pid_10] & 0xffffffff;

		if(p_00 != p_10) color = 0xffff00ff;
		if(p_00 != p_01) color = 0xffff00ff;

		// for(int dx : {-1, 0, 1})
		// for(int dy : {-1, 0, 1})
		for(int dx : {0, 1})
		for(int dy : {0, 1})
		{
			int pid_neighbor = clamp(toFramebufferIndex(x + dx, y + dy, c_target.width), 0u, numPixels - 1);
			uint32_t p_neighbor = c_target.framebuffer[pid_neighbor] & 0xffffffff;

			if(p_00 != p_neighbor) color = 0xffff00ff;
		}
	}

	// 64x64 tile grid
	// if(x % 64 == 0 || y % 64 == 0){
	// 	color = 0xffff00ff;
	// }

	if(depth != Infinity){
		uint64_t udepth = __float_as_uint(depth);
		uint64_t pixel = (udepth << 32) | color;
		c_target.colorbuffer[pixelID] = pixel;
	}else{
		// c_target.colorbuffer[pixelID] = uint64_t(__float_as_uint(INFINITY)) << 32 | 0xff776655;
		c_target.colorbuffer[pixelID] = uint64_t(__float_as_uint(INFINITY)) << 32 | 0;
	}
}