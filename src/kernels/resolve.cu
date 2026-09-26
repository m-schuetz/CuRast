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

using glm::ivec2;
using glm::i8vec4;
using glm::vec4;

__constant__ RenderTarget c_target;

uint32_t toFramebufferIndex(int x, int y, int width){
	return x + width * y;
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
	bool showInset,
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

		auto sample = [&](int x, int y){
			if(x < 0) return uint32_t(0);
			if(y < 0) return uint32_t(0);
			if(x >= source.width) return uint32_t(0);
			if(y >= source.height) return uint32_t(0);

			int pixelID = toFramebufferIndex(x, y, source.width);

			uint64_t pixel = c_target.colorbuffer[pixelID];
			float depth = __uint_as_float(pixel >> 32);
			uint32_t sampleColor = pixel & 0xffffffff;

			float edl = 1.0f;
			float ssao = 1.0f;

			if(enableEDL){
				edl = getEdlShadingFactor(c_target.colorbuffer, depth, x, y, 1);
			}

			if(enableSSAO){
				ssao = ssaoShadeBuffer[pixelID] * 0.4f + 0.6f;
			}

			if(isinf(depth)) sampleColor = backgroundColor;

			float shade = edl * ssao;

			uint8_t* rgba = (uint8_t*)&sampleColor;
			rgba[0] = shade * float(rgba[0]);
			rgba[1] = shade * float(rgba[1]);
			rgba[2] = shade * float(rgba[2]);
			rgba[3] = 255;

			return sampleColor;
		};

		uint32_t color = sample(x, y);
		surf2Dwrite(color, gl_desktop, x * 4, y);

		if(showInset){
			struct Rect{
				float x;
				float y;
				float width;
				float height;
			};
			float insetSize = 32;
			Rect insetSource = {
				mouseX - insetSize / 2,
				mouseY - insetSize / 2,
				insetSize,
				insetSize,
			};
			Rect insetTarget = {
				0, 0,
				insetSize * 16, insetSize * 16
			};

			float u = (float(x) - insetTarget.x) / insetTarget.width;
			float v = (float(y) - insetTarget.y) / insetTarget.height;

			if((u >= 0.0f && u <= 1.0f) && (v >= 0.0f && v <= 1.0f))
			{
				int source_x = insetSource.x + u * insetSize;
				int source_y = insetSource.y + v * insetSize;

				color = sample(source_x, source_y);

				if(u == 1.0f || v == 1.0f){
					color = 0xffff00ff;
				}
			}else if(
				x == int(insetSource.x)
				|| x == int(insetSource.x + insetSize)
				|| y == int(insetSource.y)
				|| y == int(insetSource.y + insetSize)
			){
				color = 0xff0000ff;
			}

			surf2Dwrite(color, gl_desktop, x * 4, y);
		}


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
