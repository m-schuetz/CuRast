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


extern "C" __global__
void kernel_enlarge(
	cudaSurfaceObject_t gl_desktop,
	uint64_t* fbo_enlarge,
	int width, 
	int height,
	int mouseX,
	int mouseY,
	DeviceState* state,
	bool enableEDL
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
void kernel_resolve_colorbuffer_to_opengl_2D(
	cudaSurfaceObject_t gl_desktop,
	int width, 
	int height,
	int mouseX,
	int mouseY,
	DeviceState* state,
	bool enableEDL,
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

			if(enableEDL){
				edl = getEdlShadingFactor(c_target.colorbuffer, depth, x, y, 1);
			}

			if(isinf(depth)) sampleColor = backgroundColor;

			float shade = edl;

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
			numSamples++;
		}

		if(enableEDL){
			edl = edl / float(numSamples);
		}else{
			edl = 1.0f;
		}

		float shade = edl;
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
	bool enableEDL,
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

	if(enableEDL){
		int supersamplingFactor = source.width / windowWidth;
		edl = getEdlShadingFactor(c_target.colorbuffer, depth, x, y, supersamplingFactor);
	}

	if(isinf(depth)) color = backgroundColor;

	float shade = edl;
	uint8_t* rgba = (uint8_t*)&color;
	rgba[0] = shade * float(rgba[0]);
	rgba[1] = shade * float(rgba[1]);
	rgba[2] = shade * float(rgba[2]);

	// surf2Dwrite(color, gl_desktop, x * 4, y);
	screenshot[pixelID] = color;
	
}
