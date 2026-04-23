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

#include "./libs/glm/glm/glm.hpp"
#include "./libs/glm/glm/gtc/matrix_transform.hpp"
#include "./libs/glm/glm/gtc/matrix_access.hpp"
#include "./libs/glm/glm/gtx/transform.hpp"
#include "./libs/glm/glm/gtc/quaternion.hpp"

#include "./utils.cuh"
#include "./HostDeviceInterface.h"
#include "./rasterization_helpers.cuh"

using glm::ivec2;
using glm::i8vec4;
using glm::vec4;

__constant__ RenderTarget c_target;

extern "C" __global__
void kernel_drawHeightmap(
	uint32_t numCells
) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	float X = grid.thread_index().x;
	float Y = grid.thread_index().y;
	if (X >= numCells || Y >= numCells) return;

	float s = float(X) / float(numCells);
	float t = float(Y) / float(numCells);
	float s_n = float(X + 1) / float(numCells);
	float t_n = float(Y + 1) / float(numCells);

	auto sample = [](float s, float t) -> vec3{

		// float u = 2.0f * 3.1415f * s;
		// float v = 3.1415f * t;

		// float r = 1.0f + 0.02f * sin(50.0f * u) * cos(50.0f * v);
		// r = 1.0f;
		
		// return r * vec3{
		// 	sin(u) * cos(v),
		// 	cos(u),
		// 	sin(u) * sin(v)
		// };

		return vec3{
			10.0f * s,
			10.0f * t,
			sin(12.0f * s) * cos(12.0f * t) + sqrt(t * s)
		};
		// return vec3{
		// 	10.0f * s,
		// 	10.0f * t,
		// 	0.0f
		// };

		// return getSeashellPos(100.0f * s, 100.0f * t, 87.0, 29.0, 4.309, 4.7, -39.944, 58.839, 9.0, 20.0, 1.0, 1.5, 2.0);
		// return getSeashellPos(100.0f * s, 100.0f * t, 86.5, 10.0, 6.071, 5.153, -39.944, 1.0, 9.0, 11.0, 1.0, 1.5, 1.5);
		// return getSeashellPos(100.0f * s, 100.0f * t, 86.5f, 10.0f, 6.071f, 5.153f, -39.944f, 100.0f, 50.0f, 70.0f, 0.5f, 4.0f, 5.0f);
		

		// return get_sh_glyph(PI * s, 2.0f * PI * t);
	};

	float f = c_target.proj[1][1];
	float aspect = float(c_target.width) / float(c_target.height);

	vec3 p00 = sample(s, t);
	vec3 p10 = sample(s_n, t);
	vec3 p01 = sample(s, t_n);
	vec3 p11 = sample(s_n, t_n);

	vec3 ndc00 = worldToNDC(p00, c_target.view, f, aspect);
	vec3 ndc10 = worldToNDC(p10, c_target.view, f, aspect);
	vec3 ndc01 = worldToNDC(p01, c_target.view, f, aspect);
	vec3 ndc11 = worldToNDC(p11, c_target.view, f, aspect);


	auto rasterize = [&](
		const vec3& a, const vec3& b, const vec3& c,
		const vec3& a_ndc, const vec3& b_ndc, const vec3& c_ndc
	){
		
		// Triangle-wise view-frustum culling
		if(a_ndc.x > +1.0f && b_ndc.x > +1.0f && c_ndc.x > +1.0f) return;
		if(a_ndc.x < -1.0f && b_ndc.x < -1.0f && c_ndc.x < -1.0f) return;
		if(a_ndc.y > +1.0f && b_ndc.y > +1.0f && c_ndc.y > +1.0f) return;
		if(a_ndc.y < -1.0f && b_ndc.y < -1.0f && c_ndc.y < -1.0f) return;
		if(a_ndc.z <= 0.0f && b_ndc.z <= 0.0f && c_ndc.z <= 0.0f) return;
		
		vec2 a_screen = ndcToScreen(a_ndc, c_target.width, c_target.height);
		vec2 b_screen = ndcToScreen(b_ndc, c_target.width, c_target.height);
		vec2 c_screen = ndcToScreen(c_ndc, c_target.width, c_target.height);

		float min_x = min(a_screen.x, min(b_screen.x, c_screen.x));
		float max_x = max(a_screen.x, max(b_screen.x, c_screen.x));
		float min_y = min(a_screen.y, min(b_screen.y, c_screen.y));
		float max_y = max(a_screen.y, max(b_screen.y, c_screen.y));

		bool isNontrivial = (a_ndc.z <= 0.0f || b_ndc.z <= 0.0f || c_ndc.z <= 0.0f) ||
			(fabsf(a_ndc.x) > 1.0f || fabsf(b_ndc.x) > 1.0f || fabsf(c_ndc.x) > 1.0f);

		// Cull triangles whose bounding box does not hit a pixel sample position
		float sample_x = floor(min_x);
		float sample_y = floor(min_y);
		if(min_x > sample_x && max_x < sample_x + 1.0f) return;
		if(min_y > sample_y && max_y < sample_y + 1.0f) return;
		
		int size_x = ceil(max_x) - floor(min_x);
		int size_y = ceil(max_y) - floor(min_y);

		int numFragments = size_x * size_y;
		if(numFragments > THRESHOLD_SMALL) isNontrivial = true;
		if(numFragments <= 0) return;

		// { // BACKFACE CULLING

		// 	// vec3 c0 = world[0];
		// 	// vec3 c1 = world[1];
		// 	// vec3 c2 = world[2];
		// 	// float s = dot(cross(c0, c1), c2);
		// 	// bool flipTriangles = s < 0.0f;

		// 	vec3 N = cross(b - a, c - a);
		// 	float res = dot(a - c_target.cameraPos, N);
		// 	// if(flipTriangles) res = -res; // TODO: instance-wise

		// 	if(res >= 0.0f) return;
		// }

		if(isNontrivial){
			// ...
		}else{

			vec2 v_ab = b_screen - a_screen;
			vec2 v_ac = c_screen - a_screen;
			
			float factor = cross(v_ab, v_ac);
			float inv_factor = 1.0f / factor; // Precompute division outside!
			float start_x = floor(min_x);
			float start_y = floor(min_y);

			// Process small triangles per-thread
			for(int fragY = 0; fragY < size_y; ++fragY) 
			for(int fragX = 0; fragX < size_x; ++fragX) 
			{

				vec2 pFrag = {
					start_x + (float)fragX,
					start_y + (float)fragY
				};
				vec2 sample = {pFrag.x - a_screen.x, pFrag.y - a_screen.y};

				float s = cross(sample, v_ac) * inv_factor;
				float t = cross(v_ab, sample) * inv_factor;
				float v = 1.0f - (s + t);

				bool isInsideTriangle = (s > 0.0f) && (t > 0.0f) && (v > 0.0f);
				
				if (isInsideTriangle)
				{
					int pixelID = int(pFrag.x) + c_target.width * int(pFrag.y);

					float depth = v * a_ndc.z + s * b_ndc.z + t * c_ndc.z;
					uint64_t udepth = __float_as_uint(depth);
					uint32_t color;
					color = 
						((grid.thread_index().x / 10) << 0) |
						((grid.thread_index().y / 10) << 8);
					uint8_t* rgba = (uint8_t*)&color;
					// uint32_t color = 0xffffffff;
					// uint32_t color = 
					// 	(uint32_t(a_st.x * 256.0f) << 0) |
					// 	(uint32_t(a_st.y * 256.0f) << 8);
					// uint32_t color = 0xff00ff00;
					// rgba[0] = a_st.x * 256.0f;
					// rgba[1] = a_st.y * 256.0f;
					// rgba[0] = clamp(a_st.x * 256.0f, 0.0f, 255.0f);
					// rgba[1] = clamp(a_st.y * 256.0f, 0.0f, 255.0f);
					// rgba[2] = 0.0f;

					rgba[0] = 50.0f * a.x;
					rgba[1] = 50.0f * a.y;
					rgba[2] = 0.2f;
					uint64_t pixel = (udepth << 32) | color;

					if(pixelID < c_target.width * c_target.height){
						atomicMin(&c_target.colorbuffer[pixelID], pixel);
					}
				}
			
			}
		}
	};

	rasterize(p00, p10, p11, ndc00, ndc10, ndc11);
	rasterize(p00, p11, p01, ndc00, ndc11, ndc01);

}


