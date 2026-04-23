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
#include "../BitEdit.h"
#include "./rasterization_helpers.cuh"

using glm::ivec2;
using glm::i8vec4;
using glm::vec4;

// Some compile-time template specializations here because for perf reasons, 
// we need each variation of getVertex to be a separately compiled function.
// Branching at runtime may increase rendering duration by a couple of percent.
template<IndexFetch INDEXING, Compression COMPRESSION>
vec4 getVertex(const CMesh& mesh, uint32_t vertexIndex){

	if constexpr(INDEXING == IndexFetch::INDEXBUFFER && COMPRESSION == Compression::UNCOMPRESSED){

		uint32_t resolvedIndex;
		if(mesh.indices){
			resolvedIndex = mesh.indices[vertexIndex];
		}else{
			resolvedIndex = vertexIndex;
		}

		vec4 pos = vec4(mesh.positions[resolvedIndex], 1.0f);
		return pos;
	}else if constexpr(INDEXING == IndexFetch::INDEXBUFFER && COMPRESSION == Compression::IX_PU16){
		uint32_t resolvedIndex = BitEdit::readU32(mesh.indices, mesh.bitsPerIndex * vertexIndex, mesh.bitsPerIndex) + mesh.index_min;

		uint16_t* positions = (uint16_t*)mesh.positions;
		uint16_t X = positions[3 * resolvedIndex + 0];
		uint16_t Y = positions[3 * resolvedIndex + 1];
		uint16_t Z = positions[3 * resolvedIndex + 2];

		vec4 pos;
		pos.x = float(X) * mesh.compressionFactor.x + mesh.aabb.min.x;
		pos.y = float(Y) * mesh.compressionFactor.y + mesh.aabb.min.y;
		pos.z = float(Z) * mesh.compressionFactor.z + mesh.aabb.min.z;
		pos.w = 1.0f;

		return pos;
	}else if constexpr(INDEXING == IndexFetch::DIRECT && COMPRESSION == Compression::UNCOMPRESSED){
		vec4 pos = vec4(mesh.positions[vertexIndex], 1.0f);
		return pos;
	}else if constexpr(INDEXING == IndexFetch::DIRECT && COMPRESSION == Compression::IX_PU16){
		uint16_t* positions = (uint16_t*)mesh.positions;
		uint16_t X = positions[3 * vertexIndex + 0];
		uint16_t Y = positions[3 * vertexIndex + 1];
		uint16_t Z = positions[3 * vertexIndex + 2];

		vec4 pos;
		pos.x = float(X) * mesh.compressionFactor.x + mesh.aabb.min.x;
		pos.y = float(Y) * mesh.compressionFactor.y + mesh.aabb.min.y;
		pos.z = float(Z) * mesh.compressionFactor.z + mesh.aabb.min.z;
		pos.w = 1.0f;

		return pos;
	}
};

template<IndexFetch INDEXING, Compression COMPRESSION>
void rasterize(
	const RasterArgs args,
	const CMesh& sh_mesh,
	int meshIndex,
	const mat4& worldView,
	int triangleIndex,
	int instanceIndex,
	vec4 a_object,
	vec4 b_object,
	vec4 c_object
){

	if(triangleIndex >= sh_mesh.numTriangles) return;

	// if(sh_mesh.numTriangles > 7) return;
	// if(triangleIndex != 3) return;

	float f = args.target.proj[1][1];
	float aspect = float(args.target.width) / float(args.target.height);

	vec3 a_view = worldView * a_object;
	vec3 b_view = worldView * b_object;
	vec3 c_view = worldView * c_object;
	vec3 a_ndc = viewToNDC(a_view, f, aspect);
	vec3 b_ndc = viewToNDC(b_view, f, aspect);
	vec3 c_ndc = viewToNDC(c_view, f, aspect);

	bool isNontrivial = (a_ndc.z <= 0.0f || b_ndc.z <= 0.0f || c_ndc.z <= 0.0f);

	// Frustum-culling triangles.
	if(!isNontrivial){
		if(a_ndc.x > +1.0f && b_ndc.x > +1.0f && c_ndc.x > +1.0f) return;
		if(a_ndc.x < -1.0f && b_ndc.x < -1.0f && c_ndc.x < -1.0f) return;
		if(a_ndc.y > +1.0f && b_ndc.y > +1.0f && c_ndc.y > +1.0f) return;
		if(a_ndc.y < -1.0f && b_ndc.y < -1.0f && c_ndc.y < -1.0f) return;
	}
	if(a_ndc.z <= 0.0f && b_ndc.z <= 0.0f && c_ndc.z <= 0.0f) return;

	vec2 a_screen = ndcToScreen(a_ndc, args.target.width, args.target.height);
	vec2 b_screen = ndcToScreen(b_ndc, args.target.width, args.target.height);
	vec2 c_screen = ndcToScreen(c_ndc, args.target.width, args.target.height);

	// NOTE: To make sure the pixel sample is in the center, 
	//       we can either add a SAMPLE_OFFSET to the computations or 
	//       offset the screen-space coordinates.
	//       For unclear reasons, specifying a SAMPLE_OFFSET of 0.5f can be up to 40% slower
	//       compared to offsetting the screen-space coordinates by 0.5f.
	//       E.g. in Zorah, it can make the difference between 74ms and 102ms per frame.
	//-------------------------------------------------
	a_screen -= 0.5f;
	b_screen -= 0.5f;
	c_screen -= 0.5f;
	constexpr float SAMPLE_OFFSET = 0.0f;
	//-------------------------------------------------
	// constexpr float SAMPLE_OFFSET = 0.5f;
	//-------------------------------------------------

	// screen-space bounding box of triangle
	float min_x = min(a_screen.x, min(b_screen.x, c_screen.x));
	float max_x = max(a_screen.x, max(b_screen.x, c_screen.x));
	float min_y = min(a_screen.y, min(b_screen.y, c_screen.y));
	float max_y = max(a_screen.y, max(b_screen.y, c_screen.y));

	// clip to screen
	min_x = max(min_x, 0.0f);
	max_x = min(max_x, float(args.target.width - 1));
	min_y = max(min_y, 0.0f);
	max_y = min(max_y, float(args.target.height - 1));

	// Cull tiny triangles whose bounding box does not intersect a pixel sample position.
	if(!isNontrivial){
		float sample_x = floorf(min_x);
		float sample_y = floorf(min_y);
		if(min_x > sample_x + SAMPLE_OFFSET && max_x < sample_x + 1.0f + SAMPLE_OFFSET) return;
		if(min_y > sample_y + SAMPLE_OFFSET && max_y < sample_y + 1.0f + SAMPLE_OFFSET) return;
	}
	
	// Samples are at lower-left corner of a pixel (computational-wise, in reality it's the center due to 0.5f transform).
	// Therefore, we start from ceil(min) 
	int size_x = ceil(max_x) - ceil(min_x);
	int size_y = ceil(max_y) - ceil(min_y);

	int numFragments = size_x * size_y;
	if(numFragments > THRESHOLD_SMALL) isNontrivial = true;
	
	{ // BACKFACE CULLING IN VIEW SPACE
		
		// Compute view-space edge vectors
		vec3 ab = b_view - a_view;
		vec3 ac = c_view - a_view;

		// Compute the view-space face normal
		vec3 N = cross(ab, ac);
		float res = dot(a_view, N);

		// Handle instances with negative scale (mirrored geometry)
		vec3 c0 = worldView[0];
		vec3 c1 = worldView[1];
		vec3 c2 = worldView[2];
		float det = dot(cross(c0, c1), c2);
		bool flipTriangles = det < 0.0f;

		if (flipTriangles) res = -res;

		if (res >= 0.0f) return;
	}

	if(isNontrivial){
		// queue nontrivial triangles (large or intersecting near) for next stage
		uint64_t index = atomicAdd(args.nontrivialTrianglesCounter, 1);
		uint64_t packed = 
			(uint64_t(sh_mesh.instances.offset + instanceIndex) << 32llu) |
			(uint64_t(triangleIndex));

		args.nontrivialTrianglesList[index] = packed;
	}else{

		// two edges spanning the triangle (at origin)
		vec2 v_ab = b_screen - a_screen;
		vec2 v_ac = c_screen - a_screen;

		// 2D-cross product of triangle edges: area of parallelogram they span.
		// float factor = 1.0f / cross(v_ab, v_ac);
		float factor = __fdividef(1.0f, cross(v_ab, v_ac));
		
		// Precompute inverse depth
		float inv_z_a = __fdividef(1.0f, a_ndc.z);
		float inv_z_b = __fdividef(1.0f, b_ndc.z);
		float inv_z_c = __fdividef(1.0f, c_ndc.z);

		// Precompute barycentric steps: How do s and t change as we move to the next pixel along the x or y axis
		float ds_dx =  v_ac.y * factor;
		float ds_dy = -v_ac.x * factor;
		float dt_dx = -v_ab.y * factor;
		float dt_dy =  v_ab.x * factor;

		// Initialize starting barycentric coordinates
		float start_x = ceil(min_x);
		float start_y = ceil(min_y);
		float sample0_x = start_x - a_screen.x + SAMPLE_OFFSET;
		float sample0_y = start_y - a_screen.y + SAMPLE_OFFSET;

		float s_row_start = (sample0_x * v_ac.y - sample0_y * v_ac.x) * factor;
		float t_row_start = (v_ab.x * sample0_y - v_ab.y * sample0_x) * factor;

		// Initialize starting pixel index
		int row_pixel_id = toFramebufferIndex((int)start_x, (int)start_y, args.target.width);
		uint32_t numPixels = args.target.width * args.target.height;

		// Process small triangles per-thread
		for(int fragY = 0; fragY < size_y; ++fragY) 
		{
			// Reset X-axis trackers at the start of each row
			float s = s_row_start;
			float t = t_row_start;
			int pixelID = row_pixel_id;

			for(int fragX = 0; fragX < size_x; fragX++){
				float v = 1.0f - (s + t);
				
				if (s > 0.0f && t > 0.0f && v > 0.0f){
					if (pixelID < numPixels){
						// Perspective-correct interpolation using precomputed inverses
						float inv_depth = v * inv_z_a + s * inv_z_b + t * inv_z_c;
						
						// Fast hardware approximation for final division
						float depth = __fdividef(1.0f, inv_depth); 
						// float depth = 1.0f / inv_depth; 

						uint64_t pixel = pack_pixel(depth, 
							sh_mesh.cummulativeTriangleCount + triangleIndex + instanceIndex * sh_mesh.numTriangles
						);
						atomicMin(&args.target.framebuffer[pixelID], pixel);
					}
				}

				// Step X-axis (Additions instead of multiplications)
				s += ds_dx;
				t += dt_dx;
				pixelID++;
			}

			// Step Y-axis for the next row
			s_row_start += ds_dy;
			t_row_start += dt_dy;
			row_pixel_id += args.target.width;
		}
	}
}


template<IndexFetch INDEXING, Compression COMPRESSION, Instancing INSTANCING>
void stage1_drawSmallTriangles(RasterArgs& args){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	// Initialize gridwide state
	if(grid.thread_rank() == 0){
		*args.numProcessedBatches = 0;
		*args.numProcessedBatches_nontrivial = 0;
		*args.hugeTrianglesCounter = 0;
		*args.nontrivialTrianglesCounter = 0;
		*args.numProcessedHugeTriangles = 0;
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
	}

	grid.sync();

	// LOOP THROUGH TRIANGLES
	while (true){

		// Claim Work: Check which batch of triangles this block should render next.
		block.sync();
		if (block.thread_rank() == 0){

			uint32_t next = atomicAdd(args.numProcessedBatches, 1);
			uint32_t diff = next - sh_blockBatchIndex;
			
			sh_blockBatchIndex = next;
			sh_blockLocalBatchIndex += diff;
			
			// Next batch is outside of current mesh.
			// Advance through meshes to the one containing the next batch
			int numBatchesInMesh = (sh_mesh.numTriangles + TRIANGLES_PER_SWEEP - 1) / TRIANGLES_PER_SWEEP;
			while (sh_blockLocalBatchIndex >= numBatchesInMesh){
				sh_meshIndex++;

				if constexpr (INSTANCING == Instancing::NO){
					// Treats list of instances as list of meshes
					if (sh_meshIndex >= args.numInstances) break;
					sh_blockLocalBatchIndex -= numBatchesInMesh;
					sh_mesh = args.instances[sh_meshIndex];
					numBatchesInMesh = (sh_mesh.numTriangles + TRIANGLES_PER_SWEEP - 1) / TRIANGLES_PER_SWEEP;
				}else if constexpr (INSTANCING == Instancing::YES){
					if (sh_meshIndex >= args.numMeshes) break;
					sh_blockLocalBatchIndex -= numBatchesInMesh;
					sh_mesh = args.meshes[sh_meshIndex];
					numBatchesInMesh = (sh_mesh.numTriangles + TRIANGLES_PER_SWEEP - 1) / TRIANGLES_PER_SWEEP;
				}
			}
		}
		block.sync();


		if constexpr (INSTANCING == Instancing::NO){
			// Treats list of instances as list of meshes
			if (sh_meshIndex >= args.numInstances) return;
		}else if constexpr (INSTANCING == Instancing::YES){
			if (sh_meshIndex >= args.numMeshes) return;
		}

		uint32_t firstTriangleInBatch = sh_blockLocalBatchIndex * TRIANGLES_PER_SWEEP;
		int triangleIndex = firstTriangleInBatch + block.thread_rank();

		if(triangleIndex >= sh_mesh.numTriangles) continue;

		vec4 a_object = getVertex<INDEXING, COMPRESSION>(sh_mesh, 3 * triangleIndex + 0);
		vec4 b_object = getVertex<INDEXING, COMPRESSION>(sh_mesh, 3 * triangleIndex + 1);
		vec4 c_object = getVertex<INDEXING, COMPRESSION>(sh_mesh, 3 * triangleIndex + 2);

		// if(triangleIndex == 200'000'000){

		// 	// int resolvedIndex = sh_mesh.indices[3 * triangleIndex + 0];
		// 	// vec3 pos = sh_mesh.positions[resolvedIndex];
		// 	// printf("%d %d\n", triangleIndex, resolvedIndex);
		// 	// printf("%.1f %.1f %.1f \n", pos.x, pos.y, pos.z);
		// 	// printf("%.1f %.1f %.1f \n", a_object.x, a_object.y, a_object.z);

		// 	int resolvedIndex = sh_mesh.indices[300'000'000];
		// 	printf("%d\n", resolvedIndex);
		// }
		
		if constexpr (INSTANCING == Instancing::NO){
			mat4 worldView = args.transforms[sh_mesh.instances.offset];
			rasterize<INDEXING, COMPRESSION>(
				args, sh_mesh, sh_meshIndex, worldView, triangleIndex, 0, 
				a_object, b_object, c_object
			);
		}else if constexpr (INSTANCING == Instancing::YES){
			for(int instanceIndex = 0; instanceIndex < sh_mesh.instances.count; instanceIndex++){
				mat4 worldView = args.transforms[sh_mesh.instances.offset + instanceIndex];
				rasterize<INDEXING, COMPRESSION>(
					args, sh_mesh, sh_meshIndex, worldView, triangleIndex, instanceIndex, 
					a_object, b_object, c_object);
			}
		}
		
	}

}



template<IndexFetch INDEXING, Compression COMPRESSION>
void stage2_drawMediumTriangles(RasterArgs& args) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	auto warp = cg::tiled_partition<32>(block);

	grid.sync();

	while (true) {
		uint32_t nontrivialTriangleIndex;

		// Acquire triangle to process
		if (warp.thread_rank() == 0) {
			nontrivialTriangleIndex = atomicAdd(args.numProcessedBatches_nontrivial, 1);
		}
		nontrivialTriangleIndex = warp.shfl(nontrivialTriangleIndex, 0);

		if (nontrivialTriangleIndex >= *args.nontrivialTrianglesCounter) return;

		// Warp-Level Triangle Setup (Thread 0 does the heavy lifting)
		// We use local variables that we will shuffle to other threads later
		uint32_t meshIndex, triangleIndex;
		float min_x, min_y, max_x, max_y;
		vec3 a_ndc, b_ndc, c_ndc;
		vec2 a_screen, v_ab, v_ac;
		float inv_det;
		uint32_t cummulativeOffset;
		int size_x, numFragments;


		if (warp.thread_rank() == 0) {
			uint64_t packed = args.nontrivialTrianglesList[nontrivialTriangleIndex];
			meshIndex = (uint32_t)(packed >> 32llu);
			triangleIndex = (uint32_t)(packed & 0xffffffff);

			CMesh mesh = args.instances[meshIndex];
			cummulativeOffset = mesh.cummulativeTriangleCount;

			// Transform Setup
			// mat4 worldView = target.view * mesh.world;
			mat4 worldView = args.transforms[mesh.instances.offset];
			float f = args.target.proj[1][1];
			float aspect = (float)args.target.width / (float)args.target.height;

			vec3 a = getVertex<INDEXING, COMPRESSION>(mesh, 3 * triangleIndex + 0);
			vec3 b = getVertex<INDEXING, COMPRESSION>(mesh, 3 * triangleIndex + 1);
			vec3 c = getVertex<INDEXING, COMPRESSION>(mesh, 3 * triangleIndex + 2);

			a_ndc = worldToNDC(a, worldView, f, aspect);
			b_ndc = worldToNDC(b, worldView, f, aspect);
			c_ndc = worldToNDC(c, worldView, f, aspect);

			// View space for clipping logic
			vec3 a_view = vec3(worldView * vec4(a, 1.0f));
			vec3 b_view = vec3(worldView * vec4(b, 1.0f));
			vec3 c_view = vec3(worldView * vec4(c, 1.0f));

			computeScreenSpaceBoundingBox(
				a_ndc, b_ndc, c_ndc, a_view, b_view, c_view,
				worldView, f, aspect, args.target.width, args.target.height,
				&min_x, &max_x, &min_y, &max_y
			);

			// Bounding boxes computed with sutherland-hodgman in stage 2 are not 100% consistent with 
			// bounding boxes computed directly from projected vertices in stage 1. 
			// Therefore, we add a small epsilon, e.g. half of a pixel.
			constexpr float epsilon = 0.5f;
			min_x -= epsilon;
			min_y -= epsilon;
			max_x += epsilon;
			max_y += epsilon;

			a_screen = ndcToScreen(a_ndc, args.target.width, args.target.height);
			vec2 b_screen = ndcToScreen(b_ndc, args.target.width, args.target.height);
			vec2 c_screen = ndcToScreen(c_ndc, args.target.width, args.target.height);

			v_ab = b_screen - a_screen;
			v_ac = c_screen - a_screen;
			inv_det = 1.0f / (v_ab.x * v_ac.y - v_ab.y * v_ac.x);

			size_x = (int)ceilf(max_x) - (int)floorf(min_x);
			int size_y = (int)ceilf(max_y) - (int)floorf(min_y);
			numFragments = size_x * size_y;
		}

		// Broadcast setup results to all threads in warp
		meshIndex = warp.shfl(meshIndex, 0);
		triangleIndex = warp.shfl(triangleIndex, 0);
		min_x = warp.shfl(min_x, 0);
		min_y = warp.shfl(min_y, 0);
		max_x = warp.shfl(max_x, 0);
		max_y = warp.shfl(max_y, 0);
		a_ndc = {warp.shfl(a_ndc.x, 0), warp.shfl(a_ndc.y, 0), warp.shfl(a_ndc.z, 0)};
		b_ndc = {warp.shfl(b_ndc.x, 0), warp.shfl(b_ndc.y, 0), warp.shfl(b_ndc.z, 0)};
		c_ndc = {warp.shfl(c_ndc.x, 0), warp.shfl(c_ndc.y, 0), warp.shfl(c_ndc.z, 0)};
		a_screen = {warp.shfl(a_screen.x, 0), warp.shfl(a_screen.y, 0)};
		v_ab = {warp.shfl(v_ab.x, 0), warp.shfl(v_ab.y, 0)};
		v_ac = {warp.shfl(v_ac.x, 0), warp.shfl(v_ac.y, 0)};
		inv_det = warp.shfl(inv_det, 0);
		cummulativeOffset = warp.shfl(cummulativeOffset, 0);
		size_x = warp.shfl(size_x, 0);
		numFragments = warp.shfl(numFragments, 0);

		if(min_x >= max_x || min_y >= max_y) continue;

		// if (numFragments >= TILE_SIZE * TILE_SIZE) {
		if (numFragments >= TILE_SIZE * TILE_SIZE || a_ndc.z <= 0.0f || b_ndc.z <= 0.0f || c_ndc.z <= 0.0f) {
			if (warp.thread_rank() == 0) {
				
				int tile_x = (int)min_x / TILE_SIZE;
				int tile_y = (int)min_y / TILE_SIZE;
				int tiles_x = ((int)max_x + TILE_SIZE - 1) / TILE_SIZE - tile_x;
				int tiles_y = ((int)max_y + TILE_SIZE - 1) / TILE_SIZE - tile_y;
				uint32_t numTiles = tiles_x * tiles_y;

				uint32_t idx = atomicAdd(args.hugeTrianglesCounter, numTiles);
				for (uint32_t i = 0; i < numTiles; i++) {
					HugeTriangle tri;
					tri.meshIndex = meshIndex;
					tri.triangleIndex = triangleIndex;
					tri.tile_x = tile_x + (i % tiles_x);
					tri.tile_y = tile_y + (i / tiles_x);
					args.hugeTriangles[idx + i] = tri;
				}
			}
			continue;
		}

		// Fragment Rasterization Loop
		const float start_x = floorf(min_x);
		const float start_y = floorf(min_y);
		const uint64_t packed_id = cummulativeOffset + triangleIndex;

		float inv_z_a = 1.0f / a_ndc.z;
		float inv_z_b = 1.0f / b_ndc.z;
		float inv_z_c = 1.0f / c_ndc.z;
		// float inv_z_a = __fdividef(1.0f, a_ndc.z);
		// float inv_z_b = __fdividef(1.0f, b_ndc.z);
		// float inv_z_c = __fdividef(1.0f, c_ndc.z);

		for (
			int fragOffset = warp.thread_rank(); 
			fragOffset < numFragments; 
			fragOffset += 32
		) {
			int fragX = fragOffset % size_x;
			int fragY = fragOffset / size_x;

			float px = start_x + (float)fragX;
			float py = start_y + (float)fragY;
			
			float sx = px - a_screen.x + 0.5f;
			float sy = py - a_screen.y + 0.5f;

			// Barycentric coordinates using pre-computed inv_det
			float s = (sx * v_ac.y - sy * v_ac.x) * inv_det;
			float t = (v_ab.x * sy - v_ab.y * sx) * inv_det;
			float v = 1.0f - (s + t);

			if (s >= 0.0f && t >= 0.0f && v >= 0.0f) {
				int pixelID = toFramebufferIndex((int)px, (int)py, args.target.width);
				if (pixelID < args.target.width * args.target.height) {

					// float depth = v * a_ndc.z + s * b_ndc.z + t * c_ndc.z;
					// perspective-correct interpolation
					float inv_depth = v * inv_z_a + s * inv_z_b + t * inv_z_c;
					float depth = 1.0f / inv_depth;
					// float depth = __fdividef(1.0f, inv_depth);

					uint64_t pixel = pack_pixel(depth, packed_id);
					atomicMin(&args.target.framebuffer[pixelID], pixel);
				}
			}
		}
	}
}




template<IndexFetch INDEXING, Compression COMPRESSION>
void stage3_drawHugeTriangles(RasterArgs args){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	__shared__ uint32_t hugeTriIndex;

	float f = args.target.proj[1][1];
	float aspect = float(args.target.width) / float(args.target.height);
	float faI = 1.0f / (f / aspect);
	float fI = 1.0f / f;
	vec3 origin = vec4(0.0f, 0.0f, 0.0f, 1.0f);
	vec3 viewDir = vec4(0.0f, 0.0f, -1.0f, 0.0f);

	// if(grid.thread_rank() == 0){
	// 	printf("%d \n", *args.hugeTrianglesCounter);
	// }

	while(true){

		if(block.thread_rank() == 0){
			hugeTriIndex = atomicAdd(args.numProcessedHugeTriangles, 1);
		}

		block.sync();

		if(hugeTriIndex >= *args.hugeTrianglesCounter) return;

		HugeTriangle tri = args.hugeTriangles[hugeTriIndex];
		CMesh mesh = args.instances[tri.meshIndex];
		mat4 transform = args.target.proj * args.target.view * mesh.world;
		// mat4 worldView = target.view * mesh.world;
		mat4 worldView = args.transforms[mesh.instances.offset];

		vec4 a_object = getVertex<INDEXING, COMPRESSION>(mesh, 3 * tri.triangleIndex + 0);
		vec4 b_object = getVertex<INDEXING, COMPRESSION>(mesh, 3 * tri.triangleIndex + 1);
		vec4 c_object = getVertex<INDEXING, COMPRESSION>(mesh, 3 * tri.triangleIndex + 2);
		vec3 a_view = worldView * a_object;
		vec3 b_view = worldView * b_object;
		vec3 c_view = worldView * c_object;
		vec3 a_ndc = viewToNDC(a_view, f, aspect);
		vec3 b_ndc = viewToNDC(b_view, f, aspect);
		vec3 c_ndc = viewToNDC(c_view, f, aspect);
		vec2 a_screen = ndcToScreen(a_ndc, args.target.width, args.target.height);
		vec2 b_screen = ndcToScreen(b_ndc, args.target.width, args.target.height);
		vec2 c_screen = ndcToScreen(c_ndc, args.target.width, args.target.height);

		vec2 v_ab = b_screen - a_screen;
		vec2 v_ac = c_screen - a_screen;

		float min_x = tri.tile_x * TILE_SIZE;
		float max_x = min_x + TILE_SIZE;
		float min_y = tri.tile_y * TILE_SIZE;
		float max_y = min_y + TILE_SIZE;

		// clamp to screen
		min_x = clamp(min_x, 0.0f, (float)args.target.width);
		min_y = clamp(min_y, 0.0f, (float)args.target.height);
		max_x = clamp(max_x, 0.0f, (float)args.target.width);
		max_y = clamp(max_y, 0.0f, (float)args.target.height);

		int size_x = ceil(max_x) - floor(min_x);
		int size_y = ceil(max_y) - floor(min_y);
		int numFragments = size_x * size_y;
		float factor = cross(v_ab, v_ac);

		// RASTERIZE, RAYTRACE
		#define RAYTRACE

		#if defined(RASTERIZE)

		// RASTERIZE
		for(
			int fragOffset = block.thread_rank();
			fragOffset < numFragments;
			fragOffset += block.num_threads()
		){
			int fragID = fragOffset; 
			int fragX = fragID % size_x;
			int fragY = fragID / size_x;

			// FIX: Offset by 0.5f to evaluate at the center of the pixel
			vec2 pFrag = {
				floor(min_x) + float(fragX) + 0.5f,
				floor(min_y) + float(fragY) + 0.5f,
			};
			
			vec2 sample = {
				pFrag.x - a_screen.x,
				pFrag.y - a_screen.y,
			};

			float s = cross(sample, v_ac) / factor;
			float t = cross(v_ab, sample) / factor;
			float v = 1.0f - (s + t);

			// Only proceed if the fragment is inside the triangle
			if(s > 0.0f && t > 0.0f && v > 0.0f) {
				int2 pixelCoords = make_int2(pFrag.x, pFrag.y);
				int pixelID = toFramebufferIndex(pixelCoords.x, pixelCoords.y, args.target.width);

				if(pixelID < args.target.width * args.target.height){
					
					// TODO: Proper perspective-correct interpolation
					float depth = v * a_ndc.z + s * b_ndc.z + t * c_ndc.z;
					
					uint64_t pixel = pack_pixel(depth, mesh.cummulativeTriangleCount + tri.triangleIndex);

					atomicMin(&args.target.framebuffer[pixelID], pixel);
				}
			}
		}

		#elif defined(RAYTRACE)

		for(int y = 0; y < 64; y++){
			int fragX = block.thread_rank();
			int fragY = y;
			
			vec2 pFrag = {
				floor(min_x) + float(fragX),
				floor(min_y) + float(fragY),
			};

			int2 pixelCoords = make_int2(pFrag.x, pFrag.y);
			// int pixelID = pixelCoords.x + pixelCoords.y * target.width;
			int pixelID = toFramebufferIndex(pixelCoords.x, pixelCoords.y, args.target.width);
			// pixelID = clamp(pixelID, 0, int(target.width * target.height) - 1);

			float u = 2.0f * (pixelCoords.x + 0.5f) / float(args.target.width) - 1.0f;
			float v = 2.0f * (pixelCoords.y + 0.5f) / float(args.target.height) - 1.0f;

			vec3 rayDir = normalize(vec3{
				(1.0f / (f / aspect)) * u,
				(1.0f / f) * v,
				-1.0f
			});

			
			float t = intersectTriangle_mt(
				origin, rayDir, 
				a_view, b_view, c_view,
				false
			);

			// rayDir = normalize(rayDir);
			// float t = intersectTriangle(
			// 	origin, rayDir, 
			// 	a_view, b_view, c_view,
			// 	false
			// );

			// DEBUG: DRAW TILE BOUNDING BOX
			// if(fragX == 0 || fragY == 0 || fragX == 63 || fragY == 63){
			// 	float depth = 0.1f;
			// 	uint64_t udepth = __float_as_uint(depth);
			// 	uint64_t pixel = udepth << 32 | 0xffff00ff;

			// 	atomicMin(&args.target.colorbuffer[pixelID], pixel);
			// }

			// Early exit for threads that miss
			if(t == Infinity || pixelID >= args.target.width * args.target.height) {
				continue;
			}

			float depth = dot(t * rayDir, viewDir);
			// float depth = t;
			// uint64_t pixel = pack_pixel(depth, tri.meshIndex, tri.triangleIndex);
			uint64_t pixel = pack_pixel(depth, mesh.cummulativeTriangleCount + tri.triangleIndex);

			atomicMin(&args.target.framebuffer[pixelID], pixel);

		}

		#endif
	}
}


//-------------------------------------------------------------
// DEFINE COMPILE-TIME SPECIALIZATIONS TO BE EXPOSED TO THE HOST
//-------------------------------------------------------------



// INDEXBUFFER; UNCOMPRESSED
extern "C" __global__
void kernel_stage1_drawSmallTriangles_indexbuffer_uncompressed(RasterArgs args){
	stage1_drawSmallTriangles<IndexFetch::INDEXBUFFER, Compression::UNCOMPRESSED, Instancing::NO>(args);
}

extern "C" __global__
void kernel_stage2_drawMediumTriangles_indexbuffer_uncompressed(RasterArgs args) {
	stage2_drawMediumTriangles<IndexFetch::INDEXBUFFER, Compression::UNCOMPRESSED>(args);
}

extern "C" __global__
void kernel_stage3_drawHugeTriangles_indexbuffer_uncompressed(RasterArgs args) {
	stage3_drawHugeTriangles<IndexFetch::INDEXBUFFER, Compression::UNCOMPRESSED>(args);
}


// INDEXBUFFER; COMPRESSED
extern "C" __global__
void kernel_stage1_drawSmallTriangles_indexbuffer_compressed(RasterArgs args){
	stage1_drawSmallTriangles<IndexFetch::INDEXBUFFER, Compression::IX_PU16, Instancing::NO>(args);
}

extern "C" __global__
void kernel_stage2_drawMediumTriangles_indexbuffer_compressed(RasterArgs args) {
	stage2_drawMediumTriangles<IndexFetch::INDEXBUFFER, Compression::IX_PU16>(args);
}

extern "C" __global__
void kernel_stage3_drawHugeTriangles_indexbuffer_compressed(RasterArgs args) {
	stage3_drawHugeTriangles<IndexFetch::INDEXBUFFER, Compression::IX_PU16>(args);
}


// DIRECT INDEXING; UNCOMPRESSED
extern "C" __global__
void kernel_stage1_drawSmallTriangles_noindexbuffer_uncompressed(RasterArgs args){
	stage1_drawSmallTriangles<IndexFetch::DIRECT, Compression::UNCOMPRESSED, Instancing::NO>(args);
}

extern "C" __global__
void kernel_stage2_drawMediumTriangles_noindexbuffer_uncompressed(RasterArgs args) {
	stage2_drawMediumTriangles<IndexFetch::DIRECT, Compression::UNCOMPRESSED>(args);
}

extern "C" __global__
void kernel_stage3_drawHugeTriangles_noindexbuffer_uncompressed(RasterArgs args) {
	stage3_drawHugeTriangles<IndexFetch::DIRECT, Compression::UNCOMPRESSED>(args);
}


// DIRECT INDEXUNG; COMPRESSED
extern "C" __global__
void kernel_stage1_drawSmallTriangles_noindexbuffer_compressed(RasterArgs args){
	stage1_drawSmallTriangles<IndexFetch::DIRECT, Compression::IX_PU16, Instancing::NO>(args);
}

extern "C" __global__
void kernel_stage2_drawMediumTriangles_noindexbuffer_compressed(RasterArgs args) {
	stage2_drawMediumTriangles<IndexFetch::DIRECT, Compression::IX_PU16>(args);
}

extern "C" __global__
void kernel_stage3_drawHugeTriangles_noindexbuffer_compressed(RasterArgs args) {
	stage3_drawHugeTriangles<IndexFetch::DIRECT, Compression::IX_PU16>(args);
}





// INDEXBUFFER; UNCOMPRESSED; INSTANCED
extern "C" __global__
void kernel_stage1_drawSmallTriangles_indexbuffer_uncompressed_instanced(RasterArgs args){
	stage1_drawSmallTriangles<IndexFetch::INDEXBUFFER, Compression::UNCOMPRESSED, Instancing::YES>(args);
}

// INDEXBUFFER; COMPRESSED; INSTANCED
extern "C" __global__
void kernel_stage1_drawSmallTriangles_indexbuffer_compressed_instanced(RasterArgs args){
	stage1_drawSmallTriangles<IndexFetch::INDEXBUFFER, Compression::IX_PU16, Instancing::YES>(args);
}

