

constexpr int dbg_pixel_x = 1000;
constexpr int dbg_pixel_y = 500;
constexpr bool enableDebugPixel = false;

bool isDbgPixel(int x, int y){
	return enableDebugPixel && (x == dbg_pixel_x && y == dbg_pixel_y);
}

#if defined(COMPRESS_IX_U16)
	vec4 getVertex(CMesh& mesh, uint32_t vertexIndex){
		
		// Note: Dynamically Checking whether a mesh is compressed or not makes rendering about 1-2% slower.
		// Therefore, we are launching different kernels for compressed and uncompressed meshes.

		// Note: Minimal performance improvements of 0.3% by using precomputed mesh.bitsPerIndex

		// uint32_t indexRange = mesh.index_max - mesh.index_min;
		// uint64_t bitsPerIndex = ceil(log2f(float(indexRange + 1)));
		// uint32_t bitsPerIndex = indexRange == 0 ? 0 : 32 - __clz(indexRange);
		// uint32_t resolvedIndex = BitEdit::readU32(mesh.indices, bitsPerIndex * vertexIndex, bitsPerIndex) + mesh.index_min;
		uint32_t resolvedIndex = BitEdit::readU32(mesh.indices, mesh.bitsPerIndex * vertexIndex, mesh.bitsPerIndex) + mesh.index_min;

		uint16_t* positions = (uint16_t*)mesh.positions;
		uint16_t X = positions[3 * resolvedIndex + 0];
		uint16_t Y = positions[3 * resolvedIndex + 1];
		uint16_t Z = positions[3 * resolvedIndex + 2];

		vec4 position;
		// "normal" way to dequantize
		// vec3 aabbSize = mesh.aabb.max - mesh.aabb.min;
		// position.x = (float(X) / 65536.0f) * aabbSize.x + mesh.aabb.min.x;
		// position.y = (float(Y) / 65536.0f) * aabbSize.y + mesh.aabb.min.y;
		// position.z = (float(Z) / 65536.0f) * aabbSize.z + mesh.aabb.min.z;
		// Using a pre-computed compression factor, we get a tiny (+0.2%) speed improvement.
		position.x = float(X) * mesh.compressionFactor.x + mesh.aabb.min.x;
		position.y = float(Y) * mesh.compressionFactor.y + mesh.aabb.min.y;
		position.z = float(Z) * mesh.compressionFactor.z + mesh.aabb.min.z;
		position.w = 1.0f;
		
		// TESTING: round to 0.001
		// pos.x = floor(pos.x * 10000.0f) / 10000.0f;
		// pos.y = floor(pos.y * 10000.0f) / 10000.0f;
		// pos.z = floor(pos.z * 10000.0f) / 10000.0f;

		return position;
	};
#else

	vec4 getVertex(CMesh& mesh, uint32_t vertexIndex){
		uint32_t resolvedIndex = mesh.indices[vertexIndex];
		vec4 pos = vec4(mesh.positions[resolvedIndex], 1.0f);

		return pos;
	};
#endif

extern "C" __global__
void kernel_stage3_drawHugeTriangles(
	CMesh* meshes,
	uint32_t numMeshes,
	mat4* transforms,
	HugeTriangle* hugeTriangles,
	uint32_t* numHugeTriangles,
	uint32_t* numHugeTrianglesProcessedCounter,
	RenderTarget target
) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	// if(grid.thread_rank() == 0){
	// 	printf("numHugeTriangles: %u \n", *numHugeTriangles);
	// }

	__shared__ uint32_t hugeTriIndex;

	float f = target.proj[1][1];
	float aspect = float(target.width) / float(target.height);
	float faI = 1.0f / (f / aspect);
	float fI = 1.0f / f;
	vec3 origin = vec4(0.0f, 0.0f, 0.0f, 1.0f);
	vec3 viewDir = vec4(0.0f, 0.0f, -1.0f, 0.0f);

	while(true){

		if(block.thread_rank() == 0){
			hugeTriIndex = atomicAdd(numHugeTrianglesProcessedCounter, 1);
		}

		block.sync();

		if(hugeTriIndex >= *numHugeTriangles) return;

		HugeTriangle tri = hugeTriangles[hugeTriIndex];
		CMesh mesh = meshes[tri.meshIndex];
		mat4 transform = target.proj * target.view * mesh.world;
		// mat4 worldView = target.view * mesh.world;
		mat4 worldView = transforms[mesh.instances.offset];

		vec4 v_0 = getVertex(mesh, 3 * tri.triangleIndex + 0);
		vec4 v_1 = getVertex(mesh, 3 * tri.triangleIndex + 1);
		vec4 v_2 = getVertex(mesh, 3 * tri.triangleIndex + 2);

		vec3 a_view = worldView * v_0;
		vec3 b_view = worldView * v_1;
		vec3 c_view = worldView * v_2;

		vec4 w_0 = mesh.world * v_0;
		vec4 w_1 = mesh.world * v_1;
		vec4 w_2 = mesh.world * v_2;

		vec4 p_0 = toScreenCoord(v_0, transform, target.width, target.height);
		vec4 p_1 = toScreenCoord(v_1, transform, target.width, target.height);
		vec4 p_2 = toScreenCoord(v_2, transform, target.width, target.height);
		
		vec2 v_01 = {p_1.x - p_0.x, p_1.y - p_0.y};
		vec2 v_02 = {p_2.x - p_0.x, p_2.y - p_0.y};

		float min_x = tri.tile_x * TILE_SIZE;
		float max_x = min_x + TILE_SIZE;
		float min_y = tri.tile_y * TILE_SIZE;
		float max_y = min_y + TILE_SIZE;

		// clamp to screen
		min_x = clamp(min_x, 0.0f, (float)target.width);
		min_y = clamp(min_y, 0.0f, (float)target.height);
		max_x = clamp(max_x, 0.0f, (float)target.width);
		max_y = clamp(max_y, 0.0f, (float)target.height);

		int size_x = ceil(max_x) - floor(min_x);
		int size_y = ceil(max_y) - floor(min_y);
		int numFragments = size_x * size_y;
		float factor = cross(v_01, v_02);

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
				floor(min_y) + float(fragY) + 0.5f
			};
			
			vec2 sample = {pFrag.x - p_0.x, pFrag.y - p_0.y};

			float s = cross(sample, v_02) / factor;
			float t = cross(v_01, sample) / factor;
			float v = 1.0f - (s + t);

			// Only proceed if the fragment is inside the triangle
			if(s > 0.0f && t > 0.0f && v > 0.0f) {
				int2 pixelCoords = make_int2(pFrag.x, pFrag.y);
				int pixelID = toFramebufferIndex(pixelCoords.x, pixelCoords.y, target.width);

				if(pixelID < target.width * target.height){
					
					// TODO: Proper perspective-correct interpolation
					float depth = v * p_0.w + s * p_1.w + t * p_2.w;
					
					uint64_t pixel = pack_pixel(depth, mesh.cummulativeTriangleCount + tri.triangleIndex);

					atomicMin(&target.framebuffer[pixelID], pixel);
				}
			}
		}

		#elif defined(RAYTRACE)

		for(int y = 0; y < 64; y++){
			int fragX = block.thread_rank();
			int fragY = y;
			
			vec2 pFrag = {
				floor(min_x) + float(fragX) + 0.5f,
				floor(min_y) + float(fragY) + 0.5f};

			int2 pixelCoords = make_int2(pFrag.x, pFrag.y);
			// int pixelID = pixelCoords.x + pixelCoords.y * target.width;
			int pixelID = toFramebufferIndex(pixelCoords.x, pixelCoords.y, target.width);
			// pixelID = clamp(pixelID, 0, int(target.width * target.height) - 1);

			float u = 2.0f * pixelCoords.x / float(target.width) - 1.0f;
			float v = 2.0f * pixelCoords.y / float(target.height) - 1.0f;

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

			// Early exit for threads that miss
			if(t == Infinity || pixelID >= target.width * target.height) {
				continue;
			}

			float depth = dot(t * rayDir, viewDir);
			// float depth = t;
			// uint64_t pixel = pack_pixel(depth, tri.meshIndex, tri.triangleIndex);
			uint64_t pixel = pack_pixel(depth, mesh.cummulativeTriangleCount + tri.triangleIndex);

			atomicMin(&target.framebuffer[pixelID], pixel);

		}

		#endif


	}

}