// mesh_visbuffer_resolve.slang — Textured visibility buffer resolve pass
// Reads a global triangle index from the visibility buffer, binary-searches
// the cumulative triangle count array to find the mesh and local triangle index,
// then reconstructs UVs via perspective-correct barycentric interpolation and samples the texture.
// slangc mesh_visbuffer_resolve.slang -target spirv -fvk-use-entrypoint-name -capability spvBindlessTextureNV -source-embed-style u32 -source-embed-name mesh_visbuffer_resolve_spv -o mesh_visbuffer_resolve.h

[[vk::binding(0, 0)]] RWTexture2D<uint> visbuffer;

struct Node {
	column_major float4x4 world;
	column_major float4x4 worldViewProj;
	float3 *position;  // unused — position comes from vertex buffer
	float2 *uv;        // unused — uv comes from vertex buffer
	uint32_t *indices; // unused — index buffer is bound separately
	uint32_t textureHandle;
	int width;
	int height;
	float3 compressionFactor;
	float min_x;
	float min_y;
	float min_z;
	uint32_t index_min;
	uint32_t bitsPerIndex;
};

[[vk::push_constant]]
cbuffer PC
{
	column_major float4x4 viewProj;
	Node*                 nodes;
	uint                  screenWidth;
	uint                  screenHeight;
	uint32_t*             cumulativeTriCounts;
	uint                  nodeCount;
	uint                  pad0;          // padding to align cameraPos to 16 bytes (offset 96)
	float4                cameraPos;     // world-space camera position (offset 96)
	float2                tanHalfFov;    // (1/proj[0][0], 1/proj[1][1])  (offset 112)
	float2                viewDirPad;    // padding to align viewInvRow0 to 16 bytes (offset 120)
	float4                viewInvRow0;   // row 0 of inverse view matrix (offset 128)
	float4                viewInvRow1;   // row 1 of inverse view matrix (offset 144)
	float4                viewInvRow2;   // row 2 of inverse view matrix (offset 160)
};

struct VertOut
{
	float4 pos : SV_Position;
};

[shader("vertex")]
VertOut main_vertex(uint vertexID : SV_VertexID)
{
	float2 positions[3] = {
		float2(-1.0, -1.0),
		float2( 3.0, -1.0),
		float2(-1.0,  3.0),
	};
	VertOut o;
	o.pos = float4(positions[vertexID], 0.0, 1.0);

	return o;
}

uint32_t readBits(uint32_t* data, uint32_t bitOffset, uint32_t bitCount)
{
	uint32_t wordIndex = bitOffset >> 5;
	uint32_t bitStart  = bitOffset & 31;
	uint32_t word      = data[wordIndex];

	if (bitStart + bitCount > 32)
	{
		uint32_t bitsInFirst  = 32 - bitStart;
		uint32_t bitsInSecond = bitCount - bitsInFirst;
		uint32_t lo = word >> bitStart;
		uint32_t hi = data[wordIndex + 1] & ((1u << bitsInSecond) - 1u);
		return lo | (hi << bitsInFirst);
	}

	uint32_t mask = (bitCount == 32) ? 0xFFFFFFFFu : ((1u << bitCount) - 1u);
	return (word >> bitStart) & mask;
}

float3 decompressPos(vector<uint16_t, 3> u16, Node node)
{
	return float3(
		float(u16.x) * node.compressionFactor.x + node.min_x,
		float(u16.y) * node.compressionFactor.y + node.min_y,
		float(u16.z) * node.compressionFactor.z + node.min_z
	);
}

// Barycentric coordinates of point p in triangle (a, b, c) — all in the same 2D space.
// Returns (w_a, w_b, w_c).
float3 barycentrics2D(float2 p, float2 a, float2 b, float2 c)
{
	float2 v0 = b - a, v1 = c - a, v2 = p - a;
	float d00 = dot(v0, v0);
	float d01 = dot(v0, v1);
	float d11 = dot(v1, v1);
	float d20 = dot(v2, v0);
	float d21 = dot(v2, v1);
	float denom = d00 * d11 - d01 * d01;
	float wb = (d11 * d20 - d01 * d21) / denom;
	float wc = (d00 * d21 - d01 * d20) / denom;
	return float3(1.0 - wb - wc, wb, wc);
}

// Screen-space gradients of the barycentric weights w_b and w_c.
// dwb/dwc are float2: .x = d/d(pixel.x), .y = d/d(pixel.y).
// w_a gradient = -(dwb + dwc).
void barycentricDerivatives(float2 a, float2 b, float2 c,
                             out float2 dwb, out float2 dwc)
{
	float2 v0 = b - a, v1 = c - a;
	float d00   = dot(v0, v0);
	float d01   = dot(v0, v1);
	float d11   = dot(v1, v1);
	float denom = d00 * d11 - d01 * d01;
	dwb = (d11 * v0 - d01 * v1) / denom;
	dwc = (d00 * v1 - d01 * v0) / denom;
}

// Compute the mip LOD for a perspective-correct UV at a screen pixel.
// Uses the UV footprint in texel space (OpenGL/Vulkan standard formula).
float computeLod(float2 uv,
                 float2 uv0, float2 uv1, float2 uv2,
                 float4 cp0, float4 cp1, float4 cp2,
                 float2 s0,  float2 s1,  float2 s2,
                 float wSum, int texW, int texH)
{
	float2 dwb, dwc;
	barycentricDerivatives(s0, s1, s2, dwb, dwc);
	float2 dwa = -(dwb + dwc);

	// Perspective-weight gradients: d(bc_i / cp_i.w) / d(pixel)
	float2 dwc0 = dwa / cp0.w;
	float2 dwc1 = dwb / cp1.w;
	float2 dwc2 = dwc / cp2.w;

	// UV numerator gradients (chain rule over N = wc0*uv0 + wc1*uv1 + wc2*uv2)
	float2 dNdx = dwc0.x * uv0 + dwc1.x * uv1 + dwc2.x * uv2;
	float2 dNdy = dwc0.y * uv0 + dwc1.y * uv1 + dwc2.y * uv2;
	float  dDdx = dwc0.x + dwc1.x + dwc2.x;
	float  dDdy = dwc0.y + dwc1.y + dwc2.y;

	// UV = N/D  ->  d(UV)/d(p) = (dN - UV*dD) / D
	float2 duvdx = (dNdx - uv * dDdx) / wSum;
	float2 duvdy = (dNdy - uv * dDdy) / wSum;

	// Convert to texel space and pick the max axis footprint
	float2 texSize    = float2(texW, texH);
	float2 duvdx_tex  = duvdx * texSize;
	float2 duvdy_tex  = duvdy * texSize;
	float  maxLenSq   = max(dot(duvdx_tex, duvdx_tex), dot(duvdy_tex, duvdy_tex));
	return max(0.5 * log2(maxLenSq), 0.0);
}

// Compute world-space ray direction for screen pixel (px, py).
// tanHalfFov.x = 1/proj[0][0], tanHalfFov.y = 1/proj[1][1].
// viewInvRow0/1/2 are rows 0-2 of the inverse view matrix.
float3 computeRayDir(float px, float py)
{
	float u = 2.0 * px / float(screenWidth)  - 1.0;
	float v = 2.0 * py / float(screenHeight) - 1.0;
	float3 d = normalize(float3(tanHalfFov.x * u, tanHalfFov.y * v, -1.0));
	// Rotate view-space direction to world space using rows of viewInv.
	return normalize(float3(
		dot(viewInvRow0.xyz, d),
		dot(viewInvRow1.xyz, d),
		dot(viewInvRow2.xyz, d)
	));
}

// Ray-plane intersection: plane defined by dot(normal, p) = d.
// Returns t such that origin + t*rayDir lies on the plane.
float intersectPlane(float3 origin, float3 rayDir, float3 normal, float d)
{
	return (d - dot(normal, origin)) / dot(normal, rayDir);
}

// 2D barycentric coordinates of p in world-space triangle (a, b, c).
// Returns (s, t) such that p ~= a + s*(b-a) + t*(c-a).
float2 worldBarycentrics(float3 p, float3 a, float3 b, float3 c)
{
	float3 v0 = b - a;
	float3 v1 = c - a;
	float3 v2 = p - a;
	float d00 = dot(v0, v0);
	float d01 = dot(v0, v1);
	float d11 = dot(v1, v1);
	float d20 = dot(v2, v0);
	float d21 = dot(v2, v1);
	float denom = d00 * d11 - d01 * d01;
	float s = (d11 * d20 - d01 * d21) / denom;
	float t = (d00 * d21 - d01 * d20) / denom;
	return float2(s, t);
}

[shader("fragment")]
float4 main_fragment_compressed(VertOut i) : SV_Target0
{
	int2 coord  = int2(i.pos.xy);
	uint packed = visbuffer[coord];

	if (packed == 0xFFFFFFFFu){
		//return float4(0.0, 0.0, 0.0, 1.0); // background
		discard;
	}

	// Binary search: find largest meshIndex such that cumulativeTriCounts[meshIndex] <= packed.
	uint lo = 0u;
	uint hi = nodeCount - 1u;
	while (lo < hi)
	{
		uint mid = (lo + hi + 1u) / 2u;
		if (cumulativeTriCounts[mid] <= packed)
			lo = mid;
		else
			hi = mid - 1u;
	}
	uint meshIndex = lo;
	uint triIndex  = packed - cumulativeTriCounts[meshIndex];

	Node node = nodes[meshIndex];

	// Retrieve the three vertex indices for this triangle.
	uint bpv = node.bitsPerIndex;
	uint i0  = readBits(node.indices, bpv * (triIndex * 3u + 0u), bpv) + node.index_min;
	uint i1  = readBits(node.indices, bpv * (triIndex * 3u + 1u), bpv) + node.index_min;
	uint i2  = readBits(node.indices, bpv * (triIndex * 3u + 2u), bpv) + node.index_min;

	// Transform vertices to clip space.
	vector<uint16_t, 3>* positionBuffer = reinterpret<vector<uint16_t, 3>*>(node.position);
	float4 cp0 = mul(viewProj, mul(node.world, float4(decompressPos(positionBuffer[i0], node), 1.0)));
	float4 cp1 = mul(viewProj, mul(node.world, float4(decompressPos(positionBuffer[i1], node), 1.0)));
	float4 cp2 = mul(viewProj, mul(node.world, float4(decompressPos(positionBuffer[i2], node), 1.0)));

	// Project to screen space (same mapping as the hardware rasterizer).
	float2 screen = float2(float(screenWidth), float(screenHeight));
	float2 s0 = (cp0.xy / cp0.w + 1.0) * 0.5 * screen;
	float2 s1 = (cp1.xy / cp1.w + 1.0) * 0.5 * screen;
	float2 s2 = (cp2.xy / cp2.w + 1.0) * 0.5 * screen;

	// Compute screen-space barycentrics for the pixel center.
	// Clamp to [0,1] and renormalize to prevent UV extrapolation at edges
	// (hardware rasterization uses sub-pixel precision; floating-point reconstruction
	// can yield slightly negative barycentrics at edge pixels, causing UV artifacts
	// on isolated triangles that have no adjacent triangle to mask the error).
	float3 bc = barycentrics2D(float2(coord) + 0.5, s0, s1, s2);
	//bc = max(bc, float3(0.0, 0.0, 0.0));
	bc /= (bc.x + bc.y + bc.z);

	float4 color = float4(0.0f, 0.0f, 0.0f, 1.0f);

	if(node.uv){
		// UV interpolation.
		float2 uv0 = node.uv[i0];
		float2 uv1 = node.uv[i1];
		float2 uv2 = node.uv[i2];

		float wc0 = bc.x / cp0.w;
		float wc1 = bc.y / cp1.w;
		float wc2 = bc.z / cp2.w;
		float wSum = wc0 + wc1 + wc2;
		float2 uv = (wc0 * uv0 + wc1 * uv1 + wc2 * uv2) / wSum;

		uv = clamp(fmod(uv, 1.0), 0.0, 1.0);
		Sampler2D tex = Sampler2D.Handle(int(node.textureHandle));

		float lod = computeLod(uv, uv0, uv1, uv2, cp0, cp1, cp2, s0, s1, s2, wSum, node.width, node.height);
		color = tex.SampleLevel(uv, lod);
	}else{
		color = float4(1.0, 1.0, 1.0, 1.0);
	}

	// Eye-Dome Lighting: darkens pixels at depth discontinuities to enhance depth perception.
	// Matches the CUDA EDL in rasterization_utils.cu (log2 response, 4 cardinal neighbors).
	//float depth   = 1.0 / (bc.x / cp0.w + bc.y / cp1.w + bc.z / cp2.w);
	//float edlSum  = 0.0;
	//edlSum += max(log2(depth) - log2(getViewDepthCompressed(coord + int2(-1,  0))), 0.0);
	//edlSum += max(log2(depth) - log2(getViewDepthCompressed(coord + int2( 1,  0))), 0.0);
	//edlSum += max(log2(depth) - log2(getViewDepthCompressed(coord + int2( 0, -1))), 0.0);
	//edlSum += max(log2(depth) - log2(getViewDepthCompressed(coord + int2( 0,  1))), 0.0);
	//float edlStrength = 1.5;
	//float shade = exp(-(edlSum / 4.0) * 300.0 * edlStrength);
	//shade = clamp(shade, 0.3, 1.0) * 0.7 + 0.3;
	//color.rgb *= shade;

	return color;
}


[shader("fragment")]
float4 main_fragment(VertOut i) : SV_Target0
{
	int2 coord = int2(i.pos.xy);
	uint totalTriangleIndex = visbuffer[coord];

	if (totalTriangleIndex == 0xFFFFFFFFu){
		discard;
		//return float4(0.0, 0.0, 0.0, 0.0); // background
	}

	// Binary search: find largest meshIndex such that cumulativeTriCounts[meshIndex] <= totalTriangleIndex.
	uint meshIndex = 0u;
	uint lo2 = 0u, hi2 = nodeCount - 1u;
	while (lo2 < hi2)
	{
		uint mid2 = (lo2 + hi2 + 1u) / 2u;
		if (cumulativeTriCounts[mid2] <= totalTriangleIndex) lo2 = mid2;
		else                                                  hi2 = mid2 - 1u;
	}
	meshIndex = lo2;

	uint triIndex = totalTriangleIndex - cumulativeTriCounts[meshIndex];
	Node node = nodes[meshIndex];

	uint i0 = node.indices[3u * triIndex + 0u];
	uint i1 = node.indices[3u * triIndex + 1u];
	uint i2 = node.indices[3u * triIndex + 2u];

	// World-space vertex positions.
	float3 a_world = mul(node.world, float4(node.position[i0], 1.0)).xyz;
	float3 b_world = mul(node.world, float4(node.position[i1], 1.0)).xyz;
	float3 c_world = mul(node.world, float4(node.position[i2], 1.0)).xyz;

	// Camera origin in world space.
	float3 origin = cameraPos.xyz;

	// Cast rays from the camera through this pixel and its three neighbours
	// (forming a 2x2 block) to estimate the UV footprint for mip LOD.
	float3 rayDir    = computeRayDir(float(coord.x) + 0.5, float(coord.y) + 0.5);
	float3 rayDir_10 = computeRayDir(float(coord.x) + 1.5, float(coord.y) + 0.5);
	float3 rayDir_01 = computeRayDir(float(coord.x) + 0.5, float(coord.y) + 1.5);
	float3 rayDir_11 = computeRayDir(float(coord.x) + 1.5, float(coord.y) + 1.5);

	// Triangle plane: dot(normal, p) = planeD.
	float3 edge1  = b_world - a_world;
	float3 edge2  = c_world - a_world;
	float3 planeN = normalize(cross(edge1, edge2));
	float  planeD = dot(a_world, planeN);

	// Intersect each ray with the triangle plane.
	float t    = intersectPlane(origin, rayDir,    planeN, planeD);
	float t_10 = intersectPlane(origin, rayDir_10, planeN, planeD);
	float t_01 = intersectPlane(origin, rayDir_01, planeN, planeD);
	float t_11 = intersectPlane(origin, rayDir_11, planeN, planeD);

	// World-space intersection points.
	float3 p    = origin + t    * rayDir;
	float3 p_10 = origin + t_10 * rayDir_10;
	float3 p_01 = origin + t_01 * rayDir_01;
	float3 p_11 = origin + t_11 * rayDir_11;

	// World-space barycentric coordinates at each intersection.
	float2 stv    = worldBarycentrics(p,    a_world, b_world, c_world);
	float2 stv_10 = worldBarycentrics(p_10, a_world, b_world, c_world);
	float2 stv_01 = worldBarycentrics(p_01, a_world, b_world, c_world);
	float2 stv_11 = worldBarycentrics(p_11, a_world, b_world, c_world);

	float4 color = float4(0.0, 0.0, 0.0, 1.0);

	if (node.uv){
		float2 uv0 = node.uv[i0];
		float2 uv1 = node.uv[i1];
		float2 uv2 = node.uv[i2];

		// Interpolate UV at the centre pixel and its three neighbours.
		float2 uv    = uv0 * (1.0 - stv.x    - stv.y   ) + uv1 * stv.x    + uv2 * stv.y;
		float2 uv_10 = uv0 * (1.0 - stv_10.x - stv_10.y) + uv1 * stv_10.x + uv2 * stv_10.y;
		float2 uv_01 = uv0 * (1.0 - stv_01.x - stv_01.y) + uv1 * stv_01.x + uv2 * stv_01.y;
		float2 uv_11 = uv0 * (1.0 - stv_11.x - stv_11.y) + uv1 * stv_11.x + uv2 * stv_11.y;

		// LOD from the UV footprint over the 2x2 pixel block (matches resolve.cu).
		float2 uvmax = max(max(uv, uv_10), max(uv_01, uv_11));
		float2 uvmin = min(min(uv, uv_10), min(uv_01, uv_11));
		float2 uvRange = uvmax - uvmin;
		float2 texSize = float2(node.width, node.height);
		float2 dx = uvRange.x * texSize;
		float2 dy = uvRange.y * texSize;
		float lod = max(0.5 * log2(max(dot(dx, dx), dot(dy, dy))), 0.0);

		uv.x = uv.x - floor(uv.x);
		uv.y = uv.y - floor(uv.y);

		Sampler2D tex = Sampler2D.Handle(int(node.textureHandle));
		color = tex.SampleLevel(uv, lod);
	} else {
		color = float4(1.0, 1.0, 1.0, 1.0);
	}

	return color;
}