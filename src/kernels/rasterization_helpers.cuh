#pragma once

using glm::ivec2;
using glm::i8vec4;
using glm::vec4;

uint32_t toFramebufferIndex(int x, int y, int width){

	return x + width * y;

	// constexpr uint32_t TILE_W = 8;
	// constexpr uint32_t TILE_H = 8;
	// constexpr uint32_t TILE_PIXELS = TILE_W * TILE_H;

	// // Tile coordinates
	// uint32_t tileX = x >> 3;   // /8
	// uint32_t tileY = y >> 3;   // /8

	// // Pixel coordinates within the tile
	// uint32_t inX = x & 7;      // %8
	// uint32_t inY = y & 7;      // %8

	// // Number of tiles per row
	// uint32_t tilesPerRow = (width + 7) / 8;

	// // Tile index in the tile grid (row-major)
	// uint32_t tileIndex = tileY * tilesPerRow + tileX;

	// // Offset within the tile (row-major)
	// uint32_t inTileIndex = inY * TILE_W + inX;

	// return tileIndex * TILE_PIXELS + inTileIndex;
}

uint64_t pack_pixel(float depth, uint64_t trianglePrefix){

	// 28 bit depth; 36 bit triangle ID
	// depth: remove leftmost bit (sign) and 3 rightmost bits (mantissa)
	// With 3 bits fewer in the mantissa, we get roughly following precision:
	// - micrometer precision at a distance of 1 meter
	// - decimeter precision at a distance of 65 kilometer
	// - meter precision at 1000 kilometer

	uint64_t udepth = __float_as_uint(depth);
	udepth = udepth & 0b01111111'11111111'11111111'11111000;
	if(isinf(depth)) udepth = 0b01111111'11111111'11111111'11111000;
	udepth = udepth << 33;

	uint64_t packed = udepth | trianglePrefix;

	return packed;
}

void unpack_pixel(uint64_t packed, float* depth, uint64_t* trianglePrefix){

	uint32_t udepth = packed >> 33;
	udepth = udepth & 0b01111111'11111111'11111111'11111000;

	if(udepth == 0b01111111'11111111'11111111'11111000){
		*depth = Infinity;
	}else{
		*depth = __uint_as_float(udepth);
	}

	*trianglePrefix = packed & 0b00001111'11111111'11111111'11111111'11111111;
}


vec3 toNDC(vec3 viewSpace, float f, float aspect){
	float depth = -viewSpace.z;
	float x_ndc = (f / aspect) * viewSpace.x / depth;
	float y_ndc = f * viewSpace.y / depth;

	return vec3{x_ndc, y_ndc, depth};
}

vec3 worldToNDC(vec4 v, mat4 view, float f, float aspect){
	vec4 viewSpace = view * v;
	float depth = -viewSpace.z;
	float x_ndc = (f / aspect) * viewSpace.x / depth;
	float y_ndc = f * viewSpace.y / depth;

	return vec3{x_ndc, y_ndc, depth};
}

vec3 worldToNDC(vec3 v, mat4 view, float f, float aspect){
	vec4 viewSpace = view * vec4(v.x, v.y, v.z, 1.0f);
	float depth = -viewSpace.z;
	float x_ndc = (f / aspect) * viewSpace.x / depth;
	float y_ndc = f * viewSpace.y / depth;

	return vec3{x_ndc, y_ndc, depth};
}

vec3 viewToNDC(vec3 viewSpace, float f, float aspect){
	float depth = -viewSpace.z;
	float x_ndc = (f / aspect) * viewSpace.x / depth;
	float y_ndc = f * viewSpace.y / depth;

	return vec3{x_ndc, y_ndc, depth};
}

vec2 ndcToScreen(vec3 ndc, float width, float height){
	return vec2{
		(ndc.x * 0.5f + 0.5f) * width,
		(ndc.y * 0.5f + 0.5f) * height,
	};
}

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

namespace SutherlandHodgman {

	// Generalizing the clipping boundary for branchless math
	struct ClipEdge {
		int axis;	 // 0 for X, 1 for Y
		float sign;   // 1.0 for Max, -1.0 for Min
		float value;  // The clipping constant (e.g., 1.0f)
	};

	__device__ inline bool isInside(const vec2 p, const ClipEdge e) {
		// Returns true if point is on the "inside" of the clipping plane
		// e.g., for Left: p.x > -1.0f  =>  p.x * 1.0 > -1.0
		float val = (e.axis == 0) ? p.x : p.y;
		return (e.sign * val >= e.sign * e.value);
	}

	__device__ inline vec2 intersect(vec2 p1, vec2 p2, const ClipEdge e) {
		float t;
		if (e.axis == 0) { // Vertical Clip (Left/Right)
			t = (e.value - p1.x) / (p2.x - p1.x);
			return { e.value, p1.y + t * (p2.y - p1.y) };
		} else { // Horizontal Clip (Top/Bottom)
			t = (e.value - p1.y) / (p2.y - p1.y);
			return { p1.x + t * (p2.x - p1.x), e.value };
		}
	}

	__device__ void windowClipping(const vec2 a, const vec2 b, const vec2 c, vec2* boxMin, vec2* boxMax) {
		// We use two small buffers on the stack (registers)
		vec2 bufferA[8];
		vec2 bufferB[8];
		
		vec2* in = bufferA;
		vec2* out = bufferB;

		in[0] = a; in[1] = b; in[2] = c;
		int inCount = 3;

		// Define the 4 boundaries: Left, Right, Bottom, Top
		const ClipEdge edges[4] = {
			{0,  1.0f, -1.0f}, // Left:   x > -1
			{0, -1.0f,  1.0f}, // Right:  x <  1
			{1,  1.0f, -1.0f}, // Bottom: y > -1
			{1, -1.0f,  1.0f}  // Top:	y <  1
		};

		#pragma unroll
		for (int e = 0; e < 4; e++) {
			int outCount = 0;
			if (inCount == 0) break;

			vec2 S = in[inCount - 1]; // Start with the last point
			for (int i = 0; i < inCount; i++) {
				vec2 E = in[i];
				bool E_inside = isInside(E, edges[e]);
				bool S_inside = isInside(S, edges[e]);

				if (E_inside) {
					if (!S_inside) {
						out[outCount++] = intersect(S, E, edges[e]);
					}
					out[outCount++] = E;
				} else if (S_inside) {
					out[outCount++] = intersect(S, E, edges[e]);
				}
				S = E;
			}
			
			// Ping-pong buffers: current 'out' becomes next 'in'
			vec2* temp = in;
			in = out;
			out = temp;
			inCount = outCount;
		}

		// printf("inCount: %d\n", inCount);
		for (int i = 0; i < inCount; i++) {
			vec2 p = in[i]; // Use 'in' because the final swap moved the result there
			boxMin->x = fminf(boxMin->x, p.x);
			boxMin->y = fminf(boxMin->y, p.y);
			boxMax->x = fmaxf(boxMax->x, p.x);
			boxMax->y = fmaxf(boxMax->y, p.y);
		}
	}
}

// Möller-Trumbore
__device__ float intersectTriangle_mt(
	vec3 orig, vec3 dir,
	vec3 v0, vec3 v1, vec3 v2,
	bool cullBackFaces
) {
	vec3 e1 = v1 - v0;
	vec3 e2 = v2 - v0;

	vec3 pvec = cross(dir, e2);
	float det = dot(e1, pvec);

	constexpr float EPSILON = 1e-7f;

	// Branch 1: Single-sided rendering
	if (cullBackFaces) {
		// If determinant is near zero, ray lies in plane of triangle. 
		// If less than zero, it's a backface.
		if (det < EPSILON) return Infinity;
		
		// OPTIMIZATION 1: Since 'orig' is (0,0,0) in view space, 
		// tvec (which is orig - v0) simply becomes -v0.
		// This saves a vector subtraction.
		vec3 tvec = -v0; 
		
		// OPTIMIZATION 2: Defer the division (1.0f / det). 
		// Instead of normalizing 'u' and 'v' to [0, 1], 
		// we check them against the un-normalized 'det'.
		float u = dot(tvec, pvec);
		if (u < 0.0f || u > det) return Infinity;

		vec3 qvec = cross(tvec, e1);
		float v = dot(dir, qvec);
		if (v < 0.0f || u + v > det) return Infinity;

		// We only divide if we are guaranteed to hit the triangle.
		float t = dot(e2, qvec) / det;
		return t > 0.0f ? t : Infinity;
	} 
	// Branch 2: Double-sided rendering (from your 'false' parameter)
	else {
		if (det > -EPSILON && det < EPSILON) return Infinity;

		float inv_det = 1.0f / det;
		
		// Same zero-origin trick applies here
		vec3 tvec = -v0; 
		
		float u = dot(tvec, pvec) * inv_det;
		if (u < 0.0f || u > 1.0f) return Infinity;

		vec3 qvec = cross(tvec, e1);
		float v = dot(dir, qvec) * inv_det;
		if (v < 0.0f || u + v > 1.0f) return Infinity;

		float t = dot(e2, qvec) * inv_det;
		return t > 0.0f ? t : Infinity;
	}
}

// see: https://github.com/mrdoob/three.js/blob/46961fbdc727f7df1c26eb0d5cc833c99ebe600a/src/math/Ray.js#L540
// LICENSE: MIT (https://github.com/mrdoob/three.js/blob/dev/LICENSE)
inline float intersectTriangle(
	vec3 origin,
	vec3 direction,
	vec3 a, vec3 b, vec3 c,
	bool backfaceCulling
){

	vec3 edge1 = b - a;
	vec3 edge2 = c - a;
	vec3 normal = cross(edge1, edge2);

	float DdN = dot(direction, normal);

	float sign;
	if(DdN > 0.0f){
		if(backfaceCulling) return Infinity;
		sign = 1.0f;
	}else if(DdN < 0.0f){
		sign = -1.0f;
		DdN = -DdN;
	}else{
		return Infinity;
	}

	vec3 diff = origin - a;
	float DdQxE2 = sign * dot(direction, cross(diff, edge2));

	if(DdQxE2 < 0.0f) return Infinity;

	float DdE1xQ = sign * dot(direction, cross(edge1, diff));

	if(DdE1xQ < 0.0f) return Infinity;

	if(DdQxE2 + DdE1xQ > DdN) return Infinity;

	float QdN = -sign * dot(diff, normal);

	if(QdN < 0.0f) return Infinity;

	return QdN / DdN;
}

// see: https://github.com/mrdoob/three.js/blob/f0628f6adbff6bd9874287d70620d29a20be7dbe/src/math/Ray.js#L362
// LICENSE: MIT (https://github.com/mrdoob/three.js/blob/dev/LICENSE)
inline float intersectPlane(
	const vec3& origin, const vec3& dir,
	const vec3& normal, const float& constant
){
	float denominator = dot(normal, dir);

	if(denominator == 0.0f) {
		return Infinity;
	}

	float t = - ( dot(origin, normal) + constant) / denominator;

	if(t >= 0.0f){
		return t;
	}else{
		return Infinity;
	}
}

inline float intersectPlane(
	vec3 origin, vec3 dir, 
	vec3 a, vec3 b, vec3 c
){
	vec3 edge1 = b - a;
	vec3 edge2 = c - a;
	vec3 normal = normalize(cross(edge1, edge2));

	float d = dot(a, normal);

	return intersectPlane(origin, dir, normal, -d);
}

__device__ void computeScreenSpaceBoundingBox(
	vec3 a_ndc, vec3 b_ndc, vec3 c_ndc,
	vec3 a_view, vec3 b_view, vec3 c_view,
	mat4 worldView, float f, float aspect,
	float width, float height,
	float* min_x, float* max_x,
	float* min_y, float* max_y
) {
	vec2 boxMin = {1e10f, 1e10f};
	vec2 boxMax = {-1e10f, -1e10f};

	// 1. Bitmask for vertices behind near plane (z <= 0 in NDC usually means behind)
	// Using a bitmask reduces branching logic.
	int mask = (a_ndc.z <= 0.0f ? 1 : 0) | 
			   (b_ndc.z <= 0.0f ? 2 : 0) | 
			   (c_ndc.z <= 0.0f ? 4 : 0);

	// 2. Early exit if all are behind
	if (mask == 7) return;

	// Optimization: View-to-plane intersection for Z-aligned planes doesn't need normalize()
	constexpr float near = 0.01f;
	auto intersectNear = [&](vec3 p1, vec3 p2) {
		float t = (-near - p1.z) / (p2.z - p1.z);
		return p1 + t * (p2 - p1);
	};

	auto extend = [&](vec2 point){
		boxMin.x = min(boxMin.x, point.x);
		boxMax.x = max(boxMax.x, point.x);
		boxMin.y = min(boxMin.y, point.y);
		boxMax.y = max(boxMax.y, point.y);
	};

	constexpr bool USE_SUTHERLAND = true;

	if (mask == 0) {
		// All in front - the fast path

		if constexpr(USE_SUTHERLAND){
			SutherlandHodgman::windowClipping(a_ndc, b_ndc, c_ndc, &boxMin, &boxMax);
		}else{
			extend(a_ndc);
			extend(b_ndc);
			extend(c_ndc);
		}

	} else {
		// Handle cases with 1 or 2 points behind using a unified triangle fan approach
		// We identify the 'unique' vertex (the one that is alone in front or alone behind)
		vec3 v[3], v_view[3];
		if (mask == 1 || mask == 6) { v_view[0] = a_view; v_view[1] = b_view; v_view[2] = c_view; }
		else if (mask == 2 || mask == 5) { v_view[0] = b_view; v_view[1] = c_view; v_view[2] = a_view; }
		else { v_view[0] = c_view; v_view[1] = a_view; v_view[2] = b_view; }

		if (mask == 1 || mask == 2 || mask == 4) {
			// TWO points in front, ONE behind (mask has 1 bit set)
			// This creates a quadrilateral, which we treat as two triangles
			vec3 I1 = intersectNear(v_view[0], v_view[1]);
			vec3 I2 = intersectNear(v_view[0], v_view[2]);
			
			vec2 p0 = viewToNDC(v_view[1], f, aspect);
			vec2 p1 = viewToNDC(v_view[2], f, aspect);
			vec2 pI1 = viewToNDC(I1, f, aspect);
			vec2 pI2 = viewToNDC(I2, f, aspect);

			if constexpr(USE_SUTHERLAND){
				SutherlandHodgman::windowClipping(p0, p1, pI1, &boxMin, &boxMax);
				SutherlandHodgman::windowClipping(pI1, p1, pI2, &boxMin, &boxMax);
			}else{
				extend(p0);
				extend(p1);
				extend(pI1);
				extend(pI2);
			}

		} else {
			// ONE point in front, TWO behind (mask has 2 bits set)
			// This creates a single smaller triangle
			vec3 I1 = intersectNear(v_view[0], v_view[1]);
			vec3 I2 = intersectNear(v_view[0], v_view[2]);

			if constexpr(USE_SUTHERLAND){
				SutherlandHodgman::windowClipping(
					viewToNDC(v_view[0], f, aspect),
					viewToNDC(I1, f, aspect),
					viewToNDC(I2, f, aspect),
					&boxMin, &boxMax
				);
			}else{
				extend(viewToNDC(v_view[0], f, aspect));
				extend(	viewToNDC(I1, f, aspect));
				extend(	viewToNDC(I2, f, aspect));
			}
		}
	}

	// clamp to ndc space
	boxMin.x = clamp(boxMin.x, -1.0f, 1.0f);
	boxMax.x = clamp(boxMax.x, -1.0f, 1.0f);
	boxMin.y = clamp(boxMin.y, -1.0f, 1.0f);
	boxMax.y = clamp(boxMax.y, -1.0f, 1.0f);

	// to screen-space coordinates
	float halfW = width * 0.5f;
	float halfH = height * 0.5f;
	*min_x = boxMin.x * halfW + halfW;
	*min_y = boxMin.y * halfH + halfH;
	*max_x = boxMax.x * halfW + halfW;
	*max_y = boxMax.y * halfH + halfH;
}


__device__ void computeScreenSpaceBoundingBox2(
	vec3 a_view, vec3 b_view, vec3 c_view,
	vec3 a_ndc, vec3 b_ndc, vec3 c_ndc,
	float f, float aspect,
	float width, float height,
	float* min_x, float* max_x,
	float* min_y, float* max_y
) {
	vec2 boxMin = {1e10f, 1e10f};
	vec2 boxMax = {-1e10f, -1e10f};

	// Bitmask for vertices behind near plane (z <= 0 in NDC usually means behind)
	// Using a bitmask reduces branching logic.
	int mask = (a_ndc.z <= 0.0f ? 1 : 0) | 
			   (b_ndc.z <= 0.0f ? 2 : 0) | 
			   (c_ndc.z <= 0.0f ? 4 : 0);

	// Early exit if all are behind
	if (mask == 7) return;

	// Optimization: View-to-plane intersection for Z-aligned planes doesn't need normalize()
	constexpr float near = 0.01f;
	auto intersectNear = [&](vec3 p1, vec3 p2) {
		float t = (-near - p1.z) / (p2.z - p1.z);
		return p1 + t * (p2 - p1);
	};

	if (mask == 0) {
		// All in front - the fast path
		SutherlandHodgman::windowClipping(a_ndc, b_ndc, c_ndc, &boxMin, &boxMax);
	} else {
		// Handle cases with 1 or 2 points behind using a unified triangle fan approach
		// We identify the 'unique' vertex (the one that is alone in front or alone behind)

		vec3 v_view[3];
		if      (mask == 1 || mask == 6) { v_view[0] = a_view; v_view[1] = b_view; v_view[2] = c_view; }
		else if (mask == 2 || mask == 5) { v_view[0] = b_view; v_view[1] = c_view; v_view[2] = a_view; }
		else                             { v_view[0] = c_view; v_view[1] = a_view; v_view[2] = b_view; }

		if (mask == 1 || mask == 2 || mask == 4) {
			// TWO points in front, ONE behind (mask has 1 bit set)
			// This creates a quadrilateral, which we treat as two triangles
			vec3 I1 = intersectNear(v_view[0], v_view[1]);
			vec3 I2 = intersectNear(v_view[0], v_view[2]);
			
			vec2 p0 = viewToNDC(v_view[1], f, aspect);
			vec2 p1 = viewToNDC(v_view[2], f, aspect);
			vec2 pI1 = viewToNDC(I1, f, aspect);
			vec2 pI2 = viewToNDC(I2, f, aspect);

			SutherlandHodgman::windowClipping(p0, p1, pI1, &boxMin, &boxMax);
			SutherlandHodgman::windowClipping(pI1, p1, pI2, &boxMin, &boxMax);
		} else {
			// ONE point in front, TWO behind (mask has 2 bits set)
			// This creates a single smaller triangle
			vec3 I1 = intersectNear(v_view[0], v_view[1]);
			vec3 I2 = intersectNear(v_view[0], v_view[2]);

			SutherlandHodgman::windowClipping(
				viewToNDC(v_view[0], f, aspect),
				viewToNDC(I1, f, aspect),
				viewToNDC(I2, f, aspect),
				&boxMin, &boxMax
			);
		}
	}

	// Final viewport transformation
	// Combine the 0.5f scale and width/height into a single FMA (Fused Multiply-Add)
	float halfW = width * 0.5f;
	float halfH = height * 0.5f;
	*min_x = boxMin.x * halfW + halfW;
	*min_y = boxMin.y * halfH + halfH;
	*max_x = boxMax.x * halfW + halfW;
	*max_y = boxMax.y * halfH + halfH;
}


float cross(vec2 a, vec2 b){ 
	return a.x * b.y - a.y * b.x; 
}


// uint32_t sampleColor_nearest(
// 	uint32_t* textureData,
// 	int width,
// 	int height,
// 	vec2 uv
// ){
// 	if(textureData == nullptr) return 0;

// 	int tx = int(uv.x * float(width) + 0.5f) % width;
// 	int ty = int(uv.y * float(height) + 0.5f) % height;
// 	int texelID = tx + ty * width;
// 	texelID = clamp(texelID, 0, width * height);

// 	uint32_t color = 0;
// 	color = textureData[texelID];
// 	uint8_t *rgb = (uint8_t *)&color;

// 	return color;
// }

// uint32_t sampleColor_linear(
// 	uint32_t* textureData,
// 	int width,
// 	int height,
// 	vec2 uv
// ){
// 	if(textureData == nullptr) return 0;

// 	uint32_t color = 0xff000000;
// 	uint8_t* rgba = (uint8_t*)&color;

// 	float ftx = (uv.x - floor(uv.x)) * float(width);
// 	float fty = (uv.y - floor(uv.y)) * float(height);

// 	auto getTexel = [&](float ftx, float fty) -> vec4 {
// 		int tx = fmodf(ftx, float(width));
// 		int ty = fmodf(fty, float(height));
// 		int texelID = tx + ty * width;
// 		texelID = clamp(texelID, 0, width * height - 1);

// 		uint32_t texel = textureData[texelID];
// 		uint8_t* rgba = (uint8_t*)&texel;

// 		return vec4{rgba[0], rgba[1], rgba[2], rgba[3]};
// 	};

// 	vec4 t00 = getTexel(ftx - 0.5f, fty - 0.5f);
// 	vec4 t01 = getTexel(ftx - 0.5f, fty + 0.5f);
// 	vec4 t10 = getTexel(ftx + 0.5f, fty - 0.5f);
// 	vec4 t11 = getTexel(ftx + 0.5f, fty + 0.5f);

// 	float wx = fmodf(ftx + 0.5f, 1.0f);
// 	float wy = fmodf(fty + 0.5f, 1.0f);

// 	vec4 interpolated = 
// 		(1.0f - wx) * (1.0f - wy) * t00 + 
// 		wx * (1.0f - wy) * t10 + 
// 		(1.0f - wx) * wy * t01 + 
// 		wx * wy * t11;

// 	rgba[0] = interpolated.r;
// 	rgba[1] = interpolated.g;
// 	rgba[2] = interpolated.b;
// 	rgba[3] = 255;

// 	return color;
// }