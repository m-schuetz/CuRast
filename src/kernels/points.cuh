#pragma once

// Shared by the point cloud kernels (laspoints.cu, potreeFileRenderer.cu, potreeDirectStorageRenderer.cu):
// How a point that projects to a pixel is written into the RenderTarget in each PointPass.
// Per point, call depthTestPoint(), and only if it passes, read the point's color and call writePoint().

#include <string>

#include "./HostDeviceInterface.h"

// Whether the point is drawn with its color in this pass.
// - POINT_PASS_DEPTH: writes the point's depth, and returns false as the pass needs no colors.
// - POINT_PASS_ACCUMULATE: false if the point is too far behind the closest point of the pixel.
__device__ inline bool depthTestPoint(const RenderTarget& target, PointPass pass, i32 pixelID, float depth){
	if(pass == POINT_PASS_DEPTH){
		// positive floats have the same order as their bits
		u32 udepth = __float_as_uint(depth);

		if(udepth < target.pointDepthbuffer[pixelID]){
			atomicMin(&target.pointDepthbuffer[pixelID], udepth);
		}

		return false;
	}else if(pass == POINT_PASS_ACCUMULATE){
		float closestDepth = __uint_as_float(target.pointDepthbuffer[pixelID]);

		return depth <= closestDepth * (1.0f + HQ_POINT_DEPTH_RANGE);
	}

	return true;
}

// Writes a point that passed depthTestPoint(). color: RGBA8, with alpha 255
__device__ inline void writePoint(const RenderTarget& target, PointPass pass, i32 pixelID, float depth, u32 color){
	if(pass == POINT_PASS_ACCUMULATE){
		// Two 64 bit atomics instead of four 32 bit ones. Sums only carry over into the next channel
		// with more than 2^32 / 255 (16.8M) points in a pixel.
		u64 r = (color >>  0) & 0xff;
		u64 g = (color >>  8) & 0xff;
		u64 b = (color >> 16) & 0xff;

		atomicAdd((unsigned long long*)&target.pointAccumbuffer[2 * pixelID + 0], (unsigned long long)(r | (g << 32)));
		atomicAdd((unsigned long long*)&target.pointAccumbuffer[2 * pixelID + 1], (unsigned long long)(b | (1ull << 32)));
	}else{
		u64 udepth = __float_as_uint(depth);
		u64 fragment = udepth << 32 | color;

		if(fragment < target.colorbuffer[pixelID]){
			atomicMin((unsigned long long*)&target.colorbuffer[pixelID], (unsigned long long)fragment);
		}
	}
}

// Label of a point cloud kernel's timings in a pass, e.g. "kernel_drawLasPoints (depth)"
inline std::string pointPassLabel(std::string kernel, PointPass pass){
	if(pass == POINT_PASS_DEPTH) return kernel + " (depth)";
	if(pass == POINT_PASS_ACCUMULATE) return kernel + " (accumulate)";

	return kernel;
}
