#pragma once

#include "kernels/HostDeviceInterface.h"

struct CuRastSettings{
	static inline bool showBoundingBoxes = false;
	static inline bool enableEDL = true;
	static inline bool enableFrustumCulling = true;
	static inline bool hideGUI = false;

	static inline bool showKernelInfos = false;
	static inline bool showMemoryInfos = false;
	static inline bool showTimingInfos = false;
	static inline bool showStats = false;
	static inline bool showOverlay = true;
	static inline bool showInset = false;
	static inline bool showBenchmarking = false;
	static inline int supersamplingFactor = 1;

	static inline bool enableLinearInterpolation = true;
	static inline bool enableMipMapping = true;
	static inline float threshold = 0.0f;
	static inline bool freezeFrustum = false;
	static inline bool enableSSAO = false;
	static inline bool enableDiffuseLighting = false;
	static inline bool disableInstancing = false;
	static inline bool enableObjectPicking = false;
	static inline shared_ptr<string> requestScreenshot = nullptr; // Set to path of screenshot, or empty string for auto path
	static inline vec4 background = {1.0f, 1.0f, 1.0f, 1.0f};


	static inline DisplayAttribute displayAttribute = DisplayAttribute::TEXTURE;
	static inline bool showWireframe = false;

	// static inline uint32_t rasterizer = RASTERIZER_VULKAN_INDEXPULLING_INSTANCED;
	// static inline uint32_t rasterizer = RASTERIZER_VISBUFFER_INDEXED;
	static inline uint32_t rasterizer = RASTERIZER_VISBUFFER_INSTANCED;
	// static inline uint32_t rasterizer = RASTERIZER_VULKAN_INDEXPULLING_VISBUFFER;
	// static inline uint32_t rasterizer = RASTERIZER_VISBUFFER_CLUSTERS;

	static inline bool benchmark_load_sponza = false;
	static inline bool benchmark_load_lantern = false;
};

// Enabling this makes CuRast allocate memory for geometry with the Vulkan API instead of CUDA.
// - Needs to be enabled to render things in Vulkan.
// - It's still shared to CUDA because LargeGlbLoader.h uses CUDA for streaming mesh data to GPU. 
// - It's off by default because allocating in Vulkan vs. CUDA has different performance implications.
// - From observations, we assume that the Vulkan buffer implicitly enables compression. 
//   This makes some scenarios faster (e.g. uncompressed geometry) but others slower (resolve).
// - Explicitly enabling compressed CUDA buffers seems to equalize the performance.
// - For benchmarking, we enable it for Vulkan measurements and disable it for CUDA measuerements.
// #define USE_VULKAN_SHARED_MEMORY