#pragma once

#include "kernels/HostDeviceInterface.h"

// How the points of Potree octree nodes get to the GPU
enum PotreeRenderPath : int {
	POTREE_MEMORY_MAPPED  = 0, // kernel reads from the memory-mapped octree.bin (requires HMM)
	POTREE_DIRECT_STORAGE = 1, // visible nodes are read into VRAM via cuFile (GPUDirect Storage) each frame
};

// Where the kernels read the clusters, vertices and triangles of clustered LOD meshes from
enum ClusterRenderPath : int {
	CLUSTERS_VRAM           = 0, // copied to VRAM on first use
	CLUSTERS_MEMORY_MAPPED  = 1, // kernels read from the memory-mapped files (requires HMM)
};

struct CuRastSettings{
	static inline bool enableEDL = true;
	static inline bool enableFrustumCulling = true;
	static inline bool hideGUI = false;

	static inline bool showKernelInfos = false;
	static inline bool showMemoryInfos = false;
	static inline bool showTimingInfos = false;
	static inline bool showStats = false;
	static inline bool showOverlay = true;
	static inline bool showInset = false;
	static inline int supersamplingFactor = 1;
	static inline int64_t pointBudget = 5'000'000; // max. number of points rendered from Potree octrees
	static inline int potreeRenderPath = POTREE_MEMORY_MAPPED;
	static inline float lodErrorThreshold = 1.0f;  // clustered LOD: max. projected simplification error, in pixels
	static inline int clusterColorMode = CLUSTER_COLOR_TEXTURE;
	static inline int clusterRenderPath = CLUSTERS_VRAM;

	static inline vec4 background = {1.0f, 1.0f, 1.0f, 1.0f};
};
