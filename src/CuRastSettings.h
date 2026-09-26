#pragma once

#include "kernels/HostDeviceInterface.h"

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

	static inline shared_ptr<string> requestScreenshot = nullptr; // Set to path of screenshot, or empty string for auto path
	static inline vec4 background = {1.0f, 1.0f, 1.0f, 1.0f};
};
