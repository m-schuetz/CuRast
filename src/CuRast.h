#pragma once

#include <vector>

#include "glm/gtc/matrix_access.hpp"

#include "CudaVirtualMemory.h"
#include "CURuntime.h"
#include "MemoryManager.h"

#include "./scene/SceneNode.h"
#include "./scene/Scene.h"
#include "./scene/SNTriangles.h"
#include "./scene/SNPoints.h"
#include "./scene/SNCPoints.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "CudaModularProgram.h"

#include "VKRenderer.h"
#include "OrbitControls.h"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "json/json.hpp"
#include "stb/stb_image_resize2.h"
#include "Runtime.h"
#include "CuRastSettings.h"
#include "Benchmarking.h"

using glm::transpose;
using glm::vec2;
using glm::quat;
using glm::vec3;
using glm::dvec3;
using glm::mat4;
using glm::dmat4;

struct CuRast{
	
	inline static CuRast* instance;

	Scene scene;

	DeviceState* deviceState = nullptr;
	CUdeviceptr cptr_state;

	CommonLaunchArgs launchArgs;

	bool requestInitScene = false;

	static void setup();

	CommonLaunchArgs getCommonLaunchArgs();

	void drawGUI();
	void resetEditor();
	void inputHandling();
	Uniforms getUniforms();
	void initCudaProgram();

	// GUI
	void makeMenubar();
	void makeStats();
	void makeDirectStats();
	void makeToolbar();
	void makeDevGUI();

	// UPDATE & DRAW 
	void update();
	void render();
	void postFrame();
	void draw(Scene* scene, vector<View> views);

};