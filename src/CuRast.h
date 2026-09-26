#pragma once

#include <vector>

#include "glm/gtc/matrix_access.hpp"

#include "CudaVirtualMemory.h"
#include "CURuntime.h"
#include "MemoryManager.h"

#include "./scene/SceneNode.h"
#include "./scene/Scene.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "CudaModularProgram.h"

#include "VKRenderer.h"
#include "OrbitControls.h"
#include "Runtime.h"
#include "CuRastSettings.h"

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

	bool requestInitScene = false;

	static void setup();

	void drawGUI();
	void resetEditor();
	void inputHandling();

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