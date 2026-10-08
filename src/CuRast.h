#pragma once

#include <vector>
#include <map>
#include <unordered_map>

#include "glm/gtc/matrix_access.hpp"

#include "CURuntime.h"
#include "MemoryManager.h"

#include "./scene/SceneNode.h"
#include "./scene/Scene.h"

#include <cuda_runtime.h>

#include "VKRenderer.h"
#include "OrbitControls.h"
#include "MouseEvents.h"
#include "CuRastSettings.h"

using glm::transpose;
using glm::vec2;
using glm::quat;
using glm::vec3;
using glm::dvec3;
using glm::mat4;
using glm::dmat4;

// Timings of the last <historySize> frames, per label
struct Timings{

	int historySize = 60;
	uint64_t counter = 0;

	map<string, vector<float>> entries;

	void add(string label, float milliseconds){

		entries[label].resize(historySize);

		int entryPos = counter % historySize;

		entries[label][entryPos] += milliseconds;
	}

	void newFrame(){

		counter++;
		int entryPos = counter % historySize;

		for(auto& [label, list] : entries){
			list[entryPos] = 0.0f;
		}
	}

	float getMean(string label){
		if(entries.find(label) == entries.end()){
			return 0.0f;
		}

		vector<float> values = entries[label]; // makes a copy before sorting
		std::sort(values.begin(), values.end());

		return values[values.size() / 2];
	}

	float getMin(string label){
		if(entries.find(label) == entries.end()){
			return 0.0f;
		}

		float min = 1000000000.0f;
		for(float value : entries[label]){
			if(value == 0.0f) continue;
			min = std::min(min, value);
		}

		if(min == 1000000000.0f) return 0.0f;

		return min;
	}

	float getMax(string label){
		if(entries.find(label) == entries.end()){
			return 0.0f;
		}

		float max = 0.0f;
		for(float value : entries[label]){
			max = std::max(max, value);
		}

		return max;
	}

};

// Global runtime state: input, camera controls, debug values and timings
struct Runtime{

	inline static vector<int> frame_keys = vector<int>();
	inline static vector<int> frame_actions = vector<int>();
	inline static vector<int> frame_mods = vector<int>();
	inline static OrbitControls* controls = new OrbitControls();
	inline static MouseEvents mouseEvents;
	inline static unordered_map<string, string> debugValues;
	inline static vector<std::pair<string, string>> debugValueList;

	inline static glm::dvec2 mousePosition = {0.0, 0.0};

	inline static bool measureTimings;
	inline static Timings timings;

};

struct CuRast{
	
	inline static CuRast* instance;

	Scene scene;

	bool requestInitScene = false;
	bool requestTdCapture = false;  // "Capture TD" button, see captureTd() in CuRast.cpp

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