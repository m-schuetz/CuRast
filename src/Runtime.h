
#pragma once

#include <string>
#include <unordered_map>
#include <map>
#include "OrbitControls.h"

#include "glm/common.hpp"

#include "MouseEvents.h"

using namespace std;

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

	float getAverage(string label){
		if(entries.find(label) == entries.end()){
			return 0.0f;
		}

		float sum = 0.0f;
		for(float value : entries[label]){
			sum += value;
		}

		float avg = sum / historySize;

		return avg;
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

	float getMedianOfMaxOver60Frames(string label){
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

struct StartStop{
	uint64_t t_start;
	uint64_t t_end;
};

struct Runtime{

	struct GuiItem{
		uint32_t type = 0;
		float min = 0.0;
		float max = 1.0;
		float oldValue = 0.5;
		float value = 0.5;
		string label = "";
	};

	inline static vector<int> keyStates = vector<int>(65536, 0);
	inline static int mods = 0;
	inline static vector<int> frame_keys = vector<int>();
	inline static vector<int> frame_actions = vector<int>();
	inline static vector<int> frame_mods = vector<int>();
	inline static OrbitControls* controls = new OrbitControls();
	inline static MouseEvents mouseEvents;
	inline static unordered_map<string, string> debugValues;
	inline static vector<std::pair<string, string>> debugValueList;

	inline static int totalTileFragmentCount;
	inline static double duration_sceneNodeUpdate;
	inline static double duration_draw;
	inline static string hovered_node_name = "";
	inline static string hovered_mesh_name = "";
	inline static uint64_t numVisibleNodes = 0;
	inline static uint64_t numVisibleTriangles = 0;
	inline static uint64_t numNodes = 0;
	inline static uint64_t numTriangles = 0;

	inline static glm::dvec2 mousePosition = {0.0, 0.0};
	inline static int mouseButtons = 0;

	inline static int64_t numRenderedTriangles = 0;
	
	struct Timing{
		string label;
		float milliseconds;
	};
	inline static bool measureTimings;
	inline static Timings timings;
	inline static vector<StartStop> profileTimings;

	Runtime(){
		
	}

	static int getKeyAction(int key){
		for(int i = 0; i < frame_keys.size(); i++){
			if(frame_keys[i] == key){
				return frame_actions[i];
			}
		}

		return -1;
	}

};