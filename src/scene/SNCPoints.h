#pragma once

#include <string>
#include <vector>

#include "SceneNode.h"
#include "./kernels/HostDeviceInterface.h"

using std::string;
using std::vector;
using glm::ivec2;

struct SNCPoints : public SceneNode{

	CUdeviceptr cptr_positions;
	CUdeviceptr cptr_colors;
	uint64_t numPoints = 0;
	
	SNCPoints(string name) : SceneNode(name){
		
	}

	uint64_t getGpuMemoryUsage(){
		return 0;
	}

};