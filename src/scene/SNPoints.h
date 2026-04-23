#pragma once

#include <string>
#include <vector>

#include "SceneNode.h"
#include "./kernels/HostDeviceInterface.h"

using std::string;
using std::vector;
using glm::ivec2;

struct SNPoints : public SceneNode{

	
	Pointcloud* pointcloud = nullptr;
	
	SNPoints(string name) : SceneNode(name){
		
	}

	uint64_t getGpuMemoryUsage(){
		return 0;
	}

};