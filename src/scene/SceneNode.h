#pragma once

#include <string>
#include <format>
#include "cuda.h"
#include "cuda_runtime.h"

#include <glm/glm.hpp>
#include <glm/common.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/matrix.hpp>
#include <glm/gtx/transform.hpp>

#include "VKRenderer.h"
#include "./kernels/HostDeviceInterface.h"

using namespace std;
using glm::ivec2;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat3;
using glm::mat4;
using glm::quat;


struct SceneNode{
	
	string name;
	uint32_t id;

	// local transform of each matrix
	mat4 transform = mat4(1.0f);

	// global transform with transformations of parents applied to it.
	// updatated via scene.updateTransformations()
	mat4 transform_global = mat4(1.0f);
	bool visible = true;
	Box3 aabb;
	vector<shared_ptr<SceneNode>> children;
	static inline uint32_t counter = 0;

	SceneNode(){

		this->name = format("node_{}", counter);
		this->id = counter;

		SceneNode::counter++;

	}
	
	SceneNode(string name){
		this->name = name;
		this->id = counter;
		SceneNode::counter++;
	}

	void traverse(function<void(SceneNode*)> callback){
		callback(this);

		for(auto child : children){
			child->traverse(callback);
		}
	}

	virtual string toString(){
		return "SceneNode";
	}

	virtual uint64_t getGpuMemoryUsage(){
		return 0;
	}

	virtual Box3 getBoundingBox(){
		Box3 box;
		return box;
	}

	SceneNode* find(string name){

		SceneNode* found = nullptr;

		this->traverse([&](SceneNode* node){
			if(node->name == name){
				found = node;
			}
		});

		return found;
	}

	void remove(SceneNode* toRemove){
		for(int i = 0; i < children.size(); i++){
			if(children[i].get() == toRemove) {
				children.erase(children.begin() + i);
				return;
			}
		}
	}

	virtual void update(View view){

		// for(auto& child : children){
		// 	child->update(views);
		// }

	}

};