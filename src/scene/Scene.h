#pragma once

#include <functional>

#include "SceneNode.h"

using namespace std;

struct Scene {

	shared_ptr<SceneNode> root = nullptr;
	shared_ptr<SceneNode> world = nullptr;

	Scene() {
		root = make_shared<SceneNode>("root");
		world = make_shared<SceneNode>("world");

		root->children.push_back(world);
	}

	template<typename T>
	void forEach(const function<void(T*)>& callback){

		root->traverse([&](SceneNode* node){
			if(dynamic_cast<T*>(node) != nullptr){

				T* sn = dynamic_cast<T*>(node);
				callback(sn);
			}
		});
	}

	void updateTransformations(){

		function<void(SceneNode*, SceneNode*)> traverse;
		traverse = [&](SceneNode* parent, SceneNode* node){
			if (parent) {
				mat4 transform = parent->transform_global * node->transform;
				node->transform_global = transform;
			} else {
				node->transform_global = node->transform;
			}

			for(auto child : node->children){
				traverse(node, child.get());
			}

		};
		traverse(nullptr, root.get());
	}

};