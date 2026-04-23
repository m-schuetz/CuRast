#pragma once

#include <string>
#include <vector>

#include "SceneNode.h"
#include "Mesh.h"
#include "Pointcloud.h"
#include "./kernels/HostDeviceInterface.h"
#include "CuRastSettings.h"
#include "morton.h"

using std::string;
using std::vector;
using glm::ivec2;


struct SNTriangles : public SceneNode{
		
	Texture* texture = nullptr;
	Mesh* mesh = nullptr;
	shared_ptr<VKTexture> vkTexture;
	uint32_t vkTextureHandle;

	SNTriangles(string name) : SceneNode(name){
		
	}

	virtual void update(View view){

		// for(auto& child : children){
		// 	child->update(views);
		// }

		// mat4 view = view.view;
		mat4 proj = view.proj;
		mat4 viewProj = proj * mat4(view.view);
		int width = view.framebuffer->width;
		int height = view.framebuffer->height;
		float aspect = float(width) / float(height);

		struct Plane {
			vec3 normal;
			float d;
		};

		auto normalizedPlane = [](const glm::vec4& p){
			float len = glm::length(glm::vec3(p));
			return Plane{ glm::vec3(p) / len, p.w / len };
		};

		vec4 row0 = glm::row(viewProj, 0);
		vec4 row1 = glm::row(viewProj, 1);
		vec4 row2 = glm::row(viewProj, 2);
		vec4 row3 = glm::row(viewProj, 3);

		Plane planes[4];
		planes[0] = normalizedPlane(row3 + row0); // Left
		planes[1] = normalizedPlane(row3 - row0); // Right
		planes[2] = normalizedPlane(row3 + row1); // Bottom
		planes[3] = normalizedPlane(row3 - row1); // Top

		auto positiveVertex = [](vec3 bmin, vec3 bmax, vec3 n){
			return vec3(
				(n.x >= 0.0f) ? bmax.x : bmin.x,
				(n.y >= 0.0f) ? bmax.y : bmin.y,
				(n.z >= 0.0f) ? bmax.z : bmin.z
			);
		};

		auto aabbOutsidePlane = [&](Plane p, Box3 aabb){
			glm::vec3 v = positiveVertex(aabb.min, aabb.max, p.normal);
			return glm::dot(p.normal, v) + p.d < 0.0f;
		};

		auto aabbOutsideFrustum = [&](Plane planes[4], Box3 aabb){
			for(int i = 0; i < 4; i++){
				Plane plane = planes[i];

				if(aabbOutsidePlane(plane, aabb)) return true;
			}

			return false;
		};

		mat4 world = this->transform_global;
		mat4 worldView = mat4(view.view) * world;

		Box3 aabb_world = this->aabb.transform(world);
		Box3 aabb_view = this->aabb.transform(worldView);

		bool isInsideFrustum = true;
		if(CuRastSettings::enableFrustumCulling){
			isInsideFrustum = !aabbOutsideFrustum(planes, aabb_world);
		}

		this->visible = isInsideFrustum;

		vec3 center_view = (aabb_view.min + aabb_view.max) / 2.0f;
		float size_view = length(aabb_view.max - aabb_view.min);
		float depth = -center_view.z;
		bool isSmall = size_view < depth * 0.2f * CuRastSettings::threshold;

		if(isSmall) this->visible = false;
	}

};