#include <cstdio>
#include <format>
#include <print>
#include <filesystem>
#include <string>
#include <queue>
#include <vector>
#include <algorithm>
#include <execution>
#include <thread>

#include "unsuck.hpp"

#include <cuda_runtime.h>

#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>


#include "CuRast.h"
#include "types.h"
#include "scene/LasfileNode.h"
#include "scene/PotreeFileNode.h"




using namespace std; // YOLO

void initCuda() {
	CURuntime::device = 0;
	CURuntime::assertCudaSuccess(cudaSetDevice(CURuntime::device));
	CURuntime::assertCudaSuccess(cudaFree(nullptr)); // creates the context

	// None of our kernels use stack/local memory, but the default limit of 1kb per thread reserves 
	// stack for all threads that can be resident on the GPU (~192MB on an RTX 4090). 
	// The driver automatically increases the limit if a kernel requires more.
	cudaDeviceSetLimit(cudaLimitStackSize, 0);
}

void initScene() {
	CuRast* editor = CuRast::instance;
	Scene& scene = editor->scene;

	// position: 124.54672426747658, -42.72048538939598, -12.2730454323992 
	Runtime::controls->yaw    = -5.179;
	Runtime::controls->pitch  = 0.108;
	Runtime::controls->radius = 142.656;
	Runtime::controls->target = { -2.859, 21.085, -5.387, };

	// string file = "/home/mschuetz/dev/resources/morro_bay_73M.las";
	// shared_ptr<LasfileNode> node = make_shared<LasfileNode>(file, "pointcloud");
	// scene.root->children.push_back(node);

	// vec3 center = (node->min + node->max) * 0.5f - node->offset;
	// Runtime::controls->target = center;
	// Runtime::controls->radius = 0.8f * length(node->max - node->min);
	// Runtime::controls->pitch  = -0.9;


	// string file = "/home/mschuetz/dev/resources/morro_bay_73M.laz_converted";
	// string file = "/run/media/mschuetz/Lightning/resources/pointclouds/iconem/Meroe_NorthNecropolis_684M.las_converted";
	string file = "/run/media/mschuetz/Lightning/resources/pointclouds/CA13_converted";
	shared_ptr<PotreeFileNode> node = make_shared<PotreeFileNode>(file, "potree");
	scene.root->children.push_back(node);

	PotreeAttribute* position = node->findAttribute("position");
	dvec3 tightMin = {position->min[0], position->min[1], position->min[2]};
	dvec3 tightMax = {position->max[0], position->max[1], position->max[2]};
	dvec3 origin   = (node->min + node->max) * 0.5;

	Runtime::controls->target = (tightMin + tightMax) * 0.5 - origin;
	Runtime::controls->radius = 0.8 * length(tightMax - tightMin);
	Runtime::controls->pitch  = -0.9;



	
}

int main(int argc, char** argv){

	std::locale::global(getSaneLocale());

	initCuda();
	VKRenderer::init();
	CuRast::setup();

	initScene();

	VKRenderer::loop(
		[&]() {CuRast::instance->update();},
		[&]() {CuRast::instance->render();},
		[&]() {CuRast::instance->postFrame();}
	);

	VKRenderer::destroy();
}
