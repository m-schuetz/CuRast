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

#include "cuda.h"
#include "cuda_runtime.h"
#include "CudaModularProgram.h"

#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include "laszip/laszip_api.h"

#include "Runtime.h"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "json/json.hpp"
#include "CuRast.h"
#include "MappedFile.h"
#include "ThreadPool.h"
#include "types.h"
#include "scene/LasfileNode.h"
#include "scene/PotreeFileNode.h"




using namespace std; // YOLO

CUcontext context;

void initCuda() {
	cuInit(0);
	
	CUctxCreateParams creation_params = {};
	cuDeviceGet(&CURuntime::device, 0);
	cuCtxCreate(&context, &creation_params, 0, CURuntime::device);

	// None of our kernels use stack/local memory, but the default limit of 1kb per thread reserves 
	// stack for all threads that can be resident on the GPU (~192MB on an RTX 4090). 
	// The driver automatically increases the limit if a kernel requires more.
	cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 0);
}

void loadPointcloud(string file){
	
	CuRast* editor = CuRast::instance;
	Scene& scene = editor->scene;
	// string file = "F:/resources/pointclouds/test/ot_35120B4303A_1.laz";
	// string file = "G:/resources/pointclouds/tuwien_baugeschichte/candi Banyunibo/candi_banyunibo.las";
	
	string filename = fs::path(file).filename().string();
	
	double t_start = now();
	
	laszip_POINTER laszip_reader;
	if(laszip_create(&laszip_reader)){
		println("ERROR: creating laszip reader for '{}'", file);
		return;
	}
	laszip_BOOL is_compressed = 0;
	if(laszip_open_reader(laszip_reader, file.c_str(), &is_compressed)){
		println("ERROR: opening laszip reader for '{}'", file);
		laszip_destroy(laszip_reader);
		return;
	}
	laszip_header* header;
	if(laszip_get_header_pointer(laszip_reader, &header)){
		println("ERROR: getting laszip header pointer for '{}'", file);
		laszip_close_reader(laszip_reader);
		laszip_destroy(laszip_reader);
		return;
	}
	
	u64 numPoints = max(u64(header->number_of_point_records), header->extended_number_of_point_records);
	// vec3 bbmin = {header->min_x, header->min_y, header->min_z};
	// vec3 bbmax = {header->max_x, header->max_y, header->max_z};
	Box3 aabb = {
		{header->min_x, header->min_y, header->min_z},
		{header->max_x, header->max_y, header->max_z}
	};
	
	println("Loading pointcloud");
	println("numPoints: {:L}", numPoints);
	
	laszip_point* laz_point;
	if(laszip_get_point_pointer(laszip_reader, &laz_point)){
		laszip_close_reader(laszip_reader);
		laszip_destroy(laszip_reader);
		return;
	}
	
	double scale_x = header->x_scale_factor;
	double scale_y = header->y_scale_factor;
	double scale_z = header->z_scale_factor;
	double offset_x = header->x_offset;
	double offset_y = header->y_offset;
	double offset_z = header->z_offset;
	dvec3 origin = {0.0, 0.0, 0.0};
	
	{
		laszip_read_point(laszip_reader);
		double x = (double)(laz_point->X * scale_x + offset_x);
		double y = (double)(laz_point->Y * scale_y + offset_y);
		double z = (double)(laz_point->Z * scale_z + offset_z);
		
		origin = {x, y, z};
	}
	
	laszip_close_reader(laszip_reader);
	
	constexpr i64 BATCH_SIZE = 1'000'000;
	i64 numBatches = (numPoints + BATCH_SIZE - 1) / BATCH_SIZE;
	
	static ThreadPool pool(16);
	
	shared_ptr<SNCPoints> points = make_shared<SNCPoints>(filename);
	points->cptr_positions = MemoryManager::alloc(numPoints * sizeof(vec3), "pos");
	points->cptr_colors = MemoryManager::alloc(numPoints * sizeof(u32), "col");
	points->numPoints = numPoints;
	points->transform = glm::translate(origin);
	points->aabb.min = aabb.min - vec3(origin);
	points->aabb.max = aabb.max - vec3(origin);
	
	for(u64 startIndex = 0; startIndex < numPoints; startIndex += BATCH_SIZE){
		u64 endIndex = min(startIndex + BATCH_SIZE, numPoints);
		u64 pointsInBatch = endIndex - startIndex;
		
		pool.enqueue([=](int threadIndex){

			cudaSetDevice(CURuntime::device);

			laszip_POINTER laszip_reader;
			if(laszip_create(&laszip_reader)){
				println("ERROR: creating laszip reader for '{}'", file);
				return;
			}
			laszip_BOOL is_compressed = 0;
			if(laszip_open_reader(laszip_reader, file.c_str(), &is_compressed)){
				println("ERROR: opening laszip reader for '{}'", file);
				laszip_destroy(laszip_reader);
				return;
			}
			laszip_header* header;
			if(laszip_get_header_pointer(laszip_reader, &header)){
				println("ERROR: getting laszip header pointer for '{}'", file);
				laszip_close_reader(laszip_reader);
				laszip_destroy(laszip_reader);
				return;
			}
			
			laszip_point* laz_point;
			if(laszip_get_point_pointer(laszip_reader, &laz_point)){
				laszip_close_reader(laszip_reader);
				laszip_destroy(laszip_reader);
				return;
			}
			if(laszip_seek_point(laszip_reader, startIndex)){
				laszip_close_reader(laszip_reader);
				laszip_destroy(laszip_reader);
				return;
			}
			
			vector<vec3> positions(pointsInBatch);
			vector<u32> colors(pointsInBatch);
			
			for(u64 i = 0; i < pointsInBatch; i++){
				
				if(laszip_read_point(laszip_reader)){
					println("error reading point");
					__debugbreak();
					exit(5124234);
				}
				
				double x = (double)(laz_point->X * scale_x + offset_x);
				double y = (double)(laz_point->Y * scale_y + offset_y);
				double z = (double)(laz_point->Z * scale_z + offset_z);
				
				positions[i] = dvec3{x, y, z} - origin;
				
				u32 r = laz_point->rgb[0] <= 255 ? laz_point->rgb[0] : laz_point->rgb[0] / 256;
				u32 g = laz_point->rgb[1] <= 255 ? laz_point->rgb[1] : laz_point->rgb[1] / 256;
				u32 b = laz_point->rgb[2] <= 255 ? laz_point->rgb[2] : laz_point->rgb[2] / 256;
				u32 color = r | (g << 8) | (b << 16);
				
				colors[i] = color;
			}
			
			laszip_close_reader(laszip_reader);
			
			cuMemcpyHtoD(points->cptr_positions + startIndex * sizeof(vec3), positions.data(), byteSizeOf(positions));
			cuMemcpyHtoD(points->cptr_colors + startIndex * sizeof(u32), colors.data(), byteSizeOf(colors));
			
		});
	}
	
	// pool.wait();
	pool.onEmpty([t_start](){
		double duration = now() - t_start;
		println("loadPoints duration: {:.1f} seconds", duration);
	});
	

	// // position: -54.13072310328936, 16.43438931570019, -20.725554240749204 
	// // vec3 target = (min + max) / 2.0f;
	// // target.z = min.z;
	// vec3 target = {0.0f, 0.0f, 0.0f};
	// Runtime::controls->yaw    = -8.239;
	// Runtime::controls->pitch  = -0.609;
	// Runtime::controls->radius = glm::length(bbmax - bbmin) / 2.0f;
	// Runtime::controls->target = target;
	
	// Box3 aabb = glb->glbNode->aabb;
	vec3 extent = aabb.max - aabb.min;
	vec3 center = (aabb.min + aabb.max) * 0.5f;

	Runtime::controls->yaw    = -7.204;
	Runtime::controls->pitch  = -0.579;
	Runtime::controls->radius = length(extent);
	Runtime::controls->target = { center.x, center.y, center.z};
	
	editor->scene.world->children.push_back(points);
	
}

void loadPointclouds(vector<string> files){
	
	if(files.size() == 0) return;
	
	for(string file : files){
		loadPointcloud(file);
	}
	
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

	VKRenderer::onFileDrop([](vector<string> files){
		
		CuRast* editor = CuRast::instance;
		Scene& scene = editor->scene;

		vector<string> lazfiles;
		for(string file : files){
			if(iEndsWith(file, ".laz") || iEndsWith(file, ".las")){
				// loadPointcloud(file);
				lazfiles.push_back(file);
			}

			if(fs::is_directory(file)){
				for(const fs::directory_entry& entry : fs::recursive_directory_iterator(file)){
					if(!entry.is_regular_file()) continue;

					string path = entry.path().string();
					if(iEndsWith(path, ".laz") || iEndsWith(path, ".las")){
						lazfiles.push_back(path);
					}
				}
			}
		}
		
		loadPointclouds(lazfiles);

	});

	initScene();

	VKRenderer::loop(
		[&]() {CuRast::instance->update();},
		[&]() {CuRast::instance->render();},
		[&]() {CuRast::instance->postFrame();}
	);

	VKRenderer::destroy();
}
