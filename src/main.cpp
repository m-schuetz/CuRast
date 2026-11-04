#include <cstdio>
#include <format>
#include "compat_print.h"
#include <filesystem>
#include <string>
#include <queue>
#include <vector>
#include <algorithm>
#include <execution>
#include <thread>

#include "unsuck.hpp"

#include "cuda_to_hip.h"

// Runtime compilation uses different classes on each backend (hiprtc vs
// nvrtc/nvJitLink), so the module program type is selected here.
#if defined(USE_HIP)
#include "HipModularProgram.h"
using CudaModularProgram = HipModularProgram;
using CudaModule = HipModule;
#else
#include "CudaModularProgram.h"
#endif

#include "CudaVulkanSharedMemory.h"
#include "VulkanCudaSharedMemory.h"
#include "jpeg/JPEGIndexer.h"

#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include "laszip/laszip_api.h"

#include "Runtime.h"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "json/json.hpp"
#include "CuRast.h"
#include "MappedFile.h"
#include "GLTFLoader.h"
#include "LargeGlbLoader.h"
#include "PlyLoader.h"
#include "types.h"


using namespace std; // YOLO

CUcontext context;

dmat4 flip = dmat4(
	1.000,  0.000, 0.000, 0.000,
	0.000,  0.000, 1.000, 0.000,
	0.000, -1.000, 0.000, 0.000,
	0.000,  0.000, 0.000, 1.000);

void initCuda() {
#if defined(USE_HIP)
	hipInit(0);
	hipDeviceGet(&CURuntime::device, 0);
	// Use the primary context: the deprecated explicit hipCtxCreate context
	// breaks regular hipModuleLaunchKernel dispatches on ROCm 7.2 (cooperative
	// launches work; regular ones memory-fault).
	hipSetDevice(CURuntime::device);
	context = nullptr;
#else
	cuInit(0);
	CUctxCreateParams creation_params = {};
	cuDeviceGet(&CURuntime::device, 0);
	cuCtxCreate(&context, &creation_params, 0, CURuntime::device);
#endif
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
			
			cuMemcpyHtoD(HIP_DEVPTR_ADD(points->cptr_positions, startIndex * sizeof(vec3)), positions.data(), byteSizeOf(positions));
			cuMemcpyHtoD(HIP_DEVPTR_ADD(points->cptr_colors, startIndex * sizeof(u32)), colors.data(), byteSizeOf(colors));
			
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

	auto loadSponza = [=](){ 
		string file = "F:/resources/meshes/sponza-png_by_Ludicon.glb";

		static auto glb = largeGlb::load(file, context, {.compress = false});
		glb->glbNode->name = "Sponza";
		glb->glbNode->transform = flip * glb->glbNode->transform;
		scene.world->children.push_back(glb->glbNode);

		Runtime::controls->yaw    = -4.731;
		Runtime::controls->pitch  = 0.009;
		Runtime::controls->radius = 336.359;
		Runtime::controls->target = { -2.986, 32.881, 119.491, };
		
		// position: 85.56503842133353, -126.16710469214773, 27.111462557161722 
		// Runtime::controls->yaw    = -8.666;
		// Runtime::controls->pitch  = -0.660;
		// Runtime::controls->radius = 45.452;
		// Runtime::controls->target = { 116.838, -152.214, 47.347};
		CuRastSettings::enableEDL = false;


	};

	auto loadSponzaJPEG = [=](){ 
		std::string file = "./resources/meshes/Sponza_70.glb";

		static auto glb = largeGlb::load(file, context, {
			.skipUVs = false, 
			.compress = true,
			.useJpegTextures = true,
		});
		glb->glbNode->transform = flip * glb->glbNode->transform;
		scene.world->children.push_back(glb->glbNode);

		// position: 1.3305474790692626, 0.45402304811990946, 1.1192273142715552 
		Runtime::controls->yaw    = -4.646;
		Runtime::controls->pitch  = 0.024;
		Runtime::controls->radius = 8.993;
		Runtime::controls->target = { -7.642, -0.142, 1.105, };
	};

	auto loadCubeJpeg = [=](){ 
		std::string file = "./resources/meshes/Cube_70.glb";

		static auto glb = largeGlb::load(file, context, {
			.skipUVs = false, 
			.compress = true,
			.useJpegTextures = true,
		});
		glb->glbNode->transform = flip * glb->glbNode->transform;
		editor->scene.world->children.push_back(glb->glbNode);

		// position: 25.15382712255294, -20.17937489109018, 12.02894404906026 
		Runtime::controls->yaw    = -5.466;
		Runtime::controls->pitch  = -0.531;
		Runtime::controls->radius = 34.148;
		Runtime::controls->target = { 0.252, -0.030, 0.196};
	};

	auto loadHakone = [=](){
		// string file = "./resources/meshes/donaukanal_urania_1M_jpeg80.glb";
		// string file = "F:/resources/meshes/hakone_lantern.glb";
		string file = "F:/resources/meshes/hakone_1M.glb";

		static auto glb = largeGlb::load(file, context, {.skipUVs = false, .compress = false});
		editor->scene.world->children.push_back(glb->glbNode);

		// Overview
		Runtime::controls->yaw    = -7.070;
		Runtime::controls->pitch  = -0.515;
		Runtime::controls->radius = 37.564;
		Runtime::controls->target = { 25.607, -17.328, 8.340, };
	};

	auto loadHakoneInstances = [=](){
		// string file = "./resources/meshes/donaukanal_urania_1M_jpeg80.glb";
		// string file = "F:/resources/meshes/hakone_lantern.glb";
		// string file = "F:/resources/meshes/hakone_lantern_optimized.glb";
		// string file = "F:/resources/meshes/hakone_lantern_3.glb";
		// string file = "F:/resources/meshes/hakone_1m.glb";
		string file = "F:/resources/meshes/hakone_1m_optimized.glb";

		static auto glb = largeGlb::load(file, context, {.skipUVs = false, .compress = false});

		shared_ptr<SNTriangles> original = dynamic_pointer_cast<SNTriangles>(glb->glbNode->children[0]);

		for(int ix = 0; ix < 50; ix++)
		for(int iy = 0; iy < 60; iy++)
		{
			shared_ptr<SNTriangles> instance = make_shared<SNTriangles>("instance");
			instance->mesh = original->mesh;
			instance->texture = original->texture;
			instance->aabb = original->aabb;
			instance->transform = glm::translate(vec3{ix * 30.0f, iy * 30.0f, 0.0f});
			editor->scene.world->children.push_back(instance);
		}

		// position: -0.9794631786208647, -40.41708964196321, 21.421843904392993 
		Runtime::controls->yaw    = -7.070;
		Runtime::controls->pitch  = -0.515;
		Runtime::controls->radius = 37.564;
		Runtime::controls->target = { 25.607, -17.328, 8.340, };
	};

	auto loadSpot = [=](){ 
		std::string file = "F:/resources/meshes/spot.glb";

		static auto glb = largeGlb::load(file, context, {.skipUVs = false, .compress = false});
		editor->scene.world->children.push_back(glb->glbNode);

		// position: 1.1620903300458982, 1.4676847017158816, -0.27796041598897114 
		Runtime::controls->yaw    = -16.358;
		Runtime::controls->pitch  = -0.381;
		Runtime::controls->radius = 1.957;
		Runtime::controls->target = { -0.023, 0.022, 0.301, };
	};

	auto loadZorah = [=](){ 
		string file = "F:/resources/meshes/zorah_main_public.gltf/zorah_main_public.gltf";
		// string file = "F:/resources/meshes/zorah_main_public.gltf_optimized/zorah_main_public.gltf";
		
		// Mesh has no textures/uvs/normals
		CuRastSettings::displayAttribute = DisplayAttribute::NONE;

		static auto glb = largeGlb::load(file, context, {.skipUVs = true, .compress = true});
		glb->glbNode->transform = flip;
		editor->scene.world->children.push_back(glb->glbNode);

		// Let's remove some less appealing billboards
		vector<shared_ptr<SceneNode>> filtered;
		for(shared_ptr<SceneNode> node : glb->glbNode->children){
			if(node->name == "FogCard") continue;
			if(node->name == "Plane") continue;
			
			filtered.push_back(node);
		}
		glb->glbNode->children = filtered;

		// Overview
		Runtime::controls->yaw    = -16.537;
		Runtime::controls->pitch  = -0.472;
		Runtime::controls->radius = 97.539;
		Runtime::controls->target = { 17.436, -6.343, 3.689, };
		
		// Closeup
		Runtime::controls->yaw    = -17.344;
		Runtime::controls->pitch  = 0.073;
		Runtime::controls->radius = 10.893;
		Runtime::controls->target = { 44.294, 1.156, 6.458, };

	};

	auto createCube = [&](){
		static shared_ptr<SNTriangles> node = make_shared<SNTriangles>("node");
		node->texture = new Texture();
		node->mesh = new Mesh();

		{ // Create Default Texture
			int64_t textureWidth = 128;
			int64_t textureHeight = 128;
			vector<uint8_t> textureData = vector<uint8_t>(2 * 4 * textureWidth * textureHeight, 255);
			
			node->texture->width = textureWidth;
			node->texture->height = textureHeight;
			node->texture->data = (uint32_t*)MemoryManager::alloc(byteSizeOf(textureData), "default texture");

			cuMemcpyHtoDAsync(CUdeviceptr(node->texture->data), textureData.data(), byteSizeOf(textureData), 0);
		}

		node->mesh->isLoaded = true;
		node->mesh->name = "default mesh";
		node->mesh->numTriangles = 0;

		vector<vec3> positions = {
			vec3{0.0f, 0.0f, 0.0f},
			vec3{1.0f, 0.0f, 0.0f},
			vec3{1.0f, 1.0f, 0.0f},
			vec3{0.0f, 1.0f, 0.0f},
		};
		vector<vec2> uvs = {
			vec2{0.0f, 0.0f},
			vec2{1.0f, 0.0f},
			vec2{1.0f, 1.0f},
			vec2{0.0f, 1.0f},
		};
		vector<uint32_t> indices = {0, 1, 2, 0, 2, 3};

		int numVertices = positions.size();
		int numTriangles = indices.size() / 3;
		node->mesh->cptr_position = MemoryManager::alloc(sizeof(vec3) * numVertices, "position");
		node->mesh->cptr_uv       = MemoryManager::alloc(sizeof(vec2) * numVertices, "uv");
		node->mesh->cptr_indices  = MemoryManager::alloc(sizeof(uint32_t) * 3 * numTriangles, "indices");
		node->aabb.extend(vec3{-1.0f, -1.0f, -1.0f});
		node->aabb.extend(vec3{1.0f, 1.0f, 1.0f});

		cuMemcpyHtoD(node->mesh->cptr_position, positions.data(), byteSizeOf(positions));
		cuMemcpyHtoD(node->mesh->cptr_uv, uvs.data(), byteSizeOf(uvs));
		cuMemcpyHtoD(node->mesh->cptr_indices, indices.data(), byteSizeOf(indices));

		node->mesh->numTriangles = indices.size() / 3;
		node->mesh->numVertices = positions.size();

		scene.world->children.push_back(node);

		CuRastSettings::displayAttribute = DisplayAttribute::TRIANGLE_ID;

		// position: 0.8369693760957783, 0.05588397571280396, 0.02743282811653472 
		Runtime::controls->yaw    = -17.426;
		Runtime::controls->pitch  = -0.272;
		Runtime::controls->radius = 0.818;
		Runtime::controls->target = { 0.028, 0.172, -0.005, };
	};

	auto loadVenice = [=](){ 
		string file = "F:/resources/meshes/iconem/VeniceGeneral-Airborne-flyover-400M-12x16k-local-binply/venice.gltf";
		// string file = "F:/resources/meshes/iconem/VeniceGeneral-Airborne-flyover-400M-12x16k-local-binply/venice_optimized.gltf";
		
		CuRastSettings::displayAttribute = DisplayAttribute::TEXTURE;

		static auto glb = largeGlb::load(file, context, {
			.skipNormals = true, // We need that previous VRAM
			.skipVertexColors = true,
			.compress = true,
			.useJpegTextures = false,
			.imageDivisionFactor = 2
		});
		editor->scene.world->children.push_back(glb->glbNode);
		
		// Distant
		Runtime::controls->yaw    = -18.968;
		Runtime::controls->pitch  = -0.769;
		Runtime::controls->radius = 4180.978;
		Runtime::controls->target = { 352.960, -1134.931, -462.529, };
	};

	// createCube();
	// loadZorah();
	// loadGraffiti();
	// loadHakone();
	// loadHakoneInstances();
	// loadXyzDragon();
	// loadSpot();
	// loadWietrznia();
	// loadSponza();
	// loadSponzaJPEG();
	// loadCubeJpeg();
	// loadPolygraphenewerkLeibzigInstances();
	// loadVenice();
	// loadPointcloud("F:/resources/pointclouds/CA13/ot_35120B4116C_1.laz");

}

void update(){

	if(Benchmarking::request_scenario){

		auto scenario = Benchmarking::request_scenario;

		string path = stringReplace(scenario->path, "DATASETPATH", Benchmarking::datasetPath);

		CuRastSettings::displayAttribute = scenario->attribute;

		static auto glb = largeGlb::load(path, context, {
			.skipUVs = scenario->skipUVs,
			.skipNormals = scenario->skipNormals,
			.compress = scenario->compress,
			.useJpegTextures = scenario->useJpegTextures,
			.imageDivisionFactor = scenario->imageDivisionFactor,
		});
		glb->glbNode->name = scenario->label;
		glb->glbNode->transform = scenario->transform * glb->glbNode->transform;

		vector<shared_ptr<SceneNode>> filtered;
		for(shared_ptr<SceneNode> node : glb->glbNode->children){

			bool accept = scenario->filter(node);
			
			if(accept){
				filtered.push_back(node);
			}
		}
		glb->glbNode->children = filtered;

		shared_ptr<SceneNode> original = glb->glbNode;

		function<shared_ptr<SceneNode>(shared_ptr<SceneNode>)> deepClone =
		[&deepClone](shared_ptr<SceneNode> node) -> shared_ptr<SceneNode> {

			shared_ptr<SceneNode> clone;

			shared_ptr<SNTriangles> tris = dynamic_pointer_cast<SNTriangles>(node);
			if(tris){
				auto triClone      = make_shared<SNTriangles>(tris->name);
				triClone->mesh     = tris->mesh;
				triClone->texture  = tris->texture;
				triClone->aabb     = tris->aabb;
				clone = triClone;
			}else{
				clone = make_shared<SceneNode>(node->name);
				clone->aabb = node->aabb;
			}

			clone->transform = node->transform;
			clone->visible   = node->visible;

			for(auto& child : node->children){
				clone->children.push_back(deepClone(child));
			}

			return clone;
		};

		for(int ix = 0; ix < scenario->instances_count.x; ix++)
		for(int iy = 0; iy < scenario->instances_count.y; iy++)
		{
			shared_ptr<SceneNode> clone = deepClone(original);
			clone->transform = glm::translate(dvec3{ix * scenario->instances_spacing.x, iy * scenario->instances_spacing.y, 0.0f}) * clone->transform;
			CuRast::instance->scene.world->children.push_back(clone);
		}

		Runtime::controls->yaw    = scenario->view_overview.yaw;
		Runtime::controls->pitch  = scenario->view_overview.pitch;
		Runtime::controls->radius = scenario->view_overview.radius;
		Runtime::controls->target = scenario->view_overview.target;

		Benchmarking::active_scenario = Benchmarking::request_scenario;
		Benchmarking::request_scenario = nullptr;
	}

}

int main(int argc, char** argv){

	Benchmarking::datasetPath = "./";

	for(int i = 1; i < argc - 1; i++){
		if(string(argv[i]) == "-b"){
			Benchmarking::datasetPath = argv[i + 1];
		}
	}

	std::locale::global(getSaneLocale());
	
	// Headless rasterization benchmark: --bench <file.glb> [width height frames]
	string benchFile = "";
	int benchW = 1920, benchH = 1080, benchFrames = 60;
	for(int i = 1; i < argc; i++){
		if(string(argv[i]) == "--bench" && i + 1 < argc){
			benchFile = argv[i + 1];
			if(i + 4 < argc){ benchW = atoi(argv[i+2]); benchH = atoi(argv[i+3]); benchFrames = atoi(argv[i+4]); }
		}
	}
	if(benchFile != ""){
		initCuda();
		CuRast::setup();
		VKRenderer::camera = make_shared<Camera>();
		VKRenderer::camera->setSize(benchW, benchH);
	
		CuRast* editor = CuRast::instance;
		Scene& scene = editor->scene;
		println("[bench] loading {}", benchFile);
		auto glb = largeGlb::load(benchFile, context, {.skipUVs = false, .compress = false});
		scene.world->children.push_back(glb->glbNode);
		scene.updateTransformations();
	
		Box3 aabb = glb->glbNode->aabb;
		vec3 extent = aabb.max - aabb.min;
		vec3 center = (aabb.min + aabb.max) * 0.5f;
		Runtime::controls->yaw    = -7.204;
		Runtime::controls->pitch  = -0.579;
		Runtime::controls->radius = length(extent);
		Runtime::controls->target = { center.x, center.y, center.z };
		println("[bench] {}x{} {} frames, model extent {:.1f}", benchW, benchH, benchFrames, length(extent));
	
		Runtime::controls->update();
		VKRenderer::camera->world = Runtime::controls->world;
		VKRenderer::camera->update();

		double best = 1e30;
		for(int f = 0; f < benchFrames; f++){
			editor->renderHeadless(benchW, benchH, f == benchFrames - 1);
			double ms = editor->lastFrameMs;
			if(f > 0 && ms > 0.0 && ms < best) best = ms;
			println("[bench] frame {:3}: visbuffer pipeline = {:7.3f} ms  ({} of {} tris visible, {} fragments)",
				f, ms, (uint64_t)Runtime::numVisibleTriangles, (uint64_t)Runtime::numTriangles, (uint64_t)editor->deviceState->dbg_fragcount);
		}
		println("[bench] best visbuffer-pipeline time: {:.3f} ms @ {}x{}", best, benchW, benchH);
		println("[bench] wrote ./bench_render.png");
		return 0;
	}

	initCuda();
	VKRenderer::init();
	CuRast::setup();

	VKRenderer::onFileDrop([](vector<string> files){
		
		CuRast* editor = CuRast::instance;
		Scene& scene = editor->scene;

		if(files.size() == 1 && iEndsWith(files[0], ".gltf") || iEndsWith(files[0], ".glb")){
			string file = files[0];
			static vector<shared_ptr<largeGlb::LoadedGlb>> loadedGlbs;
			Scene& scene = editor->scene;

			auto glb = largeGlb::load(file, context, {.skipUVs = false, .compress = false});
			scene.world->children.push_back(glb->glbNode);
			loadedGlbs.push_back(glb);

			scene.updateTransformations();

			Box3 aabb = glb->glbNode->aabb;
			vec3 extent = aabb.max - aabb.min;
			vec3 center = (aabb.min + aabb.max) * 0.5f;

			Runtime::controls->yaw    = -7.204;
			Runtime::controls->pitch  = -0.579;
			Runtime::controls->radius = length(extent);
			Runtime::controls->target = { center.x, center.y, center.z};
		}
		
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
		[&]() {
			update();
			CuRast::instance->update();
			
			DeviceState* state = CuRast::instance->deviceState;
			double stage1_millies = double(state->nanotime_stage_1 - state->nanotime_start) / 1'000'000.0;
			double stage2_millies = double(state->nanotime_stage_2 - state->nanotime_stage_1) / 1'000'000.0;
			double stage3_millies = double(state->nanotime_stage_3 - state->nanotime_stage_2) / 1'000'000.0;
			Runtime::debugValues["stage 1"] = format("{:.3f}", stage1_millies);
			Runtime::debugValues["stage 2"] = format("{:.3f}", stage2_millies);
			Runtime::debugValues["stage 3"] = format("{:.3f}", stage3_millies);
		},
		[&]() {CuRast::instance->render();},
		[&]() {CuRast::instance->postFrame();}
	);

	VKRenderer::destroy();
}
