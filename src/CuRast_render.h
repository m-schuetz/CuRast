
#include <unordered_set>
#include <execution>
#include <queue>

#include "jpeg/JpegTextures.h"

#include "Timer.h"
#include "VKRenderer.h"
#include "TextureManager.h"
#include "types.h"

using namespace std;

CudaVirtualMemory* cvm_framebuffer = nullptr;
CudaVirtualMemory* cvm_colorbuffer = nullptr;
bool initialized = false;
JpegTextures* jpegTextures = nullptr;

// Cuda-Vulkan interop
struct MappedTextures{
	vector<shared_ptr<VKTexture>> textures;
	vector<CUsurfObject> surfaces;
};

static unordered_map<int64_t, int64_t> lastImportedVersion;

// implemented in lines.cu 
void launch_drawBoundingBoxes(
	RenderTarget target,
	CMesh* meshes,
	u32 numMeshes,
	u32* numProcessedBatches
);

MappedTextures mapCudaVk(vector<shared_ptr<VKTexture>> textures){
	MappedTextures mappings;
	for(auto& tex : textures){
		if(tex->cudaSurface == 0 || lastImportedVersion[tex->ID] != tex->version){
			tex->importToCuda();
			lastImportedVersion[tex->ID] = tex->version;
		}
		mappings.textures.push_back(tex);
		mappings.surfaces.push_back(tex->cudaSurface);
	}
	return mappings;
}

void unmapCudaVk(MappedTextures& mappings){
	// Ensure CUDA writes are complete before Vulkan blits the image
	cuStreamSynchronize((CUstream)CU_STREAM_DEFAULT);
}

void saveScreenshot(RenderTarget target, View view, CUdeviceptr cptr_ssaoShadebuffer, CudaModularProgram* prog_resolve){

	u64 numPixels = target.width * target.height;
	CUdeviceptr cptr_screenshot = MemoryManager::alloc(numPixels * 4, "screenshot");

	u32 backgroundColor = 0;
	uint8_t* bgRgba = (uint8_t*)&backgroundColor;
	bgRgba[0] = clamp(CuRastSettings::background.x * 256.0f, 0.0f, 255.0f);
	bgRgba[1] = clamp(CuRastSettings::background.y * 256.0f, 0.0f, 255.0f);
	bgRgba[2] = clamp(CuRastSettings::background.z * 256.0f, 0.0f, 255.0f);
	bgRgba[3] = 255;

	void* args[] = {
		&cptr_screenshot,
		&cptr_ssaoShadebuffer,
		&CuRastSettings::enableEDL,
		&CuRastSettings::enableSSAO,
		&view.framebuffer->width,
		&view.framebuffer->height,
		&backgroundColor
	};
	prog_resolve->launch2D("kernel_resolve_colorbuffer_to_screenshot", args, target.width, target.height);

	void* screenshot_host = nullptr;
	cuMemAllocHost(&screenshot_host, 4 * numPixels);
	cuMemcpyDtoH(screenshot_host, cptr_screenshot, 4 * numPixels);

	string path = "";
	if(*CuRastSettings::requestScreenshot == ""){
		for(int i = 0; i <= 10'000'000; i++){
			fs::create_directories("./screenshots");
			path = format("./screenshots/screenshot_{}.png", i);

			if(!fs::exists(path)) break;
		}
	}else{
		path = *CuRastSettings::requestScreenshot;
	}

	int stride_in_bytes = target.width * 4;
	stbi_flip_vertically_on_write(1);
	stbi_write_png(path.c_str(), target.width, target.height, 4, screenshot_host, stride_in_bytes);

	MemoryManager::free(cptr_screenshot);
	cuMemFreeHost(screenshot_host);
}

#include "CuRast_vulkanRender.h"
#include "scene/LasfileNode.h"
#include "scene/PotreeFileNode.h"

// Whether kernels can directly access pageable host memory, e.g. memory-mapped files (HMM on linux).
bool canAccessPageableMemory(){
	static bool supported = [](){
		CUdevice device;
		cuCtxGetDevice(&device);
		int supported = 0;
		cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, device);

		if(!supported){
			println("WARNING: GPU can not access pageable host memory. Memory-mapped point clouds will not be rendered.");
		}

		return supported != 0;
	}();

	return supported;
}

void drawPoints(Scene* scene, View view, RenderTarget& target){
	
	static CudaModularProgram* prog = new CudaModularProgram({
		.modules = {"./src/kernels/points.cu",}
	});
	
	
	vector<SNCPoints*> nodes;
	scene->forEach<SNCPoints>([&](SNCPoints* node){
		nodes.push_back(node);
	});
	
	u64 totalPoints = 0;
	for(SNCPoints* node : nodes){
		
		mat4 worldView = mat4(view.view * node->transform_global);
		
		void* args[] = {
			&target,
			&node->cptr_positions,
			&node->cptr_colors,
			&node->numPoints,
			&worldView
		};
		prog->launchCooperative("kernel_drawPoints", args, {.blocksize = 256});
		
		totalPoints += node->numPoints;
	}
	
	auto& dvlist = Runtime::debugValueList;
	dvlist.push_back({"num points", format("{:L}", totalPoints)});
	
}

// Renders PotreeFileNodes directly from their memory-mapped octree.bin. 
// - Traverses the octrees of all potree files from largest to smallest nodes in screen space, 
//   skipping nodes outside the view frustum, until the point budget is reached. 
// - Points are rendered in a coordinate system centered at the bounding box of each potree file.
// Requires GPU access to pageable host memory (e.g. HMM on linux).
void drawPotreeFiles(Scene* scene, View view, RenderTarget& target){

	constexpr u64 POINT_BUDGET = 5'000'000;
	
	static CudaModularProgram* prog = new CudaModularProgram({
		.modules = {"./src/kernels/potreeFileRenderer.cu",}
	});

	if(!canAccessPageableMemory()) return;

	vector<PotreeFileNode*> files;
	scene->forEach<PotreeFileNode>([&](PotreeFileNode* file){
		if(file->mapped_octree == nullptr) return;
		if(file->hierarchyNodes.empty()) return;
		if(file->encoding != "DEFAULT") return;

		files.push_back(file);
	});

	if(files.empty()) return;

	struct Plane{
		dvec3 normal;
		double d;
	};

	// per potree file state for the traversal
	struct FileState{
		PotreeFileNode* file;
		dvec3 center;       // bounding box center, becomes the origin of the rendered coordinate system
		dmat4 worldView;
		Plane planes[4];    // frustum side planes, relative to center
		i64 offset_color;
	};

	vector<FileState> states;
	for(PotreeFileNode* file : files){
		FileState state;
		state.file = file;
		state.center = (file->min + file->max) * 0.5;
		state.worldView = view.view * file->transform_global;

		dmat4 worldViewProj = view.proj * state.worldView;
		dvec4 row0 = glm::row(worldViewProj, 0);
		dvec4 row1 = glm::row(worldViewProj, 1);
		dvec4 row3 = glm::row(worldViewProj, 3);

		auto normalizedPlane = [](dvec4 p){
			double length = glm::length(dvec3(p));
			return Plane{dvec3(p) / length, p.w / length};
		};

		state.planes[0] = normalizedPlane(row3 + row0); // Left
		state.planes[1] = normalizedPlane(row3 - row0); // Right
		state.planes[2] = normalizedPlane(row3 + row1); // Bottom
		state.planes[3] = normalizedPlane(row3 - row1); // Top

		PotreeAttribute* rgb = file->findAttribute("rgb");
		state.offset_color = rgb ? rgb->byteOffset : -1;

		states.push_back(state);
	}

	// boxes are relative to the file's center
	auto isInsideFrustum = [](const FileState& state, dvec3 min, dvec3 max){
		for(const Plane& plane : state.planes){
			dvec3 positiveVertex = {
				plane.normal.x >= 0.0 ? max.x : min.x,
				plane.normal.y >= 0.0 ? max.y : min.y,
				plane.normal.z >= 0.0 ? max.z : min.z,
			};

			if(dot(plane.normal, positiveVertex) + plane.d < 0.0) return false;
		}

		return true;
	};

	// approximate radius of the node's bounding sphere in pixels
	double projectionFactor = abs(view.proj[1][1]) * 0.5 * double(target.height);
	auto getScreenSize = [&](const FileState& state, dvec3 min, dvec3 max) -> double {
		dvec3 center_view = dvec3(state.worldView * dvec4((min + max) * 0.5, 1.0));
		double radius = 0.5 * glm::length(dvec3(state.worldView * dvec4(max - min, 0.0)));
		double distance = glm::length(center_view);

		// camera inside the node's bounding sphere
		if(distance <= radius) return std::numeric_limits<double>::infinity();

		return projectionFactor * radius / distance;
	};

	struct QueueItem{
		i32 stateIndex;
		i32 nodeIndex;
		double screenSize;
	};
	auto smallerScreenSize = [](const QueueItem& a, const QueueItem& b){ return a.screenSize < b.screenSize; };
	std::priority_queue<QueueItem, vector<QueueItem>, decltype(smallerScreenSize)> queue(smallerScreenSize);

	for(i32 stateIndex = 0; stateIndex < states.size(); stateIndex++){
		const FileState& state = states[stateIndex];
		const PotreeHierarchyNode& root = state.file->hierarchyNodes[0];
		dvec3 min = root.min - state.center;
		dvec3 max = root.max - state.center;

		if(!isInsideFrustum(state, min, max)) continue;

		queue.push({stateIndex, 0, getScreenSize(state, min, max)});
	}

	// Traverse from largest to smallest nodes in screen space, until the point budget is reached
	static vector<PotreeNode> visibleNodes;
	visibleNodes.clear();
	u64 numVisiblePoints = 0;

	while(!queue.empty()){
		QueueItem item = queue.top();
		queue.pop();

		FileState& state = states[item.stateIndex];
		PotreeFileNode* file = state.file;

		// proxies only know their point count, not yet where their points are
		file->loadHierarchyChunk(item.nodeIndex);
		const PotreeHierarchyNode& node = file->hierarchyNodes[item.nodeIndex];

		if(numVisiblePoints + node.numPoints > POINT_BUDGET) break;

		numVisiblePoints += node.numPoints;

		PotreeNode visibleNode;
		visibleNode.data            = (u8*)file->mapped_octree + node.byteOffset;
		visibleNode.numPoints       = node.numPoints;
		visibleNode.offset_color    = state.offset_color >= 0 ? state.offset_color : ~0ull;
		visibleNode.worldView       = mat4(state.worldView);
		visibleNode.bytesPerPoint   = file->bytesPerPoint;
		visibleNode.offset_position = file->findAttribute("position")->byteOffset;
		visibleNode.scale           = vec3(file->scale);
		visibleNode.offset          = vec3(file->offset - state.center);
		visibleNodes.push_back(visibleNode);

		for(i32 childIndex : node.children){
			if(childIndex < 0) continue;

			const PotreeHierarchyNode& child = file->hierarchyNodes[childIndex];
			dvec3 min = child.min - state.center;
			dvec3 max = child.max - state.center;

			if(!isInsideFrustum(state, min, max)) continue;

			queue.push({item.stateIndex, childIndex, getScreenSize(state, min, max)});
		}
	}

	if(visibleNodes.size() > 0){

		// upload list of visible nodes
		static CUdeviceptr cptr_nodes = 0;
		static u64 capacity = 0;
		if(visibleNodes.size() > capacity){
			if(cptr_nodes != 0) MemoryManager::free(cptr_nodes);

			capacity = std::max<u64>(2 * visibleNodes.size(), 1'000);
			cptr_nodes = MemoryManager::alloc(capacity * sizeof(PotreeNode), "potree visible nodes");
		}
		cuMemcpyHtoD(cptr_nodes, visibleNodes.data(), byteSizeOf(visibleNodes));

		u64 numNodes = visibleNodes.size();
		void* args[] = {
			&target,
			&cptr_nodes,
			&numNodes
		};
		prog->launch("kernel_drawPotreeFileNodes", args, {.gridsize = u32(numNodes), .blocksize = 256});
	}

	auto& dvlist = Runtime::debugValueList;
	dvlist.push_back({"potree nodes", format("{:L}", visibleNodes.size())});
	dvlist.push_back({"potree points", format("{:L}", numVisiblePoints)});
}

// Renders LasfileNodes directly from their memory-mapped files. 
// Requires GPU access to pageable host memory (e.g. HMM on linux).
void drawLasPoints(Scene* scene, View view, RenderTarget& target){
	
	static CudaModularProgram* prog = new CudaModularProgram({
		.modules = {"./src/kernels/laspoints.cu",}
	});

	if(!canAccessPageableMemory()) return;
	
	vector<LasfileNode*> nodes;
	scene->forEach<LasfileNode>([&](LasfileNode* node){
		if(node->mapped == nullptr) return;
		if(node->compressed) return;

		nodes.push_back(node);
	});
	
	u64 totalPoints = 0;
	for(LasfileNode* node : nodes){
		
		mat4 worldView         = mat4(view.view * node->transform_global);
		u8* points             = (u8*)node->mapped + node->offset_pointData;
		u64 numPoints          = node->numPoints;
		u32 pointRecordSize    = node->pointRecordSize;
		i32 offset_rgb         = node->offset_rgb;
		vec3 scale             = node->scale;
		
		void* args[] = {
			&target,
			&points,
			&numPoints,
			&pointRecordSize,
			&offset_rgb,
			&scale,
			&worldView
		};
		prog->launchCooperative("kernel_drawLasPoints", args, {.blocksize = 256});
		
		totalPoints += std::min<u64>(numPoints, MAX_LAS_POINTS);
	}
	
	auto& dvlist = Runtime::debugValueList;
	dvlist.push_back({"num las points", format("{:L}", totalPoints)});
	
}

void drawTrianglesVisbuffer(
	Scene* scene, View view, vector<CMesh>& meshes, 
	vector<CMesh>& instances, CUdeviceptr cptr_meshes,
	CUdeviceptr cptr_instances, CUdeviceptr cptr_transforms, CUdeviceptr cptr_triangleCountPrefixsum,
	RenderTarget& target, MappedTextures& mappings
){
	auto editor = CuRast::instance;

	if(meshes.size() == 0) return;

	static CUdeviceptr cptr_numProcessedBatches             = MemoryManager::alloc(4, "cptr_numProcessedBatches");
	static CUdeviceptr cptr_numProcessedBatches_nontrivial  = MemoryManager::alloc(4, "cptr_numProcessedBatches_nontrivial");
	static CUdeviceptr cptr_hugeTriangles                   = MemoryManager::alloc(MAX_HUGE_TRIANGLES * sizeof(HugeTriangle), "cptr_hugeTriangles");
	static CUdeviceptr cptr_hugeTrianglesCounter            = MemoryManager::alloc(4, "cptr_hugeTrianglesCounter");
	static CUdeviceptr cptr_nontrivialCounter               = MemoryManager::alloc(4, "cptr_nontrivialCounter");
	static CUdeviceptr cptr_nontrivialList                  = MemoryManager::alloc(8 * MAX_NONTRIVIAL_TRIANGLES, "cptr_nontrivialList");
	static CUdeviceptr cptr_numProcessedHugeTriangles       = MemoryManager::alloc(4, "cptr_numProcessedHugeTriangles");
	
	static CudaModularProgram* prog = new CudaModularProgram({
		.modules = {"./src/kernels/triangles_visbuffer.cu",}
	});

	if(instances.size() == 0) return;

	auto custart = Timer::recordCudaTimestamp();

	bool isCompressed = meshes[0].compressed;
	string strCompressed = isCompressed ? "_compressed" : "_uncompressed";
	string strInstanced = (CuRastSettings::rasterizer == RASTERIZER_VISBUFFER_INSTANCED) ? "_instanced" : "";

	string strKernelStage1 = format("kernel_stage1_drawSmallTriangles_indexbuffer{}{}", strCompressed, strInstanced);
	string strKernelStage2 = format("kernel_stage2_drawMediumTriangles_indexbuffer{}", strCompressed);
	string strKernelStage3 = format("kernel_stage3_drawHugeTriangles_indexbuffer{}", strCompressed);

	RasterArgs args;
	args.meshes                          = (CMesh*)cptr_meshes;
	args.numMeshes                       = meshes.size(); 
	args.instances                       = (CMesh*)cptr_instances;
	args.numInstances                    = instances.size();
	args.transforms                      = (mat4*)cptr_transforms;
	args.numProcessedBatches             = (u32*)cptr_numProcessedBatches;
	args.numProcessedBatches_nontrivial  = (u32*)cptr_numProcessedBatches_nontrivial;
	args.hugeTriangles                   = (HugeTriangle*)cptr_hugeTriangles;
	args.hugeTrianglesCounter            = (u32*)cptr_hugeTrianglesCounter;
	args.numProcessedHugeTriangles       = (u32*)cptr_numProcessedHugeTriangles;
	args.nontrivialTrianglesCounter      = (u32*)cptr_nontrivialCounter;
	args.nontrivialTrianglesList         = (u64*)cptr_nontrivialList;
	args.target                          = target;
	args.state                           = (DeviceState*)CuRast::instance->cptr_state;
	
	prog->launchCooperative(strKernelStage1, vector<void*>{&args}, {.blocksize = TRIANGLES_PER_SWEEP});
	prog->launchCooperative(strKernelStage2, vector<void*>{&args});
	prog->launchCooperative(strKernelStage3, vector<void*>{&args}, {.blocksize = 64});

	auto cuend = Timer::recordCudaTimestamp();
	Timer::recordDuration("<triangles visbuffer pipeline>", custart, cuend);
}

void cubSortUint64Keys(uint64_t* d_keys_in, uint64_t* d_keys_out, int num_items);

void drawTrianglesTranslucent(
	Scene* scene, View view, vector<CMesh>& meshes,
	vector<CMesh>& instances, CUdeviceptr cptr_meshes,
	CUdeviceptr cptr_instances, CUdeviceptr cptr_transforms, 
	CUdeviceptr cptr_triangleCountPrefixsum,
	RenderTarget& target, MappedTextures& mappings
){
	auto editor = CuRast::instance;

	if(meshes.size() == 0) return;

	static CUdeviceptr cptr_numProcessedBatches             = MemoryManager::alloc(4, "cptr_numProcessedBatches");
	static CUdeviceptr cptr_numProcessedBatches_nontrivial  = MemoryManager::alloc(4, "cptr_numProcessedBatches_nontrivial");
	static CUdeviceptr cptr_hugeTriangles                   = MemoryManager::alloc(MAX_HUGE_TRIANGLES * sizeof(HugeTriangle), "cptr_hugeTriangles");
	static CUdeviceptr cptr_hugeTrianglesCounter            = MemoryManager::alloc(4, "cptr_hugeTrianglesCounter");
	static CUdeviceptr cptr_nontrivialCounter               = MemoryManager::alloc(4, "cptr_nontrivialCounter");
	static CUdeviceptr cptr_nontrivialList                  = MemoryManager::alloc(8 * MAX_NONTRIVIAL_TRIANGLES, "cptr_nontrivialList");
	static CUdeviceptr cptr_numProcessedHugeTriangles       = MemoryManager::alloc(4, "cptr_numProcessedHugeTriangles");
	
	static CUdeviceptr cptr_queueTriangles                  = MemoryManager::alloc(MAX_TRANSLUCENT_TRIANGLES * sizeof(TranslucentTriangle), "cptr_queue");
	static CUdeviceptr cptr_queueKeyValue                   = MemoryManager::alloc(MAX_TRANSLUCENT_TRIANGLES * sizeof(uint64_t), "cptr_queueKeyValue");
	static CUdeviceptr cptr_queueKeyValueSorted             = MemoryManager::alloc(MAX_TRANSLUCENT_TRIANGLES * sizeof(uint64_t), "cptr_queueKeyValueSorted");
	static CUdeviceptr cptr_queueSize                       = MemoryManager::alloc(4, "cptr_queueSize");
	
	int maxTiles = 512 * 512; // Each tile is 16x16, so allows 8k x 8k 
	static CUdeviceptr cptr_tileRanges                      = MemoryManager::alloc(sizeof(ivec2) * maxTiles, "cptr_tileRanges");
	
	static CudaModularProgram* prog = new CudaModularProgram({
		.modules = {"./src/kernels/triangles_translucent.cu",}
	});

	if(instances.size() == 0) return;

	RasterArgs args;
	args.meshes                          = (CMesh*)cptr_meshes;
	args.numMeshes                       = meshes.size(); 
	args.instances                       = (CMesh*)cptr_instances;
	args.numInstances                    = instances.size();
	args.transforms                      = (mat4*)cptr_transforms;
	args.numProcessedBatches             = (u32*)cptr_numProcessedBatches;
	args.numProcessedBatches_nontrivial  = (u32*)cptr_numProcessedBatches_nontrivial;
	args.hugeTriangles                   = (HugeTriangle*)cptr_hugeTriangles;
	args.hugeTrianglesCounter            = (u32*)cptr_hugeTrianglesCounter;
	args.numProcessedHugeTriangles       = (u32*)cptr_numProcessedHugeTriangles;
	args.nontrivialTrianglesCounter      = (u32*)cptr_nontrivialCounter;
	args.nontrivialTrianglesList         = (u64*)cptr_nontrivialList;
	args.target                          = target;
	args.state                           = (DeviceState*)CuRast::instance->cptr_state;
	
	// Stage 1: Binning
	// Stage 2: Sorting
	// Stage 3: Compute Tile Ranges
	// Stage 3: Blending
	
	// string strKernelStage1 = format("kernel_binning", strCompressed, strInstanced);
	// string strKernelStage2 = format("kernel_", strCompressed);
	
	cuMemsetD8(cptr_queueSize, 0, 4);
	
	auto custart = Timer::recordCudaTimestamp();
	prog->launchCooperative("kernel_stage1_binning", vector<void*>{&args, &cptr_queueTriangles, &cptr_queueKeyValue, &cptr_queueSize}, {.blocksize = TRIANGLES_PER_SWEEP});
	// prog->launchCooperative(strKernelStage2, vector<void*>{&args});
	
	// Stage 2: Sort - read queue size to host (syncs the stream), then CUB-sort on GPU
	u32 queueSize = 0;
	cuMemcpyDtoH(&queueSize, cptr_queueSize, 4);
	if (queueSize > 0) {
		cubSortUint64Keys(
			reinterpret_cast<uint64_t*>(cptr_queueKeyValue),
			reinterpret_cast<uint64_t*>(cptr_queueKeyValueSorted),
			static_cast<int>(queueSize)
		);
	}
	
	u32 tiles_x = (target.width + 16 - 1) / 16;
	u32 tiles_y = (target.height + 16 - 1) / 16;
	u32 numTiles = tiles_x * tiles_y;
	cuMemsetD8(cptr_tileRanges, 0, sizeof(ivec2) * numTiles);
	
	prog->launchCooperative("kernel_stage3_computeRanges", vector<void*>{
		&args, 
		&cptr_queueTriangles, 
		&cptr_queueKeyValueSorted, 
		&cptr_queueSize,
		&cptr_tileRanges,
		&numTiles,
	});
	
	prog->launch("kernel_stage4_blend", vector<void*>{
		&args, 
		&cptr_queueTriangles, 
		&cptr_queueKeyValueSorted, 
		&cptr_queueSize,
		&cptr_tileRanges,
		&numTiles,
	}, {.gridsize = numTiles, .blocksize = 256});

	auto cuend = Timer::recordCudaTimestamp();
	Timer::recordDuration("<triangles translucent pipeline>", custart, cuend);
	
	auto& dvlist = Runtime::debugValueList;
	dvlist.push_back({"num tiles", format("{:L}", queueSize)});
}

void CuRast::draw(Scene* scene, vector<View> views){

	static vector<View> frustumViews;
	if(!CuRastSettings::freezeFrustum){
		frustumViews = views;
	}

	double t_start = now();

	View view = views[0]; // We discarded support for multiple views for now.
	mat4 viewI = inverse(view.view);
	vec3 cameraPos = vec3(viewI * vec4(0.0f, 0.0f, 0.0f, 1.0f));

	int supersamplingFactor = CuRastSettings::supersamplingFactor;

	RenderTarget target;
	target.framebuffer = (u64*)cvm_framebuffer->cptr;
	target.colorbuffer = (u64*)cvm_colorbuffer->cptr;
	target.width = supersamplingFactor * view.framebuffer->width;
	target.height = supersamplingFactor * view.framebuffer->height;
	target.view = view.view;
	target.viewI = viewI;
	target.proj = view.proj;
	target.cameraPos = cameraPos;

	

	// Since processing thousands of nodes can become expensive on CPU side:
	// - Use a persistent std::vector that keeps the capacity over multiple frames
	// - Collect a list of all mesh nodes
	// - Then update them concurrently
	bool hasJpegCompressedTextures = false; 
	Mesh* hoveredMesh = nullptr;
	static vector<SNTriangles*> nodes;
	nodes.clear();
	scene->forEach<SNTriangles>([&](SNTriangles* node){
		
		// if(!node->texture->isTranslucent) return;
		// if(nodes.size() >= 1) return;
		// if(node->mesh->numTriangles != 400) return;
		
		nodes.push_back(node);
		if(node->texture){
			hasJpegCompressedTextures = hasJpegCompressedTextures || node->texture->huffmanTables != nullptr;
		}
	});

	process_parallel(nodes, [&](SNTriangles* node, int64_t index){
		if(node->id == CuRast::deviceState->hovered_meshId){
			Runtime::hovered_node_name = node->name;
			Runtime::hovered_mesh_name = node->mesh->name;
			hoveredMesh = node->mesh;
		}
		node->update(view);
	});

	u64 numTotalTriangles = 0;
	u64 numTotalNodes = 0;
	u64 numVisibleTriangles = 0;
	u64 numVisibleNodes = 0;
	for(int64_t i = 0; i < nodes.size(); i++){
		SNTriangles* node = nodes[i];

		numTotalTriangles += node->mesh->numTriangles;
		numTotalNodes++;

		if(!node->visible) continue;
		if(!node->mesh->isLoaded) continue;

		nodes[numVisibleNodes] = node;

		numVisibleTriangles += node->mesh->numTriangles;
		numVisibleNodes++;
	}
	nodes.resize(numVisibleNodes);

	// Sort/Group by instance
	sort(std::execution::par, nodes.begin(), nodes.end(), [](SNTriangles* a, SNTriangles* b){

		if(a->mesh->numTriangles == b->mesh->numTriangles){
			return u64(a->mesh) < u64(b->mesh);
		}else{
			return a->mesh->numTriangles > b->mesh->numTriangles;
		}
	});

	if(CuRastSettings::rasterizer == RASTERIZER_VULKAN_INDEXED_DRAW){
		drawVulkan_indexed_draw(scene, nodes, view);
	}else if(CuRastSettings::rasterizer == RASTERIZER_VULKAN_INDEXPULLING_INSTANCED){
		drawVulkan_indexpulling_instanced_forward(scene, nodes, view);
	}else if(CuRastSettings::rasterizer == RASTERIZER_VULKAN_INDEXPULLING_VISBUFFER){
		drawVulkan_indexpulling_visibilitybuffer(scene, views);
	}else{
		VKRenderer::vulkanMeshDrawFn = nullptr; // use CUDA blit path in recordCommandBuffer

		auto toCMesh = [&](SNTriangles* node){

			u32 indexRange = node->mesh->index_max - node->mesh->index_min;
			u64 bitsPerIndex = ceil(log2f(float(indexRange + 1)));

			CMesh mesh;
			mesh.world                    = node->transform_global;
			mesh.positions                = (vec3*)node->mesh->cptr_position;
			mesh.uvs                      = (vec2*)node->mesh->cptr_uv;
			mesh.colors                   = (u32*)node->mesh->cptr_color;
			mesh.normals                  = (vec3*)node->mesh->cptr_normal;
			mesh.indices                  = (u32*)node->mesh->cptr_indices;
			mesh.index_min                = node->mesh->index_min;
			mesh.index_max                = node->mesh->index_max;
			mesh.bitsPerIndex             = bitsPerIndex;
			mesh.numTriangles             = node->mesh->numTriangles;
			mesh.numVertices              = node->mesh->numVertices;
			if (node->texture) {
				mesh.texture                  = *node->texture;
			}
			mesh.aabb                     = node->aabb;
			mesh.compressed               = node->mesh->compressed;
			mesh.compressionFactor        = (node->aabb.max - node->aabb.min) / 65536.0f;
			mesh.isLoaded                 = node->mesh->isLoaded;
			mesh.id                       = node->id;
			mesh.address                  = u64(node->mesh);

			vec3 c0 = node->transform_global[0];
			vec3 c1 = node->transform_global[1];
			vec3 c2 = node->transform_global[2];
			float s = dot(cross(c0, c1), c2);
			mesh.flipTriangles = s < 0.0f;

			return mesh;
		};

		//----------------------------------------------
		// Organize into unique meshes and per-instance data
		//----------------------------------------------
		static vector<CMesh> meshes_unique;
		static vector<CMesh> meshes_allInstances;
		static vector<mat4> transforms;
		static vector<u64> triangleCountPrefixsum;
		int64_t sum = 0;
		
		meshes_unique.resize(nodes.size());
		meshes_allInstances.resize(nodes.size());
		transforms.resize(nodes.size());
		triangleCountPrefixsum.resize(nodes.size());
		
		process_parallel(nodes, [&](SNTriangles* node, int64_t index){
			CMesh cmesh = toCMesh(node);
			meshes_allInstances[index] = cmesh;
			transforms[index] = target.view * cmesh.world;
		});

		CMesh* uniqueMesh = nullptr;
		u64 uniqueMeshCounter = 0;
		for(int i = 0; i < nodes.size(); i++){
			CMesh& cmesh = meshes_allInstances[i];

			cmesh.cummulativeTriangleCount = sum;
			cmesh.instances.offset = i;

			triangleCountPrefixsum[i] = sum;
			sum += cmesh.numTriangles;

			// Encountered a new unique mesh
			if(uniqueMesh == nullptr || uniqueMesh->address != cmesh.address || CuRastSettings::disableInstancing){
				meshes_unique[uniqueMeshCounter] = cmesh;
				uniqueMesh = &meshes_unique[uniqueMeshCounter];
				uniqueMesh->instances.count = 0;
				uniqueMeshCounter++;
			}

			uniqueMesh->instances.count++;
		}
		meshes_unique.resize(uniqueMeshCounter);
		
		// Group into opaque and translucent meshes to render each with the corresponding cuda kernels
		static vector<CMesh> meshes_unique_opaque;
		static vector<CMesh> meshes_unique_translucent;
		meshes_unique_opaque.resize(0);
		meshes_unique_translucent.resize(0);
		
		for(CMesh mesh : meshes_unique){
			bool isTranslucent = mesh.texture.isTranslucent;
			if(!CuRastSettings::enableTranslucency){
				isTranslucent = false;
			}
			
			if(isTranslucent){
				meshes_unique_translucent.push_back(mesh);
			}else{
				meshes_unique_opaque.push_back(mesh);
			}
		}

		// prep virtual memory for lots of nodes
		static CudaVirtualMemory* cvm_meshes                    = MemoryManager::allocVirtualCuda(1'000'000 * sizeof(CMesh), "cvm_meshes");
		static CudaVirtualMemory* cvm_meshes_opaque             = MemoryManager::allocVirtualCuda(1'000'000 * sizeof(CMesh), "cvm_meshes_opaque");
		static CudaVirtualMemory* cvm_meshes_translucent        = MemoryManager::allocVirtualCuda(1'000'000 * sizeof(CMesh), "cvm_meshes_translucent");
		static CudaVirtualMemory* cvm_instances                 = MemoryManager::allocVirtualCuda(1'000'000 * sizeof(CMesh), "cvm_instances");
		static CudaVirtualMemory* cvm_transforms                = MemoryManager::allocVirtualCuda(1'000'000 * sizeof(mat4), "cvm_transforms");
		static CudaVirtualMemory* cvm_triangleCountPrefixsum    = MemoryManager::allocVirtualCuda(1'000'000 * sizeof(u64), "cvm_triangleCountPrefixsum");
		
		// commit physical memory for actual amount of nodes
		cvm_meshes                 ->commit(meshes_unique.size()             * sizeof(CMesh));
		cvm_meshes_opaque          ->commit(meshes_unique_opaque.size()      * sizeof(CMesh));
		cvm_meshes_translucent     ->commit(meshes_unique_translucent.size() * sizeof(CMesh));
		cvm_instances              ->commit(meshes_allInstances.size()       * sizeof(CMesh));
		cvm_transforms             ->commit(transforms.size()                * sizeof(mat4));
		cvm_triangleCountPrefixsum ->commit(triangleCountPrefixsum.size()    * sizeof(u64));

		// submit per-frame geometry metadata to GPU
		cuMemcpyHtoDAsync(cvm_meshes->cptr                 , meshes_unique.data(),             byteSizeOf(meshes_unique), 0);
		cuMemcpyHtoDAsync(cvm_meshes_opaque->cptr          , meshes_unique_opaque.data(),      byteSizeOf(meshes_unique_opaque), 0);
		cuMemcpyHtoDAsync(cvm_meshes_translucent->cptr     , meshes_unique_translucent.data(), byteSizeOf(meshes_unique_translucent), 0);
		cuMemcpyHtoDAsync(cvm_instances->cptr              , meshes_allInstances.data(),       byteSizeOf(meshes_allInstances), 0);
		cuMemcpyHtoDAsync(cvm_transforms->cptr             , transforms.data(),                byteSizeOf(transforms), 0);
		cuMemcpyHtoDAsync(cvm_triangleCountPrefixsum->cptr , triangleCountPrefixsum.data(),    byteSizeOf(triangleCountPrefixsum), 0);

		Runtime::numVisibleNodes = numVisibleNodes;
		Runtime::numVisibleTriangles = numVisibleTriangles;
		Runtime::numNodes = numTotalNodes;
		Runtime::numTriangles = numTotalTriangles;

		auto& dvlist = Runtime::debugValueList;
		dvlist.push_back({"#total nodes           ", format("{:40L}", u64(numTotalNodes))});
		dvlist.push_back({"#total triangles       ", format("{:40L}", u64(numTotalTriangles))});
		dvlist.push_back({"#visible nodes         ", format("{:40L}", u64(numVisibleNodes))});
		dvlist.push_back({"#visible triangles     ", format("{:40L}", u64(numVisibleTriangles))});
		dvlist.push_back({"hovered mesh id        ", format("{:40L}", CuRast::deviceState->hovered_meshId)});
		dvlist.push_back({"hovered triangle index ", format("{:40L}", CuRast::deviceState->hovered_triangleIndex)});
		dvlist.push_back({"hovered node name      ", format("{:}", Runtime::hovered_node_name)});
		dvlist.push_back({"hovered mesh name      ", format("{:}", Runtime::hovered_mesh_name)});
		dvlist.push_back({"tris in hovered mesh   ", format("{:40L}", hoveredMesh ? hoveredMesh->numTriangles : 0)});
		dvlist.push_back({"verts in hovered mesh  ", format("{:40L}", hoveredMesh ? hoveredMesh->numVertices : 0)});
		dvlist.push_back({"CPU draw() duration    ", format("{:40.1f} ms", Runtime::duration_draw * 1000.0)});

		// We measure CPU draw time until here, where CPU has finished its stuff and now just invokes cuda kernels.
		Runtime::duration_draw = now() - t_start;

		int numPixels = target.width * target.height;

		vector<shared_ptr<VKTexture>> attachments = {view.framebuffer->colorAttachment};
		auto mappings = mapCudaVk(attachments);

		static CudaModularProgram* prog = new CudaModularProgram({"./src/kernels/resolve.cu",});
		// memcpy arguments to constant buffer
		CUdeviceptr cptr_target = prog->getGlobalsPointer("c_target");
		cuMemcpyHtoDAsync(cptr_target, &target, sizeof(target), 0);

		// Let the first kernel in the frame be a dummy kernel to take the hit for CUDA-OpenGL interop overhead
		// (so that we get more accurate timings for the other kernels)
		static CUdeviceptr dummydata = MemoryManager::alloc(16, "dummydata");
		prog->launch("kernel_dummy", {&dummydata}, 1);
		
		{ // resize and clear cuda framebuffer
			u32 clearColor = 0xff000000;
			float clearDepth = Infinity;

			u64 requiredBytes = numPixels * 8;
			cvm_framebuffer->commit(requiredBytes);
			cvm_colorbuffer->commit(requiredBytes);

			prog->launch("kernel_clearFramebuffer", {
				&cvm_framebuffer->cptr,
				&numPixels,
				&clearColor,
				&clearDepth
			}, numPixels);

			prog->launch("kernel_clearFramebuffer", {
				&cvm_colorbuffer->cptr,
				&numPixels,
				&clearColor,
				&clearDepth
			}, numPixels);
		}

		drawTrianglesVisbuffer(
			scene, view, meshes_unique_opaque, meshes_allInstances, 
			cvm_meshes_opaque->cptr, 
			cvm_instances->cptr, cvm_transforms->cptr, cvm_triangleCountPrefixsum->cptr,
			target, mappings
		);

		// DRAW BOUNDING BOXES
		if(CuRastSettings::showBoundingBoxes){
			RenderTarget target_lines = target;
			target_lines.framebuffer = (u64*)cvm_colorbuffer->cptr;

			vector<CMesh> boundingBoxNodes;
			scene->forEach<SNTriangles>([&](SNTriangles* node){
				CMesh mesh;
				mesh.world                    = node->transform_global;
				mesh.aabb                     = node->aabb;

				boundingBoxNodes.push_back(mesh);
			});
			
			static CUdeviceptr cptr_numProcessedBatches = MemoryManager::alloc(4, "cptr_numProcessedBatches");
			// static CUdeviceptr cptr_meshes_boxes = MemoryManager::alloc(40'000 * sizeof(CMesh), "cptr_meshes_boxes");
			static CudaVirtualMemory* cvm_meshes_boxes = MemoryManager::allocVirtualCuda(40'000 * sizeof(CMesh), "boxes");
			cvm_meshes_boxes->commit(boundingBoxNodes.size() * sizeof(CMesh));

			cuMemcpyHtoDAsync(cvm_meshes_boxes->cptr, boundingBoxNodes.data(), boundingBoxNodes.size() * sizeof(CMesh), 0);
			cuMemsetD8Async(cptr_numProcessedBatches, 0, 4, 0);
			
			u32 numMeshes = boundingBoxNodes.size();
			launch_drawBoundingBoxes(
				target_lines, 
				(CMesh*)cvm_meshes_boxes->cptr,
				numMeshes,
				(u32*)cptr_numProcessedBatches
			);
		}

		int mouse_X = Runtime::mousePosition.x;
		int mouse_Y = target.height - Runtime::mousePosition.y;
		u32 numInstances = meshes_allInstances.size();

		RasterizationSettings rasterSettings;
		rasterSettings.showWireframe = CuRastSettings::showWireframe;
		rasterSettings.enableDiffuseLighting = CuRastSettings::enableDiffuseLighting;
		rasterSettings.displayAttribute = CuRastSettings::displayAttribute;
		rasterSettings.enableObjectPicking = CuRastSettings::enableObjectPicking;

		JpegPipeline jpp;
		jpp.toDecode             = (u32*)jpegTextures->cptr_toDecode;
		jpp.toDecodeCounter      = (u32*)jpegTextures->cptr_toDecodeCounter;
		jpp.decoded              = (u32*)jpegTextures->cptr_decoded;
		jpp.TBSlots              = (u32*)jpegTextures->cptr_TBSlots;
		jpp.TBSlotsCounter       = (u32*)jpegTextures->cptr_TBSlotsCounter;
		jpp.decodedMcuMap        = *jpegTextures->decodedMcuMap;
		
		static CUdeviceptr cptr_textures = MemoryManager::alloc(MAX_TEXTURES * sizeof(Texture), "texture list");

		if(hasJpegCompressedTextures){
			cuMemcpyHtoD(cptr_textures, TextureManager::textures, TextureManager::numTextures * sizeof(Texture));
			cuMemsetD32(jpegTextures->cptr_toDecodeCounter, 0, 1);
			// cuMemsetD32(cptr_TBSlotsCounter, 0, 1);
		}


		{ // RESOLVE VISIBILITY BUFFER (write colors to colorbuffer)
			void* args[] = {
				&cvm_instances->cptr,
				&numInstances,
				&cvm_triangleCountPrefixsum->cptr,
				&mouse_X,
				&mouse_Y,
				&cptr_state,
				&rasterSettings,
				&jpp,
			};
			prog->launch2D("kernel_resolve_visbuffer_to_colorbuffer2D", args, target.width, target.height);
		}
		
		drawPoints(scene, view, target);
		drawLasPoints(scene, view, target);
		drawPotreeFiles(scene, view, target);
		
		drawTrianglesTranslucent(
			scene, view, meshes_unique_translucent, meshes_allInstances, 
			cvm_meshes_translucent->cptr, 
			cvm_instances->cptr, cvm_transforms->cptr, cvm_triangleCountPrefixsum->cptr,
			target, mappings
		);

		if(hasJpegCompressedTextures){
			u32 toDecodeCounter;
			cuMemcpyDtoH(&toDecodeCounter, (CUdeviceptr)jpp.toDecodeCounter, 4);
			dvlist.push_back({"toDecodeCounter ", format("{}", toDecodeCounter)});

			// DECODE JPEG TEXTURES
			jpegTextures->prog->launch("kernel_launch_decode", {
				&jpegTextures->cptr_toDecodeCounter,
				&jpegTextures->cptr_TBSlots, 
				&jpegTextures->cptr_TBSlotsCounter,
				&jpegTextures->cptr_toDecode,
				&jpegTextures->cptr_decoded,
				&cptr_textures,
				// &jpegTextures->cptr_texture_pointer,
				jpegTextures->decodedMcuMap,
			}, 1);


			{ // RESOLVE JPEG
				void* args[] = {
					&cvm_instances->cptr,
					&numInstances,
					&cvm_triangleCountPrefixsum->cptr,
					&mouse_X,
					&mouse_Y,
					&cptr_state,
					&rasterSettings,
					&jpp,
					&cptr_textures
				};
				prog->launch2D("kernel_resolve_jpeg", args, target.width, target.height);
			}

			{// DEBUG
				u32 C = CuRast::deviceState->dbg_hovered_decoded_color;
				uint8_t* rgba = (uint8_t*)&C;
				string strColor = format("{:3}, {:3}, {:3}", rgba[0], rgba[1], rgba[2]);

				dvlist.push_back({"CPU draw() duration    ", format("{:.1f} ms", Runtime::duration_draw * 1000.0)});
				dvlist.push_back({"hovered_textureHandle  ", format("{:12}", CuRast::deviceState->dbg_hovered_textureHandle)});
				dvlist.push_back({"hovered_mipLevel       ", format("{:12}", CuRast::deviceState->dbg_hovered_mipLevel)});
				dvlist.push_back({"hovered_tx             ", format("{:12}", CuRast::deviceState->dbg_hovered_tx)});
				dvlist.push_back({"hovered_ty             ", format("{:12}", CuRast::deviceState->dbg_hovered_ty)});
				dvlist.push_back({"hovered_mcu_x          ", format("{:12}", CuRast::deviceState->dbg_hovered_mcu_x)});
				dvlist.push_back({"hovered_mcu_y          ", format("{:12}", CuRast::deviceState->dbg_hovered_mcu_y)});
				dvlist.push_back({"hovered_mcu            ", format("{:12}", CuRast::deviceState->dbg_hovered_mcu)});
				dvlist.push_back({"hovered_decoded_color  ", format("{:12}", strColor)});
			}
		
			cuMemsetD8((CUdeviceptr)jpegTextures->decodedMcuMap_tmp->entries, 0xff, jpegTextures->decodedMcuMap_tmp->capacity * 8);
			// bool freezeCache = editor->settings.freezeCache;
			bool freezeCache = false;
			jpegTextures->prog->launch("kernel_update_cache", {
				jpegTextures->decodedMcuMap, 
				jpegTextures->decodedMcuMap_tmp, 
				&jpegTextures->cptr_TBSlots,
				&jpegTextures->cptr_TBSlotsCounter,
				&freezeCache
			}, jpegTextures->decodedMcuMap->capacity);
			cuMemcpy((CUdeviceptr)jpegTextures->decodedMcuMap->entries, (CUdeviceptr)jpegTextures->decodedMcuMap_tmp->entries, jpegTextures->decodedMcuMap_tmp->capacity * 8);

			// {
			// 	// Disable caching by fully clearing the MCU slot list and hash map at the end of each frame.
			// 	// This let's us see how much slower the decode kernel becomes.
			// 	cuMemsetD8((CUdeviceptr)jpegTextures->decodedMcuMap->entries, 0xff, jpegTextures->decodedMcuMap->capacity * 8);
			// 	u32 capacity = JPEG_NUM_DECODED_MCU_CAPACITY;
			// 	jpegTextures->prog->launch("kernel_init_availableMcuSlots", {
			// 		&jpegTextures->cptr_TBSlots, 
			// 		&jpegTextures->cptr_TBSlotsCounter, 
			// 		&capacity
			// 	}, capacity,  0);

			// 	cuMemsetD32(jpegTextures->cptr_TBSlotsCounter, 0, 1);
			// }
		}

		// { // TEST: Draw Heightmap
		// 	static CudaModularProgram* prog = new CudaModularProgram({"./src/kernels/triangles_heightmap.cu",});

		// 	CUdeviceptr cptr_target = prog->getGlobalsPointer("c_target");
		// 	cuMemcpyHtoDAsync(cptr_target, &target, sizeof(target), 0);

		// 	float w = settings.threshold;
		// 	int minCells = 128;
		// 	int maxCells = 40 * 1024;
		// 	int numCells = (1.0f - w) * float(minCells) + w * float(maxCells);
			

		// 	// int numCells = 5 * 1024;
		// 	int blocksize = 16;
		// 	int numBlocks = (numCells + blocksize - 1) / blocksize;

		// 	void* args[] = {
		// 		&numCells,
		// 		&cptr_colorbuffer
		// 	};

		// 	auto custart = Timer::recordCudaTimestamp();

		// 	auto res_launch = cuLaunchKernel(prog->kernels["kernel_drawHeightmap"],
		// 		numBlocks, numBlocks, 1,
		// 		blocksize, blocksize, 1,
		// 		0, 0, args, nullptr);

		// 	Timer::recordDuration("kernel_drawHeightmap", custart, Timer::recordCudaTimestamp());
		// }



		// SCREEN SPACE AMBIENT OCCLUSION
		static CudaVirtualMemory* cvm_ssaoShadebuffer = MemoryManager::allocVirtualCuda(2'000'000'000, "cvm_ssaoShadebuffer");
		if(CuRastSettings::enableSSAO){
			// save mem by using reusing the visibility buffer, which is no longer used in this frame
			CUdeviceptr cptr_occlusionbuffer = cvm_framebuffer->cptr;

			// But for the final ssao shading values, we need an extra buffer
			cvm_ssaoShadebuffer->commit(cvm_framebuffer->comitted / 2);

			void* argsSSAO[] = {
				&cvm_framebuffer->cptr,
				&cvm_ssaoShadebuffer->cptr
			};
			prog->launch2D("kernel_ssaoOcclusion", argsSSAO, target.width, target.height);
			prog->launch2D("kernel_ssaoBlur", argsSSAO, target.width, target.height);
		}

		// static CUdeviceptr cptr_enlarged = CURuntime::alloc("enlarge", 4096 * 4096 * 8);
		// prog->launchCooperative("kernel_enlarge", {
		// 	&mappings.surfaces[0],
		// 	&cptr_ssaoShadebuffer,
		// 	&cptr_enlarged,
		// 	&view.framebuffer->width, 
		// 	&view.framebuffer->height,
		// 	&mouse_X,
		// 	&mouse_Y,
		// 	&cptr_state,
		// 	&CuRastSettings::enableEDL,
		// 	&CuRastSettings::enableSSAO,
		// });

		{ // RESOLVE COLOR BUFFER (write to graphics API framebuffer)
			int viewWidth = view.framebuffer->width;
			int viewHeight = view.framebuffer->height;

			u32 backgroundColor = 0;
			uint8_t* bgRgba = (uint8_t*)&backgroundColor;
			bgRgba[0] = clamp(CuRastSettings::background.x * 256.0f, 0.0f, 255.0f);
			bgRgba[1] = clamp(CuRastSettings::background.y * 256.0f, 0.0f, 255.0f);
			bgRgba[2] = clamp(CuRastSettings::background.z * 256.0f, 0.0f, 255.0f);

			void* args[] = {
				&mappings.surfaces[0],
				&cvm_ssaoShadebuffer->cptr,
				&viewWidth, 
				&viewHeight,
				&mouse_X,
				&mouse_Y,
				&cptr_state,
				&CuRastSettings::enableEDL,
				&CuRastSettings::enableSSAO,
				&CuRastSettings::showInset,
				&backgroundColor
			};
			prog->launch2D("kernel_resolve_colorbuffer_to_opengl_2D", args, target.width, target.height);
		}

		if(CuRastSettings::requestScreenshot){
			saveScreenshot(target, view, cvm_ssaoShadebuffer->cptr, prog);
		}

		unmapCudaVk(mappings);
		
		cuMemcpyDtoHAsync((void*)deviceState, cptr_state, sizeof(DeviceState), 0);

		if(deviceState->dbg_fragcount > 0){
			dvlist.push_back({"fragcounter", format("{:L}", deviceState->dbg_fragcount)});
		}
	}

	CuRastSettings::requestScreenshot = nullptr;
}

void initialize(){
	if(initialized) return;

	int defaultPixels = 1920 * 1080;
	int64_t virtualCapacity = 2'147'483'648; // sufficient for up to 4096 x 4096 pixels with 16x supersampling
	// int max_SuperSamples = 16;
	cvm_framebuffer = MemoryManager::allocVirtualCuda(virtualCapacity, "framebuffer");
	cvm_framebuffer->commit(8 * defaultPixels);

	cvm_colorbuffer = MemoryManager::allocVirtualCuda(virtualCapacity, "colorbuffer");
	cvm_colorbuffer->commit(8 * defaultPixels);

	jpegTextures = new JpegTextures();

	initialized = true;
}

void CuRast::render(){

	if(VKRenderer::width * VKRenderer::height == 0){
		return;
	}

	initialize();

	VKRenderer::view.framebuffer->setSize(VKRenderer::width, VKRenderer::height);
	
	// RENDER DESKTOP
	VKRenderer::view.proj =  VKRenderer::camera->proj;
	VKRenderer::view.view =  mat4(VKRenderer::camera->view);
	
	draw(&scene, {VKRenderer::view});

	Runtime::debugValues["small"]   = format(getSaneLocale(), "{:L}", deviceState->numSmall);
	Runtime::debugValues["large"]   = format(getSaneLocale(), "{:L}", deviceState->numLarge);
	Runtime::debugValues["massive"] = format(getSaneLocale(), "{:L}", deviceState->numMassive);

	{ // DRAW GUI
		ImGui::NewFrame();
		// ImGuizmo::BeginFrame();

		drawGUI();

		ImGui::Render();
	}

	Runtime::mouseEvents.clear();
}