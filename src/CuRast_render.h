
#include <unordered_set>
#include <execution>
#include <queue>

#include "Timer.h"
#include "VKRenderer.h"
#include "types.h"

using namespace std;

CudaVirtualMemory* cvm_framebuffer = nullptr;
CudaVirtualMemory* cvm_colorbuffer = nullptr;
bool initialized = false;

// Cuda-Vulkan interop
struct MappedTextures{
	vector<shared_ptr<VKTexture>> textures;
	vector<CUsurfObject> surfaces;
};

static unordered_map<int64_t, int64_t> lastImportedVersion;

// implemented in lines.cu 
void launch_drawBoundingBoxes(
	const RenderTarget& target,
	BoundingBox* boxes,
	u32 numBoxes,
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
		if(!CuRastSettings::enableFrustumCulling) return true;

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

void CuRast::draw(Scene* scene, vector<View> views){

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

	auto& dvlist = Runtime::debugValueList;

	int numPixels = target.width * target.height;

	vector<shared_ptr<VKTexture>> attachments = {view.framebuffer->colorAttachment};
	auto mappings = mapCudaVk(attachments);

	static CudaModularProgram* prog = new CudaModularProgram({"./src/kernels/resolve.cu",});
	// memcpy arguments to constant buffer
	CUdeviceptr cptr_target = prog->getGlobalsPointer("c_target");
	cuMemcpyHtoDAsync(cptr_target, &target, sizeof(target), 0);

	// Let the first kernel in the frame be a dummy kernel to take the hit for CUDA-Vulkan interop overhead
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

	drawPoints(scene, view, target);
	drawLasPoints(scene, view, target);
	drawPotreeFiles(scene, view, target);

	// DRAW BOUNDING BOXES
	if(CuRastSettings::showBoundingBoxes){
		RenderTarget target_lines = target;
		target_lines.framebuffer = (u64*)cvm_colorbuffer->cptr;

		vector<BoundingBox> boxes;
		scene->root->traverse([&](SceneNode* node){
			if(node->aabb.isDefault()) return;

			BoundingBox box;
			box.world = node->transform_global;
			box.aabb  = node->aabb;

			boxes.push_back(box);
		});

		if(boxes.size() > 0){
			static CUdeviceptr cptr_numProcessedBatches = MemoryManager::alloc(4, "cptr_numProcessedBatches");
			static CudaVirtualMemory* cvm_boxes = MemoryManager::allocVirtualCuda(40'000 * sizeof(BoundingBox), "boxes");
			cvm_boxes->commit(boxes.size() * sizeof(BoundingBox));

			cuMemcpyHtoDAsync(cvm_boxes->cptr, boxes.data(), byteSizeOf(boxes), 0);
			cuMemsetD8Async(cptr_numProcessedBatches, 0, 4, 0);

			u32 numBoxes = boxes.size();
			launch_drawBoundingBoxes(
				target_lines,
				(BoundingBox*)cvm_boxes->cptr,
				numBoxes,
				(u32*)cptr_numProcessedBatches
			);
		}
	}

	int mouse_X = Runtime::mousePosition.x;
	int mouse_Y = target.height - Runtime::mousePosition.y;

	// SCREEN SPACE AMBIENT OCCLUSION
	static CudaVirtualMemory* cvm_ssaoShadebuffer = MemoryManager::allocVirtualCuda(2'000'000'000, "cvm_ssaoShadebuffer");
	if(CuRastSettings::enableSSAO){
		// the framebuffer is not otherwise used, so it stores the occlusion values.
		// But for the final ssao shading values, we need an extra buffer
		cvm_ssaoShadebuffer->commit(cvm_framebuffer->comitted / 2);

		void* argsSSAO[] = {
			&cvm_framebuffer->cptr,
			&cvm_ssaoShadebuffer->cptr
		};
		prog->launch2D("kernel_ssaoOcclusion", argsSSAO, target.width, target.height);
		prog->launch2D("kernel_ssaoBlur", argsSSAO, target.width, target.height);
	}

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

	{ // DRAW GUI
		ImGui::NewFrame();
		// ImGuizmo::BeginFrame();

		drawGUI();

		ImGui::Render();
	}

	Runtime::mouseEvents.clear();
}