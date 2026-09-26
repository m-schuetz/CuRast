#include <unordered_set>
#include <execution>
#include <queue>

#include "CuRast.h"
#include "VKRenderer.h"
#include "Timer.h"
#include "types.h"
#include "scene/LasfileNode.h"
#include "scene/PotreeFileNode.h"
#include "kernels/kernels.h"

using namespace std;

void CuRast::setup(){
	CuRast::instance = new CuRast();
}

void CuRast::resetEditor(){
	scene.world->children.clear();
}


void CuRast::inputHandling(){

	Runtime::controls->onMouseMove(Runtime::mouseEvents.pos_x, Runtime::mouseEvents.pos_y);
	Runtime::controls->onMouseScroll(Runtime::mouseEvents.wheel_x, Runtime::mouseEvents.wheel_y);
	Runtime::controls->update();

	VKRenderer::camera->view = inverse(Runtime::controls->world);
	VKRenderer::camera->world = Runtime::controls->world;
}

void CuRast::drawGUI() {

	if(!CuRastSettings::hideGUI){
		makeMenubar();
		makeToolbar();
		makeDevGUI();
		makeStats();
		makeDirectStats();
	}else{
		ImVec2 kernelWindowSize = {70, 25};
		ImGui::SetNextWindowPos({VKRenderer::width - kernelWindowSize.x, -8});
		ImGui::SetNextWindowSize(kernelWindowSize);

		ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar
			| ImGuiWindowFlags_NoResize
			| ImGuiWindowFlags_NoMove
			| ImGuiWindowFlags_NoScrollbar
			| ImGuiWindowFlags_NoScrollWithMouse
			| ImGuiWindowFlags_NoCollapse
			// | ImGuiWindowFlags_AlwaysAutoResize
			| ImGuiWindowFlags_NoBackground
			| ImGuiWindowFlags_NoSavedSettings
			| ImGuiWindowFlags_NoDecoration;
		static bool open;

		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1.0f, 1.0f, 1.0f, 0.0f));
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.5f));

		if(ImGui::Begin("ShowGuiWindow", &open, flags)){
			if(ImGui::Button("Show GUI")){
				CuRastSettings::hideGUI = !CuRastSettings::hideGUI;
			}
		}
		ImGui::End();
		
		ImGui::PopStyleColor(2);

	}
}

void CuRast::update(){

	Runtime::timings.newFrame();
	
	string strfps = format("CuRast | FPS: {}", int(VKRenderer::fps));
	glfwSetWindowTitle(VKRenderer::window, strfps.c_str());

	// scene.updateTransformations();
	// scene.update();
	Runtime::debugValues.clear();
	Runtime::debugValueList.clear();

	if(VKRenderer::width * VKRenderer::height == 0){
		return;
	}

	Timer::enabled = Runtime::measureTimings;

	scene.updateTransformations();
	inputHandling();
	// scene.updateTransformations();
};

void CuRast::postFrame(){
	
}

void alignRight(string text) {
	float rightBorder = ImGui::GetCursorPosX() + ImGui::GetColumnWidth();
	float width = ImGui::CalcTextSize(text.c_str()).x;
	ImGui::SetCursorPosX(rightBorder - width);
}

CudaBuffer* colorbuffer = nullptr;
bool initialized = false;

// Cuda-Vulkan interop
struct MappedTextures{
	vector<shared_ptr<VKTexture>> textures;
	vector<cudaSurfaceObject_t> surfaces;
};

static unordered_map<int64_t, int64_t> lastImportedVersion;

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
	cudaStreamSynchronize(0);
}

// Whether kernels can directly access pageable host memory, e.g. memory-mapped files (HMM on linux).
bool canAccessPageableMemory(){
	static bool supported = [](){
		int supported = 0;
		cudaDeviceGetAttribute(&supported, cudaDevAttrPageableMemoryAccess, CURuntime::device);

		if(!supported){
			println("WARNING: GPU can not access pageable host memory. Memory-mapped point clouds will not be rendered.");
		}

		return supported != 0;
	}();

	return supported;
}

// Renders PotreeFileNodes directly from their memory-mapped octree.bin. 
// - Traverses the octrees of all potree files from largest to smallest nodes in screen space, 
//   skipping nodes outside the view frustum, until the point budget is reached. 
// - Points are rendered in a coordinate system centered at the bounding box of each potree file.
// Requires GPU access to pageable host memory (e.g. HMM on linux).
void drawPotreeFiles(Scene* scene, View view, RenderTarget& target){

	u64 pointBudget = CuRastSettings::pointBudget;

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

		if(numVisiblePoints + node.numPoints > pointBudget) break;

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
		static PotreeNode* nodes = nullptr;
		static u64 capacity = 0;
		if(visibleNodes.size() > capacity){
			if(nodes != nullptr) MemoryManager::free(nodes);

			capacity = std::max<u64>(2 * visibleNodes.size(), 1'000);
			nodes = (PotreeNode*)MemoryManager::alloc(capacity * sizeof(PotreeNode), "potree visible nodes");
		}
		cudaMemcpy(nodes, visibleNodes.data(), byteSizeOf(visibleNodes), cudaMemcpyHostToDevice);

		launch_drawPotreeFileNodes(target, nodes, visibleNodes.size());
	}

	auto& dvlist = Runtime::debugValueList;
	dvlist.push_back({"potree nodes", format("{:L}", visibleNodes.size())});
	dvlist.push_back({"potree points", format("{:L}", numVisiblePoints)});
}

// Renders LasfileNodes directly from their memory-mapped files. 
// Requires GPU access to pageable host memory (e.g. HMM on linux).
void drawLasPoints(Scene* scene, View view, RenderTarget& target){

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
		
		launch_drawLasPoints(target, points, numPoints, pointRecordSize, offset_rgb, scale, worldView);
		
		totalPoints += std::min<u64>(numPoints, MAX_LAS_POINTS);
	}
	
	auto& dvlist = Runtime::debugValueList;
	dvlist.push_back({"num las points", format("{:L}", totalPoints)});
	
}

void CuRast::draw(Scene* scene, vector<View> views){

	View view = views[0]; // We discarded support for multiple views for now.

	int supersamplingFactor = CuRastSettings::supersamplingFactor;

	RenderTarget target;
	target.colorbuffer = (u64*)colorbuffer->ptr;
	target.width = supersamplingFactor * view.framebuffer->width;
	target.height = supersamplingFactor * view.framebuffer->height;
	target.proj = view.proj;

	int numPixels = target.width * target.height;

	vector<shared_ptr<VKTexture>> attachments = {view.framebuffer->colorAttachment};
	auto mappings = mapCudaVk(attachments);

	// Let the first kernel in the frame be a dummy kernel to take the hit for CUDA-Vulkan interop overhead
	// (so that we get more accurate timings for the other kernels)
	static u32* dummydata = (u32*)MemoryManager::alloc(16, "dummydata");
	launch_dummy(dummydata);

	{ // resize and clear cuda colorbuffer
		u32 clearColor = 0xff000000;
		float clearDepth = Infinity;

		// resizing may reallocate the buffer
		colorbuffer->resize(u64(numPixels) * 8);
		target.colorbuffer = (u64*)colorbuffer->ptr;

		launch_clearFramebuffer(target.colorbuffer, numPixels, clearColor, clearDepth);
	}

	drawLasPoints(scene, view, target);
	drawPotreeFiles(scene, view, target);

	int mouse_X = Runtime::mousePosition.x;
	int mouse_Y = target.height - Runtime::mousePosition.y;

	{ // RESOLVE COLOR BUFFER (write to graphics API framebuffer)
		int viewWidth = view.framebuffer->width;
		int viewHeight = view.framebuffer->height;

		u32 backgroundColor = 0;
		uint8_t* bgRgba = (uint8_t*)&backgroundColor;
		bgRgba[0] = clamp(CuRastSettings::background.x * 256.0f, 0.0f, 255.0f);
		bgRgba[1] = clamp(CuRastSettings::background.y * 256.0f, 0.0f, 255.0f);
		bgRgba[2] = clamp(CuRastSettings::background.z * 256.0f, 0.0f, 255.0f);

		launch_resolveColorbufferToSurface(
			target, mappings.surfaces[0], 
			viewWidth, viewHeight, mouse_X, mouse_Y,
			CuRastSettings::enableEDL, CuRastSettings::showInset, backgroundColor);
	}

	unmapCudaVk(mappings);
}

void initialize(){
	if(initialized) return;

	int defaultPixels = 1920 * 1080;
	colorbuffer = MemoryManager::allocBuffer(8 * defaultPixels, "colorbuffer");

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

#include "gui/menubar.h"
#include "gui/toolbar.h"
#include "gui/widget_kernels.h"
#include "gui/widget_memory.h"
#include "gui/widget_timings.h"
#include "gui/stats.h"

void CuRast::makeDevGUI(){
	makeKernels();
	makeMemory();
	makeTimings();
}

