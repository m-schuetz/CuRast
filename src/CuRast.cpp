#include <unordered_set>
#include <execution>
#include <queue>
#include <atomic>

#include "CuRast.h"
#include "VKRenderer.h"
#include "Timer.h"
#include "types.h"
#include "scene/LasfileNode.h"
#include "scene/PotreeFileNode.h"
#include "scene/ClusteredMeshNode.h"
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

// ------------------------------------------------------------------------------------------------
// Direct storage: reading octree nodes from file into VRAM via cuFile (GPUDirect Storage)
// ------------------------------------------------------------------------------------------------

// Reads are aligned to SSD pages, as required for direct transfers from SSD to GPU
constexpr u64 DIRECT_STORAGE_PAGE_SIZE = 4096;

// Opens the cuFile driver on first use. Without the nvidia-fs kernel module or PCI P2PDMA, 
// cuFile runs in compatibility mode, i.e., reads go through its pinned host memory bounce buffers.
bool initCuFile(){
	static bool initialized = false;
	static bool success = false;

	if(!initialized){
		initialized = true;

		CUfileError_t status = cuFileDriverOpen();
		success = status.err == CU_FILE_SUCCESS;

		if(!success){
			println("WARNING: cuFileDriverOpen failed (error {}). Direct storage rendering is unavailable.", int(status.err));
		}
	}

	return success;
}

// VRAM buffer that receives the visible octree nodes each frame. 
// Allocated on first use of the direct storage path, and only grows (e.g. when the point budget increases).
struct DirectStorageBuffer{
	u8* ptr = nullptr;
	u64 size = 0;
};
DirectStorageBuffer directStorageBuffer;

void reserveDirectStorageBuffer(u64 requiredSize){
	if(requiredSize <= directStorageBuffer.size) return;

	// the previous frame is complete (see unmapCudaVk), so nothing uses the old buffer anymore
	if(directStorageBuffer.ptr != nullptr){
		cuFileBufDeregister(directStorageBuffer.ptr);
		MemoryManager::free(directStorageBuffer.ptr);
	}

	directStorageBuffer.ptr = (u8*)MemoryManager::alloc(requiredSize, "potree direct storage buffer");
	directStorageBuffer.size = requiredSize;

	// Registering lets cuFile transfer directly into the buffer instead of using internal bounce buffers
	CUfileError_t status = cuFileBufRegister(directStorageBuffer.ptr, requiredSize, 0);
	if(status.err != CU_FILE_SUCCESS){
		println("WARNING: cuFileBufRegister failed (error {}). Direct storage reads may be slower.", int(status.err));
	}
}

// A page-aligned read of an octree node (plus padding) from octree.bin to directStorageBuffer
struct DirectStorageRead{
	CUfileHandle_t file;
	u64 fileOffset;    // page-aligned
	u64 size;          // page-aligned
	u64 bufferOffset;  // page-aligned
	u64 requiredSize;  // minimum number of bytes that must be read, i.e., up to the end of the node
};

// Executes the reads in parallel. cuFileRead() is synchronous, so we issue them from multiple threads. 
// Returns the number of failed reads.
int executeDirectStorageReads(const vector<DirectStorageRead>& reads){
	if(reads.empty()) return 0;

	// throughput of a Samsung 9100 PRO saturated at around 32 threads (~9GB/s)
	int numThreads = std::min<int>({32, int(std::max(1u, thread::hardware_concurrency())), int(reads.size())});

	std::atomic<int> nextRead = 0;
	std::atomic<int> numFailed = 0;
	auto worker = [&](){
		cudaSetDevice(CURuntime::device);

		for(int i = nextRead++; i < reads.size(); i = nextRead++){
			const DirectStorageRead& read = reads[i];
			ssize_t bytesRead = cuFileRead(read.file, directStorageBuffer.ptr, read.size, read.fileOffset, read.bufferOffset);

			// reads may end early at the end of the file, but must cover the node
			if(bytesRead < ssize_t(read.requiredSize)) numFailed++;
		}
	};

	vector<jthread> threads;
	for(int i = 0; i < numThreads - 1; i++){
		threads.emplace_back(worker);
	}
	worker();

	// wait for all reads to finish
	threads.clear();

	return numFailed;
}

// Renders PotreeFileNodes. 
// - Traverses the octrees of all potree files from largest to smallest nodes in screen space, 
//   skipping nodes outside the view frustum, until the point budget is reached. 
// - Points are rendered in a coordinate system centered at the bounding box of each potree file.
// - Render paths (CuRastSettings::potreeRenderPath): 
//     - Memory-mapped: The kernel reads points from the memory-mapped octree.bin. 
//       Requires GPU access to pageable host memory (e.g. HMM on linux).
//     - Direct storage: Visible nodes are read from octree.bin into VRAM via cuFile each frame, 
//       without caching, and rendered from there.
void drawPotreeFiles(Scene* scene, View view, RenderTarget& target){

	u64 pointBudget = CuRastSettings::pointBudget;
	bool directStorage = CuRastSettings::potreeRenderPath == POTREE_DIRECT_STORAGE;

	if(directStorage){
		if(!initCuFile()) return;
	}else{
		if(!canAccessPageableMemory()) return;
	}

	vector<PotreeFileNode*> files;
	scene->forEach<PotreeFileNode>([&](PotreeFileNode* file){
		if(file->mapped_hierarchy == nullptr) return;
		if(file->hierarchyNodes.empty()) return;
		if(file->encoding != "DEFAULT") return;

		if(directStorage){
			if(file->getOctreeCuFileHandle() == nullptr) return;
		}else{
			if(file->mapped_octree == nullptr) return;
		}

		files.push_back(file);
	});

	if(files.empty()) return;

	if(directStorage){
		// The buffer holds the visible points, plus padding because reads are aligned to SSD pages. 
		// Padding adds less than 2 pages per node. We reserve it for pointBudget / 1000 nodes, i.e., 
		// nodes with 1000 points on average. If a frame needs more, the traversal stops early.
		i64 maxBytesPerPoint = 0;
		for(PotreeFileNode* file : files){
			maxBytesPerPoint = std::max(maxBytesPerPoint, file->bytesPerPoint);
		}

		u64 maxNodes = std::max<u64>(pointBudget / 1000, 1000);
		u64 requiredSize = pointBudget * maxBytesPerPoint + maxNodes * 2 * DIRECT_STORAGE_PAGE_SIZE;
		reserveDirectStorageBuffer(requiredSize);
	}

	struct Plane{
		dvec3 normal;
		double d;
	};

	// per potree file state for the traversal
	struct FileState{
		PotreeFileNode* file;
		CUfileHandle_t cuFileHandle;  // direct storage only
		dvec3 center;       // bounding box center, becomes the origin of the rendered coordinate system
		dmat4 worldView;
		Plane planes[4];    // frustum side planes, relative to center
		i64 offset_color;
	};

	vector<FileState> states;
	for(PotreeFileNode* file : files){
		FileState state;
		state.file = file;
		state.cuFileHandle = directStorage ? file->getOctreeCuFileHandle() : nullptr;
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

	// direct storage: reads of the visible nodes, and the number of bytes they occupy in directStorageBuffer
	static vector<DirectStorageRead> reads;
	reads.clear();
	u64 numReadBytes = 0;

	while(!queue.empty()){
		QueueItem item = queue.top();
		queue.pop();

		FileState& state = states[item.stateIndex];
		PotreeFileNode* file = state.file;

		// proxies only know their point count, not yet where their points are
		file->loadHierarchyChunk(item.nodeIndex);
		const PotreeHierarchyNode& node = file->hierarchyNodes[item.nodeIndex];

		if(numVisiblePoints + node.numPoints > pointBudget) break;

		PotreeNode visibleNode;

		if(directStorage){
			// read whole SSD pages. The node starts somewhere in the first page.
			u64 alignedStart = (node.byteOffset / DIRECT_STORAGE_PAGE_SIZE) * DIRECT_STORAGE_PAGE_SIZE;
			u64 alignedEnd   = ((node.byteOffset + node.byteSize + DIRECT_STORAGE_PAGE_SIZE - 1) / DIRECT_STORAGE_PAGE_SIZE) * DIRECT_STORAGE_PAGE_SIZE;
			u64 paddedSize   = alignedEnd - alignedStart;

			if(numReadBytes + paddedSize > directStorageBuffer.size) break;

			DirectStorageRead read;
			read.file         = state.cuFileHandle;
			read.fileOffset   = alignedStart;
			read.size         = paddedSize;
			read.bufferOffset = numReadBytes;
			read.requiredSize = node.byteOffset + node.byteSize - alignedStart;
			reads.push_back(read);

			visibleNode.data = directStorageBuffer.ptr + numReadBytes + (node.byteOffset - alignedStart);
			numReadBytes += paddedSize;
		}else{
			visibleNode.data = (u8*)file->mapped_octree + node.byteOffset;
		}

		numVisiblePoints += node.numPoints;

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

	if(directStorage && reads.size() > 0){
		// load the visible nodes as they are stored in octree.bin; the kernel decodes them
		double tStart = now();
		int numFailed = executeDirectStorageReads(reads);
		double milliseconds = (now() - tStart) * 1000.0;

		if(numFailed > 0){
			static bool reported = false;
			if(!reported) println("ERROR: {} of {} direct storage reads failed.", numFailed, reads.size());
			reported = true;
		}

		if(Runtime::measureTimings){
			Runtime::timings.add("cuFileRead (direct storage)", milliseconds);
		}

		auto& dvlist = Runtime::debugValueList;
		dvlist.push_back({"direct storage read", format("{:.1f} MB in {:.1f} ms ({:.1f} GB/s)", 
			double(numReadBytes) / 1'000'000.0, milliseconds, double(numReadBytes) / 1'000'000.0 / milliseconds)});
		dvlist.push_back({"direct storage buffer", format("{:.1f} MB", double(directStorageBuffer.size) / 1'000'000.0)});
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

		if(directStorage){
			launch_drawPotreeDirectStorageNodes(target, nodes, visibleNodes.size());
		}else{
			launch_drawPotreeFileNodes(target, nodes, visibleNodes.size());
		}
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

// What selectClustersBvh() visited and selected
struct BvhTraversal{
	u64 visitedNodes = 0;
	u64 visitedGroups = 0;      // leaves whose clusters were tested
	u64 testedClusters = 0;
	u64 visibleTriangles = 0;
	u64 visibleVertices = 0;
};

// Selects the clusters of the LOD cut for the current view by traversing the BVH over the groups (nodes.bin) on the CPU,
// and culls them against the view frustum. Same result as kernel_selectClusters, but only visits nodes and clusters
// near the cut instead of all clusters. See tools/clodbuilder/README.md.
// Nodes, groups and clusters are read from RAM or from the memory-mapped files, see ClusteredMeshNode::getBvhData().
BvhTraversal selectClustersBvh(
	const ClusteredMeshNode::BvhData& bvh, const ClusteredMesh& mesh, const RenderTarget& target,
	vector<u32>& visibleClusters
){
	// same as projectedError() in trianglesClustered.cu
	float pixelsPerError = target.proj[1][1] * 0.5f * float(target.height);
	auto projectedError = [&](vec4 sphere, float error){
		float distance = std::max(length(vec3(sphere) - mesh.cameraPosition) - sphere.w, mesh.znear);
		return error / distance * pixelsPerError;
	};

	auto isOutsideFrustum = [&](vec4 sphere){
		if(!mesh.frustumCulling) return false;

		for(vec4 plane : mesh.frustumPlanes){
			if(dot(vec3(plane), vec3(sphere)) + plane.w < -sphere.w) return true;
		}

		return false;
	};

	float threshold = mesh.lodErrorThreshold;
	BvhTraversal traversal;

	static vector<u32> stack;
	stack.clear();
	for(u32 level = 0; level < bvh.numLevels; level++){
		stack.push_back(level);
	}

	while(!stack.empty()){
		const ClusterBvhNode& bvhNode = bvh.nodes[stack.back()];
		stack.pop_back();
		traversal.visitedNodes++;

		// The subtree is detailed enough: its clusters are replaced by coarser ones from another subtree or level
		if(projectedError(bvhNode.sphere, bvhNode.error) <= threshold) continue;
		if(isOutsideFrustum(bvhNode.sphere)) continue;

		if(bvhNode.group < 0){
			for(u32 i = 0; i < bvhNode.childCount; i++){
				stack.push_back(bvhNode.childOffset + i);
			}
		}else{
			// the group's own error is too large, so render each of its clusters that is detailed enough
			const ClusterGroup& group = bvh.groups[bvhNode.group];
			traversal.visitedGroups++;

			for(u32 clusterIndex = group.clusterOffset; clusterIndex < group.clusterOffset + group.clusterCount; clusterIndex++){
				const Cluster& cluster = bvh.clusters[clusterIndex];
				traversal.testedClusters++;

				if(cluster.refinedGroup >= 0 && projectedError(cluster.lodSphere, cluster.lodError) > threshold) continue;
				if(isOutsideFrustum(cluster.cullSphere)) continue;

				visibleClusters.push_back(clusterIndex);
				traversal.visibleTriangles += cluster.triangleCount;
				traversal.visibleVertices += cluster.vertexCount;
			}
		}
	}

	return traversal;
}

// Bytes read from a memory-mapped file of clustered meshes in the current frame, shown in the overlay.
// - Accurate (acc): the sizes of the records and elements that are read, e.g. 112 bytes per cluster and 12 bytes per 
//   vertex position. Hardware transfers are larger, as memory is accessed in whole cache lines and pages.
// - Estimate (est): texture.dds, see ClusterCounters::textureTexels. 
struct MappedFileTraffic{
	string file;
	bool accurate;
	double bytes;
};

// Renders ClusteredMeshNodes.
// - Selects the clusters of the LOD cut for the current view, and culls them against the frustum (CuRastSettings::clusterSelection):
//     - BVH: selectClustersBvh() traverses the BVH over the groups on the CPU, and uploads the list of selected clusters.
//     - Per cluster: kernel_selectClusters tests every cluster on the GPU.
// - kernel_drawClusters rasterizes the selected clusters.
// - Render paths (CuRastSettings::clusterRenderPath):
//     - VRAM: Clusters, vertices, triangles and the BC7 texture are copied to VRAM on first use.
//       The BVH traversal reads nodes, groups and clusters from RAM.
//     - Memory-mapped: The kernels read them directly from the memory-mapped files, and so does the BVH traversal.
//       Requires GPU access to pageable host memory (e.g. HMM on linux). The overlay shows how much is read from each file.
void drawClusteredMeshes(Scene* scene, View view, RenderTarget& target){

	bool memoryMapped = CuRastSettings::clusterRenderPath == CLUSTERS_MEMORY_MAPPED;
	if(memoryMapped && !canAccessPageableMemory()) return;

	bool useBvh = CuRastSettings::clusterSelection == CLUSTER_SELECTION_BVH;

	u64 numVisibleClusters = 0;
	u64 numVisibleTriangles = 0;
	u64 numVisitedNodes = 0;
	double bvhMilliseconds = 0.0;
	bool hasClusteredMeshes = false;

	vector<MappedFileTraffic> traffic;
	auto addTraffic = [&](const ClusteredMeshNode::MappedFile& file, double bytes, bool accurate){
		for(MappedFileTraffic& entry : traffic){
			if(entry.file == file.name){
				entry.bytes += bytes;
				return;
			}
		}

		traffic.push_back({file.name, accurate, bytes});
	};

	scene->forEach<ClusteredMeshNode>([&](ClusteredMeshNode* node){

		// no-ops after their first call
		node->initGpu();
		if(!memoryMapped) node->uploadToVram();

		dmat4 worldView = view.view * node->transform_global;
		dmat4 worldViewProj = view.proj * worldView;

		ClusteredMesh mesh;
		if(memoryMapped){
			mesh.clusters  = (Cluster*)node->mapped_clusters.ptr;
			mesh.positions = (vec3*)node->mapped_positions.ptr;
			mesh.uvs       = (vec2*)node->mapped_uvs.ptr;
			mesh.triangles = (u8*)node->mapped_triangles.ptr;
		}else{
			mesh.clusters  = node->gpu_clusters;
			mesh.positions = node->gpu_positions;
			mesh.uvs       = node->gpu_uvs;
			mesh.triangles = node->gpu_triangles;
		}
		mesh.texture           = node->getTexture(memoryMapped);
		mesh.numClusters       = node->clusters.size();
		mesh.worldView         = mat4(worldView);
		mesh.cameraPosition    = vec3(inverse(worldView) * dvec4(0.0, 0.0, 0.0, 1.0));
		mesh.frustumCulling    = CuRastSettings::enableFrustumCulling;
		mesh.znear             = float(view.proj[3][2]); // the infinite projection stores the near plane distance here
		mesh.lodErrorThreshold = CuRastSettings::lodErrorThreshold;
		mesh.colorMode         = CuRastSettings::clusterColorMode;

		// frustum planes in the mesh's coordinate system
		dvec4 row0 = glm::row(worldViewProj, 0);
		dvec4 row1 = glm::row(worldViewProj, 1);
		dvec4 row2 = glm::row(worldViewProj, 2);
		dvec4 row3 = glm::row(worldViewProj, 3);

		auto normalizedPlane = [](dvec4 p){
			return vec4(p / glm::length(dvec3(p)));
		};

		mesh.frustumPlanes[0] = normalizedPlane(row3 + row0); // Left
		mesh.frustumPlanes[1] = normalizedPlane(row3 - row0); // Right
		mesh.frustumPlanes[2] = normalizedPlane(row3 + row1); // Bottom
		mesh.frustumPlanes[3] = normalizedPlane(row3 - row1); // Top
		mesh.frustumPlanes[4] = normalizedPlane(row3 - row2); // Near: clip.z is the near plane distance, i.e., w >= near

		BvhTraversal traversal;

		if(useBvh){
			double tStart = now();

			static vector<u32> visibleClusters;
			visibleClusters.clear();
			traversal = selectClustersBvh(node->getBvhData(memoryMapped), mesh, target, visibleClusters);

			// the draw kernel expects the same input as produced by kernel_selectClusters
			ClusterCounters counters = {u32(visibleClusters.size()), u32(traversal.visibleTriangles), u32(traversal.visibleVertices), 0.0f};
			cudaMemcpy(node->gpu_visibleClusters, visibleClusters.data(), byteSizeOf(visibleClusters), cudaMemcpyHostToDevice);
			cudaMemcpy(node->gpu_counters, &counters, sizeof(counters), cudaMemcpyHostToDevice);

			bvhMilliseconds += (now() - tStart) * 1000.0;
			numVisitedNodes += traversal.visitedNodes;
		}else{
			cudaMemsetAsync(node->gpu_counters, 0, sizeof(ClusterCounters));
			launch_selectClusters(target, mesh, node->gpu_visibleClusters, node->gpu_counters);
		}

		launch_drawClusters(target, mesh, node->gpu_visibleClusters, node->gpu_counters);

		ClusterCounters counters;
		cudaMemcpy(&counters, node->gpu_counters, sizeof(counters), cudaMemcpyDeviceToHost);
		numVisibleClusters += counters.numVisibleClusters;
		numVisibleTriangles += counters.numVisibleTriangles;

		if(memoryMapped){
			// cluster records: those tested by the BVH traversal on the CPU (or all, by kernel_selectClusters), 
			// plus the visible ones read by kernel_drawClusters
			u64 clusterRecords = (useBvh ? traversal.testedClusters : mesh.numClusters) + counters.numVisibleClusters;

			addTraffic(node->mapped_nodes, traversal.visitedNodes * sizeof(ClusterBvhNode), true);
			addTraffic(node->mapped_groups, traversal.visitedGroups * sizeof(ClusterGroup), true);
			addTraffic(node->mapped_clusters, clusterRecords * sizeof(Cluster), true);
			addTraffic(node->mapped_positions, counters.numVisibleVertices * sizeof(vec3), true);
			addTraffic(node->mapped_uvs, counters.numVisibleVertices * sizeof(vec2), true);
			addTraffic(node->mapped_triangles, counters.numVisibleTriangles * 3, true);

			// BC7: one byte per texel
			if(mesh.texture.data != nullptr){
				addTraffic(node->mapped_texture, counters.textureTexels, false);
			}
		}

		hasClusteredMeshes = true;
	});

	if(useBvh && hasClusteredMeshes && Runtime::measureTimings){
		Runtime::timings.add("cluster BVH traversal and upload (CPU)", bvhMilliseconds);
	}

	if(hasClusteredMeshes){
		auto& dvlist = Runtime::debugValueList;
		dvlist.push_back({"clusters", format("{:L}", numVisibleClusters)});
		dvlist.push_back({"triangles", format("{:L}", numVisibleTriangles)});
		if(useBvh) dvlist.push_back({"visited BVH nodes", format("{:L}", numVisitedNodes)});

		for(const MappedFileTraffic& entry : traffic){
			dvlist.push_back({entry.file, format("{:.1f} MB / frame ({})", entry.bytes / 1'000'000.0, entry.accurate ? "acc" : "est")});
		}
	}
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
	drawClusteredMeshes(scene, view, target);

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

