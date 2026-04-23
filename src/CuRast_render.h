
#include <unordered_set>
#include <execution>
#include "jpeg/JpegTextures.h"

#include "Timer.h"
#include "VKRenderer.h"
#include "TextureManager.h"

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
	uint32_t numMeshes,
	uint32_t* numProcessedBatches
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

	uint64_t numPixels = target.width * target.height;
	CUdeviceptr cptr_screenshot = MemoryManager::alloc(numPixels * 4, "screenshot");

	uint32_t backgroundColor = 0;
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

void drawTrianglesVisbuffer(
	Scene* scene, View view, vector<CMesh>& meshes, 
	vector<CMesh>& instances, CUdeviceptr cptr_meshes,
	CUdeviceptr cptr_instances, CUdeviceptr cptr_transforms, CUdeviceptr cptr_triangleCountPrefixsum,
	RenderTarget& target, MappedTextures& mappings
){
	auto editor = CuRast::instance;

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
	args.numProcessedBatches             = (uint32_t*)cptr_numProcessedBatches;
	args.numProcessedBatches_nontrivial  = (uint32_t*)cptr_numProcessedBatches_nontrivial;
	args.hugeTriangles                   = (HugeTriangle*)cptr_hugeTriangles;
	args.hugeTrianglesCounter            = (uint32_t*)cptr_hugeTrianglesCounter;
	args.numProcessedHugeTriangles       = (uint32_t*)cptr_numProcessedHugeTriangles;
	args.nontrivialTrianglesCounter      = (uint32_t*)cptr_nontrivialCounter;
	args.nontrivialTrianglesList         = (uint64_t*)cptr_nontrivialList;
	args.target                          = target;
	
	prog->launchCooperative(strKernelStage1, vector<void*>{&args}, {.blocksize = TRIANGLES_PER_SWEEP});
	prog->launchCooperative(strKernelStage2, vector<void*>{&args});
	prog->launchCooperative(strKernelStage3, vector<void*>{&args}, {.blocksize = 64});

	auto cuend = Timer::recordCudaTimestamp();
	Timer::recordDuration("<triangles visbuffer pipeline>", custart, cuend);
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
	target.framebuffer = (uint64_t*)cvm_framebuffer->cptr;
	target.colorbuffer = (uint64_t*)cvm_colorbuffer->cptr;
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

	uint64_t numTotalTriangles = 0;
	uint64_t numTotalNodes = 0;
	uint64_t numVisibleTriangles = 0;
	uint64_t numVisibleNodes = 0;
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
			return uint64_t(a->mesh) < uint64_t(b->mesh);
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

			uint32_t indexRange = node->mesh->index_max - node->mesh->index_min;
			uint64_t bitsPerIndex = ceil(log2f(float(indexRange + 1)));

			CMesh mesh;
			mesh.world                    = node->transform_global;
			mesh.positions                = (vec3*)node->mesh->cptr_position;
			mesh.uvs                      = (vec2*)node->mesh->cptr_uv;
			mesh.colors                   = (uint32_t*)node->mesh->cptr_color;
			mesh.normals                  = (vec3*)node->mesh->cptr_normal;
			mesh.indices                  = (uint32_t*)node->mesh->cptr_indices;
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
			mesh.address                  = uint64_t(node->mesh);

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
		static vector<uint64_t> triangleCountPrefixsum;
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
		uint64_t uniqueMeshCounter = 0;
		for(int i = 0; i < nodes.size(); i++){
			CMesh& cmesh = meshes_allInstances[i];

			cmesh.cummulativeTriangleCount = sum;
			cmesh.instances.offset = i;
			// meshes_allInstances[i].instances.count = 1; // should only matter for the unique meshes entries

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

		// prep virtual memory for lots of nodes
		static CudaVirtualMemory* cvm_meshes                    = MemoryManager::allocVirtualCuda(1'000'000 * sizeof(CMesh), "cvm_meshes");
		static CudaVirtualMemory* cvm_instances                 = MemoryManager::allocVirtualCuda(1'000'000 * sizeof(CMesh), "cvm_instances");
		static CudaVirtualMemory* cvm_transforms                = MemoryManager::allocVirtualCuda(1'000'000 * sizeof(mat4), "cvm_transforms");
		static CudaVirtualMemory* cvm_triangleCountPrefixsum    = MemoryManager::allocVirtualCuda(1'000'000 * sizeof(uint64_t), "cvm_triangleCountPrefixsum");
		
		// commit physical memory for actual amount of nodes
		cvm_meshes                 ->commit(meshes_unique.size()          * sizeof(CMesh));
		cvm_instances              ->commit(meshes_allInstances.size()    * sizeof(CMesh));
		cvm_transforms             ->commit(transforms.size()             * sizeof(mat4));
		cvm_triangleCountPrefixsum ->commit(triangleCountPrefixsum.size() * sizeof(uint64_t));

		// submit per-frame geometry metadata to GPU
		cuMemcpyHtoDAsync(cvm_meshes->cptr                 , meshes_unique.data(),          byteSizeOf(meshes_unique), 0);
		cuMemcpyHtoDAsync(cvm_instances->cptr              , meshes_allInstances.data(),    byteSizeOf(meshes_allInstances), 0);
		cuMemcpyHtoDAsync(cvm_transforms->cptr             , transforms.data(),             byteSizeOf(transforms), 0);
		cuMemcpyHtoDAsync(cvm_triangleCountPrefixsum->cptr , triangleCountPrefixsum.data(), byteSizeOf(triangleCountPrefixsum), 0);

		Runtime::numVisibleNodes = numVisibleNodes;
		Runtime::numVisibleTriangles = numVisibleTriangles;
		Runtime::numNodes = numTotalNodes;
		Runtime::numTriangles = numTotalTriangles;

		auto& dvlist = Runtime::debugValueList;
		dvlist.push_back({"#total nodes           ", format("{:40L}", uint64_t(numTotalNodes))});
		dvlist.push_back({"#total triangles       ", format("{:40L}", uint64_t(numTotalTriangles))});
		dvlist.push_back({"#visible nodes         ", format("{:40L}", uint64_t(numVisibleNodes))});
		dvlist.push_back({"#visible triangles     ", format("{:40L}", uint64_t(numVisibleTriangles))});
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
			uint32_t clearColor = 0xff000000;
			float clearDepth = Infinity;

			uint64_t requiredBytes = numPixels * 8;
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
			scene, view, meshes_unique, meshes_allInstances, 
			cvm_meshes->cptr, 
			cvm_instances->cptr, cvm_transforms->cptr, cvm_triangleCountPrefixsum->cptr,
			target, mappings
		);
		

		// DRAW BOUNDING BOXES
		if(CuRastSettings::showBoundingBoxes){
			RenderTarget target_lines = target;
			target_lines.framebuffer = (uint64_t*)cvm_colorbuffer->cptr;

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
			
			uint32_t numMeshes = boundingBoxNodes.size();
			launch_drawBoundingBoxes(
				target_lines, 
				(CMesh*)cvm_meshes_boxes->cptr,
				numMeshes,
				(uint32_t*)cptr_numProcessedBatches
			);
		}

		int mouse_X = Runtime::mousePosition.x;
		int mouse_Y = target.height - Runtime::mousePosition.y;
		uint32_t numInstances = meshes_allInstances.size();

		RasterizationSettings rasterSettings;
		rasterSettings.showWireframe = CuRastSettings::showWireframe;
		rasterSettings.enableDiffuseLighting = CuRastSettings::enableDiffuseLighting;
		rasterSettings.displayAttribute = CuRastSettings::displayAttribute;
		rasterSettings.enableObjectPicking = CuRastSettings::enableObjectPicking;

		JpegPipeline jpp;
		jpp.toDecode             = (uint32_t*)jpegTextures->cptr_toDecode;
		jpp.toDecodeCounter      = (uint32_t*)jpegTextures->cptr_toDecodeCounter;
		jpp.decoded              = (uint32_t*)jpegTextures->cptr_decoded;
		jpp.TBSlots              = (uint32_t*)jpegTextures->cptr_TBSlots;
		jpp.TBSlotsCounter       = (uint32_t*)jpegTextures->cptr_TBSlotsCounter;
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

		if(hasJpegCompressedTextures){
			uint32_t toDecodeCounter;
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
				uint32_t C = CuRast::deviceState->dbg_hovered_decoded_color;
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
			// 	uint32_t capacity = JPEG_NUM_DECODED_MCU_CAPACITY;
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

			uint32_t backgroundColor = 0;
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
				&backgroundColor
			};
			prog->launch2D("kernel_resolve_colorbuffer_to_opengl_2D", args, target.width, target.height);
		}

		if(CuRastSettings::requestScreenshot){
			saveScreenshot(target, view, cvm_ssaoShadebuffer->cptr, prog);
		}

		unmapCudaVk(mappings);
		
		cuMemcpyDtoHAsync((void*)deviceState, cptr_state, sizeof(DeviceState), 0);
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