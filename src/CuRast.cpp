
#include "json/json.hpp"

#include "CuRast.h"
#include "VKRenderer.h"
#include "TextureManager.h"

using json = nlohmann::json;

void CuRast::setup(){
	CuRast::instance = new CuRast();
	CuRast* editor = CuRast::instance;

	editor->initCudaProgram();
}

void CuRast::resetEditor(){
	scene.world->children.clear();
}


Uniforms CuRast::getUniforms(){
	Uniforms uniforms;
	uniforms.time            = now();
	uniforms.frameCount      = VKRenderer::frameCount;
	//uniforms.measure         = Runtime::measureTimings;

	uniforms.inset.show      = CuRastSettings::showInset;
	uniforms.inset.start     = {16 * 60, 16 * 50};
	uniforms.inset.size      = {16, 16};

	glm::mat4 world(1.0f);
	glm::mat4 view           = VKRenderer::camera->view;
	glm::mat4 camWorld       = VKRenderer::camera->world;
	glm::mat4 proj           = VKRenderer::camera->proj;

	uniforms.world           = world;
	uniforms.camWorld        = camWorld;

	return uniforms;
}

CommonLaunchArgs CuRast::getCommonLaunchArgs(){

	CommonLaunchArgs launchArgs;
	launchArgs.uniforms       = getUniforms();
	launchArgs.state          = (DeviceState*)cptr_state;
	
	return launchArgs;
};

void CuRast::initCudaProgram(){
	cuMemAllocHost((void**)&deviceState , sizeof(DeviceState));
	cptr_state = MemoryManager::alloc(sizeof(DeviceState), "device state");
	cuMemsetD8(cptr_state, 0, sizeof(DeviceState));
}

void CuRast::inputHandling(){
	
	auto editor = CuRast::instance;
	auto& scene = editor->scene;
	auto& launchArgs = editor->launchArgs;

	bool consumed = false;

	RenderTarget target;
	target.width = VKRenderer::width;
	target.height = VKRenderer::height;
	target.view = mat4(VKRenderer::camera->view); // * scene.transform;
	target.proj = VKRenderer::camera->proj;

	bool isCtrlDown        = Runtime::keyStates[341] != 0;
	bool isAltDown         = Runtime::keyStates[342] != 0;
	bool isShiftDown       = Runtime::keyStates[340] != 0;
	bool isLeftClicked     = Runtime::mouseEvents.button == 0 && Runtime::mouseEvents.action == 1;
	static bool isLeftDown = false;
	bool isRightClicked    = false; // right click event: press and release without move

	static struct {
		vec2 startPos;
		bool isRightDown = false;
		bool hasMoved = false;
	} rightDownState;

	if(!rightDownState.isRightDown && Runtime::mouseEvents.isRightDown){
		// right mouse just pressed
		rightDownState.startPos = {Runtime::mouseEvents.pos_x, Runtime::mouseEvents.pos_y};
		rightDownState.hasMoved = false;
		rightDownState.isRightDown = true;
	}else if(rightDownState.isRightDown && Runtime::mouseEvents.isRightDown){
		// right mouse still pressed
		if(rightDownState.startPos.x != Runtime::mouseEvents.pos_x || rightDownState.startPos.y != Runtime::mouseEvents.pos_y){
			rightDownState.hasMoved = true;
		}
	}else if(rightDownState.isRightDown && !Runtime::mouseEvents.isRightDown){
		// right mouse just released
		rightDownState.isRightDown = false;

		isRightClicked = rightDownState.hasMoved == false;
	}

	if(Runtime::mouseEvents.isLeftDownEvent()) isLeftDown = true;
	if(Runtime::mouseEvents.isLeftUpEvent()) isLeftDown = false;

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

	Timer::enabled = Runtime::measureTimings || Benchmarking::measurementCountdown >= 0;

	launchArgs = getCommonLaunchArgs();

	scene.updateTransformations();
	inputHandling();
	// scene.updateTransformations();
};

void CuRast::postFrame(){
	// some special benchmarking stuff

	if(Benchmarking::measurementCountdown == 0){
		// Write result to benchmarking file

		auto& entries = Runtime::timings.entries;
		
		float mean_raster = Runtime::timings.getMean("<triangles visbuffer pipeline>");
		float mean_resolve = Runtime::timings.getMean("kernel_resolve_visbuffer_to_colorbuffer2D");
		float mean_vulkan_drawIndexed = Runtime::timings.getMean("vulkan drawIndexed");
		float mean_vulkan_draw = Runtime::timings.getMean("vulkan draw");
		float mean_vulkan_pass1 = Runtime::timings.getMean("visbuffer pass1");
		float mean_vulkan_pass2_resolve = Runtime::timings.getMean("visbuffer pass2 resolve");

		string method = "undefined";
		if(CuRastSettings::rasterizer == RASTERIZER_VISBUFFER_INSTANCED){
			method = "CUDA_VISBUFFER_INSTANCED";
		}else if(CuRastSettings::rasterizer == RASTERIZER_VISBUFFER_INDEXED){
			method = "CUDA_VISBUFFER";
		}else if(CuRastSettings::rasterizer == RASTERIZER_VULKAN_INDEXED_DRAW){
			method = "VULKAN_INDEXED_DRAW";
		}else if(CuRastSettings::rasterizer == RASTERIZER_VULKAN_INDEXPULLING_VISBUFFER){
			method = "VULKAN_INDEXPULLING_VISBUFFER";
		}else if(CuRastSettings::rasterizer == RASTERIZER_VULKAN_INDEXPULLING_INSTANCED){
			method = "VULKAN_INDEXPULLING_INSTANCED";
		}

		#ifdef USE_VULKAN_SHARED_MEMORY
			string memory = "VULKAN_MEMORY";
		#else
			string memory = "CUDA_MEMORY";
		#endif

		string sceneName = scene.world->children[0]->name;

		bool usesJpegTextures = false;
		for(int i = 0; i < TextureManager::numTextures; i++){
			Texture* texture = &TextureManager::textures[i];
			bool isJpeg = texture->huffmanTables != nullptr;

			if(isJpeg){
				usesJpegTextures = true;
				break;
			}
		}

		bool hasCompressedGeometry = false;
		scene.forEach<SNTriangles>([&](SNTriangles* node){
			if(node->mesh->compressed){
				hasCompressedGeometry = true;
			}
		});

		CUdevice device;
		cuCtxGetDevice(&device);
		char deviceName[256];
		cuDeviceGetName(deviceName, 256, device);
		string strDeviceName = deviceName;

		auto now = std::chrono::system_clock::now();
		auto local = std::chrono::zoned_time{std::chrono::current_zone(), now};
		string datetime = std::format("{:%Y-%m-%d %H:%M:%S}", local);



		println("=======================================");
		println("scene                   {}", sceneName);
		println("memory                  {}", memory);
		println("method                  {}", method);
		println("device                  {}", strDeviceName);
		println("datetime                {}", datetime);
		println("usesJpegTextures        {}", usesJpegTextures);
		println("hasCompressedGeometry   {}", hasCompressedGeometry);
		println("duration_raster(mean)   {}", mean_raster);
		println("duration_resolve(mean)  {}", mean_resolve);
		println("=======================================");

		string dir_benchmarks = "./benchmarks";
		fs::create_directories(dir_benchmarks);

		auto scenario = Benchmarking::active_scenario;

		string activeView = "";
		if(
			Runtime::controls->yaw    == scenario->view_closeup.yaw &&
			Runtime::controls->pitch  == scenario->view_closeup.pitch &&
			Runtime::controls->radius == scenario->view_closeup.radius &&
			length(Runtime::controls->target - dvec3(scenario->view_closeup.target)) == 0.0
		){
			activeView = "closeup";
		}
		if(
			Runtime::controls->yaw    == scenario->view_overview.yaw &&
			Runtime::controls->pitch  == scenario->view_overview.pitch &&
			Runtime::controls->radius == scenario->view_overview.radius &&
			length(Runtime::controls->target - dvec3(scenario->view_overview.target)) == 0.0
		){
			activeView = "overview";
		}

		// auto getDuration = [&](string label) -> json {
		// 	return {label, Runtime::timings.getMean(label)};
		// };

		json j_durations;

		for(auto [label, times] : Runtime::timings.entries){
			float mean = Runtime::timings.getMean(label);

			if(mean > 0.0f){
				// j_durations.push_back({{label, mean}});
				j_durations[label] = mean;
			}
		}

		json j = {
			{"scene",                 scenario->label},
			{"path",                  scenario->path},
			{"memory",                memory},
			{"method",                method},
			{"device",                strDeviceName},
			{"datetime",              datetime},
			{"usesJpegTextures",      usesJpegTextures},
			{"imageDivisionFactor",   scenario->imageDivisionFactor},
			{"hasCompressedGeometry", hasCompressedGeometry},
			{"instances", {
				{"count", {scenario->instances_count.x, scenario->instances_count.y}},
				{"spacing", {scenario->instances_spacing.x, scenario->instances_spacing.y}},
			}},
			{"durations", j_durations},
			{"resolution", {
				{"width", VKRenderer::view.framebuffer->width},
				{"height", VKRenderer::view.framebuffer->height},
			}},
			{"supersamplingFactor",  CuRastSettings::supersamplingFactor},
			{"activeView",           activeView},
			{"#visibleNodes",        Runtime::numVisibleNodes},
			{"#visibleTriangles",    Runtime::numVisibleTriangles},
			{"#nodes",               Runtime::numNodes},
			{"#triangles",           Runtime::numTriangles},
		};

		string strJson = j.dump(4);

		string strKey = scenario->label;
		if(scenario->compress) strKey += "_compressed";
		if(scenario->useJpegTextures) strKey += "_jpeg";
		if(scenario->path.contains("optimized")) strKey += "_optimized";
		if(scenario->imageDivisionFactor != 1) strKey += format("_texDivider{}", scenario->imageDivisionFactor);
		strKey += "_" + strDeviceName;
		strKey += "_" + method;
		strKey += "_" + activeView;
		string path = format("{}/benchmark_{}.json", dir_benchmarks, strKey);
		writeFile(path, strJson);

		string path_screenshot = format("{}/benchmark_{}.png", dir_benchmarks, strKey);
		CuRastSettings::requestScreenshot = make_shared<string>(path_screenshot);

	}

	Benchmarking::measurementCountdown--;
	if(Benchmarking::measurementCountdown < 0) Benchmarking::measurementCountdown = -1;


	

	auto editor = CuRast::instance;
}

void alignRight(string text) {
	float rightBorder = ImGui::GetCursorPosX() + ImGui::GetColumnWidth();
	float width = ImGui::CalcTextSize(text.c_str()).x;
	ImGui::SetCursorPosX(rightBorder - width);
}

#include "CuRast_render.h"
#include "gui/menubar.h"
#include "gui/toolbar.h"
#include "gui/widget_kernels.h"
#include "gui/widget_memory.h"
#include "gui/widget_benchmarking.h"
#include "gui/widget_timings.h"
#include "gui/stats.h"

void CuRast::makeDevGUI(){
	makeKernels();
	makeMemory();
	makeTimings();
	makeBenchmarking();
}

