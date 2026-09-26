
#include "json/json.hpp"

#include "CuRast.h"
#include "VKRenderer.h"

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

	Timer::enabled = Runtime::measureTimings;

	launchArgs = getCommonLaunchArgs();

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

#include "CuRast_render.h"
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

