
#include "CuRast.h"
#include "VKRenderer.h"


void CuRast::setup(){
	CuRast::instance = new CuRast();
}

void CuRast::resetEditor(){
	scene.world->children.clear();
}


void CuRast::inputHandling(){
	
	auto editor = CuRast::instance;
	auto& scene = editor->scene;

	bool consumed = false;

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

