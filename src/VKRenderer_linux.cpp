#if defined(__linux__)

#include <vector>
#include <string>
#include <sstream>
#include <functional>
#include <print>
#include <cstdint>

// todo: dnd via x11 or wayland - probably somewhat tedious, maybe just wait for glfw pr?
//#define GLFW_EXPOSE_NATIVE_X11

#include "GLFW/glfw3.h"
//#include "GLFW/glfw3native.h"

using std::vector;
using std::string;
using std::function;
using std::println;

std::vector<std::function<void(std::string)>> dragEnterCallbacks;
std::vector<std::function<void(std::string)>> dragDropCallbacks;
std::vector<std::function<void(int32_t, int32_t)>> dragOverCallbacks;

void init(GLFWwindow* window){
	static bool initialized = false;

	if(!initialized){
		glfwSetDropCallback(window, [](GLFWwindow*, int count, const char **paths){
			std::vector<string> files;
			for(int i = 0; i < count; ++i) {
			string file = paths[i];
			for (auto &listener : dragDropCallbacks) {
				listener(file);
			}
			}
		});

		// todo: x11 and/or wayland drag enter / drag move/position handlers
		/*
			* for x11 callbacks need to be handled via the display, I think? I have no experience with this
		auto displayHandle = glfwGetX11Display();
		XdndEnter = XInternAtom(displayHandle, "XdndEnter", false);
		XdndPosition= XInternAtom(displayHandle, "XdndPosition", false);

		auto windowHandle = glfwGetX11Window(window);
		auto windowHandle = glfwGetWaylandWindow(window);
		*/

		initialized = true;
	}
}


void installDragEnterHandler(GLFWwindow* window, function<void(string)> callback) {
	println("WARN: drag enter handler not available on this plattform.");
	init(window);
	dragEnterCallbacks.push_back(callback);
}

void installDragDropHandler(GLFWwindow* window, function<void(string)> callback) {
	init(window);
	dragDropCallbacks.push_back(callback);
}

void installDragOverHandler(GLFWwindow* window, function<void(int32_t, int32_t)> callback) {
	println("WARN: drag over handler not available on this plattform.");
	init(window);
	dragOverCallbacks.push_back(callback);
}

#endif