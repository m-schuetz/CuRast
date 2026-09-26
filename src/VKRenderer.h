
#pragma once

#include <functional>
#include <vector>
#include <string>
#include <span>
#include <array>

#include <vulkan/vulkan.h>

// ---------------------------------------------------------------------------
// Extension function pointers for Vulkan EXT/KHR extensions that are not
// exported by the Vulkan loader library (core functions are loaded automatically).
//
// Call loadVkExt(instance, device) once after device creation.
// Macros below shadow the official function names so call sites are unchanged.
// ---------------------------------------------------------------------------

// VK_KHR_external_memory_fd
inline PFN_vkGetMemoryFdKHR                g_vkGetMemoryFdKHR;

// VK_EXT_debug_utils
inline PFN_vkCreateDebugUtilsMessengerEXT  g_vkCreateDebugUtilsMessengerEXT;
inline PFN_vkDestroyDebugUtilsMessengerEXT g_vkDestroyDebugUtilsMessengerEXT;

#define vkGetMemoryFdKHR                g_vkGetMemoryFdKHR
#define vkCreateDebugUtilsMessengerEXT  g_vkCreateDebugUtilsMessengerEXT
#define vkDestroyDebugUtilsMessengerEXT g_vkDestroyDebugUtilsMessengerEXT

inline void loadVkExt(VkInstance instance, VkDevice device) {
#define LOAD_DEV(fn) g_##fn = (PFN_##fn)vkGetDeviceProcAddr(device, #fn)
#define LOAD_INST(fn) g_##fn = (PFN_##fn)vkGetInstanceProcAddr(instance, #fn)

    LOAD_DEV(vkGetMemoryFdKHR);

    LOAD_INST(vkCreateDebugUtilsMessengerEXT);
    LOAD_INST(vkDestroyDebugUtilsMessengerEXT);

#undef LOAD_DEV
#undef LOAD_INST
}

#include "GLFW/glfw3.h"

#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_vulkan.h"
#include "imgui_impl_glfw.h"

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

#include "unsuck.hpp"
#include "OrbitControls.h"

#include "cuda.h"

using glm::dvec3;
using glm::dvec4;
using glm::vec3;
using glm::vec4;
using glm::dmat4;

struct VKRenderer;

// Replaces GLTexture — a Vulkan image with CUDA external memory interop
struct VKTexture {
	VkImage        image  = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	VkImageView    view   = VK_NULL_HANDLE;
	VkFormat       format = VK_FORMAT_R8G8B8A8_UNORM;

	int     width  = 0;
	int     height = 0;
	int64_t version = 0;
	int64_t ID      = 0;
	inline static int64_t idcounter = 0;
	std::string label;

	// CUDA external memory interop handles — populated by importToCuda()
	CUexternalMemory cudaExtMem   = nullptr;
	CUmipmappedArray cudaMipArray = nullptr;
	CUsurfObject     cudaSurface  = 0;

	void setSize(int w, int h);
	void importToCuda();
	void destroyCuda();
	void destroy();
};

// Replaces Framebuffer — with dynamic rendering no VkRenderPass/VkFramebuffer needed
struct VKFramebuffer {
	std::shared_ptr<VKTexture> colorAttachment;
	int     width   = 0;
	int     height  = 0;
	int64_t version = 0;
	std::string label;

	void setSize(int w, int h);
	static std::shared_ptr<VKFramebuffer> create(const std::string& label);
};

struct View {
	dmat4 view;
	dmat4 proj;
	std::shared_ptr<VKFramebuffer> framebuffer = nullptr;
};

struct Camera {
	glm::dvec3 position;
	glm::dmat4 rotation;

	glm::dmat4 world;
	glm::dmat4 view;
	glm::dmat4 proj;

	double aspect = 1.0;
	double fovy   = 60.0;
	double near_  = 0.01;
	double far_   = 2'000'000.0;
	int    width  = 128;
	int    height = 128;

	Camera() {}

	void setSize(int width, int height) {
		this->width  = width;
		this->height = height;
		this->aspect = double(width) / double(height);
	}

	void update() {
		view = glm::inverse(world);

		float pi = glm::pi<float>();
		proj = Camera::createProjectionMatrix(float(near_), float(pi * fovy / 180.0), float(aspect));
	}

	vec3 getRayDir(float u, float v) {
		vec3 origin = getPosition();

		float right = float(1.0 / proj[0][0]);
		float up    = float(1.0 / proj[1][1]);
		vec4  dir_00_worldspace = inverse(view) * vec4(-right, -up,   -1.0f, 1.0f);
		vec4  dir_01_worldspace = inverse(view) * vec4(-right,  up,   -1.0f, 1.0f);
		vec4  dir_10_worldspace = inverse(view) * vec4( right, -up,   -1.0f, 1.0f);
		vec4  dir_11_worldspace = inverse(view) * vec4( right,  up,   -1.0f, 1.0f);

		auto getRayDir_ = [&](float u_, float v_) {
			float A_00 = (1.0f - u_) * (1.0f - v_);
			float A_01 = (1.0f - u_) *          v_;
			float A_10 =          u_ * (1.0f - v_);
			float A_11 =          u_ *          v_;
			vec3  dir  = (A_00 * dir_00_worldspace + A_01 * dir_01_worldspace +
			              A_10 * dir_10_worldspace + A_11 * dir_11_worldspace -
			              vec4(origin, 1.0));
			return normalize(dir);
		};
		return getRayDir_(u, v);
	}

	vec3 getPosition() {
		return dvec3(inverse(view) * dvec4(0.0, 0.0, 0.0, 1.0));
	}

	inline static glm::mat4 createProjectionMatrix(float near_, float fovy, float aspect) {
		float f = 1.0f / tan(fovy / 2.0f);
		return glm::mat4(
			f / aspect, 0.0f, 0.0f,  0.0f,
			0.0f,       f,    0.0f,  0.0f,
			0.0f,       0.0f, 0.0f, -1.0f,
			0.0f,       0.0f, near_, 0.0f);
	}
};

struct VKRenderer {
	inline static GLFWwindow*  window            = nullptr;
	inline static double       fps               = 0.0;
	inline static double       timeSinceLastFrame = 0.0;
	inline static int64_t      frameCount        = 0;

	inline static std::shared_ptr<Camera> camera = nullptr;
	inline static View view;

	inline static int         width          = 0;
	inline static int         height         = 0;

	inline static std::vector<std::function<void(std::vector<std::string>)>> fileDropListeners;

	// ---- Vulkan core objects ----
	inline static VkInstance               instance       = VK_NULL_HANDLE;
	inline static VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
	inline static VkSurfaceKHR             surface        = VK_NULL_HANDLE;
	inline static VkPhysicalDevice         physDevice   = VK_NULL_HANDLE;
	inline static VkDevice                 device       = VK_NULL_HANDLE;
	inline static VkQueue                  graphicsQueue = VK_NULL_HANDLE;
	inline static uint32_t                 graphicsQueueFamily = 0;

	// ---- Swapchain ----
	inline static VkSwapchainKHR              swapchain = VK_NULL_HANDLE;
	inline static VkFormat                    swapchainFormat;
	inline static VkExtent2D                  swapchainExtent;
	inline static std::vector<VkImage>        swapchainImages;
	inline static std::vector<VkImageView>    swapchainImageViews;

	// ---- ImGui descriptor pool ----
	inline static VkDescriptorPool imguiDescriptorPool = VK_NULL_HANDLE;

	// ---- Per-frame sync + commands ----
	static constexpr int FRAMES_IN_FLIGHT = 3;
	inline static std::vector<VkCommandPool>   commandPools;
	inline static std::vector<VkCommandBuffer> commandBuffers;
	inline static std::vector<VkSemaphore>     imageAvailableSemaphores;
	inline static std::vector<VkSemaphore>     renderFinishedSemaphores;
	inline static std::vector<VkFence>         inFlightFences;
	inline static int currentFrame = 0;

	// ---- Public API ----
	static void init();
	static void destroy();

	static void loop(
		std::function<void(void)> update,
		std::function<void(void)> render,
		std::function<void(void)> postFrame);

	inline static void onFileDrop(std::function<void(std::vector<std::string>)> callback) {
		fileDropListeners.push_back(callback);
	}

	static void assertSucces(VkResult result, stacktrace trace = stacktrace::current()){

		if(result == VK_SUCCESS) return;

		println("ERROR (Vulkan): {}", int(result));
		println("{}", trace);

		__debugbreak();
	}

private:
	VKRenderer() = delete;

	static void createInstance();
	static void createSurface();
	static void pickPhysicalDevice();
	static void createLogicalDevice();
	static void createSwapchain();
	static void createSwapchainImageViews();
	static void createCommandObjects();
	static void createSyncObjects();
	static void initImGui();
	static void recreateSwapchain();
	static void cleanupSwapchain();

	static void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex);

public:
	static uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
	static VkCommandBuffer beginSingleTimeCommands();
	static void endSingleTimeCommands(VkCommandBuffer cmd);
	static void transitionImageLayout(VkImage image,
		VkImageLayout oldLayout, VkImageLayout newLayout);
};

