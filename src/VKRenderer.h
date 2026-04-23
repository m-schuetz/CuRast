
#pragma once

#include <functional>
#include <vector>
#include <string>
#include <span>
#include <array>

#include <vulkan/vulkan.h>
#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#endif

#include "VkExt.h"

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
		proj = Camera::createProjectionMatrix(near_, pi * fovy / 180.0, aspect);
	}

	vec3 getRayDir(float u, float v) {
		vec3 origin = getPosition();

		float right = 1.0f / proj[0][0];
		float up    = 1.0f / proj[1][1];
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

	// Optional Vulkan mesh draw callback — set each frame by drawVulkan().
	// Called inside recordCommandBuffer() with the colorAttachment in
	// COLOR_ATTACHMENT_OPTIMAL layout inside an active dynamic render pass.
	// Set to nullptr to use the default CUDA→blit path.
	inline static std::function<void(VkCommandBuffer)> vulkanMeshDrawFn;

	// Cleanup callback set by drawVulkan(); called by destroy() before vkDestroyDevice
	// to release all static Vulkan resources and mesh buffers held by drawVulkan().
	inline static std::function<void()> vulkanMeshCleanupFn;

	// Depth buffer for the Vulkan mesh rasterizer pass.
	// Managed by drawVulkan(); consumed by recordCommandBuffer() for the barrier + attachment.
	inline static VkImage     vulkanMeshDepthImage = VK_NULL_HANDLE;
	inline static VkImageView vulkanMeshDepthView  = VK_NULL_HANDLE;

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

struct VKBuffer {

	string label = "none";
	VkBuffer        buffer        = VK_NULL_HANDLE;
	VkDeviceMemory  memory        = VK_NULL_HANDLE;
	VkDeviceAddress deviceAddress = 0;
	void* mapped = nullptr;
	uint64_t size = 0;
	bool isMapped = false;

	VkBufferUsageFlags usageFlags;
	VkMemoryPropertyFlags memoryPropertyFlags;

	VKBuffer(VkDeviceSize size, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags) {
		
		this->usageFlags = usageFlags;
		this->memoryPropertyFlags = memoryPropertyFlags;
		this->size = size;
		this->isMapped = false;

		resize(size);
	}

	~VKBuffer() {
		if (buffer != VK_NULL_HANDLE) vkDestroyBuffer(VKRenderer::device, buffer, nullptr);
		if (memory != VK_NULL_HANDLE) vkFreeMemory(VKRenderer::device, memory, nullptr);
	}

	void destroy(){
		if(buffer != VK_NULL_HANDLE){
			if (buffer != VK_NULL_HANDLE) vkDestroyBuffer(VKRenderer::device, buffer, nullptr);
			if (memory != VK_NULL_HANDLE) vkFreeMemory(VKRenderer::device, memory, nullptr);

			buffer = VK_NULL_HANDLE;
			memory = VK_NULL_HANDLE;
			size = 0;
			deviceAddress = 0;
			mapped = nullptr;
			isMapped = false;
		}
	}

	void map(){
		if(isMapped) return;
		VkResult result = vkMapMemory(VKRenderer::device, memory, 0, size, 0, &mapped);
		VKRenderer::assertSucces(result);

		isMapped = true;
	}

	void resize(uint64_t size){

		destroy();

		VkBufferCreateInfo bci = {};
		bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bci.size        = size;
		bci.usage       = usageFlags | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
		bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		auto res = vkCreateBuffer(VKRenderer::device, &bci, nullptr, &buffer);
		VKRenderer::assertSucces(res);

		VkMemoryRequirements memReqs;
		vkGetBufferMemoryRequirements(VKRenderer::device, buffer, &memReqs);

		VkMemoryAllocateFlagsInfo flagsInfo = {};
		flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
		flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.pNext           = &flagsInfo;
		allocInfo.allocationSize  = memReqs.size;
		allocInfo.memoryTypeIndex = VKRenderer::findMemoryType(memReqs.memoryTypeBits, memoryPropertyFlags);
		res = vkAllocateMemory(VKRenderer::device, &allocInfo, nullptr, &memory);
		VKRenderer::assertSucces(res);
		res = vkBindBufferMemory(VKRenderer::device, buffer, memory, 0);
		VKRenderer::assertSucces(res);

		VkBufferDeviceAddressInfo addrInfo = {};
		addrInfo.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
		addrInfo.buffer = buffer;
		deviceAddress = vkGetBufferDeviceAddress(VKRenderer::device, &addrInfo);

		if((memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0){
			map();
		}
	}

	
};


struct VKShader {
	using Stage = std::tuple<const VkShaderStageFlagBits, const std::span<uint32_t const>, std::string_view>;

	VKShader(const std::vector<Stage>& shaderStages,
	         const std::vector<VkPushConstantRange>& pcRanges,
	         const std::vector<VkDescriptorSetLayout>& setLayouts = {})
		: shaders(shaderStages.size(), VK_NULL_HANDLE)
		, stages(shaderStages.size())
	{
		VkPipelineLayoutCreateInfo layoutCI = {};
		layoutCI.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutCI.pushConstantRangeCount = (uint32_t)pcRanges.size();
		layoutCI.pPushConstantRanges    = pcRanges.data();
		layoutCI.setLayoutCount         = (uint32_t)setLayouts.size();
		layoutCI.pSetLayouts            = setLayouts.data();
		vkCreatePipelineLayout(VKRenderer::device, &layoutCI, nullptr, &layout);

		std::vector<VkShaderCreateInfoEXT> infos(shaderStages.size());
		VkShaderCreateFlagsEXT linkFlag = shaderStages.size() > 1u ? VK_SHADER_CREATE_LINK_STAGE_BIT_EXT : 0u;
		for (size_t i = 0; i < shaderStages.size(); ++i) {
			auto& [stage, spirv, entry] = shaderStages[i];
			infos[i] = {};
			infos[i].sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
			infos[i].flags                  = linkFlag;
			infos[i].stage                  = stage;
			infos[i].nextStage              = (i + 1 < shaderStages.size()) ? std::get<0>(shaderStages[i + 1]) : 0;
			infos[i].codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT;
			infos[i].codeSize               = spirv.size() * sizeof(uint32_t);
			infos[i].pCode                  = spirv.data();
			infos[i].pName                  = entry.data();
			infos[i].pushConstantRangeCount = (uint32_t)pcRanges.size();
			infos[i].pPushConstantRanges    = pcRanges.data();
			infos[i].setLayoutCount         = (uint32_t)setLayouts.size();
			infos[i].pSetLayouts            = setLayouts.data();
			stages[i] = stage;
		}
		vkCreateShadersEXT(VKRenderer::device, (uint32_t)infos.size(), infos.data(), nullptr, shaders.data());
	}

	~VKShader() {
		for (auto shader : shaders)
			vkDestroyShaderEXT(VKRenderer::device, shader, nullptr);
		if (layout != VK_NULL_HANDLE)
			vkDestroyPipelineLayout(VKRenderer::device, layout, nullptr);
	}

	std::vector<VkShaderEXT>          shaders;
	std::vector<VkShaderStageFlagBits> stages;
	VkPipelineLayout                   layout = VK_NULL_HANDLE;
};

void installDragEnterHandler(GLFWwindow* window, std::function<void(std::string)> callback);
void installDragDropHandler (GLFWwindow* window, std::function<void(std::string)> callback);
void installDragOverHandler (GLFWwindow* window, std::function<void(int32_t, int32_t)> callback);
