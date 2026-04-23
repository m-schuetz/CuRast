
#include "VKRenderer.h"
#include "Runtime.h"
#include "Timer.h"
#include "CURuntime.h"
#include "CuRastSettings.h"

#include <filesystem>
#include <print>
#include <set>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

static void error_callback(int error, const char* description) {
	fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key < 0 || key >= (int)Runtime::keyStates.size()) return;

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);

	Runtime::keyStates[key] = action;
	Runtime::mods           = mods;
	Runtime::frame_keys.push_back(key);
	Runtime::frame_actions.push_back(action);
	Runtime::frame_mods.push_back(mods);
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
	ImGuiIO& io = ImGui::GetIO();
	if (io.WantCaptureMouse) return;

	Runtime::mousePosition = {xpos, ypos};

	int w, h;
	glfwGetWindowSize(window, &w, &h);
	Runtime::mouseEvents.onMouseMove(xpos, h - ypos);
}

static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
	ImGuiIO& io = ImGui::GetIO();
	if (io.WantCaptureMouse) return;
	Runtime::mouseEvents.onMouseScroll(xoffset, yoffset);
}

static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	ImGuiIO& io = ImGui::GetIO();
	if (io.WantCaptureMouse) return;

	if (action == 1)
		Runtime::mouseButtons = Runtime::mouseButtons | (1 << button);
	else if (action == 0)
		Runtime::mouseButtons = Runtime::mouseButtons & ~(1 << button);

	Runtime::controls->onMouseButton(button, action, mods);
	Runtime::mouseEvents.onMouseButton(button, action, mods);
}

// ---------------------------------------------------------------------------
// VKTexture helpers
// ---------------------------------------------------------------------------

void VKTexture::destroyCuda() {
	if (cudaSurface) { cuSurfObjectDestroy(cudaSurface);         cudaSurface = 0; }
	if (cudaMipArray) { cuMipmappedArrayDestroy(cudaMipArray);    cudaMipArray = nullptr; }
	if (cudaExtMem) { cuDestroyExternalMemory(cudaExtMem);      cudaExtMem = nullptr; }
}

void VKTexture::destroy() {
	destroyCuda();
	if (view   != VK_NULL_HANDLE) { vkDestroyImageView(VKRenderer::device, view, nullptr);   view   = VK_NULL_HANDLE; }
	if (image  != VK_NULL_HANDLE) { vkDestroyImage    (VKRenderer::device, image, nullptr);  image  = VK_NULL_HANDLE; }
	if (memory != VK_NULL_HANDLE) { vkFreeMemory       (VKRenderer::device, memory, nullptr); memory = VK_NULL_HANDLE; }
}

void VKTexture::importToCuda() {
	destroyCuda();

#ifdef _WIN32
	VkMemoryGetWin32HandleInfoKHR handleInfo = {};
	handleInfo.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
	handleInfo.memory     = memory;
	handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
	HANDLE win32Handle;
	vkGetMemoryWin32HandleKHR(VKRenderer::device, &handleInfo, &win32Handle);

	CUDA_EXTERNAL_MEMORY_HANDLE_DESC extDesc{};
	extDesc.type                 = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
	extDesc.handle.win32.handle  = win32Handle;
	extDesc.handle.win32.name    = nullptr;
#else
	VkMemoryGetFdInfoKHR handleInfo = {};
	handleInfo.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
	handleInfo.memory     = memory;
	handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
	int fd;
	vkGetMemoryFdKHR(VKRenderer::device, &handleInfo, &fd);

	CUDA_EXTERNAL_MEMORY_HANDLE_DESC extDesc{};
	extDesc.type       = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
	extDesc.handle.fd  = fd;
#endif

	VkMemoryRequirements memReqs;
	vkGetImageMemoryRequirements(VKRenderer::device, image, &memReqs);
	extDesc.size = memReqs.size;

	cuImportExternalMemory(&cudaExtMem, &extDesc);

#ifdef _WIN32
	CloseHandle(win32Handle); // CUDA holds its own reference
#endif

	CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC arrDesc{};
	arrDesc.offset                = 0;
	arrDesc.arrayDesc.Width       = (size_t)width;
	arrDesc.arrayDesc.Height      = (size_t)height;
	arrDesc.arrayDesc.Depth       = 0;
	arrDesc.arrayDesc.Format      = CU_AD_FORMAT_UNSIGNED_INT8;
	arrDesc.arrayDesc.NumChannels = 4; // RGBA8
	arrDesc.arrayDesc.Flags       = CUDA_ARRAY3D_SURFACE_LDST;
	arrDesc.numLevels             = 1;
	cuExternalMemoryGetMappedMipmappedArray(&cudaMipArray, cudaExtMem, &arrDesc);

	CUarray level0;
	cuMipmappedArrayGetLevel(&level0, cudaMipArray, 0);

	CUDA_RESOURCE_DESC resDesc{};
	resDesc.resType          = CU_RESOURCE_TYPE_ARRAY;
	resDesc.res.array.hArray = level0;
	cuSurfObjectCreate(&cudaSurface, &resDesc);
}

void VKTexture::setSize(int w, int h) {
	if (this->width == w && this->height == h) return;

	// Destroy old CUDA interop
	destroyCuda();

	this->width  = w;
	this->height = h;

	// Create exportable image
#ifdef _WIN32
	VkExternalMemoryImageCreateInfo extImgInfo = {};
	extImgInfo.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
	extImgInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
	VkExternalMemoryImageCreateInfo extImgInfo = {};
	extImgInfo.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
	extImgInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

	VkImageCreateInfo ci = {};
	ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	ci.pNext         = &extImgInfo;
	ci.imageType     = VK_IMAGE_TYPE_2D;
	ci.format        = format;
	ci.extent        = { (uint32_t)w, (uint32_t)h, 1 };
	ci.mipLevels     = 1;
	ci.arrayLayers   = 1;
	ci.samples       = VK_SAMPLE_COUNT_1_BIT;
	ci.tiling        = VK_IMAGE_TILING_OPTIMAL; // required for CUDA interop
	ci.usage         = VK_IMAGE_USAGE_TRANSFER_SRC_BIT       // blit to swapchain
	                 | VK_IMAGE_USAGE_STORAGE_BIT             // CUDA writes
	                 | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;   // Vulkan mesh raster
	ci.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
	ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	if (image  != VK_NULL_HANDLE) vkDestroyImage    (VKRenderer::device, image, nullptr);
	if (view   != VK_NULL_HANDLE) vkDestroyImageView(VKRenderer::device, view,  nullptr);
	if (memory != VK_NULL_HANDLE) vkFreeMemory       (VKRenderer::device, memory, nullptr);
	image = VK_NULL_HANDLE; view = VK_NULL_HANDLE; memory = VK_NULL_HANDLE;

	vkCreateImage(VKRenderer::device, &ci, nullptr, &image);

	// Allocate exportable memory
#ifdef _WIN32
	VkExportMemoryAllocateInfo exportInfo = {};
	exportInfo.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
	exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
	VkExportMemoryAllocateInfo exportInfo = {};
	exportInfo.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
	exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

	VkMemoryRequirements memReqs;
	vkGetImageMemoryRequirements(VKRenderer::device, image, &memReqs);

	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.pNext           = &exportInfo;
	allocInfo.allocationSize  = memReqs.size;
	allocInfo.memoryTypeIndex = VKRenderer::findMemoryType(memReqs.memoryTypeBits,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	vkAllocateMemory(VKRenderer::device, &allocInfo, nullptr, &memory);
	vkBindImageMemory(VKRenderer::device, image, memory, 0);

	// Image view
	VkImageViewCreateInfo viewCI = {};
	viewCI.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewCI.image                           = image;
	viewCI.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
	viewCI.format                          = format;
	viewCI.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
	viewCI.subresourceRange.baseMipLevel   = 0;
	viewCI.subresourceRange.levelCount     = 1;
	viewCI.subresourceRange.baseArrayLayer = 0;
	viewCI.subresourceRange.layerCount     = 1;
	vkCreateImageView(VKRenderer::device, &viewCI, nullptr, &view);

	// Transition to GENERAL layout so CUDA can write to it
	VKRenderer::transitionImageLayout(image,
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_GENERAL);

	version++;
}

// ---------------------------------------------------------------------------
// VKFramebuffer
// ---------------------------------------------------------------------------

std::shared_ptr<VKFramebuffer> VKFramebuffer::create(const std::string& label) {
	auto fbo       = std::make_shared<VKFramebuffer>();
	fbo->label     = label;
	fbo->colorAttachment = std::make_shared<VKTexture>();
	fbo->colorAttachment->label = label + "_color";
	fbo->colorAttachment->ID    = VKTexture::idcounter++;
	return fbo;
}

void VKFramebuffer::setSize(int w, int h) {
	if (this->width == w && this->height == h) return;
	colorAttachment->setSize(w, h);
	this->width  = w;
	this->height = h;
	version++;
}

// ---------------------------------------------------------------------------
// VKRenderer helpers
// ---------------------------------------------------------------------------

uint32_t VKRenderer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties memProps;
	vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);
	for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
		if ((typeFilter & (1u << i)) &&
		    (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}
	println("ERROR: findMemoryType failed");
	exit(1);
}

VkCommandBuffer VKRenderer::beginSingleTimeCommands() {
	VkCommandBufferAllocateInfo allocInfo = {};
	allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool        = commandPools[0];
	allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer cb;
	vkAllocateCommandBuffers(device, &allocInfo, &cb);

	VkCommandBufferBeginInfo bi = {};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(cb, &bi);
	return cb;
}

void VKRenderer::endSingleTimeCommands(VkCommandBuffer cb) {
	vkEndCommandBuffer(cb);

	VkSubmitInfo si = {};
	si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.commandBufferCount = 1;
	si.pCommandBuffers    = &cb;
	vkQueueSubmit(graphicsQueue, 1, &si, VK_NULL_HANDLE);
	vkQueueWaitIdle(graphicsQueue);

	vkFreeCommandBuffers(device, commandPools[0], 1, &cb);
}

void VKRenderer::transitionImageLayout(VkImage image,
	VkImageLayout oldLayout, VkImageLayout newLayout) {
	auto cb = beginSingleTimeCommands();

	VkImageMemoryBarrier barrier = {};
	barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout           = oldLayout;
	barrier.newLayout           = newLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = image;
	barrier.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

	VkPipelineStageFlags srcStage, dstStage;
	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
	    newLayout == VK_IMAGE_LAYOUT_GENERAL) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = 0;
		srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		dstStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
	} else {
		barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
		dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	}

	vkCmdPipelineBarrier(cb, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
	endSingleTimeCommands(cb);
}

// ---------------------------------------------------------------------------
// VKRenderer::init()
// ---------------------------------------------------------------------------

void VKRenderer::init() {
	camera = std::make_shared<Camera>();

	// GLFW — no OpenGL context
	glfwSetErrorCallback(error_callback);
	if (!glfwInit()) {
		println("glfwInit failed");
		exit(1);
	}
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	// glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);

	int numMonitors;
	GLFWmonitor** monitors = glfwGetMonitors(&numMonitors);
	const GLFWvidmode* mode = glfwGetVideoMode(monitors[0]);
	window = glfwCreateWindow(1920, 1080, "Splat Editor", nullptr, nullptr);
	if (!window) {
		glfwTerminate();
		exit(1);
	}
	if (mode->width >= 1920 && mode->height >= 1080) {
		glfwSetWindowPos(window, (mode->width - 1920) / 2, (mode->height - 1080) / 2);
	}

	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetDropCallback(window, [](GLFWwindow*, int count, const char** paths) {
		std::vector<std::string> files;
		for (int i = 0; i < count; i++)
			files.push_back(paths[i]);
		for (auto& listener : VKRenderer::fileDropListeners)
			listener(files);
	});

	createInstance();
	createSurface();
	pickPhysicalDevice();
	createLogicalDevice();
	createSwapchain();
	createSwapchainImageViews();
	createCommandObjects();
	createSyncObjects();
	initImGui();

	Timer::initVulkan(physDevice, device, FRAMES_IN_FLIGHT);

	view.framebuffer = VKFramebuffer::create("main_fbo");
	view.framebuffer->setSize(128, 128);

	println("VKRenderer initialized");

	{ // print memory properties
		VkPhysicalDevice physDevice = VKRenderer::physDevice;
		VkDevice         device     = VKRenderer::device;

		// Query memory properties
		VkPhysicalDeviceMemoryProperties memProps;
		vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);

		// Find a heap that satisfies HOST_VISIBLE | HOST_COHERENT and is compatible with the buffer
		for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
			VkMemoryType     type      = memProps.memoryTypes[i];
			VkMemoryHeap     heap      = memProps.memoryHeaps[type.heapIndex];
			VkMemoryPropertyFlags f    = type.propertyFlags;

			// bool typeCompatible  = (memReqs.memoryTypeBits >> i) & 1;
			bool hostVisible     = f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
			bool hostCoherent    = f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
			bool hostCached      = f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
			bool deviceLocal     = f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
			bool lazilyAllocated = f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
			bool heapDeviceLocal = heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
			bool heapMultiInst   = heap.flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT;

			println("  type[{:2}]  heap={:2}  size={:6L} MB | "
			        "DEVICE_LOCAL={:5}  HOST_VISIBLE={:5}  HOST_COHERENT={:5}  HOST_CACHED={:5}  LAZILY_ALLOCATED={:5}  "
			        "| heap: DEVICE_LOCAL={:5}  MULTI_INSTANCE={:5}",
				i,
				type.heapIndex,
				heap.size / (1024 * 1024),
				// typeCompatible,
				deviceLocal, hostVisible, hostCoherent, hostCached, lazilyAllocated,
				heapDeviceLocal, heapMultiInst);
		}
	}
}

void VKRenderer::destroy() {
	vkDeviceWaitIdle(device);

	// Release all Vulkan resources held by drawVulkan() before destroying the device
	if (vulkanMeshCleanupFn) {
		vulkanMeshCleanupFn();
		vulkanMeshCleanupFn = nullptr;
	}

	if (imguiDescriptorPool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(device, imguiDescriptorPool, nullptr);
		imguiDescriptorPool = VK_NULL_HANDLE;
	}

	cleanupSwapchain();

	for (auto s : imageAvailableSemaphores) vkDestroySemaphore(device, s, nullptr);
	imageAvailableSemaphores.clear();
	for (auto f : inFlightFences) vkDestroyFence(device, f, nullptr);
	inFlightFences.clear();

	// Command buffers are freed when their pools are destroyed
	for (auto pool : commandPools) vkDestroyCommandPool(device, pool, nullptr);
	commandPools.clear();
	commandBuffers.clear();

	Timer::destroyVulkan(device);

	vkDestroyDevice(device, nullptr);
	device = VK_NULL_HANDLE;

	vkDestroySurfaceKHR(instance, surface, nullptr);
	surface = VK_NULL_HANDLE;

	if (debugMessenger && g_vkDestroyDebugUtilsMessengerEXT) {
		g_vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		debugMessenger = VK_NULL_HANDLE;
	}

	vkDestroyInstance(instance, nullptr);
	instance = VK_NULL_HANDLE;
}

// ---------------------------------------------------------------------------
// Vulkan init helpers
// ---------------------------------------------------------------------------

static VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT      severity,
	VkDebugUtilsMessageTypeFlagsEXT             type,
	const VkDebugUtilsMessengerCallbackDataEXT* data,
	void*                                       userData)
{
	if (data->messageIdNumber == (int32_t)0x9469b92a) return VK_FALSE; // NV BINDLESS
	if (data->messageIdNumber == (int32_t)2067883941) return VK_FALSE; // NV BINDLESS
	if (data->messageIdNumber == (int32_t)0x101707af) return VK_FALSE; // <blabla> not marked with NonWritable

	println("Vulkan: {}", data->pMessage);

	__debugbreak();

	return VK_FALSE;
}

void VKRenderer::createInstance() {
	VkApplicationInfo appInfo = {};
	appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName   = "Splat Editor";
	appInfo.applicationVersion = 1;
	appInfo.pEngineName        = "CuRast";
	appInfo.engineVersion      = 1;
	appInfo.apiVersion         = VK_API_VERSION_1_4;

	std::vector<const char*> extensions;
	{
		uint32_t count;
		const char** glfwExts = glfwGetRequiredInstanceExtensions(&count);
		for (uint32_t i = 0; i < count; i++)
			extensions.push_back(glfwExts[i]);
	}
	extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

	std::vector<const char*> layers = {};

	VkInstanceCreateInfo instanceInfo = {};
	instanceInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instanceInfo.pApplicationInfo        = &appInfo;
	instanceInfo.enabledExtensionCount   = (uint32_t)extensions.size();
	instanceInfo.ppEnabledExtensionNames = extensions.data();
	instanceInfo.enabledLayerCount       = (uint32_t)layers.size();
	instanceInfo.ppEnabledLayerNames     = layers.data();

	vkCreateInstance(&instanceInfo, nullptr, &instance);

	// Load debug utils instance functions and create the messenger
	g_vkCreateDebugUtilsMessengerEXT  = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	g_vkDestroyDebugUtilsMessengerEXT = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

	if (g_vkCreateDebugUtilsMessengerEXT) {
		VkDebugUtilsMessengerCreateInfoEXT messengerCI = {};
		messengerCI.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		messengerCI.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
		                            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		messengerCI.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
		                            | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
		                            | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		messengerCI.pfnUserCallback = debugUtilsCallback;
		g_vkCreateDebugUtilsMessengerEXT(instance, &messengerCI, nullptr, &debugMessenger);
	}
}

void VKRenderer::createSurface() {
	if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
		println("glfwCreateWindowSurface failed");
		exit(1);
	}
}

void VKRenderer::pickPhysicalDevice() {
	// Try to match the CUDA device by UUID
	CUuuid cudaUUID;
	cuDeviceGetUuid(&cudaUUID, CURuntime::device);

	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

	physDevice = devices[0]; // fallback

	for (auto& candidate : devices) {
		try {
			VkPhysicalDeviceIDProperties idProps = {};
			idProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;

			VkPhysicalDeviceProperties2 props2 = {};
			props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
			props2.pNext = &idProps;
			vkGetPhysicalDeviceProperties2(candidate, &props2);

			if (memcmp(cudaUUID.bytes, idProps.deviceUUID, VK_UUID_SIZE) == 0) {
				physDevice = candidate;
				println("Matched Vulkan device to CUDA device by UUID: {}",
					std::string(props2.properties.deviceName));
				return;
			}
		} catch (...) {}
	}
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(physDevice, &props);
	println("WARN: No UUID match, using first device: {}", std::string(props.deviceName));
}

void VKRenderer::createLogicalDevice() {
	// Find a queue family that supports graphics + present
	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &queueFamilyCount, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &queueFamilyCount, queueFamilies.data());

	for (uint32_t i = 0; i < queueFamilyCount; i++) {
		bool graphics = !!(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT);
		VkBool32 present = VK_FALSE;
		vkGetPhysicalDeviceSurfaceSupportKHR(physDevice, i, surface, &present);
		if (graphics && present) {
			graphicsQueueFamily = i;
			break;
		}
	}

	float queuePriority = 1.0f;
	VkDeviceQueueCreateInfo queueCI = {};
	queueCI.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueCI.queueFamilyIndex = graphicsQueueFamily;
	queueCI.queueCount       = 1;
	queueCI.pQueuePriorities = &queuePriority;

	std::vector<const char*> deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME,
		VK_EXT_SHADER_OBJECT_EXTENSION_NAME,
		VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME,
#ifdef _WIN32
		VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
		VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#else
		VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
		VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
#endif
		VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
		VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
	};

	VkPhysicalDeviceVulkan11Features vulkan11Features = {};
	vulkan11Features.sType                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
	vulkan11Features.shaderDrawParameters  = VK_TRUE;
	vulkan11Features.storagePushConstant16 = VK_TRUE;

	VkPhysicalDeviceVulkan12Features vulkan12Features = {};
	vulkan12Features.sType                                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	vulkan12Features.bufferDeviceAddress                      = VK_TRUE;
	vulkan12Features.runtimeDescriptorArray                   = VK_TRUE;
	vulkan12Features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
	vulkan12Features.descriptorBindingVariableDescriptorCount  = VK_TRUE;
	vulkan12Features.descriptorBindingPartiallyBound           = VK_TRUE;

	VkPhysicalDeviceVulkan13Features vulkan13Features = {};
	vulkan13Features.sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
	vulkan13Features.dynamicRendering = VK_TRUE;
	vulkan13Features.synchronization2 = VK_TRUE;

	VkPhysicalDeviceVulkan14Features vulkan14Features = {};
	vulkan14Features.sType          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
	vulkan14Features.hostImageCopy  = VK_TRUE;

	VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeature = {};
	shaderObjectFeature.sType        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;
	shaderObjectFeature.shaderObject = VK_TRUE;

	vulkan11Features.pNext    = &vulkan12Features;
	vulkan12Features.pNext    = &vulkan13Features;
	vulkan13Features.pNext    = &vulkan14Features;
	vulkan14Features.pNext    = &shaderObjectFeature;

	VkPhysicalDeviceFeatures2 features2 = {};
	features2.sType                         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	features2.pNext                         = &vulkan11Features;
	features2.features.multiDrawIndirect    = VK_TRUE;
	features2.features.shaderInt64          = VK_TRUE;
	features2.features.shaderInt16          = VK_TRUE;
	features2.features.sparseBinding        = VK_TRUE;

	VkDeviceCreateInfo deviceCI = {};
	deviceCI.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCI.pNext                   = &features2;
	deviceCI.queueCreateInfoCount    = 1;
	deviceCI.pQueueCreateInfos       = &queueCI;
	deviceCI.enabledExtensionCount   = (uint32_t)deviceExtensions.size();
	deviceCI.ppEnabledExtensionNames = deviceExtensions.data();

	vkCreateDevice(physDevice, &deviceCI, nullptr, &device);

	// Load extension function pointers now that we have a device
	loadVkExt(instance, device);

	vkGetDeviceQueue(device, graphicsQueueFamily, 0, &graphicsQueue);
}

void VKRenderer::createSwapchain() {
	uint32_t formatCount = 0;
	vkGetPhysicalDeviceSurfaceFormatsKHR(physDevice, surface, &formatCount, nullptr);
	std::vector<VkSurfaceFormatKHR> surfaceFormats(formatCount);
	vkGetPhysicalDeviceSurfaceFormatsKHR(physDevice, surface, &formatCount, surfaceFormats.data());

	uint32_t modeCount = 0;
	vkGetPhysicalDeviceSurfacePresentModesKHR(physDevice, surface, &modeCount, nullptr);
	std::vector<VkPresentModeKHR> presentModes(modeCount);
	vkGetPhysicalDeviceSurfacePresentModesKHR(physDevice, surface, &modeCount, presentModes.data());

	VkSurfaceCapabilitiesKHR capabilities;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physDevice, surface, &capabilities);

	// Pick format (prefer BGRA8)
	swapchainFormat = surfaceFormats[0].format;
	for (auto& f : surfaceFormats) {
		if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
			f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			swapchainFormat = f.format;
			break;
		}
	}

	VkPresentModeKHR presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;

	// Extent
	if (capabilities.currentExtent.width != UINT32_MAX) {
		swapchainExtent = capabilities.currentExtent;
	} else {
		int w, h;
		glfwGetFramebufferSize(window, &w, &h);
		swapchainExtent.width  = std::clamp((uint32_t)w, capabilities.minImageExtent.width,  capabilities.maxImageExtent.width);
		swapchainExtent.height = std::clamp((uint32_t)h, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
	}

	uint32_t imageCount = capabilities.minImageCount + 1;
	if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount)
		imageCount = capabilities.maxImageCount;

	VkSwapchainCreateInfoKHR swCI = {};
	swCI.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	swCI.surface          = surface;
	swCI.minImageCount    = imageCount;
	swCI.imageFormat      = swapchainFormat;
	swCI.imageColorSpace  = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
	swCI.imageExtent      = swapchainExtent;
	swCI.imageArrayLayers = 1;
	swCI.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	swCI.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	swCI.preTransform     = capabilities.currentTransform;
	swCI.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	swCI.presentMode      = presentMode;
	swCI.clipped          = VK_TRUE;

	vkCreateSwapchainKHR(device, &swCI, nullptr, &swapchain);

	uint32_t swImageCount = 0;
	vkGetSwapchainImagesKHR(device, swapchain, &swImageCount, nullptr);
	swapchainImages.resize(swImageCount);
	vkGetSwapchainImagesKHR(device, swapchain, &swImageCount, swapchainImages.data());
}

void VKRenderer::createSwapchainImageViews() {
	for (auto view : swapchainImageViews)
		vkDestroyImageView(device, view, nullptr);
	swapchainImageViews.clear();

	for (auto& img : swapchainImages) {
		VkImageViewCreateInfo ci = {};
		ci.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		ci.image                           = img;
		ci.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
		ci.format                          = swapchainFormat;
		ci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
		ci.subresourceRange.baseMipLevel   = 0;
		ci.subresourceRange.levelCount     = 1;
		ci.subresourceRange.baseArrayLayer = 0;
		ci.subresourceRange.layerCount     = 1;
		VkImageView v = VK_NULL_HANDLE;
		vkCreateImageView(device, &ci, nullptr, &v);
		swapchainImageViews.push_back(v);
	}
}

void VKRenderer::createCommandObjects() {
	for (auto pool : commandPools) vkDestroyCommandPool(device, pool, nullptr);
	commandPools.clear();
	commandBuffers.clear();

	VkCommandPoolCreateInfo poolCI = {};
	poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolCI.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	poolCI.queueFamilyIndex = graphicsQueueFamily;

	for (int i = 0; i < FRAMES_IN_FLIGHT; i++) {
		VkCommandPool pool = VK_NULL_HANDLE;
		vkCreateCommandPool(device, &poolCI, nullptr, &pool);
		commandPools.push_back(pool);

		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool        = pool;
		allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = 1;
		VkCommandBuffer cb = VK_NULL_HANDLE;
		vkAllocateCommandBuffers(device, &allocInfo, &cb);
		commandBuffers.push_back(cb);
	}
}

void VKRenderer::createSyncObjects() {
	for (auto s : imageAvailableSemaphores) vkDestroySemaphore(device, s, nullptr);
	for (auto s : renderFinishedSemaphores) vkDestroySemaphore(device, s, nullptr);
	for (auto f : inFlightFences)           vkDestroyFence    (device, f, nullptr);
	imageAvailableSemaphores.clear();
	renderFinishedSemaphores.clear();
	inFlightFences.clear();

	VkSemaphoreCreateInfo sci = {};
	sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fenceCI = {};
	fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	for (int i = 0; i < FRAMES_IN_FLIGHT; i++) {
		VkSemaphore sem = VK_NULL_HANDLE;
		vkCreateSemaphore(device, &sci, nullptr, &sem);
		imageAvailableSemaphores.push_back(sem);

		VkFence fence = VK_NULL_HANDLE;
		vkCreateFence(device, &fenceCI, nullptr, &fence);
		inFlightFences.push_back(fence);
	}
	// One renderFinished semaphore per swapchain image — avoids reuse while
	// the presentation engine still holds a reference to a previous signal.
	for (size_t i = 0; i < swapchainImages.size(); i++) {
		VkSemaphore sem = VK_NULL_HANDLE;
		vkCreateSemaphore(device, &sci, nullptr, &sem);
		renderFinishedSemaphores.push_back(sem);
	}
}

void VKRenderer::initImGui() {
	// Descriptor pool for ImGui
	VkDescriptorPoolSize poolSize = {};
	poolSize.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSize.descriptorCount = 1000;

	VkDescriptorPoolCreateInfo poolCI = {};
	poolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolCI.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	poolCI.maxSets       = 1000;
	poolCI.poolSizeCount = 1;
	poolCI.pPoolSizes    = &poolSize;
	vkCreateDescriptorPool(device, &poolCI, nullptr, &imguiDescriptorPool);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImGui_ImplGlfw_InitForVulkan(window, true);

	// Dynamic rendering pipeline info: tell ImGui which color format to use
	VkPipelineRenderingCreateInfoKHR pipelineRenderingCI = {};
	pipelineRenderingCI.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
	pipelineRenderingCI.colorAttachmentCount    = 1;
	pipelineRenderingCI.pColorAttachmentFormats = &swapchainFormat;

	ImGui_ImplVulkan_InitInfo info = {};
	info.ApiVersion       = VK_API_VERSION_1_4;
	info.Instance         = instance;
	info.PhysicalDevice   = physDevice;
	info.Device           = device;
	info.QueueFamily      = graphicsQueueFamily;
	info.Queue            = graphicsQueue;
	info.DescriptorPool   = imguiDescriptorPool;
	info.MinImageCount    = VKRenderer::FRAMES_IN_FLIGHT;
	info.ImageCount       = (uint32_t)swapchainImages.size();
	info.UseDynamicRendering = true;
	info.PipelineInfoMain.PipelineRenderingCreateInfo = pipelineRenderingCI;

	ImGui_ImplVulkan_Init(&info);

	ImGui::StyleColorsDark();
}

// ---------------------------------------------------------------------------
// Swapchain recreation
// ---------------------------------------------------------------------------

void VKRenderer::cleanupSwapchain() {
	for (auto s : renderFinishedSemaphores) vkDestroySemaphore(device, s, nullptr);
	renderFinishedSemaphores.clear();
	for (auto v : swapchainImageViews) vkDestroyImageView(device, v, nullptr);
	swapchainImageViews.clear();
	if (swapchain != VK_NULL_HANDLE) {
		vkDestroySwapchainKHR(device, swapchain, nullptr);
		swapchain = VK_NULL_HANDLE;
	}
}

void VKRenderer::recreateSwapchain() {
	int w = 0, h = 0;
	while (w == 0 || h == 0) {
		glfwGetFramebufferSize(window, &w, &h);
		glfwWaitEvents();
	}
	vkDeviceWaitIdle(device);
	cleanupSwapchain();
	createSwapchain();
	createSwapchainImageViews();

	VkSemaphoreCreateInfo sci = {};
	sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	for (size_t i = 0; i < swapchainImages.size(); i++) {
		VkSemaphore sem = VK_NULL_HANDLE;
		vkCreateSemaphore(device, &sci, nullptr, &sem);
		renderFinishedSemaphores.push_back(sem);
	}
}

// ---------------------------------------------------------------------------
// Command buffer recording
// ---------------------------------------------------------------------------

void VKRenderer::recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
	VkCommandBufferBeginInfo bi = {};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(cmd, &bi);

	auto& colorTex = view.framebuffer->colorAttachment;
	VkImageSubresourceRange colorSubRes = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

	// All barriers use VkImageMemoryBarrier2 (Vulkan 1.3 core synchronization2).

	if (vulkanMeshDrawFn) {
		// --- Vulkan mesh rasterizer path ---
		VkImageSubresourceRange depthSubRes = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };

		Timer::resetVulkanFrame(cmd, currentFrame);

		// 1a. Depth: UNDEFINED → GENERAL
		{
			VkImageMemoryBarrier2 b = {};
			b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
			b.srcStageMask  = VK_PIPELINE_STAGE_2_NONE;
			b.srcAccessMask = VK_ACCESS_2_NONE;
			b.dstStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
			                | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
			b.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
			                | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			b.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
			b.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
			b.image         = vulkanMeshDepthImage;
			b.subresourceRange = depthSubRes;

			VkDependencyInfo dep = {};
			dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
			dep.imageMemoryBarrierCount = 1;
			dep.pImageMemoryBarriers    = &b;
			vkCmdPipelineBarrier2(cmd, &dep);
		}

		// 1b. Dynamic rendering + mesh draw
		{
			VkClearValue clearColor = {};
			clearColor.color = { 
				CuRastSettings::background.x,
				CuRastSettings::background.y,
				CuRastSettings::background.z,
				1.f
			};
			

			VkClearValue clearDepth = {};
			clearDepth.depthStencil = { 0.0f, 0 };

			VkRenderingAttachmentInfo ca = {};
			ca.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
			ca.imageView   = colorTex->view;
			ca.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			ca.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
			ca.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
			ca.clearValue  = clearColor;

			VkRenderingAttachmentInfo da = {};
			da.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
			da.imageView   = vulkanMeshDepthView;
			da.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			da.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
			da.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			da.clearValue  = clearDepth;

			VkRenderingInfo ri = {};
			ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
			ri.renderArea           = { {0, 0}, {(uint32_t)colorTex->width, (uint32_t)colorTex->height} };
			ri.layerCount           = 1;
			ri.colorAttachmentCount = 1;
			ri.pColorAttachments    = &ca;
			ri.pDepthAttachment     = &da;
			vkCmdBeginRendering(cmd, &ri);
			vulkanMeshDrawFn(cmd);
			vkCmdEndRendering(cmd);
		}

		// 1c. Color attachment write → transfer read (stays GENERAL)
		{
			VkImageMemoryBarrier2 b = {};
			b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
			b.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
			b.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
			b.dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
			b.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
			b.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
			b.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
			b.image         = colorTex->image;
			b.subresourceRange = colorSubRes;

			VkDependencyInfo dep = {};
			dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
			dep.imageMemoryBarrierCount = 1;
			dep.pImageMemoryBarriers    = &b;
			vkCmdPipelineBarrier2(cmd, &dep);
		}
	} // else CUDA path: colorTex stays GENERAL; CUDA writes synced via cuStreamSynchronize in unmapCudaVk()

	// 2. Swapchain: UNDEFINED → GENERAL
	{
		VkImageMemoryBarrier2 b = {};
		b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
		b.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
		b.srcAccessMask = VK_ACCESS_2_NONE;
		b.dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
		b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
		b.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
		b.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
		b.image         = swapchainImages[imageIndex];
		b.subresourceRange = colorSubRes;

		VkDependencyInfo dep = {};
		dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		dep.imageMemoryBarrierCount = 1;
		dep.pImageMemoryBarriers    = &b;
		vkCmdPipelineBarrier2(cmd, &dep);
	}

	// 3. Blit CUDA output texture → swapchain image (handles RGBA↔BGRA swizzle)
	{
		VkImageBlit region = {};
		region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
		region.srcOffsets[0]  = { 0, 0, 0 };
		region.srcOffsets[1]  = { colorTex->width, colorTex->height, 1 };
		region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
		region.dstOffsets[0]  = { 0, (int32_t)swapchainExtent.height, 0 };
		region.dstOffsets[1]  = { (int32_t)swapchainExtent.width, 0, 1 };
		vkCmdBlitImage(cmd,
			colorTex->image, VK_IMAGE_LAYOUT_GENERAL,
			swapchainImages[imageIndex], VK_IMAGE_LAYOUT_GENERAL,
			1, &region, VK_FILTER_LINEAR);
	}

	// 4. Swapchain: transfer write → color attachment (GENERAL) for ImGui dynamic rendering
	{
		VkImageMemoryBarrier2 b = {};
		b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
		b.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
		b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
		b.dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
		b.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT
		                | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
		b.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
		b.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
		b.image         = swapchainImages[imageIndex];
		b.subresourceRange = colorSubRes;

		VkDependencyInfo dep = {};
		dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		dep.imageMemoryBarrierCount = 1;
		dep.pImageMemoryBarriers    = &b;
		vkCmdPipelineBarrier2(cmd, &dep);
	}

	// 5. ImGui dynamic rendering (loadOp=LOAD preserves the blit result)
	{
		VkRenderingAttachmentInfo colorAttach = {};
		colorAttach.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
		colorAttach.imageView   = swapchainImageViews[imageIndex];
		colorAttach.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		colorAttach.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
		colorAttach.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

		VkRenderingInfo ri = {};
		ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
		ri.renderArea           = { {0, 0}, swapchainExtent };
		ri.layerCount           = 1;
		ri.colorAttachmentCount = 1;
		ri.pColorAttachments    = &colorAttach;

		vkCmdBeginRendering(cmd, &ri);
		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
		vkCmdEndRendering(cmd);
	}

	// 6. Swapchain: GENERAL → PRESENT_SRC
	{
		VkImageMemoryBarrier2 b = {};
		b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
		b.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
		b.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
		b.dstStageMask  = VK_PIPELINE_STAGE_2_NONE;
		b.dstAccessMask = VK_ACCESS_2_NONE;
		b.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
		b.newLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		b.image         = swapchainImages[imageIndex];
		b.subresourceRange = colorSubRes;

		VkDependencyInfo dep = {};
		dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		dep.imageMemoryBarrierCount = 1;
		dep.pImageMemoryBarriers    = &b;
		vkCmdPipelineBarrier2(cmd, &dep);
	}

	vkEndCommandBuffer(cmd);
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

void VKRenderer::loop(
	std::function<void(void)> update,
	std::function<void(void)> render,
	std::function<void(void)> postFrame)
{
	int    fpsCounter = 0;
	double start = now();
	double tPrevious = start;
	double tPreviousFPS = start;

	while (!glfwWindowShouldClose(window)) {
		// Timing
		{
			double tCurrent = now();
			timeSinceLastFrame = tCurrent - tPrevious;
			tPrevious = tCurrent;

			double timeSinceFPS = tCurrent - tPreviousFPS;
			if (timeSinceFPS >= 1.0) {
				fps = double(fpsCounter) / timeSinceFPS;
				tPreviousFPS = tCurrent;
				fpsCounter = 0;
			}
		}

		// Window size
		int w, h;
		glfwGetWindowSize(window, &w, &h);
		if (w == 0 || h == 0) {
			glfwPollEvents();
			continue;
		}
		camera->setSize(w, h);
		width = w;
		height = h;

		EventQueue::instance->process();

		{ // Camera update
			Runtime::controls->update();
			camera->world = Runtime::controls->world;
			camera->position = camera->world * glm::dvec4(0.0, 0.0, 0.0, 1.0);
		}

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();

		// Update & render
		camera->update();
		update();
		camera->update();

		// Resize the main framebuffer to match window
		if (view.framebuffer->width != w || view.framebuffer->height != h) vkDeviceWaitIdle(device);
		view.framebuffer->setSize(w, h);

		// Wait for previous frame
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		render(); // CUDA kernels + kernel_resolve + ImGui::Render() (but not RenderDrawData)

		auto recordings = Timer::resolve();
		for (auto recording : recordings) {
			Runtime::timings.add(recording.label, recording.milliseconds);
		}

		Runtime::frame_keys.clear();
		Runtime::frame_actions.clear();
		Runtime::frame_mods.clear();

		for (auto& r : Timer::resolveVulkan(device, currentFrame)){
			Runtime::timings.add(r.label, r.milliseconds);
		}

		// Acquire next swapchain image
		uint32_t imageIndex;
		VkResult acquireResult = vkAcquireNextImageKHR(
			device, swapchain, UINT64_MAX,
			imageAvailableSemaphores[currentFrame],
			VK_NULL_HANDLE, &imageIndex);

		if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwapchain();
			continue;
		}

		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		// Record command buffer: blit CUDA result + ImGui dynamic rendering
		vkResetCommandBuffer(commandBuffers[currentFrame], 0);
		recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

		// Submit
		VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		VkSubmitInfo submit = {};
		submit.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit.waitSemaphoreCount   = 1;
		submit.pWaitSemaphores      = &imageAvailableSemaphores[currentFrame];
		submit.pWaitDstStageMask    = &waitStage;
		submit.commandBufferCount   = 1;
		submit.pCommandBuffers      = &commandBuffers[currentFrame];
		submit.signalSemaphoreCount = 1;
		submit.pSignalSemaphores    = &renderFinishedSemaphores[imageIndex]; // per-image, not per-frame
		vkQueueSubmit(graphicsQueue, 1, &submit, inFlightFences[currentFrame]);

		// Present
		VkPresentInfoKHR presentInfo = {};
		presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores    = &renderFinishedSemaphores[imageIndex];
		presentInfo.swapchainCount     = 1;
		presentInfo.pSwapchains        = &swapchain;
		presentInfo.pImageIndices      = &imageIndex;
		VkResult presentResult = vkQueuePresentKHR(graphicsQueue, &presentInfo);
		if (presentResult == VK_ERROR_OUT_OF_DATE_KHR ||
		    presentResult == VK_SUBOPTIMAL_KHR) {
			recreateSwapchain();
		}

		postFrame();
		glfwPollEvents();

		currentFrame = (currentFrame + 1) % FRAMES_IN_FLIGHT;
		frameCount++;
		fpsCounter++;
	}

	vkDeviceWaitIdle(device);

	// Explicitly destroy the CUDA-interop texture before device destruction.
	if (view.framebuffer && view.framebuffer->colorAttachment) {
		view.framebuffer->colorAttachment->destroy();
	}

	// ImGui cleanup
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();
}
