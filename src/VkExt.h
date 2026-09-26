#pragma once

// Extension function pointers for Vulkan EXT/KHR extensions
// not included in vulkan-1.lib (core functions are loaded automatically).
//
// Call loadVkExt(instance, device) once after device creation.
// Macros below shadow the official function names so call sites are unchanged.

#include <vulkan/vulkan.h>
#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#endif

// ---------------------------------------------------------------------------
// Global function pointer variables
// ---------------------------------------------------------------------------

// VK_KHR_external_memory_win32 / VK_KHR_external_memory_fd
#ifdef _WIN32
inline PFN_vkGetMemoryWin32HandleKHR      g_vkGetMemoryWin32HandleKHR;
#else
inline PFN_vkGetMemoryFdKHR               g_vkGetMemoryFdKHR;
#endif

// VK_EXT_debug_utils
inline PFN_vkCreateDebugUtilsMessengerEXT  g_vkCreateDebugUtilsMessengerEXT;
inline PFN_vkDestroyDebugUtilsMessengerEXT g_vkDestroyDebugUtilsMessengerEXT;

// ---------------------------------------------------------------------------
// Macro aliases — make call sites look like standard vk calls
// ---------------------------------------------------------------------------
#ifdef _WIN32
#define vkGetMemoryWin32HandleKHR       g_vkGetMemoryWin32HandleKHR
#else
#define vkGetMemoryFdKHR                g_vkGetMemoryFdKHR
#endif

#define vkCreateDebugUtilsMessengerEXT  g_vkCreateDebugUtilsMessengerEXT
#define vkDestroyDebugUtilsMessengerEXT g_vkDestroyDebugUtilsMessengerEXT

// ---------------------------------------------------------------------------
// Loader — call once after logical device creation
// ---------------------------------------------------------------------------
inline void loadVkExt(VkInstance instance, VkDevice device) {
#define LOAD_DEV(fn) g_##fn = (PFN_##fn)vkGetDeviceProcAddr(device, #fn)
#define LOAD_INST(fn) g_##fn = (PFN_##fn)vkGetInstanceProcAddr(instance, #fn)

#ifdef _WIN32
    LOAD_DEV(vkGetMemoryWin32HandleKHR);
#else
    LOAD_DEV(vkGetMemoryFdKHR);
#endif

    LOAD_INST(vkCreateDebugUtilsMessengerEXT);
    LOAD_INST(vkDestroyDebugUtilsMessengerEXT);

#undef LOAD_DEV
#undef LOAD_INST
}
