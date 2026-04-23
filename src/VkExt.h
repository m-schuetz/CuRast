#pragma once

// Extension function pointers for Vulkan EXT/KHR/NVX extensions
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

// VK_EXT_shader_object
inline PFN_vkCreateShadersEXT             g_vkCreateShadersEXT;
inline PFN_vkDestroyShaderEXT             g_vkDestroyShaderEXT;
inline PFN_vkCmdBindShadersEXT            g_vkCmdBindShadersEXT;
inline PFN_vkCmdSetVertexInputEXT         g_vkCmdSetVertexInputEXT;
inline PFN_vkCmdSetPolygonModeEXT         g_vkCmdSetPolygonModeEXT;
inline PFN_vkCmdSetRasterizationSamplesEXT g_vkCmdSetRasterizationSamplesEXT;
inline PFN_vkCmdSetSampleMaskEXT          g_vkCmdSetSampleMaskEXT;
inline PFN_vkCmdSetAlphaToCoverageEnableEXT g_vkCmdSetAlphaToCoverageEnableEXT;
inline PFN_vkCmdSetAlphaToOneEnableEXT    g_vkCmdSetAlphaToOneEnableEXT;
inline PFN_vkCmdSetLogicOpEnableEXT       g_vkCmdSetLogicOpEnableEXT;
inline PFN_vkCmdSetColorBlendEnableEXT    g_vkCmdSetColorBlendEnableEXT;
inline PFN_vkCmdSetColorBlendEquationEXT  g_vkCmdSetColorBlendEquationEXT;
inline PFN_vkCmdSetColorWriteMaskEXT      g_vkCmdSetColorWriteMaskEXT;

// VK_KHR_external_memory_win32 / VK_KHR_external_memory_fd
#ifdef _WIN32
inline PFN_vkGetMemoryWin32HandleKHR      g_vkGetMemoryWin32HandleKHR;
#else
inline PFN_vkGetMemoryFdKHR               g_vkGetMemoryFdKHR;
#endif

// VK_NVX_image_view_handle
inline PFN_vkGetImageViewHandleNVX        g_vkGetImageViewHandleNVX;

// VK_EXT_debug_utils
inline PFN_vkCreateDebugUtilsMessengerEXT  g_vkCreateDebugUtilsMessengerEXT;
inline PFN_vkDestroyDebugUtilsMessengerEXT g_vkDestroyDebugUtilsMessengerEXT;

// ---------------------------------------------------------------------------
// Macro aliases — make call sites look like standard vk calls
// ---------------------------------------------------------------------------
#define vkCreateShadersEXT              g_vkCreateShadersEXT
#define vkDestroyShaderEXT              g_vkDestroyShaderEXT
#define vkCmdBindShadersEXT             g_vkCmdBindShadersEXT
#define vkCmdSetVertexInputEXT          g_vkCmdSetVertexInputEXT
#define vkCmdSetPolygonModeEXT          g_vkCmdSetPolygonModeEXT
#define vkCmdSetRasterizationSamplesEXT g_vkCmdSetRasterizationSamplesEXT
#define vkCmdSetSampleMaskEXT           g_vkCmdSetSampleMaskEXT
#define vkCmdSetAlphaToCoverageEnableEXT g_vkCmdSetAlphaToCoverageEnableEXT
#define vkCmdSetAlphaToOneEnableEXT     g_vkCmdSetAlphaToOneEnableEXT
#define vkCmdSetLogicOpEnableEXT        g_vkCmdSetLogicOpEnableEXT
#define vkCmdSetColorBlendEnableEXT     g_vkCmdSetColorBlendEnableEXT
#define vkCmdSetColorBlendEquationEXT   g_vkCmdSetColorBlendEquationEXT
#define vkCmdSetColorWriteMaskEXT       g_vkCmdSetColorWriteMaskEXT

#ifdef _WIN32
#define vkGetMemoryWin32HandleKHR       g_vkGetMemoryWin32HandleKHR
#else
#define vkGetMemoryFdKHR                g_vkGetMemoryFdKHR
#endif

#define vkGetImageViewHandleNVX         g_vkGetImageViewHandleNVX

#define vkCreateDebugUtilsMessengerEXT  g_vkCreateDebugUtilsMessengerEXT
#define vkDestroyDebugUtilsMessengerEXT g_vkDestroyDebugUtilsMessengerEXT

// ---------------------------------------------------------------------------
// Loader — call once after logical device creation
// ---------------------------------------------------------------------------
inline void loadVkExt(VkInstance instance, VkDevice device) {
#define LOAD_DEV(fn) g_##fn = (PFN_##fn)vkGetDeviceProcAddr(device, #fn)
#define LOAD_INST(fn) g_##fn = (PFN_##fn)vkGetInstanceProcAddr(instance, #fn)

    LOAD_DEV(vkCreateShadersEXT);
    LOAD_DEV(vkDestroyShaderEXT);
    LOAD_DEV(vkCmdBindShadersEXT);
    LOAD_DEV(vkCmdSetVertexInputEXT);
    LOAD_DEV(vkCmdSetPolygonModeEXT);
    LOAD_DEV(vkCmdSetRasterizationSamplesEXT);
    LOAD_DEV(vkCmdSetSampleMaskEXT);
    LOAD_DEV(vkCmdSetAlphaToCoverageEnableEXT);
    LOAD_DEV(vkCmdSetAlphaToOneEnableEXT);
    LOAD_DEV(vkCmdSetLogicOpEnableEXT);
    LOAD_DEV(vkCmdSetColorBlendEnableEXT);
    LOAD_DEV(vkCmdSetColorBlendEquationEXT);
    LOAD_DEV(vkCmdSetColorWriteMaskEXT);

#ifdef _WIN32
    LOAD_DEV(vkGetMemoryWin32HandleKHR);
#else
    LOAD_DEV(vkGetMemoryFdKHR);
#endif

    LOAD_DEV(vkGetImageViewHandleNVX);

    LOAD_INST(vkCreateDebugUtilsMessengerEXT);
    LOAD_INST(vkDestroyDebugUtilsMessengerEXT);

#undef LOAD_DEV
#undef LOAD_INST
}
