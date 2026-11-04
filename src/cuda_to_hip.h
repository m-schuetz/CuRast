// cuda_to_hip.h -- CUDA-to-HIP compatibility header for CuRast
// This header aliases CUDA spellings to HIP equivalents on AMD GPUs.
// On NVIDIA it is a no-op include of the CUDA headers.
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>

#pragma once

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

// Under hiprtc (runtime kernel compilation) the device runtime and cooperative
// groups are provided by hiprtc's auto-included builtin header; the host
// runtime headers are not on the hiprtc include path, so skip them there.
#if !defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#else
// hiprtc provides the core device runtime but not these CUDA-style bit-cast
// intrinsics (they normally come from hip_runtime.h, skipped above under RTC).
// Defined as macros (not inline functions) because in one combined hiprtc TU
// this header is reached via two relative paths (src/jpeg/../, src/kernels/../)
// and #pragma once does not dedupe; identical macro redefinition is allowed,
// whereas duplicate function definitions are an error.
#define __float_as_uint(x) __builtin_bit_cast(unsigned int, (float)(x))
#define __float_as_int(x)  __builtin_bit_cast(int, (float)(x))
#define __uint_as_float(x) __builtin_bit_cast(float, (unsigned int)(x))
#define __int_as_float(x)  __builtin_bit_cast(float, (int)(x))
#endif

// Runtime API aliases
#define cudaMalloc                  hipMalloc
#define cudaFree                    hipFree
#define cudaMemcpy                  hipMemcpy
#define cudaMemcpyAsync             hipMemcpyAsync
#define cudaMemcpyHostToDevice      hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost      hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice    hipMemcpyDeviceToDevice
#define cudaMemset                  hipMemset
#define cudaMemsetAsync             hipMemsetAsync
#define cudaDeviceSynchronize       hipDeviceSynchronize
#define cudaGetLastError            hipGetLastError
#define cudaPeekAtLastError         hipPeekAtLastError
#define cudaGetErrorString          hipGetErrorString
#define cudaGetDevice               hipGetDevice
#define cudaSetDevice               hipSetDevice
#define cudaGetDeviceCount          hipGetDeviceCount
#define cudaGetDeviceProperties     hipGetDeviceProperties

#define cudaStream_t                hipStream_t
#define cudaStreamCreate            hipStreamCreate
#define cudaStreamDestroy           hipStreamDestroy
#define cudaStreamSynchronize       hipStreamSynchronize

#define cudaEvent_t                 hipEvent_t
#define cudaEventCreate             hipEventCreate
#define cudaEventDestroy            hipEventDestroy
#define cudaEventRecord             hipEventRecord
#define cudaEventSynchronize        hipEventSynchronize
#define cudaEventElapsedTime        hipEventElapsedTime

#define cudaError_t                 hipError_t
#define cudaSuccess                 hipSuccess

// Device properties
#define cudaDeviceProp              hipDeviceProp_t
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaLaunchCooperativeKernel hipLaunchCooperativeKernel
#define cudaMemcpyToSymbol          hipMemcpyToSymbol

// Surface objects
#define cudaSurfaceObject_t         hipSurfaceObject_t
#define surf2Dwrite                 surf2Dwrite  // HIP provides surf2Dwrite directly

// Driver API aliases
#define CUDA_SUCCESS                hipSuccess
#define CUresult                    hipError_t
#define CUdevice                    hipDevice_t
#define CUcontext                   hipCtx_t
#define CUmodule                    hipModule_t
#define CUfunction                  hipFunction_t
#define CUdeviceptr                 hipDeviceptr_t
#define CUstream                    hipStream_t
#define CUevent                     hipEvent_t
// hipDeviceptr_t is void*, so arithmetic on CUdeviceptr is ill-formed in strict C++.
// Windows Clang rejects it even in gnu++ mode. Use this helper for offset arithmetic.
#define HIP_DEVPTR_ADD(ptr, off)    ((hipDeviceptr_t)((uint8_t*)(ptr) + (uint64_t)(off)))

#define cuInit                      hipInit
#define cuDeviceGet                 hipDeviceGet
#define cuDeviceGetAttribute        hipDeviceGetAttribute
#define cuCtxCreate                 hipCtxCreate
#define cuCtxDestroy                hipCtxDestroy
#define cuCtxGetDevice              hipCtxGetDevice
#define cuCtxSynchronize            hipCtxSynchronize

#define cuModuleLoadData            hipModuleLoadData
#define cuModuleUnload              hipModuleUnload
#define cuModuleGetFunction         hipModuleGetFunction
#define cuModuleGetGlobal           hipModuleGetGlobal
#define cuModuleGetFunctionCount    hipModuleGetFunctionCount
#define cuModuleEnumerateFunctions  hipModuleEnumerateFunctions
#define cuFuncGetName               hipFuncGetName

#define cuLaunchKernel              hipModuleLaunchKernel
#define cuLaunchCooperativeKernel   hipModuleLaunchCooperativeKernel

#define cuOccupancyMaxActiveBlocksPerMultiprocessor hipModuleOccupancyMaxActiveBlocksPerMultiprocessor

#define cuEventCreate               hipEventCreateWithFlags
#define cuEventRecord               hipEventRecord
#define cuEventSynchronize          hipEventSynchronize
#define cuEventElapsedTime          hipEventElapsedTime
#define cuEventDestroy              hipEventDestroy
#define CU_EVENT_DEFAULT            hipEventDefault

#define cuMemcpyHtoD                hipMemcpyHtoD
#define cuMemcpyDtoH                hipMemcpyDtoH
#define cuMemcpyDtoD                hipMemcpyDtoD
#define cuMemcpyHtoDAsync           hipMemcpyHtoDAsync
#define cuMemcpy                    hipMemcpy
#define cuMemAlloc                  hipMalloc
#define cuMemFree                   hipFree
#define cuMemAllocHost              hipHostMalloc
#define cuMemFreeHost               hipHostFree

#define cuStreamSynchronize         hipStreamSynchronize
#define CU_STREAM_DEFAULT           hipStreamDefault
#define cuCtxSetCurrent             hipCtxSetCurrent
#define cuMemsetD8Async             hipMemsetD8Async
#define cuMemsetD8                  hipMemsetD8
#define cuMemsetD32                 hipMemsetD32
#define cuMemcpyDtoHAsync           hipMemcpyDtoHAsync

// Function attributes
#define cuFuncGetAttribute          hipFuncGetAttribute
#define CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
#define CU_FUNC_ATTRIBUTE_NUM_REGS              HIP_FUNC_ATTRIBUTE_NUM_REGS
#define CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES     HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES

#define cuGetErrorString            hipDrvGetErrorString
#define cuGetErrorName              hipDrvGetErrorName

// Virtual memory API
#define CUmemGenericAllocationHandle      hipMemGenericAllocationHandle_t
#define CUmemAllocationProp               hipMemAllocationProp
#define CUmemAccessDesc                   hipMemAccessDesc
#define CU_MEM_ALLOCATION_TYPE_PINNED     hipMemAllocationTypePinned
#define CU_MEM_LOCATION_TYPE_DEVICE       hipMemLocationTypeDevice
#define CU_MEM_HANDLE_TYPE_WIN32          hipMemHandleTypeWin32
#define CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR hipMemHandleTypePosixFileDescriptor
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#define CU_MEM_ALLOC_GRANULARITY_MINIMUM  hipMemAllocationGranularityMinimum
#define CU_MEM_ALLOCATION_COMP_GENERIC    0  // HIP has no compression-type enum

#define cuMemGetAllocationGranularity     hipMemGetAllocationGranularity
#define cuMemAddressReserve               hipMemAddressReserve
#define cuMemAddressFree                  hipMemAddressFree
#define cuMemCreate                       hipMemCreate
#define cuMemRelease                      hipMemRelease
#define cuMemMap                          hipMemMap
#define cuMemUnmap                        hipMemUnmap
#define cuMemSetAccess                    hipMemSetAccess
#define cuMemExportToShareableHandle      hipMemExportToShareableHandle
#define cuMemImportFromShareableHandle    hipMemImportFromShareableHandle

// Device attributes
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR hipDeviceAttributeComputeCapabilityMajor
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR hipDeviceAttributeComputeCapabilityMinor
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT     hipDeviceAttributeMultiprocessorCount

// Device functions
#define cuDeviceGetName                 hipDeviceGetName
#define CUuuid                          hipUUID
#define cuDeviceGetUuid                 hipDeviceGetUuid

// External memory (CUDA-Vulkan interop -> HIP-Vulkan interop)
#define CUexternalMemory               hipExternalMemory_t
#define CUmipmappedArray               hipMipmappedArray_t
#define CUsurfObject                   hipSurfaceObject_t
#define CUarray                        hipArray_t

#define CUDA_EXTERNAL_MEMORY_HANDLE_DESC          hipExternalMemoryHandleDesc
#define CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC hipExternalMemoryMipmappedArrayDesc
#define CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD  hipExternalMemoryHandleTypeOpaqueFd
#define CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 hipExternalMemoryHandleTypeOpaqueWin32

#define cuImportExternalMemory                    hipImportExternalMemory
#define cuExternalMemoryGetMappedMipmappedArray   hipExternalMemoryGetMappedMipmappedArray
#define cuDestroyExternalMemory                   hipDestroyExternalMemory
#define cuMipmappedArrayGetLevel                  hipMipmappedArrayGetLevel
#define cuMipmappedArrayDestroy                   hipMipmappedArrayDestroy
#define cuSurfObjectCreate                        hipCreateSurfaceObject
#define cuSurfObjectDestroy                       hipDestroySurfaceObject

// Array format and descriptor types
#define CUDA_ARRAY3D_DESCRIPTOR                   hipArray3DCreate  // Note: different struct name
#define CU_AD_FORMAT_UNSIGNED_INT8                HIP_AD_FORMAT_UNSIGNED_INT8
#define CUDA_ARRAY_3D_SURFACE_LDST                hipArraySurfaceLoadStore
#define CU_RESOURCE_TYPE_ARRAY                    hipResourceTypeArray
#define CUDA_RESOURCE_DESC                        hipResourceDesc

// Context creation params (CUDA 12+ feature, may not have direct HIP equivalent)
#ifdef CUctxCreateParams
#undef CUctxCreateParams
#endif
struct CUctxCreateParams_hip { int dummy; };
#define CUctxCreateParams CUctxCreateParams_hip

// cuCtxCreate with params: HIP uses simpler cuCtxCreate, so adapt signature
// The CUDA 12 signature is: cuCtxCreate(CUcontext*, CUctxCreateParams*, unsigned int, CUdevice)
// HIP signature is: hipCtxCreate(hipCtx_t*, unsigned int flags, hipDevice_t)
// We override this in code where used

// Intrinsics and device functions
// These are mostly compatible between CUDA and HIP
#define __ldg(ptr)                  (*(ptr))  // HIP has __ldg but with subtle differences; safe fallback

// Debug break
#ifdef __debugbreak
#undef __debugbreak
#endif
#define __debugbreak()              __builtin_trap()

// GLM force GPU flag
#ifdef GLM_FORCE_CUDA
#undef GLM_FORCE_CUDA
#endif
// GLM does not need special HIP flag; it works with standard defines

#else // NVIDIA CUDA path

#include "cuda.h"
#include "cuda_runtime.h"
#include <cooperative_groups.h>

// On NVIDIA, CUdeviceptr is uint64_t, so arithmetic is well-formed.
// Provide the same helper name so shared source compiles on both paths.
#define HIP_DEVPTR_ADD(ptr, off)    ((CUdeviceptr)(ptr) + (uint64_t)(off))

#endif // USE_HIP
