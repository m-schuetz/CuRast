#pragma once

#include <print>
#include <mutex>
#include <vector>

#include <vulkan/vulkan.h>
#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#endif

#include "cuda.h"
#include "unsuck.hpp"
#include "VKRenderer.h"
#include "VkExt.h"
#include "CURuntime.h"

using std::println;
using std::mutex;
using std::lock_guard;
using std::vector;

// Growable GPU buffer backed by a single reserved virtual address range on both
// the CUDA and Vulkan sides.  Physical memory is allocated in chunks by commit()
// and mapped into both ranges so CUDA kernels and Vulkan shaders share the same
// underlying pages.
//
// Vulkan is the allocator: each chunk is a VkDeviceMemory with export flags.
// CUDA imports each chunk's handle via cuMemImportFromShareableHandle and maps
// it into a reserved CUDA virtual address range.
//
// Requires sparseBinding device feature (enabled in VKRenderer::createLogicalDevice).
//
// Usage:
//   auto buf = VulkanCudaSharedMemory::create(8ULL * 1024 * 1024 * 1024);
//   buf->commit(needed_bytes);   // grows lazily
//   // buf->cptr          — CUdeviceptr for CUDA kernels
//   // buf->deviceAddress — VkDeviceAddress for Vulkan shaders
struct VulkanCudaSharedMemory {

	static constexpr uint64_t DEFAULT_VIRTUAL_SIZE = 8ULL * 1024 * 1024 * 1024; // 8 GB

	string label = "";
	uint64_t virtualSize   = 0;
	uint64_t comitted = 0;
	uint64_t granularity   = 0;

	CUdeviceptr     cptr          = 0;
	VkBuffer        vk_buffer     = VK_NULL_HANDLE;
	VkDeviceAddress deviceAddress = 0;

	// Hack if something refers to bytes at an offset into the memory.
	// e.g., index buffer is located at byte offset 1234 from the start of this buffer.
	uint64_t offset = 0;

	struct Chunk {
		CUmemGenericAllocationHandle cuHandle = 0;  // obtained via cuMemImportFromShareableHandle
		VkDeviceMemory               vkMemory = VK_NULL_HANDLE;
		uint64_t                     offset   = 0;
		uint64_t                     size     = 0;
	};

	vector<Chunk> chunks;

	void destroy() {
		println("freeing memory. size: {:L}", comitted);

		// Destroy the sparse Vulkan buffer (automatically unbinds all sparse ranges)
		if (vk_buffer != VK_NULL_HANDLE) {
			vkDestroyBuffer(VKRenderer::device, vk_buffer, nullptr);
			vk_buffer     = VK_NULL_HANDLE;
			deviceAddress = 0;
		}

		// Unmap and release CUDA side first (Vulkan owns the physical memory),
		// then free Vulkan memory.
		for (auto& chunk : chunks) {
			if (chunk.cuHandle) {
				cuMemUnmap(cptr + chunk.offset, chunk.size);
				cuMemRelease(chunk.cuHandle);
			}
			if (chunk.vkMemory != VK_NULL_HANDLE)
				vkFreeMemory(VKRenderer::device, chunk.vkMemory, nullptr);
		}
		chunks.clear();

		if (cptr) {
			cuMemAddressFree(cptr, virtualSize);
			cptr = 0;
		}
	}

	// Reserves virtual address space on both the CUDA and Vulkan sides.
	// No physical memory is allocated yet.
	static VulkanCudaSharedMemory* create(uint64_t virtualSize = DEFAULT_VIRTUAL_SIZE, string label = "none") {
		VulkanCudaSharedMemory* mem = new VulkanCudaSharedMemory();

		CUdevice cuDevice;
		cuDeviceGet(&cuDevice, 0);

		// CUDA granularity is still needed for the virtual address range and
		// cuMemMap — use a plain (non-exportable) prop just to query it.
		CUmemAllocationProp granProp{};
		granProp.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
		granProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		granProp.location.id   = cuDevice;

		uint64_t granularity = 0;
		cuMemGetAllocationGranularity(&granularity, &granProp, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

		uint64_t paddedSize   = roundUp(virtualSize, granularity);
		mem->granularity      = granularity;
		mem->virtualSize      = paddedSize;

		// Reserve CUDA virtual address range (no physical memory yet)
		CUdeviceptr cptr = 0;
		CUresult cuRes = cuMemAddressReserve(&cptr, paddedSize, 0, 0, 0);
		if (cuRes != CUDA_SUCCESS) {
			println("VulkanCudaSharedMemory::create: cuMemAddressReserve failed ({})", int(cuRes));
			return nullptr;
		}
		mem->cptr = cptr;

		// Create a sparse Vulkan buffer covering the full virtual range.
		// VkExternalMemoryBufferCreateInfo tells Vulkan that the memory bound
		// to this buffer will be exportable (Vulkan allocates and exports to CUDA).
#ifdef _WIN32
		VkExternalMemoryBufferCreateInfo extBufInfo = {};
		extBufInfo.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
		extBufInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
		VkExternalMemoryBufferCreateInfo extBufInfo = {};
		extBufInfo.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
		extBufInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

		VkBufferCreateInfo bci = {};
		bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bci.pNext       = &extBufInfo;
		bci.size        = paddedSize;
		bci.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
		                | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
		                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		                | VK_BUFFER_USAGE_TRANSFER_DST_BIT
		                | VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT
		                | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
		bci.flags       = VK_BUFFER_CREATE_SPARSE_BINDING_BIT;
		bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VkResult vkRes = vkCreateBuffer(VKRenderer::device, &bci, nullptr, &mem->vk_buffer);
		if (vkRes != VK_SUCCESS) {
			println("VulkanCudaSharedMemory::create: vkCreateBuffer failed ({})", int(vkRes));
			return nullptr;
		}

		// For sparse buffers the device address is stable for the buffer's lifetime
		// and can be queried before any memory is bound.
		VkBufferDeviceAddressInfo addrInfo = {};
		addrInfo.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
		addrInfo.buffer = mem->vk_buffer;
		mem->deviceAddress = vkGetBufferDeviceAddress(VKRenderer::device, &addrInfo);

		mem->label = label;

		return mem;
	}

	// Ensures at least <requested_size> bytes of physical memory are committed.
	// If more is already committed, this is a no-op.
	//
	// Three-phase design:
	//   Phase 1 — Vulkan: vkAllocateMemory (with VkExportMemoryAllocateInfo) for
	//             each new chunk. Physical memory is owned by Vulkan.
	//   Phase 2 — CUDA: export the Vulkan handle, import it with
	//             cuMemImportFromShareableHandle, cuMemMap into the virtual range,
	//             then ONE cuMemSetAccess covering only the new range.
	//             cuMemSetAccess must not cover previously Vulkan-sparse-bound
	//             pages; doing so causes CUDA_ERROR_NOT_READY on the 13th+
	//             call because Vulkan's sparse binds leave the shared NVIDIA
	//             GPU MMU in a deferred-update state that CUDA detects.
	//   Phase 3 — Vulkan: batch all new sparse binds in a single
	//             vkQueueBindSparse call (after CUDA has set access).
	void commit(uint64_t requested_size) {
		static mutex mtx;
		lock_guard<mutex> lock(mtx);

		VkMemoryRequirements memReqs;
		vkGetBufferMemoryRequirements(VKRenderer::device, vk_buffer, &memReqs);

		uint64_t phaseStartOffset = comitted; // first new byte
		size_t   firstNewChunk    = chunks.size(); // index of first new chunk

		// ----------------------------------------------------------------
		// Phase 1: Vulkan — allocate exportable memory for all new chunks.
		// ----------------------------------------------------------------
		while (comitted < requested_size) {
			uint64_t diff             = requested_size - comitted;
			uint64_t stepSize         = min(diff, 1'000'000'000llu);
			uint64_t currentRequested = comitted + stepSize;
			uint64_t padded           = roundUp(currentRequested, granularity);
			if (padded <= comitted) break;

			uint64_t chunkOffset = comitted;
			uint64_t chunkSize   = padded - comitted;

#ifdef _WIN32
			SECURITY_ATTRIBUTES secAttr{};
			secAttr.nLength        = sizeof(SECURITY_ATTRIBUTES);
			secAttr.bInheritHandle = FALSE;

			VkExportMemoryWin32HandleInfoKHR exportWin32Info = {};
			exportWin32Info.sType      = VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
			exportWin32Info.pAttributes = &secAttr;
			exportWin32Info.dwAccess   = GENERIC_ALL;

			VkExportMemoryAllocateInfo exportInfo = {};
			exportInfo.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
			exportInfo.pNext       = &exportWin32Info;
			exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
			VkExportMemoryAllocateInfo exportInfo = {};
			exportInfo.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
			exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

			VkMemoryAllocateFlagsInfo allocFlags = {};
			allocFlags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
			allocFlags.pNext = &exportInfo;
			allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

			VkMemoryAllocateInfo allocInfo = {};
			allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.pNext           = &allocFlags;
			allocInfo.allocationSize  = chunkSize;
			allocInfo.memoryTypeIndex = VKRenderer::findMemoryType(
				memReqs.memoryTypeBits, 
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT //| VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
			);

			VkDeviceMemory vkMemory = VK_NULL_HANDLE;
			VkResult vkRes = vkAllocateMemory(VKRenderer::device, &allocInfo, nullptr, &vkMemory);
			if (vkRes != VK_SUCCESS) {
				println("VulkanCudaSharedMemory::commit: vkAllocateMemory failed ({})", int(vkRes));
				println("{}", stacktrace::current());
				exit(26245762354);
			}

			chunks.push_back({ 0, vkMemory, chunkOffset, chunkSize });
			comitted = padded;
		}

		if (comitted == phaseStartOffset) return; // nothing new to do

		// ----------------------------------------------------------------
		// Phase 2: CUDA — export Vulkan handles, import, map, set access.
		//
		// cuMemSetAccess covers only the new range — do not touch previously
		// Vulkan-sparse-bound pages (they already have access and including
		// them triggers CUDA_ERROR_NOT_READY from Vulkan MMU state).
		// ----------------------------------------------------------------
		CUdevice cuDevice;
		cuDeviceGet(&cuDevice, 0);

		for (size_t i = firstNewChunk; i < chunks.size(); i++) {
			auto& chunk = chunks[i];

#ifdef _WIN32
			VkMemoryGetWin32HandleInfoKHR getHandleInfo = {};
			getHandleInfo.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
			getHandleInfo.memory     = chunk.vkMemory;
			getHandleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

			HANDLE win32Handle = nullptr;
			VkResult vkRes = vkGetMemoryWin32HandleKHR(VKRenderer::device, &getHandleInfo, &win32Handle);
			if (vkRes != VK_SUCCESS) {
				println("VulkanCudaSharedMemory::commit: vkGetMemoryWin32HandleKHR failed ({})", int(vkRes));
				println("{}", stacktrace::current());
				exit(26245762355);
			}

			CUmemGenericAllocationHandle cuHandle = 0;
			CUresult cuRes = cuMemImportFromShareableHandle(&cuHandle, (void*)win32Handle, CU_MEM_HANDLE_TYPE_WIN32);
			CloseHandle(win32Handle); // CUDA duplicates the NT handle internally
			CURuntime::assertCudaSuccess(cuRes);
#else
			VkMemoryGetFdInfoKHR getFdInfo = {};
			getFdInfo.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
			getFdInfo.memory     = chunk.vkMemory;
			getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

			int fd = -1;
			VkResult vkRes = vkGetMemoryFdKHR(VKRenderer::device, &getFdInfo, &fd);
			if (vkRes != VK_SUCCESS) {
				println("VulkanCudaSharedMemory::commit: vkGetMemoryFdKHR failed ({})", int(vkRes));
				println("{}", stacktrace::current());
				exit(26245762355);
			}

			CUmemGenericAllocationHandle cuHandle = 0;
			// fd is consumed (closed) by cuMemImportFromShareableHandle on Linux
			CUresult cuRes = cuMemImportFromShareableHandle(&cuHandle, (void*)(uintptr_t)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
			CURuntime::assertCudaSuccess(cuRes);
#endif

			chunk.cuHandle = cuHandle;

			cuCtxSynchronize();
			cuRes = cuMemMap(cptr + chunk.offset, chunk.size, 0, cuHandle, 0);
			cuCtxSynchronize();
			CURuntime::assertCudaSuccess(cuRes);
		}

		// Set access ONCE for the new range only
		CUmemAccessDesc accessDesc{};
		accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		accessDesc.location.id   = cuDevice;
		accessDesc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

		cuCtxSynchronize();
		CUresult cuRes = cuMemSetAccess(cptr + phaseStartOffset,
		                                comitted - phaseStartOffset,
		                                &accessDesc, 1);
		cuCtxSynchronize();
		CURuntime::assertCudaSuccess(cuRes);

		// ----------------------------------------------------------------
		// Phase 3: Vulkan — sparse-bind all new chunks in a single call.
		// ----------------------------------------------------------------
		vector<VkSparseMemoryBind> binds;
		binds.reserve(chunks.size() - firstNewChunk);

		for (size_t i = firstNewChunk; i < chunks.size(); i++) {
			auto& chunk = chunks[i];
			binds.push_back({ chunk.offset, chunk.size, chunk.vkMemory, 0, 0 });
		}

		if (!binds.empty()) {
			VkSparseBufferMemoryBindInfo bufferBind = {};
			bufferBind.buffer    = vk_buffer;
			bufferBind.bindCount = (uint32_t)binds.size();
			bufferBind.pBinds    = binds.data();

			VkBindSparseInfo sparseInfo = {};
			sparseInfo.sType           = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
			sparseInfo.bufferBindCount = 1;
			sparseInfo.pBufferBinds    = &bufferBind;

			vkQueueBindSparse(VKRenderer::graphicsQueue, 1, &sparseInfo, VK_NULL_HANDLE);
			vkQueueWaitIdle(VKRenderer::graphicsQueue);
			vkDeviceWaitIdle(VKRenderer::device);
		}
	}

	void memcopyHtoD(uint64_t offset, void* source, uint64_t size, stacktrace trace = stacktrace::current()){

		bool validRange = true;
		validRange = validRange && offset >= 0 && offset < comitted;
		validRange = validRange && (offset + size) < comitted;

		if(!validRange){
			println("ERROR: Attempted to memcpy to unallocated or uncomitted range.");
			println("    cptr:          {:15L}", cptr);
			println("    comitted:      {:15L}", comitted);
			println("    target offset: {:15L}", offset);
			println("    source size:   {:15L}", size);

			println("{}", trace);
			__debugbreak();
			exit(652345345);
		}

		CUresult result = cuMemcpyHtoD(cptr + offset, source, size);
		CURuntime::assertCudaSuccess(result);
	}
};
