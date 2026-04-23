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
// Requires sparseBinding device feature (enabled in VKRenderer::createLogicalDevice).
//
// Usage:
//   auto buf = CudaVulkanSharedMemory::create(8ULL * 1024 * 1024 * 1024);
//   buf->commit(needed_bytes);   // grows lazily
//   // buf->cptr          — CUdeviceptr for CUDA kernels
//   // buf->deviceAddress — VkDeviceAddress for Vulkan shaders
struct CudaVulkanSharedMemory {

	static constexpr uint64_t DEFAULT_VIRTUAL_SIZE = 8ULL * 1024 * 1024 * 1024; // 8 GB

	uint64_t virtualSize   = 0;
	uint64_t committedSize = 0;
	uint64_t granularity   = 0;

	CUdeviceptr     cptr          = 0;
	VkBuffer        vk_buffer     = VK_NULL_HANDLE;
	VkDeviceAddress deviceAddress = 0;

	struct Chunk {
		CUmemGenericAllocationHandle cuHandle = 0;
		VkDeviceMemory               vkMemory = VK_NULL_HANDLE;
		uint64_t                     offset   = 0;
		uint64_t                     size     = 0;
	};

	vector<Chunk> chunks;

	CudaVulkanSharedMemory() = default;

	~CudaVulkanSharedMemory() {
		destroy();
	}

	void destroy() {
		// Destroy the sparse Vulkan buffer (automatically unbinds all sparse ranges)
		if (vk_buffer != VK_NULL_HANDLE) {
			vkDestroyBuffer(VKRenderer::device, vk_buffer, nullptr);
			vk_buffer     = VK_NULL_HANDLE;
			deviceAddress = 0;
		}

		// Free Vulkan memory and unmap CUDA chunks
		for (auto& chunk : chunks) {
			if (chunk.vkMemory != VK_NULL_HANDLE)
				vkFreeMemory(VKRenderer::device, chunk.vkMemory, nullptr);
			cuMemUnmap(cptr + chunk.offset, chunk.size);
			if (chunk.cuHandle)
				cuMemRelease(chunk.cuHandle);
		}
		chunks.clear();

		if (cptr) {
			cuMemAddressFree(cptr, virtualSize);
			cptr = 0;
		}
	}

	// Reserves virtual address space on both the CUDA and Vulkan sides.
	// No physical memory is allocated yet.
	static std::shared_ptr<CudaVulkanSharedMemory> create(uint64_t virtualSize = DEFAULT_VIRTUAL_SIZE) {
		auto mem = std::make_shared<CudaVulkanSharedMemory>();

		CUdevice cuDevice;
		cuDeviceGet(&cuDevice, 0);

		// Determine minimum granularity for exportable CUDA allocations
		CUmemAllocationProp prop{};
		prop.type              = CU_MEM_ALLOCATION_TYPE_PINNED;
		prop.location.type     = CU_MEM_LOCATION_TYPE_DEVICE;
		prop.location.id       = cuDevice;
#ifdef _WIN32
		prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_WIN32;
		SECURITY_ATTRIBUTES secAttr{};
		secAttr.nLength        = sizeof(SECURITY_ATTRIBUTES);
		secAttr.bInheritHandle = FALSE;
		prop.win32HandleMetaData = &secAttr;
#else
		prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif

		uint64_t granularity = 0;
		cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

		uint64_t paddedSize   = roundUp(virtualSize, granularity);
		mem->granularity      = granularity;
		mem->virtualSize      = paddedSize;

		// Reserve CUDA virtual address range (no physical memory yet)
		CUdeviceptr cptr = 0;
		CUresult cuRes = cuMemAddressReserve(&cptr, paddedSize, 0, 0, 0);
		if (cuRes != CUDA_SUCCESS) {
			println("CudaVulkanSharedMemory::create: cuMemAddressReserve failed ({})", int(cuRes));
			return nullptr;
		}
		mem->cptr = cptr;

		// Create a sparse Vulkan buffer covering the full virtual range.
		// VK_BUFFER_CREATE_SPARSE_BINDING_BIT means memory is bound incrementally
		// via vkQueueBindSparse rather than a single vkBindBufferMemory call.
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
		                | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		bci.flags       = VK_BUFFER_CREATE_SPARSE_BINDING_BIT;
		bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VkResult vkRes = vkCreateBuffer(VKRenderer::device, &bci, nullptr, &mem->vk_buffer);
		if (vkRes != VK_SUCCESS) {
			println("CudaVulkanSharedMemory::create: vkCreateBuffer failed ({})", int(vkRes));
			return nullptr;
		}

		// For sparse buffers the device address is stable for the buffer's lifetime
		// and can be queried before any memory is bound.
		VkBufferDeviceAddressInfo addrInfo = {};
		addrInfo.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
		addrInfo.buffer = mem->vk_buffer;
		mem->deviceAddress = vkGetBufferDeviceAddress(VKRenderer::device, &addrInfo);

		return mem;
	}

	// Ensures at least <requested_size> bytes of physical memory are committed.
	// If more is already committed, this is a no-op.
	//
	// Two-phase design to avoid GPU MMU conflicts:
	//   Phase 1 — all CUDA work:  cuMemCreate + cuMemMap for every new chunk,
	//             then ONE cuMemSetAccess covering only the new range.
	//             cuMemSetAccess must not cover previously Vulkan-sparse-bound
	//             pages; doing so causes CUDA_ERROR_NOT_READY on the 13th+
	//             call because Vulkan's sparse binds leave the shared NVIDIA
	//             GPU MMU in a deferred-update state that CUDA detects.
	//   Phase 2 — all Vulkan work: export handles, vkAllocateMemory, batch
	//             all new sparse binds in a single vkQueueBindSparse call.
	void commit(uint64_t requested_size) {
		static mutex mtx;
		lock_guard<mutex> lock(mtx);

		CUdevice cuDevice;
		cuDeviceGet(&cuDevice, 0);

		CUmemAllocationProp prop{};
		prop.type              = CU_MEM_ALLOCATION_TYPE_PINNED;
		prop.location.type     = CU_MEM_LOCATION_TYPE_DEVICE;
		prop.location.id       = cuDevice;
#ifdef _WIN32
		prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_WIN32;
		SECURITY_ATTRIBUTES secAttr{};
		secAttr.nLength        = sizeof(SECURITY_ATTRIBUTES);
		secAttr.bInheritHandle = FALSE;
		prop.win32HandleMetaData = &secAttr;
#else
		prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif

		// ----------------------------------------------------------------
		// Phase 1: CUDA — map all new chunks, then set access once.
		//
		// Commit in steps of < 2^32 bytes because shared allocations
		// cannot handle a single chunk that large.
		// ----------------------------------------------------------------
		uint64_t phaseStartOffset = committedSize; // first new byte

		while (committedSize < requested_size) {
			uint64_t diff             = requested_size - committedSize;
			uint64_t stepSize         = min(diff, 1'000'000'000llu);
			uint64_t currentRequested = committedSize + stepSize;
			uint64_t padded           = roundUp(currentRequested, granularity);
			if (padded <= committedSize) break;

			uint64_t chunkOffset = committedSize;
			uint64_t chunkSize   = padded - committedSize;

			cuCtxSynchronize();
			CUmemGenericAllocationHandle cuHandle = 0;
			CUresult cuRes = cuMemCreate(&cuHandle, chunkSize, &prop, 0);
			CURuntime::assertCudaSuccess(cuRes);

			cuCtxSynchronize();
			cuRes = cuMemMap(cptr + chunkOffset, chunkSize, 0, cuHandle, 0);
			cuCtxSynchronize();
			CURuntime::assertCudaSuccess(cuRes);

			// Store handle; Vulkan import happens in Phase 2.
			chunks.push_back({ cuHandle, VK_NULL_HANDLE, chunkOffset, chunkSize });
			committedSize = padded;
		}

		if (committedSize == phaseStartOffset) return; // nothing new to do

		// Set access ONCE for the new range only — do not touch previously
		// Vulkan-sparse-bound pages (they already have access and including
		// them triggers CUDA_ERROR_NOT_READY from Vulkan MMU state).
		CUmemAccessDesc accessDesc{};
		accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		accessDesc.location.id   = cuDevice;
		accessDesc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

		cuCtxSynchronize();
		CUresult cuRes = cuMemSetAccess(cptr + phaseStartOffset,
		                                committedSize - phaseStartOffset,
		                                &accessDesc, 1);
		cuCtxSynchronize();
		CURuntime::assertCudaSuccess(cuRes);

		// ----------------------------------------------------------------
		// Phase 2: Vulkan — import and sparse-bind all new chunks at once.
		// ----------------------------------------------------------------
		VkMemoryRequirements memReqs;
		vkGetBufferMemoryRequirements(VKRenderer::device, vk_buffer, &memReqs);

		vector<VkSparseMemoryBind> binds;
		binds.reserve(chunks.size());

		//for (auto& chunk : chunks) {
		for (int chunkIndex = 0; chunkIndex < chunks.size(); chunkIndex++) {
			auto& chunk = chunks[chunkIndex];
			
			if (chunk.vkMemory != VK_NULL_HANDLE) continue; // already bound

#ifdef _WIN32
			HANDLE win32Handle = nullptr;
			cuRes = cuMemExportToShareableHandle(&win32Handle, chunk.cuHandle, CU_MEM_HANDLE_TYPE_WIN32, 0);
			CURuntime::assertCudaSuccess(cuRes);

			VkImportMemoryWin32HandleInfoKHR importInfo = {};
			importInfo.sType      = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
			importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
			importInfo.handle     = win32Handle;
#else
			int fd = -1;
			cuRes = cuMemExportToShareableHandle(&fd, chunk.cuHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
			CURuntime::assertCudaSuccess(cuRes);

			VkImportMemoryFdInfoKHR importInfo = {};
			importInfo.sType      = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
			importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
			importInfo.fd         = fd;
#endif

			VkMemoryAllocateFlagsInfo allocFlags = {};
			allocFlags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
			allocFlags.pNext = &importInfo;
			allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

			VkMemoryAllocateInfo allocInfo = {};
			allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.pNext           = &allocFlags;
			allocInfo.allocationSize  = chunk.size;
			allocInfo.memoryTypeIndex = VKRenderer::findMemoryType(
				memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

			VkResult vkRes = vkAllocateMemory(VKRenderer::device, &allocInfo, nullptr, &chunk.vkMemory);
			if (vkRes != VK_SUCCESS) {
				println("CudaVulkanSharedMemory::commit: vkAllocateMemory failed ({})", int(vkRes));
#ifdef _WIN32
				CloseHandle(win32Handle);
#endif
				println("{}", stacktrace::current());
				exit(26245762354);
			}

#ifdef _WIN32
			CloseHandle(win32Handle); // Vulkan holds its own NT handle reference
#endif

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
		validRange = validRange && offset >= 0 && offset < committedSize;
		validRange = validRange && (offset + size) < committedSize;

		if(!validRange){
			println("ERROR: Attempted to memcpy to unallocated or uncomitted range.");
			println("    cptr:          {:15L}", cptr);
			println("    comitted:      {:15L}", committedSize);
			println("    target offset: {:15L}", offset);
			println("    source size:   {:15L}", size);

			println("{}", trace);

			exit(652345345);
		}

		CUresult result = cuMemcpyHtoD(cptr + offset, source, size);
		CURuntime::assertCudaSuccess(result);
	}
};
