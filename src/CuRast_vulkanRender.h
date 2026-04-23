#include "shaders/mesh_indexpulling.h"
#include "shaders/mesh.h"
#include "shaders/mesh_visbuffer.h"
#include "shaders/mesh_visbuffer_resolve.h"

struct VKNode {
	glm::mat4 world;
	glm::mat4 worldViewProj;
	uint64_t  ptr_position;
	uint64_t  ptr_uv;
	uint64_t  ptr_indices;
	uint32_t  textureHandle;
	int32_t   width;
	int32_t   height;
	vec3      compressionFactor;
	float     min_x;
	float     min_y;
	float     min_z;
	uint32_t  index_min;
	uint32_t  bitsPerIndex;
};

// Must match PC push constant in mesh_indexpulling.slang:
//   offset  0: viewProj   (column_major float4x4, 64 bytes)
//   offset 64: nodes      (Node* BDA, 8 bytes)
struct MeshPC {
	glm::mat4 viewProj;
	uint64_t  nodesAddr;
};

// Must match PC push constant in mesh_visbuffer.slang (pass 1)
struct MeshPC_VisbufPass1 {
	glm::mat4 viewProj;
	uint64_t  nodesAddr;
	uint64_t  cumulativeTriCountsAddr;
};

// Must match PC push constant in mesh_visbuffer_resolve.slang
// Layout (std430, all offsets in bytes):
//   0:   viewProj              (mat4,  64)
//  64:   nodesAddr             (u64,    8)
//  72:   screenWidth           (u32,    4)
//  76:   screenHeight          (u32,    4)
//  80:   cumulativeTriCounts   (u64,    8)
//  88:   nodeCount             (u32,    4)
//  92:   pad0                  (u32,    4)
//  96:   cameraPos             (vec4,  16)  world-space camera position
// 112:   tanHalfFov            (vec2,   8)  (1/proj[0][0], 1/proj[1][1])
// 120:   viewDirPad            (vec2,   8)  alignment padding
// 128:   viewInvRow0           (vec4,  16)  row 0 of inverse-view matrix
// 144:   viewInvRow1           (vec4,  16)  row 1
// 160:   viewInvRow2           (vec4,  16)  row 2
// Total: 176 bytes
struct MeshPC_Resolve {
	glm::mat4 viewProj;
	uint64_t  nodesAddr;
	uint32_t  screenWidth;
	uint32_t  screenHeight;
	uint64_t  cumulativeTriCountsAddr;
	uint32_t  nodeCount;
	uint32_t  pad0;
	glm::vec4 cameraPos;
	glm::vec2 tanHalfFov;
	glm::vec2 viewDirPad;
	glm::vec4 viewInvRow0;
	glm::vec4 viewInvRow1;
	glm::vec4 viewInvRow2;
};
static_assert(offsetof(MeshPC_Resolve, cameraPos)   ==  96);
static_assert(offsetof(MeshPC_Resolve, tanHalfFov)  == 112);
static_assert(offsetof(MeshPC_Resolve, viewInvRow0) == 128);
static_assert(sizeof(MeshPC_Resolve)                == 176);


void lazyInitVulkanFromCudaResources(SNTriangles* node, VkSampler sampler){

	static unordered_map<Texture*, shared_ptr<VKTexture>> textureCache;

	// VKTexture (plain sampled image — no CUDA interop needed)
	if(!node->vkTexture){

		if(textureCache.contains(node->texture)){
			node->vkTexture = textureCache[node->texture];

			VkImageViewHandleInfoNVX handleInfo = {};
			handleInfo.sType      = VK_STRUCTURE_TYPE_IMAGE_VIEW_HANDLE_INFO_NVX;
			handleInfo.imageView  = node->vkTexture->view;
			handleInfo.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			handleInfo.sampler    = sampler;
			node->vkTextureHandle = vkGetImageViewHandleNVX(VKRenderer::device, &handleInfo);
		}else{

			node->vkTexture = std::make_shared<VKTexture>();
			VKTexture& tex  = *node->vkTexture;
			tex.width  = node->texture->width;
			tex.height = node->texture->height;
			tex.format = VK_FORMAT_R8G8B8A8_UNORM;
			tex.label  = node->name + "_tex";
			tex.ID     = VKTexture::idcounter++;

			VkDeviceSize imageBytes = (VkDeviceSize)tex.width * tex.height * 4;

			println("Copying image data from cuda memory to host memory");
			vector<uint8_t> hostData(imageBytes);
			CUresult cures = cuMemcpyDtoH(hostData.data(), (CUdeviceptr)node->texture->data, imageBytes);
			CURuntime::assertCudaSuccess(cures);

			// HACK: Remove CUDA allocation of textures because for the venice data set, 
			// we don't have enough GPU memory to have the textures in GPU twice. 
			MemoryManager::free((CUdeviceptr)node->texture->data);

			// Device-local sampled image (no external memory export — Vulkan-only)
			uint32_t mipLevels = 1 + (uint32_t)floor(log2((float)max(tex.width, tex.height)));

			VkImageCreateInfo ici = {};
			ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			ici.imageType     = VK_IMAGE_TYPE_2D;
			ici.format        = tex.format;
			ici.extent        = { (uint32_t)tex.width, (uint32_t)tex.height, 1 };
			ici.mipLevels     = mipLevels;
			ici.arrayLayers   = 1;
			ici.samples       = VK_SAMPLE_COUNT_1_BIT;
			ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
			ici.usage         = VK_IMAGE_USAGE_SAMPLED_BIT
							| VK_IMAGE_USAGE_TRANSFER_SRC_BIT
							| VK_IMAGE_USAGE_TRANSFER_DST_BIT;
			ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
			vkCreateImage(VKRenderer::device, &ici, nullptr, &tex.image);
			
			VkMemoryRequirements memReqs;
			vkGetImageMemoryRequirements(VKRenderer::device, tex.image, &memReqs);
			VkMemoryAllocateInfo allocInfo = {};
			allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize  = memReqs.size;
			allocInfo.memoryTypeIndex = VKRenderer::findMemoryType(memReqs.memoryTypeBits,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			
			println("Allocating image memory: {:10L}", memReqs.size);
			VkResult res = vkAllocateMemory(VKRenderer::device, &allocInfo, nullptr, &tex.memory);
			VKRenderer::assertSucces(res);
			res = vkBindImageMemory(VKRenderer::device, tex.image, tex.memory, 0);
			VKRenderer::assertSucces(res);

			VkImageViewCreateInfo vci = {};
			vci.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			vci.image            = tex.image;
			vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
			vci.format           = tex.format;
			vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1 };
			res = vkCreateImageView(VKRenderer::device, &vci, nullptr, &tex.view);
			VKRenderer::assertSucces(res);


			VkBuffer stagingBuffer = VK_NULL_HANDLE;
			VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
			{
				VkBufferCreateInfo bci = {};
				bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
				bci.size  = imageBytes;
				bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
				res = vkCreateBuffer(VKRenderer::device, &bci, nullptr, &stagingBuffer);
				VKRenderer::assertSucces(res);

				VkMemoryRequirements stagingReqs;
				vkGetBufferMemoryRequirements(VKRenderer::device, stagingBuffer, &stagingReqs);
				VkMemoryAllocateInfo stagingAlloc = {};
				stagingAlloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
				stagingAlloc.allocationSize  = stagingReqs.size;
				stagingAlloc.memoryTypeIndex = VKRenderer::findMemoryType(stagingReqs.memoryTypeBits,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

				println("vkAllocateMemory, size: {:L}", stagingAlloc.allocationSize);
				res = vkAllocateMemory(VKRenderer::device, &stagingAlloc, nullptr, &stagingMemory);
				VKRenderer::assertSucces(res);
				println("vkBindBufferMemory");
				res = vkBindBufferMemory(VKRenderer::device, stagingBuffer, stagingMemory, 0);
				VKRenderer::assertSucces(res);

				println("DEBUG: vkMapMemory, size: {:L}", imageBytes);
				void* mapped;
				res = vkMapMemory(VKRenderer::device, stagingMemory, 0, imageBytes, 0, &mapped);
				VKRenderer::assertSucces(res);
				memcpy(mapped, hostData.data(), imageBytes);
				
				println("DEBUG: vkUnmapMemory");
				vkUnmapMemory(VKRenderer::device, stagingMemory);
			}

			if (mipLevels > 1){
				auto cmd = VKRenderer::beginSingleTimeCommands();

				// mip 0: UNDEFINED → TRANSFER_DST_OPTIMAL, copy from staging buffer, then → GENERAL
				{
					VkImageMemoryBarrier2 b = {};
					b.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
					b.srcStageMask     = VK_PIPELINE_STAGE_2_NONE;
					b.srcAccessMask    = VK_ACCESS_2_NONE;
					b.dstStageMask     = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
					b.dstAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT;
					b.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
					b.newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
					b.image            = tex.image;
					b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
					VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
					dep.imageMemoryBarrierCount = 1;
					dep.pImageMemoryBarriers    = &b;
					vkCmdPipelineBarrier2(cmd, &dep);
				}
				{
					VkBufferImageCopy2 region = {};
					region.sType             = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
					region.bufferOffset      = 0;
					region.imageSubresource  = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
					region.imageExtent       = { (uint32_t)tex.width, (uint32_t)tex.height, 1 };
					VkCopyBufferToImageInfo2 copyInfo = {};
					copyInfo.sType          = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2;
					copyInfo.srcBuffer      = stagingBuffer;
					copyInfo.dstImage       = tex.image;
					copyInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
					copyInfo.regionCount    = 1;
					copyInfo.pRegions       = &region;
					vkCmdCopyBufferToImage2(cmd, &copyInfo);
				}
				{
					VkImageMemoryBarrier2 b = {};
					b.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
					b.srcStageMask     = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
					b.srcAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT;
					b.dstStageMask     = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
					b.dstAccessMask    = VK_ACCESS_2_TRANSFER_READ_BIT;
					b.oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
					b.newLayout        = VK_IMAGE_LAYOUT_GENERAL;
					b.image            = tex.image;
					b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
					VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
					dep.imageMemoryBarrierCount = 1;
					dep.pImageMemoryBarriers    = &b;
					vkCmdPipelineBarrier2(cmd, &dep);
				}

				int32_t mipW = tex.width;
				int32_t mipH = tex.height;
				for(uint32_t level = 1; level < mipLevels; level++){
					int32_t nextW = max(1, mipW / 2);
					int32_t nextH = max(1, mipH / 2);

					// Transition this mip level UNDEFINED → TRANSFER_DST_OPTIMAL
					{
						VkImageMemoryBarrier2 dstBarrier = {};
						dstBarrier.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
						dstBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_NONE;
						dstBarrier.srcAccessMask = VK_ACCESS_2_NONE;
						dstBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
						dstBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
						dstBarrier.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
						dstBarrier.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
						dstBarrier.image         = tex.image;
						dstBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, 1 };

						VkDependencyInfo dep = {};
						dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
						dep.imageMemoryBarrierCount = 1;
						dep.pImageMemoryBarriers    = &dstBarrier;
						vkCmdPipelineBarrier2(cmd, &dep);
					}

					// Blit from level-1 (GENERAL) → level (TRANSFER_DST_OPTIMAL)
					{
						VkImageBlit2 blit = {};
						blit.sType          = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;
						blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1 };
						blit.srcOffsets[0]  = { 0, 0, 0 };
						blit.srcOffsets[1]  = { mipW, mipH, 1 };
						blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1 };
						blit.dstOffsets[0]  = { 0, 0, 0 };
						blit.dstOffsets[1]  = { nextW, nextH, 1 };

						VkBlitImageInfo2 blitInfo = {};
						blitInfo.sType          = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
						blitInfo.srcImage       = tex.image;
						blitInfo.srcImageLayout = VK_IMAGE_LAYOUT_GENERAL;
						blitInfo.dstImage       = tex.image;
						blitInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
						blitInfo.regionCount    = 1;
						blitInfo.pRegions       = &blit;
						blitInfo.filter         = VK_FILTER_LINEAR;
						vkCmdBlitImage2(cmd, &blitInfo);
					}

					// Transition this level TRANSFER_DST_OPTIMAL → GENERAL
					{
						VkImageMemoryBarrier2 readyBarrier = {};
						readyBarrier.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
						readyBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
						readyBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
						readyBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT
												| VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
						readyBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT
												| VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
						readyBarrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
						readyBarrier.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
						readyBarrier.image         = tex.image;
						readyBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, 1 };

						VkDependencyInfo dep = {};
						dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
						dep.imageMemoryBarrierCount = 1;
						dep.pImageMemoryBarriers    = &readyBarrier;
						vkCmdPipelineBarrier2(cmd, &dep);
					}

					mipW = nextW;
					mipH = nextH;
				}

				VkImageViewHandleInfoNVX handleInfo = {};
				handleInfo.sType      = VK_STRUCTURE_TYPE_IMAGE_VIEW_HANDLE_INFO_NVX;
				handleInfo.imageView  = node->vkTexture->view;
				handleInfo.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				handleInfo.sampler    = sampler;
				node->vkTextureHandle = vkGetImageViewHandleNVX(VKRenderer::device, &handleInfo);

				VKRenderer::endSingleTimeCommands(cmd);
			}

			vkDestroyBuffer(VKRenderer::device, stagingBuffer, nullptr);
			vkFreeMemory(VKRenderer::device, stagingMemory, nullptr);

			textureCache[node->texture] = node->vkTexture;
		}
	}

}

static VkSampler sampler = VK_NULL_HANDLE;
static VKShader* sh_mesh_instancepulling = nullptr;
static VKShader* sh_mesh_instancepulling_compressed = nullptr;
static VKShader* sh_mesh_indexedDraw = nullptr;

static VKShader* sh_visbuffer_pass1 = nullptr;
static VKShader* sh_visbuffer_resolve = nullptr;
static VKShader* sh_visbuffer_pass1_compressed = nullptr;
static VKShader* sh_visbuffer_resolve_compressed = nullptr;

static VKBuffer* buffer_nodes = nullptr;
static VKBuffer* buffer_drawCmds = nullptr;
static VKBuffer* buffer_cumulativeTriCounts = nullptr;
static vector<VkImage>       depthImages{VKRenderer::FRAMES_IN_FLIGHT};
static vector<VkDeviceMemory> depthMemories{VKRenderer::FRAMES_IN_FLIGHT};
static vector<VkImageView>   depthViews{VKRenderer::FRAMES_IN_FLIGHT};
static vector<VkImage>       visbufImages{VKRenderer::FRAMES_IN_FLIGHT};
static vector<VkDeviceMemory> visbufMemories{VKRenderer::FRAMES_IN_FLIGHT};
static vector<VkImageView>   visbufViews{VKRenderer::FRAMES_IN_FLIGHT};
static VkDescriptorSetLayout visbufDSL  = VK_NULL_HANDLE;
static VkDescriptorPool      visbufPool = VK_NULL_HANDLE;
static vector<VkDescriptorSet> visbufSets{VKRenderer::FRAMES_IN_FLIGHT};

static void init(){
	static bool initialized = false;
	if(initialized) return;

	sh_mesh_instancepulling = new VKShader(
		vector<VKShader::Stage>{
			{VK_SHADER_STAGE_VERTEX_BIT,   mesh_shaders_indexpulling_spv, "main_vertex_instanced"},
			{VK_SHADER_STAGE_FRAGMENT_BIT, mesh_shaders_indexpulling_spv, "main_fragment"},
		},
		std::vector<VkPushConstantRange>{{
			VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			0, sizeof(MeshPC)
		}}
	);

	sh_mesh_instancepulling_compressed = new VKShader(
		vector<VKShader::Stage>{
			{VK_SHADER_STAGE_VERTEX_BIT,   mesh_shaders_indexpulling_spv, "main_vertex_compressed_instanced"},
			{VK_SHADER_STAGE_FRAGMENT_BIT, mesh_shaders_indexpulling_spv, "main_fragment"},
		},
		std::vector<VkPushConstantRange>{{
			VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			0, sizeof(MeshPC)
		}}
	);

	sh_mesh_indexedDraw = new VKShader(
		vector<VKShader::Stage>{
			{VK_SHADER_STAGE_VERTEX_BIT,   mesh_shaders_indexedDraw_spv, "main_vertex"},
			{VK_SHADER_STAGE_FRAGMENT_BIT, mesh_shaders_indexedDraw_spv, "main_fragment"},
		},
		std::vector<VkPushConstantRange>{{
			VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			0, sizeof(MeshPC)
		}}
	);

	buffer_nodes = MemoryManager::allocVulkan(1024 * 1024, "buffer_nodes");
	buffer_drawCmds = MemoryManager::allocVulkan(1024 * 1024, "buffer_drawCmds");
	buffer_cumulativeTriCounts = MemoryManager::allocVulkan(1024 * 1024, "buffer_cumulativeTriCounts");

	VkSamplerCreateInfo sci = {};
	sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	sci.magFilter    = VK_FILTER_LINEAR;
	sci.minFilter    = VK_FILTER_LINEAR;
	sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sci.maxLod       = VK_LOD_CLAMP_NONE;
	vkCreateSampler(VKRenderer::device, &sci, nullptr, &sampler);

	sh_visbuffer_pass1 = new VKShader(
		vector<VKShader::Stage>{
			{VK_SHADER_STAGE_VERTEX_BIT,   mesh_visbuffer_spv, "main_vertex"},
			{VK_SHADER_STAGE_FRAGMENT_BIT, mesh_visbuffer_spv, "main_fragment"},
		},
		std::vector<VkPushConstantRange>{{
			VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			0, sizeof(MeshPC_VisbufPass1)
		}}
	);

	sh_visbuffer_pass1_compressed = new VKShader(
		vector<VKShader::Stage>{
			{VK_SHADER_STAGE_VERTEX_BIT,   mesh_visbuffer_spv, "main_vertex_compressed"},
			{VK_SHADER_STAGE_FRAGMENT_BIT, mesh_visbuffer_spv, "main_fragment"},
		},
		std::vector<VkPushConstantRange>{{
			VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			0, sizeof(MeshPC_VisbufPass1)
		}}
	);

	// Descriptor set layout: binding 0 = storage image (R32_UINT visbuffer)
	{
		VkDescriptorSetLayoutBinding b = {};
		b.binding         = 0;
		b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		b.descriptorCount = 1;
		b.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo dslCI = {};
		dslCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		dslCI.bindingCount = 1;
		dslCI.pBindings    = &b;
		vkCreateDescriptorSetLayout(VKRenderer::device, &dslCI, nullptr, &visbufDSL);

		VkDescriptorPoolSize poolSize = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, (uint32_t)VKRenderer::FRAMES_IN_FLIGHT };
		VkDescriptorPoolCreateInfo poolCI = {};
		poolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolCI.maxSets       = (uint32_t)VKRenderer::FRAMES_IN_FLIGHT;
		poolCI.poolSizeCount = 1;
		poolCI.pPoolSizes    = &poolSize;
		vkCreateDescriptorPool(VKRenderer::device, &poolCI, nullptr, &visbufPool);

		vector<VkDescriptorSetLayout> layouts(VKRenderer::FRAMES_IN_FLIGHT, visbufDSL);
		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool     = visbufPool;
		allocInfo.descriptorSetCount = (uint32_t)VKRenderer::FRAMES_IN_FLIGHT;
		allocInfo.pSetLayouts        = layouts.data();
		vkAllocateDescriptorSets(VKRenderer::device, &allocInfo, visbufSets.data());
	}

	sh_visbuffer_resolve = new VKShader(
		vector<VKShader::Stage>{
			{VK_SHADER_STAGE_VERTEX_BIT,   mesh_visbuffer_resolve_spv, "main_vertex"},
			{VK_SHADER_STAGE_FRAGMENT_BIT, mesh_visbuffer_resolve_spv, "main_fragment"},
		},
		std::vector<VkPushConstantRange>{{
			VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			0, sizeof(MeshPC_Resolve)
		}},
		{visbufDSL} // descriptor set 0 = visbuffer storage image
	);

	sh_visbuffer_resolve_compressed = new VKShader(
		vector<VKShader::Stage>{
			{VK_SHADER_STAGE_VERTEX_BIT,   mesh_visbuffer_resolve_spv, "main_vertex"},
			{VK_SHADER_STAGE_FRAGMENT_BIT, mesh_visbuffer_resolve_spv, "main_fragment_compressed"},
		},
		std::vector<VkPushConstantRange>{{
			VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			0, sizeof(MeshPC_Resolve)
		}},
		{visbufDSL} // descriptor set 0 = visbuffer storage image
	);

	initialized = true;
}

void updateDepthbuffer(
	uint32_t width, uint32_t height,
	vector<VkImage>& depthImages,
	vector<VkDeviceMemory>& depthMemories,
	vector<VkImageView>& depthViews
){
	static uint32_t depthWidth  = 0;
	static uint32_t depthHeight = 0;

	if(depthImages[0] == VK_NULL_HANDLE || depthWidth != width || depthHeight != height){
		vkDeviceWaitIdle(VKRenderer::device);
		depthWidth  = width;
		depthHeight = height;

		for(int i = 0; i < VKRenderer::FRAMES_IN_FLIGHT; i++){
			if(depthViews[i]   != VK_NULL_HANDLE) { vkDestroyImageView(VKRenderer::device, depthViews[i],   nullptr); depthViews[i]   = VK_NULL_HANDLE; }
			if(depthImages[i]  != VK_NULL_HANDLE) { vkDestroyImage    (VKRenderer::device, depthImages[i],  nullptr); depthImages[i]  = VK_NULL_HANDLE; }
			if(depthMemories[i]!= VK_NULL_HANDLE) { vkFreeMemory       (VKRenderer::device, depthMemories[i],nullptr); depthMemories[i]= VK_NULL_HANDLE; }

			VkImageCreateInfo ici = {};
			ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			ici.imageType     = VK_IMAGE_TYPE_2D;
			ici.format        = VK_FORMAT_D32_SFLOAT;
			ici.extent        = { width, height, 1 };
			ici.mipLevels     = 1;
			ici.arrayLayers   = 1;
			ici.samples       = VK_SAMPLE_COUNT_1_BIT;
			ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
			ici.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
			vkCreateImage(VKRenderer::device, &ici, nullptr, &depthImages[i]);

			VkMemoryRequirements memReqs;
			vkGetImageMemoryRequirements(VKRenderer::device, depthImages[i], &memReqs);
			VkMemoryAllocateInfo allocInfo = {};
			allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize  = memReqs.size;
			allocInfo.memoryTypeIndex = VKRenderer::findMemoryType(memReqs.memoryTypeBits,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			vkAllocateMemory(VKRenderer::device, &allocInfo, nullptr, &depthMemories[i]);
			vkBindImageMemory(VKRenderer::device, depthImages[i], depthMemories[i], 0);

			VkImageViewCreateInfo vci = {};
			vci.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			vci.image            = depthImages[i];
			vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
			vci.format           = VK_FORMAT_D32_SFLOAT;
			vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
			vkCreateImageView(VKRenderer::device, &vci, nullptr, &depthViews[i]);
		}
	}
}

static void updateVisbuffer(uint32_t width, uint32_t height)
{
	static uint32_t visbufWidth  = 0;
	static uint32_t visbufHeight = 0;

	if(visbufImages[0] == VK_NULL_HANDLE || visbufWidth != width || visbufHeight != height){
		vkDeviceWaitIdle(VKRenderer::device);
		visbufWidth  = width;
		visbufHeight = height;

		for(int i = 0; i < VKRenderer::FRAMES_IN_FLIGHT; i++){
			if(visbufViews[i]    != VK_NULL_HANDLE) { vkDestroyImageView(VKRenderer::device, visbufViews[i],    nullptr); visbufViews[i]    = VK_NULL_HANDLE; }
			if(visbufImages[i]   != VK_NULL_HANDLE) { vkDestroyImage    (VKRenderer::device, visbufImages[i],   nullptr); visbufImages[i]   = VK_NULL_HANDLE; }
			if(visbufMemories[i] != VK_NULL_HANDLE) { vkFreeMemory      (VKRenderer::device, visbufMemories[i], nullptr); visbufMemories[i] = VK_NULL_HANDLE; }

			VkImageCreateInfo ici = {};
			ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			ici.imageType     = VK_IMAGE_TYPE_2D;
			ici.format        = VK_FORMAT_R32_UINT;
			ici.extent        = { width, height, 1 };
			ici.mipLevels     = 1;
			ici.arrayLayers   = 1;
			ici.samples       = VK_SAMPLE_COUNT_1_BIT;
			ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
			ici.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
			vkCreateImage(VKRenderer::device, &ici, nullptr, &visbufImages[i]);

			VkMemoryRequirements memReqs;
			vkGetImageMemoryRequirements(VKRenderer::device, visbufImages[i], &memReqs);
			VkMemoryAllocateInfo allocInfo = {};
			allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize  = memReqs.size;
			allocInfo.memoryTypeIndex = VKRenderer::findMemoryType(memReqs.memoryTypeBits,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			vkAllocateMemory(VKRenderer::device, &allocInfo, nullptr, &visbufMemories[i]);
			vkBindImageMemory(VKRenderer::device, visbufImages[i], visbufMemories[i], 0);

			VkImageViewCreateInfo vci = {};
			vci.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			vci.image            = visbufImages[i];
			vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
			vci.format           = VK_FORMAT_R32_UINT;
			vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
			vkCreateImageView(VKRenderer::device, &vci, nullptr, &visbufViews[i]);

			// Update the resolve shader's descriptor set to point to this frame's visbuffer
			VkDescriptorImageInfo imgInfo = {};
			imgInfo.imageView   = visbufViews[i];
			imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

			VkWriteDescriptorSet write = {};
			write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			write.dstSet          = visbufSets[i];
			write.dstBinding      = 0;
			write.descriptorCount = 1;
			write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			write.pImageInfo      = &imgInfo;
			vkUpdateDescriptorSets(VKRenderer::device, 1, &write, 0, nullptr);
		}
	}
}

// - First draws into a visibility buffer, then uses a full-screen resolve pass.
// - Draws the entire scene in a single draw call
// - Uses programmable indexpulling and vertex pulling.
void drawVulkan_indexpulling_visibilitybuffer(Scene* scene, vector<View> views){
	if(views.empty()) return;

	init();

	// Wait for all in-flight frames to finish before overwriting the shared buffer_nodes / buffer_drawCmds that the previous frame's GPU commands may still be reading.
	// May hurt fps, but for perf evaluation we only care about duration of the draw.
	// And this way may actually be more accurate since the ongoing previous draw may not impact 
	// the draw we launch now.
	vkWaitForFences(
		VKRenderer::device,
		(uint32_t)VKRenderer::inFlightFences.size(),
		VKRenderer::inFlightFences.data(),
		VK_TRUE, UINT64_MAX);

	// Collect visible, loaded nodes
	vector<SNTriangles*> nodes;
	scene->forEach<SNTriangles>([&](SNTriangles* node) {
		if(!node->visible)           return;
		if(!node->mesh)              return;
		if(!node->mesh->isLoaded)    return;
		if(!node->texture)           return;
		nodes.push_back(node);
	});

	// Create Vulkan Resources from CUDA Resources
	for(SNTriangles* node : nodes){
		lazyInitVulkanFromCudaResources(node, sampler);
	}

	// Ultra Hack: We currently only load one glb file, where either all meshes are compressed or uncompressed.
	// So let's pick the shader based on whether the first node is compressed
	VKShader* sh_visbuffer;
	VKShader* sh_resolve;

	if(nodes[0]->mesh->compressed){
		sh_visbuffer = sh_visbuffer_pass1_compressed;
		sh_resolve = sh_visbuffer_resolve_compressed;
	}else{
		sh_visbuffer = sh_visbuffer_pass1;
		sh_resolve = sh_visbuffer_resolve;
	}


	uint32_t fboWidth  = (uint32_t)views[0].framebuffer->width;
	uint32_t fboHeight = (uint32_t)views[0].framebuffer->height;

	{ // Depth + Visibility buffers
		updateDepthbuffer(fboWidth, fboHeight, depthImages, depthMemories, depthViews);
		updateVisbuffer(fboWidth, fboHeight);

		VKRenderer::vulkanMeshDepthImage = depthImages[VKRenderer::currentFrame];
		VKRenderer::vulkanMeshDepthView  = depthViews[VKRenderer::currentFrame];
	}

	// Capture per-frame handles for the draw lambda
	VkImageView colorView   = views[0].framebuffer->colorAttachment->view;
	VkImageView visbufView  = visbufViews[VKRenderer::currentFrame];
	VkImage     visbufImage = visbufImages[VKRenderer::currentFrame];
	VkImageView depthView   = depthViews[VKRenderer::currentFrame];
	VkDescriptorSet descSet = visbufSets[VKRenderer::currentFrame];

	// Even with no geometry still run the two-pass structure so the outer
	// vkCmdEndRendering has a matching vkCmdBeginRendering.
	if(nodes.empty()){
		VKRenderer::vulkanMeshDrawFn = [=](VkCommandBuffer cmd){
			vkCmdEndRendering(cmd);

			// Transition visbuffer UNDEFINED → GENERAL and clear it.
			{
				VkImageMemoryBarrier2 b = {};
				b.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
				b.srcStageMask     = VK_PIPELINE_STAGE_2_NONE;
				b.srcAccessMask    = VK_ACCESS_2_NONE;
				b.dstStageMask     = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
				b.dstAccessMask    = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
				b.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
				b.newLayout        = VK_IMAGE_LAYOUT_GENERAL;
				b.image            = visbufImage;
				b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
				VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
				dep.imageMemoryBarrierCount = 1;
				dep.pImageMemoryBarriers    = &b;
				vkCmdPipelineBarrier2(cmd, &dep);
			}
			{
				VkClearColorValue clearVis = {}; clearVis.uint32[0] = 0xFFFFFFFFu;
				VkClearValue cvVis; cvVis.color = clearVis;
				VkRenderingAttachmentInfo ca = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
				ca.imageView = visbufView; ca.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
				ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
				ca.clearValue = cvVis;
				VkRenderingInfo ri = { VK_STRUCTURE_TYPE_RENDERING_INFO };
				ri.renderArea = { {0,0}, {fboWidth, fboHeight} }; ri.layerCount = 1;
				ri.colorAttachmentCount = 1; ri.pColorAttachments = &ca;
				vkCmdBeginRendering(cmd, &ri);
				vkCmdEndRendering(cmd);
			}
			{
				VkImageMemoryBarrier2 b = {};
				b.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
				b.srcStageMask     = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
				b.srcAccessMask    = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
				b.dstStageMask     = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
				b.dstAccessMask    = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
				b.oldLayout        = VK_IMAGE_LAYOUT_GENERAL;
				b.newLayout        = VK_IMAGE_LAYOUT_GENERAL;
				b.image            = visbufImage;
				b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
				VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
				dep.imageMemoryBarrierCount = 1;
				dep.pImageMemoryBarriers    = &b;
				vkCmdPipelineBarrier2(cmd, &dep);
			}
			// Begin pass 2 (LOAD — already cleared to black); outer code ends it.
			{
				VkRenderingAttachmentInfo ca = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
				ca.imageView = colorView; ca.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
				ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
				VkRenderingInfo ri = { VK_STRUCTURE_TYPE_RENDERING_INFO };
				ri.renderArea = { {0,0}, {fboWidth, fboHeight} }; ri.layerCount = 1;
				ri.colorAttachmentCount = 1; ri.pColorAttachments = &ca;
				vkCmdBeginRendering(cmd, &ri);
			}
		};
		return;
	}

	// ---- Build per-frame CPU arrays ----
	uint32_t nodeCount = (uint32_t)nodes.size();
	vector<VKNode> vkNodes(nodeCount);
	vector<VkDrawIndirectCommand> drawCmds(nodeCount);

	for(uint32_t i = 0; i < nodeCount; i++){
		SNTriangles* node              = nodes[i];
		Mesh* mesh                     = node->mesh;
		VKNode* vkNode                 = &vkNodes[i];
		VkDrawIndirectCommand* drawCmd = &drawCmds[i];

		uint32_t indexRange   = node->mesh->index_max - node->mesh->index_min;
		uint64_t bitsPerIndex = (uint64_t)ceil(log2f(float(indexRange + 1)));

		vkNode->world             = node->transform_global;
		vkNode->ptr_position      = mesh->vkc_position.deviceAddress;
		vkNode->ptr_uv            = mesh->vkc_uv.deviceAddress;
		vkNode->ptr_indices       = mesh->vkc_indices.deviceAddress;
		vkNode->textureHandle     = node->vkTextureHandle;
		vkNode->width             = node->texture->width;
		vkNode->height            = node->texture->height;
		vkNode->compressionFactor = (node->aabb.max - node->aabb.min) / 65536.0f;
		vkNode->min_x             = node->aabb.min.x;
		vkNode->min_y             = node->aabb.min.y;
		vkNode->min_z             = node->aabb.min.z;
		vkNode->bitsPerIndex      = (uint32_t)bitsPerIndex;

		drawCmd->vertexCount   = mesh->numTriangles * 3;
		drawCmd->instanceCount = 1;
		drawCmd->firstVertex   = 0;
		drawCmd->firstInstance = 0;
	}

	// Upload nodes and draw commands to host-visible buffers
	uint64_t requiredNodeBufferSize = uint64_t(nodeCount) * sizeof(VKNode);
	if(buffer_nodes->size < requiredNodeBufferSize) buffer_nodes->resize(requiredNodeBufferSize);
	memcpy(buffer_nodes->mapped, vkNodes.data(), requiredNodeBufferSize);

	uint64_t requiredDrawBufferSize = uint64_t(nodeCount) * sizeof(VkDrawIndirectCommand);
	if(buffer_drawCmds->size < requiredDrawBufferSize) buffer_drawCmds->resize(requiredDrawBufferSize);
	memcpy(buffer_drawCmds->mapped, drawCmds.data(), requiredDrawBufferSize);

	// Build exclusive prefix sum of triangle counts across meshes.
	// cumulativeTriCounts[i] = total triangles in meshes 0..i-1.
	vector<uint32_t> cumulativeTriCounts(nodeCount);
	uint32_t runningTotal = 0;
	for(uint32_t i = 0; i < nodeCount; i++){
		cumulativeTriCounts[i] = runningTotal;
		runningTotal += nodes[i]->mesh->numTriangles;
	}
	uint64_t requiredCumulativeBufferSize = uint64_t(nodeCount) * sizeof(uint32_t);
	if(buffer_cumulativeTriCounts->size < requiredCumulativeBufferSize)
		buffer_cumulativeTriCounts->resize(requiredCumulativeBufferSize);
	memcpy(buffer_cumulativeTriCounts->mapped, cumulativeTriCounts.data(), requiredCumulativeBufferSize);

	MeshPC_VisbufPass1 pc;
	pc.viewProj                  = glm::mat4(views[0].proj * views[0].view);
	pc.nodesAddr                 = buffer_nodes->deviceAddress;
	pc.cumulativeTriCountsAddr   = buffer_cumulativeTriCounts->deviceAddress;

	VKRenderer::vulkanMeshDrawFn = [=](VkCommandBuffer cmd){

		// End the outer rendering pass (applies the clear to color + depth).
		vkCmdEndRendering(cmd);

		// ── PASS 1: Geometry → Visibility Buffer ─────────────────────────────

		// Transition visbuffer UNDEFINED → GENERAL (color attachment write).
		{
			VkImageMemoryBarrier2 b = {};
			b.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
			b.srcStageMask     = VK_PIPELINE_STAGE_2_NONE;
			b.srcAccessMask    = VK_ACCESS_2_NONE;
			b.dstStageMask     = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
			b.dstAccessMask    = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
			b.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
			b.newLayout        = VK_IMAGE_LAYOUT_GENERAL;
			b.image            = visbufImage;
			b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
			VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
			dep.imageMemoryBarrierCount = 1;
			dep.pImageMemoryBarriers    = &b;
			vkCmdPipelineBarrier2(cmd, &dep);
		}

		// Begin pass 1: visbuffer (R32_UINT) + depth.
		{
			VkClearColorValue clearVis = {}; clearVis.uint32[0] = 0xFFFFFFFFu;
			VkClearValue cvVis;   cvVis.color          = clearVis;
			VkClearValue cvDepth; cvDepth.depthStencil = { 0.f, 0 };

			VkRenderingAttachmentInfo ca = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
			ca.imageView   = visbufView;
			ca.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			ca.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
			ca.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
			ca.clearValue  = cvVis;

			VkRenderingAttachmentInfo da = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
			da.imageView   = depthView;
			da.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			da.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
			da.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			da.clearValue  = cvDepth;

			VkRenderingInfo ri = { VK_STRUCTURE_TYPE_RENDERING_INFO };
			ri.renderArea           = { {0, 0}, {fboWidth, fboHeight} };
			ri.layerCount           = 1;
			ri.colorAttachmentCount = 1;
			ri.pColorAttachments    = &ca;
			ri.pDepthAttachment     = &da;
			vkCmdBeginRendering(cmd, &ri);
		}

		// Render state for pass 1
		{
			VkViewport vp = { 0.f, 0.f, (float)fboWidth, (float)fboHeight, 0.f, 1.f };
			vkCmdSetViewportWithCount(cmd, 1, &vp);
			VkRect2D sc = { {0, 0}, {fboWidth, fboHeight} };
			vkCmdSetScissorWithCount(cmd, 1, &sc);

			vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
			vkCmdSetPrimitiveRestartEnable(cmd, VK_FALSE);
			vkCmdSetRasterizerDiscardEnable(cmd, VK_FALSE);
			vkCmdSetPolygonModeEXT(cmd, VK_POLYGON_MODE_FILL);
			vkCmdSetCullMode(cmd, VK_CULL_MODE_FRONT_BIT);
			vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);
			vkCmdSetLineWidth(cmd, 1.f);
			vkCmdSetRasterizationSamplesEXT(cmd, VK_SAMPLE_COUNT_1_BIT);
			VkSampleMask sm = 0xFFFFFFFF;
			vkCmdSetSampleMaskEXT(cmd, VK_SAMPLE_COUNT_1_BIT, &sm);
			vkCmdSetAlphaToCoverageEnableEXT(cmd, VK_FALSE);
			vkCmdSetAlphaToOneEnableEXT(cmd, VK_FALSE);
			vkCmdSetDepthTestEnable(cmd, VK_TRUE);
			vkCmdSetDepthWriteEnable(cmd, VK_TRUE);
			vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_GREATER);
			vkCmdSetDepthBiasEnable(cmd, VK_FALSE);
			vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
			vkCmdSetStencilTestEnable(cmd, VK_FALSE);

			VkBool32 blendOff = VK_FALSE;
			vkCmdSetLogicOpEnableEXT(cmd, VK_FALSE);
			vkCmdSetColorBlendEnableEXT(cmd, 0, 1, &blendOff);
			VkColorBlendEquationEXT blendEq = {};
			blendEq.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
			blendEq.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
			blendEq.colorBlendOp        = VK_BLEND_OP_ADD;
			blendEq.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
			blendEq.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
			blendEq.alphaBlendOp        = VK_BLEND_OP_ADD;
			vkCmdSetColorBlendEquationEXT(cmd, 0, 1, &blendEq);
			VkColorComponentFlags colorMask = VK_COLOR_COMPONENT_R_BIT;
			vkCmdSetColorWriteMaskEXT(cmd, 0, 1, &colorMask);

			vkCmdBindShadersEXT(cmd,
				(uint32_t)sh_visbuffer->stages.size(),
				sh_visbuffer->stages.data(),
				sh_visbuffer->shaders.data());
			vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);
			vkCmdPushConstants(cmd, sh_visbuffer->layout,
				VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				0, sizeof(pc), &pc);
		}

		auto vkTsStart = Timer::recordVulkanTimestamp(cmd, VKRenderer::currentFrame);
		vkCmdDrawIndirect(cmd, buffer_drawCmds->buffer, 0, nodeCount, sizeof(VkDrawIndirectCommand));
		auto vkTsEnd = Timer::recordVulkanTimestamp(cmd, VKRenderer::currentFrame);
		Timer::recordVulkanDuration("visbuffer pass1", vkTsStart, vkTsEnd, VKRenderer::currentFrame);

		vkCmdEndRendering(cmd);

		// Barrier: visbuffer color-att write → storage image read.
		{
			VkImageMemoryBarrier2 b = {};
			b.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
			b.srcStageMask     = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
			b.srcAccessMask    = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
			b.dstStageMask     = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
			b.dstAccessMask    = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
			b.oldLayout        = VK_IMAGE_LAYOUT_GENERAL;
			b.newLayout        = VK_IMAGE_LAYOUT_GENERAL;
			b.image            = visbufImage;
			b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
			VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
			dep.imageMemoryBarrierCount = 1;
			dep.pImageMemoryBarriers    = &b;
			vkCmdPipelineBarrier2(cmd, &dep);
		}

		// ── PASS 2: Resolve — colorize each pixel by triangle index ──────────

		// Begin pass 2 on main color (LOAD — already cleared to black by outer pass).
		// The outer vkCmdEndRendering in recordCommandBuffer closes this pass.
		{
			VkRenderingAttachmentInfo ca = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
			ca.imageView   = colorView;
			ca.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			ca.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
			ca.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

			VkRenderingInfo ri = { VK_STRUCTURE_TYPE_RENDERING_INFO };
			ri.renderArea           = { {0, 0}, {fboWidth, fboHeight} };
			ri.layerCount           = 1;
			ri.colorAttachmentCount = 1;
			ri.pColorAttachments    = &ca;
			vkCmdBeginRendering(cmd, &ri);
		}

		// Render state for pass 2 (fullscreen triangle, depth disabled)
		{
			VkViewport vp = { 0.f, 0.f, (float)fboWidth, (float)fboHeight, 0.f, 1.f };
			vkCmdSetViewportWithCount(cmd, 1, &vp);
			VkRect2D sc = { {0, 0}, {fboWidth, fboHeight} };
			vkCmdSetScissorWithCount(cmd, 1, &sc);

			vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
			vkCmdSetPrimitiveRestartEnable(cmd, VK_FALSE);
			vkCmdSetRasterizerDiscardEnable(cmd, VK_FALSE);
			vkCmdSetPolygonModeEXT(cmd, VK_POLYGON_MODE_FILL);
			vkCmdSetCullMode(cmd, VK_CULL_MODE_NONE);
			vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);
			vkCmdSetLineWidth(cmd, 1.f);
			vkCmdSetRasterizationSamplesEXT(cmd, VK_SAMPLE_COUNT_1_BIT);
			VkSampleMask sm = 0xFFFFFFFF;
			vkCmdSetSampleMaskEXT(cmd, VK_SAMPLE_COUNT_1_BIT, &sm);
			vkCmdSetAlphaToCoverageEnableEXT(cmd, VK_FALSE);
			vkCmdSetAlphaToOneEnableEXT(cmd, VK_FALSE);
			vkCmdSetDepthTestEnable(cmd, VK_FALSE);
			vkCmdSetDepthWriteEnable(cmd, VK_FALSE);
			vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_ALWAYS);
			vkCmdSetDepthBiasEnable(cmd, VK_FALSE);
			vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
			vkCmdSetStencilTestEnable(cmd, VK_FALSE);

			VkBool32 blendOff = VK_FALSE;
			vkCmdSetLogicOpEnableEXT(cmd, VK_FALSE);
			vkCmdSetColorBlendEnableEXT(cmd, 0, 1, &blendOff);
			VkColorBlendEquationEXT blendEq = {};
			blendEq.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
			blendEq.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
			blendEq.colorBlendOp        = VK_BLEND_OP_ADD;
			blendEq.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
			blendEq.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
			blendEq.alphaBlendOp        = VK_BLEND_OP_ADD;
			vkCmdSetColorBlendEquationEXT(cmd, 0, 1, &blendEq);
			VkColorComponentFlags colorMask =
				VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
				VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
			vkCmdSetColorWriteMaskEXT(cmd, 0, 1, &colorMask);

			vkCmdBindShadersEXT(cmd,
				(uint32_t)sh_resolve->stages.size(),
				sh_resolve->stages.data(),
				sh_resolve->shaders.data());
			vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
				sh_resolve->layout, 0, 1, &descSet, 0, nullptr);

			MeshPC_Resolve resolvePC;
			resolvePC.viewProj                = pc.viewProj;
			resolvePC.nodesAddr               = pc.nodesAddr;
			resolvePC.screenWidth             = fboWidth;
			resolvePC.screenHeight            = fboHeight;
			resolvePC.cumulativeTriCountsAddr = pc.cumulativeTriCountsAddr;
			resolvePC.nodeCount               = nodeCount;
			resolvePC.pad0                    = 0;
			{ // World-space ray casting parameters for main_fragment.
				glm::mat4 viewI = glm::mat4(glm::inverse(views[0].view));
				// Camera position = last column of inverse-view matrix.
				resolvePC.cameraPos  = glm::vec4(viewI[3][0], viewI[3][1], viewI[3][2], 0.0f);
				// tan(halfFov) = 1 / proj focal length terms.
				float thx = 1.0f / float(views[0].proj[0][0]);
				float thy = 1.0f / float(views[0].proj[1][1]);
				resolvePC.tanHalfFov = glm::vec2(thx, thy);
				resolvePC.viewDirPad = glm::vec2(0.0f);
				// Rows of viewI (GLM is col-major: viewI[col][row]).
				resolvePC.viewInvRow0 = glm::vec4(viewI[0][0], viewI[1][0], viewI[2][0], viewI[3][0]);
				resolvePC.viewInvRow1 = glm::vec4(viewI[0][1], viewI[1][1], viewI[2][1], viewI[3][1]);
				resolvePC.viewInvRow2 = glm::vec4(viewI[0][2], viewI[1][2], viewI[2][2], viewI[3][2]);
			}
			vkCmdPushConstants(cmd, sh_resolve->layout,
				VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				0, sizeof(MeshPC_Resolve), &resolvePC);
		}

		{ // Fullscreen triangle; outer vkCmdEndRendering closes pass 2.
			auto vkTsStart = Timer::recordVulkanTimestamp(cmd, VKRenderer::currentFrame);
			vkCmdDraw(cmd, 3, 1, 0, 0);
			auto vkTsEnd = Timer::recordVulkanTimestamp(cmd, VKRenderer::currentFrame);
			Timer::recordVulkanDuration("visbuffer pass2 resolve", vkTsStart, vkTsEnd, VKRenderer::currentFrame);
		}
	};
}

// - One draw call per unique mesh, drawing all instances per draw.
// - Uses programmable index pulling and vertex pulling.
void drawVulkan_indexpulling_instanced_forward(Scene* scene, vector<SNTriangles*>& nodes, View view){

	init();

	// Wait for all in-flight frames to finish before overwriting the shared buffer_nodes / buffer_drawCmds that the previous frame's GPU commands may still be reading.
	// May hurt fps, but for perf evaluation we only care about duration of the draw.
	// And this way may actually be more accurate since the ongoing previous draw may not impact 
	// the draw we launch now.
	vkWaitForFences(
		VKRenderer::device,
		(uint32_t)VKRenderer::inFlightFences.size(),
		VKRenderer::inFlightFences.data(),
		VK_TRUE, UINT64_MAX);

	// Create Vulkan Resources from CUDA Resources
	for(SNTriangles* node : nodes){
		lazyInitVulkanFromCudaResources(node, sampler);
	}
	
	{ // Depth Buffer
		uint32_t fboW = (uint32_t)view.framebuffer->width;
		uint32_t fboH = (uint32_t)view.framebuffer->height;
		
		updateDepthbuffer(fboW, fboH, depthImages, depthMemories, depthViews);

		VKRenderer::vulkanMeshDepthImage = depthImages[VKRenderer::currentFrame];
		VKRenderer::vulkanMeshDepthView  = depthViews[VKRenderer::currentFrame];
	}

	// Even with no geometry, set a valid fn so recordCommandBuffer clears to black.
	// Depth image must be set first (above) so the barrier in recordCommandBuffer has a valid handle.
	if(nodes.empty()){
		VKRenderer::vulkanMeshDrawFn = [](VkCommandBuffer){};
		return;
	}

	// Key: Mesh* pointer — uniquely identifies the mesh object regardless of CUDA state.
	// (cptr_indices is 0 for non-CUDA-loaded meshes, making it an unreliable key.)
	vector<SNTriangles*> nodes_unique;

	Mesh* uniqueMesh = nullptr;
	unordered_map<Mesh*, int64_t> instanceCounts;
	for(int i = 0; i < nodes.size(); i++){

		SNTriangles* node = nodes[i];

		instanceCounts[node->mesh]++;
		
		if(uniqueMesh != node->mesh || CuRastSettings::disableInstancing){
			nodes_unique.push_back(node);
			uniqueMesh = node->mesh;
		}
	}

	// ---- Build per-frame CPU arrays (instanced) ----
	// One VKNode per instance (flat, ordered by unique mesh).
	// One DrawCall per unique mesh.
	// The shader indexes nodes[] via SV_InstanceID = firstInstance + localIndex.
	uint32_t instanceCount = (uint32_t)nodes.size();
	uint32_t uniqueCount   = (uint32_t)nodes_unique.size();

	struct DrawCall {
		uint32_t vertexCount;
		uint32_t instanceCount;
		uint32_t firstInstance;
	};

	vector<VKNode>    vkNodes(instanceCount);
	vector<DrawCall>  drawCalls(uniqueCount);

	for(uint32_t i = 0; i < instanceCount; i++){
		SNTriangles* node = nodes[i];
		Mesh* mesh        = node->mesh;
		VKNode& vkNode    = vkNodes[i];

		uint32_t indexRange   = mesh->index_max - mesh->index_min;
		uint32_t bitsPerIndex = (uint32_t)ceil(log2f(float(indexRange + 1)));

		vkNode.world             = node->transform_global;
		vkNode.worldViewProj     = view.proj * view.view * dmat4(node->transform_global);
		vkNode.ptr_position      = mesh->vkc_position.deviceAddress;
		vkNode.ptr_uv            = mesh->vkc_uv.deviceAddress;
		vkNode.ptr_indices       = mesh->vkc_indices.deviceAddress;
		vkNode.textureHandle     = node->vkTextureHandle;
		vkNode.width             = node->texture->width;
		vkNode.height            = node->texture->height;
		vkNode.compressionFactor = (node->aabb.max - node->aabb.min) / 65536.0f;
		vkNode.min_x             = node->aabb.min.x;
		vkNode.min_y             = node->aabb.min.y;
		vkNode.min_z             = node->aabb.min.z;
		vkNode.index_min         = node->mesh->index_min;
		vkNode.bitsPerIndex      = bitsPerIndex;
	}

	uint32_t firstInstance = 0;
	for(uint32_t i = 0; i < uniqueCount; i++){
		SNTriangles* node     = nodes_unique[i];
		uint32_t numInstances = (uint32_t)instanceCounts[node->mesh];

		drawCalls[i].vertexCount   = node->mesh->numTriangles * 3;
		drawCalls[i].instanceCount = numInstances;
		drawCalls[i].firstInstance = firstInstance;
		firstInstance += numInstances;
	}

	// Upload nodes to host-visible buffer
	uint64_t requiredNodeBufferSize = uint64_t(instanceCount) * sizeof(VKNode);
	if(buffer_nodes->size < requiredNodeBufferSize) buffer_nodes->resize(requiredNodeBufferSize);
	memcpy(buffer_nodes->mapped, vkNodes.data(), requiredNodeBufferSize);

	// Flush host writes to GPU-visible memory (required for non-coherent mappings)
	{
		VkMappedMemoryRange range = {};
		range.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
		range.memory = buffer_nodes->memory;
		range.offset = 0;
		range.size   = VK_WHOLE_SIZE;
		vkFlushMappedMemoryRanges(VKRenderer::device, 1, &range);
	}

	MeshPC pc{
		.viewProj  = glm::mat4(view.proj * view.view),
		.nodesAddr = buffer_nodes->deviceAddress
	};

	uint32_t fboWidth  = (uint32_t)view.framebuffer->width;
	uint32_t fboHeight = (uint32_t)view.framebuffer->height;

	VKRenderer::vulkanMeshDrawFn = [=](VkCommandBuffer cmd){

		// Viewport / scissor
		VkViewport vp = { 0.f, 0.f, (float)fboWidth, (float)fboHeight, 0.f, 1.f };
		vkCmdSetViewportWithCount(cmd, 1, &vp);
		VkRect2D sc = { {0, 0}, {fboWidth, fboHeight} };
		vkCmdSetScissorWithCount(cmd, 1, &sc);

		// Input assembly
		vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
		vkCmdSetPrimitiveRestartEnable(cmd, VK_FALSE);

		// Rasterization
		vkCmdSetRasterizerDiscardEnable(cmd, VK_FALSE);
		vkCmdSetPolygonModeEXT(cmd, VK_POLYGON_MODE_FILL);
		vkCmdSetCullMode(cmd, VK_CULL_MODE_FRONT_BIT);
		vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);
		vkCmdSetLineWidth(cmd, 1.f);

		// MSAA
		vkCmdSetRasterizationSamplesEXT(cmd, VK_SAMPLE_COUNT_1_BIT);
		VkSampleMask sm = 0xFFFFFFFF;
		vkCmdSetSampleMaskEXT(cmd, VK_SAMPLE_COUNT_1_BIT, &sm);
		vkCmdSetAlphaToCoverageEnableEXT(cmd, VK_FALSE);
		vkCmdSetAlphaToOneEnableEXT(cmd, VK_FALSE);

		// Depth / stencil
		vkCmdSetDepthTestEnable(cmd, VK_TRUE);
		vkCmdSetDepthWriteEnable(cmd, VK_TRUE);
		vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_GREATER);
		vkCmdSetDepthBiasEnable(cmd, VK_FALSE);
		vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
		vkCmdSetStencilTestEnable(cmd, VK_FALSE);

		// Color blending — opaque write
		VkBool32 blendEnable = VK_FALSE;
		vkCmdSetLogicOpEnableEXT(cmd, VK_FALSE);
		vkCmdSetColorBlendEnableEXT(cmd, 0, 1, &blendEnable);
		VkColorBlendEquationEXT blendEq = {};
		blendEq.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendEq.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
		blendEq.colorBlendOp        = VK_BLEND_OP_ADD;
		blendEq.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		blendEq.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		blendEq.alphaBlendOp        = VK_BLEND_OP_ADD;
		vkCmdSetColorBlendEquationEXT(cmd, 0, 1, &blendEq);
		VkColorComponentFlags colorMask =
			VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		vkCmdSetColorWriteMaskEXT(cmd, 0, 1, &colorMask);

		{// Setup Shader

			// Ultra Hack: We currently only load one glb file, where either all meshes are compressed or uncompressed.
			// So let's pick the shader based on whether the first node is compressed
			VKShader* shader;
			
			if(nodes[0]->mesh->compressed){
				shader = sh_mesh_instancepulling_compressed;
			}else{
				shader = sh_mesh_instancepulling;
			}

			vkCmdBindShadersEXT(cmd, 
				(uint32_t)shader->stages.size(),
				shader->stages.data(), 
				shader->shaders.data());
				
			// No vertex buffers — all geometry accessed via BDA in the shader
			vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

			// Push constants (viewProj + BDA to node array)
			vkCmdPushConstants(cmd, shader->layout,
				VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				0, sizeof(pc), &pc);
		}

		auto vkTsStart = Timer::recordVulkanTimestamp(cmd, VKRenderer::currentFrame);

		for(const DrawCall& dc : drawCalls){
			vkCmdDraw(cmd, dc.vertexCount, dc.instanceCount, 0, dc.firstInstance);
		}

		auto vkTsEnd = Timer::recordVulkanTimestamp(cmd, VKRenderer::currentFrame);
		Timer::recordVulkanDuration("vulkan draw", vkTsStart, vkTsEnd, VKRenderer::currentFrame);
	};
}

// 
// - One draw call per unique mesh, drawing all instances per draw.
// - Uses index draw call.
// - Uses vertex buffers.
// - Does not support compressed index buffers.
void drawVulkan_indexed_draw(Scene* scene, vector<SNTriangles*>& nodes, View view){

	init();

	// Wait for all in-flight frames to finish before overwriting the shared buffer_nodes / buffer_drawCmds that the previous frame's GPU commands may still be reading.
	// May hurt fps, but for perf evaluation we only care about duration of the draw.
	// And this way may actually be more accurate since the ongoing previous draw may not impact
	// the draw we launch now.
	vkWaitForFences(
		VKRenderer::device,
		(uint32_t)VKRenderer::inFlightFences.size(),
		VKRenderer::inFlightFences.data(),
		VK_TRUE, UINT64_MAX);

	// Create Vulkan Resources from CUDA Resources
	for(SNTriangles* node : nodes){
		lazyInitVulkanFromCudaResources(node, sampler);
	}

	{ // Depth Buffer
		uint32_t fboW = (uint32_t)view.framebuffer->width;
		uint32_t fboH = (uint32_t)view.framebuffer->height;

		updateDepthbuffer(fboW, fboH, depthImages, depthMemories, depthViews);

		VKRenderer::vulkanMeshDepthImage = depthImages[VKRenderer::currentFrame];
		VKRenderer::vulkanMeshDepthView  = depthViews[VKRenderer::currentFrame];
	}

	// Indexed draw does not support compressed indexbuffers.
	// Also, if no nodes, let's quit early.
	if(nodes.size() == 0 || nodes[0]->mesh->compressed){
		VKRenderer::vulkanMeshDrawFn = [](VkCommandBuffer){};
		return;
	}

	// Key: Mesh* pointer — uniquely identifies the mesh object regardless of CUDA state.
	// (cptr_indices is 0 for non-CUDA-loaded meshes, making it an unreliable key.)
	vector<SNTriangles*> nodes_unique;

	Mesh* uniqueMesh = nullptr;
	unordered_map<Mesh*, int64_t> instanceCounts;
	for(int i = 0; i < nodes.size(); i++){

		SNTriangles* node = nodes[i];

		instanceCounts[node->mesh]++;
		
		if(uniqueMesh != node->mesh || CuRastSettings::disableInstancing){
			nodes_unique.push_back(node);
			uniqueMesh = node->mesh;
		}
	}

	uint32_t instanceCount = (uint32_t)nodes.size();
	uint32_t uniqueCount   = (uint32_t)nodes_unique.size();

	// Per-instance VKNode data (world, ptrs, textureHandle — the shader Node struct maps to this layout).
	vector<VKNode> vkNodes(instanceCount);
	for(uint32_t i = 0; i < instanceCount; i++){
		SNTriangles* node  = nodes[i];
		VKNode&      vkNode = vkNodes[i];

		vkNode.world         = node->transform_global;
		vkNode.worldViewProj = view.proj * view.view * dmat4(node->transform_global);
		vkNode.ptr_position  = 0;
		vkNode.ptr_uv        = 0;
		vkNode.ptr_indices   = 0;
		vkNode.textureHandle = node->vkTextureHandle;
	}

	// Per-unique-mesh draw calls.
	struct DrawCall {
		VkBuffer  indexBuffer;
		uint64_t  indexBufferOffset;
		VkBuffer  posBuffer;
		uint64_t  posBufferOffset;
		VkBuffer  uvBuffer;
		uint64_t  uvBufferOffset;
		uint32_t  indexCount;
		uint32_t  instanceCount;
		uint32_t  firstInstance;
	};
	vector<DrawCall> drawCalls(uniqueCount);
	{
		uint32_t firstInstance = 0;
		for(uint32_t i = 0; i < uniqueCount; i++){
			SNTriangles* node      = nodes_unique[i];
			Mesh*        mesh      = node->mesh;
			uint32_t     numInst   = (uint32_t)instanceCounts[node->mesh];
			
			drawCalls[i].indexBuffer       = mesh->vkc_indices.vk_buffer;
			drawCalls[i].indexBufferOffset = mesh->vkc_indices.offset;
			drawCalls[i].posBuffer         = mesh->vkc_position.vk_buffer;
			drawCalls[i].posBufferOffset   = mesh->vkc_position.offset;
			drawCalls[i].uvBuffer          = mesh->vkc_uv.vk_buffer;
			drawCalls[i].uvBufferOffset    = mesh->vkc_uv.offset;
			drawCalls[i].indexCount        = mesh->numTriangles * 3;
			drawCalls[i].instanceCount     = numInst;
			drawCalls[i].firstInstance     = firstInstance;
			firstInstance += numInst;
		}
	}

	// Upload node data to host-visible buffer.
	uint64_t requiredNodeBufferSize = uint64_t(instanceCount) * sizeof(VKNode);
	if(buffer_nodes->size < requiredNodeBufferSize) buffer_nodes->resize(requiredNodeBufferSize);
	memcpy(buffer_nodes->mapped, vkNodes.data(), requiredNodeBufferSize);

	{
		VkMappedMemoryRange range = {};
		range.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
		range.memory = buffer_nodes->memory;
		range.offset = 0;
		range.size   = VK_WHOLE_SIZE;
		vkFlushMappedMemoryRanges(VKRenderer::device, 1, &range);
	}

	MeshPC pc;
	pc.viewProj  = glm::mat4(view.proj * view.view);
	pc.nodesAddr = buffer_nodes->deviceAddress;

	uint32_t fboWidth  = (uint32_t)view.framebuffer->width;
	uint32_t fboHeight = (uint32_t)view.framebuffer->height;

	VKRenderer::vulkanMeshDrawFn = [=, drawCalls = std::move(drawCalls)](VkCommandBuffer cmd) mutable {
		vkCmdBindShadersEXT(cmd, (uint32_t)sh_mesh_indexedDraw->stages.size(),
			sh_mesh_indexedDraw->stages.data(), sh_mesh_indexedDraw->shaders.data());

		// binding 0 = position (vec3), binding 1 = uv (vec2)
		VkVertexInputBindingDescription2EXT bindings[2] = {};
		bindings[0].sType     = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT;
		bindings[0].binding   = 0;
		bindings[0].stride    = sizeof(float) * 3;
		bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		bindings[0].divisor   = 1;
		bindings[1].sType     = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT;
		bindings[1].binding   = 1;
		bindings[1].stride    = sizeof(float) * 2;
		bindings[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		bindings[1].divisor   = 1;

		VkVertexInputAttributeDescription2EXT attribs[2] = {};
		attribs[0].sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT;
		attribs[0].location = 0;
		attribs[0].binding  = 0;
		attribs[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
		attribs[0].offset   = 0;
		attribs[1].sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT;
		attribs[1].location = 1;
		attribs[1].binding  = 1;
		attribs[1].format   = VK_FORMAT_R32G32_SFLOAT;
		attribs[1].offset   = 0;

		vkCmdSetVertexInputEXT(cmd, 2, bindings, 2, attribs);

		// Viewport / scissor
		VkViewport vp = { 0.f, 0.f, (float)fboWidth, (float)fboHeight, 0.f, 1.f };
		vkCmdSetViewportWithCount(cmd, 1, &vp);
		VkRect2D sc = { {0, 0}, {fboWidth, fboHeight} };
		vkCmdSetScissorWithCount(cmd, 1, &sc);

		// Input assembly
		vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
		vkCmdSetPrimitiveRestartEnable(cmd, VK_FALSE);

		// Rasterization
		vkCmdSetRasterizerDiscardEnable(cmd, VK_FALSE);
		vkCmdSetPolygonModeEXT(cmd, VK_POLYGON_MODE_FILL);
		vkCmdSetCullMode(cmd, VK_CULL_MODE_FRONT_BIT);
		vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);
		vkCmdSetLineWidth(cmd, 1.f);

		// MSAA
		vkCmdSetRasterizationSamplesEXT(cmd, VK_SAMPLE_COUNT_1_BIT);
		VkSampleMask sm = 0xFFFFFFFF;
		vkCmdSetSampleMaskEXT(cmd, VK_SAMPLE_COUNT_1_BIT, &sm);
		vkCmdSetAlphaToCoverageEnableEXT(cmd, VK_FALSE);
		vkCmdSetAlphaToOneEnableEXT(cmd, VK_FALSE);

		// Depth / stencil
		vkCmdSetDepthTestEnable(cmd, VK_TRUE);
		vkCmdSetDepthWriteEnable(cmd, VK_TRUE);
		vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_GREATER);
		vkCmdSetDepthBiasEnable(cmd, VK_FALSE);
		vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
		vkCmdSetStencilTestEnable(cmd, VK_FALSE);

		// Color blending — opaque write
		VkBool32 blendEnable = VK_FALSE;
		vkCmdSetLogicOpEnableEXT(cmd, VK_FALSE);
		vkCmdSetColorBlendEnableEXT(cmd, 0, 1, &blendEnable);
		VkColorBlendEquationEXT blendEq = {};
		blendEq.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendEq.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
		blendEq.colorBlendOp        = VK_BLEND_OP_ADD;
		blendEq.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		blendEq.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		blendEq.alphaBlendOp        = VK_BLEND_OP_ADD;
		vkCmdSetColorBlendEquationEXT(cmd, 0, 1, &blendEq);
		VkColorComponentFlags colorMask =
			VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		vkCmdSetColorWriteMaskEXT(cmd, 0, 1, &colorMask);

		vkCmdPushConstants(cmd, sh_mesh_indexedDraw->layout,
			VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			0, sizeof(pc), &pc);

		auto vkTsStart = Timer::recordVulkanTimestamp(cmd, VKRenderer::currentFrame);

		// One draw call per unique mesh; bind position and UV vertex buffers per mesh.
		for(const DrawCall& dc : drawCalls){
			VkBuffer      vbufs[2]    = { dc.posBuffer,       dc.uvBuffer       };
			VkDeviceSize  voffsets[2] = { dc.posBufferOffset, dc.uvBufferOffset };
			vkCmdBindVertexBuffers(cmd, 0, 2, vbufs, voffsets);
			vkCmdBindIndexBuffer(cmd, dc.indexBuffer, dc.indexBufferOffset, VK_INDEX_TYPE_UINT32);
			vkCmdDrawIndexed(cmd, dc.indexCount, dc.instanceCount, 0, 0, dc.firstInstance);
		}

		auto vkTsEnd = Timer::recordVulkanTimestamp(cmd, VKRenderer::currentFrame);
		Timer::recordVulkanDuration("vulkan drawIndexed", vkTsStart, vkTsEnd, VKRenderer::currentFrame);
	};

}
