
#include <thread>
#include <queue>
#include <stacktrace>
#include <set>

#include "jpg/turbojpeg.h"

#include "GLTFLoader.h"
#include "scene/SceneNode.h"
#include "scene/SNTriangles.h"
#include "MemoryManager.h"
#include "CudaModularProgram.h"
#include "ThreadPool.h"
#include "CudaModularProgram.h"
#include "Timer.h"
#include "BitEdit.h"
#include "kernels/textureTools.cuh"
#include "stb/stb_image_resize2.h"
#include "CuRastSettings.h"
#include "TextureManager.h"


using namespace std; // YOLO

namespace largeGlb{

	constexpr float JPEG_QUALITY = 80;

	bool resize_jpeg_buffer_turbo(const std::vector<uint8_t>& jpeg_input,
		int new_width,
		int new_height,
		std::vector<uint8_t>& jpeg_output,
		int quality = JPEG_QUALITY
	){
		tjhandle tjInstance = tjInitDecompress();
		if (!tjInstance) {
			std::cerr << "TurboJPEG init decompress error: " << tjGetErrorStr() << std::endl;
			return false;
		}

		int width, height, jpegSubsamp, jpegColorspace;
		if (tjDecompressHeader3(tjInstance, jpeg_input.data(), jpeg_input.size(),
			&width, &height, &jpegSubsamp, &jpegColorspace) != 0)
		{
			std::cerr << "TurboJPEG decompress header error: " << tjGetErrorStr() << std::endl;
			tjDestroy(tjInstance);
			return false;
		}

		std::vector<uint8_t> decodedRGB(width * height * 3);
		if (tjDecompress2(tjInstance, jpeg_input.data(), jpeg_input.size(),
			decodedRGB.data(), width, 0, height, TJPF_RGB, TJFLAG_FASTDCT) != 0)
		{
			std::cerr << "TurboJPEG decompress error: " << tjGetErrorStr() << std::endl;
			tjDestroy(tjInstance);
			return false;
		}

		tjDestroy(tjInstance);

		std::vector<uint8_t> resizedRGB(new_width * new_height * 3);

			stbir_resize_uint8_linear(
			decodedRGB.data(), width, height, 0,
			resizedRGB.data(), new_width, new_height, 0,
			STBIR_RGB
		);


		tjhandle tjComp = tjInitCompress();
		if (!tjComp) {
			std::cerr << "TurboJPEG init compress error: " << tjGetErrorStr() << std::endl;
			return false;
		}

		unsigned char* jpegBuf = nullptr;
		unsigned long jpegSize = 0;

		if (tjCompress2(tjComp, resizedRGB.data(), new_width, 0, new_height, TJPF_RGB,
			&jpegBuf, &jpegSize, TJSAMP_420, quality, TJFLAG_FASTDCT) != 0)
		{
			std::cerr << "TurboJPEG compress error: " << tjGetErrorStr() << std::endl;
			tjDestroy(tjComp);
			return false;
		}
		jpeg_output.assign(jpegBuf, jpegBuf + jpegSize);
		tjFree(jpegBuf);
		tjDestroy(tjComp);
		return true;
	}

	Texture* createMipMappedTexture(void* data, int64_t width, int64_t height){
		int64_t levels = log2(max(width, height));
		int64_t bytesPerPixel = 4;

		int64_t mipWidth = width;
		int64_t mipHeight = height;

		int64_t byteSize = 0;
		for(int i = 0; i < levels; i++){

			byteSize += mipWidth * mipHeight * 4;

			mipWidth = (mipWidth + 2 - 1) / 2;
			mipHeight = (mipHeight + 2 - 1) / 2;
		}

		CUdeviceptr cptr_texture = MemoryManager::alloc(byteSize, "texture");
		cuMemcpyHtoD(cptr_texture, data, width * height * 4);

		uint32_t blockSize = 16;
		uint32_t gridWidth = (width / 2 + blockSize) / blockSize;
		uint32_t gridHeight = (height / 2 + blockSize) / blockSize;

		computeMipMap((uint32_t*)cptr_texture, width, height);

		Texture* texture = TextureManager::create();
		texture->width = width;
		texture->height = height;
		texture->data = (uint32_t*)cptr_texture;

		return texture;
	}

	

	// Here we apply a bit of a hack:
	// Jpeg Texture Mip Levels are stored as separate textures 
	// because we can't compute the byte offset to another level from width and height.
	// Therefore we create 8 mip levels and put them into the texture manager,
	// But we only return the pointer to the original resolution.
	Texture* createMipMappedJpegTexture(void* data, int64_t byteLength, int64_t textureWidth, int64_t textureHeight){

		vector<uint8_t> u8data((uint8_t*)data, (uint8_t*)data + byteLength);
		JPEGIndexer* jpegIndexer = new JPEGIndexer(u8data);

		int numLevels = 8;
		Texture* levels[8];

		static mutex mtx;
		mtx.lock(); // Lock because we need these 8 mip levels to be consecutive in the texture manager.
		for (int i = 0; i < numLevels; i++) {
			levels[i] = TextureManager::create();
		}
		mtx.unlock();

		for (int i = 0; i < numLevels; i++) {

			JPEGIndexer* indexer = nullptr;
			int resizedWidth = max(textureWidth / pow(2, i), 4.0);
			int resizedHeight = max(textureHeight / pow(2, i), 4.0);

			if (i == 0) {
				indexer = new JPEGIndexer(u8data);
			} else {
				vector<uint8_t> resized;
				resize_jpeg_buffer_turbo(u8data, resizedWidth, resizedHeight, resized, JPEG_QUALITY);

				indexer = new JPEGIndexer(resized);
			}
			
			indexer->mipMapLevel = i;

			Texture* texture = levels[i];
			texture->width = resizedWidth;
			texture->height = resizedHeight;

			texture->data = (uint32_t*)MemoryManager::alloc(indexer->only_ac_data.size() + 384, "jpeg data");
			cuMemcpyHtoD((CUdeviceptr)texture->data, indexer->only_ac_data.data(), indexer->only_ac_data.size() * sizeof(uint8_t));

			texture->mcuPositions = (uint32_t*)MemoryManager::alloc(indexer->mcu_index.size() * sizeof(uint32_t), "mcuPositions");
			cuMemcpyHtoD((CUdeviceptr)texture->mcuPositions, indexer->mcu_index.data(), indexer->mcu_index.size() * sizeof(uint32_t));

			vector<HuffmanTable> huffman_table_vector;
			for (const auto& class_entry : indexer->huffman_tables_components) {
				for (const auto& table_entry : class_entry.second) {
					HuffmanTable huff_table = {};
					int temp_keys[256] = {};
					int value_index = 0;
					for (const auto& code_value : table_entry.second) {
						const string& code = code_value.first;
						int value = code_value.second;

						int code_len = code.size();
						huff_table.num_codes_per_bit_length[code_len - 1] += 1;
						temp_keys[value_index] = std::stoi(code, nullptr, 2);
						huff_table.huffman_values[value_index] = value;
						value_index++;
					}
					// println("Number of Huffman Codes: {}", table_entry.second.size());

					// Build packed[i] = (codelength << 16) | key, sorted by length (same order as
					// map iteration for canonical JPEG Huffman codes).
					int codeIndex = 0;
					for (int i = 0; i < 16; i++) {
						int codeLength = i + 1;
						int numCodes = huff_table.num_codes_per_bit_length[i];
						for (int j = 0; j < numCodes; j++) {
							huff_table.packed[codeIndex] = (uint32_t(codeLength) << 16) | uint32_t(temp_keys[codeIndex]);
							codeIndex++;
						}
					}

					huffman_table_vector.push_back(huff_table);
				}
			}

			texture->huffmanTables = (HuffmanTable*)MemoryManager::alloc(huffman_table_vector.size() * sizeof(HuffmanTable), "huffmanTables");
			cuMemcpyHtoD((CUdeviceptr)texture->huffmanTables, huffman_table_vector.data(), huffman_table_vector.size() * sizeof(HuffmanTable));

			vector<QuantizationTable> quant_table_vector;
			for (const auto& quant_entry : indexer->quantization_tables) {
				QuantizationTable quant_table = {};
				std::copy(quant_entry.second.begin(), quant_entry.second.end(), quant_table.values);
				quant_table_vector.push_back(quant_table);
			}
			texture->quanttables = (QuantizationTable*)MemoryManager::alloc(quant_table_vector.size() * sizeof(QuantizationTable), "quanttables");
			cuMemcpyHtoD((CUdeviceptr)texture->quanttables, quant_table_vector.data(), quant_table_vector.size() * sizeof(QuantizationTable));
			
		}

		return levels[0];
	}

	struct LoadConfig{
		bool skipUVs = false;
		bool skipNormals = false;
		bool skipVertexColors = false;
		bool compress = false;
		bool useJpegTextures = false;
		int imageDivisionFactor = 1;
	};

	struct LoadedGlb{
		string path;
		shared_ptr<SceneNode> glbNode = nullptr;
		Texture* defaultTexture;
		vector<Texture*> textures;
		vector<Mesh> meshes;

		#if defined(USE_VULKAN_SHARED_MEMORY)
			VulkanCudaSharedMemory* memory = nullptr;
		#else 
			CudaVirtualMemory* memory = nullptr;
		#endif
	};

	struct LargeGlbLoader{
		string path;
		CUcontext context;
		gltfloader::GLTF gltf;
		shared_ptr<UnbufferedFile> file = nullptr;
		shared_ptr<Mapping::MappedFile> mappedFile = nullptr;
		LoadConfig config;

		struct DeviceBuffer{
			CUdeviceptr cptr = 0;
			uint64_t size = 0;
			uint32_t index_min = 0;
			uint32_t index_max = 0;
		};

		vector<DeviceBuffer> accessorToDevicebufferMapping;
		

		shared_ptr<LoadedGlb> run(){

			shared_ptr<LoadedGlb> loaded = make_shared<LoadedGlb>();
			loaded->path = path;

			{ // Create Default Texture
				int64_t textureWidth = 128;
				int64_t textureHeight = 128;
				vector<uint8_t> textureData = vector<uint8_t>(2 * 4 * textureWidth * textureHeight, 255);
				
				loaded->defaultTexture = TextureManager::create();
				loaded->defaultTexture->width = textureWidth;
				loaded->defaultTexture->height = textureHeight;
				loaded->defaultTexture->data = (uint32_t*)MemoryManager::alloc(byteSizeOf(textureData), "default texture");

				cuMemcpyHtoDAsync(CUdeviceptr(loaded->defaultTexture->data), textureData.data(), byteSizeOf(textureData), 0);
			}
			
			double t_start = now();

			gltf = gltfloader::loadMetadata(path);
			// gltf = gltfloader::filter(gltf, 20'000); 

			file = UnbufferedFile::open(gltf.buffers[0].uri);
			mappedFile = Mapping::mapFile(gltf.buffers[0].uri);

			uint64_t totalBytes_uncompressed = 0;
			uint64_t totalBytes_compressed = 0;
			uint64_t maxByteSize = 0;

			for(int accessorIndex = 0; accessorIndex < gltf.accessors.size(); accessorIndex++){
				gltfloader::Accessor accessor = gltf.accessors[accessorIndex];
				uint64_t byteSize = accessor.getByteSize();
				maxByteSize = max(maxByteSize, byteSize);
			}

			accessorToDevicebufferMapping.resize(gltf.accessors.size());
			loaded->textures.resize(gltf.images.size(), loaded->defaultTexture);

			//==================================================================
			// ASSEMBLE LIST OF LOADER TASKS
			//==================================================================
			struct LoadTask{
				int accessorIndex;
				uint64_t byteOffset;
				uint64_t byteSize;
			};

			struct ImageLoadTask{
				int imageIndex;
				uint64_t byteOffset;
				uint64_t byteSize;
			};

			vector<ImageLoadTask> imageLoadTasks;

			set<int> accessors_indices;
			set<int> accessors_positions;
			set<int> accessors_normals;
			set<int> accessors_colors;
			set<int> accessors_uvs;

			for(gltfloader::Mesh& mesh : gltf.meshes){
				for(gltfloader::Primitive& primitive : mesh.primitives){
					
					if (primitive.indices != -1) {
						accessors_indices.insert(primitive.indices);
					}

					if(primitive.attributes.contains("POSITION")){
						accessors_positions.insert(primitive.attributes["POSITION"]);
					}
					if(primitive.attributes.contains("NORMAL")){
						accessors_normals.insert(primitive.attributes["NORMAL"]);
					}
					if(primitive.attributes.contains("COLOR_0")){
						accessors_colors.insert(primitive.attributes["COLOR_0"]);
					}
					if(primitive.attributes.contains("TEXCOORD_0")){
						accessors_uvs.insert(primitive.attributes["TEXCOORD_0"]);
					}
				}
			}

			// IMAGE LOADER TASKS
			for(int imageIndex = 0; imageIndex < gltf.images.size(); imageIndex++){
				gltfloader::Image image = gltf.images[imageIndex];
				gltfloader::BufferView bufferView = gltf.bufferViews[image.bufferView];
				gltfloader::Buffer& buffer = gltf.buffers[bufferView.buffer];

				uint64_t byteOffset = buffer.offset + bufferView.byteOffset;
				uint64_t byteSize = bufferView.byteLength;

				ImageLoadTask task;
				task.imageIndex = imageIndex;
				task.byteOffset = byteOffset;
				task.byteSize = byteSize;

				imageLoadTasks.push_back(task);
			}

			// SIZE OF LARGEST INDEX BUFFER
			uint64_t largestIndexbufferSize = 0;
			uint64_t uncompressedIndicesSize = 0;
			for(int accessorIndex : accessors_indices){
				uint64_t indexbufferu32Size = gltf.accessors[accessorIndex].count * sizeof(uint32_t);
				largestIndexbufferSize = max(largestIndexbufferSize, indexbufferu32Size);
				
				uncompressedIndicesSize += indexbufferu32Size;
			}
			println("largestIndexbufferSize: {:L}", largestIndexbufferSize);

			uint64_t indices_uncompressed = 0;
			uint64_t indices_compressed = 0;

			#if defined(USE_VULKAN_SHARED_MEMORY)
				loaded->memory = MemoryManager::allocVulkanCudaShared(40'000'000'000, "glb memory");
			#else
				loaded->memory = MemoryManager::allocVirtualCuda(40'000'000'000, "glb memory");
			#endif
			
			loaded->memory->commit(uncompressedIndicesSize);
			atomic_uint64_t gpu_memory_counter = 0;
			// "prime" the buffer to avoid measuring cold-start memcpy perf latter on. 
			// cuMemsetD8(memory->cptr, 0, memory->comitted);
			// void* tmpHostMem;
			// cuMemAllocHost(&tmpHostMem, 10'000'000);
			// cuMemcpyHtoD(memory->cptr, tmpHostMem, min(uncompressedIndicesSize, 10'000'000llu));
			// cuMemFreeHost(tmpHostMem);

			// Hack: Let's allocate enough pinned host memory to store the largest index buffer.
			// - Assume indices are 32 bit, because we may convert to 32 bit in uncompressed mode.
			// - Add sectorSize padding so that we can use unbuffered reads that may require padding.
			struct PinnedBuffer{
				void* buffer;
				uint64_t size;
			};
			ThreadPool pool(16);
			vector<PinnedBuffer> pinnedBuffers;
			for(int i = 0; i < pool.numThreads; i++){
				PinnedBuffer pinnedBuffer;
				pinnedBuffer.size = roundUp(
					max(int64_t(largestIndexbufferSize), 100'000'000ll), 
					file->sectorSize) + file->sectorSize;

				auto result = cuMemAllocHost(&pinnedBuffer.buffer, pinnedBuffer.size);
				CURuntime::assertCudaSuccess(result);

				memset(pinnedBuffer.buffer, 0, pinnedBuffer.size);
				pinnedBuffers.push_back(pinnedBuffer);
			}

			double t_start_index_tasks = now();

			atomic_uint64_t bytesRead = 0;

			//==================================================================
			// PROCESS INDEX BUFFERS
			//==================================================================
			atomic_uint64_t size_indexbuffer_gltf = 0;
			atomic_uint64_t size_indexbuffer_curast = 0;
			println("process index buffers");
			for(int accessorIndex : accessors_indices){

				gltfloader::Accessor accessor = gltf.accessors[accessorIndex];
				gltfloader::BufferView bufferView = gltf.bufferViews[accessor.bufferView];
				gltfloader::Buffer& buffer = gltf.buffers[bufferView.buffer];

				LoadTask task;
				task.accessorIndex = accessorIndex;
				task.byteOffset = buffer.offset + bufferView.byteOffset + accessor.byteOffset;
				task.byteSize = accessor.getByteSize();

				size_indexbuffer_gltf += task.byteSize;
				
				pool.enqueue([&, this, task](int threadIndex){
					cuCtxSetCurrent(context);

					gltfloader::Accessor accessor = gltf.accessors[task.accessorIndex];
					gltfloader::BufferView view = gltf.bufferViews[accessor.bufferView];
					gltfloader::Buffer buffer = gltf.buffers[view.buffer];
					uint64_t numIndices = accessor.count;
					PinnedBuffer pinned = pinnedBuffers[threadIndex];

					uint64_t padding = task.byteOffset % file->sectorSize;
					uint64_t paddedOffset = task.byteOffset - padding;
					uint64_t paddedSize = roundUp(int64_t(task.byteSize + padding), file->sectorSize);
					file->read(paddedOffset, paddedSize, pinned.buffer);

					bytesRead += paddedSize;

					auto readValue = [&](int i) -> uint32_t {
						if(accessor.componentType == gltfloader::COMPONENT_TYPE_UINT8){
							return read<uint8_t>(pinned.buffer, padding + 1 * i);
						}else if(accessor.componentType == gltfloader::COMPONENT_TYPE_UINT16){
							return read<uint16_t>(pinned.buffer, padding + 2 * i);
						}else if(accessor.componentType == gltfloader::COMPONENT_TYPE_UINT32){
							return read<uint32_t>(pinned.buffer, padding + 4 * i);
						}else{
							println("unhandled component type: {}", accessor.componentType);
							__debugbreak();
							exit(2354356);
						}
					};

					uint64_t gpu_memory_offset = 0;
					uint64_t indexbufferByteSize = 0;
					uint32_t index_min = 0;
					uint32_t index_max = 0;

					if(config.compress){
						index_min = 0xffffffff;
						index_max = 0;
						
						for(int i = 0; i < numIndices; i++){
							uint32_t value = readValue(i);

							index_min = min(index_min, value);
							index_max = max(index_max, value);
						}

						uint64_t indexRange = index_max - index_min;
						uint64_t requiredBits = ceil(log2f(float(indexRange + 1)));

						for(int i = 0; i < numIndices; i++){
							uint32_t value = readValue(i) - index_min;

							BitEdit::writeU32((uint32_t*)pinned.buffer, i * requiredBits, requiredBits, value);
						}

						indexbufferByteSize = roundUp((numIndices * requiredBits + 7llu) / 8llu, 4llu);
						gpu_memory_offset = gpu_memory_counter.fetch_add(indexbufferByteSize);
						loaded->memory->memcopyHtoD(gpu_memory_offset, pinned.buffer, indexbufferByteSize);
					}else if(accessor.componentType == gltfloader::COMPONENT_TYPE_UINT32){
						// nothing to do
						indexbufferByteSize = numIndices * sizeof(uint32_t);
						gpu_memory_offset = gpu_memory_counter.fetch_add(indexbufferByteSize);
						loaded->memory->memcopyHtoD(gpu_memory_offset, (uint8_t*)pinned.buffer + padding, indexbufferByteSize);
					}else if(accessor.componentType == gltfloader::COMPONENT_TYPE_UINT16){
						// - Convert to 32 bit indices.
						// - in-place: Process back to front so that read<u16> and write<u32> in same buffer won't overlap.

						for(int i = numIndices - 1; i >= 0; i--){
							uint32_t value = read<uint16_t>((uint8_t*)pinned.buffer + padding, i * sizeof(uint16_t));
							write<uint32_t>((uint8_t*)pinned.buffer + padding, i * sizeof(uint32_t), value);
						}

						indexbufferByteSize = numIndices * sizeof(uint32_t);
						gpu_memory_offset = gpu_memory_counter.fetch_add(indexbufferByteSize);
						loaded->memory->memcopyHtoD(gpu_memory_offset, (uint8_t*)pinned.buffer + padding, indexbufferByteSize);
					}

					DeviceBuffer db;
					db.cptr = loaded->memory->cptr + gpu_memory_offset;
					db.size = indexbufferByteSize;
					db.index_min = index_min;
					db.index_max = index_max;
					accessorToDevicebufferMapping[task.accessorIndex] = db;

					size_indexbuffer_curast += indexbufferByteSize;
				});
			}

			pool.wait();

			//==================================================================
			// PROCESS IMAGES/TEXTURES
			//==================================================================

			for(auto task : imageLoadTasks){

				pool.enqueue([&, task](int threadIndex){
					cuCtxSetCurrent(context);

					gltfloader::Image& image = gltf.images[task.imageIndex];
					gltfloader::BufferView& view = gltf.bufferViews[image.bufferView];
					auto& buffer = gltf.buffers[view.buffer];

					uint8_t* ptr = ((uint8_t*)buffer.data) + view.byteOffset;

					int width, height, channels = 0;
					double t_start = now();
					uint8_t* imageData = stbi_load_from_memory(ptr, view.byteLength, &width, &height, &channels, 4);

					if(config.useJpegTextures){
						Texture* texture = createMipMappedJpegTexture(ptr, view.byteLength, width, height);
						loaded->textures[task.imageIndex] = texture;
					}else if(config.imageDivisionFactor == 1){
						Texture* texture = createMipMappedTexture(imageData, width, height);
						loaded->textures[task.imageIndex] = texture;
					}else{
						// RESIZE
						int newWidth = width / config.imageDivisionFactor;
						int newHeight = height / config.imageDivisionFactor;
						unsigned char* resizedData = (unsigned char*)malloc(newWidth * newHeight * 4);

						stbir_resize_uint8_linear(
							imageData,  width,  height, 0,   // src, w, h, stride (0 = tightly packed)
							resizedData, newWidth, newHeight, 0,   // dst, w, h, stride
							STBIR_RGBA
						);

						Texture* texture = createMipMappedTexture(resizedData, newWidth, newHeight);
						loaded->textures[task.imageIndex] = texture;

						free(resizedData);
					}
					
					stbi_image_free(imageData);
				});

			}

			//==================================================================
			// PROCESS VERTEX BUFFERS
			//==================================================================

			// Now we know how many bytes we need for (compressed) indices, 
			// and we know how much more we will need for positions. 
			// Commit sufficient amount of memory.
			uint64_t bytesNeeded = uint64_t(gpu_memory_counter);
			for(int accessorIndex : accessors_positions){
				gltfloader::Accessor accessor = gltf.accessors[accessorIndex];
				if(config.compress){
					bytesNeeded += roundUp(accessor.getByteSize() / 2ll, 12ll);
				}else{
					bytesNeeded += roundUp(accessor.getByteSize(), 12ll);
				}
			}
			bytesNeeded += 256;
			println("Allocating {:L} bytes of GPU memory", bytesNeeded);
			loaded->memory->commit(bytesNeeded);
			
			// Enforce some alignment for start of next batch of data.
			gpu_memory_counter = roundUp(uint64_t(gpu_memory_counter), 16llu);

			println("process vertex buffers");
			atomic_uint64_t size_positions_gltf = 0;
			atomic_uint64_t size_positions_curast = 0;
			for(int accessorIndex : accessors_positions){
				
				gltfloader::Accessor accessor = gltf.accessors[accessorIndex];
				gltfloader::BufferView bufferView = gltf.bufferViews[accessor.bufferView];
				gltfloader::Buffer& buffer = gltf.buffers[bufferView.buffer];

				uint64_t byteStride = accessor.getPackedStride();
				if(bufferView.byteStride != -1) byteStride = bufferView.byteStride;

				LoadTask task;
				task.accessorIndex = accessorIndex;
				task.byteOffset = buffer.offset + bufferView.byteOffset + accessor.byteOffset;
				task.byteSize = accessor.getByteSize();

				size_positions_gltf += task.byteSize;
				
				pool.enqueue([=, this, &gpu_memory_counter, &size_positions_curast](int threadIndex){
					cuCtxSetCurrent(context);

					gltfloader::Accessor accessor = gltf.accessors[task.accessorIndex];
					uint64_t numVertices = accessor.count;
					PinnedBuffer pinned = pinnedBuffers[threadIndex];

					uint64_t verticesPerBatch = (pinned.size - 2 * file->sectorSize) / byteStride;
					uint64_t numBatches = (numVertices + verticesPerBatch - 1) / verticesPerBatch;

					// We are reusing the staging buffers for indexbuffers, which may not be large enough for vertex buffers.
					// So process in batches that fit in staging buffers.

					if(config.compress){
						// Compressing coordinates to 3x16 bit integers
						uint64_t compressedSize = numVertices * 6;
						uint64_t gpu_memory_offset = gpu_memory_counter.fetch_add(compressedSize);

						DeviceBuffer db;
						db.cptr = loaded->memory->cptr + gpu_memory_offset;
						db.size = compressedSize;
						accessorToDevicebufferMapping[task.accessorIndex] = db;

						vec3 aabbSize = accessor.max - accessor.min;
						for(uint64_t batchIndex = 0; batchIndex < numBatches; batchIndex++){

							uint64_t processedVertices = batchIndex * verticesPerBatch;
							uint64_t verticesInBatch = min(verticesPerBatch, numVertices - processedVertices);
							uint64_t byteOffset = task.byteOffset + processedVertices * byteStride;
							uint64_t byteSize = verticesInBatch * byteStride;
							uint64_t padding = byteOffset % file->sectorSize;
							uint64_t paddedOffset = byteOffset - padding;
							uint64_t paddedSize = roundUp(int64_t(byteSize + padding), file->sectorSize);

							file->read(paddedOffset, paddedSize, pinned.buffer);

							for(uint64_t vertexIndex = 0; vertexIndex < verticesInBatch; vertexIndex++){
								vec3 position = read<vec3>(pinned.buffer, padding + byteStride * vertexIndex);
								
								uint16_t X = clamp(65536.0 * (position.x - accessor.min.x) / aabbSize.x, 0.0, 65535.0);
								uint16_t Y = clamp(65536.0 * (position.y - accessor.min.y) / aabbSize.y, 0.0, 65535.0);
								uint16_t Z = clamp(65536.0 * (position.z - accessor.min.z) / aabbSize.z, 0.0, 65535.0);

								write<uint16_t>(pinned.buffer, 6 * vertexIndex + 0, X);
								write<uint16_t>(pinned.buffer, 6 * vertexIndex + 2, Y);
								write<uint16_t>(pinned.buffer, 6 * vertexIndex + 4, Z);
							}

							uint64_t batch_compressedSize = verticesInBatch * 6;
							loaded->memory->memcopyHtoD(gpu_memory_offset, pinned.buffer, batch_compressedSize);

							gpu_memory_offset += batch_compressedSize;
							size_positions_curast += batch_compressedSize;
						}
					}else if(accessor.type == "VEC3"){
						uint64_t vertexbufferSize = numVertices * sizeof(vec3);
						uint64_t gpu_memory_offset = gpu_memory_counter.fetch_add(vertexbufferSize);

						DeviceBuffer db;
						db.cptr = loaded->memory->cptr + gpu_memory_offset;
						db.size = vertexbufferSize;
						accessorToDevicebufferMapping[task.accessorIndex] = db;

						vec3 aabbSize = accessor.max - accessor.min;
						for(uint64_t batchIndex = 0; batchIndex < numBatches; batchIndex++){

							uint64_t processedVertices = batchIndex * verticesPerBatch;
							uint64_t verticesInBatch = min(verticesPerBatch, numVertices - processedVertices);
							uint64_t byteOffset = task.byteOffset + processedVertices * byteStride;
							uint64_t byteSize = verticesInBatch * byteStride;
							uint64_t padding = byteOffset % file->sectorSize;
							uint64_t paddedOffset = byteOffset - padding;
							uint64_t paddedSize = roundUp(int64_t(byteSize + padding), file->sectorSize);

							file->read(paddedOffset, paddedSize, pinned.buffer);

							for(uint64_t vertexIndex = 0; vertexIndex < verticesInBatch; vertexIndex++){
								vec3 position = read<vec3>(pinned.buffer, padding + byteStride * vertexIndex);
								write<vec3>(pinned.buffer, 12 * vertexIndex, position);
							}

							uint64_t batch_buffersize = verticesInBatch * sizeof(vec3);
							loaded->memory->memcopyHtoD(gpu_memory_offset, (uint8_t*)pinned.buffer, batch_buffersize);

							gpu_memory_offset += batch_buffersize;
							size_positions_curast += batch_buffersize;
						}
					}else{
						println("unsupported bytes stride or component format");
						println("{}", stacktrace::current());
						__debugbreak();
						exit(12346347);
					}

				});
			}

			pool.wait();

			//==================================================================
			// PROCESS COLOR BUFFERS
			//==================================================================

			if(!config.skipVertexColors){
				bytesNeeded = uint64_t(gpu_memory_counter);
				for(int accessorIndex : accessors_colors){
					gltfloader::Accessor accessor = gltf.accessors[accessorIndex];
					bytesNeeded += accessor.count * sizeof(uint32_t);
				}
				bytesNeeded += 256;
				println("Allocating {:L} bytes of GPU memory", bytesNeeded);
				loaded->memory->commit(bytesNeeded);
				// Enforce some alignment for start of next batch of data.
				gpu_memory_counter = roundUp(uint64_t(gpu_memory_counter), 16llu);

				println("process color buffers");
				for(int accessorIndex : accessors_colors){
					
					gltfloader::Accessor accessor = gltf.accessors[accessorIndex];
					gltfloader::BufferView bufferView = gltf.bufferViews[accessor.bufferView];
					gltfloader::Buffer& buffer = gltf.buffers[bufferView.buffer];

					uint64_t byteOffset = buffer.offset + bufferView.byteOffset + accessor.byteOffset;
					uint64_t byteSize = accessor.getByteSize();
					
					pool.enqueue([=, &gpu_memory_counter](int threadIndex){
						cuCtxSetCurrent(context);

						uint64_t numVertices = accessor.count;
						PinnedBuffer pinned = pinnedBuffers[threadIndex];

						// We are reusing the staging buffers for indexbuffers, which may not be large enough for vertex buffers.
						// So process in batches that fit in staging buffers.
						uint64_t bytesPerVertex = sizeof(uint32_t);
						uint64_t verticesPerBatch = (pinned.size - 2 * file->sectorSize) / bytesPerVertex;
						uint64_t numBatches = (numVertices + verticesPerBatch - 1) / verticesPerBatch;
						uint64_t vertexbufferSize = numVertices * bytesPerVertex;
						uint64_t gpu_memory_offset = gpu_memory_counter.fetch_add(vertexbufferSize);

						DeviceBuffer db;
						db.cptr = loaded->memory->cptr + gpu_memory_offset;
						db.size = vertexbufferSize;
						accessorToDevicebufferMapping[accessorIndex] = db;

						// proto code for future support when source and target mem layout do not match.
						// But if component type is uint8_t and it's a vec4 or a vec3 with stride 4, 
						// then source layout matches our expected layout, and we don't need to convert.
						uint64_t typeCount = gltfloader::typeCount(accessor.type);
						auto readValue_vec3_u8 = [&](uint64_t i, uint64_t padding, uint64_t stride) -> uint32_t {
							uint32_t result = 0;
							uint8_t* result_u8 = (uint8_t*)&result;
							
							result_u8[0] = read<uint8_t>(pinned.buffer, padding + i * stride + 0);
							result_u8[1] = read<uint8_t>(pinned.buffer, padding + i * stride + 1);
							result_u8[2] = read<uint8_t>(pinned.buffer, padding + i * stride + 2);
							result_u8[3] = 255;

							return result;
						};

						auto readValue_vec3_f32 = [&](uint64_t i, uint64_t padding, uint64_t stride) -> uint32_t {
							uint32_t result = 0;
							uint8_t* result_u8 = (uint8_t*)&result;

							float r = read<float>(pinned.buffer, padding + i * stride + 0);
							float g = read<float>(pinned.buffer, padding + i * stride + 4);
							float b = read<float>(pinned.buffer, padding + i * stride + 8);

							while(r > 1.0f) r = r / 256.0f;
							while(g > 1.0f) g = g / 256.0f;
							while(b > 1.0f) b = b / 256.0f;
							
							result_u8[0] = r;
							result_u8[1] = g;
							result_u8[2] = b;
							result_u8[3] = 255;

							return result;
						};

						uint64_t sourceByteStride = 0;
						if(bufferView.byteStride != -1){
							sourceByteStride = bufferView.byteStride;
						}else{
							sourceByteStride = accessor.getPackedStride();
						}

						function<uint32_t(uint64_t, uint64_t, uint64_t)> readValue;
						bool sourceMatchesTarget = false;
						
						if(accessor.type == "VEC3" && accessor.componentType == gltfloader::COMPONENT_TYPE_UINT8 && sourceByteStride == 4){
							sourceMatchesTarget = true;
						}else if(accessor.type == "VEC4" && accessor.componentType == gltfloader::COMPONENT_TYPE_UINT8){
							// fits target
							sourceMatchesTarget = true;
						}else if(accessor.type == "VEC3" && accessor.componentType == gltfloader::COMPONENT_TYPE_FLOAT){
							readValue = readValue_vec3_f32;
						}else{
							println("ERROR: unsupported combination of accessor type and component type");
							println("{}", stacktrace::current());
							__debugbreak();
							exit(123572465);
						}
						
						for(uint64_t batchIndex = 0; batchIndex < numBatches; batchIndex++){

							uint64_t processedVertices = batchIndex * verticesPerBatch;
							uint64_t verticesInBatch = min(verticesPerBatch, numVertices - processedVertices);
							uint64_t batchByteOffset = byteOffset + processedVertices * sourceByteStride;
							uint64_t batchByteSize = verticesInBatch * sourceByteStride;
							uint64_t padding = batchByteOffset % file->sectorSize;
							uint64_t paddedOffset = batchByteOffset - padding;
							uint64_t paddedSize = roundUp(int64_t(batchByteSize + padding), file->sectorSize);

							file->read(paddedOffset, paddedSize, pinned.buffer);

							if(sourceMatchesTarget){
								uint64_t batch_buffersize = verticesInBatch * bytesPerVertex;
								loaded->memory->memcopyHtoD(gpu_memory_offset, (uint8_t*)pinned.buffer + padding, batch_buffersize);
								gpu_memory_offset += batch_buffersize;
							}else{
								// assume that source is bigger than target, which should be the case unless source is <4bytes

								for(uint64_t vertexIndex = 0; vertexIndex < verticesInBatch; vertexIndex++){
									uint32_t C = readValue(vertexIndex, padding, sourceByteStride);

									write<uint32_t>(pinned.buffer, 4 * vertexIndex, C);
								}

								uint64_t batch_buffersize = verticesInBatch * bytesPerVertex;
								loaded->memory->memcopyHtoD(gpu_memory_offset, pinned.buffer, batch_buffersize);
								gpu_memory_offset += batch_buffersize;
							}
						}
					});
				}

				pool.wait();
			}

			//==================================================================
			// PROCESS NORMAL BUFFERS
			//==================================================================

			if(!config.skipNormals){
				bytesNeeded = uint64_t(gpu_memory_counter);
				for(int accessorIndex : accessors_normals){
					gltfloader::Accessor accessor = gltf.accessors[accessorIndex];
					bytesNeeded += accessor.count * sizeof(vec3);
					
				}
				println("Allocating {:L} bytes of GPU memory", bytesNeeded);
				loaded->memory->commit(bytesNeeded + 256);
				// Enforce some alignment for start of next batch of data.
				gpu_memory_counter = roundUp(uint64_t(gpu_memory_counter), 16llu);

				println("process normal buffers");
				for(int accessorIndex : accessors_normals){
					
					gltfloader::Accessor accessor = gltf.accessors[accessorIndex];
					gltfloader::BufferView bufferView = gltf.bufferViews[accessor.bufferView];
					gltfloader::Buffer& buffer = gltf.buffers[bufferView.buffer];

					uint64_t byteOffset = buffer.offset + bufferView.byteOffset + accessor.byteOffset;
					uint64_t byteSize = accessor.getByteSize();
					uint64_t sourceByteStride = accessor.getPackedStride();
					if(bufferView.byteStride != -1) sourceByteStride = bufferView.byteStride;
					
					pool.enqueue([=, &gpu_memory_counter](int threadIndex){
						cuCtxSetCurrent(context);

						uint64_t numVertices = accessor.count;
						PinnedBuffer pinned = pinnedBuffers[threadIndex];

						// We are reusing the staging buffers for indexbuffers, which may not be large enough for vertex buffers.
						// So process in batches that fit in staging buffers.
						uint64_t bytesPerVertex = sizeof(vec3);
						uint64_t verticesPerBatch = (pinned.size - 2 * file->sectorSize) / sourceByteStride;
						uint64_t numBatches = (numVertices + verticesPerBatch - 1) / verticesPerBatch;
						uint64_t targetByteSize = numVertices * bytesPerVertex;
						uint64_t gpu_memory_offset = gpu_memory_counter.fetch_add(targetByteSize);

						DeviceBuffer db;
						db.cptr = loaded->memory->cptr + gpu_memory_offset;
						db.size = targetByteSize;
						accessorToDevicebufferMapping[accessorIndex] = db;

						function<uint32_t(uint64_t, uint64_t, uint64_t)> readValue;
						bool sourceMatchesTarget = false;
						
						for(uint64_t batchIndex = 0; batchIndex < numBatches; batchIndex++){

							uint64_t processedVertices = batchIndex * verticesPerBatch;
							uint64_t verticesInBatch = min(verticesPerBatch, numVertices - processedVertices);
							uint64_t batchByteOffset = byteOffset + processedVertices * sourceByteStride;
							uint64_t batchByteSize = verticesInBatch * sourceByteStride;
							uint64_t padding = batchByteOffset % file->sectorSize;
							uint64_t paddedOffset = batchByteOffset - padding;
							uint64_t paddedSize = roundUp(int64_t(batchByteSize + padding), file->sectorSize);

							file->read(paddedOffset, paddedSize, pinned.buffer);

							uint64_t batch_buffersize = verticesInBatch * bytesPerVertex;
							if(accessor.type == "VEC3" && accessor.componentType == gltfloader::COMPONENT_TYPE_FLOAT && sourceByteStride == 12){
								loaded->memory->memcopyHtoD(gpu_memory_offset, (uint8_t*)pinned.buffer + padding, batch_buffersize);
							}else if(accessor.type == "VEC3" && accessor.componentType == gltfloader::COMPONENT_TYPE_FLOAT){
								for(int vertexIndex = 0; vertexIndex < verticesInBatch; vertexIndex++){
									vec3 normal = read<vec3>(pinned.buffer, padding + sourceByteStride * vertexIndex);
									write<vec3>(pinned.buffer, sizeof(vec3) * vertexIndex, normal);
								}
								loaded->memory->memcopyHtoD(gpu_memory_offset, (uint8_t*)pinned.buffer, batch_buffersize);
							}else{
								println("unsupported");
								println("{}", stacktrace::current());
								exit(623546);
							}
							gpu_memory_offset += batch_buffersize;
						}
					});
				}

			}

			pool.wait();
			
			//==================================================================
			// PROCESS UV BUFFERS
			//==================================================================

			if(!config.skipUVs){
				println("process uvs buffers");

				// Commit sufficient amount of memory for UVs
				uint64_t bytesNeeded = uint64_t(gpu_memory_counter);
				for(int accessorIndex : accessors_uvs){
					gltfloader::Accessor accessor = gltf.accessors[accessorIndex];

					bytesNeeded += accessor.getByteSize();
				}
				bytesNeeded = roundUp(bytesNeeded, 256llu);
				println("Allocating {:L} bytes of GPU memory", bytesNeeded);
				loaded->memory->commit(bytesNeeded);
				// Enforce some alignment for start of next batch of data.
				gpu_memory_counter = roundUp(uint64_t(gpu_memory_counter), 16llu);
				
				for(int accessorIndex : accessors_uvs){

					gltfloader::Accessor accessor = gltf.accessors[accessorIndex];
					gltfloader::BufferView bufferView = gltf.bufferViews[accessor.bufferView];
					gltfloader::Buffer& buffer = gltf.buffers[bufferView.buffer];

					uint64_t byteStride = accessor.getPackedStride();
					if(bufferView.byteStride != -1) byteStride = bufferView.byteStride;

					LoadTask task;
					task.accessorIndex = accessorIndex;
					task.byteOffset = buffer.offset + bufferView.byteOffset + accessor.byteOffset;
					task.byteSize = accessor.getByteSize();
					
					pool.enqueue([=, this, &gpu_memory_counter](int threadIndex){
						cuCtxSetCurrent(context);

						gltfloader::Accessor accessor = gltf.accessors[task.accessorIndex];
						uint64_t numVertices = accessor.count;
						PinnedBuffer pinned = pinnedBuffers[threadIndex];

						uint64_t verticesPerBatch = (pinned.size - 2 * file->sectorSize) / byteStride;
						uint64_t numBatches = (numVertices + verticesPerBatch - 1) / verticesPerBatch;
						uint64_t targetByteSize = numVertices * sizeof(vec2);
						uint64_t gpu_memory_offset = gpu_memory_counter.fetch_add(targetByteSize);

						DeviceBuffer db;
						db.cptr = loaded->memory->cptr + gpu_memory_offset;
						db.size = targetByteSize;
						accessorToDevicebufferMapping[task.accessorIndex] = db;

						vec3 aabbSize = accessor.max - accessor.min;

						if(byteStride < sizeof(vec2)){
							println("Currently unsupported: Source byte stride {} must be larger than target byte stride {}",
								byteStride, sizeof(vec2));
							println("{}", stacktrace::current());
							exit(635367824);
						}

						for(uint64_t batchIndex = 0; batchIndex < numBatches; batchIndex++){

							uint64_t processedVertices = batchIndex * verticesPerBatch;
							uint64_t verticesInBatch = min(verticesPerBatch, numVertices - processedVertices);
							uint64_t byteOffset = task.byteOffset + processedVertices * byteStride;
							uint64_t sourceByteSize = verticesInBatch * byteStride;
							uint64_t padding = byteOffset % file->sectorSize;
							uint64_t paddedOffset = byteOffset - padding;
							uint64_t paddedSize = roundUp(int64_t(sourceByteSize + padding), file->sectorSize);

							file->read(paddedOffset, paddedSize, pinned.buffer);

							for(int vertexIndex = 0; vertexIndex < verticesInBatch; vertexIndex++){
								vec2 uv = read<vec2>(pinned.buffer, padding + byteStride * vertexIndex);
								write<vec2>(pinned.buffer, sizeof(vec2) * vertexIndex, uv);
							}

							uint64_t batch_buffersize = verticesInBatch * sizeof(vec2);
							loaded->memory->memcopyHtoD(gpu_memory_offset, pinned.buffer, batch_buffersize);

							gpu_memory_offset += batch_buffersize;
						}

					});
				}
			}

			pool.wait();

			

			//==================================================================
			// CREATE MESHES
			//==================================================================
			// We basically flatten the mesh-primitive structure of gltf to a 
			// mesh-only array, creating one curast mesh for each gltf primitive.
			// This vector gives us the index of the first curast mesh for a specific gltf mesh.
			// # gltf:
			//     mesh[0]
			//         primitive[0]
			//         primitive[1]
			//     mesh[1]
			//         primitive[0]
			//         primitive[1]
			//         primitive[2]
			// # CuRast:
			//     mesh[0]  (corresponds to mesh[0].primitive[0])
			//     mesh[1]  (corresponds to mesh[0].primitive[1])
			//     mesh[2]  (corresponds to mesh[1].primitive[0])
			//     mesh[3]  (corresponds to mesh[1].primitive[1])
			//     mesh[4]  (corresponds to mesh[1].primitive[2])
			// gltfToCurastOffset[0] gives "0", mapping gltf mesh[0] to curast mesh[0]
			// gltfToCurastOffset[1] gives "2", mapping gltf mesh[1] to curast mesh[2]
			vector<int> gltfToCurastOffset;
			uint64_t numUniqueTriangles = 0;
			uint64_t numUniqueNodes = 0;
			uint64_t numUniqueVertices = 0;
			for(int meshIndex = 0; meshIndex < gltf.meshes.size(); meshIndex++){

				gltfloader::Mesh& gltfmesh = gltf.meshes[meshIndex];

				gltfToCurastOffset.push_back(loaded->meshes.size());

				for(int primitiveIndex = 0; primitiveIndex < gltfmesh.primitives.size(); primitiveIndex++){
					gltfloader::Primitive& primitive = gltfmesh.primitives[primitiveIndex];

					Mesh mesh;
					mesh.isLoaded = true;
					mesh.cptr_uv = 0;
					mesh.cptr_color = 0;
					mesh.cptr_normal = 0;
					mesh.compressed = config.compress;

					if(primitive.attributes.contains("POSITION")){
						int accessorIndex = primitive.attributes["POSITION"];
						auto& accessor = gltf.accessors[accessorIndex];
						mesh.aabb.min = accessor.min;
						mesh.aabb.max = accessor.max;
						

						auto mapping = accessorToDevicebufferMapping[accessorIndex];
						assert(mapping.size != 0);

						mesh.cptr_position = mapping.cptr;

						#if defined(USE_VULKAN_SHARED_MEMORY)
							uint64_t offset = mapping.cptr - loaded->memory->cptr;
							mesh.vkc_position = *loaded->memory;
							mesh.vkc_position.deviceAddress = loaded->memory->deviceAddress + offset;
							mesh.vkc_position.offset = offset;
						#endif

						mesh.numVertices = accessor.count;
					}

					// if(mesh->numVertices != 533) continue;

					if(primitive.attributes.contains("TEXCOORD_0") && !config.skipUVs){
						int accessorIndex = primitive.attributes["TEXCOORD_0"];
						auto& accessor = gltf.accessors[accessorIndex];

						auto mapping = accessorToDevicebufferMapping[accessorIndex];
						assert(mapping.size != 0);

						mesh.cptr_uv = mapping.cptr;

						#if defined(USE_VULKAN_SHARED_MEMORY)
							uint64_t offset = mapping.cptr - loaded->memory->cptr;
							mesh.vkc_uv = *loaded->memory;
							mesh.vkc_uv.deviceAddress = loaded->memory->deviceAddress + offset;
							mesh.vkc_uv.offset = offset;
						#endif
					}

					if(primitive.attributes.contains("COLOR_0") && !config.skipVertexColors){
						int accessorIndex = primitive.attributes["COLOR_0"];
						auto& accessor = gltf.accessors[accessorIndex];

						auto mapping = accessorToDevicebufferMapping[accessorIndex];
						assert(mapping.size != 0);

						mesh.cptr_color = mapping.cptr;

						#if defined(USE_VULKAN_SHARED_MEMORY)
							uint64_t offset = mapping.cptr - loaded->memory->cptr;
							mesh.vkc_color = *loaded->memory;
							mesh.vkc_color.deviceAddress = loaded->memory->deviceAddress + offset;
						#endif
					}

					if(primitive.attributes.contains("NORMAL") && !config.skipNormals){
						int accessorIndex = primitive.attributes["NORMAL"];
						auto& accessor = gltf.accessors[accessorIndex];

						auto mapping = accessorToDevicebufferMapping[accessorIndex];
						assert(mapping.size != 0);

						mesh.cptr_normal = mapping.cptr;

						#if defined(USE_VULKAN_SHARED_MEMORY)
							uint64_t offset = mapping.cptr - loaded->memory->cptr;
							mesh.vkc_normal = *loaded->memory;
							mesh.vkc_normal.deviceAddress = loaded->memory->deviceAddress + offset;
						#endif
					}

					int64_t numTriangles = 0;
					if(primitive.indices != -1){
						auto& accessor = gltf.accessors[primitive.indices];
						numTriangles = accessor.count / 3;

						auto mapping = accessorToDevicebufferMapping[primitive.indices];
						assert(mapping.size != 0);

						mesh.index_min = mapping.index_min;
						mesh.index_max = mapping.index_max;
						mesh.cptr_indices = mapping.cptr;
						mesh.numTriangles = numTriangles;

						#if defined(USE_VULKAN_SHARED_MEMORY)
							uint64_t offset = mapping.cptr - loaded->memory->cptr;
							mesh.vkc_indices = *loaded->memory;
							mesh.vkc_indices.deviceAddress = loaded->memory->deviceAddress + offset;
							mesh.vkc_indices.offset = offset;
						#endif
					}else{
						mesh.index_min = 0;
						mesh.index_max = 0;
						mesh.cptr_indices = 0;
						mesh.numTriangles = mesh.numVertices / 3;
					}

					numUniqueTriangles += mesh.numTriangles;
					numUniqueVertices += mesh.numVertices;
					numUniqueNodes++;
					
					loaded->meshes.push_back(mesh);
				}

			}

			//==================================================================
			// CREATE SCENE NODES
			//==================================================================
			// This part is a bit hacky because our scene representation does not match gltf's.
			// - We don't have mesh&primitives, we only have meshes
			// - Makes representing gltf instances tricky. In gltf, meshes are instanced 
			//   when multiple nodes refer to the same mesh. 
			//   But a mesh is a collection of primitives that are also instanced.
			// - We also do instancing, but only on a mesh not a collection of primitives.
			// - So let's try this:
			//     - Make each primitive a separate mesh.
			//     - If gltf instances a mesh(primitive collection), we create separate instances for each primitive. 

			loaded->glbNode = make_shared<SceneNode>(path);

			vector<vector<shared_ptr<SceneNode>>> sceneNodes;

			uint64_t totalTriangles = 0;
			for(int nodeIndex = 0; nodeIndex < gltf.nodes.size(); nodeIndex++){

				// if(nodeIndex > 1) break;

				gltfloader::Node& gltfNode = gltf.nodes[nodeIndex];

				vector<shared_ptr<SceneNode>> nodesInGltfNode;

				if(gltfNode.mesh != -1){
					// // TODO:  skip meshes with Ivy
					gltfloader::Mesh& gltfmesh = gltf.meshes[gltfNode.mesh];
					// if(path.contains("zorah_main_public") &&  gltfmesh.name.contains("Ivy")){
					// 	println("dropped mesh {}", gltfmesh.name);
					// 	continue;
					// }

					for(int primitiveIndex = 0; primitiveIndex < gltfmesh.primitives.size(); primitiveIndex++){
						gltfloader::Primitive& primitive = gltfmesh.primitives[primitiveIndex];

						int meshOffset = gltfToCurastOffset[gltfNode.mesh] + primitiveIndex;
						Mesh& mesh = loaded->meshes[meshOffset];

						// only add floor of sponza to scene
						// if(mesh.numTriangles > 7) continue;

						totalTriangles += mesh.numTriangles;

						auto sceneNode = make_shared<SNTriangles>(gltfNode.name);
						sceneNode->transform = gltfNode.matrix;
						sceneNode->name = gltfNode.name;
						sceneNode->mesh = &mesh;
						sceneNode->aabb = mesh.aabb;
						sceneNode->texture = loaded->defaultTexture;

						if (primitive.material != -1) {
							gltfloader::Material& material = gltf.materials[primitive.material];

							if(material.texture != -1){
								gltfloader::Texture texture = gltf.textures[material.texture];
								sceneNode->texture = loaded->textures[texture.source];
							}
						}
						
						// loaded->glbNode->children.push_back(sceneNode);
						// Box3 aabb_world = mesh.aabb.transform(sceneNode->transform);
						// loaded->glbNode->aabb.extend(aabb_world.min);
						// loaded->glbNode->aabb.extend(aabb_world.max);

						nodesInGltfNode.push_back(sceneNode);
					}
				}else{
					auto sceneNode = make_shared<SceneNode>(gltfNode.name);
					sceneNode->transform = gltfNode.matrix;
					sceneNode->name = gltfNode.name;

					nodesInGltfNode.push_back(sceneNode);
				}

				sceneNodes.push_back(nodesInGltfNode);

				// println("Loaded {:L} triangles", totalTriangles);
			}

			// Establish node hierarchy.
			// A bit wonky because we have one scene node for each mesh primitive. 
			vector<bool> hasParentFlags(sceneNodes.size(), false);
			for(int nodeIndex = 0; nodeIndex < gltf.nodes.size(); nodeIndex++){

				gltfloader::Node& gltfNode = gltf.nodes[nodeIndex];

				shared_ptr<SceneNode> representative = sceneNodes[nodeIndex][0];

				for(int childIndex : gltfNode.children){
					hasParentFlags[childIndex] = true;

					vector<shared_ptr<SceneNode>>& childNodes = sceneNodes[childIndex];

					for(shared_ptr<SceneNode> childNode : childNodes){
						representative->children.push_back(childNode);
					}
				}
			}

			// Add all nodes without parent to glb node
			for(int nodeIndex = 0; nodeIndex < gltf.nodes.size(); nodeIndex++){

				bool hasParent = hasParentFlags[nodeIndex];
				if(hasParent) continue;

				for(auto sceneNode : sceneNodes[nodeIndex]){
					loaded->glbNode->children.push_back(sceneNode);
				}
			}

			{// Update transformations and compute global bounding box

				Box3 aabb_global;
				function<void(SceneNode*, SceneNode*)> traverse;
				traverse = [&](SceneNode* parent, SceneNode* node){
					if (parent) {
						mat4 transform = parent->transform_global * node->transform;
						node->transform_global = transform;
					} else {
						node->transform_global = node->transform;
					}

					for(auto child : node->children){
						traverse(node, child.get());
					}

					bool isMesh = dynamic_cast<SNTriangles*>(node) != nullptr;
					if(isMesh){
						Box3 aabb_world = node->aabb.transform(node->transform_global);
						aabb_global.extend(aabb_world.min);
						aabb_global.extend(aabb_world.max);
					}
				};
				traverse(nullptr, loaded->glbNode.get());

				loaded->glbNode->aabb = aabb_global;
			}

			// Upon request, we drop the few nodes named Ivy from Zorah.
			if(path.contains("zorah_main_public")){
				vector<std::shared_ptr<SceneNode>> filtered;
				for(int i = 0; i < loaded->glbNode->children.size(); i++){
					auto child = loaded->glbNode->children[i];

					if(child->name.contains("Ivy")){
						println("dropped node {}", child->name);
						continue;
					}else{
						filtered.push_back(child);
					}
				}

				loaded->glbNode->children = filtered;
			}

			// println("bytesRead:          {:L}", uint64_t(bytesRead));
			// println("gpu_memory_counter: {:L}", uint64_t(gpu_memory_counter));

			double seconds = now() - t_start;

			// uint64_t totalGpuMemory = MemoryManager::getTotalAllocatedMemory();

			// println("duration: {:.1f} seconds", seconds);
			// println("totalGpuMemory: {:.1f} GB", double(totalGpuMemory) / 1'000'000'000.0);

			// PRINT STATS
			println("================ STATS =================");
			println("path                      {}", path);
			println("nodes                       ");
			println("    unique                {:L}", numUniqueNodes);
			println("    with instances        {:L}", loaded->glbNode->children.size());
			println("triangles                   ");
			println("    unique                {:L}", numUniqueTriangles);
			println("    with instances        {:L}", totalTriangles);
			println("========================================");
			println("size_indexbuffer_gltf     {:L}", uint64_t(size_indexbuffer_gltf));
			println("size_indexbuffer_curast   {:L}", uint64_t(size_indexbuffer_curast));
			println("size_positions_gltf       {:L}", uint64_t(size_positions_gltf));
			println("size_positions_curast     {:L}", uint64_t(size_positions_curast));
			println("========================================");
			uint64_t numTexels = 0;
			for(int i = 0; i < loaded->textures.size(); i++){
				Texture* texture = loaded->textures[i];
				numTexels += texture->width * texture->height;
				println("texture, size: {} x {}", texture->width, texture->height);
			}
			println("total pixels: {:L} M", double(numTexels) / 1'000'000.0);
			println("duration: {:.1f} sec", seconds);
			println("========================================");

			return loaded;
		}

		void close(){
			
		}
	};

	shared_ptr<LoadedGlb> load(string path, CUcontext context, LoadConfig config = LoadConfig()){
		auto loader = make_shared<LargeGlbLoader>();
		loader->path = path;
		loader->context = context;
		loader->config = config;

		auto glb = loader->run();

		return glb;
	};



};




