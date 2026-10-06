#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <print>
#include <fstream>
#include <filesystem>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "json/json.hpp"
#include "stb/stb_image.h"

#include "SceneNode.h"
#include "types.h"
#include "unsuck.hpp"
#include "CURuntime.h"
#include "MemoryManager.h"
#include "./kernels/HostDeviceInterface.h"

using std::string;
using std::vector;
using std::println;

// Clustered LOD mesh, as produced by tools/clodbuilder (see tools/clodbuilder/README.md).
// - The constructor loads all files into RAM, and memory-maps the cluster, vertex and triangle files. Linux-only for now.
// - Kernels read clusters, vertices and triangles either from VRAM or from the memory-mapped files,
//   see CuRastSettings::clusterRenderPath. The texture is always in VRAM.
// - GPU resources are created on first draw, because the CUDA context is not yet available while the scene is set up.
struct ClusteredMeshNode : public SceneNode{

	struct MappedFile{
		void* ptr = nullptr;
		i64 size = 0;
	};

	string dir = "";

	// RAM
	vector<Cluster> clusters;
	vector<vec3> positions;
	vector<vec2> uvs;
	vector<u8> triangles;      // 3 cluster-local vertex indices per triangle
	vector<u8> textureRGBA;    // decoded base color texture, empty if there is none
	int textureWidth = 0;
	int textureHeight = 0;

	// memory-mapped files, with the same content as clusters, positions, uvs and triangles
	MappedFile mapped_clusters;
	MappedFile mapped_positions;
	MappedFile mapped_uvs;
	MappedFile mapped_triangles;

	// VRAM
	bool gpuInitialized = false;     // per-frame buffers and texture, see initGpu()
	bool geometryUploaded = false;   // clusters, vertices and triangles, see uploadGeometry()
	Cluster* gpu_clusters = nullptr;
	vec3* gpu_positions = nullptr;
	vec2* gpu_uvs = nullptr;
	u8* gpu_triangles = nullptr;
	cudaMipmappedArray_t gpu_textureArray = nullptr;
	cudaTextureObject_t gpu_texture = 0;
	u32* gpu_visibleClusters = nullptr;  // indices of the clusters selected for the current frame
	u32* gpu_counters = nullptr;         // [0]: number of visible clusters, [1]: number of their triangles

	ClusteredMeshNode(string dir, string name) : SceneNode(name){
		this->dir = dir;

		double tStart = now();

		string metadataPath = dir + "/metadata.json";
		if(!fs::exists(metadataPath)){
			println("ERROR: clustered mesh metadata not found: {}", metadataPath);
			exit(7234561);
		}

		using json = nlohmann::json;
		json j;
		try{
			std::ifstream stream(metadataPath);
			j = json::parse(stream);
		}catch(const json::exception& e){
			println("ERROR: failed to parse {}: {}", metadataPath, e.what());
			exit(7234562);
		}

		const json& files = j["files"];
		clusters  = readArray<Cluster>(files["clusters"]);
		positions = readArray<vec3>(files["positions"]);
		uvs       = readArray<vec2>(files["uvs"]);
		triangles = readArray<u8>(files["triangles"]);

		// file sizes were already validated by readArray
		mapped_clusters  = mapFile(dir + "/" + files["clusters"]["file"].get<string>());
		mapped_positions = mapFile(dir + "/" + files["positions"]["file"].get<string>());
		mapped_uvs       = mapFile(dir + "/" + files["uvs"]["file"].get<string>());
		mapped_triangles = mapFile(dir + "/" + files["triangles"]["file"].get<string>());

		auto toVec3 = [](const json& arr) -> vec3 {
			return {arr[0].get<float>(), arr[1].get<float>(), arr[2].get<float>()};
		};
		aabb.min = toVec3(j["boundingBox"]["min"]);
		aabb.max = toVec3(j["boundingBox"]["max"]);

		if(j.contains("texture")){
			string texturePath = dir + "/" + j["texture"]["file"].get<string>();

			int numChannels;
			u8* data = stbi_load(texturePath.c_str(), &textureWidth, &textureHeight, &numChannels, 4);
			if(data == nullptr){
				println("ERROR: failed to load texture {}: {}", texturePath, stbi_failure_reason());
				exit(7234563);
			}

			textureRGBA.assign(data, data + u64(textureWidth) * u64(textureHeight) * 4);
			stbi_image_free(data);
		}

		println("loaded {} clusters, {} triangles, {}x{} texture from {} in {:.1f}s",
			clusters.size(), triangles.size() / 3, textureWidth, textureHeight, dir, now() - tStart);
	}

	// owns the GPU resources and the mappings
	ClusteredMeshNode(const ClusteredMeshNode&) = delete;
	ClusteredMeshNode& operator=(const ClusteredMeshNode&) = delete;

	~ClusteredMeshNode(){
		if(geometryUploaded){
			MemoryManager::free(gpu_clusters);
			MemoryManager::free(gpu_positions);
			MemoryManager::free(gpu_uvs);
			MemoryManager::free(gpu_triangles);
		}

		if(gpuInitialized){
			MemoryManager::free(gpu_visibleClusters);
			MemoryManager::free(gpu_counters);

			if(gpu_texture != 0) cudaDestroyTextureObject(gpu_texture);
			if(gpu_textureArray != nullptr) cudaFreeMipmappedArray(gpu_textureArray);
		}

		for(MappedFile* file : {&mapped_clusters, &mapped_positions, &mapped_uvs, &mapped_triangles}){
			if(file->ptr != nullptr) munmap(file->ptr, file->size);
			file->ptr = nullptr;
		}
	}

	// Allocates the per-frame buffers and uploads the texture. Needed by both render paths.
	void initGpu(){
		if(gpuInitialized) return;

		gpu_visibleClusters = (u32*)MemoryManager::alloc(clusters.size() * sizeof(u32), name + " visible clusters");
		gpu_counters        = (u32*)MemoryManager::alloc(2 * sizeof(u32), name + " counters");

		if(!textureRGBA.empty()){
			uploadTexture();
		}

		gpuInitialized = true;
	}

	// Copies clusters, vertices and triangles to VRAM, for the VRAM render path.
	// They stay in VRAM when switching to the memory-mapped render path.
	void uploadGeometry(){
		if(geometryUploaded) return;

		double tStart = now();

		gpu_clusters  = (Cluster*)upload(clusters, name + " clusters");
		gpu_positions = (vec3*)upload(positions, name + " positions");
		gpu_uvs       = (vec2*)upload(uvs, name + " uvs");
		gpu_triangles = (u8*)upload(triangles, name + " triangles");

		geometryUploaded = true;

		println("uploaded {} to VRAM in {:.1f}s", name, now() - tStart);
	}

private:

	static MappedFile mapFile(string path){
		MappedFile file;

		int fd = open(path.c_str(), O_RDONLY);
		if(fd == -1){
			println("ERROR: failed to open {}", path);
			exit(7234566);
		}

		struct stat st;
		if(fstat(fd, &st) != 0){
			println("ERROR: fstat failed for {}", path);
			close(fd);
			exit(7234567);
		}
		file.size = st.st_size;

		// mmap does not support empty files
		if(file.size == 0){
			close(fd);
			return file;
		}

		file.ptr = mmap(nullptr, file.size, PROT_READ, MAP_SHARED, fd, 0);
		close(fd); // the mapping stays valid after closing the file descriptor

		if(file.ptr == MAP_FAILED){
			println("ERROR: mmap failed for {}", path);
			exit(7234568);
		}

		return file;
	}

	// Reads a file listed in metadata.json, and checks that its size matches count and stride
	template<typename T>
	vector<T> readArray(const nlohmann::json& file){
		string path = dir + "/" + file["file"].get<string>();
		u64 count   = file["count"].get<u64>();
		u64 stride  = file["stride"].get<u64>();

		u64 byteSize = count * stride;
		if(byteSize % sizeof(T) != 0 || !fs::exists(path) || fs::file_size(path) != byteSize){
			println("ERROR: {} is missing or does not contain {} elements with {} bytes each", path, count, stride);
			exit(7234564);
		}

		vector<T> data(byteSize / sizeof(T));
		std::ifstream stream(path, std::ios::binary);
		stream.read((char*)data.data(), byteSize);

		if(!stream){
			println("ERROR: failed to read {}", path);
			exit(7234565);
		}

		return data;
	}

	template<typename T>
	static void* upload(const vector<T>& data, string label){
		void* ptr = MemoryManager::alloc(byteSizeOf(data), label);
		CURuntime::assertCudaSuccess(cudaMemcpy(ptr, data.data(), byteSizeOf(data), cudaMemcpyHostToDevice));

		return ptr;
	}

	// Uploads the texture with a full mip chain, so that minified textures don't alias
	void uploadTexture(){
		int numLevels = 1 + int(std::floor(std::log2(double(std::max(textureWidth, textureHeight)))));

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
		cudaExtent extent = make_cudaExtent(textureWidth, textureHeight, 0);
		CURuntime::assertCudaSuccess(cudaMallocMipmappedArray(&gpu_textureArray, &channelDesc, extent, numLevels));

		const u8* levelData = textureRGBA.data();
		vector<u8> nextLevel;
		vector<u8> currentLevel;
		int levelWidth = textureWidth;
		int levelHeight = textureHeight;

		for(int level = 0; level < numLevels; level++){
			cudaArray_t levelArray;
			CURuntime::assertCudaSuccess(cudaGetMipmappedArrayLevel(&levelArray, gpu_textureArray, level));
			CURuntime::assertCudaSuccess(cudaMemcpy2DToArray(
				levelArray, 0, 0, levelData, levelWidth * 4, levelWidth * 4, levelHeight, cudaMemcpyHostToDevice));

			if(level + 1 == numLevels) break;

			// next level: average of 2x2 texels
			int nextWidth = std::max(1, levelWidth / 2);
			int nextHeight = std::max(1, levelHeight / 2);
			nextLevel.resize(u64(nextWidth) * u64(nextHeight) * 4);

			for(int y = 0; y < nextHeight; y++)
			for(int x = 0; x < nextWidth; x++)
			for(int c = 0; c < 4; c++){
				int x0 = std::min(2 * x, levelWidth - 1);
				int x1 = std::min(2 * x + 1, levelWidth - 1);
				int y0 = std::min(2 * y, levelHeight - 1);
				int y1 = std::min(2 * y + 1, levelHeight - 1);

				u32 sum = levelData[4 * (u64(y0) * levelWidth + x0) + c]
				        + levelData[4 * (u64(y0) * levelWidth + x1) + c]
				        + levelData[4 * (u64(y1) * levelWidth + x0) + c]
				        + levelData[4 * (u64(y1) * levelWidth + x1) + c];

				nextLevel[4 * (u64(y) * nextWidth + x) + c] = (sum + 2) / 4;
			}

			std::swap(currentLevel, nextLevel);
			levelData = currentLevel.data();
			levelWidth = nextWidth;
			levelHeight = nextHeight;
		}

		cudaResourceDesc resourceDesc = {};
		resourceDesc.resType = cudaResourceTypeMipmappedArray;
		resourceDesc.res.mipmap.mipmap = gpu_textureArray;

		// clamp to edge, as in the glTF sampler of the source mesh
		cudaTextureDesc textureDesc = {};
		textureDesc.addressMode[0]      = cudaAddressModeClamp;
		textureDesc.addressMode[1]      = cudaAddressModeClamp;
		textureDesc.filterMode          = cudaFilterModeLinear;
		textureDesc.mipmapFilterMode    = cudaFilterModeLinear;
		textureDesc.readMode            = cudaReadModeNormalizedFloat;
		textureDesc.normalizedCoords    = 1;
		textureDesc.maxMipmapLevelClamp = float(numLevels - 1);

		CURuntime::assertCudaSuccess(cudaCreateTextureObject(&gpu_texture, &resourceDesc, &textureDesc, nullptr));
	}

};
