#pragma once

#include <string>
#include <vector>
#include <cstring>
#include <print>
#include <fstream>
#include <filesystem>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "json/json.hpp"

#include "SceneNode.h"
#include "types.h"
#include "unsuck.hpp"
#include "CURuntime.h"
#include "MemoryManager.h"
#include "./kernels/HostDeviceInterface.h"

using std::string;
using std::vector;
using std::println;

// Records of groups.bin, see tools/clodbuilder/README.md
struct ClusterGroup{
	vec4 sphere;           // xyz: center, w: radius
	float error;           // FLT_MAX for terminal groups
	u32 level;
	u32 clusterOffset;     // first cluster of this group
	u32 clusterCount;
};
static_assert(sizeof(ClusterGroup) == 32);

// Records of nodes.bin: A BVH over the groups, with one tree per level. The root of level i is node i.
struct ClusterBvhNode{
	vec4 sphere;           // encloses all groups in this subtree
	float error;           // maximum error of all groups in this subtree
	i32 group;             // leaf: index into groups, internal node: -1
	u32 childOffset;       // internal node: children are nodes[childOffset, childOffset + childCount)
	u32 childCount;
};
static_assert(sizeof(ClusterBvhNode) == 32);

// Clustered LOD mesh, as produced by tools/clodbuilder (see tools/clodbuilder/README.md),
// with its texture converted to a BC7-compressed texture.dds.
// - The constructor loads all files into RAM, and memory-maps them. Linux-only for now.
// - With the memory-mapped render path (CuRastSettings::clusterRenderPath), the BVH traversal on the CPU reads nodes,
//   groups and clusters from the memory-mapped files, and kernels read clusters, vertices, triangles and the texture
//   from them. Otherwise, the traversal uses the copies in RAM, and kernels the copies in VRAM.
// - GPU resources are created on first draw, because the CUDA context is not yet available while the scene is set up.
struct ClusteredMeshNode : public SceneNode{

	struct MappedFile{
		void* ptr = nullptr;
		i64 size = 0;
	};

	string dir = "";

	// RAM
	vector<ClusterGroup> groups;
	vector<ClusterBvhNode> nodes;
	u32 numLevels = 0;         // number of BVH roots
	vector<Cluster> clusters;
	vector<vec3> positions;
	vector<vec2> uvs;
	vector<u8> triangles;      // 3 cluster-local vertex indices per triangle

	// BC7-compressed texture with mip levels. Empty if there is none.
	vector<u8> textureDds;     // the whole dds file
	u64 textureDataOffset = 0; // the blocks of all levels start here
	u32 textureWidth = 0;
	u32 textureHeight = 0;
	u32 textureLevels = 0;

	// memory-mapped files, with the same content as groups, nodes, clusters, positions, uvs, triangles and textureDds
	MappedFile mapped_groups;
	MappedFile mapped_nodes;
	MappedFile mapped_clusters;
	MappedFile mapped_positions;
	MappedFile mapped_uvs;
	MappedFile mapped_triangles;
	MappedFile mapped_texture;

	// VRAM
	bool gpuInitialized = false;     // per-frame buffers, see initGpu()
	bool vramUploaded = false;       // see uploadToVram()
	Cluster* gpu_clusters = nullptr;
	vec3* gpu_positions = nullptr;
	vec2* gpu_uvs = nullptr;
	u8* gpu_triangles = nullptr;
	u8* gpu_texture = nullptr;           // the blocks of all levels, without the dds header
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
		groups    = readArray<ClusterGroup>(files["groups"]);
		nodes     = readArray<ClusterBvhNode>(files["nodes"]);
		numLevels = j["counts"]["levels"].get<u32>();
		clusters  = readArray<Cluster>(files["clusters"]);
		positions = readArray<vec3>(files["positions"]);
		uvs       = readArray<vec2>(files["uvs"]);
		triangles = readArray<u8>(files["triangles"]);

		// file sizes were already validated by readArray
		mapped_groups    = mapFile(dir + "/" + files["groups"]["file"].get<string>());
		mapped_nodes     = mapFile(dir + "/" + files["nodes"]["file"].get<string>());
		mapped_clusters  = mapFile(dir + "/" + files["clusters"]["file"].get<string>());
		mapped_positions = mapFile(dir + "/" + files["positions"]["file"].get<string>());
		mapped_uvs       = mapFile(dir + "/" + files["uvs"]["file"].get<string>());
		mapped_triangles = mapFile(dir + "/" + files["triangles"]["file"].get<string>());

		auto toVec3 = [](const json& arr) -> vec3 {
			return {arr[0].get<float>(), arr[1].get<float>(), arr[2].get<float>()};
		};
		aabb.min = toVec3(j["boundingBox"]["min"]);
		aabb.max = toVec3(j["boundingBox"]["max"]);

		// clodbuilder writes the source's textures as they are. We use texture.dds, the BC7-compressed version
		// (or atlas, if there are multiple textures) created by tools/clodbuilder/convert_textures.py.
		if(j.contains("texture")){
			fs::path texturePath = fs::path(dir) / "texture.dds";

			if(fs::exists(texturePath)){
				loadTexture(texturePath.string());
			}else{
				println("WARNING: {} not found, rendering without texture. See tools/clodbuilder/README.md on how to create it.", texturePath.string());
			}
		}

		println("loaded {} clusters, {} triangles, {}x{} texture with {} levels from {} in {:.1f}s",
			clusters.size(), triangles.size() / 3, textureWidth, textureHeight, textureLevels, dir, now() - tStart);
	}

	// owns the GPU resources and the mappings
	ClusteredMeshNode(const ClusteredMeshNode&) = delete;
	ClusteredMeshNode& operator=(const ClusteredMeshNode&) = delete;

	~ClusteredMeshNode(){
		if(vramUploaded){
			MemoryManager::free(gpu_clusters);
			MemoryManager::free(gpu_positions);
			MemoryManager::free(gpu_uvs);
			MemoryManager::free(gpu_triangles);
			if(gpu_texture != nullptr) MemoryManager::free(gpu_texture);
		}

		if(gpuInitialized){
			MemoryManager::free(gpu_visibleClusters);
			MemoryManager::free(gpu_counters);
		}

		for(MappedFile* file : {&mapped_groups, &mapped_nodes, &mapped_clusters, &mapped_positions, &mapped_uvs, &mapped_triangles, &mapped_texture}){
			if(file->ptr != nullptr) munmap(file->ptr, file->size);
			file->ptr = nullptr;
		}
	}

	// Allocates the per-frame buffers. Needed by both render paths.
	void initGpu(){
		if(gpuInitialized) return;

		gpu_visibleClusters = (u32*)MemoryManager::alloc(clusters.size() * sizeof(u32), name + " visible clusters");
		gpu_counters        = (u32*)MemoryManager::alloc(2 * sizeof(u32), name + " counters");

		gpuInitialized = true;
	}

	// Copies clusters, vertices, triangles and the texture's blocks to VRAM, for the VRAM render path.
	// They stay in VRAM when switching to the memory-mapped render path.
	void uploadToVram(){
		if(vramUploaded) return;

		double tStart = now();

		gpu_clusters  = (Cluster*)upload(clusters.data(), byteSizeOf(clusters), name + " clusters");
		gpu_positions = (vec3*)upload(positions.data(), byteSizeOf(positions), name + " positions");
		gpu_uvs       = (vec2*)upload(uvs.data(), byteSizeOf(uvs), name + " uvs");
		gpu_triangles = (u8*)upload(triangles.data(), byteSizeOf(triangles), name + " triangles");

		if(!textureDds.empty()){
			gpu_texture = (u8*)upload(textureDds.data() + textureDataOffset, textureDds.size() - textureDataOffset, name + " texture (BC7)");
		}

		vramUploaded = true;

		println("uploaded {} to VRAM in {:.1f}s", name, now() - tStart);
	}

	// What the BVH traversal on the CPU reads, from RAM or from the memory-mapped files
	struct BvhData{
		const ClusterBvhNode* nodes;
		const ClusterGroup* groups;
		const Cluster* clusters;
		u32 numLevels;
	};

	BvhData getBvhData(bool memoryMapped){
		if(memoryMapped){
			return {(ClusterBvhNode*)mapped_nodes.ptr, (ClusterGroup*)mapped_groups.ptr, (Cluster*)mapped_clusters.ptr, numLevels};
		}else{
			return {nodes.data(), groups.data(), clusters.data(), numLevels};
		}
	}

	// The texture's blocks, in VRAM or in the memory-mapped dds file
	BC7Texture getTexture(bool memoryMapped){
		BC7Texture texture = {};
		if(textureDds.empty()) return texture;

		texture.data      = memoryMapped ? (u8*)mapped_texture.ptr + textureDataOffset : gpu_texture;
		texture.width     = textureWidth;
		texture.height    = textureHeight;
		texture.numLevels = textureLevels;

		return texture;
	}

private:

	// Loads a dds file with a BC7-compressed 2D texture into RAM, and memory-maps it
	void loadTexture(string path){
		std::ifstream stream(path, std::ios::binary);
		textureDds.resize(fs::file_size(path));
		stream.read((char*)textureDds.data(), textureDds.size());

		auto read32 = [&](u64 offset){
			u32 value;
			memcpy(&value, &textureDds[offset], 4);
			return value;
		};

		// "DDS", the 124 byte header, and the 20 byte DX10 header with the format.
		// 98 and 99: DXGI_FORMAT_BC7_UNORM and DXGI_FORMAT_BC7_UNORM_SRGB, 3: 2D texture.
		bool isBC7 = stream
			&& textureDds.size() >= 148
			&& memcmp(&textureDds[0], "DDS ", 4) == 0
			&& memcmp(&textureDds[84], "DX10", 4) == 0
			&& (read32(128) == 98 || read32(128) == 99)
			&& read32(132) == 3
			&& read32(140) == 1;

		if(!isBC7){
			println("ERROR: {} is not a BC7-compressed 2D texture", path);
			exit(7234569);
		}

		textureHeight     = read32(12);
		textureWidth      = read32(16);
		textureLevels     = std::max(read32(28), 1u);
		textureDataOffset = 148;

		// levels are stored one after another, starting with the largest
		u64 dataSize = 0;
		for(u32 level = 0; level < textureLevels && level < 32; level++){
			u64 levelWidth  = std::max(textureWidth >> level, 1u);
			u64 levelHeight = std::max(textureHeight >> level, 1u);
			dataSize += 16 * ((levelWidth + 3) / 4) * ((levelHeight + 3) / 4);
		}

		if(textureLevels > 32 || textureDataOffset + dataSize > textureDds.size()){
			println("ERROR: {} is smaller than its {}x{} texture with {} levels", path, textureWidth, textureHeight, textureLevels);
			exit(7234570);
		}

		mapped_texture = mapFile(path);
	}

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

	static void* upload(const void* data, u64 size, string label){
		void* ptr = MemoryManager::alloc(size, label);
		CURuntime::assertCudaSuccess(cudaMemcpy(ptr, data, size, cudaMemcpyHostToDevice));

		return ptr;
	}

};
