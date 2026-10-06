#pragma once

#include <string>
#include <vector>
#include <cstring>
#include <print>
#include <fstream>
#include <filesystem>

#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cufile.h>

#include "json/json.hpp"

#include "SceneNode.h"
#include "types.h"
#include "./kernels/HostDeviceInterface.h"

using std::string;
using std::vector;
using std::println;
using glm::ivec2;
using glm::dvec3;

// Attribute description from potree's metadata.json
struct PotreeAttribute{
	string name;
	string description;
	string type;              // e.g. "int32", "uint16", "uint8", ...
	i64 size = 0;             // bytes per point for this attribute
	i64 numElements = 0;
	i64 elementSize = 0;
	i64 byteOffset = 0;       // byte offset of this attribute within a point record
	vector<double> min;       // numElements values
	vector<double> max;
	vector<double> scale;
	vector<double> offset;
	vector<i64> histogram;    // only present for some attributes, e.g. classification
};

// Octree node from potree's hierarchy.bin
struct PotreeHierarchyNode{
	static constexpr u8 TYPE_NORMAL = 0;
	static constexpr u8 TYPE_LEAF   = 1;
	static constexpr u8 TYPE_PROXY  = 2;  // hierarchy chunk of this node and its descendants is not yet loaded

	u8 type = TYPE_NORMAL;
	u8 childMask = 0;
	u32 numPoints = 0;
	u64 byteOffset = 0;           // location of this node's points in octree.bin. Only valid if not a proxy.
	u64 byteSize = 0;
	u64 hierarchyByteOffset = 0;  // proxies: location of the hierarchy chunk in hierarchy.bin
	u64 hierarchyByteSize = 0;
	i32 level = 0;
	dvec3 min;                    // bounding box, same coordinate system as metadata's boundingBox
	dvec3 max;
	i32 children[8] = {-1, -1, -1, -1, -1, -1, -1, -1}; // indices into PotreeFileNode::hierarchyNodes
};

// Loads a pointcloud converted to the Potree 2.0 format (https://github.com/potree/PotreeConverter),
// and memory-maps its hierarchy.bin and octree.bin. octree.bin can also be read via cuFile, 
// for the direct storage render path. Linux-only for now.
struct PotreeFileNode : public SceneNode{

	string dir = "";
	void* mapped_hierarchy = nullptr;
	void* mapped_octree = nullptr;
	i64 hierarchyFileSize = 0;
	i64 octreeFileSize = 0;

	// octree.bin, opened for reads via cuFile (GPUDirect Storage) by getOctreeCuFileHandle()
	int octreeDirectFd = -1;
	CUfileHandle_t octreeCuFileHandle = nullptr;
	bool octreeCuFileFailed = false;

	// metadata from metadata.json
	string version;
	string metadataName;      // "name" in metadata.json. Not to be confused with SceneNode::name
	string description;
	string projection;
	string encoding;          // "DEFAULT" (uncompressed) or "BROTLI"
	i64 numPoints = 0;
	i64 hierarchy_firstChunkSize = 0;  // byte size of the root hierarchy chunk
	i64 hierarchy_stepSize = 0;        // levels per hierarchy chunk
	i64 hierarchy_depth = 0;
	dvec3 offset;
	dvec3 scale;
	double spacing = 0.0;
	dvec3 min;                // bounding box of the octree (cubic)
	dvec3 max;
	vector<PotreeAttribute> attributes;
	i64 bytesPerPoint = 0;    // sum of all attribute sizes

	// Octree nodes, hierarchyNodes[0] is the root. Grows as proxy nodes are expanded via loadHierarchyChunk().
	vector<PotreeHierarchyNode> hierarchyNodes;

	PotreeFileNode(string dir, string name) : SceneNode(name){
		// dir will be something like: /home/mschuetz/dev/resources/morro_bay_73M.laz_converted

		this->dir = dir;

		// Load metadata from metadata.json.
		string metadataPath = dir + "/metadata.json";
		if(!std::filesystem::exists(metadataPath)){
			println("ERROR: potree metadata not found: {}", metadataPath);
			exit(9234561);
		}

		using json = nlohmann::json;
		json j;
		try{
			std::ifstream stream(metadataPath);
			j = json::parse(stream);
		}catch(const json::exception& e){
			println("ERROR: failed to parse {}: {}", metadataPath, e.what());
			exit(9234562);
		}

		auto toDvec3 = [](const json& arr) -> dvec3 {
			return {arr[0].get<double>(), arr[1].get<double>(), arr[2].get<double>()};
		};

		version                  = j.value("version", "");
		metadataName             = j.value("name", "");
		description              = j.value("description", "");
		projection               = j.value("projection", "");
		encoding                 = j.value("encoding", "DEFAULT");
		numPoints                = j["points"].get<i64>();
		hierarchy_firstChunkSize = j["hierarchy"]["firstChunkSize"].get<i64>();
		hierarchy_stepSize       = j["hierarchy"]["stepSize"].get<i64>();
		hierarchy_depth          = j["hierarchy"]["depth"].get<i64>();
		offset                   = toDvec3(j["offset"]);
		scale                    = toDvec3(j["scale"]);
		spacing                  = j["spacing"].get<double>();
		min                      = toDvec3(j["boundingBox"]["min"]);
		max                      = toDvec3(j["boundingBox"]["max"]);

		if(version != "2.0"){
			println("WARNING: expected potree format version 2.0, got '{}' in {}", version, metadataPath);
		}

		// attributes are stored interleaved per point, in the order listed in metadata.json
		bytesPerPoint = 0;
		for(const json& jattribute : j["attributes"]){
			PotreeAttribute attribute;
			attribute.name        = jattribute.value("name", "");
			attribute.description = jattribute.value("description", "");
			attribute.type        = jattribute.value("type", "");
			attribute.size        = jattribute["size"].get<i64>();
			attribute.numElements = jattribute["numElements"].get<i64>();
			attribute.elementSize = jattribute["elementSize"].get<i64>();
			attribute.byteOffset  = bytesPerPoint;

			if(jattribute.contains("min"))       attribute.min       = jattribute["min"].get<vector<double>>();
			if(jattribute.contains("max"))       attribute.max       = jattribute["max"].get<vector<double>>();
			if(jattribute.contains("scale"))     attribute.scale     = jattribute["scale"].get<vector<double>>();
			if(jattribute.contains("offset"))    attribute.offset    = jattribute["offset"].get<vector<double>>();
			if(jattribute.contains("histogram")) attribute.histogram = jattribute["histogram"].get<vector<i64>>();

			bytesPerPoint += attribute.size;
			attributes.push_back(attribute);
		}

		// Points are rendered relative to the center of the octree's bounding box. 
		// The position attribute's min/max is the tight bounding box of the actual points.
		dvec3 center = (min + max) * 0.5;
		dvec3 tightMin = min;
		dvec3 tightMax = max;
		if(PotreeAttribute* position = findAttribute("position"); position && position->min.size() == 3 && position->max.size() == 3){
			tightMin = {position->min[0], position->min[1], position->min[2]};
			tightMax = {position->max[0], position->max[1], position->max[2]};
		}
		aabb.min = tightMin - center;
		aabb.max = tightMax - center;

		// memory-map hierarchy.bin and octree.bin
		mapped_hierarchy = mapFile(dir + "/hierarchy.bin", hierarchyFileSize);
		mapped_octree    = mapFile(dir + "/octree.bin", octreeFileSize);

		// root starts as a proxy that points to the first hierarchy chunk
		PotreeHierarchyNode root;
		root.type                = PotreeHierarchyNode::TYPE_PROXY;
		root.hierarchyByteOffset = 0;
		root.hierarchyByteSize   = hierarchy_firstChunkSize;
		root.min                 = min;
		root.max                 = max;
		hierarchyNodes.push_back(root);

		loadHierarchyChunk(0);

		if(encoding != "DEFAULT"){
			println("WARNING: {} uses {} encoding, point data can not be accessed directly from the mapped octree.bin.", dir, encoding);
		}
	}

	// owns the mappings
	PotreeFileNode(const PotreeFileNode&) = delete;
	PotreeFileNode& operator=(const PotreeFileNode&) = delete;

	~PotreeFileNode(){
		if(octreeCuFileHandle != nullptr){
			cuFileHandleDeregister(octreeCuFileHandle);
			octreeCuFileHandle = nullptr;
		}
		if(octreeDirectFd != -1){
			::close(octreeDirectFd);
			octreeDirectFd = -1;
		}
		if(mapped_hierarchy != nullptr){
			munmap(mapped_hierarchy, hierarchyFileSize);
			mapped_hierarchy = nullptr;
		}
		if(mapped_octree != nullptr){
			munmap(mapped_octree, octreeFileSize);
			mapped_octree = nullptr;
		}
	}

	// Opens octree.bin for reads via cuFile on first use. The cuFile driver must already be open.
	// Returns nullptr if the file can't be used with cuFile.
	CUfileHandle_t getOctreeCuFileHandle(){
		if(octreeCuFileHandle != nullptr) return octreeCuFileHandle;
		if(octreeCuFileFailed) return nullptr;

		string path = dir + "/octree.bin";

		// Direct transfers from the SSD to the GPU require O_DIRECT, i.e., bypassing the page cache. 
		// If the file system doesn't support it, cuFile can still read via its compatibility mode.
		octreeDirectFd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
		if(octreeDirectFd == -1){
			println("WARNING: failed to open {} with O_DIRECT ({}). Trying without.", path, strerror(errno));
			octreeDirectFd = ::open(path.c_str(), O_RDONLY);
		}
		if(octreeDirectFd == -1){
			println("ERROR: failed to open {} for direct storage reads: {}", path, strerror(errno));
			octreeCuFileFailed = true;
			return nullptr;
		}

		CUfileDescr_t descr = {};
		descr.handle.fd = octreeDirectFd;
		descr.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

		CUfileError_t status = cuFileHandleRegister(&octreeCuFileHandle, &descr);
		if(status.err != CU_FILE_SUCCESS){
			println("ERROR: cuFileHandleRegister failed for {} (error {})", path, int(status.err));
			::close(octreeDirectFd);
			octreeDirectFd = -1;
			octreeCuFileHandle = nullptr;
			octreeCuFileFailed = true;
			return nullptr;
		}

		return octreeCuFileHandle;
	}

	// Expands a proxy node by parsing its hierarchy chunk from the mapped hierarchy.bin. 
	// The chunk contains the node itself, followed by its descendants in breadth-first order. 
	// Descendants at the chunk boundary are again proxies. 
	// Note: Appends to hierarchyNodes, so references to its elements become invalid.
	void loadHierarchyChunk(i32 proxyIndex){
		if(hierarchyNodes[proxyIndex].type != PotreeHierarchyNode::TYPE_PROXY) return;

		constexpr i64 BYTES_PER_NODE = 22;

		u8* chunk = (u8*)mapped_hierarchy + hierarchyNodes[proxyIndex].hierarchyByteOffset;
		i64 numChunkNodes = hierarchyNodes[proxyIndex].hierarchyByteSize / BYTES_PER_NODE;

		vector<i32> chunkNodes = {proxyIndex};
		chunkNodes.reserve(numChunkNodes);

		for(i64 i = 0; i < numChunkNodes && i < chunkNodes.size(); i++){
			u8* record = chunk + i * BYTES_PER_NODE;

			u8 type        = record[0];
			u8 childMask   = record[1];
			u32 numPoints  = readAt<u32>(record + 2);
			u64 byteOffset = readAt<u64>(record + 6);
			u64 byteSize   = readAt<u64>(record + 14);

			i32 currentIndex = chunkNodes[i];
			PotreeHierarchyNode& current = hierarchyNodes[currentIndex];

			if(current.type == PotreeHierarchyNode::TYPE_PROXY){
				// the proxy we are expanding - replace with its actual data
				current.byteOffset = byteOffset;
				current.byteSize   = byteSize;
			}else if(type == PotreeHierarchyNode::TYPE_PROXY){
				// a proxy within this chunk - remember where its own chunk is
				current.hierarchyByteOffset = byteOffset;
				current.hierarchyByteSize   = byteSize;
			}else{
				current.byteOffset = byteOffset;
				current.byteSize   = byteSize;
			}

			current.type      = type;
			current.childMask = childMask;
			current.numPoints = numPoints;

			if(type == PotreeHierarchyNode::TYPE_PROXY) continue;

			for(int childIndex = 0; childIndex < 8; childIndex++){
				if((childMask & (1 << childIndex)) == 0) continue;

				PotreeHierarchyNode child;
				child.level = hierarchyNodes[currentIndex].level + 1;
				computeChildBox(hierarchyNodes[currentIndex].min, hierarchyNodes[currentIndex].max, childIndex, child.min, child.max);

				i32 newIndex = hierarchyNodes.size();
				hierarchyNodes.push_back(child);  // invalidates "current"
				hierarchyNodes[currentIndex].children[childIndex] = newIndex;
				chunkNodes.push_back(newIndex);
			}
		}
	}

	// childIndex bits: 0b100 -> upper half in x, 0b010 -> y, 0b001 -> z
	static void computeChildBox(dvec3 parentMin, dvec3 parentMax, int childIndex, dvec3& childMin, dvec3& childMax){
		dvec3 halfSize = (parentMax - parentMin) * 0.5;
		childMin = parentMin;
		childMax = parentMax;

		if(childIndex & 0b100) childMin.x += halfSize.x; else childMax.x -= halfSize.x;
		if(childIndex & 0b010) childMin.y += halfSize.y; else childMax.y -= halfSize.y;
		if(childIndex & 0b001) childMin.z += halfSize.z; else childMax.z -= halfSize.z;
	}

	// returns nullptr if there is no attribute with that name
	PotreeAttribute* findAttribute(string name){
		for(PotreeAttribute& attribute : attributes){
			if(attribute.name == name) return &attribute;
		}

		return nullptr;
	}

	uint64_t getGpuMemoryUsage(){
		return 0;
	}

private:

	template<typename T>
	static T readAt(const u8* ptr){
		T value;
		memcpy(&value, ptr, sizeof(T));

		return value;
	}

	static void* mapFile(string path, i64& fileSize){
		int fd = open(path.c_str(), O_RDONLY);
		if(fd == -1){
			println("ERROR: failed to open {}", path);
			exit(9234563);
		}

		struct stat st;
		if(fstat(fd, &st) != 0){
			println("ERROR: fstat failed for {}", path);
			close(fd);
			exit(9234564);
		}
		fileSize = st.st_size;

		// mmap does not support empty files
		if(fileSize == 0){
			close(fd);
			return nullptr;
		}

		void* mapped = mmap(nullptr, fileSize, PROT_READ, MAP_SHARED, fd, 0);
		close(fd); // the mapping stays valid after closing the file descriptor

		if(mapped == MAP_FAILED){
			println("ERROR: mmap failed for {}", path);
			exit(9234565);
		}

		return mapped;
	}

};
