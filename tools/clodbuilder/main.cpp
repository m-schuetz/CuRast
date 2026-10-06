// clodbuilder: converts a single-primitive GLB mesh (POSITION, TEXCOORD_0, indices, embedded base color texture)
// into a clustered LOD representation. The clusters, groups and LOD errors are computed by clodBuild() from
// meshoptimizer's demo/clusterlod.h; this program only loads the GLB and writes the results to flat binary files.
// The output format is described in README.md next to this file.
//
// Usage: clodbuilder <input.glb> <outputDir>

#include <cassert>
#include <cfloat>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <string>
#include <vector>

#include "meshoptimizer.h"

#define CLUSTERLOD_IMPLEMENTATION
#include "clusterlod.h"

#include "json/json.hpp"

using std::string;
using std::vector;
using std::println;
using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

// Records of clusters.bin, see README.md
struct Cluster{
	float    aabbMin[3];
	uint32_t vertexOffset;      // first vertex in positions.bin and uvs.bin
	float    aabbMax[3];
	uint32_t triangleOffset;    // first triangle in triangles.bin
	float    cullSphere[4];     // tight bounding sphere of this cluster's triangles. xyz: center, w: radius
	float    lodSphere[4];      // bounds of the group this cluster was simplified from
	float    parentSphere[4];   // bounds of the group this cluster belongs to
	float    lodError;          // 0 for clusters of the original mesh
	float    parentError;       // FLT_MAX if this cluster's group was not simplified any further
	uint32_t vertexCount;
	uint32_t triangleCount;
	uint32_t level;             // DAG depth of this cluster's group, 0 = original mesh
	uint32_t group;             // the group this cluster belongs to
	int32_t  refinedGroup;      // the group this cluster was simplified from, -1 for clusters of the original mesh
	uint32_t padding;
};
static_assert(sizeof(Cluster) == 112);

// Records of groups.bin, see README.md
struct Group{
	float    sphere[4];         // simplified bounds of the group. xyz: center, w: radius
	float    error;             // FLT_MAX for terminal groups
	uint32_t level;
	uint32_t clusterOffset;     // first cluster of this group in clusters.bin
	uint32_t clusterCount;
};
static_assert(sizeof(Group) == 32);

// clodNode is written as is to nodes.bin
static_assert(sizeof(clodNode) == 32);

struct InputMesh{
	vector<float> positions;    // xyz per vertex
	vector<float> uvs;          // uv per vertex
	vector<uint32_t> indices;
	vector<uint8_t> image;      // encoded base color texture, empty if there is none
	string imageMimeType;
};

template<typename... Args>
[[noreturn]] void fail(std::format_string<Args...> format, Args&&... args){
	println(stderr, "ERROR: {}", std::format(format, std::forward<Args>(args)...));
	exit(1);
}

static vector<uint8_t> readBinaryFile(string path){
	std::ifstream file(path, std::ios::binary);
	if(!file) fail("failed to open {}", path);

	vector<uint8_t> data(fs::file_size(path));
	file.read((char*)data.data(), data.size());
	if(!file) fail("failed to read {}", path);

	return data;
}

static void writeBinaryFile(fs::path path, const void* data, size_t size){
	std::ofstream file(path, std::ios::binary);
	file.write((const char*)data, size);
	if(!file) fail("failed to write {}", path.string());
}

template<typename T>
static void writeBinaryFile(fs::path path, const vector<T>& data){
	writeBinaryFile(path, data.data(), data.size() * sizeof(T));
}

// Location of an accessor's elements in the GLB's BIN chunk
struct AccessorView{
	const uint8_t* data;
	size_t stride;
	size_t count;
	int componentType;
};

static AccessorView getAccessor(const json& gltf, const uint8_t* bin, size_t binSize, int index, string expectedType){
	const json& accessor = gltf["accessors"][index];

	if(accessor.contains("sparse")) fail("accessor {}: sparse accessors are not supported", index);
	if(!accessor.contains("bufferView")) fail("accessor {} has no bufferView", index);
	if(accessor["type"] != expectedType) fail("accessor {} has type {}, expected {}", index, accessor["type"].get<string>(), expectedType);

	const json& view = gltf["bufferViews"][accessor["bufferView"].get<int>()];
	if(view.value("buffer", 0) != 0) fail("accessor {} does not refer to the GLB's BIN chunk", index);

	int componentType = accessor["componentType"];
	size_t componentSize = (componentType == 5126 || componentType == 5125) ? 4 : (componentType == 5123 || componentType == 5122) ? 2 : 1;
	size_t numComponents = expectedType == "VEC3" ? 3 : expectedType == "VEC2" ? 2 : 1;
	size_t elementSize = componentSize * numComponents;

	size_t count = accessor["count"];
	size_t viewOffset = view.value("byteOffset", size_t(0));
	size_t viewLength = view["byteLength"];
	size_t accessorOffset = accessor.value("byteOffset", size_t(0));
	size_t stride = view.value("byteStride", elementSize);

	if(viewOffset + viewLength > binSize) fail("bufferView of accessor {} exceeds the BIN chunk", index);
	if(count > 0 && accessorOffset + (count - 1) * stride + elementSize > viewLength) fail("accessor {} exceeds its bufferView", index);

	return {bin + viewOffset + accessorOffset, stride, count, componentType};
}

static vector<float> readFloats(const AccessorView& view, size_t numComponents){
	if(view.componentType != 5126) fail("expected a float accessor, got componentType {}", view.componentType);

	vector<float> values(view.count * numComponents);
	for(size_t i = 0; i < view.count; i++){
		memcpy(&values[i * numComponents], view.data + i * view.stride, numComponents * sizeof(float));
	}

	return values;
}

static vector<uint32_t> readIndices(const AccessorView& view){
	vector<uint32_t> values(view.count);

	for(size_t i = 0; i < view.count; i++){
		const uint8_t* source = view.data + i * view.stride;

		if(view.componentType == 5125){
			memcpy(&values[i], source, 4);
		}else if(view.componentType == 5123){
			uint16_t value;
			memcpy(&value, source, 2);
			values[i] = value;
		}else if(view.componentType == 5121){
			values[i] = source[0];
		}else{
			fail("unsupported index componentType {}", view.componentType);
		}
	}

	return values;
}

static InputMesh loadGlb(string path){
	vector<uint8_t> file = readBinaryFile(path);

	auto u32At = [&](size_t offset){
		uint32_t value;
		memcpy(&value, &file[offset], 4);
		return value;
	};

	// GLB: 12 byte header, followed by a JSON chunk and a BIN chunk, each starting with an 8 byte chunk header
	if(file.size() < 20 || u32At(0) != 0x46546C67) fail("not a GLB file: {}", path);
	if(u32At(4) != 2) fail("unsupported GLB version {}", u32At(4));
	if(u32At(16) != 0x4E4F534A) fail("first GLB chunk is not JSON");

	size_t jsonLength = u32At(12);
	size_t binHeader = 20 + jsonLength;
	if(binHeader + 8 > file.size()) fail("GLB has no BIN chunk");
	if(u32At(binHeader + 4) != 0x004E4942) fail("second GLB chunk is not BIN");

	size_t binLength = u32At(binHeader);
	if(binHeader + 8 + binLength > file.size()) fail("GLB BIN chunk exceeds the file size");
	const uint8_t* bin = &file[binHeader + 8];

	json gltf = json::parse(file.begin() + 20, file.begin() + 20 + jsonLength);

	// Only a single primitive without node transformations is supported
	if(gltf["meshes"].size() != 1 || gltf["meshes"][0]["primitives"].size() != 1){
		fail("expected exactly one mesh with exactly one primitive");
	}

	int meshNodes = 0;
	for(const json& node : gltf["nodes"]){
		if(node.contains("matrix") || node.contains("translation") || node.contains("rotation") || node.contains("scale")){
			fail("node transformations are not supported");
		}
		if(node.contains("mesh")) meshNodes++;
	}
	if(meshNodes != 1) fail("expected exactly one node that references the mesh, found {}", meshNodes);

	const json& primitive = gltf["meshes"][0]["primitives"][0];
	const json& attributes = primitive["attributes"];

	if(primitive.value("mode", 4) != 4) fail("only triangle lists are supported");
	if(!attributes.contains("POSITION") || !attributes.contains("TEXCOORD_0") || !primitive.contains("indices")){
		fail("expected a primitive with POSITION, TEXCOORD_0 and indices");
	}

	InputMesh mesh;
	mesh.positions = readFloats(getAccessor(gltf, bin, binLength, attributes["POSITION"], "VEC3"), 3);
	mesh.uvs = readFloats(getAccessor(gltf, bin, binLength, attributes["TEXCOORD_0"], "VEC2"), 2);
	mesh.indices = readIndices(getAccessor(gltf, bin, binLength, primitive["indices"], "SCALAR"));

	size_t numVertices = mesh.positions.size() / 3;
	if(mesh.uvs.size() / 2 != numVertices) fail("POSITION and TEXCOORD_0 have different counts");
	if(mesh.indices.size() % 3 != 0) fail("index count is not a multiple of 3");
	for(uint32_t index : mesh.indices){
		if(index >= numVertices) fail("index {} out of range", index);
	}

	// Base color texture, if the material has one
	if(primitive.contains("material")){
		const json& material = gltf["materials"][primitive["material"].get<int>()];

		if(material.contains("pbrMetallicRoughness") && material["pbrMetallicRoughness"].contains("baseColorTexture")){
			const json& textureInfo = material["pbrMetallicRoughness"]["baseColorTexture"];
			if(textureInfo.value("texCoord", 0) != 0) fail("base color texture does not use TEXCOORD_0");

			const json& texture = gltf["textures"][textureInfo["index"].get<int>()];
			const json& image = gltf["images"][texture["source"].get<int>()];
			if(!image.contains("bufferView")) fail("only images embedded in the GLB are supported");

			const json& view = gltf["bufferViews"][image["bufferView"].get<int>()];
			size_t offset = view.value("byteOffset", size_t(0));
			size_t length = view["byteLength"];
			if(offset + length > binLength) fail("image bufferView exceeds the BIN chunk");

			mesh.image.assign(bin + offset, bin + offset + length);
			mesh.imageMimeType = image.value("mimeType", "");
		}
	}

	return mesh;
}

static void setSphere(float* target, const clodBounds& bounds){
	target[0] = bounds.center[0];
	target[1] = bounds.center[1];
	target[2] = bounds.center[2];
	target[3] = bounds.radius;
}

static void run(string inputPath, fs::path outputDir){

	auto start = std::chrono::steady_clock::now();
	auto seconds = [&](){ return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };

	println("loading {}", inputPath);
	InputMesh input = loadGlb(inputPath);
	size_t numVertices = input.positions.size() / 3;
	size_t numInputTriangles = input.indices.size() / 3;
	println("    {} vertices, {} triangles ({:.1f}s)", numVertices, numInputTriangles, seconds());

	// Drop triangles that reference the same vertex more than once
	size_t numTriangles = 0;
	for(size_t i = 0; i < numInputTriangles; i++){
		uint32_t a = input.indices[3 * i + 0];
		uint32_t b = input.indices[3 * i + 1];
		uint32_t c = input.indices[3 * i + 2];
		if(a == b || b == c || c == a) continue;

		input.indices[3 * numTriangles + 0] = a;
		input.indices[3 * numTriangles + 1] = b;
		input.indices[3 * numTriangles + 2] = c;
		numTriangles++;
	}
	input.indices.resize(3 * numTriangles);
	size_t numDegenerate = numInputTriangles - numTriangles;
	println("    removed {} degenerate triangles", numDegenerate);

	float boxMin[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
	float boxMax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
	for(size_t i = 0; i < numVertices; i++){
		for(int j = 0; j < 3; j++){
			boxMin[j] = std::min(boxMin[j], input.positions[3 * i + j]);
			boxMax[j] = std::max(boxMax[j], input.positions[3 * i + j]);
		}
	}

	// meshoptimizer's default configuration for rasterization with 128 triangles per cluster
	clodConfig config = clodDefaultConfig(128);
	config.partition_sort = true;   // store spatially adjacent groups next to each other in the output files
	config.optimize_bounds = true;  // tight cullSphere for simplified clusters

	// UVs are not weighted during simplification, but UV seams are protected. Same setup as meshoptimizer's demo/nanite.cpp
	clodMesh mesh = {};
	mesh.indices = input.indices.data();
	mesh.index_count = input.indices.size();
	mesh.vertex_count = numVertices;
	mesh.vertex_positions = input.positions.data();
	mesh.vertex_positions_stride = 3 * sizeof(float);
	mesh.vertex_attributes = input.uvs.data();
	mesh.vertex_attributes_stride = 2 * sizeof(float);
	mesh.attribute_weights = nullptr;
	mesh.attribute_count = 0;
	mesh.attribute_protect_mask = (1 << 0) | (1 << 1);

	vector<clodGroup> groups;   // as received from clodBuild, for the LOD bounds of refined groups and for the hierarchy
	vector<Group> outGroups;
	vector<Cluster> outClusters;
	vector<float> outPositions;
	vector<float> outUvs;
	vector<uint8_t> outTriangles;

	vector<uint32_t> localVertices(config.max_triangles * 3);
	vector<uint8_t> localTriangles(config.max_triangles * 3);
	int currentLevel = -1;

	println("building clustered LOD");
	clodBuild(config, mesh, [&](clodGroup group, const clodCluster* clusters, size_t clusterCount) -> int {

		if(group.depth != currentLevel){
			currentLevel = group.depth;
			println("    level {} ({:.1f}s)", currentLevel, seconds());
		}

		uint32_t groupIndex = uint32_t(groups.size());

		Group outGroup = {};
		setSphere(outGroup.sphere, group.simplified);
		outGroup.error = group.simplified.error;
		outGroup.level = group.depth;
		outGroup.clusterOffset = uint32_t(outClusters.size());
		outGroup.clusterCount = uint32_t(clusterCount);
		outGroups.push_back(outGroup);

		for(size_t i = 0; i < clusterCount; i++){
			const clodCluster& cluster = clusters[i];

			size_t vertexCount = clodLocalIndices(localVertices.data(), localTriangles.data(), cluster.indices, cluster.index_count);
			if(vertexCount != cluster.vertex_count || vertexCount > 256) fail("unexpected cluster vertex count {}", vertexCount);

			Cluster outCluster = {};
			outCluster.vertexOffset = uint32_t(outPositions.size() / 3);
			outCluster.triangleOffset = uint32_t(outTriangles.size() / 3);
			outCluster.vertexCount = uint32_t(vertexCount);
			outCluster.triangleCount = uint32_t(cluster.index_count / 3);

			for(int j = 0; j < 3; j++){
				outCluster.aabbMin[j] = FLT_MAX;
				outCluster.aabbMax[j] = -FLT_MAX;
			}

			for(size_t j = 0; j < vertexCount; j++){
				const float* position = &input.positions[3 * localVertices[j]];
				const float* uv = &input.uvs[2 * localVertices[j]];

				outPositions.insert(outPositions.end(), position, position + 3);
				outUvs.insert(outUvs.end(), uv, uv + 2);

				for(int k = 0; k < 3; k++){
					outCluster.aabbMin[k] = std::min(outCluster.aabbMin[k], position[k]);
					outCluster.aabbMax[k] = std::max(outCluster.aabbMax[k], position[k]);
				}
			}

			outTriangles.insert(outTriangles.end(), localTriangles.begin(), localTriangles.begin() + cluster.index_count);

			setSphere(outCluster.cullSphere, cluster.bounds);

			if(cluster.refined < 0){
				setSphere(outCluster.lodSphere, cluster.bounds);
				outCluster.lodError = 0.0f;
			}else{
				setSphere(outCluster.lodSphere, groups[cluster.refined].simplified);
				outCluster.lodError = groups[cluster.refined].simplified.error;
			}

			setSphere(outCluster.parentSphere, group.simplified);
			outCluster.parentError = group.simplified.error;
			outCluster.level = uint32_t(group.depth);
			outCluster.group = groupIndex;
			outCluster.refinedGroup = cluster.refined;

			outClusters.push_back(outCluster);
		}

		groups.push_back(group);

		return int(groupIndex);
	});

	// Spatial hierarchy over the groups: a forest with one tree per level, roots at nodes[0, numLevels)
	size_t numLevels = 0;
	for(const clodGroup& group : groups){
		numLevels = std::max(numLevels, size_t(group.depth) + 1);
	}

	size_t nodeWidth = 8;
	vector<clodNode> nodes(clodBuildHierarchyBound(groups.size(), nodeWidth, numLevels));
	nodes.resize(clodBuildHierarchy(nodes.data(), groups.data(), groups.size(), nodeWidth, numLevels));

	println("    {} clusters, {} groups, {} nodes, {} vertices, {} triangles ({:.1f}s)",
		outClusters.size(), outGroups.size(), nodes.size(), outPositions.size() / 3, outTriangles.size() / 3, seconds());

	// Per-level statistics
	struct LevelStats{
		size_t groups = 0;
		size_t clusters = 0;
		size_t triangles = 0;
		size_t vertices = 0;
		size_t terminalGroups = 0;
		size_t terminalTriangles = 0;
		float maxLodError = 0.0f;
	};

	vector<LevelStats> levelStats(numLevels);
	for(const Group& group : outGroups){
		LevelStats& stats = levelStats[group.level];
		stats.groups++;
		stats.clusters += group.clusterCount;

		if(group.error == FLT_MAX) stats.terminalGroups++;

		for(uint32_t i = group.clusterOffset; i < group.clusterOffset + group.clusterCount; i++){
			const Cluster& cluster = outClusters[i];
			stats.triangles += cluster.triangleCount;
			stats.vertices += cluster.vertexCount;
			stats.maxLodError = std::max(stats.maxLodError, cluster.lodError);
			if(group.error == FLT_MAX) stats.terminalTriangles += cluster.triangleCount;
		}
	}

	println("    level   groups  clusters   triangles  terminal triangles  max lodError");
	for(size_t i = 0; i < numLevels; i++){
		const LevelStats& stats = levelStats[i];
		println("    {:5} {:8} {:9} {:11} {:19} {:13.6f}", i, stats.groups, stats.clusters, stats.triangles, stats.terminalTriangles, stats.maxLodError);
	}

	// Write results
	println("writing to {}", outputDir.string());
	fs::create_directories(outputDir);

	writeBinaryFile(outputDir / "clusters.bin", outClusters);
	writeBinaryFile(outputDir / "groups.bin", outGroups);
	writeBinaryFile(outputDir / "nodes.bin", nodes);
	writeBinaryFile(outputDir / "positions.bin", outPositions);
	writeBinaryFile(outputDir / "uvs.bin", outUvs);
	writeBinaryFile(outputDir / "triangles.bin", outTriangles);

	string textureFile = "";
	if(!input.image.empty()){
		textureFile = input.imageMimeType == "image/png" ? "texture.png" : "texture.jpg";
		writeBinaryFile(outputDir / textureFile, input.image);
	}

	json metadata;
	metadata["format"] = "clustered LOD, see tools/clodbuilder/README.md";
	metadata["version"] = 1;
	metadata["source"] = inputPath;
	metadata["generator"] = std::format("tools/clodbuilder with meshoptimizer {}.{} clusterlod.h", MESHOPTIMIZER_VERSION / 1000, (MESHOPTIMIZER_VERSION % 1000) / 10);

	metadata["boundingBox"] = {
		{"min", {boxMin[0], boxMin[1], boxMin[2]}},
		{"max", {boxMax[0], boxMax[1], boxMax[2]}},
	};

	metadata["input"] = {
		{"vertices", numVertices},
		{"triangles", numInputTriangles},
		{"degenerateTrianglesRemoved", numDegenerate},
	};

	metadata["config"] = {
		{"max_vertices", config.max_vertices},
		{"min_triangles", config.min_triangles},
		{"max_triangles", config.max_triangles},
		{"partition_spatial", config.partition_spatial},
		{"partition_sort", config.partition_sort},
		{"partition_size", config.partition_size},
		{"cluster_spatial", config.cluster_spatial},
		{"cluster_split_factor", config.cluster_split_factor},
		{"simplify_ratio", config.simplify_ratio},
		{"simplify_threshold", config.simplify_threshold},
		{"simplify_error_merge_previous", config.simplify_error_merge_previous},
		{"simplify_error_merge_additive", config.simplify_error_merge_additive},
		{"simplify_error_factor_sloppy", config.simplify_error_factor_sloppy},
		{"simplify_error_clamped", config.simplify_error_clamped},
		{"simplify_permissive", config.simplify_permissive},
		{"simplify_fallback_permissive", config.simplify_fallback_permissive},
		{"simplify_fallback_sloppy", config.simplify_fallback_sloppy},
		{"optimize_bounds", config.optimize_bounds},
		{"optimize_clusters", config.optimize_clusters},
		{"attribute_protect_mask", mesh.attribute_protect_mask},
		{"hierarchy_node_width", nodeWidth},
	};

	metadata["counts"] = {
		{"levels", numLevels},
		{"clusters", outClusters.size()},
		{"groups", outGroups.size()},
		{"nodes", nodes.size()},
		{"vertices", outPositions.size() / 3},
		{"triangles", outTriangles.size() / 3},
	};

	metadata["levels"] = json::array();
	for(size_t i = 0; i < numLevels; i++){
		const LevelStats& stats = levelStats[i];
		metadata["levels"].push_back({
			{"level", i},
			{"groups", stats.groups},
			{"clusters", stats.clusters},
			{"triangles", stats.triangles},
			{"vertices", stats.vertices},
			{"terminalGroups", stats.terminalGroups},
			{"terminalTriangles", stats.terminalTriangles},
			{"maxLodError", stats.maxLodError},
		});
	}

	metadata["files"] = {
		{"clusters",  {{"file", "clusters.bin"},  {"count", outClusters.size()},       {"stride", sizeof(Cluster)}}},
		{"groups",    {{"file", "groups.bin"},    {"count", outGroups.size()},         {"stride", sizeof(Group)}}},
		{"nodes",     {{"file", "nodes.bin"},     {"count", nodes.size()},             {"stride", sizeof(clodNode)}}},
		{"positions", {{"file", "positions.bin"}, {"count", outPositions.size() / 3}, {"stride", 12}, {"type", "float32 x3"}}},
		{"uvs",       {{"file", "uvs.bin"},       {"count", outUvs.size() / 2},       {"stride", 8},  {"type", "float32 x2"}}},
		{"triangles", {{"file", "triangles.bin"}, {"count", outTriangles.size() / 3}, {"stride", 3},  {"type", "uint8 x3, cluster-local vertex indices"}}},
	};

	if(!textureFile.empty()){
		metadata["texture"] = {
			{"file", textureFile},
			{"mimeType", input.imageMimeType},
		};
	}

	std::ofstream(outputDir / "metadata.json") << metadata.dump(1, '\t') << "\n";

	println("done ({:.1f}s)", seconds());
}

int main(int argc, char** argv){

	if(argc != 3){
		println("usage: clodbuilder <input.glb> <outputDir>");
		return 1;
	}

	try{
		run(argv[1], argv[2]);
	}catch(const std::exception& e){
		fail("{}", e.what());
	}

	return 0;
}
