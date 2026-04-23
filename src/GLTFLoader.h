#pragma once

#include "json/json.hpp"
#include <thread>
#include <mutex>
#include <stacktrace>
#include <unordered_set>

#include "MappedFile.h"



using json = nlohmann::json;
using namespace std;

namespace gltfloader{

struct Node{
	string name = "undefined";
	mat4 matrix;
	int mesh = -1;
	vector<int> children;

	// One scene node per mesh primitive
	// vector<shared_ptr<SNTriangles>> sceneNodes;
};

static inline mutex mtx_primitive;
struct Primitive{
	unordered_map<string, int> attributes;
	int indices = -1;
	int mode = -1;
	int material = -1;

	shared_ptr<SNTriangles> prepper = nullptr; // The SNTriangles instance that prepped the data and owns the GPU buffers
};

static inline mutex mtx_mesh;
struct Mesh{
	string name = "undefined";
	vector<Primitive> primitives;
};

struct Material{
	int texture = -1;
	string name = "";
};

struct Texture{
	int sampler = -1;
	string name = "";
	int source = -1;
};

struct Image{
	int bufferView = -1;
	string mimeType = "";
	string name = "";
};

constexpr int COMPONENT_TYPE_UINT8 = 5121;
constexpr int COMPONENT_TYPE_UINT16 = 5123;
constexpr int COMPONENT_TYPE_UINT32 = 5125;
constexpr int COMPONENT_TYPE_FLOAT = 5126;

// enum class ComponentType{
// 	UINT8 = 5121,
// 	UINT16 = 5123,
// 	UINT32 = 5125,
// 	FLOAT = 5126,
// };

int64_t componentTypeByteSize(int componentType){
	if(componentType == COMPONENT_TYPE_UINT8) return 1;
	if(componentType == COMPONENT_TYPE_UINT16) return 2;
	if(componentType == COMPONENT_TYPE_UINT32) return 4;
	if(componentType == COMPONENT_TYPE_FLOAT) return 4;

	println("ERROR: unsupported component type: {}", componentType);
	println("{}", stacktrace::current());
	exit(12035756);
}

int64_t typeCount(string type){
	if(type== "SCALAR") return 1;
	if(type== "VEC2")   return 2;
	if(type== "VEC3")   return 3;
	if(type== "VEC4")   return 4;
	if(type== "MAT2")   return 2 * 2;
	if(type== "MAT3")   return 3 * 3;
	if(type== "MAT4")   return 4 * 4;

	println("ERROR: unsupported type: {}", type);
	println("{}", stacktrace::current());
	exit(12035756);
}

// spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-accessor
// 
// from https://chatgpt.com/c/694431ec-9008-8326-9915-20157fb7ea17
// componentType    Data Type             Signed                       Bits
// 5120             signed byte           Signed, two’s complement        8
// 5121             unsigned byte         Unsigned                        8
// 5122             signed short          Signed, two’s complement       16
// 5123             unsigned short        Unsigned                       16
// 5125             unsigned int          Unsigned                       32
// 5126             float                 Signed                         32
// ============================================================================
// | Type   | Number of Components |
// | ------ | -------------------- |
// | SCALAR | 1                    |
// | VEC2   | 2                    |
// | VEC3   | 3                    |
// | VEC4   | 4                    |
// | MAT2   | 4                    |
// | MAT3   | 9                    |
// | MAT4   | 16                   |
struct Accessor{
	int bufferView      = -1;
	int64_t byteOffset  = 0;
	int64_t count       = -1;
	int componentType   = -1;
	string type         = "undefined";
	vec3 min            = {Infinity, Infinity, Infinity};
	vec3 max            = {-Infinity, -Infinity, -Infinity};

	int64_t getPackedStride() const {
		int64_t componentSize = componentTypeByteSize(componentType);
		return componentSize * typeCount(type);
	}

	int64_t getByteSize() const {
		return count * getPackedStride();
	}
};

struct BufferView{
	int buffer          = -1;
	int64_t byteOffset  = 0;
	int64_t byteStride  = -1;
	int64_t byteLength  = -1;
	int target          = -1;
};

struct Buffer{
	int64_t byteLength = -1;
	int64_t offset = 0;
	string uri = "undefined";
	// string canonicalPath = "undefined";
	shared_ptr<Mapping::MappedFile> file = nullptr;
	void* data = nullptr;
};

struct GLTF{

	vector<Node> nodes;
	vector<Mesh> meshes;
	vector<Accessor> accessors;
	vector<BufferView> bufferViews;
	vector<Buffer> buffers;
	vector<Image> images;
	vector<Texture> textures;
	vector<Material> materials;

	int64_t sectorSize = 0;
};



GLTF loadMetadata(string path){

	println("loading gltf metadata from '{}'", path);
	
	string strJson = "";
	if(iEndsWith(path, ".gltf")){
		strJson = readFile(path);
	}else if(iEndsWith(path, ".glb")){
		auto header = readBinaryFile(path, 0, 20);
		uint32_t jsonLength = header->get<uint32_t>(12);
		auto jsonBuffer = readBinaryFile(path, 20, jsonLength);
		strJson = string(jsonBuffer->data_char, jsonBuffer->size);
	}

	json j = json::parse(strJson);

	{ // DEBUG
		std::ofstream out("./dbg.glb.json");
		out << j.dump(4);
	}

	if (j.contains("asset") && j["asset"].contains("generator")) {
		string generator = j["asset"]["generator"];
		println("generator: {}", generator);
	}
	

	GLTF gltf;
	gltf.sectorSize = getPhysicalSectorSize(path);

	// LOAD IMAGES
	for(json jimage : j["images"]){
		
		Image image;
		if(jimage.contains("bufferView")) image.bufferView = jimage["bufferView"];
		if(jimage.contains("mimeType")) image.mimeType = jimage["mimeType"];
		if(jimage.contains("name")) image.name = jimage["name"];
		
		gltf.images.push_back(image);
	}

	// LOAD TEXTURES
	for(json jtexture : j["textures"]){
		
		Texture texture;
		if(jtexture.contains("sampler")) texture.sampler = jtexture["sampler"];
		if(jtexture.contains("name")) texture.name = jtexture["name"];
		if(jtexture.contains("source")) texture.source = jtexture["source"];
		
		gltf.textures.push_back(texture);
	}

	// LOAD MATERIALS
	for(json jmaterial : j["materials"]){
		
		Material material;
		if(jmaterial.contains("name")) material.name = jmaterial["name"];

		if(jmaterial.contains("pbrMetallicRoughness")){
			if(jmaterial["pbrMetallicRoughness"].contains("baseColorTexture")){
				json jbaseColorTexture = jmaterial["pbrMetallicRoughness"]["baseColorTexture"];
				material.texture = jbaseColorTexture["index"];
			}
		}
		
		gltf.materials.push_back(material);
	}

	// LOAD MESHES
	for(json jmesh : j["meshes"]){
		Mesh mesh;

		if(jmesh.contains("name")){
			mesh.name = jmesh["name"];
		}else{
			mesh.name = format("mesh_{}", gltf.meshes.size());
		}
		
		for(json jprimitive : jmesh["primitives"]){
			Primitive primitive;
			if(jprimitive.contains("indices")){
				primitive.indices = jprimitive["indices"];
			}
			if(jprimitive.contains("mode")){
				primitive.mode = jprimitive["mode"];
			}

			if(jprimitive.contains("material")) {
				primitive.material = jprimitive["material"];
			}

			for(auto& [key, value] : jprimitive["attributes"].items()){
				primitive.attributes[key] = int(value);
			}

			mesh.primitives.push_back(primitive);
		}

		gltf.meshes.push_back(mesh);
	}

	// LOAD NODES
	float matrixBuffer[16];
	for(json jnode : j["nodes"]){

		Node node;
		if(jnode.contains("name")){
			node.name = jnode["name"];
		} else {
			node.name = "undefined";
		}

		if (jnode.contains("mesh")) {
			node.mesh = jnode["mesh"];
		}

		node.matrix = mat4(1.0f);
		
		if (jnode.contains("matrix")) {
			for(int i = 0; i < 16; i++){
				matrixBuffer[i] = jnode["matrix"][i];
			}
			memcpy(&node.matrix, matrixBuffer, 16 * 4);
		}

		if(jnode.contains("children")){
			node.children = jnode["children"].get<std::vector<int>>();
		}

		gltf.nodes.push_back(node);
	}

	// LOAD ACCESSORS
	for(json jaccessor : j["accessors"]){
		Accessor accessor;

		if(jaccessor.contains("bufferView")) accessor.bufferView = jaccessor["bufferView"];
		if(jaccessor.contains("byteOffset")) accessor.byteOffset = jaccessor["byteOffset"];
		if(jaccessor.contains("componentType")) accessor.componentType = jaccessor["componentType"];
		if(jaccessor.contains("count")) accessor.count = jaccessor["count"];
		if(jaccessor.contains("type")) accessor.type = jaccessor["type"];

		if(accessor.type == "VEC3"){
			if(jaccessor.contains("min")){
				accessor.min.x = jaccessor["min"][0];
				accessor.min.y = jaccessor["min"][1];
				accessor.min.z = jaccessor["min"][2];
			}
			if(jaccessor.contains("max")){
				accessor.max.x = jaccessor["max"][0];
				accessor.max.y = jaccessor["max"][1];
				accessor.max.z = jaccessor["max"][2];
			}
		}

		gltf.accessors.push_back(accessor);
	}

	// LOAD BUFFER VIEWS
	for(json jview : j["bufferViews"]){
		BufferView view;

		if(jview.contains("buffer"))     view.buffer        = jview["buffer"];
		if(jview.contains("byteOffset")) view.byteOffset    = jview["byteOffset"];
		if(jview.contains("byteStride")) view.byteStride    = jview["byteStride"];
		if(jview.contains("byteLength")) view.byteLength    = jview["byteLength"];
		if(jview.contains("target"))     view.target        = jview["target"];

		gltf.bufferViews.push_back(view);
	}

	// LOAD BUFFERS
	auto mapped = Mapping::mapFile(path);
	uint32_t jsonChunkLength = read<uint32_t>(mapped->data, 12);
	uint32_t currentBufferOffset = 12 + 8 + jsonChunkLength;
	for(json jbuffer : j["buffers"]){
		Buffer buffer;

		if(jbuffer.contains("byteLength")) buffer.byteLength  = jbuffer["byteLength"];
		if(jbuffer.contains("uri"))        buffer.uri         = jbuffer["uri"];

		if(buffer.uri != "undefined"){
			string bufferPath = format("{}/../{}", path, buffer.uri);
			string canonicalPath = fs::canonical(fs::absolute(bufferPath)).string();
			buffer.uri = bufferPath;
			buffer.file = Mapping::mapFile(bufferPath);
			buffer.data = buffer.file->data;
		}else{
			int64_t chunkLength = read<uint32_t>(mapped->data, currentBufferOffset);

			if(chunkLength != buffer.byteLength){
				println("Missmatch between chunkLength and byteLength. {} vs {}", chunkLength, buffer.byteLength);
			}

			buffer.uri = path;
			buffer.file = Mapping::mapFile(path);
			buffer.offset = 8llu + currentBufferOffset;
			buffer.data = ((uint8_t*)buffer.file->data) + 8llu + currentBufferOffset;
			currentBufferOffset += 8 + chunkLength;
		}

		gltf.buffers.push_back(buffer);
	}

	// // Print Nodes
	// println("");
	// println("# Nodes ({}) ", gltf.nodes.size());
	// for(int i = 0; i < gltf.nodes.size(); i++){
	// 	Node& node = gltf.nodes[i];
				
	// 	println("{}, {}, matrix: {}, {}, {}, {}, ...", 
	// 		node.name, node.mesh,
	// 		node.matrix[0][0], node.matrix[0][1], node.matrix[0][2], node.matrix[0][3]
	// 		);

	// 	if(i > 10) break;
	// }

	// // Print Meshes
	// int numPrimitives = 0;
	// for(Mesh& mesh : gltf.meshes){
	// 	numPrimitives += mesh.primitives.size();
	// }
	// println("");
	// println("# Meshes ({}) ", gltf.meshes.size());
	// println("# Primitives ({}) ", numPrimitives);
	// for(int i = 0; i < gltf.meshes.size(); i++){
	// 	Mesh& mesh = gltf.meshes[i];
		
	// 	string strMesh = format("{}\n", mesh.name);
	// 	for(Primitive primitive : mesh.primitives){
	// 		for(auto [key, value] : primitive.attributes){
	// 			strMesh += format("    [{:12} : {}]\n", key, value);
	// 		}
	// 	}
		
	// 	println("{}", strMesh);

	// 	if(i > 10) break;
	// }

	// // Print Accessors
	// println("");
	// println("# Accessors ({}) ", gltf.accessors.size());
	// for(int i = 0; i < gltf.accessors.size(); i++){
	// 	Accessor& accessor = gltf.accessors[i];
				
	// 	println("bufferView: {}, byteOffset: {}, componentType: {}, count: {}, type: {}", 
	// 		accessor.bufferView, accessor.byteOffset, accessor.componentType, accessor.count, accessor.type
	// 	);

	// 	if(i > 10) break;
	// }

	// // Print Buffer Views
	// println("");
	// println("# Buffer Views ({}) ", gltf.bufferViews.size());
	// for(int i = 0; i < gltf.bufferViews.size(); i++){
	// 	BufferView& view = gltf.bufferViews[i];
				
	// 	println("buffer: {}, byteOffset: {}, byteLength: {}, byteStride: {}, target: {}", 
	// 		view.buffer, view.byteOffset, view.byteLength, view.byteStride, view.target
	// 	);

	// 	if(i > 10) break;
	// }

	// { // Print largest buffer views
	// 	vector<gltfloader::BufferView> views = gltf.bufferViews;
	// 	sort(views.begin(), views.end(), [](auto& a, auto& b){
	// 		return a.byteLength > b.byteLength;
	// 	});

	// 	std::locale::global(getSaneLocale());

	// 	println("");
	// 	println("# Largest Buffer Views ({:L}) ", gltf.bufferViews.size());
	// 	for(int i = 0; i < views.size(); i++){
	// 		BufferView& view = views[i];
					
	// 		println("buffer: {}, byteOffset: {}, byteLength: {}, byteStride: {}, target: {}", 
	// 			view.buffer, view.byteOffset, view.byteLength, view.byteStride, view.target
	// 		);

	// 		if(i > 10) break;
	// 	}
	// }

	// // Print Buffers
	// println("");
	// println("Buffers: ");
	// for(int i = 0; i < gltf.buffers.size(); i++){
	// 	Buffer& buffer = gltf.buffers[i];
				
	// 	println("byteLength: {}, uri: {}", buffer.byteLength, buffer.uri);

	// 	if(i > 10) break;
	// }

	return gltf;
}

GLTF filter(GLTF source, std::function<bool(Mesh)> predicate){

	GLTF result;
	result.sectorSize = source.sectorSize;
	result.buffers    = source.buffers;

	// Step 1: Filter meshes by predicate
	unordered_map<int, int> meshRemap;
	unordered_set<int> usedAccessorIndices;
	unordered_set<int> usedMaterialIndices;

	for(int oldIdx = 0; oldIdx < (int)source.meshes.size(); oldIdx++){
		if(!predicate(source.meshes[oldIdx])) continue;

		meshRemap[oldIdx] = (int)result.meshes.size();
		Mesh mesh = source.meshes[oldIdx];
		for(Primitive& prim : mesh.primitives){
			if(prim.indices  >= 0) usedAccessorIndices.insert(prim.indices);
			if(prim.material >= 0) usedMaterialIndices.insert(prim.material);
			for(auto& [key, val] : prim.attributes)
				usedAccessorIndices.insert(val);
		}
		result.meshes.push_back(mesh);
	}

	// Step 2: Keep only nodes that reference a kept mesh
	unordered_map<int, int> nodeRemap;
	for(int oldIdx = 0; oldIdx < (int)source.nodes.size(); oldIdx++){
		Node node = source.nodes[oldIdx];
		if(node.mesh < 0 || !meshRemap.count(node.mesh)) continue;
		nodeRemap[oldIdx] = (int)result.nodes.size();
		node.mesh = meshRemap[node.mesh];
		result.nodes.push_back(node);
	}

	// Step 3: Filter accessors; collect used bufferView indices
	unordered_set<int> usedBufferViewIndices;
	unordered_map<int, int> accessorRemap;
	for(int oldIdx = 0; oldIdx < (int)source.accessors.size(); oldIdx++){
		if(!usedAccessorIndices.count(oldIdx)) continue;
		accessorRemap[oldIdx] = (int)result.accessors.size();
		Accessor acc = source.accessors[oldIdx];
		if(acc.bufferView >= 0) usedBufferViewIndices.insert(acc.bufferView);
		result.accessors.push_back(acc);
	}

	// Step 4: Filter materials; collect used texture indices
	unordered_set<int> usedTextureIndices;
	unordered_map<int, int> materialRemap;
	for(int oldIdx = 0; oldIdx < (int)source.materials.size(); oldIdx++){
		if(!usedMaterialIndices.count(oldIdx)) continue;
		materialRemap[oldIdx] = (int)result.materials.size();
		Material mat = source.materials[oldIdx];
		if(mat.texture >= 0) usedTextureIndices.insert(mat.texture);
		result.materials.push_back(mat);
	}

	// Step 5: Filter textures; collect used image indices
	unordered_set<int> usedImageIndices;
	unordered_map<int, int> textureRemap;
	for(int oldIdx = 0; oldIdx < (int)source.textures.size(); oldIdx++){
		if(!usedTextureIndices.count(oldIdx)) continue;
		textureRemap[oldIdx] = (int)result.textures.size();
		Texture tex = source.textures[oldIdx];
		if(tex.source >= 0) usedImageIndices.insert(tex.source);
		// Also track bufferView of the image (via image.bufferView)
		result.textures.push_back(tex);
	}

	// Step 6: Filter images; collect bufferViews used by images
	unordered_map<int, int> imageRemap;
	for(int oldIdx = 0; oldIdx < (int)source.images.size(); oldIdx++){
		if(!usedImageIndices.count(oldIdx)) continue;
		imageRemap[oldIdx] = (int)result.images.size();
		Image img = source.images[oldIdx];
		if(img.bufferView >= 0) usedBufferViewIndices.insert(img.bufferView);
		result.images.push_back(img);
	}

	// Step 7: Filter bufferViews
	unordered_map<int, int> bufferViewRemap;
	for(int oldIdx = 0; oldIdx < (int)source.bufferViews.size(); oldIdx++){
		if(!usedBufferViewIndices.count(oldIdx)) continue;
		bufferViewRemap[oldIdx] = (int)result.bufferViews.size();
		result.bufferViews.push_back(source.bufferViews[oldIdx]);
	}

	// Patch mesh primitive references
	for(Mesh& mesh : result.meshes){
		for(Primitive& prim : mesh.primitives){
			if(prim.indices  >= 0) prim.indices  = accessorRemap.count(prim.indices)  ? accessorRemap[prim.indices]  : -1;
			if(prim.material >= 0) prim.material = materialRemap.count(prim.material) ? materialRemap[prim.material] : -1;
			for(auto& [key, val] : prim.attributes)
				val = accessorRemap.count(val) ? accessorRemap[val] : -1;
		}
	}

	// Patch accessor bufferView references
	for(Accessor& acc : result.accessors){
		if(acc.bufferView >= 0) acc.bufferView = bufferViewRemap.count(acc.bufferView) ? bufferViewRemap[acc.bufferView] : -1;
	}

	// Patch material texture references
	for(Material& mat : result.materials){
		if(mat.texture >= 0) mat.texture = textureRemap.count(mat.texture) ? textureRemap[mat.texture] : -1;
	}

	// Patch texture image references
	for(Texture& tex : result.textures){
		if(tex.source >= 0) tex.source = imageRemap.count(tex.source) ? imageRemap[tex.source] : -1;
	}

	// Patch image bufferView references
	for(Image& img : result.images){
		if(img.bufferView >= 0) img.bufferView = bufferViewRemap.count(img.bufferView) ? bufferViewRemap[img.bufferView] : -1;
	}

	return result;
}

GLTF filter(GLTF source, int maxNumberNodes){

	GLTF result;
	result.sectorSize = source.sectorSize;
	result.buffers    = source.buffers;
	result.bufferViews = source.bufferViews;

	// Step 1: Determine which nodes to keep
	int nodeCount = min((int)source.nodes.size(), maxNumberNodes);

	// Step 2: Collect used mesh indices from kept nodes
	unordered_set<int> usedMeshIndices;
	for(int i = 0; i < nodeCount; i++){
		if(source.nodes[i].mesh >= 0)
			usedMeshIndices.insert(source.nodes[i].mesh);
	}

	// Step 3: Filter meshes; collect used accessor and material indices
	unordered_map<int, int> meshRemap;
	unordered_set<int> usedAccessorIndices;
	unordered_set<int> usedMaterialIndices;

	for(int oldIdx = 0; oldIdx < (int)source.meshes.size(); oldIdx++){
		if(!usedMeshIndices.count(oldIdx)) continue;

		meshRemap[oldIdx] = (int)result.meshes.size();
		Mesh mesh = source.meshes[oldIdx];
		for(Primitive& prim : mesh.primitives){
			if(prim.indices  >= 0) usedAccessorIndices.insert(prim.indices);
			if(prim.material >= 0) usedMaterialIndices.insert(prim.material);
			for(auto& [key, val] : prim.attributes)
				usedAccessorIndices.insert(val);
		}
		result.meshes.push_back(mesh);
	}

	// Step 4: Filter accessors
	unordered_map<int, int> accessorRemap;
	for(int oldIdx = 0; oldIdx < (int)source.accessors.size(); oldIdx++){
		if(!usedAccessorIndices.count(oldIdx)) continue;
		accessorRemap[oldIdx] = (int)result.accessors.size();
		result.accessors.push_back(source.accessors[oldIdx]);
	}

	// Step 5: Filter materials; collect used texture indices
	unordered_set<int> usedTextureIndices;
	unordered_map<int, int> materialRemap;
	for(int oldIdx = 0; oldIdx < (int)source.materials.size(); oldIdx++){
		if(!usedMaterialIndices.count(oldIdx)) continue;
		materialRemap[oldIdx] = (int)result.materials.size();
		Material mat = source.materials[oldIdx];
		if(mat.texture >= 0) usedTextureIndices.insert(mat.texture);
		result.materials.push_back(mat);
	}

	// Step 6: Filter textures; collect used image indices
	unordered_set<int> usedImageIndices;
	unordered_map<int, int> textureRemap;
	for(int oldIdx = 0; oldIdx < (int)source.textures.size(); oldIdx++){
		if(!usedTextureIndices.count(oldIdx)) continue;
		textureRemap[oldIdx] = (int)result.textures.size();
		Texture tex = source.textures[oldIdx];
		if(tex.source >= 0) usedImageIndices.insert(tex.source);
		result.textures.push_back(tex);
	}

	// Step 7: Filter images
	unordered_map<int, int> imageRemap;
	for(int oldIdx = 0; oldIdx < (int)source.images.size(); oldIdx++){
		if(!usedImageIndices.count(oldIdx)) continue;
		imageRemap[oldIdx] = (int)result.images.size();
		result.images.push_back(source.images[oldIdx]);
	}

	// Patch nodes
	for(int i = 0; i < nodeCount; i++){
		Node node = source.nodes[i];
		node.mesh = (node.mesh >= 0 && meshRemap.count(node.mesh)) ? meshRemap[node.mesh] : -1;
		result.nodes.push_back(node);
	}

	// Patch mesh primitive references
	for(Mesh& mesh : result.meshes){
		for(Primitive& prim : mesh.primitives){
			if(prim.indices  >= 0) prim.indices  = accessorRemap.count(prim.indices)  ? accessorRemap[prim.indices]  : -1;
			if(prim.material >= 0) prim.material = materialRemap.count(prim.material) ? materialRemap[prim.material] : -1;
			for(auto& [key, val] : prim.attributes)
				val = accessorRemap.count(val) ? accessorRemap[val] : -1;
		}
	}

	// Patch material texture references
	for(Material& mat : result.materials){
		if(mat.texture >= 0) mat.texture = textureRemap.count(mat.texture) ? textureRemap[mat.texture] : -1;
	}

	// Patch texture image references
	for(Texture& tex : result.textures){
		if(tex.source >= 0) tex.source = imageRemap.count(tex.source) ? imageRemap[tex.source] : -1;
	}

	return result;
}


}