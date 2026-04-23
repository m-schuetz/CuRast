
#include <string>
#include <thread>
#include <queue>
#include <stacktrace>
#include <set>
#include <filesystem>
#include <print>
#include <format>
#include <algorithm>
#include <execution>
#include <memory>

#include "./glm/glm/glm.hpp"
#include "./MappedFile.h"
 #include "unsuck.hpp"
 #include "json/json.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

using namespace std;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat4;

constexpr float Infinity = __builtin_bit_cast(float, 0x7f800000);

struct Box3 {
	vec3 min = { Infinity, Infinity, Infinity };
	vec3 max = { -Infinity, -Infinity, -Infinity };

	bool isDefault() {
		return min.x == Infinity && min.y == Infinity && min.z == Infinity && max.x == -Infinity && max.y == -Infinity && max.z == -Infinity;
	}

	bool isEqual(Box3 box, float epsilon) {
		float diff_min = length(box.min - min);
		float diff_max = length(box.max - max);

		if (diff_min >= epsilon) return false;
		if (diff_max >= epsilon) return false;

		return true;
	}

	void extend(vec3 v){
		this->min.x = ::min(this->min.x, v.x);
		this->min.y = ::min(this->min.y, v.y);
		this->min.z = ::min(this->min.z, v.z);
		this->max.x = ::max(this->max.x, v.x);
		this->max.y = ::max(this->max.y, v.y);
		this->max.z = ::max(this->max.z, v.z);
	}

	Box3 transform(mat4 matrix){

		Box3 result;

		vec3 corners[8] = {
			{min.x, min.y, min.z},
			{max.x, min.y, min.z},
			{min.x, max.y, min.z},
			{max.x, max.y, min.z},
			{min.x, min.y, max.z},
			{max.x, min.y, max.z},
			{min.x, max.y, max.z},
			{max.x, max.y, max.z},
		};

		for(auto& c : corners){
			result.extend(vec3(matrix * vec4(c, 1.0f)));
		}

		return result;
	}
};

void writeGltf(
	string gltfPath,
	uint64_t numVertices,
	uint64_t numTriangles,
	vec3* positions,
	uint32_t* colors,
	vec2* uvs,
	uint32_t* indices,
	vec3* normals,
	string texturePath
){
	// Compute AABB for POSITION accessor (required by glTF spec)
	vec3 posMin(FLT_MAX), posMax(-FLT_MAX);
	for(uint32_t i = 0; i < numVertices; i++){
		posMin = glm::min(posMin, positions[i]);
		posMax = glm::max(posMax, positions[i]);
	}

	// Binary buffer layout: positions | colors | indices
	uint64_t posSize      = numVertices  * 12;
	uint64_t colorSize    = numVertices  * 4;
	uint64_t uvSize       = numVertices  * 8;
	uint64_t indexSize    = numTriangles * 12;
	uint64_t normalSize  = numVertices * 12;

	uint64_t offset_pos     = 0;
	uint64_t offset_color   = offset_pos + posSize;
	uint64_t offset_uv      = offset_color + colorSize;
	uint64_t offset_index   = offset_uv + uvSize;
	uint64_t offset_normal  = offset_index + indexSize;

	uint64_t binarySize = roundUp(posSize + colorSize + uvSize + indexSize + normalSize, 4llu);
	vector<uint8_t> binary(binarySize);
	memcpy(binary.data() + offset_pos   ,   positions, posSize);
	memcpy(binary.data() + offset_color ,   colors,    colorSize);
	memcpy(binary.data() + offset_uv    ,   uvs,       uvSize);
	memcpy(binary.data() + offset_index ,   indices,   indexSize);
	memcpy(binary.data() + offset_normal ,  normals,   normalSize);

	string dir = fs::path(gltfPath).parent_path().string();
	string basename = fs::path(gltfPath).stem().string();
	string binPath = format("{}/{}.bin", dir, basename);
	string binPath_relative = format("./{}.bin", basename);


	json j = {
		{"asset", {{"version", "2.0"}}},
		{"scene", 0},
		{"scenes", {{{"nodes", {0}}}}},
		{"nodes", {{{"mesh", 0}}}},
		{"meshes", {{
			{"primitives", {{
				{"attributes", {
					{"POSITION", 0}, 
					{"COLOR_0", 1},
					{"TEXCOORD_0", 3},
					{"NORMAL", 4},
				}},
				{"indices", 2},
				{"mode", 4}
			}}}
		}}},
		{"accessors", {
			{
				{"bufferView", 0}, {"byteOffset", 0}, {"componentType", 5126},
				{"count", numVertices}, {"type", "VEC3"},
				{"min", {posMin.x, posMin.y, posMin.z}},
				{"max", {posMax.x, posMax.y, posMax.z}}
			},
			{
				{"bufferView", 1}, {"byteOffset", 0}, {"componentType", 5121},
				{"count", numVertices}, {"type", "VEC4"}, {"normalized", true}
			},
			{
				{"bufferView", 2}, {"byteOffset", 0}, {"componentType", 5125},
				{"count", numTriangles * 3}, {"type", "SCALAR"}
			},
			{
				{"bufferView", 3}, {"byteOffset", 0}, {"componentType", 5126},
				{"count", numVertices}, {"type", "VEC2"},
			},
			{
				{"bufferView", 4}, {"byteOffset", 0}, {"componentType", 5126},
				{"count", numVertices}, {"type", "VEC3"},
			},
		}},
		{"bufferViews", {
			{{"buffer", 0}, {"byteOffset", offset_pos},     {"byteLength", posSize},        {"target", 34962}},
			{{"buffer", 0}, {"byteOffset", offset_color},   {"byteLength", colorSize},      {"target", 34962}},
			{{"buffer", 0}, {"byteOffset", offset_index},   {"byteLength", indexSize},      {"target", 34963}},
			{{"buffer", 0}, {"byteOffset", offset_uv},      {"byteLength", uvSize},         {"target", 34962}},
			{{"buffer", 0}, {"byteOffset", offset_normal},  {"byteLength", normalSize},     {"target", 34962}},
		}},
		{"buffers", {{
			{"byteLength", binarySize},
			{"uri", binPath_relative},
		}}}
	};

	if(texturePath != ""){
		j["images"] = {
			{
				{"bufferView", 5},
				{"mimeType", "image/jpeg"}
			}
		};

		j["textures"] = {
			{
				{"sampler", 0},
				{"source", 0}
			}
		};

		j["samplers"] = {
			{
				{"magFilter", 9729},
				{"minFilter", 9729},
				{"wrapS", 33071},
				{"wrapT", 33071},
			}
		};

		j["materials"] = {{
			{"name", "default_tex0"},
			{"pbrMetallicRoughness", {
				{"baseColorTexture", {{"index", 0}}},
				{"metallicFactor", 0.0},
			}},
		}};

		string textureFilename = fs::path(texturePath).filename().string();
		j["buffers"].push_back({
			{"byteLength", fs::file_size(textureFilename)},
			{"uri", textureFilename},
		});

		j["bufferViews"].push_back({
			{"buffer", 1},
			{"byteLength", fs::file_size(textureFilename)},
			{"byteOffset", 0},
		});

		j["meshes"][0]["primitives"][0]["material"] = 0;

	}

	// We need to split into gltf and bin files, since glb does not support buffers >4GB
	string strJson = j.dump(4);
	writeFile(gltfPath, strJson);
	writeBinaryFile(binPath, binary);
	println("wrote {} vertices, {} triangles to", numVertices, numTriangles);
	println("    {}", gltfPath);
	println("    {}", binPath);
}

void writeTexturedGltf(
	string plyPath,
	string gltfPath,
	uint64_t numVertices,
	uint64_t numTriangles,
	vec3* positions,
	uint32_t* colors,
	vec2* uvs,
	uint32_t* indices,
	vec3* normals
){
	println("start writeTexturedGltf()");

	// Handle Textures.
	// Some ply files might be accompanied with textures with similar names, e.g.:
	// somemodel.ply
	// - somemodel_u1_v1_diffuse.jpg
	// - somemodel_u1_v2_diffuse.jpg
	// - somemodel_u2_v1_diffuse.jpg
	// - somemodel_u2_v2_diffuse.jpg
	vector<string> texturePaths;
	vector<string> files = listFiles(fs::path(plyPath).parent_path().string());
	string ply_basename = fs::path(plyPath).stem().string();
	for(string file : files){
		string file_basename = fs::path(file).stem().string();
		if(file_basename.contains(ply_basename) && iEndsWith(file, ".jpg")){

			int pos = file_basename.find(ply_basename);
			string sub = file_basename.substr(ply_basename.size() + 1);
			auto tokens = split(sub, '_');

			if(tokens.size() == 3 && tokens[2] == "diffuse"){
				texturePaths.push_back(file);
			}
		}
	}

	// The textures determine the chunks/grid of meshes we split into
	struct Chunk{
		string texturePath = "";
		uint64_t numTriangles = 0;
		uint64_t counter = 0;
		uint32_t* indices = nullptr;
		Box3 aabb;
	};

	int size_u = 0;
	int size_v = 0;
	for(string texturePath : texturePaths){
		string textureBasename = fs::path(texturePath).stem().string();
		int pos = textureBasename.find(ply_basename);
		string sub = textureBasename.substr(ply_basename.size() + 1);
		auto tokens = split(sub, '_');

		int u = stoi(tokens[0].substr(1));
		int v = stoi(tokens[1].substr(1));

		size_u = max(size_u, u);
		size_v = max(size_v, v);
	}

	vector<Chunk> chunks(size_u * size_v);
	for(string texturePath : texturePaths){
		string textureBasename = fs::path(texturePath).stem().string();
		int pos = textureBasename.find(ply_basename);
		string sub = textureBasename.substr(ply_basename.size() + 1);
		auto tokens = split(sub, '_');

		// Appears to start at 1, so let's make it start from 0.
		int u = stoi(tokens[0].substr(1)) - 1;
		int v = stoi(tokens[1].substr(1)) - 1;
		int chunkID = u + size_u * v;
		chunks[chunkID].texturePath = texturePath;
	}

	// Now we need to split the mesh into one mesh per texture.
	// We'll keep the vertex buffer in one piece, but create separate index buffers.
	// First, let's count the number of vertices and triangles in each mesh. 
	println("count number of triangles per mesh");
	for(int triangleIndex = 0; triangleIndex < numTriangles; triangleIndex++){
		int i0 = indices[3 * triangleIndex + 0];
		int i1 = indices[3 * triangleIndex + 1];
		int i2 = indices[3 * triangleIndex + 2];

		vec2 uv0 = uvs[i0];
		vec2 uv1 = uvs[i1];
		vec2 uv2 = uvs[i2];

		bool usesSameTexture = true;
		if(floor(uv0.x) != floor(uv1.x) || floor(uv0.x) != floor(uv2.x)) usesSameTexture = false;
		if(floor(uv0.y) != floor(uv1.y) || floor(uv0.y) != floor(uv2.y)) usesSameTexture = false;

		if(!usesSameTexture){
			println("Unsupported: triangle has vertices that access different textures");
			exit(1235);
		}

		int u = floor(uv0.x);
		int v = floor(uv0.y);

		int chunkID = u + size_u * v;

		if(chunkID >= chunks.size()){
			println("huh?");
			exit(1235765);
		}

		chunks[chunkID].numTriangles++;
	}

	// Now that we know the number of triangles per mesh, we can allocate the corresponding index buffers
	println("allocate chunk index buffers");
	for (Chunk& chunk : chunks) {
		chunk.indices = (uint32_t*)malloc(chunk.numTriangles * 12);
	}

	// Fill chunk index buffers
	println("fill chunk index buffers");
	for(int triangleIndex = 0; triangleIndex < numTriangles; triangleIndex++){
		int i0 = indices[3 * triangleIndex + 0];
		int i1 = indices[3 * triangleIndex + 1];
		int i2 = indices[3 * triangleIndex + 2];

		vec2 uv0 = uvs[i0];
		vec2 uv1 = uvs[i1];
		vec2 uv2 = uvs[i2];

		int u = floor(uv0.x);
		int v = floor(uv0.y);
		int chunkID = u + size_u * v;

		Chunk& chunk = chunks[chunkID];
		int targetTriangleIndex = chunk.counter;

		chunk.indices[3 * targetTriangleIndex + 0] = i0;
		chunk.indices[3 * targetTriangleIndex + 1] = i1;
		chunk.indices[3 * targetTriangleIndex + 2] = i2;

		chunk.counter++;
	}

	// Adjust the uv coordinates
	println("adjust uv coordinates");
	for (int vertexIndex = 0; vertexIndex < numVertices; vertexIndex++) {
		vec2 uv = uvs[vertexIndex];

		uv.x = uv.x - floor(uv.x);
		uv.y = uv.y - floor(uv.y);
		uv.y = 1.0f - uv.y;

		uvs[vertexIndex] = uv;
	}

	// Compute AABB for each chunk and total AABB
	println("compute AABBs");
	Box3 aabb;
	for(Chunk& chunk : chunks){

		for(int triangleIndex = 0; triangleIndex < chunk.numTriangles; triangleIndex++){
			int i0 = chunk.indices[3 * triangleIndex + 0];
			int i1 = chunk.indices[3 * triangleIndex + 1];
			int i2 = chunk.indices[3 * triangleIndex + 2];

			chunk.aabb.extend(positions[i0]);
			chunk.aabb.extend(positions[i1]);
			chunk.aabb.extend(positions[i2]);

			aabb.extend(positions[i0]);
			aabb.extend(positions[i1]);
			aabb.extend(positions[i2]);
		}

	}

	// __debugbreak();

	// Binary buffer layout: positions | colors | indices
	uint64_t posSize    = numVertices  * 12;
	uint64_t colorSize  = numVertices  * 4;
	uint64_t uvSize     = numVertices  * 8;
	uint64_t normalSize = numVertices * 12;

	uint64_t offset_pos     = 0;
	uint64_t offset_color   = offset_pos + posSize;
	uint64_t offset_uv      = offset_color + colorSize;
	uint64_t offset_normal  = offset_uv + uvSize;

	string dir = fs::path(gltfPath).parent_path().string();
	string basename = fs::path(gltfPath).stem().string();
	string binPath = format("{}/{}.bin", dir, basename);
	string binPath_relative = format("./{}.bin", basename);

	ofstream fout(binPath, std::ios::binary);
	println("writing vertex buffers");
	fout.write((const char*)positions, posSize);
	fout.write((const char*)colors, colorSize);
	fout.write((const char*)uvs, uvSize);
	fout.write((const char*)normals, normalSize);

	json j_bufferViews = {
		{{"buffer", 0}, {"byteOffset", offset_pos},     {"byteLength", posSize},        {"target", 34962}},
		{{"buffer", 0}, {"byteOffset", offset_color},   {"byteLength", colorSize},      {"target", 34962}},
		{{"buffer", 0}, {"byteOffset", offset_uv},      {"byteLength", uvSize},         {"target", 34962}},
		{{"buffer", 0}, {"byteOffset", offset_normal},  {"byteLength", normalSize},     {"target", 34962}},
	};

	json j_accessors = {
		{
			{"bufferView", 0}, {"byteOffset", 0}, {"componentType", 5126},
			{"count", numVertices}, {"type", "VEC3"},
			{"min", {aabb.min.x, aabb.min.y, aabb.min.z}},
			{"max", {aabb.max.x, aabb.max.y, aabb.max.z}}
		},
		{
			{"bufferView", 1}, {"byteOffset", 0}, {"componentType", 5121},
			{"count", numVertices}, {"type", "VEC4"}, {"normalized", true}
		},
		{
			{"bufferView", 2}, {"byteOffset", 0}, {"componentType", 5126},
			{"count", numVertices}, {"type", "VEC2"},
		},
		{
			{"bufferView", 3}, {"byteOffset", 0}, {"componentType", 5126},
			{"count", numVertices}, {"type", "VEC3"},
		},
	};

	// Write mesh indices and create mesh entries
	println("writing meshes/indexbuffers");
	json j_meshes;
	json j_nodes;

	uint64_t byteOffset = posSize + colorSize + uvSize + normalSize;
	int bufferViewCount = 4;
	int nodeCount = 0;
	for(Chunk& chunk : chunks){

		uint64_t byteLength = chunk.numTriangles * 12;
		fout.write((const char*)chunk.indices, byteLength);

		j_accessors.push_back({
			{"bufferView", bufferViewCount},
			{"byteOffset", 0},
			{"componentType", 5125},
			{"count", 3 * chunk.numTriangles},
			{"type", "SCALAR"},
		});

		j_bufferViews.push_back({
			{"buffer", 0},
			{"byteLength", byteLength},
			{"byteOffset", byteOffset},
		});

		j_meshes.push_back({
			{"primitives", {{
				{"attributes", {
					{"POSITION", 0}, 
					{"COLOR_0", 1},
					{"TEXCOORD_0", 2},
					{"NORMAL", 3},
				}},
				{"indices", bufferViewCount},
				{"mode", 4},
				{"material", nodeCount},
			}}}
		});

		j_nodes.push_back({{"mesh", nodeCount}});
		nodeCount++;

		byteOffset += byteLength;
		bufferViewCount++;
	}

	json j_images;
	json j_textures;
	json j_materials;
	json j_samplers = {
		{
			{"magFilter", 9729},
			{"minFilter", 9729},
			{"wrapS", 33071},
			{"wrapT", 3307},
		}
	};

	json j_buffers = {{
		{"byteLength", byteOffset},
		{"uri", binPath_relative},
	}};

	int imageCount = 0;
	for(Chunk& chunk : chunks){

		string filename = fs::path(chunk.texturePath).filename().string();
		uint64_t filesize = fs::file_size(chunk.texturePath);

		j_buffers.push_back(
			{
				{"byteLength", filesize},
				{"uri", filename}
			}
		);

		j_bufferViews.push_back(
			{
				{"buffer", 1 + imageCount},
				{"byteLength", filesize},
				{"byteOffset", 0},
			}
		);

		j_images.push_back(
			{
				{"bufferView", bufferViewCount},
				{"mimeType", "image/jpg"},
			}
		);
		bufferViewCount++;

		j_textures.push_back(
			{
				{"sampler", 0},
				{"source", imageCount},
			}
		);

		j_materials.push_back(
			{
				{"name", "default_tex0"},
				{"pbrMetallicRoughness", {
					{"baseColorTexture", {
						{"index", imageCount}
					}},
					{"metallicFactor", 0.0},
				}}
			}
		);

		imageCount++;
	}


	json j = {
		{"asset", {{"version", "2.0"}}},
		{"scene", 0},
		{"scenes", {{{"nodes", {0}}}}},
		{"nodes", j_nodes},
		{"meshes", j_meshes},
		{"accessors", j_accessors},
		{"bufferViews", j_bufferViews},
		{"buffers", j_buffers},
		{"materials", j_materials},
		{"textures", j_textures},
		{"images", j_images},
		{"samplers", j_samplers},
	};

	println("writing gltf");
	string strJson = j.dump(4);
	writeFile(gltfPath, strJson);
	// writeBinaryFile(binPath, binary);
	println("wrote {} vertices, {} triangles to", numVertices, numTriangles);
	println("    {}", gltfPath);
	println("    {}", binPath);
}

void convert(string path, string gltfPath){
	
	if(!fs::exists(path)){
		println("file not found: {}", path);
		return;
	}

	auto mappedFile = Mapping::mapFile(path);

	string strProbableHeader((const char*)mappedFile->data, min(fs::file_size(path), 10'000llu));
	size_t pos_headerToken = strProbableHeader.find("end_header");
	if(pos_headerToken == string::npos){
		println("could not find end of header in ply file");
		return;
	}

	uint64_t vertex_start = pos_headerToken + 11;
	string strHeader = string((const char*)mappedFile->data, vertex_start);

	println("===== PLY HEADER");
	println("{}", strHeader);
	println("================");

	uint64_t numVertices = 0;
	uint64_t numTriangles = 0;
	uint64_t stride = 0;
	uint64_t OFFSET_X = 0;
	uint64_t OFFSET_Y = 0;
	uint64_t OFFSET_Z = 0;
	uint64_t OFFSET_NX = 0;
	uint64_t OFFSET_NY = 0;
	uint64_t OFFSET_NZ = 0;
	uint64_t OFFSET_S = 0;
	uint64_t OFFSET_T = 0;
	uint64_t OFFSET_RED = 0;
	uint64_t OFFSET_GREEN = 0;
	uint64_t OFFSET_BLUE = 0;


	vector<string> lines = split(strHeader, '\n');
	for(int lineIndex = 0; lineIndex < lines.size(); lineIndex++){
		string line = lines[lineIndex];
		vector<string> tokens = split(line, ' ');

		if(tokens[0] == "ply"){
			// ...
		}else if(tokens[0] == "format"){
			// ...
		}else if(tokens[0] == "comment"){
			// ...
		}else if(tokens[0] == "end_header"){
			// ...
		}else if(tokens[0] == "element" && tokens[1] == "vertex"){
			numVertices = stoi(tokens[2]);
		}else if(tokens[0] == "element" && tokens[1] == "face"){
			numTriangles = stoi(tokens[2]);
		}else if(tokens[0] == "property" && tokens[1] == "float" && tokens[2] == "x"){
			OFFSET_X = stride;
			stride += 4;
		}else if(tokens[0] == "property" && tokens[1] == "float" && tokens[2] == "y"){
			OFFSET_Y = stride;
			stride += 4;
		}else if(tokens[0] == "property" && tokens[1] == "float" && tokens[2] == "z"){
			OFFSET_Z = stride;
			stride += 4;
		}else if(tokens[0] == "property" && tokens[1] == "float" && tokens[2] == "nx"){
			OFFSET_NX = stride;
			stride += 4;
		}else if(tokens[0] == "property" && tokens[1] == "float" && tokens[2] == "ny"){
			OFFSET_NY = stride;
			stride += 4;
		}else if(tokens[0] == "property" && tokens[1] == "float" && tokens[2] == "nz"){
			OFFSET_NZ = stride;
			stride += 4;
		}else if(tokens[0] == "property" && tokens[1] == "float" && tokens[2] == "s"){
			OFFSET_S = stride;
			stride += 4;
		}else if(tokens[0] == "property" && tokens[1] == "float" && tokens[2] == "t"){
			OFFSET_T = stride;
			stride += 4;
		}else if(tokens[0] == "property" && tokens[1] == "uchar" && tokens[2] == "red"){
			OFFSET_RED = stride;
			stride += 1;
		}else if(tokens[0] == "property" && tokens[1] == "uchar" && tokens[2] == "green"){
			OFFSET_GREEN = stride;
			stride += 1;
		}else if(tokens[0] == "property" && tokens[1] == "uchar" && tokens[2] == "blue"){
			OFFSET_BLUE = stride;
			stride += 1;
		}else if(tokens[0] == "property" && tokens[1] == "list"){
			if(tokens[2] == "uchar" && tokens[3] == "int"){
				// take for granted
			}else{
				println("unsupported: {}", line);
				return;
			}
		}else if(tokens[0] == "property"){
			println("NOTE: Unhandled ply property: {}", line);

			if(tokens[1] == "float") stride += 4;
			if(tokens[1] == "uchar") stride += 1;
		}else {
			println("unsupported {}", line);
			return;
		}
	}

	uint64_t face_start = vertex_start + numVertices * stride;
	println("vertex_start: {:L}", vertex_start);
	println("face_start:   {:L}", face_start);

	println("numVertices  {:L}", numVertices);
	println("numTriangles {:L}", numTriangles);
	println("stride       {:L}", stride);
	println("OFFSET_X     {:L}", OFFSET_X);
	println("OFFSET_Y     {:L}", OFFSET_Y);
	println("OFFSET_Z     {:L}", OFFSET_Z);
	println("OFFSET_NX    {:L}", OFFSET_NX);
	println("OFFSET_NY    {:L}", OFFSET_NY);
	println("OFFSET_NZ    {:L}", OFFSET_NZ);
	println("OFFSET_S     {:L}", OFFSET_S);
	println("OFFSET_T     {:L}", OFFSET_T);

	shared_ptr<Buffer> positions = make_shared<Buffer>(numVertices * 12);
	shared_ptr<Buffer> colors    = make_shared<Buffer>(numVertices * 4);
	shared_ptr<Buffer> uvs       = make_shared<Buffer>(numVertices * 8);
	shared_ptr<Buffer> normals   = make_shared<Buffer>(numVertices * 12);
	shared_ptr<Buffer> indices   = make_shared<Buffer>(numTriangles * 12);

	Box3 aabb;
	{// LOAD POSITIONS
		double t_start = now();

		uint64_t CHUNKSIZE = 1'000'000;
		vector<uint64_t> chunkStarts;
		for(uint64_t i = 0; i < numVertices; i += CHUNKSIZE){
			chunkStarts.push_back(i);
		}

		mutex mtx;
		for_each(std::execution::par, chunkStarts.begin(), chunkStarts.end(), [&](uint64_t chunkStart){
			Box3 threadlocalAABB;

			uint64_t verticesInChunk = min(numVertices - chunkStart, CHUNKSIZE);
			for(uint64_t i = chunkStart; i < chunkStart + verticesInChunk; i++){
				vec3 position;
				position.x = mappedFile->read<float>(vertex_start + i * stride + OFFSET_X);
				position.y = mappedFile->read<float>(vertex_start + i * stride + OFFSET_Y);
				position.z = mappedFile->read<float>(vertex_start + i * stride + OFFSET_Z);

				positions->set<vec3>(position, 12 * i);

				threadlocalAABB.extend(position);
			}

			lock_guard<mutex> lock(mtx);
			aabb.extend(threadlocalAABB.min);
			aabb.extend(threadlocalAABB.max);
		});

		printElapsedTime("load vertices", t_start);
	}

	// LOAD VERTEX COLORS
	if(OFFSET_RED > 0 && OFFSET_GREEN > 0 && OFFSET_BLUE > 0){
		double t_start = now();

		uint64_t CHUNKSIZE = 1'000'000;
		vector<uint64_t> chunkStarts;
		for(uint64_t i = 0; i < numVertices; i += CHUNKSIZE){
			chunkStarts.push_back(i);
		}

		for_each(std::execution::par, chunkStarts.begin(), chunkStarts.end(), [&](uint64_t chunkStart){

			uint64_t verticesInChunk = min(numVertices - chunkStart, CHUNKSIZE);
			for(uint64_t i = chunkStart; i < chunkStart + verticesInChunk; i++){
				uint8_t r = mappedFile->read<uint8_t>(vertex_start + i * stride + OFFSET_RED);
				uint8_t g = mappedFile->read<uint8_t>(vertex_start + i * stride + OFFSET_GREEN);
				uint8_t b = mappedFile->read<uint8_t>(vertex_start + i * stride + OFFSET_BLUE);
				
				colors->set<uint8_t>(r, 4 * i + 0);
				colors->set<uint8_t>(g, 4 * i + 1);
				colors->set<uint8_t>(b, 4 * i + 2);
				colors->set<uint8_t>(255, 4 * i + 3);
			}

		});

		printElapsedTime("load vertex colors", t_start);
	}

	// LOAD UVS
	if(OFFSET_S > 0 && OFFSET_T > 0){
		double t_start = now();

		uint64_t CHUNKSIZE = 1'000'000;
		vector<uint64_t> chunkStarts;
		for(uint64_t i = 0; i < numVertices; i += CHUNKSIZE){
			chunkStarts.push_back(i);
		}

		for_each(std::execution::par, chunkStarts.begin(), chunkStarts.end(), [&](uint64_t chunkStart){

			uint64_t verticesInChunk = min(numVertices - chunkStart, CHUNKSIZE);
			for(uint64_t i = chunkStart; i < chunkStart + verticesInChunk; i++){
				float s = mappedFile->read<float>(vertex_start + i * stride + OFFSET_S);
				float t = mappedFile->read<float>(vertex_start + i * stride + OFFSET_T);

				uvs->set<float>(s, 8 * i + 0);
				uvs->set<float>(t, 8 * i + 4);
			}

		});

		printElapsedTime("load uvs", t_start);
	}

	// LOAD NORMALS
	if(OFFSET_NX > 0 && OFFSET_NY > 0 && OFFSET_NZ > 0){
		double t_start = now();

		uint64_t CHUNKSIZE = 1'000'000;
		vector<uint64_t> chunkStarts;
		for(uint64_t i = 0; i < numVertices; i += CHUNKSIZE){
			chunkStarts.push_back(i);
		}

		for_each(std::execution::par, chunkStarts.begin(), chunkStarts.end(), [&](uint64_t chunkStart){

			uint64_t verticesInChunk = min(numVertices - chunkStart, CHUNKSIZE);
			for(uint64_t i = chunkStart; i < chunkStart + verticesInChunk; i++){
				float nx = mappedFile->read<float>(vertex_start + i * stride + OFFSET_NX);
				float ny = mappedFile->read<float>(vertex_start + i * stride + OFFSET_NY);
				float nz = mappedFile->read<float>(vertex_start + i * stride + OFFSET_NZ);

				normals->set<float>(nx, 12 * i + 0);
				normals->set<float>(ny, 12 * i + 4);
				normals->set<float>(nz, 12 * i + 8);
			}

		});

		printElapsedTime("load normals", t_start);
	}

	{ // LOAD INDICES
		double t_start = now();

		uint64_t CHUNKSIZE = 1'000'000;
		vector<uint64_t> chunkStarts;
		for(uint64_t i = 0; i < numTriangles; i += CHUNKSIZE){
			chunkStarts.push_back(i);
		}

		for_each(std::execution::par, chunkStarts.begin(), chunkStarts.end(), [&](uint64_t chunkStart){

			uint64_t facesInChunk = min(numTriangles - chunkStart, CHUNKSIZE);
			for(uint64_t faceIndex = chunkStart; faceIndex < chunkStart + facesInChunk; faceIndex++){
				int count = mappedFile->read<uint8_t>(face_start + 13llu * faceIndex);

				int i0 = mappedFile->read<uint32_t>(face_start + 13llu * faceIndex + 1);
				int i1 = mappedFile->read<uint32_t>(face_start + 13llu * faceIndex + 5);
				int i2 = mappedFile->read<uint32_t>(face_start + 13llu * faceIndex + 9);

				indices->set<int>(i0, 12 * faceIndex + 0);
				indices->set<int>(i1, 12 * faceIndex + 4);
				indices->set<int>(i2, 12 * faceIndex + 8);
			}
		});

		printElapsedTime("load indices", t_start);
	}

	// writeGltf(gltfPath,
	// 	numVertices,
	// 	numTriangles,
	// 	(vec3*)positions->data,
	// 	(uint32_t*)colors->data,
	// 	(vec2*)uvs->data,
	// 	(uint32_t*)indices->data,
	// 	(vec3*)normals->data,
	// 	"F:/resources/meshes/iconem/VeniceGeneral-Airborne-flyover-400M-12x16k-local-binply/VeniceGeneral-Airborne-flyover-400M-12x16k_u1_v1_diffuse.jpg"
	// );

	writeTexturedGltf(
		path,
		gltfPath,
		numVertices,
		numTriangles,
		(vec3*)positions->data,
		(uint32_t*)colors->data,
		(vec2*)uvs->data,
		(uint32_t*)indices->data,
		(vec3*)normals->data
	);

}


int main(){

	convert(
		"F:/resources/meshes/iconem/VeniceGeneral-Airborne-flyover-400M-12x16k-local-binply/VeniceGeneral-Airborne-flyover-400M-12x16k.ply",
		"F:/resources/meshes/iconem/VeniceGeneral-Airborne-flyover-400M-12x16k-local-binply/venice.gltf"
	);

	return 0;
}