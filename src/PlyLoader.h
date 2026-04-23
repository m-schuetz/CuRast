
#include <thread>
#include <queue>
#include <stacktrace>
#include <set>

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

// YOLO
using namespace std;

namespace PlyLoader{

	#define USE_VULKAN_SHARED_MEMORY

	struct PlyData{
		string path = "";
		shared_ptr<SceneNode> node = nullptr;
		Texture defaultTexture;
		// vector<Texture> textures;
		vector<Mesh> meshes;

		#if defined(USE_VULKAN_SHARED_MEMORY)
			VulkanCudaSharedMemory* mem_positions = nullptr;
			VulkanCudaSharedMemory* mem_indices = nullptr;
			VulkanCudaSharedMemory* mem_colors = nullptr;
			VulkanCudaSharedMemory* mem_uvs = nullptr;
		#else 
			CudaVirtualMemory* memory = nullptr;
		#endif
	};

	// void loadIndices_basic(
	// 	PlyData* data,
	// 	shared_ptr<Mapping::MappedFile> mappedFile,
	// 	uint64_t numFaces,
	// 	uint64_t face_start
	// ){
	// 	double t_start = now();

	// 	// numFaces = min(numFaces, 1'000'000ll);
	// 	uint32_t* host_indices = nullptr;
	// 	cuMemAllocHost((void**)&host_indices, numFaces * 12);
	// 	data->mem_indices = MemoryManager::allocVulkanCudaShared(numFaces * 12, "ply indices");
	// 	data->mem_indices->commit(numFaces * 12);

	// 	uint64_t CHUNKSIZE = 1'000'000;
	// 	vector<uint64_t> chunkStarts;
	// 	for(uint64_t i = 0; i < numFaces; i += CHUNKSIZE){
	// 		chunkStarts.push_back(i);
	// 	}

	// 	for_each(std::execution::par, chunkStarts.begin(), chunkStarts.end(), [&](uint64_t chunkStart){

	// 		uint64_t facesInChunk = min(numFaces - chunkStart, CHUNKSIZE);
	// 		for(uint64_t faceIndex = chunkStart; faceIndex < chunkStart + facesInChunk; faceIndex++){
	// 			int count = mappedFile->read<uint8_t>(face_start + 13llu * faceIndex);

	// 			host_indices[3 * faceIndex + 0] = mappedFile->read<uint32_t>(face_start + 13llu * faceIndex + 1);
	// 			host_indices[3 * faceIndex + 1] = mappedFile->read<uint32_t>(face_start + 13llu * faceIndex + 5);
	// 			host_indices[3 * faceIndex + 2] = mappedFile->read<uint32_t>(face_start + 13llu * faceIndex + 9);
	// 		}
	// 	});

	// 	data->mem_indices->memcopyHtoD(0, host_indices, numFaces * 12);
	// 	cuMemFreeHost(host_indices);

	// 	printElapsedTime("load indices", t_start);
	// }

	PlyData* load(string path, CUcontext context){

		if(!fs::exists(path)){
			println("file not found: {}", path);
			return nullptr;
		}


		PlyData* data = new PlyData();

		string basename = fs::path(path).stem().string();
		auto mappedFile = Mapping::mapFile(path);

		println("basename: {}", basename);

		// Handle Textures.
		// Some ply files might be accompanied with textures with similar names, e.g.:
		// somemodel.ply
		// - somemodel_u1_v1_diffuse.jpg
		// - somemodel_u1_v2_diffuse.jpg
		// - somemodel_u2_v1_diffuse.jpg
		// - somemodel_u2_v2_diffuse.jpg
		// vector<string> texturePaths;
		// vector<string> files = listFiles(fs::path(path).parent_path().string());
		// for(string file : files){
		// 	string file_basename = fs::path(file).stem().string();
		// 	if(file_basename.contains(basename) && iEndsWith(file, ".jpg")){

		// 		int pos = file_basename.find(basename);
		// 		string sub = file_basename.substr(basename.size() + 1);
		// 		auto tokens = split(sub, '_');
		// 		// println("{} {} {}", tokens[0], tokens[1], tokens[2]);

		// 		if(tokens.size() == 3 && tokens[2] == "diffuse"){
		// 			texturePaths.push_back(file);
		// 		}
		// 	}
		// }

		
		string strProbableHeader((const char*)mappedFile->data, min(fs::file_size(path), 10'000llu));
		size_t pos_headerToken = strProbableHeader.find("end_header");
		if(pos_headerToken == string::npos){
			println("could not find end of header in ply file");
			return nullptr;
		}

		uint64_t vertex_start = pos_headerToken + 11;
		string strHeader = string((const char*)mappedFile->data, vertex_start);

		println("===== PLY HEADER");
		println("{}", strHeader);
		println("================");

		uint64_t numVertices = 0;
		uint64_t numFaces = 0;
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
				numFaces = stoi(tokens[2]);
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
					return nullptr;
				}
			}else if(tokens[0] == "property"){
				println("NOTE: Unhandled ply property: {}", line);

				if(tokens[1] == "float") stride += 4;
				if(tokens[1] == "uchar") stride += 1;
			}else {
				println("unsupported {}", line);
				return nullptr;
			}
		}

		uint64_t face_start = vertex_start + numVertices * stride;
		println("vertex_start: {:L}", vertex_start);
		println("face_start:   {:L}", face_start);

		println("numVertices  {:L}", numVertices);
		println("numFaces     {:L}", numFaces);
		println("stride       {:L}", stride);
		println("OFFSET_X     {:L}", OFFSET_X);
		println("OFFSET_Y     {:L}", OFFSET_Y);
		println("OFFSET_Z     {:L}", OFFSET_Z);
		println("OFFSET_NX    {:L}", OFFSET_NX);
		println("OFFSET_NY    {:L}", OFFSET_NY);
		println("OFFSET_NZ    {:L}", OFFSET_NZ);
		println("OFFSET_S     {:L}", OFFSET_S);
		println("OFFSET_T     {:L}", OFFSET_T);

		Box3 aabb;
		{// LOAD POSITIONS
			double t_start = now();

			vec3* host_positions = nullptr;
			cuMemAllocHost((void**)&host_positions, numVertices * 12);

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
					vec3& position = host_positions[i];
					position.x = mappedFile->read<float>(vertex_start + i * stride + OFFSET_X);
					position.y = mappedFile->read<float>(vertex_start + i * stride + OFFSET_Y);
					position.z = mappedFile->read<float>(vertex_start + i * stride + OFFSET_Z);
					threadlocalAABB.extend(position);
				}

				lock_guard<mutex> lock(mtx);
				aabb.extend(threadlocalAABB.min);
				aabb.extend(threadlocalAABB.max);
			});

			data->mem_positions = MemoryManager::allocVulkanCudaShared(numVertices * 12, "ply positions");
			data->mem_positions->commit(numVertices * 12);
			data->mem_positions->memcopyHtoD(0, host_positions, numVertices * 12);
			cuMemFreeHost(host_positions);

			printElapsedTime("load vertices", t_start);
		}

		// LOAD VERTEX COLORS
		if(OFFSET_RED > 0 && OFFSET_GREEN > 0 && OFFSET_BLUE > 0){
			double t_start = now();

			uint8_t* host_colors = nullptr;
			cuMemAllocHost((void**)&host_colors, numVertices * 4);

			uint64_t CHUNKSIZE = 1'000'000;
			vector<uint64_t> chunkStarts;
			for(uint64_t i = 0; i < numVertices; i += CHUNKSIZE){
				chunkStarts.push_back(i);
			}

			for_each(std::execution::par, chunkStarts.begin(), chunkStarts.end(), [&](uint64_t chunkStart){

				uint64_t verticesInChunk = min(numVertices - chunkStart, CHUNKSIZE);
				for(uint64_t i = chunkStart; i < chunkStart + verticesInChunk; i++){
					host_colors[4 * i + 0] = mappedFile->read<uint8_t>(vertex_start + i * stride + OFFSET_RED);
					host_colors[4 * i + 1] = mappedFile->read<uint8_t>(vertex_start + i * stride + OFFSET_GREEN);
					host_colors[4 * i + 2] = mappedFile->read<uint8_t>(vertex_start + i * stride + OFFSET_BLUE);
					host_colors[4 * i + 3] = 255;
				}

			});

			data->mem_colors = MemoryManager::allocVulkanCudaShared(numVertices * 4, "ply vertex colors");
			data->mem_colors->commit(numVertices * 4);
			data->mem_colors->memcopyHtoD(0, host_colors, numVertices * 4);
			cuMemFreeHost(host_colors);

			printElapsedTime("load vertex colors", t_start);
		}

		// LOAD UV COORDINATES
		vec2* host_uvs = nullptr;
		cuMemAllocHost((void**)&host_uvs, numVertices * 8);
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
					vec2 uv;
					uv.x = mappedFile->read<float>(vertex_start + i * stride + OFFSET_S);
					uv.y = mappedFile->read<float>(vertex_start + i * stride + OFFSET_T);
					host_uvs[i] = uv;
				}

			});

			data->mem_uvs = MemoryManager::allocVulkanCudaShared(numVertices * 8, "ply vertex uvs");
			data->mem_uvs->commit(numVertices * 8);
			data->mem_uvs->memcopyHtoD(0, host_uvs, numVertices * 8);

			printElapsedTime("load vertex uvs", t_start);
		}

		{ // LOAD INDICES
			double t_start = now();

			// numFaces = min(numFaces, 1'000'000ll);
			uint32_t* host_indices = nullptr;
			cuMemAllocHost((void**)&host_indices, numFaces * 12);

			uint64_t CHUNKSIZE = 1'000'000;
			vector<uint64_t> chunkStarts;
			for(uint64_t i = 0; i < numFaces; i += CHUNKSIZE){
				chunkStarts.push_back(i);
			}

			for_each(std::execution::par, chunkStarts.begin(), chunkStarts.end(), [&](uint64_t chunkStart){

				uint64_t facesInChunk = min(numFaces - chunkStart, CHUNKSIZE);
				for(uint64_t faceIndex = chunkStart; faceIndex < chunkStart + facesInChunk; faceIndex++){
					int count = mappedFile->read<uint8_t>(face_start + 13llu * faceIndex);

					host_indices[3 * faceIndex + 0] = mappedFile->read<uint32_t>(face_start + 13llu * faceIndex + 1);
					host_indices[3 * faceIndex + 1] = mappedFile->read<uint32_t>(face_start + 13llu * faceIndex + 5);
					host_indices[3 * faceIndex + 2] = mappedFile->read<uint32_t>(face_start + 13llu * faceIndex + 9);
				}
			});

			data->mem_indices = MemoryManager::allocVulkanCudaShared(numFaces * 12, "ply indices");
			data->mem_indices->commit(numFaces * 12);
			data->mem_indices->memcopyHtoD(0, host_indices, numFaces * 12);
			cuMemFreeHost(host_indices);

			printElapsedTime("load indices", t_start);
		}

		Mesh* mesh = new Mesh();
		mesh->isLoaded = true;
		mesh->cptr_uv = 0;
		mesh->cptr_color = 0;
		mesh->cptr_normal = 0;
		mesh->compressed = false;
		mesh->aabb = aabb;
		mesh->cptr_position = data->mem_positions->cptr;
		mesh->numVertices = numVertices;
		mesh->numTriangles = numFaces;
		if(data->mem_indices) mesh->cptr_indices = data->mem_indices->cptr;
		if(data->mem_colors)  mesh->cptr_color = data->mem_colors->cptr;
		// if(data->mem_uvs)     mesh->cptr_uv = data->mem_uvs->cptr;

		shared_ptr<SNTriangles> node = make_shared<SNTriangles>("ply node");
		node->mesh = mesh;
		node->aabb = mesh->aabb;

		data->node = node;

		// if(texturePaths.size() == 0){
		// 	// Easy, one mesh
		// 	loadIndices_basic(data, mappedFile, numFaces, face_start);
		// }else{
		// 	// We only support one texture per mesh, so if there are textures we need to split into multiple meshes
		// 	loadIndices_textured(data, mappedFile, numFaces, face_start);
		// }

		// shared_ptr<SceneNode> node = make_shared<SceneNode>("ply node");
		// data->node = node;

		// {
		// 	double t_start = now();

		// 	uint32_t* host_indices = nullptr;
		// 	cuMemAllocHost((void**)&host_indices, numFaces * 12);

		// 	uint64_t CHUNKSIZE = 1'000'000;
		// 	vector<uint64_t> chunkStarts;
		// 	for(uint64_t i = 0; i < numFaces; i += CHUNKSIZE){
		// 		chunkStarts.push_back(i);
		// 	}

		// 	for_each(std::execution::par, chunkStarts.begin(), chunkStarts.end(), [&](uint64_t chunkStart){
		// 		uint64_t facesInChunk = min(numFaces - chunkStart, CHUNKSIZE);
		// 		for(uint64_t faceIndex = chunkStart; faceIndex < chunkStart + facesInChunk; faceIndex++){
		// 			int count = mappedFile->read<uint8_t>(face_start + 13llu * faceIndex);

		// 			host_indices[3 * faceIndex + 0] = mappedFile->read<uint32_t>(face_start + 13llu * faceIndex + 1);
		// 			host_indices[3 * faceIndex + 1] = mappedFile->read<uint32_t>(face_start + 13llu * faceIndex + 5);
		// 			host_indices[3 * faceIndex + 2] = mappedFile->read<uint32_t>(face_start + 13llu * faceIndex + 9);
		// 		}
		// 	});

		// 	// Here is where it gets tricky: Some PLY files may come with textures.
		// 	// However, we only support one texture per mesh, so in that case we need to produce multiple meshes.

		// 	// if(texturePaths.size() == 0){
		// 		// Mesh* mesh = new Mesh();
		// 		// mesh->isLoaded = true;
		// 		// mesh->cptr_uv = 0;
		// 		// mesh->cptr_color = 0;
		// 		// mesh->cptr_normal = 0;
		// 		// mesh->compressed = false;
		// 		// mesh->aabb = aabb;
		// 		// mesh->cptr_position = data->mem_positions->cptr;
		// 		// mesh->numVertices = numVertices;
		// 		// mesh->numTriangles = numFaces;
		// 		// if(data->mem_indices) mesh->cptr_indices = data->mem_indices->cptr;
		// 		// if(data->mem_colors)  mesh->cptr_color = data->mem_colors->cptr;
		// 		// // if(data->mem_uvs)     mesh->cptr_uv = data->mem_uvs->cptr;

		// 		// shared_ptr<SNTriangles> meshNode = make_shared<SNTriangles>("ply node");
		// 		// meshNode->mesh = mesh;
		// 		// meshNode->aabb = mesh->aabb;

		// 		// node->children.push_back(meshNode);

		// 		// data->mem_indices = MemoryManager::allocVulkanCudaShared(numFaces * 12, "ply indices");
		// 		// data->mem_indices->commit(numFaces * 12);
		// 		// data->mem_indices->memcopyHtoD(0, host_indices, numFaces * 12);
		// 	// }else{

		// 	// 	// The textures determine the chunks/grid of meshes we split into
		// 	// 	struct Chunk{
		// 	// 		string texturePath = "";
		// 	// 		uint64_t numTriangles = 0;
		// 	// 	};

		// 	// 	int size_u = 0;
		// 	// 	int size_v = 0;
		// 	// 	for(string texturePath : texturePaths){
		// 	// 		string textureBasename = fs::path(texturePath).stem().string();
		// 	// 		int pos = textureBasename.find(basename);
		// 	// 		string sub = textureBasename.substr(basename.size() + 1);
		// 	// 		auto tokens = split(sub, '_');

		// 	// 		int u = stoi(tokens[0].substr(1));
		// 	// 		int v = stoi(tokens[1].substr(1));

		// 	// 		size_u = max(size_u, u);
		// 	// 		size_v = max(size_v, v);
		// 	// 	}

		// 	// 	vector<Chunk> chunks(size_u * size_v);
		// 	// 	for(string texturePath : texturePaths){
		// 	// 		string textureBasename = fs::path(texturePath).stem().string();
		// 	// 		int pos = textureBasename.find(basename);
		// 	// 		string sub = textureBasename.substr(basename.size() + 1);
		// 	// 		auto tokens = split(sub, '_');

		// 	// 		// Appears to start at 1, so let's make it start from 0.
		// 	// 		int u = stoi(tokens[0].substr(1)) - 1;
		// 	// 		int v = stoi(tokens[1].substr(1)) - 1;
		// 	// 		int chunkID = u + size_u * v;
		// 	// 		chunks[chunkID].texturePath = texturePath;
		// 	// 	}

		// 	// 	// Now we need to process all triangles and check which chunk they belong to.
		// 	// 	for(int triangleIndex = 0; triangleIndex < numFaces; triangleIndex++){

		// 	// 		int i0 = host_indices[3 * triangleIndex + 0];
		// 	// 		int i1 = host_indices[3 * triangleIndex + 1];
		// 	// 		int i2 = host_indices[3 * triangleIndex + 2];

		// 	// 		vec2 uv0 = host_uvs[i0];
		// 	// 		vec2 uv1 = host_uvs[i1];
		// 	// 		vec2 uv2 = host_uvs[i2];

		// 	// 		int u0 = floor(uv0.x);
		// 	// 		int v0 = floor(uv0.y);
		// 	// 		int chunkID = u0 + v0 * size_u;
		// 	// 		chunks[chunkID].numTriangles++;

		// 	// 		// sanity check
		// 	// 		bool triangleUsesDifferentTextures = false;
		// 	// 		triangleUsesDifferentTextures = floor(uv0.x) != floor(uv1.x) || floor(uv0.x) != floor(uv2.x);
		// 	// 		triangleUsesDifferentTextures = floor(uv0.y) != floor(uv1.y) || floor(uv0.y) != floor(uv2.y);

		// 	// 		if(triangleUsesDifferentTextures){
		// 	// 			println("Not supported: triangle uses different textures.");
		// 	// 			return nullptr;
		// 	// 		}

		// 	// 	}

		// 	// 	__debugbreak();
		// 	// }


		// 	cuMemFreeHost(host_indices);
		// 	cuMemFreeHost(host_uvs);
		// 	printElapsedTime("load indices", t_start);
		// }

		// exit(123);

		

		return data;
	}

};




