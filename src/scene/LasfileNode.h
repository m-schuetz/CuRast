#pragma once

#include <string>
#include <vector>
#include <cstring>
#include <print>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "SceneNode.h"
#include "types.h"
#include "./kernels/HostDeviceInterface.h"

using std::string;
using std::vector;
using std::println;
using glm::ivec2;

// Linux-only for now: the las file is accessed via mmap.
struct LasfileNode : public SceneNode{

	string file = "";
	void* mapped = nullptr;
	i64 fileSize = 0;

	vec3 min;
	vec3 max;
	vec3 scale;
	vec3 offset;
	i32 format;
	i64 numPoints;
	i64 pointRecordSize;  // bytes per point
	i64 offset_pointData; // byte offset to first point in file
	i64 offset_rgb;       // byte offset to rgb within a point, -1 if the format has no rgb
	bool compressed;      // LAZ - point data can not be read directly from the mapped file

	LasfileNode(string file, string name) : SceneNode(name){

		this->file = file;

		// memory-map the lasfile
		int fd = open(file.c_str(), O_RDONLY);
		if(fd == -1){
			println("ERROR: failed to open las file {}", file);
			exit(8234561);
		}

		struct stat st;
		if(fstat(fd, &st) != 0){
			println("ERROR: fstat failed for las file {}", file);
			close(fd);
			exit(8234562);
		}
		fileSize = st.st_size;

		if(fileSize < 227){
			println("ERROR: file too small to be a las file: {}", file);
			close(fd);
			exit(8234563);
		}

		mapped = mmap(nullptr, fileSize, PROT_READ, MAP_SHARED, fd, 0);
		close(fd); // the mapping stays valid after closing the file descriptor

		if(mapped == MAP_FAILED){
			mapped = nullptr;
			println("ERROR: mmap failed for las file {}", file);
			exit(8234564);
		}

		// parse header of lasfile
		// see https://www.asprs.org/wp-content/uploads/2019/07/LAS_1_4_r15.pdf
		if(memcmp(mapped, "LASF", 4) != 0){
			println("ERROR: not a las file (missing LASF signature): {}", file);
			exit(8234565);
		}

		u8 versionMajor          = read<u8>(24);
		u8 versionMinor          = read<u8>(25);
		u16 headerSize           = read<u16>(94);
		offset_pointData         = read<u32>(96);
		u8 pointFormatByte       = read<u8>(104);
		pointRecordSize          = read<u16>(105);
		u32 numPoints_legacy     = read<u32>(107);

		// LAZ sets bit 7 (and historically bit 6) of the point data format
		compressed = (pointFormatByte & 0b1100'0000) != 0;
		format     = pointFormatByte & 0b0011'1111;

		numPoints = numPoints_legacy;
		bool isLas14 = versionMajor == 1 && versionMinor >= 4 && headerSize >= 375 && fileSize >= 375;
		if(isLas14){
			u64 numPoints_extended = read<u64>(247);
			if(numPoints_extended > 0) numPoints = numPoints_extended;
		}

		scale = {
			read<double>(131),
			read<double>(139),
			read<double>(147),
		};
		offset = {
			read<double>(155),
			read<double>(163),
			read<double>(171),
		};
		// header stores max before min for each axis
		max = {read<double>(179), read<double>(195), read<double>(211)};
		min = {read<double>(187), read<double>(203), read<double>(219)};

		switch(format){
			case 2:  offset_rgb = 20; break;
			case 3:  offset_rgb = 28; break;
			case 5:  offset_rgb = 28; break;
			case 7:  offset_rgb = 30; break;
			case 8:  offset_rgb = 30; break;
			case 10: offset_rgb = 30; break;
			default: offset_rgb = -1; break;
		}

		// points are rendered without the offset
		aabb.min = min - offset;
		aabb.max = max - offset;

		if(compressed){
			println("WARNING: {} is LAZ-compressed, point data can not be accessed from the mapped file.", file);
		}else if(offset_pointData + numPoints * pointRecordSize > fileSize){
			println("WARNING: {} is smaller than expected from its header ({} points, {} bytes per point).", file, numPoints, pointRecordSize);
		}
	}

	// owns the mapping
	LasfileNode(const LasfileNode&) = delete;
	LasfileNode& operator=(const LasfileNode&) = delete;

	~LasfileNode(){
		if(mapped != nullptr){
			munmap(mapped, fileSize);
			mapped = nullptr;
		}
	}

	template<typename T>
	T read(i64 byteOffset){
		T value;
		memcpy(&value, (u8*)mapped + byteOffset, sizeof(T));

		return value;
	}

	uint64_t getGpuMemoryUsage(){
		return 0;
	}

};
