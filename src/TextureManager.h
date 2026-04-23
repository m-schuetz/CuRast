#pragma once

#include <unordered_map>
#include <vector>

#include "kernels/HostDeviceInterface.h"

using namespace std;

constexpr int64_t MAX_TEXTURES = 10'000;

// Serves the purpose of a global descriptor heap
// TODO: allow removal of textures
struct TextureManager{

	// static unordered_map<void*, int> stagedTextures;
	// static inline vector<Texture> textures;
	static inline Texture textures[MAX_TEXTURES];
	static inline int numTextures = 0;

	static inline Texture* create(){

		Texture* texture = &textures[numTextures];
		texture->handle = numTextures;
		numTextures++;

		return texture;
	}

};

// inline vector<Texture> TextureManager::textures;