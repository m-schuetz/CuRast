// This is where the implementation of stb is compiled once

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "json/json.hpp"
#include "stb/stb_image_resize2.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "stb/stb_image_resize2.h"

// #define TINYGLTF_NO_INCLUDE_JSON
// #define TINYGLTF_NO_INCLUDE_STB_IMAGE
// #define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
// #define TINYGLTF_IMPLEMENTATION
// #include "tinygltf/tiny_gltf.h"

 