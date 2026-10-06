cmake_minimum_required(VERSION 3.25)

# Builds a clustered LOD representation of a GLB mesh using meshoptimizer's clusterlod.h.
# Standalone, not part of the CuRast build:
#   cmake -S tools/clodbuilder -B tools/clodbuilder/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build tools/clodbuilder/build -j
#   ./tools/clodbuilder/build/clodbuilder <input.glb> <outputDir>

project(clodbuilder LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# meshoptimizer is downloaded at configure time. clusterlod.h is not part of the library itself, it lives in demo/.
include(FetchContent)
FetchContent_Declare(meshoptimizer
	GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
	GIT_TAG v1.3
	GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(meshoptimizer)

add_executable(clodbuilder main.cpp)
target_include_directories(clodbuilder PRIVATE
	${meshoptimizer_SOURCE_DIR}/demo
	${CMAKE_CURRENT_SOURCE_DIR}/../../libs
)
target_link_libraries(clodbuilder PRIVATE meshoptimizer)
