cmake_minimum_required(VERSION 3.24)

set(TARGET_NAME CuRast) 

# nvcc is usually not on PATH. Pick up the default toolkit location.
if (NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX} AND EXISTS "/usr/local/cuda/bin/nvcc")
	set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

# By default, compile kernels for the GPU(s) in this machine. 
# Override with e.g. -DCMAKE_CUDA_ARCHITECTURES="89;120"
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Default to a Release build unless a build type is given (e.g. -DCMAKE_BUILD_TYPE=Debug)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

project(${TARGET_NAME} LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 20)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(cmake/common.cmake)


# --------------------------------------------------------
# Source files
# --------------------------------------------------------
file(GLOB PROJECT_SOURCES CONFIGURE_DEPENDS 
	src/*.[ch]pp
	src/kernels/*.cu
)

add_executable(${TARGET_NAME} ${PROJECT_SOURCES})
target_compile_options(${TARGET_NAME} PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>
)
target_compile_definitions(${TARGET_NAME} PUBLIC
	IMGUI_DEFINE_MATH_OPERATORS
	GLFW_INCLUDE_NONE
)

# Include paths
target_include_directories(${TARGET_NAME} PRIVATE
	./src
	./libs
)

# std::stacktrace lives in a separate library with libstdc++
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	target_link_libraries(${TARGET_NAME} PRIVATE stdc++exp)
endif()

# libstdc++ implements std::execution::par via TBB if its headers are found
find_package(TBB QUIET)
if (TBB_FOUND)
	target_link_libraries(${TARGET_NAME} PRIVATE TBB::tbb)
endif()

# Add other third-party libs
ADD_CUDA(${TARGET_NAME})
ADD_VULKAN(${TARGET_NAME})
ADD_GLM(${TARGET_NAME})
ADD_IMGUI(${TARGET_NAME})
