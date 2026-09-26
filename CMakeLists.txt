cmake_minimum_required(VERSION 3.5)

set(TARGET_NAME CuRast) 

# On Linux, nvcc is usually not on PATH. Pick up the default toolkit location.
if (NOT WIN32 AND NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX} AND EXISTS "/usr/local/cuda/bin/nvcc")
	set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

project(${TARGET_NAME} LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 23)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(cmake/common.cmake)


# --------------------------------------------------------
# Source files (C++ and CUDA)
# --------------------------------------------------------
file(GLOB PROJECT_SOURCES CONFIGURE_DEPENDS 
	src/*.[ch]pp
	src/*.cu
	src/kernels/lines.cu
)

# Choose GPU architectures (adjust for your machine). 
# Must be set before add_executable to take effect. 
set(CMAKE_CUDA_ARCHITECTURES 75 86 89 90)

add_executable(${TARGET_NAME} ${PROJECT_SOURCES})
if (MSVC)
	target_compile_options(${TARGET_NAME} PRIVATE
		$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
		$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /Zc:preprocessor>
	)
endif()
target_compile_definitions(${TARGET_NAME} PUBLIC
	IMGUI_DEFINE_MATH_OPERATORS
	_HAS_STD_BYTE=0
	GLFW_INCLUDE_NONE
)
if (WIN32)
	target_compile_definitions(${TARGET_NAME} PUBLIC VK_USE_PLATFORM_WIN32_KHR)
	set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)..")
endif()

# Include paths
target_include_directories(${TARGET_NAME} PRIVATE
	./src
	./libs
)

add_subdirectory(libs/laszip)
target_link_libraries(${TARGET_NAME} PRIVATE 
	laszip
)

# std::stacktrace lives in a separate library with libstdc++
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	target_link_libraries(${TARGET_NAME} PRIVATE stdc++exp)
endif()

# libstdc++ implements std::execution::par via TBB if its headers are found
if (NOT WIN32)
	find_package(TBB QUIET)
	if (TBB_FOUND)
		target_link_libraries(${TARGET_NAME} PRIVATE TBB::tbb)
	endif()
endif()

# Add other third-party libs
ADD_CUDA(${TARGET_NAME})
ADD_VULKAN(${TARGET_NAME})
ADD_GLM(${TARGET_NAME})
ADD_IMGUI(${TARGET_NAME})

# --------------------------------------------------------
# CUDA configuration
# --------------------------------------------------------
# Enable separable compilation (for multi-file CUDA builds)
set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Treat warnings as errors (optional)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror all-warnings")

# set proper startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${TARGET_NAME})
