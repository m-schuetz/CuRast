cmake_minimum_required(VERSION 3.21)

set(TARGET_NAME CuRast)

option(USE_HIP "Build with HIP for AMD GPUs" OFF)

# Always start with CXX; add CUDA or HIP depending on option
project(${TARGET_NAME} LANGUAGES CXX)

if(USE_HIP)
    # enable_language(HIP) honors -DCMAKE_HIP_ARCHITECTURES, else auto-detects the
    # host GPU, else errors on a no-GPU host.
    enable_language(HIP)
    add_compile_definitions(USE_HIP=1)
else()
    enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 23)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(cmake/common.cmake)

# Paths to TurboJPEG
if(WIN32)
    set(TURBOJPEG_LIBRARY "${CMAKE_SOURCE_DIR}/libs/jpg/turbojpeg-static.lib")
else()
    # Use system TurboJPEG on Linux
    find_library(TURBOJPEG_LIBRARY turbojpeg REQUIRED)
endif() 

# --------------------------------------------------------
# Source files (C++ and CUDA)
# --------------------------------------------------------
file(GLOB PROJECT_SOURCES CONFIGURE_DEPENDS 
	src/*.[ch]pp
	src/*.cu
	src/kernels/textureTools.cu
	src/kernels/lines.cu
	#src/kernels/rasterization_utils.cu
)

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
)
if (WIN32)
	target_compile_definitions(${TARGET_NAME} PUBLIC VK_USE_PLATFORM_WIN32_KHR)
	set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)..")
endif()

# Include paths
target_include_directories(${TARGET_NAME} PRIVATE
	./src
	./libs
	${TURBOJPEG_INCLUDE_DIR}
)

add_subdirectory(libs/laszip)
target_link_libraries(${TARGET_NAME} PRIVATE
	${TURBOJPEG_LIBRARY}
	laszip
)

# For std::stacktrace support (GCC 13+) and TBB
find_package(TBB QUIET)
if(TBB_FOUND)
	target_link_libraries(${TARGET_NAME} PRIVATE TBB::tbb)
endif()
# Link stdc++_libbacktrace for std::stacktrace (GCC 13/14; not needed on Windows)
if(NOT WIN32)
	target_link_libraries(${TARGET_NAME} PRIVATE -lstdc++exp)
endif()

# Add other third-party libs
ADD_CUDA(${TARGET_NAME})
ADD_VULKAN(${TARGET_NAME})
ADD_GLM(${TARGET_NAME})
ADD_IMGUI(${TARGET_NAME})

# --------------------------------------------------------
# CUDA configuration
# --------------------------------------------------------
# Choose GPU architectures (adjust for your machine)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89 90)

# Enable separable compilation (for multi-file CUDA builds)
set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Treat warnings as errors (optional)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror all-warnings")

# set proper startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${TARGET_NAME})







# Second project
add_executable(PlyToGlb
	src/tools/PlyToGlb.cpp
	src/unsuck_platform_specific.cpp
)
set_target_properties(PlyToGlb PROPERTIES
	CXX_STANDARD 23
	CXX_STANDARD_REQUIRED ON
)
target_compile_options(PlyToGlb PRIVATE -std=c++23)
target_include_directories(PlyToGlb PRIVATE libs/glm)
target_include_directories(PlyToGlb PRIVATE
	./src
	./libs
)
find_package(TBB QUIET)
if(TBB_FOUND)
	target_link_libraries(PlyToGlb PRIVATE TBB::tbb)
endif()