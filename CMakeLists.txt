cmake_minimum_required(VERSION 3.5)

set(TARGET_NAME CuRast) 

# CUDA kernels are compiled at runtime with NVRTC, so no CUDA compiler is needed at build time.
project(${TARGET_NAME} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(cmake/common.cmake)


# --------------------------------------------------------
# Source files
# --------------------------------------------------------
file(GLOB PROJECT_SOURCES CONFIGURE_DEPENDS 
	src/*.[ch]pp
)

add_executable(${TARGET_NAME} ${PROJECT_SOURCES})
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
