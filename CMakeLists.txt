cmake_minimum_required(VERSION 3.18)
project(pylittle LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(PYLITTLE_BUILD_PYBIND "Build Python bindings" ON)

add_subdirectory(core)
add_subdirectory(bindings)
add_subdirectory(backends/cpu)
# add_subdirectory(backends/cuda) # enable when CUDA toolchain available
# add_subdirectory(backends/vulkan) # enable when Vulkan SDK available
