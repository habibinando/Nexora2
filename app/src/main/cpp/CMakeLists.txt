cmake_minimum_required(VERSION 3.22.1)
project(nexora_native CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(nexora-native SHARED
    native.cpp
    vulkan_compute.cpp
)

# CPU kernels use NEON + a native thread pool. GPU GEMM uses Vulkan compute.
find_library(log-lib log)
find_library(android-lib android)
find_library(vulkan-lib vulkan)

if(NOT vulkan-lib)
    message(FATAL_ERROR "Android NDK Vulkan loader library was not found")
endif()

target_compile_options(nexora-native PRIVATE
    -O3 -ffast-math -fno-exceptions -fno-rtti
    -fopenmp-simd
)

target_link_libraries(nexora-native
    ${log-lib}
    ${android-lib}
    ${vulkan-lib}
)
