cmake_minimum_required(VERSION 3.22.1)
project(nexora_native CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(nexora-native SHARED
    native.cpp
    gles_compute.cpp
)

# Real CPU maximization: Neon + multi-threading. GPU via GLES 3.1 compute.
find_library(log-lib log)
find_library(android-lib android)
find_library(glesv3-lib GLESv3)
find_library(egl-lib EGL)

target_compile_options(nexora-native PRIVATE
    -O3 -ffast-math -fno-exceptions -fno-rtti
    -fopenmp-simd
)

target_link_libraries(nexora-native
    ${log-lib}
    ${android-lib}
    ${glesv3-lib}
    ${egl-lib}
)
