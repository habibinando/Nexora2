cmake_minimum_required(VERSION 3.22.1)
project("nexora_native" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(
    nexora_native
    SHARED
    nexora_native.cpp
    nexora_vulkan.cpp
    nexora_lm5m_trainer.cpp
)

find_library(
    log-lib
    log
)

find_library(
    vulkan-lib
    vulkan
)

target_link_libraries(
    nexora_native
    ${log-lib}
    ${vulkan-lib}
)

# Compile the GLSL sources into the exact SPIR-V bytecode consumed by the app.
# Never silently use a stale checked-in binary after a shader source is changed.
# Install glslang-tools for local native builds; CI installs it automatically.
find_program(GLSLANG_VALIDATOR NAMES glslangValidator)
if(GLSLANG_VALIDATOR)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(NEXORA_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    set(NEXORA_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
    set(NEXORA_SHADER_SOURCES
        "${NEXORA_SHADER_DIR}/matmul_tiled.comp"
        "${NEXORA_SHADER_DIR}/adamw.comp"
        "${NEXORA_SHADER_DIR}/swiglu.comp"
        "${NEXORA_SHADER_DIR}/swiglu_backward.comp"
        "${NEXORA_SHADER_DIR}/rmsnorm.comp"
    )
    add_custom_command(
        OUTPUT "${NEXORA_GENERATED_DIR}/nexora_spv.h"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${NEXORA_GENERATED_DIR}"
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/generate_spv_header.py"
            --compiler "${GLSLANG_VALIDATOR}"
            --source-dir "${NEXORA_SHADER_DIR}"
            --output "${NEXORA_GENERATED_DIR}/nexora_spv.h"
        DEPENDS ${NEXORA_SHADER_SOURCES} "${CMAKE_CURRENT_SOURCE_DIR}/generate_spv_header.py"
        VERBATIM
        COMMENT "Compiling Nexora compute shaders to validated SPIR-V modules"
    )
    add_custom_target(nexora_spv_header DEPENDS "${NEXORA_GENERATED_DIR}/nexora_spv.h")
    add_dependencies(nexora_native nexora_spv_header)
    target_include_directories(nexora_native BEFORE PRIVATE "${NEXORA_GENERATED_DIR}")
else()
    message(FATAL_ERROR "glslangValidator is required: refusing to build with a stale checked-in SPIR-V header. Install glslang-tools and retry.")
endif()

target_include_directories(nexora_native PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
