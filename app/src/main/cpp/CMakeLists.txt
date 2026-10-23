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
