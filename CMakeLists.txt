cmake_minimum_required(VERSION 3.18)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(PINN LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_BUILD_TYPE Debug)

set(SOURCES
    src/main.cu
    src/network.cu
    src/runtime.cu
    src/utils.cu
)

add_executable(PINN ${SOURCES})

target_include_directories(PINN
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(PINN PROPERTIES
    CUDA_ARCHITECTURES native
)