cmake_minimum_required(VERSION 3.18)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(PINN LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_BUILD_TYPE Debug)

include_directories(include)

set(SOURCES
    src/main.cu
    src/network.cu
    src/utils.cu
)

add_executable(PINN ${SOURCES})

set_target_properties(PINN PROPERTIES
    CUDA_ARCHITECTURES "native"
)
