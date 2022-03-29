cmake_minimum_required(VERSION 3.10)
project(1dStencil CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(1dStencil main.cu)

set_target_properties(1dStencil PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
