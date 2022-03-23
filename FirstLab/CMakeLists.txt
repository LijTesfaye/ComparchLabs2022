cmake_minimum_required(VERSION 3.0)
project(untitled CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(untitled Lab1.cu)

set_target_properties(untitled PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
