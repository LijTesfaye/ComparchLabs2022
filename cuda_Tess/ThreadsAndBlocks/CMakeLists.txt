cmake_minimum_required(VERSION 3.10)
project(ThreadsAndBlocks CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(ThreadsAndBlocks main.cu)

set_target_properties(ThreadsAndBlocks PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
