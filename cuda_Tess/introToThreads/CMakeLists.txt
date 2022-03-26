cmake_minimum_required(VERSION 3.10)
project(introToThreads CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(introToThreads main.cu)

set_target_properties(introToThreads PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
