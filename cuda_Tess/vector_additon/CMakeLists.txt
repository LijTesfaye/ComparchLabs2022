cmake_minimum_required(VERSION 3.10)
project(vector_additon CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(vector_additon main.cu)

set_target_properties(vector_additon PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
