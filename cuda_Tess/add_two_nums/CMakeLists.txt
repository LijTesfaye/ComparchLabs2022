cmake_minimum_required(VERSION 3.10)
project(add_two_nums CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(add_two_nums main.cu)

set_target_properties(add_two_nums PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
