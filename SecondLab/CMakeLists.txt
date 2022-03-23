cmake_minimum_required(VERSION 3.10)
project(SecondLab CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(SecondLab main.cu)

set_target_properties(SecondLab PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
