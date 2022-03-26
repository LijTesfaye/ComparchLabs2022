cmake_minimum_required(VERSION 3.10)
project(HelloWorld CUDA)

set(CMAKE_CXX_STANDARD 14)

add_executable(HelloWorld main.cu)


set_target_properties(HelloWorld PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
