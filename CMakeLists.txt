# The minimum version that supports CUDA Language
cmake_minimum_required(VERSION 3.11.4)

project(cuda_template
        VERSION 1.0
        LANGUAGES CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_subdirectory(examples)


