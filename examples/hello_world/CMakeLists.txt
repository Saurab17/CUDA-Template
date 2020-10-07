cmake_minimum_required(VERSION 3.11.4)

project(cuda_hello_world
        VERSION 1.0
        LANGUAGES CUDA)

file(GLOB SOURCES *.cu)

set(EXECUTABLE ${PROJECT_NAME})
add_executable(${EXECUTABLE} ${SOURCES})