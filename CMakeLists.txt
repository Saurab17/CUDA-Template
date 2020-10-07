# The minimum version that supports CUDA Language
cmake_minimum_required(VERSION 3.11.4)

project(cuda_app 
        VERSION 1.0
        LANGUAGES CUDA)

# file(GLOB SOURCES *.cu)
set(SOURCES hello_world.cu)
set(EXECUTABLE ${PROJECT_NAME})
add_executable(${EXECUTABLE} ${SOURCES})

