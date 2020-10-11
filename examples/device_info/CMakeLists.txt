cmake_minimum_required(VERSION 3.11.4)

project(cuda_device_info
        VERSION 1.0
        LANGUAGES CUDA)

set(SOURCES device_info.cu)

set(EXECUTABLE ${PROJECT_NAME})
add_executable(${EXECUTABLE} ${SOURCES})