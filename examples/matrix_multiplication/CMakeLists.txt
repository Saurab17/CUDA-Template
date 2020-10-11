cmake_minimum_required(VERSION 3.11.4)

project(cuda_matrix_multiplication  
    VERSION 1.0  
    LANGUAGES CUDA)

set(TILED_METHOD FALSE)

if(NOT TILED_METHOD)
    set(SOURCES matrix_multiplication.cu)
else()
    set(SOURCES tiled_matrix_multiplication.cu)
endif()


set(EXECUTABLE ${PROJECT_NAME})
add_executable(${EXECUTABLE} ${SOURCES})