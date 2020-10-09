cmake_minimum_required(VERSION 3.11.4)

project(cuda_matrix_multiplication  
    VERSION 1.0  
    LANGUAGES CUDA)

set(SOURCES matrix_multiplication.cu)

set(EXECUTABLE ${PROJECT_NAME})
add_executable(${EXECUTABLE} ${SOURCES})