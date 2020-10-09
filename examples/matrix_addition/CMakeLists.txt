cmake_minimum_required(VERSION 3.11.4)

project(cuda_matrix_addition  
    VERSION 1.0  
    LANGUAGES CUDA)

set(SOURCES matrix_addition.cu)

set(EXECUTABLE ${PROJECT_NAME})
add_executable(${EXECUTABLE} ${SOURCES})