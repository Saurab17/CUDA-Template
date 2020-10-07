cmake_minimum_required(VERSION 3.11.4)

project(cuda_vector_addition  
    VERSION 1.0  
    LANGUAGES CUDA)

set(NEW_METHOD TRUE)

if(NOT NEW_METHOD)
    set(SOURCES vector_add.cu)
else()
    set(SOURCES vector_add_new.cu)
endif()

set(EXECUTABLE ${PROJECT_NAME})
add_executable(${EXECUTABLE} ${SOURCES})