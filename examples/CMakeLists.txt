# The minimum version that supports CUDA Language
cmake_minimum_required(VERSION 3.11.4)

project(cuda_examples
        VERSION 1.0
        LANGUAGES CUDA)


# add_subdirectory(hello_world)
# add_subdirectory(vector_addition)
# add_subdirectory(matrix_addition)
# add_subdirectory(matrix_multiplication)
add_subdirectory(device_info)


