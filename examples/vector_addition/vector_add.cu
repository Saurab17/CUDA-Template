// Copyright (c) 2020 Saurabh Yadav
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_OF_ELEMENTS  40000U
#define ARRAY_A_ELEMENT  ((int) 'A')
#define ARRAY_B_ELEMENT  ((int) 'B')

//Compute vector sum C = A+B
//Each thread performs one pair-wise addition

__global__
void vecAddKernel(const int *arr_A, const int *arr_B, int *arr_C, int n) {

    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if(index < n) {
        arr_C[index] = arr_A[index] + arr_B[index];
    } 
}

int main() {

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    size_t arr_size = NUM_OF_ELEMENTS * sizeof(int);

    // Allocate the host input and outout vectors
    int *host_arr_A = (int *)malloc(arr_size);
    int *host_arr_B = (int *)malloc(arr_size);
    int *host_arr_C = (int *)malloc(arr_size);

    // Verify that all allocations succeeded
    if (host_arr_A == NULL || host_arr_B == NULL || host_arr_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < NUM_OF_ELEMENTS; ++i)
    {
        host_arr_A[i] = ARRAY_A_ELEMENT;
        host_arr_B[i] = ARRAY_B_ELEMENT;
    }

    // Allocate the device input vector A
    int *dev_arr_A = NULL;
    err = cudaMalloc((void **)&dev_arr_A, arr_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    int *dev_arr_B = NULL;
    err = cudaMalloc((void **)&dev_arr_B, arr_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    int *dev_arr_C = NULL;
    err = cudaMalloc((void **)&dev_arr_C, arr_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(dev_arr_A, host_arr_A, arr_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(dev_arr_B, host_arr_B, arr_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(NUM_OF_ELEMENTS + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_arr_A, dev_arr_B, dev_arr_C, NUM_OF_ELEMENTS);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(host_arr_C, dev_arr_C, arr_size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < NUM_OF_ELEMENTS; ++i)
    {
        if (fabs(host_arr_A[i] + host_arr_B[i] - host_arr_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(dev_arr_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(dev_arr_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(dev_arr_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(host_arr_A);
    free(host_arr_B);
    free(host_arr_C);

    printf("Done\n");
    return EXIT_SUCCESS;
}