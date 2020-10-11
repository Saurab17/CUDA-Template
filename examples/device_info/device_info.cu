// Copyright (c) 2020 Saurabh Yadav
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include <stdio.h>        
#include <unistd.h> 
#include <stdlib.h> 
     
#include <cuda_runtime.h>

int main()
{
    cudaError_t err = cudaSuccess;

    int device_count;
    err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get device count (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Number of GPUs:\t\t %d \n",device_count);

    cudaDeviceProp dev_prop;

    for(int i=0; i<device_count; i++) {

        err = cudaGetDeviceProperties(&dev_prop, i);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to get device %d properties (error code %s)!\n", i, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        printf("GPU Name:\t\t %s \n",dev_prop.name);
        printf("Clock rate:\t\t %d \n",dev_prop.clockRate);
        printf("Max Threads per block:\t %d \n",dev_prop.maxThreadsPerBlock);
        printf("Shared Memory per block: %lu \n",dev_prop.sharedMemPerBlock);
    }
    
    return 0;
}
