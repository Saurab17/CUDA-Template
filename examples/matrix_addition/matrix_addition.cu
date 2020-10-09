// Copyright (c) 2020 Saurabh Yadav
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TOTAL_ROWS      1000U
#define TOTAL_COLS      2000U

__global__
void init_matrix(float *matrix, int width, int height, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < width * height; i += gridDim.x * blockDim.x) {
        matrix[i]=val;
    }
}

__global__
void add_matrices(float * mat_A_arr, float * mat_B_arr, float * mat_C_arr,
                  int num_cols, int num_rows) {

	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(row < num_rows && col < num_cols) {
		mat_C_arr[row*num_cols + col] = mat_A_arr[row*num_cols + col] + 
										mat_B_arr[row*num_cols + col]; 
	}
}

int main() {
    
	cudaError_t err = cudaSuccess;

	float *mat_A, *mat_B, *mat_C;
	size_t memsize = TOTAL_COLS * TOTAL_ROWS * sizeof(float);

	/* Allocate memories for the matrices*/
	err = cudaMallocManaged(&mat_A, memsize);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate memory for matrix A (error code %s)!\n", 
				cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaMallocManaged(&mat_B, memsize);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate memory for matrix B (error code %s)!\n", 
				cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaMallocManaged(&mat_C, memsize);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate memory for matrix C (error code %s)!\n", 
				cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	/* Initialize matrices A and B */
	int blocksize_for_init = 256;
    int blocks_for_init = (TOTAL_ROWS*TOTAL_COLS + blocksize_for_init - 1) 
													/ (blocksize_for_init);
	init_matrix<<<blocks_for_init, blocksize_for_init>>>(mat_A, TOTAL_COLS, TOTAL_ROWS, 1);
	init_matrix<<<blocks_for_init, blocksize_for_init>>>(mat_B, TOTAL_COLS, TOTAL_ROWS, 2);
	err = cudaGetLastError();
	if( err != cudaSuccess) {
		fprintf(stderr, "Failed to initialize matrix (error code %s)!\n", 
				cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	/* Do the matrix addition */
	size_t blocksizeX = 16;
	size_t blocksizeY = 16;

	dim3 DimGrid( (TOTAL_COLS-1)/blocksizeX + 1, (TOTAL_ROWS-1)/blocksizeY + 1);
	dim3 DimBlock( blocksizeX, blocksizeY);
	add_matrices<<<DimGrid, DimBlock>>>(mat_A, mat_B, mat_C, TOTAL_COLS, TOTAL_ROWS);
	err = cudaGetLastError();
	if( err != cudaSuccess) {
		fprintf(stderr, "Failed to perform matrix addition (error code %s)!\n", 
				cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();
	// Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < (TOTAL_ROWS*TOTAL_COLS); i++)
        maxError = fmax(maxError, fabs(mat_C[i]-3.0f));
    printf("Max error: %f\n", maxError);
	return EXIT_SUCCESS;
}