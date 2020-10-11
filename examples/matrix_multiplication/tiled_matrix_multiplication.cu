// Copyright (c) 2020 Saurabh Yadav
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* This example to analyse practically the performance benefits of
using tiled algorithms that use shared memory of the gpu */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define MAT_A_TOTAL_ROWS      4000U
#define MAT_A_TOTAL_COLS      5000U
#define MAT_B_TOTAL_ROWS      MAT_A_TOTAL_COLS
#define MAT_B_TOTAL_COLS      6000U

#define TILE_WIDTH            16


__global__
void init_matrix(float *matrix, int width, int height, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < width * height; i += gridDim.x * blockDim.x) {
        matrix[i]=val;
    }
}

__global__
void tiled_matrix_multiplication(float * mat_A_arr, float * mat_B_arr, float * mat_C_arr,
                  int num_A_rows, int num_A_cols, int num_B_cols) {

    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH]; // tiled shared memory for matrix A
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH]; // tiled shared memory for matrix B

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

	int row = ty + by * blockDim.y;
	int col = tx + bx * blockDim.x;

    float c_value = 0.0; 

    for(size_t t=0; t<((num_A_cols-1)/TILE_WIDTH+1); t++) {

        if( row < num_A_rows && (t*TILE_WIDTH+tx) < num_A_cols ) {
            ds_A[ty][tx] = mat_A_arr[row*num_A_cols + t*TILE_WIDTH+tx];
        } else {
            ds_A[ty][tx] = 0.0;
        }

        if( (t*TILE_WIDTH+ty) < num_A_cols && col < num_B_cols ) {
            ds_B[ty][tx] = mat_B_arr[(t*TILE_WIDTH+ty)*num_B_cols + col];
        } else {
            ds_B[ty][tx] = 0.0;
        }
        __syncthreads();

        for(size_t i=0; i<TILE_WIDTH; i++) {
            c_value += ds_A[ty][i] * ds_B[i][tx];
        }

        __syncthreads();
    }

    if (row < num_A_rows && col < num_B_cols) {
        mat_C_arr[row*num_B_cols + col] = c_value;   
    }
}

int main() {
    
	cudaError_t err = cudaSuccess;

	float *mat_A, *mat_B, *mat_C;
	size_t memsize_A = MAT_A_TOTAL_ROWS * MAT_A_TOTAL_COLS * sizeof(float);
	size_t memsize_B = MAT_B_TOTAL_ROWS * MAT_B_TOTAL_COLS * sizeof(float);
	size_t memsize_C = MAT_A_TOTAL_ROWS * MAT_B_TOTAL_COLS * sizeof(float);

	/* Allocate memories for the matrices*/
	err = cudaMallocManaged(&mat_A, memsize_A);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate memory for matrix A (error code %s)!\n", 
				cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaMallocManaged(&mat_B, memsize_B);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate memory for matrix B (error code %s)!\n", 
				cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaMallocManaged(&mat_C, memsize_C);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate memory for matrix C (error code %s)!\n", 
				cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	/* Initialize matrices A and B */
	int blocksize_for_init = 256;
    int blocks_for_matA = (MAT_A_TOTAL_ROWS*MAT_A_TOTAL_COLS + blocksize_for_init - 1) 
													/ (blocksize_for_init);
    int blocks_for_matB = (MAT_B_TOTAL_ROWS*MAT_B_TOTAL_COLS + blocksize_for_init - 1) 
													/ (blocksize_for_init);
	init_matrix<<<blocks_for_matA, blocksize_for_init>>>(mat_A, MAT_A_TOTAL_COLS, 
                                                            MAT_A_TOTAL_ROWS, 1);
	init_matrix<<<blocks_for_matB, blocksize_for_init>>>(mat_B, MAT_B_TOTAL_COLS, 
                                                            MAT_B_TOTAL_ROWS, 2);
	err = cudaGetLastError();
	if( err != cudaSuccess) {
		fprintf(stderr, "Failed to initialize matrix (error code %s)!\n", 
				cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	/* Do the matrix addition */
	size_t blocksizeX = TILE_WIDTH;
	size_t blocksizeY = TILE_WIDTH;

	dim3 DimGrid( (MAT_B_TOTAL_COLS-1)/blocksizeX + 1, (MAT_A_TOTAL_ROWS-1)/blocksizeY + 1);
	dim3 DimBlock( blocksizeX, blocksizeY);
	tiled_matrix_multiplication<<<DimGrid, DimBlock>>>(mat_A, mat_B, mat_C, 
        MAT_A_TOTAL_ROWS, MAT_A_TOTAL_COLS, MAT_B_TOTAL_COLS);
	err = cudaGetLastError();
	if( err != cudaSuccess) {
		fprintf(stderr, "Failed to perform matrix addition (error code %s)!\n", 
				cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();
	
	return EXIT_SUCCESS;
}