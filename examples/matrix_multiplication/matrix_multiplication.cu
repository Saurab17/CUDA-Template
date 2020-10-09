// Copyright (c) 2020 Saurabh Yadav
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define MAT_A_TOTAL_ROWS      4U
#define MAT_A_TOTAL_COLS      5U
#define MAT_B_TOTAL_ROWS      MAT_A_TOTAL_COLS
#define MAT_B_TOTAL_COLS      6U


__global__
void init_matrix(float *matrix, int width, int height, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < width * height; i += gridDim.x * blockDim.x) {
        matrix[i]=val;
    }
}

__global__
void multiply_matrices(float * mat_A_arr, float * mat_B_arr, float * mat_C_arr,
                  int num_A_rows, int num_A_cols, int num_B_cols) {

	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(row < num_A_rows && col < num_B_cols) {
        float value = 0.0;
        for(int i=0; i<num_A_cols; i++) {
            value += mat_A_arr[row*num_A_cols+i] * mat_B_arr[col + i*num_B_cols];
        }
        mat_C_arr[row*num_B_cols + col] = value;
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
	size_t blocksizeX = 16;
	size_t blocksizeY = 16;

	dim3 DimGrid( (MAT_B_TOTAL_COLS-1)/blocksizeX + 1, (MAT_A_TOTAL_ROWS-1)/blocksizeY + 1);
	dim3 DimBlock( blocksizeX, blocksizeY);
	multiply_matrices<<<DimGrid, DimBlock>>>(mat_A, mat_B, mat_C, 
        MAT_A_TOTAL_ROWS, MAT_A_TOTAL_COLS, MAT_B_TOTAL_COLS);
	err = cudaGetLastError();
	if( err != cudaSuccess) {
		fprintf(stderr, "Failed to perform matrix addition (error code %s)!\n", 
				cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();
	
    //Print the matrices for visualization
    printf("\nMatrix A: \n");
    for(int row=0; row<MAT_A_TOTAL_ROWS; row++) {
        for(int col=0; col<MAT_A_TOTAL_COLS; col++) {
            printf("%f ",mat_A[row*MAT_A_TOTAL_COLS + col]);
        }
        printf("\n");
    }
    printf("\nMatrix B: \n");
    for(int row=0; row<MAT_B_TOTAL_ROWS; row++) {
        for(int col=0; col<MAT_B_TOTAL_COLS; col++) {
            printf("%f ",mat_B[row*MAT_B_TOTAL_COLS + col]);
        }
        printf("\n");
    }
    printf("\nMatrix C: \n");
    for(int row=0; row<MAT_A_TOTAL_ROWS; row++) {
        for(int col=0; col<MAT_B_TOTAL_COLS; col++) {
            printf("%f ",mat_C[row*MAT_B_TOTAL_COLS + col]);
        }
        printf("\n");
    }
	return EXIT_SUCCESS;
}