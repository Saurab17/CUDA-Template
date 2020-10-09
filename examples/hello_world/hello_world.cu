// Copyright (c) 2020 Saurabh Yadav
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ---------------------------------------------------
   My Hello world for CUDA programming
   --------------------------------------------------- */

#include <stdio.h>        
#include <unistd.h>  
     
#include <cuda_runtime.h>

/* ------------------------------------
   Kernel (GPU function)
   ------------------------------------ */
__global__ void hello(void)
{
   printf("Hello From the GPU !\n");
}

int main()
{
   /* ------------------------------------
      Call to the hello( ) kernel function
      ------------------------------------ */
    hello<<<1, 1>>>();

    printf("Hello From the CPU ! \n");
    
    cudaDeviceSynchronize();
    return 0;
}
