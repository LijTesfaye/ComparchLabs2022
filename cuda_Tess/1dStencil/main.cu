
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
//define vector length, stencil radius,
#define N (4*4)
#define RADIUS 3
#define GRIDSIZE 8 //N,M
#define BLOCKSIZE 32
//
int gridSize  = GRIDSIZE;
int blockSize = BLOCKSIZE;
/*
-------------------------------------------------------
CUDA device function that performs 1D stencil operation
-------------------------------------------------------
*/
__global__
void stencil_1D(int *in, int *out, int dim)
{
    __shared__ int temp[BLOCKSIZE + 2*RADIUS];

    int gindex = threadIdx.x + blockDim.x * blockIdx.x; //global index
    int stride = gridDim.x * blockDim.x;  //reason  on the slide.
    int tid = threadIdx.x;
    int lindex = threadIdx.x + RADIUS; //shared memory local index

    // Go through all data
    // Step all threads in a block to avoid synchronization problem
    while ( gindex < dim + blockDim.x)
    {
        if(gindex < dim)
        {
            temp[lindex] = in[gindex];
        }
        else
        {
            temp[lindex] = 0;
        }
        if(tid < RADIUS)
        {
            if(gindex < RADIUS)
            {
                temp[lindex - RADIUS] = 0;
            }
            else
            {
                temp[lindex - RADIUS] = in[gindex - RADIUS];
            }
            if(gindex + BLOCKSIZE >= dim)
            {
                temp[lindex + BLOCKSIZE] = 0;
            }
            else
            {
                temp[lindex + BLOCKSIZE] = in[gindex + BLOCKSIZE];
            }
        }
        __syncthreads(); // this is for the synchronization of the threads, avoi the race condition.

        // Apply the stencil
        int result = 0;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            if ( lindex + offset < dim && lindex + offset > -1)
                result += temp[lindex + offset];
        }
        // Store the result
        if (gindex < dim)
            out[gindex] = result;

        // Update global index and quit if we are done
        gindex += stride;
        __syncthreads(); // avoid the race condition
    }
}
//displays the results of the 1D stencil program
void display(int *a,int n) {
    for (int i = 0; i < n; i++) {
        printf("out is :%d", a[i]);
        printf("\n");
    }
}
//
// this function assigns a random integer to the input variable
void random_ints(int *a, int n) {
    for (int i = 0; i < n; ++i)
        a[i] = rand() % 100 + 1;
}
/*
------------
main program
------------
*/
int main(void) {

    int *in, *out;
    int *d_in, *d_out;
    int size = N * sizeof(int);

    //allocate a pace to the host memory
    in = (int *) malloc(size);
    random_ints(in, N);
    out = (int *) malloc(size);
    random_ints(out, N);

    // allocate device memory
    cudaMalloc((void **) &d_in, size);
    cudaMalloc((void **) &d_out, size);

    // copy input data to device
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

    //----------------------------------------------------------
    // CODE TO RUN AND TIME THE STENCIL KERNEL.
    //----------------------------------------------------------
    stencil_1D<<<gridSize, blockSize>>>(d_in, d_out, N);
    //----------------------------------------------------------

    //copy the result from the Device to the Host
    //cudaMemcpy(c,d_c,size,cudaMemcpyDeviceToHost);
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    //display the results
    display(out, N);
    //
    // deallocate device memory
    cudaFree(d_in);
    cudaFree(d_out);
    // deallocate host memory
    free(in);
    free(out);
    return 0;
}