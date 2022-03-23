#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <time.h>
#include <cstdlib>

#define N 4*4		//number of blocks to be run
#define M 2
using namespace std;

//case of <<<1, 1>>> (1 block, with 1 thread)
__global__  void multiply1_1(int *a, int *b, int *result) {
    *result = *a * *b;
}


//case of <<<N, 1>>> (N blocks with 1 thread per block)
//using blockIdx.x to index into the passed array, we make each block handle
//a different element of the array
/*
	if N = 4 with a single thread for each block we will have

	Block 0 -> result[0] = a[0] + b[0]
	Block 1 -> result[1] = a[1] + b[1]
	Block 2 -> result[2] = a[2] + b[2]
	Block 3 -> result[3] = a[3] + b[3]
	executed in parallel on the GPU
*/
__global__ void multiplyN_1(int *a, int *b, int *result) {
    result[blockIdx.x] = a[blockIdx.x] * b[blockIdx.x];
}


/*

*/
__global__ void multiply(int *a, int *b, int *result, int n) {
    result[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
    //printf("%d x %d = %d\n", a[threadIdx.x], b[threadIdx.x], result[threadIdx.x]);
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) result[index] = a[index] * b[index];
}




void random_ints(int *a, int n) {
    for (int i = 0; i < n; ++i)
        a[i] = rand() % 100 + 1;
}

void display(int *a, int *b, int *result, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d  x  %d  =  %d\n", a[i], b[i], result[i]);
    }
}



int main(void) {
    int *a, *b, *result; // host copies of a, b, result
    int *d_a, *d_b, *d_result; // device copies of a, b, result
    int size = N * sizeof(int);


    // Allocate space for device copies of a, b, c
    /*
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_result, size);
    */

    // Alloc space for host copies of a, b, c and setup input values
    a = (int*)malloc(size);
    random_ints(a, N);
    b = (int*)malloc(size);
    random_ints(b, N);
    result = (int*)malloc(size);

    // Alloc space for device copies of a, b, c
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_result, size);


    //setup variable values
    /*
    scanf("%d", &a);
    scanf("%d", &b);
    */


    // Copy inputs from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);


    //multiply1_1 <<< 1, 1>>> (d_a, d_b, d_result);

    //Launches multiplyN_1 on GPU with N blocks
    multiply <<<(N + M - 1) / M, M>>> (d_a, d_b, d_result, N);



    // Copy result back to host
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);



    //showing the result
    //printf("%d x %d = %d\n", a, b, result);

    display(a, b, result, N);



    // Cleanup
    free(a);
    free(b);
    free(result);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    return 0;
}