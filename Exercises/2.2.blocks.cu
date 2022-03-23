#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define N 10

void random_ints(int* a, int M)
{
    for (int i = 0; i < M; i++)
        a[i] = rand() % 10;
}

__global__ void mult(int *a, int *b, int *c)
{
    c[blockIdx.x] = a[blockIdx.x] * b[blockIdx.x];
}

int main()
{
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);

    // Allocate space for device copies of a, b, c​
    a = (int*)malloc(size); random_ints(a, N);
    b = (int*)malloc(size); random_ints(b, N);
    c = (int*)malloc(size);

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    mult<<<N,1>>>(d_a,d_b,d_c);

    // Copy result back to host​
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup​
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    for (int i = 0; i < N; i++)
        printf("[%d]: %d x %d = %d\n", i, a[i], b[i], c[i]);

    return 0;
}