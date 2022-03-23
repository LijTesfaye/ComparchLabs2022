#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define N 3
#define M 5

void random_ints(int* a, int F)
{
    for (int i = 0; i < F; i++)
        a[i] = rand() % 10;
}

__global__ void mult(int* a, int* b, int* c, int MAX)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < MAX)
        c[index] = a[index] * b[index];
}

int main()
{
    int* a, * b, * c;
    int* d_a, * d_b, * d_c;
    int size = M * N * sizeof(int);

    // Allocate space for device copies of a, b, c​
    a = (int*)malloc(size); random_ints(a, M * N);
    b = (int*)malloc(size); random_ints(b, M * N);
    c = (int*)malloc(size);

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    mult <<<N, M>>>(d_a, d_b, d_c, M * N);

    // Copy result back to host​
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup​
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    printf("c = \n");
    for (int i = 0; i < N * M; i++)
        printf("[%d]: %d x %d = %d\n", i, a[i], b[i], c[i]);

    return 0;
}