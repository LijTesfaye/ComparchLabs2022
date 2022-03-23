#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void mult(int* a, int* b, int* c)
{
    *c = *a * *b;
}

int main()
{
    int a, b, c;
    int* d_a, * d_b, * d_c;
    int size = sizeof(int);

    // Allocate space for device copies of a, b, c​
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    a = 10;
    b = 3;

    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    mult << <1, 1 >> > (d_a, d_b, d_c);

    // Copy result back to host​
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    printf("c = a x b = 10 x 3 = %d", c);

    // Cleanup​
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}