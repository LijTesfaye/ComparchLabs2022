/* holds the cuda memory management functions like cudaMemcpy,
 cudaFree,cudaMemcpyHostToDevice,cudaMemcpyDeviceToHost
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h" // helps to use threadId and blockId
//
#include <iostream>

__global__
void intro_threads(int *a,int *b, int *c) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}
//
void random_ints(int *a, int n) {
    for (int i = 0; i < n; ++i)
        a[i] = rand() % 100 + 1;
}
//
void display(int *a, int *b, int *result, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d  +  %d  =  %d\n", a[i], b[i], result[i]);
    }
}
//
#define N 4
#define M 6
int main() {
    // the Host copies
    int *a , *b , *c ;
    // the Device copies
    int *d_a , *d_b , *d_c ;
    //size
    int size=N* sizeof(int);

    //allocating space for the Device copies
    cudaMalloc((void **) &d_a,size);
    cudaMalloc((void **) &d_b,size);
    cudaMalloc((void **)&d_c,size);
    //allocate space for host copies of the variables, we dont accept values from a user
    //rather we randomly passing a number to each of them,
    a=(int *) malloc(size);
    random_ints(a,N);
    b=(int*) malloc(size);
    random_ints(b,N);
    c=(int *) malloc(size);


    //coping  to Device
    cudaMemcpy(d_a , a , size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_b , b , size , cudaMemcpyHostToDevice);
    // launch the kernel
    intro_threads<<<1, N>>>(d_a, d_b, d_c ); // N threads for a single  block

    //coping results back to Host
    cudaMemcpy(c,d_c,size,cudaMemcpyDeviceToHost);
    //display the results
    display(a,b,c,N);

    //time to the cleanup stuff
        //for the Device variables
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    //for the Host variables
    free(a);
    free(b);
    free(c);
    return 0;
}
