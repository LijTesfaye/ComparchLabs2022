#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

__global__
void threads_and_blocks(int *a, int *b,  int *c){
    int index=threadIdx.x+blockIdx.x*blockDim.x; // the index for a particular thread.
    /* threadId.x is the id of the thread
     * blockId.x is the id of that particular block
     * blockDim is the number of threads per a block, sometimes is referred to as M
     */
    c[index]=a[index]+b[index];
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
#define N   (4*4)
#define M   2
//
int main() {
    //these are the host copies
    int *a, *b, *c ;
    // these are the device copies
    int *d_a , *d_b , *d_c;
    //the size is required
    int size=N*sizeof(int);
    //allocate space for the device copies
    cudaMalloc((void **)&d_a,size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c,size);
    //allocate space for the host copies too
    a=(int *) malloc(size);
    random_ints(a,N);
    b=(int *) malloc(size);
    random_ints(b,N);
    c=(int *) malloc(size);
    //copy the variable to device
    cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,size,cudaMemcpyHostToDevice);
    //launch the kernel
    threads_and_blocks<<<N/M,M>>>(d_a,d_b,d_c);
    //copy the results back to the Host
    cudaMemcpy(c,d_c,size,cudaMemcpyDeviceToHost);
    //display the results
    display(a,b,c,N);
    //clean up
        //the Device variables
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
            // for the Host variables
            free(a);
            free(b);
            free(c);

    return 0;
}
