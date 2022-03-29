#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

/*
Each parallel invocation of add_vector() is referred to as a 'block'.
 a set of blocks is called a grid.
if there is a questions saying how this  could be a parallel program, look how the addition of each
 * block is done independently.
 * ******
 * Block 0
 c[0]=a[0]+b[0]
 *********
 Block 1
 c[1]=a[1]+b[1]
 *
 e.t.c
 */
__global__
void add_vector(int *a, int *b, int *c){
    c[blockIdx.x]=a[blockIdx.x]+b[blockIdx.x];
    // bear in mind the usage of blockIdx, next example we substitute it with
    //threadId
}
//
void random_ints(int *a, int n) {
    for (int i = 0; i < n; ++i)
        a[i] = rand() % 100 + 1;
}
//
void display(int *a, int *b, int *result, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d  + %d  =  %d\n", a[i], b[i], result[i]);
    }
}
//
//for the main
//
#define N 4
int main(void) {
    int *a , *b ,*c ; //host copies of a b c
    int *d_a, *d_b , *d_c; // device
    //size variable
    int size= N*sizeof(int);
    // allocate space for the device copies of a b c
    cudaMalloc((void **) &d_a , size);
    cudaMalloc((void **) &d_b , size);
    cudaMalloc((void **) &d_c , size);
    //
    //get input values from the user, this part is a bit different from the previous two examples
    a=(int *) malloc(size);
    random_ints(a,N);
    b=(int *) malloc(size);
    random_ints(b,N);
    c=(int *)malloc(size);

    //copy the values to the Device
    cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,size,cudaMemcpyHostToDevice);



    //call the kernel, N copies of add
    add_vector<<<N,1>>>(d_a,d_b,d_c); //N blocks with each one thread
    //copy the result to the Host
    cudaMemcpy(c,d_c,size, cudaMemcpyDeviceToHost);
    //display the results
    display(a,b,c,N);
    //cleanup
     //for cuda
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    //for c
    free(a);
    free(b);
    free(c);
    return 0;
}
