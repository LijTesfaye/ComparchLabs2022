//for a cuda to run these the following two must be included.
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include <iostream> // for input output
//
__global__
void add_two_num(int *a, int *b, int *c){
    *c=*a + *b;
}
void disp(int a, int b,int c){
    printf("%d + %d= %d",a,b,c);
}
// still this code is not parallel, but it is better than the Helloworld example in the previous  one.
// In this part ypu will understand the procedures to
//1. allocate space for the Device variables
//1.0 Specify the size tha type that you want e.g. size of int is used in this example;
//2. copy from  host
//3. copy from devices
//4.launch the kernel
//5. clean up the memory space allocated in step 1
//
int main() {
    //host copies of the numbers
    int a, b,c;
    //the copies of the Device
    int *d_a,*d_b,*d_c;
    //size specification
    int size=sizeof (int );
    //allocate space for device copies
    cudaMalloc((void **) &d_a, size );
    cudaMalloc((void **) &d_b, size);
    cudaMalloc((void **) &d_c, size);

    //accept the values from the user
    printf("Enter the values of a: ");
    scanf("%d",&a);
    printf("Enter the values of b: ");
    //strtol('%d',&b);
    scanf("%d",&b);

    //copy the values to the Device
    cudaMemcpy(d_a,&a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,&b,size,cudaMemcpyHostToDevice);

    //launch the kernel
    add_two_num<<<1,1>>>(d_a,d_b,d_c); //it is executing the kernel 1 times.
    //In the upcoming exercise we will see the vector addition for executing the kernel multiple times.
    //copy the result to the Host
    cudaMemcpy(&c,d_c,size,cudaMemcpyDeviceToHost);
    //call the display function to show the results
    disp(a,b,c);
    //cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
