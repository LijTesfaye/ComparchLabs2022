#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>
//
// Learn these names by heart.
// the device is the name for the GPU
// the Host is the name for the CPU
//the __global__ command is a manifestation od a cuda C++ in your code. Hence, the code runs in the Device( i.e the GPU)
//
__global__
void hello_world(void ){
    printf("This is Hello world from Kernel");

}
// The saddest thing is that the Kernel is going to do nothing, which is annoying for real.But in the next part
// i.e adding numbers will be a bit better.
int main(void) {
    hello_world<<<1,1>>>(); // This is the kernel call.
    printf("This is Hello World from Main ! \n"); // this part of the code is displayed on the screen
    return 0;
}
