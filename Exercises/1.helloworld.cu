#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void cuda_hello()
{
    printf("Hello world!");
}

int main()
{
    
    cuda_hello<<<1,1>>>();

    return 0;
}
