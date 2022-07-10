
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "audio.cuh"


int main()
{
    printf("%d\n", test(1000000, 1));

    return 0;
}
