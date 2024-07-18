// - Build
// nvcc -o "01 CUDA 기본/04 CUDA ErrorCheck" "01 CUDA 기본/04 CUDA ErrorCheck.cu" --expt-relaxed-constexpr -lcurand -lcuda -lcudart -lcublas
// - Run
// "./01 CUDA 기본/04 CUDA ErrorCheck"

#include<stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_CHECK(val) { \
    if (val != cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(val), __LINE__, __FILE__); \
        exit(1); \
    } \
}

__global__ void kernel(int* a) {
	*a+=5;
}

int main() {
    int host[3] = {10};
	int* device;

	CUDA_CHECK(cudaMalloc((void**)&device, sizeof(int) * 10000000000000));

	cudaFree(device);
}