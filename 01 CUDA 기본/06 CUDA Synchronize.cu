// - Build
// nvcc -o "01 CUDA 기본/06 CUDA Synchronize"  "01 CUDA 기본/06 CUDA Synchronize.cu" --expt-relaxed-constexpr -lcurand -lcuda -lcudart -lcublas
// - Run
// "./01 CUDA 기본/06 CUDA Synchronize"

#include<stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void kernel() {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int a;
	int b=0;
	a+= i; b+= i;
	printf("S[%d][%d] : (%d, %d)\n", blockIdx.x, threadIdx.x, a, b);
	__syncthreads();
	printf("E[%d][%d] : (%d, %d)\n", blockIdx.x, threadIdx.x, a, b);
}

int main() {
	kernel <<<3,3>>> ();
	cudaDeviceSynchronize();
}