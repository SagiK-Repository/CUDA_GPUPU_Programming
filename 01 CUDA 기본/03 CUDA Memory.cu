// - Build
// nvcc -o "01 CUDA 기본/03 CUDA Memory" "01 CUDA 기본/03 CUDA Memory.cu" --expt-relaxed-constexpr -lcurand -lcuda -lcudart -lcublas
// - Run
// "./01 CUDA 기본/03 CUDA Memory"

#include<stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void kernel(int* a) {
	*a+=5;
}

int main() {
    int host = 10;
	int* device;

	cudaMalloc((void**)&device, sizeof(int));
	cudaMemcpy(device, &host, sizeof(int), cudaMemcpyHostToDevice);
	
	printf("\nbefore host : %d\n", host);

	kernel <<<1,1>>> (device);
	cudaDeviceSynchronize();

	cudaMemcpy(&host, device, sizeof(int), cudaMemcpyDeviceToHost);

	printf("\nafter host : %d\n", host);

	cudaFree(device);
}