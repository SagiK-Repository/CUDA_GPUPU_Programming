// - Build
// nvcc -o "01 CUDA 기본/05 CUDA Elapsed" "01 CUDA 기본/05 CUDA Elapsed.cu" --expt-relaxed-constexpr -lcurand -lcuda -lcudart -lcublas
// - Run
// "./01 CUDA 기본/05 CUDA Elapsed"

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
    int host = 10;
	int* device;

	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	CUDA_CHECK(cudaMalloc((void**)&device, sizeof(int)));
	cudaMemcpy(device, &host, sizeof(int), cudaMemcpyHostToDevice);
	
	printf("\nbefore host : %d\n", host);

	CUDA_CHECK(cudaEventRecord(start));
	kernel <<<2,2>>> (device);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaEventRecord(stop));

	cudaMemcpy(&host, device, sizeof(int), cudaMemcpyDeviceToHost);

	float msec_time = 0;
	CUDA_CHECK(cudaEventElapsedTime(&msec_time, start, stop));
	printf("\nTime = %.3fms\n", msec_time);

	printf("\nafter host : %d\n", host);

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	cudaFree(device);
}