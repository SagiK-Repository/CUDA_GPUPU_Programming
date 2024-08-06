// - Build
// nvcc -o "01 CUDA 기본/08 CUDA Pinned Memory"  "01 CUDA 기본/08 CUDA Pinned Memory.cu" --expt-relaxed-constexpr -lcurand -lcuda -lcudart -lcublas
// - Run
// "./01 CUDA 기본/08 CUDA Pinned Memory"

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
	
    //TimeCheck
	cudaEvent_t start, stop, start0, stop0;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventCreate(&start0));
	CUDA_CHECK(cudaEventCreate(&stop0));
	
	int* device;
	
	CUDA_CHECK(cudaEventRecord(start));
	
	CUDA_CHECK(cudaMalloc((void**)&device, sizeof(int)));
	CUDA_CHECK(cudaMemcpy(device, &host, sizeof(int), cudaMemcpyHostToDevice));
	printf("before host : %d\n", host);
	kernel <<<2, 2>>> (device);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaMemcpy(&host, device, sizeof(int), cudaMemcpyDeviceToHost));
	printf("after host : %d\n", host);
	
	CUDA_CHECK(cudaEventRecord(stop));

	float msec_time = 0;
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaEventElapsedTime(&msec_time, start, stop));
	printf("\nTime = %.3f ms\n\n\n", msec_time);

	// Pinned
	int* hosts;
	CUDA_CHECK(cudaEventRecord(start0));
	
	CUDA_CHECK(cudaMallocHost((void**)&hosts, sizeof(int))); //Host Pinned

	hosts = &host;
	CUDA_CHECK(cudaMalloc((void**)&device, sizeof(int)));
	CUDA_CHECK(cudaMemcpy(device, hosts, sizeof(int), cudaMemcpyHostToDevice));
	printf("before host : %d\n", *hosts);
	kernel <<<2, 2>>> (device);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaMemcpy(hosts, device, sizeof(int), cudaMemcpyDeviceToHost));
	printf("after host : %d\n", *hosts);

	CUDA_CHECK(cudaEventRecord(stop0));

	msec_time = 0;
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaEventElapsedTime(&msec_time, start0, stop0));
	printf("\nTime = %.3f ms\n", msec_time);

	CUDA_CHECK(cudaFree(device));
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	CUDA_CHECK(cudaEventDestroy(start0));
	CUDA_CHECK(cudaEventDestroy(stop0));

    return 0;
}