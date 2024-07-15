// - Build
// nvcc -o "CUDA 기본/CudaNormal" "CUDA 기본/CudaNormal.cu" --expt-relaxed-constexpr -lcurand -lcuda -lcudart -lcublas
// - Run
// "./CUDA 기본/CudaNormal"

#include <cuda_runtime.h>
#include <iostream>

__global__ void Kernel(void) {
	printf("Hello, GPU World!\n");
}

int main() {
    Kernel<<<1, 4>>>();
	cudaDeviceSynchronize();
}