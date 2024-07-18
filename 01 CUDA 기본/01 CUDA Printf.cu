// - Build
// nvcc -o "01 CUDA 기본/01 CUDA Printf" "01 CUDA 기본/01 CUDA Printf.cu" --expt-relaxed-constexpr -lcurand -lcuda -lcudart -lcublas
// - Run
// "./01 CUDA 기본/01 CUDA Printf"

#include <cuda_runtime.h>
#include <iostream>

__global__ void Kernel(void) {
	printf("Hello, GPU World!\n");
}

int main() {
    Kernel<<<1, 4>>>();
	cudaDeviceSynchronize();
}