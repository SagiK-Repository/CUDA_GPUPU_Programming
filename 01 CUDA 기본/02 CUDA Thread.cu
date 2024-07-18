// - Build
// nvcc -o "01 CUDA 기본/02 CUDA Thread" "01 CUDA 기본/02 CUDA Thread.cu" --expt-relaxed-constexpr -lcurand -lcuda -lcudart -lcublas
// - Run
// "./01 CUDA 기본/02 CUDA Thread"

#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel(void) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	printf("\nGPU_Thread_num[%d] = (%d, %d, %d)", i, blockIdx.x, blockDim.x, threadIdx.x);
}

int main() {
	dim3 dimBlock(2, 1, 1);
	kernel <<<3,dimBlock>>>();
	cudaDeviceSynchronize();
	printf("\ndimBlock.x = %d\n\n", dimBlock.x);

	dim3 dimBlock2(2, 2, 1);
	kernel <<<3,dimBlock2>>>();
	cudaDeviceSynchronize();
	printf("\ndimBlock.y = %d\n\n", dimBlock.y);
    
	dim3 dimBlock3(2, 2, 2);
	kernel <<<3,dimBlock3>>>();
	cudaDeviceSynchronize();
	printf("\ndimBlock.z = %d\n\n", dimBlock.z);
}