// - Build
// nvcc -o "01 CUDA 기타/01 CUDA Mapped Memory"  "01 CUDA 기타/01 CUDA Mapped Memory.cu" --expt-relaxed-constexpr -lcurand -lcuda -lcudart -lcublas
// - Run
// "./01 CUDA 기타/01 CUDA Mapped Memory"

#include <iostream>
#include <cuda_runtime.h>

__global__ void square(int *d_data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        d_data[idx] = d_data[idx] * d_data[idx];
    }
}

int main() {
    const int N = 512;
    size_t size = N * sizeof(int);

    // Mapped Memory
    int *h_data;
    cudaHostAlloc((void**)&h_data, size, cudaHostAllocMapped);

    for (int i = 0; i < N; ++i)
        h_data[i] = i;

    // Device Memory Pointer
    int *d_data;
    cudaHostGetDevicePointer(&d_data, h_data, 0);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    square<<<numBlocks, blockSize>>>(d_data, N);

    for (int i = 0; i < N; ++i)
        std::cout << h_data[i] << " ";
    std::cout << std::endl;

    cudaFreeHost(h_data);
    return 0;
}