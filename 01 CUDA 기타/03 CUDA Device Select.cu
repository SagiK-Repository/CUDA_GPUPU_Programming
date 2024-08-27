// - Build
// nvcc -o "01 CUDA 기타/03 CUDA Device Select"  "01 CUDA 기타/03 CUDA Device Select.cu" --expt-relaxed-constexpr -lcurand -lcuda -lcudart -lcublas
// - Run
// "./01 CUDA 기타/03 CUDA Device Select"

#include <iostream>
#include <cuda_runtime.h>


__global__ void square(int *d_data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        d_data[idx] = d_data[idx] * d_data[idx];
    }
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices available." << std::endl;
        return -1;
    }
    
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    int device = 0;
    cudaSetDevice(device);

    // Device Info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Using Device: " << prop.name << std::endl;

    const int N = 512;
    size_t size = N * sizeof(int);

    int *h_data = (int*)malloc(size);
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;
    }

    int *d_data;
    cudaMalloc((void**)&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    square<<<numBlocks, blockSize>>>(d_data, N);

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_data);
    free(h_data);

    return 0;
}