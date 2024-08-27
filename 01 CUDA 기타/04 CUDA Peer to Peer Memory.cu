// - Build
// nvcc -o "01 CUDA 기타/04 CUDA Peer to Peer Memory"  "01 CUDA 기타/04 CUDA Peer to Peer Memory.cu" --expt-relaxed-constexpr -lcurand -lcuda -lcudart -lcublas
// - Run
// "./01 CUDA 기타/04 CUDA Peer to Peer Memory"

#include <iostream>
#include <cuda_runtime.h>

__global__ void initializeArray(int *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        data[idx] = idx;
    }
}

void printArray(int *data, int N) {
    for (int i = 0; i < N; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount < 2) {
        std::cerr << "At least two CUDA devices are required for this example." << std::endl;
        return -1;
    }

    // GPU 0
    int N = 512;
    size_t size = N * sizeof(int);
    int *d_data0, *d_data1;

    // GPU 0
    cudaSetDevice(0);
    cudaMalloc((void**)&d_data0, size);
    
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    initializeArray<<<numBlocks, blockSize>>>(d_data0, N);
    cudaDeviceSynchronize();

    // GPU 1
    cudaSetDevice(1);
    cudaMalloc((void**)&d_data1, size);

    // P2P Memory
    cudaDeviceEnablePeerAccess(0, 0); // GPU 1 Access to GPU 0
    cudaDeviceEnablePeerAccess(1, 0); // GPU 0 Access to GPU 1

    // GPU 0, Copy to GPU 1
    cudaSetDevice(0);
    cudaMemcpyPeer(d_data1, 1, d_data0, 0, size); // Copy GPU 0 d_data0 -> GPU 1 d_data1

    cudaSetDevice(1);
    int *h_data1 = (int*)malloc(size);
    cudaMemcpy(h_data1, d_data1, size, cudaMemcpyDeviceToHost);
    
    std::cout << "Data in GPU 1 after copying from GPU 0: ";
    printArray(h_data1, N);

    cudaSetDevice(0);
    cudaFree(d_data0);
    cudaSetDevice(1);
    cudaFree(d_data1);
    free(h_data1);

    return 0;
}