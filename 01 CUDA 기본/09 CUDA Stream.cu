// - Build
// nvcc -o "01 CUDA 기본/09 CUDA Stream"  "01 CUDA 기본/09 CUDA Stream.cu" --expt-relaxed-constexpr -lcurand -lcuda -lcudart -lcublas
// - Run
// "./01 CUDA 기본/09 CUDA Stream"

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void processDataKernel(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i] = data[i] * data[i];
    }
}

void processWithoutStreams(float* h_data, int N) {
    float *d_data;

    cudaMalloc((void**)&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    processDataKernel<<<numBlocks, blockSize>>>(d_data, N);

    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
}

void processWithStreams(float* h_data, int N) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float *d_data;

    cudaMalloc((void**)&d_data, N * sizeof(float));
    cudaMemcpyAsync(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice, stream); // Async

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    processDataKernel<<<numBlocks, blockSize, 0, stream>>>(d_data, N);

    cudaMemcpyAsync(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost, stream); // Async

    cudaStreamSynchronize(stream);

    cudaFree(d_data);
    cudaStreamDestroy(stream);
}

int main() {
    const int N = 1 << 20; // 1M elements
    float *h_data;

    h_data = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i)
        h_data[i] = static_cast<float>(i); // initial

    // without stream
    auto start = std::chrono::high_resolution_clock::now();
    processWithoutStreams(h_data, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_without_stream = end - start;

    // with stream
    start = std::chrono::high_resolution_clock::now();
    processWithStreams(h_data, N);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_with_stream = end - start;

    std::cout << "without stream: " << duration_without_stream.count() * 1000 << "ms" << std::endl;
    std::cout << "with stream: " << duration_with_stream.count() * 1000 << "ms" << std::endl;

    for (int i = 0; i < 10; ++i)
        std::cout << h_data[i] << " ";
    std::cout << std::endl;

    free(h_data);

    return 0;
}