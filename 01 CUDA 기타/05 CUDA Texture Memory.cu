// - Build
// nvcc -o "01 CUDA 기타/05 CUDA Texture Memory"  "01 CUDA 기타/05 CUDA Texture Memory.cu" --expt-relaxed-constexpr -lcurand -lcuda -lcudart -lcublas
// - Run
// "./01 CUDA 기타/05 CUDA Texture Memory"

#include <iostream>
#include <cuda_runtime.h>

texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

__global__ void processImage(float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float pixelValue = tex2D(texRef, x, y); // read pixel from texture memory
        output[y * width + x] = pixelValue * 2.0f; // pixel * 2
    }
}

int main() {
    const int width = 512;
    const int height = 512;
    size_t size = width * height * sizeof(float);

    float *h_image = (float*)malloc(size);
    for (int i = 0; i < width * height; ++i) {
        h_image[i] = static_cast<float>(i % 256); // sample pattern
    }

    float *d_output;
    cudaMalloc((void**)&d_output, size);

    // Set 2D Array Texture Memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray *d_textureArray;
    cudaMallocArray(&d_textureArray, &channelDesc, width, height);
    cudaMemcpyToArray(d_textureArray, 0, 0, h_image, size, cudaMemcpyHostToDevice);

    // Binding
    cudaBindTextureToArray(texRef, d_textureArray);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    processImage<<<gridSize, blockSize>>>(d_output, width, height);
    cudaDeviceSynchronize();

    float *h_output = (float*)malloc(size);
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    std::cout << "Processed image (first 10 pixels): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_output);
    cudaFreeArray(d_textureArray);
    free(h_image);
    free(h_output);

    return 0;
}