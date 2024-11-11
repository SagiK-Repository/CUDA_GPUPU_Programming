// - Build
// nvcc -o "01 CUDA 기타/06 CUDA Layerd Texture"  "01 CUDA 기타/06 CUDA Layerd Texture.cu" --expt-relaxed-constexpr -lcurand -lcuda -lcudart -lcublas
// - Run
// "./01 CUDA 기타/06 CUDA Layerd Texture"

#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

texture<float, cudaTextureType2DArray, cudaReadModeElementType> texRef;

__global__ void processLayeredTexture(float *output, int width, int height, int layer) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float pixelValue = tex2DLayered(texRef, x, y, layer);  // read pixel from texture memory
        output[y * width + x] = pixelValue * 2.0f; // pixel * 2
    }
}

int main() {
    const int width = 512;
    const int height = 512;
    const int layers = 3;
    size_t size = width * height * sizeof(float);

    float *h_layers = (float*)malloc(size * layers);
    for (int layer = 0; layer < layers; ++layer) {
        for (int i = 0; i < width * height; ++i) {
            h_layers[layer * width * height + i] = static_cast<float>(i % 256) + layer; // 레이어별로 다른 패턴 생성
        }
    }

    float *d_output;
    cudaMalloc((void**)&d_output, size);

    // Set 2D Array Texture Memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray *d_textureArray;
    cudaMallocArray(&d_textureArray, &channelDesc, width, height, layers);

    // Copy Host Memory to Texture Array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(h_layers, width * sizeof(float), width, height * layers);
    copyParams.dstArray = d_textureArray;
    copyParams.extent = make_cudaExtent(width, height, layers);
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    // Binding
    cudaBindTextureToArray(texRef, d_textureArray);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    processLayeredTexture<<<gridSize, blockSize>>>(d_output, width, height, 0); // First Layer
    cudaDeviceSynchronize();

    float *h_output = (float*)malloc(size);
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    std::cout << "Processed image from layer 0 (first 10 pixels): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_output);
    cudaFreeArray(d_textureArray);
    free(h_layers);
    free(h_output);

    return 0;
}
