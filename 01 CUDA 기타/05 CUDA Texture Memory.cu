// - Build
// nvcc -o "01 CUDA 기타/05 CUDA Texture Memory"  "01 CUDA 기타/05 CUDA Texture Memory.cu" --expt-relaxed-constexpr -lcurand -lcuda -lcudart -lcublas
// - Run
// "./01 CUDA 기타/05 CUDA Texture Memory"

#include <iostream>
#include <cuda_runtime.h>

cudaTextureObject_t texObj; // 텍스처 객체 선언

__global__ void processImage(float *output, int width, int height, cudaTextureObject_t texObj) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float pixelValue = tex2D<float>(texObj, x, y); // read pixel from texture memory
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

    // 2D 배열로 데이터 복사
    cudaMemcpy2DToArray(d_textureArray, 0, 0, h_image, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);

    // 텍스처 객체 생성
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_textureArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    processImage<<<gridSize, blockSize>>>(d_output, width, height, texObj);
    cudaDeviceSynchronize();

    float *h_output = (float*)malloc(size);
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    std::cout << "Processed image (first 10 pixels): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    cudaDestroyTextureObject(texObj);
    cudaFree(d_output);
    cudaFreeArray(d_textureArray);
    free(h_image);
    free(h_output);

    return 0;
}
