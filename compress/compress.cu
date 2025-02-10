#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "compress.h"

// Kernel to apply the threshold to the transformed data
__global__ void applyThreshold(float* d_data, int width, int height, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;
        // Apply threshold: set values below the threshold to zero
        if (fabs(d_data[index]) < threshold) {
            d_data[index] = 0.0f;
        }
    }
}

// Function to perform compression and apply the threshold
extern "C" void compressWithThreshold(float* d_data, int width, int height, float threshold) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Apply thresholding
    applyThreshold<<<gridSize, blockSize>>>(d_data, width, height, threshold);
    cudaDeviceSynchronize();
}
