#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

extern "C" void decompressFrame(cufftComplex* input, float* output, int width, int height) {
    cufftHandle plan;
    size_t size = width * height * sizeof(float);
    
    cufftComplex* d_input;
    float* d_output;
    cudaMalloc((void**)&d_input, width * height * sizeof(cufftComplex));
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, input, width * height * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    cufftPlan2d(&plan, height, width, CUFFT_C2R);
    cufftExecC2R(plan, d_input, d_output);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_input);
    cudaFree(d_output);
}
