#ifndef DECOMPRESS_H
#define DECOMPRESS_H

#include <cuda_runtime.h>
#include <cufft.h>

// Function prototype for decompressing the frame
extern "C" void decompressFrame(cufftComplex* input, float* output, int width, int height);

#endif // DECOMPRESS_H
