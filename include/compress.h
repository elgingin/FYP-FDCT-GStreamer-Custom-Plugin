#ifndef COMPRESS_H
#define COMPRESS_H

#include <cuda_runtime.h>
#include <cufft.h>

#ifdef __cplusplus
extern "C" {
#endif

// Function prototype for compressing a frame with thresholding
void compressWithThreshold(float* d_data, int width, int height, float threshold);

#ifdef __cplusplus
}
#endif

#endif /* COMPRESS_H */
