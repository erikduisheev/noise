#ifndef GPULib
#define GPULib

#include <cuda.h>
#include <curand.h>                 // includes random num stuff
#include <curand_kernel.h>          // more rand stuff
#include <cuda_texture_types.h>

#include "params.h"

#define cH2D            cudaMemcpyHostToDevice
#define cD2D            cudaMemcpyDeviceToDevice
#define cD2H            cudaMemcpyDeviceToHost

struct GPU_Palette{

    dim3 gThreads;  // threads (TPB, TPB, 1)
    dim3 gBlocks;   // blocks (DIM/TPB, DIM/TPB, 1)

    int gTPB;       // threads per block
    int hDIM;       // height of image (square)
    int wDIM;       // width of image (square)

    unsigned long gSize;      // size of vector of data for image
    float* gray;              // grayscale data
    float* red;
    float* green;
    float* greenMax;
    float* blue;

    curandState* theRand; // random numbers
};

int updatePalette(GPU_Palette* P, AParams* PARAMS);
GPU_Palette initGPUPalette(AParams* PARAMS);
int freeGPUPalette(GPU_Palette* P1);

// kernel calls:
__global__ void setup_kernel(curandState*, unsigned long);
__global__ void updateReds(float*, curandState*);
__global__ void updateGreens(float* green, float);
__global__ void updateBlues(float* blue);


#endif  // GPULib
