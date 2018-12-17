/**************************************************************************
*
*
*
**************************************************************************/

#include <cuda.h>

#include <stdio.h>
#include "gpu_main.h"
#include "params.h"

texture<float, 2>  blueTex; // used for 2D grid for task 1

/*************************************************************************/
int updatePalette(GPU_Palette* P, AParams* PARAMS){
  if (PARAMS->runMode == 1 || PARAMS->runMode == 2)
    updateBlues <<< P->gBlocks, P->gThreads >>> (P->blue);
  if (PARAMS->runMode == 1 || PARAMS->runMode == 3)
    updateReds <<< P->gBlocks, P->gThreads >>> (P->red, P->theRand);
  if (PARAMS->runMode == 1 || PARAMS->runMode == 4) {
    float value = sin(clock()/PARAMS->sinusoidal); // used for oscilation for task 3
    updateGreens <<< P->gBlocks, P->gThreads >>> (P->green, value);
  }

  return 0;
}

/*************************************************************************/
__global__ void setup_kernel(curandState* state, unsigned long seed){
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);
  curand_init(seed, vecIdx, 0, &state[vecIdx]);
}

/*************************************************************************/
__global__ void updateReds(float* red, curandState* theRand){
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);
  curandState localState = theRand[vecIdx];
  float goo = curand_uniform(&localState);
  theRand[vecIdx] = localState;
  red[vecIdx] = (goo + red[vecIdx])/2; // get an average (task 2)
}

/*************************************************************************/
__global__ void updateBlues(float* blue){
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  float blue_sum = 0;
  int blue_count = 0;

  // get surrounding values
  for (int dx = -5; dx <= 6; dx++) {
    for(int dy = -5; dy <= 6; dy++) {
      blue_sum += tex2D(blueTex,x+dx,y+dy);
      blue_count += 1;
    }
  }

  float goo = blue_sum / blue_count; // the average value of surroundings
  blue[vecIdx] = goo;
}

/*************************************************************************/
__global__ void updateGreens(float* green, float value){
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  green[vecIdx] = value;
}

/*************************************************************************/
GPU_Palette initGPUPalette(AParams* PARAMS){

  // load
  GPU_Palette P;

  P.gTPB = 32;      // threads per block
  P.hDIM = PARAMS->height;     // assumes the image is 800x800
  P.wDIM = PARAMS->width;

  printf("PARAMS->height = %d\n", PARAMS->height);
  printf("PARAMS->width = %d\n", PARAMS->width);
  printf("PARAMS->size = %d\n", PARAMS->size);

  // 800x800 palette = 25x25 grid of 32x32 threadblocks
  P.gSize = P.wDIM * P.hDIM * sizeof(float);

  P.gThreads.x = P.gTPB;
  P.gThreads.y = P.gTPB;
  P.gThreads.z = 1;         // 3D of threads allowed
  P.gBlocks.x = P.wDIM/P.gTPB;
  P.gBlocks.y = P.hDIM/P.gTPB;
  P.gBlocks.z = 1;          // only 2D of blocks allowed

  // allocate memory for the palette
  cudaMalloc((void**) &P.gray, P.gSize);    // black and white (avg of rgb)
  cudaMalloc((void**) &P.red, P.gSize);   // r
  cudaMalloc((void**) &P.green, P.gSize); // g
  cudaMalloc((void**) &P.greenMax, P.gSize); // green Amplitude
  cudaMalloc((void**) &P.blue, P.gSize);  // b
  cudaMalloc((void**) &P.theRand, P.hDIM * P.wDIM * sizeof(curandState));
  setup_kernel <<< P.gBlocks, P.gThreads >>> (P.theRand, time(NULL));

  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  cudaBindTexture2D( NULL, blueTex, P.blue, desc, P.wDIM, P.hDIM, sizeof(float) * P.wDIM );

  return P;
}


/*************************************************************************/
int freeGPUPalette(GPU_Palette* P) {
  cudaUnbindTexture( blueTex );
  return 0;
}

/*************************************************************************/
