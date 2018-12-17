/**************************************************************************
*
*
*
**************************************************************************/

#include <cuda.h>

#include <stdio.h>
#include "gpu_main.h"
#include "params.h"


/*************************************************************************/
int updatePalette(GPU_Palette* P){

  printf("made it here\n");
//  do something <<< P->gBlocks, P->gThreads >>> (P->red);

  return 0;
}

/*************************************************************************/
GPU_Palette initGPUPalette(AParams* PARAMS){

  // load
  GPU_Palette P;

  P.gTPB = 32;      // threads per block
  P.gDIM = 800;     // assumes the image is 800x800

  // 800x800 palette = 25x25 grid of 32x32 threadblocks
  P.gSize = P.gDIM * P.gDIM * sizeof(float);

  P.gThreads.x = P.gTPB;
  P.gThreads.y = P.gTPB;
  P.gThreads.z = 1;         // 3D of threads allowed
  P.gBlocks.x = P.gDIM/P.gTPB;
  P.gBlocks.y = P.gDIM/P.gTPB;
  P.gBlocks.z = 1;          // only 2D of blocks allowed

  // allocate memory for the palette
  cudaMalloc((void**) &P.gray, P.gSize);    // black and white (avg of rgb)
  cudaMalloc((void**) &P.red, P.gSize);   // r
  cudaMalloc((void**) &P.green, P.gSize); // g
  cudaMalloc((void**) &P.blue, P.gSize);  // b

  return P;
}


/*************************************************************************/
int freeGPUPalette(GPU_Palette* P) {

  return 0;
}

/*************************************************************************/
