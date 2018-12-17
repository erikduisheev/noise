#ifndef hInterfaceLib
#define hInterfaceLib

#include "params.h"
#include "gpu_main.h"

/*
class AParams {
public:
        
  int             verbosity;
  int             runMode;
  
  int             imageHeight;
  int             imageWidth;
};
*/

int openBMP(AParams *PARAMS, GPU_Palette* P1, char* fileName);
int runIt(GPU_Palette* P1, AParams* PARAMS);

int setDefaults(AParams *PARAMS);
int usage();
int viewParams(const AParams *PARAMS);
char crack(int argc, char** argv, char* flags, int ignore_unknowns);


#endif