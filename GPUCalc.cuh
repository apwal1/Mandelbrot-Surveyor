#pragma once
#include <cuda.h>
#include <cuComplex.h>
#include "fracState.h"

__device__
void coordsToComplexGPU(const int* x, const int* y, const fracState* state, cuDoubleComplex* result);
__device__
void getNumItersGPU(const cuDoubleComplex* complexNum, int* iters);
__global__
void makeFracGPU(int* resultArr, const fracState* state);