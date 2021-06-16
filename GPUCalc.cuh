#pragma once
#include <cuda.h>
#include <cuComplex.h>
#include "fracState.h"

__device__
void coordsToComplexGPU(const int* x, const int* y, const fracState* state, cuDoubleComplex* result);
__device__
void getNumItersGPU(const cuDoubleComplex* complexNum, double* smooth);
__device__
void calcSmoothColorGPU(const cuDoubleComplex* complexNum, const int* iters, double* smooth);
__global__
void makeFracGPU(RGB* resultArr, const fracState* state);