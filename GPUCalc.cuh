#pragma once
#include <cuda.h>
#include <cuComplex.h>
#include "fracState.hpp"

__device__
void coordsToComplexGPU(const int* x, const int* y, const fracState* state, cuDoubleComplex& result);
__device__
void getSmoothColorGPU(const cuDoubleComplex* complexNum, double& smooth, int maxIters);
__device__
void calcSmoothColorGPU(const cuDoubleComplex* complexNum, const int* iters, double& smooth);
__global__
void makeFracGPU(RGB* resultArr, const fracState* state);