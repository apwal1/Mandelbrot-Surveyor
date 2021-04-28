#include "GPUCalc.cuh"

//Converts a pixel's coordinates to a complex number, which will be stored in result
__device__
void coordsToComplexGPU(const int* x, const int* y, const fracState* state, cuDoubleComplex* result)
{
    result->x = ((double)*x + state->xPanOffset) / state->xZoomScale;
    result->y = ((double)*y + state->yPanOffset) / state->yZoomScale;
}

/*Calculates the number of iterations required to determine whether the passed complex number is
in the mandelbrot set or not. The result will be placed in the passed int* iters*/
__device__
void getNumItersGPU(const cuDoubleComplex* complexNum, int* iters)
{
    cuDoubleComplex z = make_cuDoubleComplex(0, 0);
    /*(z.imag() * z.imag()) + (z.real() * z.real()) <= 4 is equivalent to abs(z) <= 2
    While the former may look more complicated, it is about twice as efficient as the latter*/
    for (*iters = 0; (z.y * z.y) + (z.x * z.x) <= 4 && *iters < MAX_ITER; (*iters)++)
    {
        z = cuCmul(z, z);
        z = cuCadd(z, *complexNum);
    }
}

//Calculates results for a section of the fractal
__global__
void makeFracGPU(int* resultArr, const fracState* state)
{
    cuDoubleComplex complexPixel = make_cuDoubleComplex(0, 0);

    int itrs;
    int sectionHeight = WINDOW_HEIGHT / gridDim.x;
    int sectionWidth = WINDOW_WIDTH / blockDim.x;
    for (int y = blockIdx.x * sectionHeight; y < (blockIdx.x + 1) * sectionHeight; y++)
    {
        for (int x = threadIdx.x * sectionWidth; x < (threadIdx.x + 1) * sectionWidth; x++)
        {
            coordsToComplexGPU(&x, &y, state, &complexPixel);
            getNumItersGPU(&complexPixel, &itrs);
            resultArr[y * WINDOW_WIDTH + x] = itrs;
        }
    }
    return;
}