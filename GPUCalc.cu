#include "GPUCalc.cuh"
#include "hsvrgb_GPU.cuh"

//Converts a pixel's coordinates to a complex number, which will be stored in result
__device__
void coordsToComplexGPU(const int* x, const int* y, const fracState* state, cuDoubleComplex* result)
{
    result->x = ((double)*x + state->xPanOffset) / state->xZoomScale;
    result->y = ((double)*y + state->yPanOffset) / state->yZoomScale;
}

/*Calculates the number of iterations required to determine whether the passed complex number is 
in the mandelbrot set or not and calculates a smooth color based on the number of iterations. 
The result will be placed in the passed double* smooth and will be -1.0 if the pixel is in the 
mandelbrot set and should be colored black*/
__device__
void getSmoothColorGPU(const cuDoubleComplex* complexNum, double* smooth)
{
    int iters = 0;
    cuDoubleComplex z = make_cuDoubleComplex(0, 0);
    /*(z.y * z.y) + (z.x * z.x) <= 4 is equivalent to abs(z) <= 2
    While the former may look more complicated, it is about twice as efficient as the latter*/
    for (; (z.y * z.y) + (z.x * z.x) <= 4 && iters < MAX_ITER; (iters)++)
    {
        z = cuCmul(z, z);
        z = cuCadd(z, *complexNum);
    }
    iters == MAX_ITER ? *smooth = -1.0 : calcSmoothColorGPU(&z, &iters, smooth);
}

//Uses the mandelbrot smooth-coloring algorithm (https://stackoverflow.com/questions/369438/smooth-spectrum-for-mandelbrot-set-rendering)
//to calculate a value between 0 and 1 which will be used to determine the color of a pixel
__device__
void calcSmoothColorGPU(const cuDoubleComplex* complexNum, const int* iters, double* smooth)
{
    //sqrt((complexNum->x * complexNum->x) + (complexNum->y * complexNum->y))
    //is much faster than abs(z) but gives the same result
    double complexAbs = sqrt((complexNum->x * complexNum->x) + (complexNum->y * complexNum->y));
    double complexDoubleLog = log10(log10(complexAbs));
    *smooth = *iters + 1 - (complexDoubleLog / log10(2.0));
}

//Calculates results for a section of the fractal
__global__
void makeFracGPU(RGB* resultArr, const fracState* state)
{
    double smooth;
    int sectionHeight = (int)floor((double)(WINDOW_HEIGHT / gridDim.x));
    int sectionWidth = (int)floor((double)(WINDOW_WIDTH / blockDim.x));
    float h, s = 0.7, v = 1.0, r, g, b;
    cuDoubleComplex complexPixel = make_cuDoubleComplex(0, 0);

    for (int y = blockIdx.x * sectionHeight; y < (blockIdx.x + 1) * sectionHeight; y++)
    {
        for (int x = threadIdx.x * sectionWidth; x < (threadIdx.x + 1) * sectionWidth; x++)
        {
            coordsToComplexGPU(&x, &y, state, &complexPixel);
            getSmoothColorGPU(&complexPixel, &smooth);

            if (smooth == -1.0)
                r = g = b = 0;
            else
            {
                h = smooth + 255;
                HSVtoRGB_GPU(r, g, b, h, s, v);
            }

            resultArr[y * WINDOW_WIDTH + x].r = r * 255;
            resultArr[y * WINDOW_WIDTH + x].g = g * 255;
            resultArr[y * WINDOW_WIDTH + x].b = b * 255;
        }
    }
    return;
}