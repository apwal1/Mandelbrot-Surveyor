#include "fracThread.hpp"
#include "hsvrgb.h"

using std::unique_lock;

fracThread::fracThread(int width, int height, int iterations, pair<SDL_Point, SDL_Point> bounds, RGB* arr, const fracState* state)
    : windowWidth(double(width)), windowHeight(double(height)), maxIters(iterations), sectionBounds(bounds)
{
    xOffset = &state->xPanOffset;
    yOffset = &state->yPanOffset;
    xZoom = &state->xZoomScale;
    yZoom = &state->yZoomScale;

    subThread = thread(&fracThread::makeFractal, this, arr);
}

//Calculates the thread's portion of the fractal
void fracThread::makeFractal(RGB* resultArr)
{
    float h, s = 0.7, v = 1.0, r, g, b;
    complex<double> complexPixel;
    double smooth = 0;

    while (running)
    {
        unique_lock<mutex> lock(mx);
        start.wait(lock);

        for (int y = sectionBounds.first.y; y < sectionBounds.second.y; y++)
        {
            for (int x = sectionBounds.first.x; x < sectionBounds.second.x; x++)
            {
                /*Creates a complex number based on the coordinates of whichever pixel we are
                drawing and calculates the color the pixel should be*/
                coordsToComplex(&x, &y, &complexPixel);
                getSmoothColor(&complexPixel, &smooth);

                if (smooth == -1.0)
                    r = g = b = 0;
                else
                {
                    h = smooth + 255;
                    HSVtoRGB(r, g, b, h, s, v);
                }

                //Saves our result into the 1 dimensional result array that we are using to mimic a 2d array
                resultArr[y * WINDOW_WIDTH + x].r = r * 255;
                resultArr[y * WINDOW_WIDTH + x].g = g * 255;
                resultArr[y * WINDOW_WIDTH + x].b = b * 255;
            }
        }
    }
}

//Lets the thread calculate its portion of the current frame using the arguments which represent the current pan/zoom state
void fracThread::run()
{
    unique_lock<mutex>(mx);
    start.notify_one();
}

//Hangs until thread exits
void fracThread::join()
{
    running = false;
    start.notify_one();
	subThread.join();
}

//Hangs until the thread is done calculating its portion of the frame
void fracThread::waitUntilDone()
{
    //We will know the thread is done once it releases the mutex. This lock will be released once the unique_lock is deallocated
    unique_lock<mutex> lock(mx);
}

//Returns true if the thread is able to be joined, false otherwise
bool fracThread::joinable()
{
    return subThread.joinable();
}

//Converts a pixel's coordinates to a complex number, which will be stored in result
void fracThread::coordsToComplex(const int* x, const int* y, complex<double>* result)
{
    result->real(((double)*x + *xOffset) / *xZoom);
    result->imag(((double)*y + *yOffset) / *yZoom);
}

/*Calculates the number of iterations required to determine whether the passed complex number is
in the mandelbrot set or not. The result will be placed in the passed double* smooth and will be -1.0
if the pixel is in the mandelbrot set and should be colored black*/
void fracThread::getSmoothColor(const complex<double>* complexNum, double* smooth)
{
    int iters = 0;
    complex<double> z = 0;
    /*(z.imag() * z.imag()) + (z.real() * z.real()) <= 4 is equivalent to abs(z) <= 2
    While the former may look more complicated, it is about twice as efficient as the latter*/
    for (; (z.imag() * z.imag()) + (z.real() * z.real()) <= 4 && iters < maxIters; iters++)
        z = z * z + *complexNum;
    iters == maxIters ? *smooth = -1.0 : calcSmoothColor(&z, &iters, smooth);
}

//Uses the mandelbrot smooth-coloring algorithm (https://stackoverflow.com/questions/369438/smooth-spectrum-for-mandelbrot-set-rendering)
//to calculate a value between 0 and 1 which will be used to determine the color of a pixel
void fracThread::calcSmoothColor(const complex<double>* complexNum, const int* iters, double* smooth)
{
    //sqrt((complexNum->imag() * complexNum->imag()) + (complexNum->real() * complexNum->real()))
    //is much faster than abs(z) but gives the same result
    double complexAbs = sqrt((complexNum->imag() * complexNum->imag()) + (complexNum->real() * complexNum->real()));
    double complexDoubleLog = log10(log10(complexAbs));
    *smooth = *iters + 1 - (complexDoubleLog / log10(2));
}