#include "fracThread.h"

using std::unique_lock;

fracThread::fracThread(int width, int height, int iterations, pair<SDL_Point, SDL_Point> bounds, int** arr, fracState* state)
    : windowWidth(double(width)), windowHeight(double(height)), maxIters(iterations), sectionBounds(bounds)
{
    xOffset = &state->xPanOffset;
    yOffset = &state->yPanOffset;
    xZoom = &state->xZoomScale;
    yZoom = &state->yZoomScale;

    subThread = thread(&fracThread::makeFractal, this, arr);
}

//Calculates the thread's portion of the fractal
void fracThread::makeFractal(int** iterVec)
{
    complex<double> complexPixel;
    int iterations = 0;

    while (running)
    {
        unique_lock<mutex> lock(mx);
        start.wait(lock);

        for (int y = sectionBounds.first.y; y < sectionBounds.second.y; y++)
        {
            for (int x = sectionBounds.first.x; x < sectionBounds.second.x; x++)
            {
                /*Creates a complex number based on the coordinates of whichever pixel we are
                drawing and calculates how many iterations were needed to decide whether it is
                in the mandelbrot set or not*/
                coordsToComplex(&x, &y, &complexPixel);
                getNumIters(&complexPixel, &iterations);
                iterVec[x][y] = iterations;
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

//Converts a pixel's coordinates to a complex number, which will be stored in result
void fracThread::coordsToComplex(const int* x, const int* y, complex<double>* result)
{
    result->real(((double)*x + *xOffset) / *xZoom);
    result->imag(((double)*y + *yOffset) / *yZoom);
}

/*Calculates the number of iterations required to determine whether the passed complex number is
in the mandelbrot set or not. The result will be placed in the passed int* iters*/
void fracThread::getNumIters(const complex<double>* complexNum, int* iters)
{
    complex<double> z = 0;
    /*(z.imag() * z.imag()) + (z.real() * z.real()) <= 4 is equivalent to abs(z) <= 2
    While the former may look more complicated, it is about twice as efficient as the latter*/
    for (*iters = 0; (z.imag() * z.imag()) + (z.real() * z.real()) <= 4 && *iters < maxIters; (*iters)++)
        z = z * z + *complexNum;
}