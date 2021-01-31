#include "fracThread.h"

using std::unique_lock;

fracThread::fracThread(double width, double height, int iterations, SDL_Point startCoord, SDL_Point endCoord, int** arr) : windowWidth(width), windowHeight(height), maxIters(iterations), startPoint(startCoord), endPoint(endCoord)
{
    subThread = thread(&fracThread::makeFractal, this, arr);
}

//Calculates the thread's portion of the fractal
void fracThread::makeFractal(int** iterVec)
{
    complex<double> complexPixel;
    int iterations;

    while (running)
    {
        unique_lock<mutex> lock(mx);
        start.wait(lock);

        for (int y = startPoint.y; y < endPoint.y; y++)
        {
            for (int x = startPoint.x; x < endPoint.x; x++)
            {
                /*Creates a complex number based on the coordinates of whichever pixel we are
                drawing and calculates how many iterations were needed to decide whether it is
                in the mandelbrot set or not*/
                coordsToComplex(&x, &y, &complexPixel);
                iterVec[x][y] = getNumIters(&complexPixel);
            }
        }
        done = true;
    }
}

//Lets the thread calculate its portion of the current frame using the arguments which represent the current pan/zoom state
void fracThread::run(const double* xOffset, const double* yOffset, const double* xZoom, const double* yZoom)
{
    unique_lock<mutex>(mx);
    this->xOffset = xOffset;
    this->yOffset = yOffset;
    this->xZoom = xZoom;
    this->yZoom = yZoom;
    done = false;
    start.notify_one();
}

//Hangs until the thread exits
void fracThread::join()
{
    running = false;
    start.notify_one();
	subThread.join();
}

//Checks if the thread is done with the current frame
bool fracThread::isThreadDone()
{
    return done;
}

//Converts a pixel's coordinates to a complex number (result)
void fracThread::coordsToComplex(const int* x, const int* y, complex<double>* result)
{
    result->real(*xOffset + ((*x / *xZoom) / windowWidth));
    result->imag(*yOffset + (*y / windowHeight) / *yZoom);
}

/*Calculates the number of iterations required to determine whether the passed complex number is
in the mandelbrot set or not*/
int fracThread::getNumIters(const complex<double>* complexNum)
{
    int iters = 0;
    complex<double> z = 0;
    for (; (z.imag() * z.imag()) + (z.real() * z.real()) <= 4 && iters < maxIters; iters++)
        z = z * z + *complexNum;
    return iters;
}