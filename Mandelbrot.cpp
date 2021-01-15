#include <iostream>
#include <complex>
#include <chrono>
#include <SDL.h>

//Window width and height as a double to avoid int/double conversion for each pixel (very small performance increase)
#define WINDOW_WIDTH 1280.0
#define WINDOW_HEIGHT 720.0

/*Max iterations before a complex number is considered to not be in the mandelbrot set.
Increasing this will result in a better quality fractal with reduced performance.
Reducing this will result in a lower quality fractal with improved performance.*/
#define MAX_ITER 128

using std::complex;
using std::cout;
using std::endl;
using namespace std::chrono;

int getNumIters(const complex<double>* complexNum);
bool eventHandler(SDL_Event* event, bool* mousePanning, SDL_Point* mouseCoords, double* xOffset, double* yOffset, double* xZoom, double* yZoom);
void coordsToComplex(const int* x, const int* y, const double* xOffset, const double* yOffset, const double* xZoom, const double* yZoom, complex<double>* result);

int main(int argc, char* argv[]) {
    int iterations;
    complex<double> complexPixel;

    SDL_Event event;
    SDL_Renderer* renderer;
    SDL_Window* window;

    //Used to keep track of panning
    double xPanOffset = -2;
    double yPanOffset = -1.5;
    bool mousePanning = false;

    //Used to keep track of panning
    double xZoomScale = 4;
    double yZoomScale = 3;

    SDL_Point mousePanningCoords;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(int(WINDOW_WIDTH), int(WINDOW_HEIGHT), 0, &window, &renderer);
    while (!eventHandler(&event, &mousePanning, &mousePanningCoords, &xPanOffset, &yPanOffset, &xZoomScale, &yZoomScale))
    {
        auto start = high_resolution_clock::now();
        for (int y = 0; y < WINDOW_HEIGHT; y++)
            for (int x = 0; x < WINDOW_WIDTH; x++)
            {
                /*Creates a complex number based on the coordinates of whichever pixel we are
                drawing and calculates how many iterations were needed to decide whether it is
                in the mandelbrot set*/
                coordsToComplex(&x, &y, &xPanOffset, &yPanOffset, &xZoomScale, &yZoomScale, &complexPixel);
                iterations = getNumIters(&complexPixel);

                if (iterations == MAX_ITER)
                    //Each pixel that is in the mandelbrot set is black
                    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
                else
                {
                    //Makes the fractal look cool (decides the color of a pixel that isn't in the mandelbrot set)
                    switch (iterations % 16)
                    {
                    case 0: SDL_SetRenderDrawColor(renderer, 66, 30, 15, 255); break;
                    case 1: SDL_SetRenderDrawColor(renderer, 25, 7, 26, 255); break;
                    case 2: SDL_SetRenderDrawColor(renderer, 9, 1, 47, 255); break;
                    case 3: SDL_SetRenderDrawColor(renderer, 4, 4, 73, 255); break;
                    case 4: SDL_SetRenderDrawColor(renderer, 0, 7, 100, 255); break;
                    case 5: SDL_SetRenderDrawColor(renderer, 12, 44, 138, 255); break;
                    case 6: SDL_SetRenderDrawColor(renderer, 24, 82, 177, 255); break;
                    case 7: SDL_SetRenderDrawColor(renderer, 57, 125, 209, 255); break;
                    case 8: SDL_SetRenderDrawColor(renderer, 134, 181, 229, 255); break;
                    case 9: SDL_SetRenderDrawColor(renderer, 211, 236, 248, 255); break;
                    case 10: SDL_SetRenderDrawColor(renderer, 241, 233, 191, 255); break;
                    case 11: SDL_SetRenderDrawColor(renderer, 248, 201, 95, 255); break;
                    case 12: SDL_SetRenderDrawColor(renderer, 255, 170, 0, 255); break;
                    case 13: SDL_SetRenderDrawColor(renderer, 204, 128, 0, 255); break;
                    case 14: SDL_SetRenderDrawColor(renderer, 153, 87, 0, 255); break;
                    case 15: SDL_SetRenderDrawColor(renderer, 106, 52, 3, 255);
                    }
                }
                //Draws a pixel to the window with coordinates x, y and the color chosen above
                SDL_RenderDrawPoint(renderer, x, y);
            }
        SDL_RenderPresent(renderer);
        cout << "Rendered in " << duration_cast<milliseconds>(high_resolution_clock::now() - start).count() << "ms" << endl;
    }

    //Cleaning up
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}

/*Calculates the number of iterations required to determine whether the passed complex number is
in the mandelbrot set or not*/
int getNumIters(const complex<double>* complexNum)
{
    int iters = 0;
    complex<double> z = 0;
    for (; abs(z) <= 2 && iters < MAX_ITER; iters++)
        z = z * z + *complexNum;
    return iters;
}

//Converts a pixel's coordinates to a complex number (result)
void coordsToComplex(const int* x, const int* y, const double* xOffset, const double* yOffset, const double* xZoom, const double* yZoom, complex<double>* result)
{
    result->real(*xOffset + (*x / WINDOW_WIDTH) * *xZoom);
    result->imag(*yOffset + (*y / WINDOW_HEIGHT) * *yZoom);
}

/*Handles mouse/keyboard events. Will return true if the user has chosen to close the window
or false otherwise*/
bool eventHandler(SDL_Event* event, bool* mousePanning, SDL_Point* mouseCoords, double* xOffset, double* yOffset, double* xZoom, double* yZoom)
{
    SDL_PollEvent(event);
    if (event->type == SDL_QUIT)
        return true;

    //Handles panning events
    if (*mousePanning)
    {
        if (SDL_GetRelativeMouseState(NULL, NULL) & SDL_BUTTON_LMASK)
        {
            *xOffset += (0.001 * *xZoom) * double(mouseCoords->x - event->button.x);
            *yOffset += (0.001 * *yZoom) * double(mouseCoords->y - event->button.y);
            mouseCoords->x = event->button.x;
            mouseCoords->y = event->button.y;
        }
        else
            *mousePanning = false;
    }

    if (event->type == SDL_MOUSEBUTTONDOWN && event->button.button == SDL_BUTTON_LEFT)
    {
        *mousePanning = true;
        mouseCoords->x = event->button.x;
        mouseCoords->y = event->button.y;
    }
    else if (event->type == SDL_MOUSEBUTTONUP && event->button.button == SDL_BUTTON_LEFT)
        *mousePanning = false;

    //Handles scroll wheel zoom events
    const unsigned char* kbState = SDL_GetKeyboardState(NULL);
    if (event->type == SDL_MOUSEWHEEL)
    {
        if (event->wheel.y > 0)
        {
            *xZoom *= 0.9;
            *yZoom *= 0.9;
        }
        else if (event->wheel.y < 0)
        {
            *xZoom *= 1.1;
            *yZoom *= 1.1;
        }
    }

    //Handles keyboard zoom events (zoom in with W, zoom out with s)
    if (kbState[SDL_SCANCODE_W])
    {
        *xZoom *= 0.9;
        *yZoom *= 0.9;
    }
    else if (kbState[SDL_SCANCODE_S])
    {
        *xZoom *= 1.1;
        *yZoom *= 1.1;
    }

    return false;
}