#include <iostream>
#include <complex>
#include <chrono>
#include <vector>
#include <thread>
#include "fracThread.h"
#include <SDL.h>

//Window width and height as a double to avoid int/double conversion for each pixel (very small performance increase)
#define WINDOW_WIDTH 1280.0
#define WINDOW_HEIGHT 720.0

/*Max iterations before a complex number is considered to not be in the mandelbrot set.
Increasing this will result in a better quality fractal with reduced performance.
Reducing this will result in a lower quality fractal with improved performance.*/
#define MAX_ITER 1024

//Number of concurrently running threads
#define NUM_THREADS 32

using std::complex;
using std::cout;
using std::endl;
using std::vector;
using std::thread;
using std::atomic;
using namespace std::chrono;

bool eventHandler(SDL_Event* event, bool* mousePanning, SDL_Point* mouseCoords, double* xOffset, double* yOffset, double* xZoom, double* yZoom);
void zoomIn(double* xZoom, double* yZoom);
void zoomOut(double* xZoom, double* yZoom);

int main(int argc, char* argv[]) {
    complex<double> complexPixel;

    SDL_Event event;
    SDL_Renderer* renderer;
    SDL_Window* window;

    //Used to keep track of panning
    double xPanOffset = -2;
    double yPanOffset = -1.5;
    bool mousePanning = false;
    SDL_Point mousePanningCoords;

    //Used to keep track of zooming
    double xZoomScale = .25;
    double yZoomScale = .33;

    fracThread* threads[NUM_THREADS];

    //Creating a 2d array of ints (the hard way)
    int** result = new int* [int(WINDOW_WIDTH)];
    for (int i = 0; i < WINDOW_WIDTH; i++)
        result[i] = new int [int(WINDOW_HEIGHT)];

    for (int i = 0; i < NUM_THREADS; i++)
    {
        SDL_Point start = { i * int(WINDOW_WIDTH / NUM_THREADS), 0 };
        SDL_Point end = { (i + 1) * int(WINDOW_WIDTH / NUM_THREADS), int(WINDOW_HEIGHT) };
        threads[i] = new fracThread(WINDOW_WIDTH, WINDOW_HEIGHT, MAX_ITER, start, end, result);
    }


    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(int(WINDOW_WIDTH), int(WINDOW_HEIGHT), 0, &window, &renderer);
    while (!eventHandler(&event, &mousePanning, &mousePanningCoords, &xPanOffset, &yPanOffset, &xZoomScale, &yZoomScale))
    {
        auto start = high_resolution_clock::now();
        //Starts the threads
        for (auto& i : threads)
            i->run(&xPanOffset, &yPanOffset, &xZoomScale, &yZoomScale);

        //Waits for every thread to finish the frame
        for (auto& i : threads)
            while(!i->isThreadDone());
        cout << "Rendered in " << duration_cast<milliseconds>(high_resolution_clock::now() - start).count() << "ms" << endl;

        //Determines the color of each pixel and draws it to the screen
        for (int x = 0; x < WINDOW_WIDTH; x++)
            for (int y = 0; y < WINDOW_HEIGHT; y++)
            {
                if (result[x][y] == MAX_ITER)
                    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
                else
                {
                    //Makes the fractal look nice (color choosing)
                    switch (result[x][y] % 16)
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
                SDL_RenderDrawPoint(renderer, x, y);
            }

        SDL_RenderPresent(renderer);
    }

    //Cleaning up SDL
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    //Cleaning up threads
    for (int i = 0; i < NUM_THREADS; i++)
    {
        threads[i]->join();
        delete threads[i];
    }

    //Cleaning up 2D array of results
    for (int x = 0; x < WINDOW_WIDTH; x++)
    {
        delete[] result[x];
    }
    delete[] result;

    return 0;
}

/*Handles mouse/keyboard events. Will return true if the user has chosen to close the window
or false otherwise*/
bool eventHandler(SDL_Event* event, bool* mousePanning, SDL_Point* mouseCoords, double* xOffset, double* yOffset, double* xZoom, double* yZoom)
{
    SDL_FlushEvents(SDL_KEYDOWN, SDL_MOUSEMOTION);
    SDL_PollEvent(event);
    if (event->type == SDL_QUIT)
        return true;

    //Gets the keyboard's state (which keys are active, which aren't)
    const unsigned char* kbState = SDL_GetKeyboardState(NULL);

    //Handles panning events
    if (*mousePanning)
    {
        //Pans if the left mouse button is still being held down
        if (SDL_GetRelativeMouseState(NULL, NULL) & SDL_BUTTON_LMASK)
        {
            *xOffset += (0.001 * *xZoom) * double(mouseCoords->x - event->button.x);
            *yOffset += (0.001 * *yZoom) * double(mouseCoords->y - event->button.y);
            mouseCoords->x = event->button.x;
            mouseCoords->y = event->button.y;
        }
        //If the left mouse button is no longer being held down, stop panning
        else
            *mousePanning = false;
    }
    //Runs when user starts panning
    else if (event->type == SDL_MOUSEBUTTONDOWN && event->button.button == SDL_BUTTON_LEFT)
    {
        *mousePanning = true;
        mouseCoords->x = event->button.x;
        mouseCoords->y = event->button.y;
    }

    //Handles scroll wheel zoom events 
    else if (event->type == SDL_MOUSEWHEEL)
    {
        if (event->wheel.y > 0)
        {
            zoomIn(xZoom, yZoom);
        }
        else if (event->wheel.y < 0)
            zoomOut(xZoom, yZoom);
        event->wheel.y = 0;
    }

    //Handles keyboard zoom events (zoom in with W, zoom out with S)
    if (kbState[SDL_SCANCODE_W])
        zoomIn(xZoom, yZoom);
    else if (kbState[SDL_SCANCODE_S])
        zoomOut(xZoom, yZoom);

    return false;
}

//Raises the zoom scale
void zoomIn(double* xZoom, double* yZoom)
{
    *xZoom *= 1.1;
    *yZoom *= 1.1;
}

//Lowers the zoom scale
void zoomOut(double* xZoom, double* yZoom)
{
    *xZoom *= 0.9;
    *yZoom *= 0.9;
}