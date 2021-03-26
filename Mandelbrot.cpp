#include <iostream>
#include <complex>
#include <chrono>
#include <vector>
#include <thread>
#include "fracThread.h"
#include <SDL.h>

//Window width and height
#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720

/*Max iterations before a complex number is considered to be in the mandelbrot set.
Increasing this will result in a better quality fractal with reduced performance.
Reducing this will result in a lower quality fractal with improved performance.*/
#define MAX_ITER 1024

//Number of concurrently running threads
#define NUM_THREADS 32

using std::complex;
using std::string;
using std::cout;
using std::endl;
using std::vector;
using std::thread;
using std::atomic;
using std::pair;

using namespace std::chrono;

bool eventHandler(SDL_Event* event, bool* mousePanning, SDL_Point* mouseCoords, fracState* state);
void zoom(fracState* state, double zoomAmount);
void pan(int* beforeX, int* beforeY, const int* afterX, const int* afterY, fracState* state);
void SDLError(string errorMsg, bool* errorFlag);

int main(int argc, char* argv[]) {
    SDL_Event event;
    SDL_Renderer* renderer;
    SDL_Window* window;

    //Used to keep track of panning
    bool mousePanning = false;
    SDL_Point mousePanningCoords;

    //Initializes the fractal to the middle of the screen
    fracState state(-WINDOW_WIDTH / 2, -WINDOW_HEIGHT / 2);

    fracThread* threads[NUM_THREADS];

    //This will be true if there was an error in the initialization of the SDL window/renderer
    bool initError = false;

    //Creating a 2d array of ints
    int** result = new int* [WINDOW_WIDTH];
    for (int i = 0; i < WINDOW_WIDTH; i++)
        result[i] = new int [WINDOW_HEIGHT];

    //Initializes our array of fracThreads
    for (int i = 0; i < NUM_THREADS; i++)
    {
        SDL_Point start = { i * WINDOW_WIDTH / NUM_THREADS, 0 };
        SDL_Point end = { (i + 1) * WINDOW_WIDTH / NUM_THREADS, WINDOW_HEIGHT };
        pair<SDL_Point, SDL_Point> bounds(start, end);
        threads[i] = new fracThread(WINDOW_WIDTH, WINDOW_HEIGHT, MAX_ITER, bounds, result, &state);
    }

    //Initializes SDL video and checks for errors
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        SDLError("Error during SDL_Init", &initError);
    //Initializes SDL window and checks for errors
    if ((window = SDL_CreateWindow("Mandelbrot Surveyor", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WINDOW_WIDTH, WINDOW_HEIGHT, 0)) == NULL)
        SDLError("Error during SDL_CreateWindow", &initError);
    //Initializes SDL renderer and checks for errors
    if ((renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED)) == NULL)
        SDLError("Error during SDL_CreateRenderer", &initError);
    
    if (!initError)
    {
        while (!eventHandler(&event, &mousePanning, &mousePanningCoords, &state))
        {
            auto start = high_resolution_clock::now();

            //Starts the threads
            for (auto& i : threads)
                i->run();

            //Waits for every thread to finish its portion of the frame
            for (auto& i : threads)
                i->waitUntilDone();

            duration<double> time = high_resolution_clock::now() - start;
            cout << "Created result array in " << time.count() << " seconds\n";

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
                            //These values were taken from https://stackoverflow.com/questions/16500656/which-color-gradient-is-used-to-color-mandelbrot-in-wikipedia
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

    //Cleaning up 2D result array
    for (int x = 0; x < WINDOW_WIDTH; x++)
        delete[] result[x];
    delete[] result;

    return 0;
}

/*Raises/lowers the zoom scale by a passed amount. Zooming will occur relative to the center of the window
  Amounts > 1 will result in a zoom-in, amounts < 1 will result in a zoom-out*/
void zoom(fracState* state, double zoomAmount)
{
    pair<double, double> beforeCenter = { 
        ((double)(WINDOW_WIDTH / 2) + state->xPanOffset) / state->xZoomScale,
        ((double)(WINDOW_HEIGHT / 2) + state->yPanOffset) / state->yZoomScale,
    };

    state->xZoomScale *= zoomAmount;
    state->yZoomScale *= zoomAmount;

    pair<double, double> afterCenter = {
        ((double)(WINDOW_WIDTH / 2) + state->xPanOffset) / state->xZoomScale,
        ((double)(WINDOW_HEIGHT / 2) + state->yPanOffset) / state->yZoomScale,
    };

    state->xPanOffset += (beforeCenter.first - afterCenter.first) * state->xZoomScale;
    state->yPanOffset += (beforeCenter.second - afterCenter.second) * state->yZoomScale;
}

//Handles panning calculations
void pan(int* beforeX, int* beforeY, const int* afterX, const int* afterY, fracState* state)
{
    state->xPanOffset += (double)(*beforeX - *afterX);
    state->yPanOffset += (double)(*beforeY - *afterY);
    *beforeX = *afterX;
    *beforeY = *afterY;
}

/*Handles mouse/keyboard events. Will return true if the user has chosen to close the window
or false otherwise*/
bool eventHandler(SDL_Event* event, bool* mousePanning, SDL_Point* mouseCoords, fracState* state)
{
    SDL_FlushEvents(SDL_KEYDOWN, SDL_MOUSEMOTION);
    SDL_PollEvent(event);

    if (event->type == SDL_QUIT)
        return true;

    //Gets the keyboard's state (which keys are active, which aren't)
    const unsigned char* kbState = SDL_GetKeyboardState(NULL);

    //Runs when user starts panning
    if (event->type == SDL_MOUSEBUTTONDOWN && event->button.button == SDL_BUTTON_LEFT)
    {
        *mousePanning = true;
        mouseCoords->x = event->button.x;
        mouseCoords->y = event->button.y;
    }

    //Handles panning events. Runs while the user is panning
    if (*mousePanning)
    {
        int newMouseX;
        int newMouseY;

        //Pans if the left mouse button is still being held down
        if (SDL_GetMouseState(&newMouseX, &newMouseY) & SDL_BUTTON_LMASK)
            pan(&mouseCoords->x, &mouseCoords->y, &newMouseX, &newMouseY, state);
        //If the left mouse button is no longer being held down, stop panning
        else
            *mousePanning = false;
    }
    //Panning and zooming at the same time seems to cause problems, so zooming
    //should only be considered if the user is not panning
    else
    {
        //Handles scroll wheel zoom events 
        if (event->type == SDL_MOUSEWHEEL)
        {
            if (event->wheel.y > 0)
                zoom(state, 1.1);
            else if (event->wheel.y < 0)
                zoom(state, 0.9);
            event->wheel.y = 0;
        }

        //Handles keyboard zoom events (zoom in with W, zoom out with S)
        if (kbState[SDL_SCANCODE_W])
            zoom(state, 1.1);
        else if (kbState[SDL_SCANCODE_S])
            zoom(state, 0.9);
    }

    return false;
}

//Logs an SDL error
void SDLError(string errorMsg, bool* errorFlag)
{
    SDL_LogError(SDL_LOG_CATEGORY_ERROR, *errorMsg.c_str() + ": " + *SDL_GetError());
    *errorFlag = true;
}