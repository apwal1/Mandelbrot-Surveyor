#include <iostream>
#include <complex>
#include <chrono>
#include <vector>
#include "fracThread.h"
#include "GPUCalc.cuh"
#include <SDL.h>

//Number of GPU blocks
#define NUM_BLOCKS 20
//Number of GPU threads per block
#define NUM_GPU_THREADS 640

//Number of CPU threads
#define NUM_CPU_THREADS 32

using std::complex;
using std::string;
using std::cout;
using std::endl;
using std::vector;
using std::atomic;
using std::pair;

bool eventHandler(SDL_Event* event, bool* mousePanning, SDL_Point* mouseCoords, fracState* state);
void zoom(fracState* state, double zoomAmount, int mouseX, int mouseY);
void pan(int* beforeX, int* beforeY, const int* afterX, const int* afterY, fracState* state);
void SDLError(string errorMsg, bool* errorFlag);

int main(int argc, char* argv[]) {
    SDL_Event event;
    SDL_Renderer* renderer;
    SDL_Window* window;

    //Used to help keep track of panning
    bool mousePanning = false;
    SDL_Point mousePanningCoords;

    fracState state;
    fracState* d_state;
    cudaMalloc((void**)&d_state, sizeof(fracState));

    fracThread* threads[NUM_CPU_THREADS];

    //This will be true if there was an error in the initialization of the SDL window/renderer
    bool initError = false;

    //Initializes the fractal to the middle of the screen
    state.initialize(-WINDOW_WIDTH / 2, -WINDOW_HEIGHT / 2, GPU);

    //Creating a 1d array of ints
    int* result = new int [WINDOW_WIDTH * WINDOW_HEIGHT];

    //Initializes our array of fracThreads
    for (int i = 0; i < NUM_CPU_THREADS; i++)
    {
        SDL_Point start = { i * WINDOW_WIDTH / NUM_CPU_THREADS, 0 };
        SDL_Point end = { (i + 1) * WINDOW_WIDTH / NUM_CPU_THREADS, WINDOW_HEIGHT };
        pair<SDL_Point, SDL_Point> bounds(start, end);
        threads[i] = new fracThread(WINDOW_WIDTH, WINDOW_HEIGHT, MAX_ITER, bounds, result, &state);
    }

    //Creating 1d array of ints in GPU mem
    int* d_result;
    cudaMalloc((void**)&d_result, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(int));

    //Initializes SDL video and checks for errors
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        SDLError("Error during SDL_Init", &initError);
    //Initializes SDL window and checks for errors
    if ((window = SDL_CreateWindow("Mandelbrot Surveyor", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WINDOW_WIDTH, WINDOW_HEIGHT, 0)) == NULL)
        SDLError("Error during SDL_CreateWindow", &initError);
    //Initializes SDL renderer and checks for errors
    if ((renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED)) == NULL)
        SDLError("Error during SDL_CreateRenderer", &initError);

    int temp = 0;

    if (!initError)
    {
        while (!eventHandler(&event, &mousePanning, &mousePanningCoords, &state))
        {
            auto start = std::chrono::high_resolution_clock::now();
            //If we are rendering the fractal in GPU mode, this runs
            if (state.calcMethod == GPU)
            {
                SDL_SetWindowTitle(window, "Mandelbrot Surveyor - GPU");

                //Copy the state of the fractal to GPU mem
                cudaMemcpy(d_state, &state, sizeof(fracState), cudaMemcpyHostToDevice);

                //Run the GPU calculations and wait for all threads to finish
                makeFracGPU << <NUM_BLOCKS, NUM_GPU_THREADS >> > (d_result, d_state);
                cudaDeviceSynchronize();

                //Copy results back from GPU mem
                cudaMemcpy(result, d_result, WINDOW_HEIGHT * WINDOW_WIDTH * sizeof(int), cudaMemcpyDeviceToHost);
            }
            //If we are rendering the fractal in CPU mode, this runs
            else
            {
                SDL_SetWindowTitle(window, "Mandelbrot Surveyor - CPU");

                //Starts the threads
                for (auto& i : threads)
                    i->run();

                //Waits for every thread to finish its portion of the frame
                for (auto& i : threads)
                    i->waitUntilDone();
            }
            //Prints performance metric and current state of fractal
            std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;
            cout << "Created and copied result array in " << time.count() << " seconds\n";
            cout << "Zoom X: " << (double)(state.xZoomScale / (WINDOW_WIDTH / 4)) << " Zoom Y: " << (double)(state.yZoomScale / (WINDOW_HEIGHT / 3)) << endl;
            cout << "Pan X: " << state.xPanOffset << " Pan Y: " << state.yPanOffset << endl;

            //Determines the color of each pixel and draws it to the screen
            for (int x = 0; x < WINDOW_WIDTH; x++)
                for (int y = 0; y < WINDOW_HEIGHT; y++)
                {
                    temp = result[y * WINDOW_WIDTH + x];
                    if (temp == MAX_ITER)
                        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
                    else
                    {
                        //Makes the fractal look nice (color choosing)
                        switch (temp % 16)
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
    for (int i = 0; i < NUM_CPU_THREADS; i++)
    {
        threads[i]->join();
        delete threads[i];
    }

    //Cleaning up 1D result array from GPU mem
    cudaFree(d_result);

    //Cleaning up fracState from GPU mem
    cudaFree(d_state);

    //Cleaning up 1D result array
    delete[] result;

    return 0;
}

/*Raises/lowers the zoom scale by a passed amount. Zooming will occur relative to the
  current mouse position. zoomAmount > 1 will result in a zoom-in, zoomAmount < 1 will result in a zoom-out*/
void zoom(fracState* state, double zoomAmount, int mouseX, int mouseY)
{
    pair<double, double> beforeCenter = {
        ((double)mouseX + state->xPanOffset) / state->xZoomScale,
        ((double)mouseY + state->yPanOffset) / state->yZoomScale,
    };

    state->xZoomScale *= zoomAmount;
    state->yZoomScale *= zoomAmount;

    pair<double, double> afterCenter = {
        ((double)mouseX + state->xPanOffset) / state->xZoomScale,
        ((double)mouseY + state->yPanOffset) / state->yZoomScale,
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
    SDL_FlushEvents(SDL_KEYUP, SDL_MOUSEMOTION);
    SDL_PollEvent(event);

    if (event->type == SDL_QUIT)
        return true;

    //Holds the mouse coordinates
    int mouseX, mouseY;

    //Gets the keyboard and mouse state
    const unsigned char* kbState = SDL_GetKeyboardState(NULL);
    const unsigned int mouseState = SDL_GetMouseState(&mouseX, &mouseY);

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
        //Pans if the left mouse button is still being held down
        if (mouseState & SDL_BUTTON_LMASK)
            pan(&mouseCoords->x, &mouseCoords->y, &mouseX, &mouseY, state);
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
                zoom(state, 1.1, mouseX, mouseY);
            else if (event->wheel.y < 0)
                zoom(state, 0.9, mouseX, mouseY);
            event->wheel.y = 0;
        }

        //Handles keyboard zoom events (zoom in with W, zoom out with S)
        if (kbState[SDL_SCANCODE_W])
            zoom(state, 1.1, mouseX, mouseY);
        else if (kbState[SDL_SCANCODE_S])
            zoom(state, 0.9, mouseX, mouseY);
    }
    //Handles keyboard render method events (switch render method between CPU and GPU with M key)
    if (kbState[SDL_SCANCODE_M])
    {
        if (state->calcMethod == CPU)
            state->calcMethod = GPU;
        else
            state->calcMethod = CPU;
    }

    return false;
}

//Logs an error during SDL initialization
void SDLError(string errorMsg, bool* errorFlag)
{
    SDL_LogError(SDL_LOG_CATEGORY_ERROR, *errorMsg.c_str() + ": " + *SDL_GetError());
    *errorFlag = true;
}