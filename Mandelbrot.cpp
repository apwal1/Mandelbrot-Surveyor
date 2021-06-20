#include <iostream>
#include <complex>
#include <chrono>
#include <vector>
#include "fracThread.hpp"
#include "GPUCalc.cuh"
#include <SDL.h>

#define FPS_CAP 60

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
    SDL_Window* window;
    SDL_Surface* fracSurface;
    SDL_Surface* windowSurface;

    //Used to help keep track of panning
    bool mousePanning = false;
    SDL_Point mousePanningCoords;

    fracState state;
    fracState* d_state;
    cudaMalloc((void**)&d_state, sizeof(fracState));

    fracThread* threads[NUM_CPU_THREADS];

    RGB temp;

    //This will be true if there was an error in the initialization of the SDL window/renderer
    bool initError = false;

    //Initializes the fractal to the middle of the screen in GPU rendering mode
    state.initialize(-WINDOW_WIDTH / 2, -WINDOW_HEIGHT / 2, GPU);

    //Creating a 1d array of ints
    RGB* result = new RGB [WINDOW_WIDTH * WINDOW_HEIGHT];

    //Initializes our array of fracThreads
    for (int i = 0; i < NUM_CPU_THREADS; i++)
    {
        SDL_Point start = { i * WINDOW_WIDTH / NUM_CPU_THREADS, 0 };
        SDL_Point end = { (i + 1) * WINDOW_WIDTH / NUM_CPU_THREADS, WINDOW_HEIGHT };
        pair<SDL_Point, SDL_Point> bounds(start, end);
        threads[i] = new fracThread(WINDOW_WIDTH, WINDOW_HEIGHT, MAX_ITER, bounds, result, &state);
    }

    //Creating 1d array of ints in GPU mem
    RGB* d_result;
    cudaMalloc((void**)&d_result, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(RGB));

    //Initializes SDL video and checks for errors
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        SDLError("Error during SDL_Init", &initError);
    //Initializes SDL window and checks for errors
    if ((window = SDL_CreateWindow("Mandelbrot Surveyor", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WINDOW_WIDTH, WINDOW_HEIGHT, 0)) == NULL)
        SDLError("Error during SDL_CreateWindow", &initError);

    windowSurface = SDL_GetWindowSurface(window);

    if (!initError)
    {
        while (!eventHandler(&event, &mousePanning, &mousePanningCoords, &state))
        {
            auto start = std::chrono::high_resolution_clock::now();
            //If we are rendering the fractal in GPU mode, this runs
            if (state.calcMethod == GPU)
            {
                //Copy the state of the fractal to GPU mem
                cudaMemcpy(d_state, &state, sizeof(fracState), cudaMemcpyHostToDevice);

                //Run the GPU calculations and wait for all threads to finish
                makeFracGPU<<<NUM_BLOCKS, NUM_GPU_THREADS>>> (d_result, d_state);
                cudaDeviceSynchronize();

                //Copy results back from GPU mem
                cudaMemcpy(result, d_result, WINDOW_HEIGHT * WINDOW_WIDTH * sizeof(RGB), cudaMemcpyDeviceToHost);
            }
            //If we are rendering the fractal in CPU mode, this runs
            else
            {
                //Starts the threads
                for (auto& i : threads)
                    i->run();

                //Waits for every thread to finish its portion of the frame
                for (auto& i : threads)
                    i->waitUntilDone();
            }

            //Creates a surface out of our RGB result array and displays it on the window
            fracSurface = SDL_CreateRGBSurfaceFrom((void*) result, WINDOW_WIDTH, WINDOW_HEIGHT, 24, WINDOW_WIDTH * 3, 0x0000FF, 0x00FF00, 0xFF0000, 0);
            SDL_BlitSurface(fracSurface, NULL, windowSurface, NULL);
            SDL_UpdateWindowSurface(window);
            SDL_FreeSurface(fracSurface);

            //Measures FPS and caps it at FPS_CAP by making this thread sleep
            std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;
            if (floor(1 / time.count()) > FPS_CAP)
            {
                state.fps = 1 / (1 / (double)FPS_CAP) - time.count();
                std::this_thread::sleep_for(std::chrono::milliseconds((int)floor(((1 / (double)FPS_CAP) - time.count()) * 1000)));
            }
            else
                state.fps = floor(1 / time.count());

            //Prints performance metric and fractal state
            cout << "Rendered frame in " << time.count() << " seconds. " << state.fps << "fps\nMode: ";
            state.calcMethod == GPU ? cout << "GPU" : cout << "CPU";
            cout << "\tZoom Scale: " << (double)(state.yZoomScale / (WINDOW_HEIGHT / 3)) << endl;
            cout << "Pan X: " << state.xPanOffset << "\tPan Y: " << state.yPanOffset << endl;
        }
    }

    //Cleaning up SDL
    SDL_DestroyWindow(window);
    SDL_Quit();

    //Cleaning up CPU threads
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
    SDL_FlushEvents(SDL_KEYDOWN, SDL_MOUSEMOTION);
    SDL_PollEvent(event);
    SDL_FlushEvent(SDL_MOUSEWHEEL);

    if (event->type == SDL_QUIT || event->type == SDL_APP_TERMINATING)
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

    //Runs when user stops panning
    if (event->type == SDL_MOUSEBUTTONUP && event->button.button == SDL_BUTTON_LEFT)
        *mousePanning = false;

    //Handles panning events. Runs while the user is panning
    if (*mousePanning)
        pan(&mouseCoords->x, &mouseCoords->y, &mouseX, &mouseY, state);
    else
    {
        //Handles keyboard zoom events (zoom in with W, zoom out with S)
        if (kbState[SDL_SCANCODE_W] && event->type != SDL_MOUSEWHEEL)
            zoom(state, 1.05 - (state->fps / (21 * FPS_CAP)), mouseX, mouseY);
        else if (kbState[SDL_SCANCODE_S] && event->type != SDL_MOUSEWHEEL)
            zoom(state, 0.95 + (state->fps / (21 * FPS_CAP)), mouseX, mouseY);
        //Handles scroll wheel zoom events 
        else if (event->type == SDL_MOUSEWHEEL)
        {
            if (event->wheel.y > 0)
                zoom(state, 1.1, mouseX, mouseY);
            else if (event->wheel.y < 0)
                zoom(state, 0.9, mouseX, mouseY);
            event->wheel.y = 0;
        }
    }
    //Responds to the M key being pressed by switching render method
    if (kbState[SDL_SCANCODE_M])
        state->calcMethod == CPU ? state->calcMethod = GPU : state->calcMethod = CPU;

    return false;
}

//Logs an error during SDL initialization
void SDLError(string errorMsg, bool* errorFlag)
{
    SDL_LogError(SDL_LOG_CATEGORY_ERROR, *errorMsg.c_str() + ": " + *SDL_GetError());
    *errorFlag = true;
}