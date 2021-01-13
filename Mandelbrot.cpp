#include <iostream>
#include <complex>
#include <SDL.h>
#undef main

#define WINDOW_WIDTH 300
#define WINDOW_HEIGHT 300
#define MAX_ITER 128

using std::complex;

int getNumIters(const complex<double>* complexNum);

int main(void) {
    int iterations;
    complex<double> complexPixel;

    SDL_Event event;
    SDL_Renderer* renderer;
    SDL_Window* window;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);
    for (int y = 0; y < WINDOW_HEIGHT; y++)
        for (int x = 0; x < WINDOW_WIDTH; x++)
        {
            complexPixel.real(-2 + (x / double(WINDOW_WIDTH)) * 3);
            complexPixel.imag( -1.5 + (y / double(WINDOW_HEIGHT)) * 3);
            iterations = getNumIters(&complexPixel);

            if (iterations == MAX_ITER)
                SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
            else
            {
                //Makes the fractal look cool (decides the color of a pixel in the mandelbrot set)
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
            SDL_RenderDrawPoint(renderer, x, y);
        }

    SDL_RenderPresent(renderer);

    while (1)
    {
        SDL_PollEvent(&event);
        if (event.type == SDL_QUIT)
            break;
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return EXIT_SUCCESS;
}

//Calculates the number of iterations required to determine whether the passed complex number is in the mandelbrot set or not
int getNumIters(const complex<double>* complexNum)
{
    int iters = 0;
    complex<double> z = 0;
    for (; abs(z) <= 2 && iters < MAX_ITER; iters++)
        z = z * z + *complexNum;
    return iters;
}