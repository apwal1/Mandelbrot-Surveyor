#pragma once

#include <iostream>
#include <chrono>
#include "fracThreadPool.hpp"
#include "GPUCalc.cuh"
#include <SDL.h>
#include <SDL_ttf.h>

class fracGen
{
public:
	fracGen(int windowWidth, int windowHeight, int maxIters);
	void start();
	~fracGen();
private:
	bool eventHandler();
	void zoom(double zoomAmount, int mouseX, int mouseY);
	void pan(const int& afterX, const int& afterY);
	void logError(std::string errorMsg, const char* (*errorDebugInfo)(), bool* errorFlag, bool spawnWindow);
	std::string getStateString();

	const int FPS_CAP = 30;
	//Number of GPU blocks
	const int NUM_BLOCKS = 40;
	//Number of GPU threads per block
	const int NUM_GPU_THREADS = 640;
	//Number of CPU threads
	const int NUM_CPU_THREADS = 32;
	const int WINDOW_WIDTH;
	const int WINDOW_HEIGHT;
	const int MAX_ITERS;

	SDL_Event event;
	SDL_Window* window = nullptr;
	SDL_Surface* fracSurface = nullptr;
	SDL_Surface* windowSurface = nullptr;
	SDL_Surface* ttfSurface = nullptr;
	TTF_Font* font = nullptr;

	fracState* state = nullptr;
	fracState* d_state = nullptr;

	//This 1d array will be used to store the RGB values of every pixel
	RGB* result = nullptr;

	//Creating 1d result array in GPU mem
	RGB* d_result = nullptr;

	//This will be true if there was an error during SDL or SDL_ttf
	bool initError = false;

	/*Creates and initializes our CPU thread pool. Threads will be properly joined when the
	fracThreadPool is deallocated*/
	fracThreadPool* threadPool;

	//Used to help keep track of panning
	SDL_Point mousePanningCoords;
};