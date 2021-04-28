#pragma once

#include <SDL.h>
#include <thread>
#include <mutex>
#include <vector>
#include <complex>
#include "fracState.h"

using std::complex;
using std::thread;
using std::mutex;
using std::vector;
using std::atomic;
using std::condition_variable;
using std::pair;

class fracThread
{
public:
	fracThread(int width, int height, int iterations, pair<SDL_Point, SDL_Point> bounds, int* arr, fracState* state);
	void makeFractal(int* resultArr);
	void run();
	void join();
	void waitUntilDone();
private:
	void coordsToComplex(const int* x, const int* y, complex<double>* result);
	void getNumIters(const complex<double>* complexNum, int* iters);
	thread subThread;
	condition_variable start;
	const double windowWidth = 0;
	const double windowHeight = 0;
	const double maxIters = 0;
	const pair<SDL_Point, SDL_Point> sectionBounds = { { 0, 0 }, { 0, 0 } };
	const double* xOffset;
	const double* yOffset;
	const double* xZoom;
	const double* yZoom;
	atomic<bool> running = true;
	mutex mx;
};
