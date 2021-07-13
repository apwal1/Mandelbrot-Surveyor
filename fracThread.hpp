#pragma once

#include <SDL.h>
#include <thread>
#include <mutex>
#include <vector>
#include <complex>
#include "fracState.hpp"

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
	fracThread(int width, int height, int iterations, pair<SDL_Point, SDL_Point> bounds, RGB* arr, const fracState* state);
	void run();
	void join();
	void waitUntilDone();
	bool joinable();
private:
	void makeFractal(RGB* resultArr);
	void coordsToComplex(const int* x, const int* y, complex<double>* result);
	void getSmoothColor(const complex<double>* complexNum, double* smooth);
	void calcSmoothColor(const complex<double>* complexNum, const int* iters, double* smooth);
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