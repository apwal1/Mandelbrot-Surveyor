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
	fracThread(int width, int height, int iterations, pair<SDL_Point, SDL_Point> bounds, RGB* arr, const fracState& state);
	fracThread(const fracThread& rhs);
	void run();
	void join();
	void waitUntilDone();
	bool joinable() const;
private:
	void makeFractal();
	void coordsToComplex(const int* x, const int* y, complex<double>& result) const;
	void getSmoothColor(const complex<double>* complexNum, double& smooth) const;
	void calcSmoothColor(const complex<double>* complexNum, const int* iters, double& smooth) const;
	thread subThread;
	condition_variable start;
	int windowWidth = 0;
	int windowHeight = 0;
	double maxIters = 0;
	const pair<SDL_Point, SDL_Point> sectionBounds = { { 0, 0 }, { 0, 0 } };
	const double* xOffset;
	const double* yOffset;
	const double* xZoom;
	const double* yZoom;
	RGB* resultArr;
	atomic<bool> running = true;
	mutex mx;
};