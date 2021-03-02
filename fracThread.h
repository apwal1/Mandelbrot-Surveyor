#pragma once

#include <SDL.h>
#include <thread>
#include <mutex>
#include <vector>
#include <complex>

using std::complex;
using std::thread;
using std::mutex;
using std::vector;
using std::atomic;
using std::condition_variable;
using std::pair;

//Used to keep track of pan/zoom state of the fractal
struct fracState {
	double xZoomScale;
	double yZoomScale;
	double xPanOffset;
	double yPanOffset;

	//Initializes the starting x and y offsets of the fractal
	fracState(int xPan, int yPan)
	{
		xPanOffset = xPan;
		yPanOffset = yPan;
		xZoomScale = xPan / -2;
		yZoomScale = yPan / -1.5;
	}
};

class fracThread
{
public:
	fracThread(int width, int height, int iterations, pair<SDL_Point, SDL_Point> bounds, int** arr, fracState* state);
	void makeFractal(int** iterVec);
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