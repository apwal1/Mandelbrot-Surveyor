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

class fracThread
{
public:
	fracThread(double width, double height, int iterations, SDL_Point startCoord, SDL_Point endCoord, int** arr);
	void makeFractal(int** iterVec);
	void run(const double* xOffset, const double* yOffset, const double* xZoom, const double* yZoom);
	void join();
	bool isThreadDone();
private:
	void coordsToComplex(const int* x, const int* y, complex<double>* result);
	int getNumIters(const complex<double>* complexNum);
	thread subThread;
	condition_variable start;
	const double windowWidth = 0;
	const double windowHeight = 0;
	const double maxIters = 0;
	const SDL_Point startPoint = { 0, 0 };
	const SDL_Point endPoint = { 0, 0 };
	const double* xOffset;
	const double* yOffset;
	const double* xZoom;
	const double* yZoom;
	atomic<bool> running = true;
	atomic<bool> done = false;
	mutex mx;
};