#include "fracThreadPool.hpp"
//Creates and starts the threads in our threadpool, threads will be idle until calcFrame is called
fracThreadPool::fracThreadPool(unsigned int threadCount, RGB* resultArr, const fracState* state)
{
	for (int i = 0; i < threadCount; i++)
	{
		SDL_Point start = { i * state->windowWidth / threadCount, 0 };
		SDL_Point end = { (i + 1) * state->windowWidth / threadCount, state->windowHeight };
		pair<SDL_Point, SDL_Point> bounds(start, end);
		threads.push_back(new fracThread(state->windowWidth, state->windowHeight, state->maxIters, bounds, resultArr, state));
	}
}

//Properly joins and deallocates threads
fracThreadPool::~fracThreadPool()
{
	for (auto i : threads)
	{
		if (i->joinable())
			i->join();

		delete i;
	}
}

//Calculates a single frame of our fractal
void fracThreadPool::calcFrame()
{
	for (auto i : threads)
		i->run();
	for (auto i : threads)
		i->waitUntilDone();
}