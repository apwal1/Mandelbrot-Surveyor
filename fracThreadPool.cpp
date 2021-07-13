#include "fracThreadPool.hpp"
//Creates and starts the threads in our threadpool, threads will be idle until calcFrame is called
fracThreadPool::fracThreadPool(unsigned int threadCount, RGB* resultArr, const fracState* state)
{
	for (int i = 0; i < threadCount; i++)
	{
		SDL_Point start = { i * WINDOW_WIDTH / threadCount, 0 };
		SDL_Point end = { (i + 1) * WINDOW_WIDTH / threadCount, WINDOW_HEIGHT };
		pair<SDL_Point, SDL_Point> bounds(start, end);
		threads.push_back(new fracThread(WINDOW_WIDTH, WINDOW_HEIGHT, MAX_ITER, bounds, resultArr, state));
	}
}

//Properly deallocates threads (if the result array is still pointing to valid, allocated memory)
fracThreadPool::~fracThreadPool()
{
	for (auto i : threads)
	{
		/*This will cause an access violation exceotion if the result array has been deallocated
		before this destructor is called, which indicates to the user of this class that something wrong*/
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

//Properly makes threads exit (if the result array is still pointing to valid, allocated memory)
void fracThreadPool::joinThreads()
{
	/*This will cause an access violation exceotion if the result array has been deallocated
	before this function is called, which indicates to the user of this class that something wrong*/
	for (auto i : threads)
		i->join();
}