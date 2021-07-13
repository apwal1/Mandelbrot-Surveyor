#pragma once
#include "fracThread.hpp"
using std::vector;
class fracThreadPool
{
public:
	fracThreadPool(unsigned int threadCount, RGB* resultArr, const fracState* state);
	~fracThreadPool();
	void calcFrame();
	void joinThreads();
private:
	vector<fracThread*> threads;
};