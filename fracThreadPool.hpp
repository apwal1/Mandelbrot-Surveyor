#pragma once
#include "fracThread.hpp"
using std::vector;
class fracThreadPool
{
public:
	fracThreadPool(unsigned int threadCount, RGB* resultArr, const fracState& state);
	fracThreadPool(const fracThreadPool& rhs);
	virtual ~fracThreadPool();
	virtual void calcFrame();
private:
	vector<fracThread*> threads;
};