#include "mypthread.h"
#include <iostream>
#include <pthread.h>
#include <windows.h>
#pragma comment(lib,"pthreadVC2.lib")

double test(int numThreads)
{
	float *data;
	int len;
	auto t = generate();// read((char *)"record2.wav");
	data = t.first;
	len = t.second;

	clock_t start = clock();

	simd::stretch_prepare(len, len * 0.8);
	t = simd::stretch(data, len, len * 0.8);
	data = t.first;
	len = t.second;
	t = ptime_scale(data, len, 1 / 0.8, numThreads, true);
	data = t.first;
	len = t.second;
	t = simd::filter(data, len);
	data = t.first;
	len = t.second;

	double timeUsed = (double)(clock() - start) / CLOCKS_PER_SEC;

	//write((char *)"record2.wav",
	//	(char *)"record2_test_dft.wav", data, len);

	return timeUsed;
}
double test_nosimd(int numThreads)
{
	float *data;
	int len;
	auto t = generate();
	data = t.first;
	len = t.second;

	clock_t start = clock();// printf("test start\n");

	t = stretch(data, len, len * 0.8);
	data = t.first;
	len = t.second;
	t = ptime_scale(data, len, 1 / 0.8, numThreads, false);
	data = t.first;
	len = t.second;
	t = filter(data, len);
	data = t.first;
	len = t.second;

	double timeUsed = (double)(clock() - start) / CLOCKS_PER_SEC;

	return timeUsed;
}

int main() 
{
	int testCases[] = { 1,2,3,4,5,6,7,8,10,12,14,16 };
	double t[100] = {}, t2[100] = {};

	for (int i = 0; i < sizeof(testCases) / sizeof(int); i++)
	{
		int n = 5;
		for (int j = 0; j < n; j++)
			t[i] += test(testCases[i]), t2[i] += test_nosimd(testCases[i]);
		printf("%d threads: %lf, %lf ms\n", testCases[i], t[i] / n * 1000, t2[i] / n * 1000);
	}

	return 0;
}