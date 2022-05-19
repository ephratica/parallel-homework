#include "mypthread.h"
#include <iostream>
#include <pthread.h>
#include <windows.h>
#pragma comment(lib,"pthreadVC2.lib")

double test(int numThreads, bool omp = true)
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
	t = omptime_scale(data, len, 1 / 0.8, numThreads, true);
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
double test_nosimd(int numThreads, bool omp = true)
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
	t = omptime_scale(data, len, 1 / 0.8, numThreads, false);
	data = t.first;
	len = t.second;
	t = filter(data, len);
	data = t.first;
	len = t.second;

	double timeUsed = (double)(clock() - start) / CLOCKS_PER_SEC;

	return timeUsed;
}
double ptest(int numThreads)
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
double ptest_nosimd(int numThreads)
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
	test(16);
	test(16);
	return 0;

	printf("%lf %lf\n", ptest(16), test(16));
	return 0;
	int testCases[] = { 1,2,3,4,5,6,7,8,9,10 };
	double t[100] = {}, t2[100] = {}, t3[100] = {}, t4[100] = {};

	for (int i = 0; i < sizeof(testCases) / sizeof(int); i++)
	{
		int n = 5;
		for (int j = 0; j < n; j++)
			t[i] += ptest(testCases[i]), t2[i] += ptest_nosimd(testCases[i]),
			t3[i] += test(testCases[i]), t4[i] += test_nosimd(testCases[i]);
		t[i] = t[i] / n * 1000;
		t2[i] = t2[i] / n * 1000;
		t3[i] = t3[i] / n * 1000;
		t4[i] = t4[i] / n * 1000;
		printf("%d threads: %lf, %lf, %lf, %lf ms\n", testCases[i], t[i], t2[i], t3[i], t4[i]);
	}
	for (int i = 0; i < sizeof(testCases) / sizeof(int); i++)
		printf("%.2lf\t", t[i]);
	printf("\n");
	for (int i = 0; i < sizeof(testCases) / sizeof(int); i++)
		printf("%.2lf\t", t2[i]);
	printf("\n");
	for (int i = 0; i < sizeof(testCases) / sizeof(int); i++)
		printf("%.2lf\t", t3[i]);
	printf("\n");
	for (int i = 0; i < sizeof(testCases) / sizeof(int); i++)
		printf("%.2lf\t", t4[i]);
	printf("\n");

	return 0;
}