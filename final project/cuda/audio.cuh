
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <utility>
#include <time.h>
#include "cufft.cuh"

//void fft(Complex *input_array, int lenth, int flag, cudaStream_t stream);

std::pair<Complex *, int> stretch(Complex *data, int len, int newlen)
{
	float rate = (float)newlen / len;
	Complex *newdata = new Complex[newlen];
	for (int i = 0; i < newlen; i++)
	{
		float pos = i / rate;
		int left = (int)pos, right = left + 1;
		newdata[i] = data[left] * (right - pos) + data[right] * (pos - left);
	}
	newdata[newlen - 1] = data[len - 1];
	return { newdata, newlen };
}
std::pair<float *, int> stretch(float *data, int len, int newlen)
{
	float rate = (float)newlen / len;
	float *newdata = new float[newlen];
	for (int i = 0; i < newlen; i++)
	{
		float pos = i / rate;
		int left = (int)pos, right = left + 1;
		newdata[i] = data[left] * (right - pos) + data[right] * (pos - left);
	}
	return { newdata, newlen };
}

std::pair<float *, int> omp_time_scale(float *data, int len, float rate, int n)
{		//  data为输入数组，len为输入数组长度，rate为拉伸比例，n为线程数
	len = len / 1024 * 1024;
	int num = len / 1024;
	int newlen = 1024 * rate;
	float *ret = new float[newlen * num];
	
	Complex *t = new Complex[len];
	for (int i = 0; i < len; i++)
		t[i].x = data[i], t[i].y = 0;
	fft(t, len, 1024, 1);
	t = stretch(t, len, len * 2).first;
	fft(t, len, 2048, -1);
	for (int i = 0; i < num; i++)
		for (int j = 0; j < newlen; j++)
			ret[i * newlen + j] = t[i * 2048 + j].x;
	return { ret,num * newlen };				//  返回输出数组的地址与长度
}

std::pair<float *, int> process(std::pair<float *, int> input, int num_threads, float rate = 1.25)
{
	auto t = input, tmp = t;
	rate = 1 / rate;

	t = stretch(t.first, t.second, t.second * rate); tmp = t;
	t = omp_time_scale(t.first, t.second, 1 / rate, num_threads); delete[] tmp.first;

	return t;
}

std::pair<float *, int> generate(int len)
{
	float *data = new float[len];
	for (int i = 0; i < len; i++)
		data[i] = rand();
	return { data,len };
}

int test(int len, int num_threads)
{
	auto input = generate(len);
	int times = 1;
	time_t start = clock();
	while (times--)
	{
		delete[] process(input, num_threads).first;
	}
	return (int)((double)(clock() - start) / CLOCKS_PER_SEC * 1000);
}