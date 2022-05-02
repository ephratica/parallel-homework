#pragma once

#include "AudioFile.h"
#include "FFT.h"
#include <utility>
#include <math.h>
#include <immintrin.h>
#include <pthread.h>
#pragma comment(lib,"pthreadVC2.lib")

__pragma(warning(disable : 4996))

class Constants {
public:
	static const int MAXN = 3000000;
	int *numbers;
	int *ones;

	Constants()
	{
		numbers = new int[MAXN];
		ones = new int[MAXN];
		for (int i = 0; i < MAXN; i++)
			numbers[i] = i, ones[i] = 1;
	}
	~Constants() { delete[] numbers, delete[] ones; }
} constants;

namespace simd {

	std::pair<float *, int> filter(float *data, int len)
	{
		float *ret = new float[len + 4];
		memset(ret, 0, sizeof(float) * (len + 4));
		ret += 2;
		int len8 = len / 8 * 8;
		for (int i = -2; i <= 2; i++)
		{
			for (int j = 0; j < len8; j += 8)
			{
				__m256 vdata = _mm256_load_ps(data + j);
				__m256 vret = _mm256_load_ps(ret + j + i);
				vret = _mm256_add_ps(vret, vdata);
				_mm256_store_ps(ret + j + i, vret);
			}
			for (int j = len8; j < len; j++)
				ret[j + i] += data[j];
		}
		float *tmp = new float[len];
		int lim = 2 + (len - 4) / 8 * 8;
		for (int i = 2; i < lim; i += 8)
		{
			__m256 vret = _mm256_load_ps(ret + i);
			vret = _mm256_mul_ps(vret, _mm256_set1_ps(0.2));
			_mm256_store_ps(tmp + i, vret);
		}
		for (int i = lim; i < len - 2; i++)
			tmp[i] = ret[i] / 5;
		tmp[1] = ret[1] / 4, tmp[len - 2] = ret[len - 2] / 4;
		tmp[0] = ret[0] / 3, tmp[len - 1] = ret[len - 1] / 3;
		delete[](ret - 2);
		return { tmp,len };
	}

	float *tmpl, *tmpr;
	void stretch_prepare(int len, int newlen)
	{
		float rate = (float)newlen / len;
		tmpl = new float[newlen];
		tmpr = new float[newlen];
		for (int i = 0; i < newlen; i++)
		{
			float pos = i / rate;
			tmpl[i] = pos - (int)pos;
			tmpr[i] = 1 - tmpl[i];
		}
	}
	std::pair<float *, int> stretch(float *data, int len, int newlen)
	{
		float rate = (float)newlen / len, _rate = 1 / rate;
		float *newdata = new float[newlen];
		int newlen8 = newlen / 8 * 8;
		float tmp[8];
		for (int i = 0; i < newlen8; i += 8)
		{
			__m256 vl = _mm256_load_ps(tmpl + i);
			__m256 vr = _mm256_load_ps(tmpr + i);

			int l0 = i / rate, l1 = (i + 1) / rate, l2 = (i + 2) / rate, l3 = (i + 3) / rate,
				l4 = (i + 4) / rate, l5 = (i + 5) / rate, l6 = (i + 6) / rate, l7 = (i + 7) / rate;
			tmp[0] = data[l0]; tmp[1] = data[l1];
			tmp[2] = data[l2]; tmp[3] = data[l3];
			tmp[4] = data[l4]; tmp[5] = data[l5];
			tmp[6] = data[l6]; tmp[7] = data[l7];
			__m256 vdataleft = _mm256_load_ps(tmp);
			tmp[0] = data[l0 + 1]; tmp[1] = data[l1 + 1];
			tmp[2] = data[l2 + 1]; tmp[3] = data[l3 + 1];
			tmp[4] = data[l4 + 1]; tmp[5] = data[l5 + 1];
			tmp[6] = data[l6 + 1]; tmp[7] = data[l7 + 1];
			__m256 vdataright = _mm256_load_ps(tmp);

			__m256 vterm1 = _mm256_mul_ps(vdataleft, vr);
			__m256 vterm2 = _mm256_fmadd_ps(vdataright, vl, vterm1);
			_mm256_store_ps(newdata + i, vterm2);
		}
		for (int i = newlen8; i < newlen; i++)	//下面处理边界情形
		{
			float pos = i / rate;
			int left = (int)pos, right = left + 1;
			newdata[i] = data[left] * (right - pos) + data[right] * (pos - left);
		}
		newdata[newlen - 1] = data[len - 1];
		return { newdata, newlen };
	}
	std::pair<Complex *, int> stretch(Complex *data, int len, int newlen)
	{
		float *x = new float[len], *y = new float[len];
		Complex *ret = new Complex[newlen];
		for (int i = 0; i < len; i++)
			x[i] = data[i].x, y[i] = data[i].y;
		float *s = stretch(x, len, newlen).first;
		float *t = stretch(y, len, newlen).first;
		for (int i = 0; i < newlen; i++)
			ret[i] = { s[i],t[i] };
		delete[] x;
		delete[] y;
		delete[] s;
		delete[] t;
		return { ret,newlen };
	}
}

class DFT {
	static bool ispow2(int len)
	{
		while (len > 1)
			if (len & 1)
				return false;
			else
				len >>= 1;
		return true;
	}
public:
	static void mydft(Complex *data, int len)	//普通DFT
	{
		Complex *tmp = new Complex[len];
		memset(tmp, 0, sizeof(Complex) * len);
		for (int i = 0; i < len; i++)
		{
			Complex wni(cos(-2 * PI * i / len), sin(-2 * PI * i / len));
			Complex t(1, 0);
			for (int j = 0; j < len; j++)
				tmp[i] = tmp[i] + data[j] * t, t = t * wni;
		}
		for (int i = 0; i < len; i++)
			data[i] = tmp[i];
		delete[] tmp;
	}
	static void myidft(Complex *data, int len)	//普通DFT
	{
		Complex *tmp = new Complex[len];
		memset(tmp, 0, sizeof(Complex) * len);
		for (int i = 0; i < len; i++)
		{
			Complex wni(cos(2 * PI * i / len), sin(2 * PI * i / len));
			Complex t(1, 0);
			for (int j = 0; j < len; j++)
				tmp[i] = tmp[i] + data[j] * t, t = t * wni;
		}
		for (int i = 0; i < len; i++)
			data[i] = (1.0 / len) * tmp[i];
		delete[] tmp;
	}
	static void dft(Complex *data, int len)		// fft的名字被抢了 QAQ
	{
		assert(ispow2(len));
		fft(data, len, 1);
	}
	static void idft(Complex *data, int len)		// fft的名字被抢了 QAQ
	{
		assert(ispow2(len));
		fft(data, len, -1);
	}
};

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
std::pair<float *, int> filter(float *data, int len)
{
	float *ret = new float[len];
	memset(ret, 0, sizeof(float) * len);
	for (int i = -2; i <= 2; i++)
		for (int j = 0; j < len; j++)
			if (j + i >= 0 && j + i < len)
				ret[j + i] += data[j];
	for (int i = 2; i < len - 2; i++)
		ret[i] /= 5;
	ret[1] /= 4, ret[len - 2] /= 4;
	ret[0] /= 3, ret[len - 1] /= 3;
	return { ret,len };
}


bool compare(float *a, float *b, int len)
{
	for (int i = 0; i < len; i++)
		if (fabs(a[i] - b[i]) > 1e-2)
			return false;
	return true;
}
bool compare(Complex *a, Complex *b, int len)
{
	for (int i = 0; i < len; i++)
		if (fabs(a[i].x - b[i].x) + fabs(a[i].y - b[i].y) > 1e-1)
			return false;
	return true;
}

std::pair<float *, int> time_scale(float *data, int len, float rate, bool useFFT = true)
{
	const int seglen = 1024;
	len = len / seglen * seglen;
	assert(rate > 0 && rate < 2);

	if (rate < 1)
	{
		int newlen = len * rate;
		float *newdata = new float[newlen];
		int p = 0, t = 1024 * rate;
		for (int i = 0; i < len; i += seglen)
			for (int j = 0; j < t; j++)
				newdata[p++] = data[i + j];
		while (p < newlen) newdata[p++] = 0;
		return { newdata,newlen };
	}

	int newlen = len * rate;
	float *ret = new float[newlen];
	int cur = 0, len_per_step = seglen * rate;

	simd::stretch_prepare(seglen, useFFT ? 2048 : len_per_step);

	for (int i = 0; i < len; i += seglen)
	{
		Complex *t = new Complex[seglen];
		for (int j = 0; j < seglen; j++)
			t[j] = data[i + j];
		(useFFT ? DFT::dft : DFT::mydft)(t, seglen);
		
		Complex *p = simd::stretch(t, seglen, useFFT ? 2048 : len_per_step).first;

		(useFFT ? DFT::idft : DFT::myidft)(p, useFFT ? 2048 : len_per_step);
		for (int j = 0; j < len_per_step; j++)
			ret[cur + j] = p[j].x;
		cur += len_per_step;
		delete[] t;
		delete[] p;
	}
//	while (cur < newlen) ret[cur++] = 0;
	return { ret,cur };
}
std::pair<float *, int> time_scale_nosimd(float *data, int len, float rate, bool useFFT = true)
{
	const int seglen = 1024;
	len = len / seglen * seglen;
	assert(rate > 0 && rate < 2);

	if (rate < 1)
	{
		int newlen = len * rate;
		float *newdata = new float[newlen];
		int p = 0, t = 1024 * rate;
		for (int i = 0; i < len; i += seglen)
			for (int j = 0; j < t; j++)
				newdata[p++] = data[i + j];
		while (p < newlen) newdata[p++] = 0;
		return { newdata,newlen };
	}

	int newlen = len * rate;
	float *ret = new float[newlen];
	int cur = 0, len_per_step = seglen * rate;

	for (int i = 0; i < len; i += seglen)
	{
		Complex *t = new Complex[seglen];
		for (int j = 0; j < seglen; j++)
			t[j] = data[i + j];
		(useFFT ? DFT::dft : DFT::mydft)(t, seglen);

		Complex *p = stretch(t, seglen, useFFT ? 2048 : len_per_step).first;

		(useFFT ? DFT::idft : DFT::myidft)(p, useFFT ? 2048 : len_per_step);
		for (int j = 0; j < len_per_step; j++)
			ret[cur + j] = p[j].x;
		cur += len_per_step;
		delete[] t;
		delete[] p;
	}
	return { ret,cur };
}

struct run_args {	//	线程参数
	float *data;	//  需要处理的数据的起始地址
	int len;		//  需要处理的数据的长度
	float rate;		//  需要拉长的比例
	float *out;		//  储存输出数列的长度
	bool useSIMD;	//  是否使用SIMD（用于测试）
};
void *run(void *arg)//  每个线程执行的函数，对串行算法进行了简单包装
{
	run_args *args = (run_args *)arg;
	auto t = (args->useSIMD ? time_scale : time_scale_nosimd)(args->data, args->len, args->rate, true);
	for (int i = 0; i < t.second; i++)
		args->out[i] = t.first[i];
	delete[] t.first;
	return NULL;
}
std::pair<float *, int> ptime_scale(float *data, int len, float rate, int n, bool useSIMD)
{		//  data为输入数组，len为输入数组长度，rate为拉伸比例，n为线程数，useSIMD表示是否使用SIMD
	len = len / 1024 * 1024;
	int num = len / 1024;					//  将输入切割为适合处理的片段，num为片段数
	int step = (num + n - 1) / n * 1024;	//  每个线程需要处理的子数列的长度，保证它是1024的倍数
	int newlen = 0;
	for (int i = 0; i < len; i += step)		//  计算输出的长度
		newlen += std::min(step, len - i) / 1024 * floor(1024 * rate);
	float *ret = new float[newlen];			//  为输出数列分配空间
	float *cur = ret;
	pthread_t *tids = new pthread_t[n]; int t = 0;
	run_args *args = new run_args[n];
	for (int i = 0; i < len; i += step, t++)
	{
		args[t] = { data + i,std::min(step, len - i),rate,cur,useSIMD };	//  确定参数
		cur += (int)(std::min(step, len - i) / 1024 * floor(1024 * rate));
		pthread_create(tids + t, NULL, run, args + t);						//  创建线程
	}
	
	for(int i = 0; i < n; i++)
		pthread_join(tids[i],NULL);		//  等待所有线程结束
	delete[] tids;
	delete[] args;
	return { ret,newlen };				//  返回输出数组的地址与长度
}

std::pair<float *, int> read(char *path)
{
	AudioFile<float> f(path);
	int numSamples = f.getNumSamplesPerChannel();
	float *data = new float[numSamples];
	for (int i = 0; i < numSamples; i++)
		data[i] = f.samples[0][i];
	return { data,numSamples };
}

void write(char *oldpath, char *newpath, float *data, int len)
{
	AudioFile<float> f(oldpath);
	AudioFile<float>::AudioBuffer buffer;
	buffer.resize(2);
	buffer[0].resize(len);
	buffer[1].resize(len);
	for (int i = 0; i < len; i++)
		buffer[0][i] = buffer[1][i] = data[i];
	f.setAudioBuffer(buffer);
	f.save(newpath);
}

std::pair<float *, int> generate()
{
	const int len = 3000000;
	float *data = new float[len];
	for (int i = 0; i < len; i++)
		data[i] = rand();
	return { data,len };
}