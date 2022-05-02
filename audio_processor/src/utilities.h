#pragma once

#include "AudioFile.h"
#include "FFT.h"
#include <utility>
#include <math.h>

__pragma(warning(disable : 4996))

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

float *CtoF(Complex *c, int len)
{
	float *f = new float[len];
	for (int i = 0; i < len; i++)
		f[i] = c[i].x;
	return f;
}
Complex *FtoC(float *f, int len)
{
	Complex *c = new Complex[len];
	for (int i = 0; i < len; i++)
		c[i] = f[i];
	return c;
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

std::pair<float *, int> time_scale(float *data, int len, float rate, bool usefft = true)
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
		(usefft ? DFT::dft : DFT::mydft)(t, seglen);
		Complex *p = stretch(t, seglen, usefft?2*seglen:len_per_step).first;
		(usefft ? DFT::idft : DFT::myidft)(p, usefft ? 2 * seglen : len_per_step);
		for (int j = 0; j < len_per_step; j++)
			ret[cur + j] = p[j].x;// sqrt(p[j].x * p[j].x + p[j].y * p[j].y);
		cur += len_per_step;
		delete[] t;
		delete[] p;
	}
	while (cur < newlen) ret[cur++] = 0;
	return { ret,newlen };
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
	const int len = 10000000;
	float *data = new float[len];
	for (int i = 0; i < len; i++)
		data[i] = rand();
	return { data,len };
}