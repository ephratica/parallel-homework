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
Complex *double_length(Complex *c, int len)
{
	int len2 = 2 * len;
	Complex *c2 = new Complex[len2];
	for (int i = 0; i < len - 1; i++)
		c2[i * 2] = c[i], c2[i * 2 + 1] = c[i];// 0.5 * (c[i] + c[i + 1]);
	c2[len * 2 - 2] = c2[len * 2 - 1] = c[len - 1];
	return c2;
}

std::pair<float *, int> stretch(float *data, int len, float rate)
{
	int newlen = len * rate;
	float *newdata = new float[newlen];
	for (int i = 0; i < newlen; i++)
		newdata[i] = data[(int)(i / rate)];
	return { newdata, newlen };
}

std::pair<float *, int> time_scale(float *data, int len, float rate)
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
//	rate = 2;
	int newlen = len * rate;
	float *ret = new float[newlen];
	int cur = 0, len_per_step = seglen * rate;
	for (int i = 0; i < len; i += seglen)
	{
		Complex *t = new Complex[seglen];
		for (int j = 0; j < seglen; j++)
			t[j] = data[i + j];
		DFT::dft(t, seglen);
		Complex *p = double_length(t, seglen);
		DFT::idft(p, seglen * 2);
		for (int j = 0; j < len_per_step; j++)
			ret[cur + j] = p[j].x;// sqrt(p[j].x * p[j].x + p[j].y * p[j].y);
		cur += len_per_step;
		delete[] t;
		delete[] p;
	}
	while (cur < newlen) ret[cur++] = 0;
	return { ret,newlen };
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