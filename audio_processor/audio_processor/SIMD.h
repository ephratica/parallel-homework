#pragma once

#include<cstdio>
#include <arm_neon.h>

class Constants {
public:
	static const int MAXN = 1000000;
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
		int len4 = len / 4 * 4;
		for (int i = -2; i <= 2; i++)
		{
			for (int j = 0; j < len4; j += 4)
			{
				float32x4_t vdata = vld1q_f32(data + j);
				float32x4_t vret = vld1q_f32(ret + j + i);
				vret = vaddq_f32(vret, vdata);
				vst1q_f32(ret + j + i, vret);
			}
			for (int j = len4; j < len; j++)
				ret[j + i] += data[j];
		}
		int lim = 2 + (len - 4) / 4 * 4;
		for (int i = 2; i < lim; i += 4)
		{
			float32x4_t vret = vld1q_f32(ret + i);
			vret = vmulq_n_f32(vret, 0.2);
			vst1q_f32(ret + i, vret);
		}
		for (int i = lim; i < len - 2; i++)
			ret[i] /= 5;
		ret[1] /= 4, ret[len - 2] /= 4;
		ret[0] /= 3, ret[len - 1] /= 3;
		return { ret,len };
	}

	std::pair<float *, int> stretch(float *data, int len, int newlen)
	{
		float rate = (float)newlen / len, _rate = 1 / rate;
		float *newdata = new float[newlen];
		int newlen4 = newlen / 4 * 4;
		int32x4_t vone = vld1q_s32(constants.ones);
		int tmpi[4];
		float tmpf[4];
		for (int i = 0; i < newlen4; i += 4)
		{
			int32x4_t vint = vld1q_s32(constants.numbers + i);
			float32x4_t vpos = vcvtq_f32_s32(vint);
			vpos = vmulq_n_f32(vpos, _rate);
			int32x4_t vintl = vcvtq_s32_f32(vpos);
			int32x4_t vintr = vaddq_s32(vintl, vone);
			float32x4_t vfl = vcvtq_f32_s32(vintl);
			float32x4_t vfr = vcvtq_f32_s32(vintr);
			vfl = vsubq_f32(vpos, vfl);
			vfr = vsubq_f32(vfr, vpos);

			vst1q_s32(tmpi, vintl);
			tmpf[0] = data[tmpi[0]];
			tmpf[1] = data[tmpi[1]];
			tmpf[2] = data[tmpi[2]];
			tmpf[3] = data[tmpi[3]];
			float32x4_t vdataleft = vld1q_f32(tmpf);
			vst1q_s32(tmpi, vintr);
			tmpf[0] = data[tmpi[0]];
			tmpf[1] = data[tmpi[1]];
			tmpf[2] = data[tmpi[2]];
			tmpf[3] = data[tmpi[3]];
			float32x4_t vdataright = vld1q_f32(tmpf);

			float32x4_t vterm1 = vmulq_f32(vdataleft, vfr);
			float32x4_t vterm2 = vmulq_f32(vdataright, vfl);
			vterm1 = vaddq_f32(vterm1, vterm2);
			vst1q_f32(newdata + i, vterm1);
		
		}
		for (int i = newlen4; i < newlen; i++)
		{
			float pos = i / rate;
			int left = (int)pos, right = left + 1;
			newdata[i] = data[left] * (right - pos) + data[right] * (pos - left);
		}
		newdata[newlen - 1] = data[len - 1];
		return { newdata, newlen };
	}

}

class Test {
	float *data;
	int len;
	int newlen;
	int times;
public:
	Test(int n, int m, int times)
	{
		len = n;
		newlen = m;
		this->times = times;
		data = new float[n];
		for (int i = 0; i < n; i++)
			data[i] = sin(i);
	}
	~Test() { delete[] data; }

	double operator () (std::pair<float *, int> (*f)(float *, int, int))
	{
		long long st = clock();
		for (int i = 0; i < times; i++)
			f(data, len, newlen);
		return (1.0 * clock() - st) / CLOCKS_PER_SEC;
	}
};