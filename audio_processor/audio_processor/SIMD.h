#pragma once

#include<cstdio>
#include <arm_neon.h>

namespace simd {

	class Constants {
	public:
		static const int MAXN = 100000;
		static int numbers[MAXN];
		static int ones[MAXN];

		Constants()
		{
			for (int i = 0; i < MAXN; i++)
				numbers[i] = i, ones[i] = 1;
		}
	} constants;

	void test()
	{
		float a[] = { 1,2,3,4 };
		float b[] = { 0.10,0.20,0.30,0.40 };
		int n[4];
		float32x4_t va = vld1q_f32(a);
		float32x4_t vb = vld1q_f32(b);
		float32x4_t vc = vaddq_f32(va, vb);
		vst1q_f32(a, vc);
		printf("%f %f %f %f\n", a[0], a[1], a[2], a[3]);
		int32x4_t vint = vcvtq_s32_f32(va);
		vst1q_s32(n, vint);
		printf("%d %d %d %d\n", n[0], n[1], n[2], n[3]);
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