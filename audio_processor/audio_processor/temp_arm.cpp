#include<cstdio>
#include<arm_neon.h>
#include<utility>
#include<time.h>
#include<math.h>
#include<iostream>
#include<cstring>
#include<vector>

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
		float *tmp = new float[len];
		int lim = 2 + (len - 4) / 4 * 4;
		for (int i = 2; i < lim; i += 4)
		{
			float32x4_t vret = vld1q_f32(ret + i);
			vret = vmulq_n_f32(vret, 0.2);
			vst1q_f32(tmp + i, vret);
		}
		for (int i = lim; i < len - 2; i++)
			tmp[i] = ret[i] / 5;
		tmp[1] = ret[1] / 4, tmp[len - 2] = ret[len - 2] / 4;
		tmp[0] = ret[0] / 3, tmp[len - 1] = ret[len - 1] / 3;
		delete[](ret - 2);
		return { tmp,len };
	}

	float *tmpl, *tmpr;
	void stretch2_prepare(int len, int newlen)
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
	std::pair<float *, int> stretch2(float *data, int len, int newlen)
	{
		float rate = (float)newlen / len, _rate = 1 / rate;
		float *newdata = new float[newlen];
		int newlen4 = newlen / 4 * 4;
		float tmp[4];
		for (int i = 0; i < newlen4; i += 4)
		{
			float32x4_t vl = vld1q_f32(tmpl + i);
			float32x4_t vr = vld1q_f32(tmpr + i);
			
			int l0 = i / rate, l1 = (i + 1) / rate, l2 = (i + 2) / rate, l3 = (i + 3) / rate;
			tmp[0] = data[l0]; tmp[1] = data[l1];
			tmp[2] = data[l2]; tmp[3] = data[l3];
			float32x4_t vdataleft = vld1q_f32(tmp);
			tmp[0] = data[l0 + 1]; tmp[1] = data[l1 + 1];
			tmp[2] = data[l2 + 1]; tmp[3] = data[l3 + 1];
			float32x4_t vdataright = vld1q_f32(tmp);

			float32x4_t vterm1 = vmulq_f32(vdataleft, vr);
			float32x4_t vterm2 = vmlaq_f32(vterm1, vdataright, vl);
			vst1q_f32(newdata + i, vterm2);
		}
		for (int i = newlen4; i < newlen; i++)	//下面处理边界情形
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
		float rate = (float)newlen / len, _rate = 1 / rate;
		float *newdata = new float[newlen];
		int newlen4 = newlen / 4 * 4;
		int32x4_t vone = vld1q_s32(constants.ones);
		int tmpi[4];
		float tmpf[4];
		for (int i = 0; i < newlen4; i += 4)
		{
			int32x4_t vint = vld1q_s32(constants.numbers + i);	//numbers[i] = i
			float32x4_t vpos = vcvtq_f32_s32(vint);
			vpos = vmulq_n_f32(vpos, _rate);	//_rate = 1 / rate
			int32x4_t vleft = vcvtq_s32_f32(vpos);	//floor(i / rate)
			int32x4_t vright = vaddq_s32(vleft, vone);
			float32x4_t vfl = vcvtq_f32_s32(vleft);
			float32x4_t vfr = vcvtq_f32_s32(vright);
			vfl = vsubq_f32(vpos, vfl);
			vfr = vsubq_f32(vfr, vpos);

		/*	float32x4_t vdataleft;
			vdataleft = vsetq_lane_f32(data[vgetq_lane_s32(vleft, 0)], vdataleft, 0);
			vdataleft = vsetq_lane_f32(data[vgetq_lane_s32(vleft, 1)], vdataleft, 1);
			vdataleft = vsetq_lane_f32(data[vgetq_lane_s32(vleft, 2)], vdataleft, 2);
			vdataleft = vsetq_lane_f32(data[vgetq_lane_s32(vleft, 3)], vdataleft, 3);
			float32x4_t vdataright;
			vdataright = vsetq_lane_f32(data[vgetq_lane_s32(vright, 0)], vdataright, 0);
			vdataright = vsetq_lane_f32(data[vgetq_lane_s32(vright, 1)], vdataright, 1);
			vdataright = vsetq_lane_f32(data[vgetq_lane_s32(vright, 2)], vdataright, 2);
			vdataright = vsetq_lane_f32(data[vgetq_lane_s32(vright, 3)], vdataright, 3);*/
			vst1q_s32(tmpi, vleft);
			tmpf[0] = data[tmpi[0]];
			tmpf[1] = data[tmpi[1]];
			tmpf[2] = data[tmpi[2]];
			tmpf[3] = data[tmpi[3]];
			float32x4_t vdataleft = vld1q_f32(tmpf);
			vst1q_s32(tmpi, vright);
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
public:
	float *data;
	int len;
	int newlen;
	int times;
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

	double operator () (std::pair<float *, int>(*f)(float *, int, int))
	{
		long long st = clock();
		for (int i = 0; i < times; i++) 
			delete[] f(data, len, newlen).first;
		return (1.0 * clock() - st) / CLOCKS_PER_SEC;
	}
	double operator () (std::pair<float *, int>(*f)(float *, int))
	{
		long long st = clock();
		for (int i = 0; i < times; i++)
			delete[] f(data, len).first;
		return (1.0 * clock() - st) / CLOCKS_PER_SEC;
	}
};

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

void test_time2()
{
	Test test(1000000, 1800000, 50);
	double t1 = test(stretch);
	simd::stretch2_prepare(test.len, test.newlen);
	double t2 = test(simd::stretch2);
	printf("%lf %lf\n", t1, t2);
}
void test_time()
{
	Test tests[] = { Test(1000, 1800, 50000), Test(10000, 18000, 5000),
		Test(100000, 180000, 500),Test(1000000, 1800000, 50) };
	std::vector<float> t1, t2;

	for (Test &test : tests) {
		t1.push_back(test(filter));
		t2.push_back(test(simd::filter));
	}
	printf("  filter_no_simd: ");
	for (float t : t1)
		printf("%f ", t);
	printf("\nfilter_with_simd: ");
	for (float t : t2)
		printf("%f ", t);
	printf("\n");

	t1.clear();
	t2.clear();
	std::vector<float> t3;
	for (Test &test : tests) {
		t1.push_back(test(stretch));
		t2.push_back(test(simd::stretch));
		simd::stretch2_prepare(test.len, test.newlen);
		t3.push_back(test(simd::stretch2));
	}
	printf("   stretch_no_simd: ");
	for (float t : t1)
		printf("%f ", t);
	printf("\nstretch_with_simd1: ");
	for (float t : t2)
		printf("%f ", t);
	printf("\nstretch_with_simd2: ");
	for (float t : t3)
		printf("%f ", t);
	printf("\n");
/*	auto r = simd::filter(test.data, test.len);
	auto r2 = filter(test.data, test.len);
	float sum = 0;
	for (int i = 0; i < test.len; i++)
		sum += fabs(r.first[i] - r2.first[i]);
	printf("avgdiff: %f\n", sum / test.len);*/
}

void test_correctness()
{
	float data[] = { 1,2,3,4 };
	int len = 4, newlen = 7;
	auto r = simd::stretch(data, len, newlen);
	auto r2 = stretch(data, len, newlen);
	for (int i = 0; i < len; i++)
		printf("%.1f%c", data[i], i == len - 1 ? '\n' : ' ');
	for (int i = 0; i < newlen; i++)
		printf("%.1f%c", r.first[i], i == newlen - 1 ? '\n' : ' ');
	for (int i = 0; i < newlen; i++)
		printf("%.1f%c", r2.first[i], i == newlen - 1 ? '\n' : ' ');
}

int main()
{
	test_time();
	/*test_correctness();*/

	return 0;
}