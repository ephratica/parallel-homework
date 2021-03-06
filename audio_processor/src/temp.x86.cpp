#include <immintrin.h>
#include <cstdio>
#include <utility>
#include <cstring>
#include <vector>
#include <time.h>
#include "utilities.h"

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
		for (int i = newlen8; i < newlen; i++)	//????????????????
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

//std::pair<float *, int> stretch(float *data, int len, int newlen)
//{
//	float rate = (float)newlen / len;
//	float *newdata = new float[newlen];
//	for (int i = 0; i < newlen; i++)
//	{
//		float pos = i / rate;
//		int left = (int)pos, right = left + 1;
//		newdata[i] = data[left] * (right - pos) + data[right] * (pos - left);
//	}
//	return { newdata, newlen };
//}
//
//std::pair<float *, int> filter(float *data, int len)
//{
//	float *ret = new float[len];
//	memset(ret, 0, sizeof(float) * len);
//	for (int i = -2; i <= 2; i++)
//		for (int j = 0; j < len; j++)
//			if (j + i >= 0 && j + i < len)
//				ret[j + i] += data[j];
//	for (int i = 2; i < len - 2; i++)
//		ret[i] /= 5;
//	ret[1] /= 4, ret[len - 2] /= 4;
//	ret[0] /= 3, ret[len - 1] /= 3;
//	return { ret,len };
//}

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
		simd::stretch2_prepare(test.len, test.newlen);
		t3.push_back(test(simd::stretch2));
	}
	printf("   stretch_no_simd: ");
	for (float t : t1)
		printf("%f ", t);
	printf("\nstretch_with_simd2: ");
	for (float t : t3)
		printf("%f ", t);
	printf("\n");
}

void test_correctness()
{
	float data[] = { 1,2,3,4 };
	int len = 4, newlen = 7;
	simd::stretch2_prepare(len, newlen);
	auto r = simd::stretch2(data, len, newlen);
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
	//test_correctness();

	return 0;
}