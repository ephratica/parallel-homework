#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <immintrin.h>
#include <omp.h>
#include <assert.h>
#include <math.h>
#include <cstring>
#pragma warning(disable:4996)

#define PI 3.1415926
struct Complex {
	float x, y;
	Complex(float _x = 0, float _y = 0) :x(_x), y(_y) {}

};
Complex operator + (Complex a, Complex b) { return Complex(a.x + b.x, a.y + b.y); }
Complex operator - (Complex a, Complex b) { return Complex(a.x - b.x, a.y - b.y); }
Complex operator * (Complex a, Complex b) { return Complex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
void swap(Complex &a, Complex &b) {
	Complex temp = a;
	a = b;
	b = temp;
	return;
}
void change(Complex *y, int len) {
	int *rev = new int[len];
	rev[0] = 0;
	for (int i = 0; i < len; ++i) {
		rev[i] = rev[i >> 1] >> 1;
		if (i & 1) {
			rev[i] |= len >> 1;
		}
	}
	for (int i = 0; i < len; ++i) {
		if (i < rev[i]) {
			swap(y[i], y[rev[i]]);
		}
	}
	return;
}
Complex *fft(Complex *y, int len, int on) {
	change(y, len);
	for (int h = 2; h <= len; h <<= 1) {
		Complex wn(cos(2 * PI / h), sin(on * 2 * PI / h));
		for (int j = 0; j < len; j += h) {
			Complex w(1, 0);
			for (int k = j; k < j + h / 2; k++) {
				Complex u = y[k];
				Complex t = w * y[k + h / 2];
				y[k] = u + t;
				y[k + h / 2] = u - t;
				w = w * wn;
			}
		}
	}
	if (on == -1) {
		for (int i = 0; i < len; i++) {
			y[i].x /= len; y[i].y /= len;
		}
	}
	return y;
}

class Constants {
public:
	static const int MAXN = 30000000;
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
				__m256 vdata = _mm256_loadu_ps(data + j);
				__m256 vret = _mm256_loadu_ps(ret + j + i);
				vret = _mm256_add_ps(vret, vdata);
				_mm256_storeu_ps(ret + j + i, vret);
			}
			for (int j = len8; j < len; j++)
				ret[j + i] += data[j];
		}
		float *tmp = new float[len];
		int lim = 2 + (len - 4) / 8 * 8;
		for (int i = 2; i < lim; i += 8)
		{
			__m256 vret = _mm256_loadu_ps(ret + i);
			vret = _mm256_mul_ps(vret, _mm256_set1_ps(0.2));
			_mm256_storeu_ps(tmp + i, vret);
		}
		for (int i = lim; i < len - 2; i++)
			tmp[i] = ret[i] / 5;
		tmp[1] = ret[1] / 4, tmp[len - 2] = ret[len - 2] / 4;
		tmp[0] = ret[0] / 3, tmp[len - 1] = ret[len - 1] / 3;
		delete[](ret - 2);
		return { tmp,len };
	}

	float tmpl[Constants::MAXN], tmpr[Constants::MAXN];
	void stretch_prepare(int len, int newlen)
	{
		float rate = (float)newlen / len;
	//	tmpl = new float[newlen];
	//	tmpr = new float[newlen];
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
		int newlen8 = newlen / 8 * 8 - 8;
		float tmp[8];
		for (int i = 0; i < newlen8; i += 8)
		{
			__m256 vl = _mm256_loadu_ps(tmpl + i);
			__m256 vr = _mm256_loadu_ps(tmpr + i);

			int l0 = i / rate, l1 = (i + 1) / rate, l2 = (i + 2) / rate, l3 = (i + 3) / rate,
				l4 = (i + 4) / rate, l5 = (i + 5) / rate, l6 = (i + 6) / rate, l7 = (i + 7) / rate;
			tmp[0] = data[l0]; tmp[1] = data[l1];
			tmp[2] = data[l2]; tmp[3] = data[l3];
			tmp[4] = data[l4]; tmp[5] = data[l5];
			tmp[6] = data[l6]; tmp[7] = data[l7];
			__m256 vdataleft = _mm256_loadu_ps(tmp);
			tmp[0] = data[l0 + 1]; tmp[1] = data[l1 + 1];
			tmp[2] = data[l2 + 1]; tmp[3] = data[l3 + 1];
			tmp[4] = data[l4 + 1]; tmp[5] = data[l5 + 1];
			tmp[6] = data[l6 + 1]; tmp[7] = data[l7 + 1];
			__m256 vdataright = _mm256_loadu_ps(tmp);

			__m256 vterm1 = _mm256_mul_ps(vdataleft, vr);
			__m256 vterm2 = _mm256_fmadd_ps(vdataright, vl, vterm1);
			_mm256_storeu_ps(newdata + i, vterm2);
		}
		for (int i = newlen8; i < newlen; i++)	//下面处理边界情形
		{
			float pos = i / rate;
			int left = std::min((int)pos, len - 1), right = std::min(left + 1, len - 1);
			newdata[i] = data[left] * (right - pos) + data[right] * (pos - left);
		}
		//newdata[newlen - 1] = data[len - 1];
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
		int left = std::min((int)pos, len - 1), right = std::min(left + 1, len - 1);
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
        int left = std::min((int)pos, len - 1), right = std::min(left + 1, len - 1);
        newdata[i] = data[left] * (right - pos) + data[right] * (pos - left);
    }
    return { newdata, newlen };
}

void segment_time_scale(float *data, float *out, float rate, bool useFFT = true)
{
	const int len = 1024;
	int newlen = len * rate;
	Complex *t = new Complex[len];
	for (int j = 0; j < len; j++)
		t[j] = data[j];

	(useFFT ? DFT::dft : DFT::mydft)(t, len);
	Complex *p = simd::stretch(t, len, useFFT ? 2048 : newlen).first;
//	Complex *p = stretch(t, len, useFFT ? 2048 : newlen).first;
	(useFFT ? DFT::idft : DFT::myidft)(p, useFFT ? 2048 : newlen);

	for (int j = 0; j < newlen; j++)
		out[j] = p[j].x;
}
void segment_time_scale_nosimd(float *data, float *out, float rate, bool useFFT = true)
{
	const int len = 1024;
	int newlen = len * rate;
	Complex *t = new Complex[len];
	for (int j = 0; j < len; j++)
		t[j] = data[j];

	(useFFT ? DFT::dft : DFT::mydft)(t, len);
	Complex *p = stretch(t, len, useFFT ? 2048 : newlen).first;
	(useFFT ? DFT::idft : DFT::myidft)(p, useFFT ? 2048 : newlen);

	for (int j = 0; j < newlen; j++)
		out[j] = p[j].x;
}
std::pair<float *, int> omptime_scale(float *data, int len, float rate, int n, bool useSIMD)
{		//  data为输入数组，len为输入数组长度，rate为拉伸比例，n为线程数，useSIMD表示是否使用SIMD
//	fprintf(stderr, "omp: len = %d\n", len);
	assert(len % 1024 == 0);
	len = len / 1024 * 1024;
	int num = len / 1024;
	int newlen = 1024 * rate;
	float *ret = new float[newlen * num];

	bool useFFT = true;
	simd::stretch_prepare(1024, useFFT ? 2048 : newlen);
#pragma omp parallel for num_threads(n)
	for (int i = 0; i < num; i++)
	{
		(useSIMD ? segment_time_scale : segment_time_scale_nosimd)
			(data + i * 1024, ret + i * newlen, rate, true);
	}

	return { ret,num * newlen };				//  返回输出数组的地址与长度
}

std::pair<float *, int> generate(int len = 1000000)
{
    float *data = new float[len];
    for (int i = 0; i < len; i++)
        data[i] = 1;// rand();
    return { data,len };
}

double work(int argc, char *argv[], std::pair<float *, int> input, int numThreads = 1, bool use_simd = true)
{
    int myid, numprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();

	int times = 1;
	auto bak = input;
	while (times--)
	{
		input = bak;
		std::pair<float *, int> output, data;
		int num_slice, num_slice_per_process;

		if (myid == 0)
		{
			//    printf("number of processes: %d\n", numprocs);
			input = stretch(input.first, input.second, input.second * 0.8);
			num_slice = input.second / 1024;
			num_slice_per_process = num_slice / numprocs;

			data.second = num_slice_per_process * 1024;
			data.first = new float[data.second];
			memcpy(data.first, input.first, data.second * sizeof(float));

			for (int i = 1, p = data.second / 1024; i < numprocs; i++)
			{
				int cnt = (i == numprocs - 1 ? num_slice - p : num_slice_per_process) * 1024;
				MPI_Send(&cnt, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
				MPI_Send(input.first + p * 1024, cnt, MPI_FLOAT, i, 2, MPI_COMM_WORLD);
				p += cnt / 1024;
			}
		}
		else
		{
			MPI_Recv(&data.second, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			data.first = new float[data.second];
			MPI_Recv(data.first, data.second, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		//printf("Hello world from process %d \n", myid);

	//	for (int i = 0; i < data.second; i++) assert(data.first[i] == 1);
		//fprintf(stderr, "%d compute start: %d\n", myid, data.second);
		auto temp = data;
		//data = time_scale(data.first, data.second, 1 / 0.8, true);
		data = omptime_scale(data.first, data.second, 1 / 0.8, numThreads, use_simd);
		delete[] temp.first;
		float rate = 1 / 0.8;
		int new_slice_len = 1024 * rate;
		//fprintf(stderr, "%d compute ok\n", myid);

		if (myid == 0)
		{
			output.second = num_slice * new_slice_len;
			output.first = new float[output.second];
			memcpy(output.first, data.first, data.second * sizeof(float));
			for (int i = 1, p = data.second / new_slice_len; i < numprocs; i++)
			{
				int cnt = (i == numprocs - 1 ? num_slice - p : num_slice_per_process) * new_slice_len;
				MPI_Recv(output.first + p * new_slice_len, cnt, MPI_FLOAT, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				p += cnt / new_slice_len;
				//		printf("%d recved\n", i);
			}
			delete[] input.first;
		}
		else
		{
			//	printf("%d send start\n", myid);
			MPI_Send(data.first, data.second, MPI_FLOAT, 0, 3, MPI_COMM_WORLD);
			//	printf("%d send ok\n", myid);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	if (myid == 0)
	{
	//	printf("done!\n");
		printf("%d %d %d %d: ", bak.second, numprocs, numThreads, use_simd);
		printf("time: %d\n", (int)((MPI_Wtime() - start) * 1000 / 1));
	}

    MPI_Finalize();

    return 0;
}

void get_time(int argc, char *argv[], int N, int num_threads, bool simd)
{
	auto input = generate(N);
	work(argc, argv, input, num_threads, simd);
}

int main(int argc, char *argv[])
{
	int N, ntrd, simd;
	sscanf(argv[1], "%d", &N);
	sscanf(argv[2], "%d", &ntrd);
	sscanf(argv[3], "%d", &simd);
	get_time(argc, argv, N, ntrd, simd);

	return 0;
}