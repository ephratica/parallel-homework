#include "utilities.h"

void test_stretch()
{
	float *data;
	int len;
	auto t = read((char *)"Record2.wav");
	data = t.first;
	len = t.second;

	t = stretch(data, len, 0.8f);
	data = t.first;
	len = t.second;

	write((char *)"Record2.wav",
		(char *)"Record2_s.wav", data, len);
}

void test_DFT()
{
	float *data;
	int len;
	auto t = read((char *)"Record1.wav");
	data = t.first;
	len = t.second;

	auto cdata = FtoC(data, len);
	DFT::dft(cdata, len);
	DFT::idft(cdata, len);
	delete[] data;
	data = CtoF(cdata, len);

	write((char *)"Record1.wav",
		(char *)"Record1_fft.wav", data, len);
}

void test()
{
	float *data;
	int len;
	auto t = read((char *)"Record2.wav");
	data = t.first;
	len = t.second;

	t = stretch(data, len, 0.8);
	data = t.first;
	len = t.second;
	t = time_scale(data, len, 1 / 0.8);
	data = t.first;
	len = t.second;

	write((char *)"Record2.wav",
		(char *)"Record2_test.wav", data, len);
}

int main()
{
	test();


	return 0;
}