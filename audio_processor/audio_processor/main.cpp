#include "utilities.h"
#include "SIMD.h"

void test()
{
	float *data;
	int len;
	auto t = read((char *)"Record2.wav");
	data = t.first;
	len = t.second;

	t = stretch(data, len, len * 0.8);
	data = t.first;
	len = t.second;
	t = time_scale(data, len, 1 / 0.8, false);
	data = t.first;
	len = t.second;
//	t = filter(data, len);
//	data = t.first;
//	len = t.second;

	write((char *)"Record2.wav",
		(char *)"Record2_test_dft.wav", data, len);
}

int main()
{
	simd::test();


	return 0;
}