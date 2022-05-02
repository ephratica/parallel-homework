//
// Created by HP on 2022/4/1.
//

#ifndef REMAKE2_0_FFT_H
#define REMAKE2_0_FFT_H
#include <math.h>
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
Complex *linear(Complex *y, int len, int lenth) {
	int block = len / lenth, t = 0;
	Complex *yy = new Complex[lenth];
	for (int i = 0; t < lenth; i += block) {
		yy[t] = y[i] + y[i + 1];
		yy[t].x /= 2;
		yy[t++].y /= 2;
	}
	return yy;
}

#endif //REMAKE2_0_FFT_H
