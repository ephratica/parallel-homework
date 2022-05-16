#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <queue>
#include <math.h>
#include<sys/time.h>
#include <unistd.h>
#include <omp.h>

#define nullptr NULL
struct timeval tv_begin, tv_end;

using namespace std;
#define PI acos(-1)
struct Complex{
    float x,y;
    Complex(float _x=0,float _y=0):x(_x),y(_y){}
};
Complex operator + (Complex a,Complex b){ return Complex(a.x+b.x , a.y+b.y);}
Complex operator - (Complex a,Complex b){ return Complex(a.x-b.x , a.y-b.y);}
Complex operator * (Complex a,Complex b){ return Complex(a.x*b.x-a.y*b.y , a.x*b.y+a.y*b.x);}
void swap(Complex &a,Complex &b){
    Complex temp=a;
    a=b;
    b=temp;
}
void change(Complex *y, int len) {
    int *rev=new int[len];
    rev[0]=0;
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
    delete[] rev;
}
Complex* fft(Complex *y, int len, int on) {
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
            y[i].x /= len;y[i].y/=len;
        }
    }
    return y;
}
Complex* fft_omp(Complex *y, int len, int on) {
    change(y, len);
    for (int h = 2; h <= len; h <<= 1) {
        Complex wn(cos(2 * PI / h), sin(on * 2 * PI / h));
#pragma omp parallel for num_threads(7)
        {
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

    }
#pragma omp parallel for num_threads(7)
    {
        if (on == -1) {
            for (int i = 0; i < len; i++) {
                y[i].x /= len;y[i].y/=len;
            }
        }
    }
    return y;
}
Complex* fft1(Complex *y, int len, int on) {
    change(y, len);
    for (int h = 2; h <= len; h <<= 1) {
        Complex wn(cos(2 * PI / h), sin(on * 2 * PI / h));
#pragma omp parallel for simd num_threads(7)
        {
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

    }
#pragma omp parallel for simd num_threads(7)
    {
        if (on == -1) {
            for (int i = 0; i < len; i++) {
                y[i].x /= len;y[i].y/=len;
            }
        }
    }

    return y;
}
Complex y[1<<28],yy[1<<28];
int main(){

    for (int i = 0; i < (1<<28); i++) {
        y[i].x = yy[i].x = rand();
        y[i].y = yy[i].y = rand();
    }
    gettimeofday(&tv_begin,NULL);
    fft(yy,1<<25,1);
    gettimeofday(&tv_end,NULL);
    long long sb=tv_begin.tv_sec*(1e6)+tv_begin.tv_usec,se=tv_end.tv_sec*(1e6)+tv_end.tv_usec;
    printf("normal: %lld\n",se-sb);
    gettimeofday(&tv_begin,NULL);
    fft_omp(y,1<<25,1);
    gettimeofday(&tv_end,NULL);
    sb=tv_begin.tv_sec*(1e6)+tv_begin.tv_usec,se=tv_end.tv_sec*(1e6)+tv_end.tv_usec;
    printf("omp: %lld\n",se-sb);

    gettimeofday(&tv_begin,NULL);
    fft1(y,1<<25,1);
    gettimeofday(&tv_end,NULL);
    sb=tv_begin.tv_sec*(1e6)+tv_begin.tv_usec,se=tv_end.tv_sec*(1e6)+tv_end.tv_usec;
    printf("omp with omp simd: %lld\n",se-sb);
    return 0;
}