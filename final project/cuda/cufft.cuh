#ifndef REMAKE2_0_FFT_H
#define REMAKE2_0_FFT_H

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cstring>

#define PI acos(-1)

struct Complex {
    double x, y;
    Complex(double _x = 0, double _y = 0) :x(_x), y(_y) {}

};
Complex operator + (Complex a, Complex b) { return Complex(a.x + b.x, a.y + b.y); }
Complex operator - (Complex a, Complex b) { return Complex(a.x - b.x, a.y - b.y); }
Complex operator * (Complex a, Complex b) { return Complex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

void swap(cuDoubleComplex &a, cuDoubleComplex &b) {
    cuDoubleComplex temp = a;
    a = b;
    b = temp;
    return;
}

void change(cuDoubleComplex *y, int len) {
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

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

void initWith(cuDoubleComplex *pow, int h, int on) {
    pow[0] = make_cuDoubleComplex(1, 0);
    cuDoubleComplex wn = make_cuDoubleComplex(cos(2 * PI / h), sin(on * 2 * PI / h));
    for (int i = 1; i < h / 2; i++) {
        pow[i] = cuCmul(pow[i - 1], wn);
    }
}


__global__
void fft_cuda(cuDoubleComplex *y, cuDoubleComplex *temp, int n, int h, cuDoubleComplex *pow)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;//开始位置的索引
    int stride = blockDim.x * gridDim.x;//跳过其他由线程处理的索引


    for (int i = index; i < n; i += stride) {
        int j = (i / h) * h;
        if (i - j < h / 2) {
            cuDoubleComplex w = pow[i - j];
            cuDoubleComplex u = temp[i];
            cuDoubleComplex t = cuCmul(temp[i + h / 2], w);
            y[i] = cuCadd(u, t);
            y[i + h / 2] = cuCsub(u, t);
        }
    }
}

__global__
void idft(cuDoubleComplex *y, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;//开始位置的索引
    int stride = blockDim.x * gridDim.x;//跳过其他由线程处理的索引

    for (int i = index; i < n; i += stride) {
        y[i] = cuCdiv(y[i], make_cuDoubleComplex(n, 0));
    }
}

void fft(Complex *x, int N, int block, int on) {
    size_t size = N * sizeof(cuDoubleComplex);

    cuDoubleComplex *y;
    cuDoubleComplex *temp;
    cuDoubleComplex *pow;

    //cudaMallocManaged(&a, size)的作用相当于a = (float *)malloc(size);
    //区别在于申请的是gpu可以访问的内存
    checkCuda(cudaMallocManaged(&y, size));
    checkCuda(cudaMallocManaged(&temp, size));
    checkCuda(cudaMallocManaged(&pow, size));

    for (int i = 0; i < N; i++) {
        y[i] = make_cuDoubleComplex(x[i].x, x[i].y);
    }

    //选定网络上的block大小
    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    size_t threadsPerBlock;
    size_t numberOfBlocks;

    threadsPerBlock = 256;
    numberOfBlocks = 32 * numberOfSMs;

    for(int i = 0; i < N; i += block)
        change(y + i, block);
    for (int h = 2; h <= block; h <<= 1) {
        initWith(pow, h, on);
        memcpy(temp, y, size);
        fft_cuda <<<numberOfBlocks, threadsPerBlock, 0 >>> (y, temp, N, h, pow);
        cudaDeviceSynchronize();
    };

    if (on == -1)idft <<<numberOfBlocks, threadsPerBlock, 0 >>> (y, N);
    cudaDeviceSynchronize();

    //释放内存
    checkCuda(cudaFree(y));
    checkCuda(cudaFree(temp));
    checkCuda(cudaFree(pow));
}

#endif
