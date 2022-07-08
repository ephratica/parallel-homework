#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include<sys/time.h>
#define PI acos(-1)

struct timeval tv_begin, tv_end;

void swap(cuDoubleComplex &a,cuDoubleComplex &b){
    cuDoubleComplex temp=a;
    a=b;
    b=temp;
    return;
}

void change(cuDoubleComplex *y, int len) {
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

void initWith(cuDoubleComplex* pow,int h,int on){
    pow[0]= make_cuDoubleComplex(1,0);
    cuDoubleComplex wn= make_cuDoubleComplex(cos(2 * PI / h),sin(on * 2 * PI / h));
    for(int i=1;i<h/2;i++){
        pow[i]= cuCmul(pow[i-1],wn);
    }
}


__global__
void fft_cuda(cuDoubleComplex* y,cuDoubleComplex* temp,int n,int h,cuDoubleComplex* pow)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;//开始位置的索引
    int stride = blockDim.x * gridDim.x;//跳过其他由线程处理的索引


    for(int i = index; i < n; i += stride){
        int j=(i/h)*h;
        if(i-j<h/2) {
            cuDoubleComplex w= pow[i-j];
            //if(h==4)printf("fft_cuda_before\nw:%lf %lf,y[%d]:%lf %lf,y[%d]:%lf %lf\n",w.x,w.y,i,y[i].x,y[i].y,i+h/2,y[i+h/2].x,y[i+h/2].y);
            cuDoubleComplex u=temp[i];
            cuDoubleComplex t= cuCmul(temp[i+h/2],w);
            y[i]= cuCadd(u,t);
            y[i+h/2]= cuCsub(u,t);
            //if(h==4)printf("fft_cuda_after\nw:%lf %lf,y[%d]:%lf %lf,y[%d]:%lf %lf\n",w.x,w.y,i,y[i].x,y[i].y,i+h/2,y[i+h/2].x,y[i+h/2].y);
        }
    }
}

void check(cuDoubleComplex* y,cuDoubleComplex* yy,int n){
    for (int i = 0; i < n; i++) {
        if(abs(y[i].x-yy[i].x)>(0.00001)||abs(y[i].y-yy[i].y)>(0.00001)){
            printf("wrong\n");
            printf("%d\n",i);
            printf("%lf %lf\n",y[i].x,y[i].y);
            printf("%lf %lf\n",yy[i].x,yy[i].y);
            printf("%lf,%lf",abs(y[i].x-yy[i].x),abs(y[i].y-yy[i].y));
            return;
        }
    }
    printf("correct\n");
}

void Print(cuDoubleComplex* y,cuDoubleComplex* yy,int n){
    for (int i = 0; i < n; i++) {
        printf("%f %f\n",y[i].x,y[i].y);
        printf("%f %f\n",yy[i].x,yy[i].y);
        printf("&&&&&&&&&&&&&&&&\n");
    }
}

void fft(cuDoubleComplex *y, int len, int on) {
    change(y, len);
    for (int h = 2; h <= len; h <<= 1) {
        cuDoubleComplex wn=make_cuDoubleComplex(cos(2 * PI / h), sin(on * 2 * PI / h));
        for (int j = 0; j < len; j += h) {
            cuDoubleComplex w=make_cuDoubleComplex(1, 0);
            for (int k = j; k < j + h / 2; k++) {
                cuDoubleComplex u = y[k];
                cuDoubleComplex t = cuCmul(w,y[k+h/2]);
                //if(h==4)printf("fft_before\nw:%lf %lf,y[%d]:%lf %lf,y[%d]:%lf %lf\n",w.x,w.y,k,y[k].x,y[k].y,k+h/2,y[k+h/2].x,y[k+h/2].y);
                y[k] = cuCadd(u,t);
                y[k + h / 2] = cuCsub(u,t);
                w = cuCmul(w,wn);
                //if(h==4)printf("fft_after\nw:%lf %lf,y[%d]:%lf %lf,y[%d]:%lf %lf\n",w.x,w.y,k,y[k].x,y[k].y,k+h/2,y[k+h/2].x,y[k+h/2].y);
            }
        }
    }
    if (on == -1) {
        for (int i = 0; i < len; i++) {
            y[i].x /= len;y[i].y/=len;
        }
    }
}

int main()
{
    const int N = 1<<25;
    size_t size = N * sizeof(cuDoubleComplex);

    cuDoubleComplex* y;
    cuDoubleComplex* yy;
    cuDoubleComplex* temp;
    cuDoubleComplex* pow;

    //cudaMallocManaged(&a, size)的作用相当于a = (float *)malloc(size);
    //区别在于申请的是gpu可以访问的内存
    checkCuda( cudaMallocManaged(&y, size) );
    checkCuda( cudaMallocManaged(&yy, size) );
    checkCuda( cudaMallocManaged(&temp, size) );
    checkCuda( cudaMallocManaged(&pow, size) );

    //选定网络上的block大小
    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    size_t threadsPerBlock;
    size_t numberOfBlocks;

    threadsPerBlock = 256;
    numberOfBlocks = 32 * numberOfSMs;

    for (int i = 0; i < N; i++) {
        y[i].x = yy[i].x = rand();
        y[i].y = yy[i].y = rand();
    }

    gettimeofday(&tv_begin,NULL);
    fft(yy,N,1);
    gettimeofday(&tv_end,NULL);
    long long sb=tv_begin.tv_sec*(1e6)+tv_begin.tv_usec,se=tv_end.tv_sec*(1e6)+tv_end.tv_usec;
    printf("normal: %lld\n",se-sb);

    gettimeofday(&tv_begin,NULL);
    change(y,N);
    for (int h=2;h<=N;h<<=1){
        initWith(pow,h,1);
        memcpy(temp,y, size);
        //类似串行程序的函数调用，我们把这种“函数调用”称为“a kernel is launched”
        //<<<numberOfBlocks, threadsPerBlock>>>顾名思义，就是块数和线程数
        fft_cuda<<<numberOfBlocks, threadsPerBlock>>>(y, temp,N, h, pow);

        //检查错误
        //checkCuda( cudaGetLastError() );
        //cpu程序不会继续执行直到gpu的kernel全部完成
        //checkCuda( cudaDeviceSynchronize() );
        cudaDeviceSynchronize();
    }
    gettimeofday(&tv_end,NULL);
    sb=tv_begin.tv_sec*(1e6)+tv_begin.tv_usec,se=tv_end.tv_sec*(1e6)+tv_end.tv_usec;
    printf("cuda: %lld\n",se-sb);



    //Print(y,yy,8);
    check(y,yy,N);

    //释放内存
    checkCuda( cudaFree(y) );
    checkCuda( cudaFree(yy) );
    checkCuda( cudaFree(temp) );
    checkCuda( cudaFree(pow) );
}
