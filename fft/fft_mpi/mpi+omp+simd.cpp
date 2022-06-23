#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <queue>
#include <math.h>
#include<sys/time.h>
#include <unistd.h>
#include <omp.h>
#include <mpi.h>
#include "stdint.h"
#include <cstring>
#include <arm_neon.h>
#define nullptr NULL
struct timeval tv_begin, tv_end;
long long sb,se;

using namespace std;
#define PI acos(-1)
struct Complex{
    float x,y;
    Complex(float _x=0,float _y=0):x(_x),y(_y){}
};
Complex operator + (Complex a,Complex b){ return Complex(a.x+b.x , a.y+b.y);}
Complex operator - (Complex a,Complex b){ return Complex(a.x-b.x , a.y-b.y);}
Complex operator * (Complex a,Complex b){ return Complex(a.x*b.x-a.y*b.y , a.x*b.y+a.y*b.x);}
float32x4x2_t f_add(float32x4x2_t a,float32x4x2_t b){
    float32x4_t a_re=a.val[0],a_im=a.val[1];
    float32x4_t b_re=b.val[0],b_im=b.val[1];
    a_re= vaddq_f32(a_re,b_re),a_im= vaddq_f32(a_im,b_im);
    float32x4x2_t std;
    std.val[0]=a_re;std.val[1]=a_im;
    return std;
}
float32x4x2_t f_sub(float32x4x2_t a,float32x4x2_t b){
    float32x4_t a_re=a.val[0],a_im=a.val[1];
    float32x4_t b_re=b.val[0],b_im=b.val[1];
    a_re= vsubq_f32(a_re,b_re),a_im= vsubq_f32(a_im,b_im);
    a.val[0]=a_re,a.val[1]=a_im;
    return a;
}
float32x4x2_t f_mul(float32x4x2_t a,float32x4x2_t b){
    float32x4_t a_re=a.val[0],a_im=a.val[1];
    float32x4_t b_re=b.val[0],b_im=b.val[1];
    float32x4x2_t st;
    st.val[0]= vsubq_f32(vmulq_f32(a_re,b_re), vmulq_f32(a_im,b_im));
    st.val[1]= vaddq_f32(vmulq_f32(a_re,b_im), vmulq_f32(a_im,b_re));
    return st;
}
float32x4x2_t f_div(float32x4x2_t a,float len){
    float32x4_t a_re=a.val[0],a_im=a.val[1];
    float32x4_t Len=vmovq_n_f32((float32_t)((1.0)/len));
    a_re= vmulq_f32(a_re,Len),a_im= vmulq_f32(a_im,Len);
    a.val[0]=a_re,a.val[1]=a_im;
    return a;
}
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
Complex* fft_mpi(Complex *y, int len, int on, int H=2) {
    for (int h = H; h <= len; h <<= 1) {
        Complex wn(cos(2 * PI / h), sin(on * 2 * PI / h));
#pragma omp parallel for num_threads(3)
        for (int j = 0; j < len; j += h) {
            Complex w(1, 0);
            if(h/2>=4){
                Complex b[4]={wn*wn*wn*wn,wn*wn*wn*wn,wn*wn*wn*wn,wn*wn*wn*wn},
                        c[4]={w, w*wn, w*wn*wn, w*wn*wn*wn};
                float32x4x2_t temp= vld2q_f32((const float32_t*)b),w1= vld2q_f32((const float32_t*)c);
                for (int k = j; k < j + h / 2; k+=4){
                    float32x4x2_t u= vld2q_f32((const float32_t*)(y+k)),t= vld2q_f32((const float32_t*)(y+k+h/2));
                    t= f_mul(t,w1);
                    vst2q_f32((float32_t*)(y+k), f_add(u,t));
                    vst2q_f32((float32_t*)(y+k+h/2), f_sub(u,t));
                    w1= f_mul(w1,temp);
                }
            }
            else{
                for (int k = j; k < j + h / 2; k++) {
                    Complex u = y[k];
                    Complex t = w*y[k + h / 2];
                    y[k] = u+t;
                    y[k + h / 2] = u-t;
                    w = w*wn;
                }
            }
        }

    }

    if (on == -1&&H>2) {
         if(len>=4){
            #pragma omp for
            for(int i=0;i<len;i+=4){
                float32x4x2_t t=vld2q_f32((const float32_t*)(y+i));
                t= f_div(t,len);
                vst2q_f32((float32_t*)(y+i),t);
            }
        }
        else{
            #pragma omp for
            for (int i = 0; i < len; i++) {
                y[i].x /= len; y[i].y /= len;
            }
        }
    }
    return y;
}


const int N=20;
const int Core=1<<3;
const int node=8;
Complex y[(1<<N)+10],yy[(1<<N)+10];
Complex ty[(1<<N)+10];
void Init(){
    for (int i = 0; i < (1<<N); i++) {
        y[i].x = yy[i].x = rand();
        y[i].y = yy[i].y = rand();
    }
}
void check(){
    for (int i = 0; i < (1<<N); i++) {
        if(y[i].x!=yy[i].x||y[i].y!=yy[i].y){
            printf("wrong\n");
            return;
        }
    }
    printf("correct\n");
}
void Print(){
    for (int i = 0; i < (1<<3); i++) {
        printf("%f %f\n",y[i].x,y[i].y);
        printf("%f %f\n",yy[i].x,yy[i].y);
        printf("&&&&&&&&&&&&&&&&\n");
    }
}
void mympi(){
    int rank=0,dest=0,i=0,provided;
    int block=(1<<N)/Core;
    MPI_Status status;
    MPI_Request request;
    MPI_Init_thread(NULL,NULL, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED)
        MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0) {
        gettimeofday(&tv_begin,NULL);
        fft(yy,1<<N,0);
        gettimeofday(&tv_end,NULL);
        sb=tv_begin.tv_sec*(1e6)+tv_begin.tv_usec,se=tv_end.tv_sec*(1e6)+tv_end.tv_usec;
        printf("normal: %lld\n",(se-sb));

        gettimeofday(&tv_begin,NULL);
        change(y, (1<<N));
        for(i=0;dest<node;i+=block,dest++){
            MPI_Isend(y+i, block*2, MPI_FLOAT, dest+1, i/block, MPI_COMM_WORLD,&request);
        }
        dest%=node;
        while (dest<node){
            MPI_Recv(ty, block*2, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            memcpy(y+(status.MPI_TAG)*block, ty, sizeof(float) * block*2);
            if(i<(1<<N))MPI_Isend(y+i, block*2, MPI_FLOAT, status.MPI_SOURCE, i/block, MPI_COMM_WORLD,&request),i+=block;
            else MPI_Isend(y, block*2, MPI_FLOAT, status.MPI_SOURCE, (1<<N)/block+1, MPI_COMM_WORLD,&request),dest++;
        }
        fft_mpi(y,(1<<N),0,block*2);
        gettimeofday(&tv_end,NULL);
        sb=tv_begin.tv_sec*(1e6)+tv_begin.tv_usec,se=tv_end.tv_sec*(1e6)+tv_end.tv_usec;
        printf("mpi: %lld\n",(se-sb));


        check();
    }
    else{
        while (true){
            MPI_Recv(ty, block*2, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG >= (1<<N)/block)break;
            fft_mpi(ty,block,0);
            MPI_Isend(ty, block*2, MPI_FLOAT, 0, status.MPI_TAG, MPI_COMM_WORLD,&request);
        }
    }
    MPI_Finalize();
}
void serialize(){
    change(y, N);
    int block=N/Core;
    for(int i=0;i<N;i+=block){
        memcpy(ty, y+i, sizeof(float) * block*2);
        fft_mpi(ty,block,0);
        memcpy(y+i, ty, sizeof(float) * block*2);
    }
    fft_mpi(y,N,0,block*2);
    fft(yy,N,0);
    check();
}
int main(){
    Init();
    mympi();
    return 0;
}
