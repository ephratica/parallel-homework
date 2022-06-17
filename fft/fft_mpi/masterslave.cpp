#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <queue>
#include <math.h>
#include<sys/time.h>
#include <unistd.h>
//#include <omp.h>
#include <mpi.h>
#include <cstring>
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
//#pragma omp parallel for num_threads(7)
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
//#pragma omp for
    {
        if (on == -1) {
            for (int i = 0; i < len; i++) {
                y[i].x /= len;y[i].y/=len;
            }
        }
    }
    return y;
}
Complex* fft_mpi(Complex *y, int len, int on, int H=2) {
    for (int h = H; h <= len; h <<= 1) {
        Complex wn(cos(2 * PI / h), sin(on * 2 * PI / h));
//#pragma omp parallel for num_threads(7)
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
//#pragma omp for
    {
        if (on == -1&&H>2) {
            for (int i = 0; i < len; i++) {
                y[i].x /= len;y[i].y/=len;
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
    int rank=0,dest=0,i=0;
    int block=(1<<N)/Core;
    MPI_Status status;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0) {
        //master
        change(y, (1<<N));
        for(i=0;dest<node;i+=block,dest++){
            MPI_Send(y+i, block*2, MPI_FLOAT, dest+1, i/block, MPI_COMM_WORLD);
        }
        //printf("arrange finished\n");
        dest%=node;
        while (dest<node){
            MPI_Recv(ty, block*2, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            memcpy(y+(status.MPI_TAG)*block, ty, sizeof(float) * block*2);
            if(i<(1<<N))MPI_Send(y+i, block*2, MPI_FLOAT, status.MPI_SOURCE, i/block, MPI_COMM_WORLD),i+=block;
            else MPI_Send(y, block*2, MPI_FLOAT, status.MPI_SOURCE, (1<<N)/block+1, MPI_COMM_WORLD),dest++;
        }
        fft_mpi(y,(1<<N),0,block*2);

        //检查正确性
        check();
    }
    else{
        //slave
        while (true){
            MPI_Recv(ty, block*2, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG >= (1<<N)/block)break;
            fft_mpi(ty,block,0);
            MPI_Send(ty, block*2, MPI_FLOAT, 0, status.MPI_TAG, MPI_COMM_WORLD);
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