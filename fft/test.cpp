#include <iostream>
#include<stdio.h>
#include<sys/time.h>
#include<unistd.h>
#include <time.h>
#include <stdlib.h>
#include <typeinfo>
#include "FFT.h"
using namespace std;
const int n=(1<<17)+3;
struct timeval tv_begin,tv_end;
Complex y[10005];


int main(){
    for(int i=0;i<8;i++){
        y[i].x=rand();
        y[i].y=rand();
        cout<<y[i].x<<" "<<y[i].y<<endl;
    }
    cout<<"************"<<endl;
    fft(y,8,1);
    for(int i=0;i<8;i++)cout<<y[i].x<<" "<<y[i].y<<endl;
    fft(y,8,-1);
    cout<<"************"<<endl;
    for(int i=0;i<8;i++)cout<<y[i].x<<" "<<y[i].y<<endl;
    Complex *z= linear(y,8,6);
    cout<<"************"<<endl;
    for(int i=0;i<6;i++)cout<<z[i].x<<" "<<z[i].y<<endl;
    return 0;
}
//        long long sb=tv_begin.tv_sec*(1e6)+tv_begin.tv_usec,se=tv_end.tv_sec*(1e6)+tv_end.tv_usec;
//        cout<<"平凡算法："<<m<<" "<< typeid(a[0]).name()<<" "<<se-sb<<"um"<<endl;