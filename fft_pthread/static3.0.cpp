#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <queue>
#include <math.h>
#include <semaphore.h>
#include<sys/time.h>
#include <unistd.h>

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
struct Node{
    Complex *cp;
    int temp[3];
    Node(Complex *_cp=NULL,int temp0=0,int temp1=0,int temp2=0) { cp=_cp,temp[0]=temp0,temp[1]=temp1,temp[2]=temp2;}
};
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


sem_t	sem_children[7],sem_p[7],sem_q[7];
//pthread_t pth;
queue<Node *>pool[7];
//pthread_mutex_t mutex_for_todo = PTHREAD_MUTEX_INITIALIZER;
void *fft_pthread(void *a){
    int id=*((int *)a);
    while (true){
        sem_wait(&sem_children[id]);
        while (!pool[id].empty()) {
            Node* tt = pool[id].front();
            pool[id].pop();
            if(tt== nullptr){
                return nullptr;
            }
            if(tt->cp== nullptr){ sem_post(&sem_p[id]);sem_wait(&sem_q[id]);break;}
            int j=tt->temp[0],h=tt->temp[1],on=tt->temp[2];
            Complex *y=tt->cp;
            Complex w(1, 0);
            Complex wn(cos(2 * PI / h), sin(on * 2 * PI / h));
            for (int k = j; k < j + h / 2; k++) {
                Complex u = y[k];
                Complex t = w * y[k + h / 2];
                y[k] = u + t;
                y[k + h / 2] = u - t;
                w = w * wn;
            }
            delete tt;
        }
    }
}
void *idft(void *a){
    Node *tt=(Node*)a;
    Complex *y=tt->cp;
    int l=tt->temp[0],r=tt->temp[1],len=tt->temp[2];
    for (int i = l; i < r; i++) {
        y[i].x /= len;y[i].y/=len;
    }
    delete tt;
    return nullptr;
}
Complex* fft_p(Complex *y, int len, int on) {
    //pthread_mutex_init(&mutex_for_todo, NULL);
    for(int i=0;i<7;i++)
        sem_init(&sem_children[i], 0, 0),
                sem_init(&sem_p[i], 0, 0),
                sem_init(&sem_q[i], 0, 0);
    pthread_t threads[7];
    for(int i=0;i<7;i++) {int *a=new int(i);pthread_create(&threads[i], nullptr, fft_pthread, a);}
    for(int i=0;i<7;i++) {pthread_detach(threads[i]);}
    change(y, len);
    for (int h = 2; h <= len; h <<= 1) {
        if(h<=4){
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
            continue;
        }
        for (int j = 0; j < len; j += h) {
            Node *temp=new Node(y,j,h,on);
            pool[(j/h)%7].push(temp);
            //sem_post(&sem_children[(j/h)%7]);
        }
        for(int i=0;i<7;i++){
            Node *temp=new Node(nullptr,0,0,0);
            pool[i].push(temp);
            sem_post(&sem_children[i]);
        }
        //pthread_mutex_unlock(&mutex_for_todo);
        for(int i=0;i<7;i++){ sem_wait(&sem_p[i]);sem_post(&sem_q[i]);}
    }
    for(int i=0;i<7;i++) {
        pool[i].push(nullptr);
        sem_post(&sem_children[i]);
    }
    if (on == -1) {
        int s=0,block=len/7+1;
        pthread_t threads[7];
        for (int i=0;i<7;i++) {
            Node *p=new Node(y,s,min(len,s+block),len);
            s+=block;
            pthread_create(&threads[i], nullptr, idft, p);
        }
        for (int i=0;i<7;i++) {pthread_detach(threads[i]);}
    }
    return y;
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
Complex y[1<<25],yy[1<<25];
int main(){

    for (int i = 0; i < (1<<25); i++) {
        y[i].x = yy[i].x = rand();
        y[i].y = yy[i].y = rand();
    }
    for (int i = 0; i < 10; ++i)
        fft(yy,1<<20,1);
    gettimeofday(&tv_begin,NULL);
    fft(yy,1<<20,1);
    gettimeofday(&tv_end,NULL);
    long long sb=tv_begin.tv_sec*(1e6)+tv_begin.tv_usec,se=tv_end.tv_sec*(1e6)+tv_end.tv_usec;
    printf("normal: %lld\n",se-sb);
    gettimeofday(&tv_begin,NULL);
    fft_p(y,1<<20,1);
    gettimeofday(&tv_end,NULL);
    sb=tv_begin.tv_sec*(1e6)+tv_begin.tv_usec,se=tv_end.tv_sec*(1e6)+tv_end.tv_usec;
    printf("pyhread: %lld\n",se-sb);
    return 0;
}
