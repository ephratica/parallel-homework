#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <queue>
#include <math.h>
#include <semaphore.h>

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

//pthread part
sem_t	sem_children;
pthread_t pth;
void *fft_pthread(void *a){
    Node *t=(Node*)a;
    int j=t->temp[0],h=t->temp[1],on=t->temp[2];
    Complex *y=t->cp;
    Complex w(1, 0);
    Complex wn(cos(2 * PI / h), sin(on * 2 * PI / h));
    for (int k = j; k < j + h / 2; k++) {
        Complex u = y[k];
        Complex t = w * y[k + h / 2];
        y[k] = u + t;
        y[k + h / 2] = u - t;
        w = w * wn;
    }
    delete t;
    sem_post(&sem_children);
}
Complex* fft_p(Complex *y, int len, int on) {
    change(y, len);
    for (int h = 2; h <= len; h <<= 1) {
        sem_init(&sem_children, 0, 7);
        for (int j = 0; j < len; j += h) {
            Node *temp=new Node(y,j,h,on);
            sem_wait(&sem_children);
            pthread_create(&pth, nullptr, fft_pthread, (void *)temp);
            pthread_detach(pth);
        }
        for(int i=0;i<7;i++)sem_wait(&sem_children);
    }
    if (on == -1) {
        for (int i = 0; i < len; i++) {
            y[i].x /= len;y[i].y/=len;
        }
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
int main(){
    Complex y[1<<8],yy[1<<8];
    for(int i=0;i<(1<<8);i++){
        yy[i].x=y[i].x=rand();
        yy[i].y=y[i].y=rand();
    }
    fft_p(y,1<<8,1);
    fft(yy,1<<8,1);
    for(int i=0;i<(1<<8);i++){
        printf("%f %f\n",y[i].x,y[i].y);
        printf("%f %f\n",yy[i].x,yy[i].y);
        printf("**************\n");
    }
    return 0;
}

//std::queue<int> todo;
//int working_threads = 7;
//pthread_mutex_t mutex_for_todo = PTHREAD_MUTEX_INITIALIZER;
//pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
//int ans[100];
//
//void* calculate_sum(void*) {
//    while (true) {
//        pthread_mutex_lock(&mutex_for_todo);
//        if (todo.empty()) {
//            working_threads--;
//            pthread_cond_signal(&cond);
//            pthread_mutex_unlock(&mutex_for_todo);
//            return nullptr;
//        }
//        int seed = todo.front();
//        todo.pop();
//        pthread_mutex_unlock(&mutex_for_todo);
//        srand(seed);
//        for (int i = 0; i < 100; ++i) {
//            ans[seed] += rand();
//        }
//    }
//}
//
//int main() {
//    pthread_t threads[7];
//    for (auto& th : threads) {
//        pthread_create(&th, nullptr, calculate_sum, nullptr);
//    }
//    for (int i = 0; i < 100; ++i)
//        todo.push(i);
//    for (auto& th : threads) {
//        pthread_detach(th);
//    }
//    while (true) {
//        pthread_cond_wait(&cond, nullptr);
//    }
//    for (int i = 0; i < 100; ++i)
//        std::cout << ans[i] << std::endl;
//    return 0;
//}