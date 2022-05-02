#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <queue>
#include <math.h>
#include <semaphore.h>
#include<sys/time.h>
#include <unistd.h>
#include <arm_neon.h>

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


sem_t	sem_children[7],sem_p[7],sem_q[7];
queue<Node *>pool[7];
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
            delete tt;
        }
    }
}
void *idft(void *a){
    Node *tt=(Node*)a;
    Complex *y=tt->cp;
    int l=tt->temp[0],r=tt->temp[1],len=tt->temp[2];
    if (len >= 4) {
        for (int i = 0; i < len; i += 4) {
            float32x4x2_t t = vld2q_f32((const float32_t *) (y + i));
            t = f_div(t, len);
            vst2q_f32((float32_t * )(y + i), t);
        }
    } else {
        for (int i = 0; i < len; i++) {
            y[i].x /= len;
            y[i].y /= len;
        }
    }
    delete tt;
    return nullptr;
}
Complex* fft_p(Complex *y, int len, int on) {
    for(int i=0;i<7;i++)
        sem_init(&sem_children[i], 0, 0),
        sem_init(&sem_p[i], 0, 0),
        sem_init(&sem_q[i], 0, 0);
    pthread_t threads[7];
    for(int i=0;i<7;i++) {int *a=new int(i);pthread_create(&threads[i], nullptr, fft_pthread, a);}
    for (auto& th : threads) {pthread_detach(th);}
    change(y, len);
    for (int h = 2; h <= len; h <<= 1) {
        for (int j = 0; j < len; j += h) {
            Node *temp=new Node(y,j,h,on);
            pool[(j/h)%7].push(temp);
        }
        for(int i=0;i<7;i++){
            Node *temp=new Node(nullptr,0,0,0);
            pool[i].push(temp);
            sem_post(&sem_children[i]);
        }
        for(int i=0;i<7;i++){ sem_wait(&sem_p[i]);sem_post(&sem_q[i]);}
    }
    for(int i=0;i<7;i++) {
        pool[i].push(nullptr);
        sem_post(&sem_children[i]);
    }
    if (on == -1) {
        int s=0,block=len/7+1;
        pthread_t threads[7];
        for (auto& th : threads) {
            Node *p=new Node(y,s,min(len,s+block),len);
            s+=block;
            pthread_create(&th, nullptr, idft, p);
        }
        for (auto& th : threads) {pthread_detach(th);}
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
    printf("pthread: %lld\n",se-sb);
    return 0;
}
