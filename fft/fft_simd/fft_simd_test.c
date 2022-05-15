#include<stdio.h>
#include <time.h>
#include <stdlib.h>
#include <arm_neon.h>
#include<sys/time.h>
#define ll long long


struct timeval tv_begin, tv_end;
struct Complex {
    float x, y;
};

#include <math.h>
#define Complex struct Complex
#define PI 3.1415926

Complex make_Complex(float a,float b){
    Complex t;
    t.x=a,t.y=b;
    return t;
}

Complex add(Complex a,Complex b){return make_Complex(a.x + b.x, a.y + b.y);}
Complex sub(Complex a,Complex b){return make_Complex(a.x - b.x, a.y - b.y);}
Complex mul(Complex a, Complex b) { return make_Complex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

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
void swap(Complex *a, Complex *b) {
    Complex temp;
    temp = *a;
    *a = *b;
    *b = temp;
}
int rev[1<<20];
void change(Complex* y, int len) {
    rev[0] = 0;
    for (int i = 0; i < len; ++i) {
        rev[i] = rev[i >> 1] >> 1;
        if (i & 1) {
            rev[i] |= len >> 1;
        }
    }
    for (int i = 0; i < len; ++i) {
        if (i < rev[i]) {
            swap(&y[i], &y[rev[i]]);
        }
    }
}
Complex* fft_simd(Complex* y, int len, int on) {
    change(y, len);
    for (int h = 2; h <= len; h <<= 1) {
        Complex wn= make_Complex(cos(2.0 * PI / h),sin(on * 2.0 * PI / h));
        for (int j = 0; j < len; j += h) {
            Complex w= make_Complex(1,0);
            if(h/2>=4){
                Complex b[4]={mul(wn,mul(wn, mul(wn, wn))),mul(wn,mul(wn, mul(wn, wn))),mul(wn,mul(wn, mul(wn, wn))),mul(wn,mul(wn, mul(wn, wn)))},
                        c[4]={w, mul(w,wn), mul(w,mul(wn,wn)), mul(w,mul(wn, mul(wn, wn)))};
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
                    Complex t = mul(w,y[k + h / 2]);
                    y[k] = add(u,t);
                    y[k + h / 2] = sub(u,t);
                    w = mul(w,wn);
                }
            }
        }
    }
    if (on == -1) {
        if(len>=4){
            for(int i=0;i<len;i+=4){
                float32x4x2_t t=vld2q_f32((const float32_t*)(y+i));
                t= f_div(t,len);
                vst2q_f32((float32_t*)(y+i),t);
            }
        }
        else{
            for (int i = 0; i < len; i++) {
                y[i].x /= len; y[i].y /= len;
            }
        }

    }
    return y;
}
Complex* fft(Complex* y, int len, int on) {
    change(y, len);
    for (int h = 2; h <= len; h <<= 1) {
        Complex wn= make_Complex(cos(2.0 * PI / h),sin(on * 2.0 * PI / h));
        for (int j = 0; j < len; j += h) {
            Complex w= make_Complex(1,0);
            for (int k = j; k < j + h / 2; k++) {
                Complex u = y[k];
                Complex t = mul(w,y[k + h / 2]);
                y[k] = add(u,t);
                y[k + h / 2] = sub(u,t);
                w = mul(w,wn);
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

Complex y[(1<<20)+10],yy[(1<<20)+10];
int main() {
    for (int i = 0; i < (1<<20); i++) {
        y[i].x = yy[i].x = rand();
        y[i].y = yy[i].y = rand();
    }
    gettimeofday(&tv_begin,NULL);
    fft(yy,1<<20,1);
    gettimeofday(&tv_end,NULL);
    long long sb=tv_begin.tv_sec*(1e6)+tv_begin.tv_usec,se=tv_end.tv_sec*(1e6)+tv_end.tv_usec;
    printf("normal: %lld\n",se-sb);
    gettimeofday(&tv_begin,NULL);
    fft_simd(y,1<<20,1);
    gettimeofday(&tv_end,NULL);
    sb=tv_begin.tv_sec*(1e6)+tv_begin.tv_usec,se=tv_end.tv_sec*(1e6)+tv_end.tv_usec;
    printf("simd: %lld\n",se-sb);

    return 0;
}