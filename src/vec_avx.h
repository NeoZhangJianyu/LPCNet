/* Copyright (c) 2018 Mozilla
                 2012-2017 Jean-Marc Valin */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/*
  AVX implementation of vector operations, compile with -mavx
  AVX2/FMA implementation of vector operations, compile with -mavx2 -mfma
*/

#ifndef VEC_AVX_H
#define VEC_AVX_H

#include <immintrin.h>

/* Use 8-bit dot products unless disabled or if stuck with SSE2. */
#if (defined(__AVX2__) || defined(__SSSE3__)) && !defined(DISABLE_DOT_PROD)
#define DOT_PROD
#define USE_SU_BIAS

#else

#warning "Only SSE and SSE2 are available. On newer machines, enable SSSE3/AVX/AVX2 using -march= to get better performance"

#endif


#ifndef __SSE_4_1__
static inline __m128 mm_floor_ps(__m128 x) {
  __m128 half = _mm_set1_ps(0.5);
  return _mm_cvtepi32_ps(_mm_cvtps_epi32(_mm_sub_ps(x, half)));
}
#undef _mm_floor_ps
#define _mm_floor_ps(x) mm_floor_ps(x)
#endif


/* If we don't have AVX available, emulate what we need with SSE up to 4.1. */
#ifndef __AVX__

typedef struct {
  __m128 lo;
  __m128 hi;
} mm256_emu;
#define __m256 mm256_emu

static inline mm256_emu mm256_loadu_ps(const float *src) {
  mm256_emu ret;
  ret.lo = _mm_loadu_ps(&src[0]);
  ret.hi = _mm_loadu_ps(&src[4]);
  return ret;
}
#define _mm256_loadu_ps(src) mm256_loadu_ps(src)


static inline void mm256_storeu_ps(float *dst, mm256_emu src) {
  _mm_storeu_ps(dst, src.lo);
  _mm_storeu_ps(&dst[4], src.hi);
}
#define _mm256_storeu_ps(dst, src) mm256_storeu_ps(dst, src)


static inline mm256_emu mm256_setzero_ps() {
  mm256_emu ret;
  ret.lo = _mm_setzero_ps();
  ret.hi = ret.lo;
  return ret;
}
#define _mm256_setzero_ps mm256_setzero_ps

static inline mm256_emu mm256_broadcast_ss(const float *x) {
  mm256_emu ret;
  ret.lo = _mm_set1_ps(*x);
  ret.hi = ret.lo;
  return ret;
}
#define _mm256_broadcast_ss(x) mm256_broadcast_ss(x)

static inline mm256_emu mm256_set1_ps(float x) {
  mm256_emu ret;
  ret.lo = _mm_set1_ps(x);
  ret.hi = ret.lo;
  return ret;
}
#define _mm256_set1_ps(x) mm256_set1_ps(x)



static inline mm256_emu mm256_mul_ps(mm256_emu a, mm256_emu b) {
  mm256_emu ret;
  ret.lo = _mm_mul_ps(a.lo, b.lo);
  ret.hi = _mm_mul_ps(a.hi, b.hi);
  return ret;
}
#define _mm256_mul_ps(a,b) mm256_mul_ps(a,b)

static inline mm256_emu mm256_add_ps(mm256_emu a, mm256_emu b) {
  mm256_emu ret;
  ret.lo = _mm_add_ps(a.lo, b.lo);
  ret.hi = _mm_add_ps(a.hi, b.hi);
  return ret;
}
#define _mm256_add_ps(a,b) mm256_add_ps(a,b)


static inline mm256_emu mm256_max_ps(mm256_emu a, mm256_emu b) {
  mm256_emu ret;
  ret.lo = _mm_max_ps(a.lo, b.lo);
  ret.hi = _mm_max_ps(a.hi, b.hi);
  return ret;
}
#define _mm256_max_ps(a,b) mm256_max_ps(a,b)

static inline mm256_emu mm256_min_ps(mm256_emu a, mm256_emu b) {
  mm256_emu ret;
  ret.lo = _mm_min_ps(a.lo, b.lo);
  ret.hi = _mm_min_ps(a.hi, b.hi);
  return ret;
}
#define _mm256_min_ps(a,b) mm256_min_ps(a,b)

static inline mm256_emu mm256_rcp_ps(mm256_emu a) {
  mm256_emu ret;
  ret.lo = _mm_rcp_ps(a.lo);
  ret.hi = _mm_rcp_ps(a.hi);
  return ret;
}
#define _mm256_rcp_ps(a) mm256_rcp_ps(a)


static inline __m128 mm256_extractf128_ps(mm256_emu x, int i) {
    return (i==0) ? x.lo : x.hi;
}
#undef _mm256_extractf128_ps
#define _mm256_extractf128_ps(x,i) mm256_extractf128_ps(x,i)

static inline mm256_emu mm256_insertf128_ps(mm256_emu dst, __m128 src, int i) {
    if (i==0) dst.lo = src;
    else dst.hi = src;
    return dst;
}
#undef _mm256_insertf128_ps
#define _mm256_insertf128_ps(dst,src,i) mm256_insertf128_ps(dst,src,i)

#endif /* __AVX__ */



/* If we don't have AVX2 available, emulate what we need with SSE up to 4.1. */
#ifndef __AVX2__

typedef struct {
  __m128i lo;
  __m128i hi;
} mm256i_emu;
typedef __m256i real_m256i;
#define __m256i mm256i_emu


static inline mm256i_emu mm256_loadu_si256(const mm256i_emu *src) {
  mm256i_emu ret;
  ret.lo = _mm_loadu_si128((const __m128i*)src);
  ret.hi = _mm_loadu_si128((const __m128i*)(&((const char *)src)[16]));
  return ret;
}
#define _mm256_loadu_si256(src) mm256_loadu_si256(src)


static inline void mm256_storeu_si256(mm256i_emu *dst, mm256i_emu src) {
  _mm_storeu_si128((__m128i*)dst, src.lo);
  _mm_storeu_si128((__m128i*)(&((char *)dst)[16]), src.hi);
}
#define _mm256_storeu_si256(dst, src) mm256_storeu_si256(dst, src)


static inline mm256i_emu mm256_set1_epi32(int x) {
  mm256i_emu ret;
  ret.lo = _mm_set1_epi32(x);
  ret.hi = ret.lo;
  return ret;
}
#define _mm256_set1_epi32(x) mm256_set1_epi32(x)

static inline mm256i_emu mm256_set1_epi16(int x) {
  mm256i_emu ret;
  ret.lo = _mm_set1_epi16(x);
  ret.hi = ret.lo;
  return ret;
}
#define _mm256_set1_epi16(x) mm256_set1_epi16(x)


static inline mm256i_emu mm256_add_epi32(mm256i_emu a, mm256i_emu b) {
  mm256i_emu ret;
  ret.lo = _mm_add_epi32(a.lo, b.lo);
  ret.hi = _mm_add_epi32(a.hi, b.hi);
  return ret;
}
#define _mm256_add_epi32(a,b) mm256_add_epi32(a,b)

static inline mm256i_emu mm256_madd_epi16(mm256i_emu a, mm256i_emu b) {
  mm256i_emu ret;
  ret.lo = _mm_madd_epi16(a.lo, b.lo);
  ret.hi = _mm_madd_epi16(a.hi, b.hi);
  return ret;
}
#define _mm256_madd_epi16(a,b) mm256_madd_epi16(a,b)

static inline mm256i_emu mm256_maddubs_epi16(mm256i_emu a, mm256i_emu b) {
  mm256i_emu ret;
  ret.lo = _mm_maddubs_epi16(a.lo, b.lo);
  ret.hi = _mm_maddubs_epi16(a.hi, b.hi);
  return ret;
}
#define _mm256_maddubs_epi16(a,b) mm256_maddubs_epi16(a,b)



/* Emulating the conversion functions is tricky because they use __m256i but are defined in AVX.
   So we need to make a special when only AVX is available. */
#ifdef __AVX__

typedef union {
  mm256i_emu fake;
  real_m256i real;
} mm256_union;

static inline __m256 mm256_cvtepi32_ps(mm256i_emu a) {
  mm256_union src;
  src.fake = a;
  return _mm256_cvtepi32_ps(src.real);
}
#define _mm256_cvtepi32_ps(a) mm256_cvtepi32_ps(a)

static inline mm256i_emu mm256_cvtps_epi32(__m256 a) {
  mm256_union ret;
  ret.real =   _mm256_cvtps_epi32(a);
  return ret.fake;
}
#define _mm256_cvtps_epi32(a) mm256_cvtps_epi32(a)


#else

static inline mm256_emu mm256_cvtepi32_ps(mm256i_emu a) {
  mm256_emu ret;
  ret.lo = _mm_cvtepi32_ps(a.lo);
  ret.hi = _mm_cvtepi32_ps(a.hi);
  return ret;
}
#define _mm256_cvtepi32_ps(a) mm256_cvtepi32_ps(a)

static inline mm256i_emu mm256_cvtps_epi32(mm256_emu a) {
  mm256i_emu ret;
  ret.lo = _mm_cvtps_epi32(a.lo);
  ret.hi = _mm_cvtps_epi32(a.hi);
  return ret;
}
#define _mm256_cvtps_epi32(a) mm256_cvtps_epi32(a)

#endif /* __AVX__ */


#endif /* __AVX2__ */

/* In case we don't have FMA, make it a mul and an add. */
#if !(defined(__FMA__) && defined(__AVX__))
#define _mm256_fmadd_ps(a,b,c) _mm256_add_ps(_mm256_mul_ps(a, b), c)
#define _mm_fmadd_ps(a,b,c) _mm_add_ps(_mm_mul_ps(a, b), c)
#endif

#ifdef __AVX2__
static inline __m256 exp8_approx(__m256 X)
{
   const __m256 K0 = _mm256_set1_ps(0.99992522f);
   const __m256 K1 = _mm256_set1_ps(0.69583354f);
   const __m256 K2 = _mm256_set1_ps(0.22606716f);
   const __m256 K3 = _mm256_set1_ps(0.078024523f);
   const __m256 log2_E = _mm256_set1_ps(1.44269504);
   const __m256 max_in = _mm256_set1_ps(50.f);
   const __m256 min_in = _mm256_set1_ps(-50.f);
   __m256 XF, Y;
   __m256i I;
   X = _mm256_mul_ps(X, log2_E);
   X = _mm256_max_ps(min_in, _mm256_min_ps(max_in, X));
   XF = _mm256_floor_ps(X);
   I = _mm256_cvtps_epi32(XF);
   X = _mm256_sub_ps(X, XF);
   Y = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(K3, X, K2), X, K1), X, K0);
   I = _mm256_slli_epi32(I, 23);
   Y = _mm256_castsi256_ps(_mm256_add_epi32(I, _mm256_castps_si256(Y)));
   return Y;
}

static inline void vector_ps_to_epi8(unsigned char *x, const float *_x, int len) {
    int i;
   __m256 const127 = _mm256_set1_ps(127.f);
    for (i=0;i<len;i+=8) {
       __m256 xf;
       __m256i xi;
       xf = _mm256_loadu_ps(&_x[i]);
       //xf = _mm256_mul_ps(xf, const127);
       //xf = _mm256_add_ps(xf, const127);
       xf = _mm256_fmadd_ps(xf, const127, const127);
       xi = _mm256_cvtps_epi32(xf);
       xi = _mm256_packus_epi32(xi,  _mm256_setzero_si256());
       xi = _mm256_permute4x64_epi64(xi, 0xD8);
       xi = _mm256_packus_epi16(xi, _mm256_setzero_si256());
       xi = _mm256_permutevar8x32_epi32(xi, _mm256_setr_epi32(0,1, 0,0, 0,0, 0,0));
       //xi = _mm256_permute4x64_epi64(xi, 0x);
       _mm256_storeu_si256 ((__m256i *)&x[i], xi);
   }
}

#else
static inline __m128 exp4_approx(__m128 X)
{
   const __m128 K0 = _mm_set1_ps(0.99992522f);
   const __m128 K1 = _mm_set1_ps(0.69583354f);
   const __m128 K2 = _mm_set1_ps(0.22606716f);
   const __m128 K3 = _mm_set1_ps(0.078024523f);
   const __m128 log2_E = _mm_set1_ps(1.44269504);
   const __m128 max_in = _mm_set1_ps(50.f);
   const __m128 min_in = _mm_set1_ps(-50.f);
   const __m128i mask = _mm_set1_epi32(0x7fffffff);
   __m128 XF, Y;
   __m128i I;
   X = _mm_mul_ps(X, log2_E);
   X = _mm_max_ps(min_in, _mm_min_ps(max_in, X));
   XF = _mm_floor_ps(X);
   I = _mm_cvtps_epi32(XF);
   X = _mm_sub_ps(X, XF);
   Y = _mm_fmadd_ps(_mm_fmadd_ps(_mm_fmadd_ps(K3, X, K2), X, K1), X, K0);
   I = _mm_slli_epi32(I, 23);
   Y = _mm_castsi128_ps(_mm_and_si128(mask, _mm_add_epi32(I, _mm_castps_si128(Y))));
   return Y;
}
static inline __m256 exp8_approx(__m256 X)
{
   __m256 Y;
   __m128 Xhi, Xlo, Yhi, Ylo;
   Xhi = _mm256_extractf128_ps(X, 1);
   Xlo = _mm256_extractf128_ps(X, 0);
   Yhi = exp4_approx(Xhi);
   Ylo = exp4_approx(Xlo);
   Y = _mm256_insertf128_ps(_mm256_setzero_ps(), Yhi, 1);
   Y = _mm256_insertf128_ps(Y, Ylo, 0);
   return Y;
}

static inline void vector_ps_to_epi8(unsigned char *x, const float *_x, int len) {
    int i;
    for (i=0;i<len;i++) x[i] = 127+floor(.5+127*_x[i]);
}

#endif


#ifdef __AVX__

/* Approximating tanh() using a Padé-like rational function:
   tanh(x) ~= x * (N0 + N1*x^2 + N2*x^4)/(D0 + D1*x^2 + D2*x^4)
   subject to the +/- 1 bounds.
   The coefficients were determined by gradient descent trying to minimize
   the maximum deviation over the whole range (this is only possible because
   of the bounds). The max error is around 3e-4 and is dominated by the
   reciprocal approximation (the max error of the rational function is
   around 6e-5).
   */
static inline __m256 tanh8_approx(__m256 X)
{
   const __m256 N0 = _mm256_set1_ps(952.52801514f);
   const __m256 N1 = _mm256_set1_ps(96.39235687f);
   const __m256 N2 = _mm256_set1_ps(0.60863042f);
   const __m256 D0 = _mm256_set1_ps(952.72399902f);
   const __m256 D1 = _mm256_set1_ps(413.36801147f);
   const __m256 D2 = _mm256_set1_ps(11.88600922f);
   const __m256 max_out = _mm256_set1_ps(1.f);
   const __m256 min_out = _mm256_set1_ps(-1.f);
   __m256 X2, num, den;
   X2 = _mm256_mul_ps(X, X);
   num = _mm256_fmadd_ps(_mm256_fmadd_ps(N2, X2, N1), X2, N0);
   den = _mm256_fmadd_ps(_mm256_fmadd_ps(D2, X2, D1), X2, D0);
   num = _mm256_mul_ps(num, X);
   den = _mm256_rcp_ps(den);
   num = _mm256_mul_ps(num, den);
   return _mm256_max_ps(min_out, _mm256_min_ps(max_out, num));
}

/* Sigmoid approximation using a Padé-like rational function:
   1/(1+exp(-x)) ~= 0.5 + x * (N0 + N1*x^2 + N2*x^4)/(D0 + D1*x^2 + D2*x^4)
   subject to the [0, 1] bounds.
   The coefficients are directly derived by dividing the tanh() coefficients
   by powers of two to get the correct scaling. The max error is around 1.5e-4
   and is dominated by the reciprocal approximation (the max error of the
   rational function is around 3e-5).
   */
static inline __m256 sigmoid8_approx(__m256 X)
{
   const __m256 N0 = _mm256_set1_ps(238.13200378f);
   const __m256 N1 = _mm256_set1_ps(6.02452230f);
   const __m256 N2 = _mm256_set1_ps(0.00950985f);
   const __m256 D0 = _mm256_set1_ps(952.72399902f);
   const __m256 D1 = _mm256_set1_ps(103.34200287f);
   const __m256 D2 = _mm256_set1_ps(0.74287558f);
   const __m256 half = _mm256_set1_ps(0.5);
   const __m256 max_out = _mm256_set1_ps(1.f);
   const __m256 min_out = _mm256_set1_ps(0.f);
   __m256 X2, num, den;
   X2 = _mm256_mul_ps(X, X);
   num = _mm256_fmadd_ps(_mm256_fmadd_ps(N2, X2, N1), X2, N0);
   den = _mm256_fmadd_ps(_mm256_fmadd_ps(D2, X2, D1), X2, D0);
   num = _mm256_mul_ps(num, X);
   den = _mm256_rcp_ps(den);
   num = _mm256_fmadd_ps(num, den, half);
   return _mm256_max_ps(min_out, _mm256_min_ps(max_out, num));
}

static inline float tanh_approx(float x)
{
   float out[8];
   __m256 X, Y;
   X = _mm256_set1_ps(x);
   Y = tanh8_approx(X);
   _mm256_storeu_ps(out, Y);
   return out[0];
}

static inline float sigmoid_approx(float x)
{
   float out[8];
   __m256 X, Y;
   X = _mm256_set1_ps(x);
   Y = sigmoid8_approx(X);
   _mm256_storeu_ps(out, Y);
   return out[0];
}

#else

static inline __m128 tanh4_approx(__m128 X)
{
   const __m128 N0 = _mm_set1_ps(952.52801514f);
   const __m128 N1 = _mm_set1_ps(96.39235687f);
   const __m128 N2 = _mm_set1_ps(0.60863042f);
   const __m128 D0 = _mm_set1_ps(952.72399902f);
   const __m128 D1 = _mm_set1_ps(413.36801147f);
   const __m128 D2 = _mm_set1_ps(11.88600922f);
   const __m128 max_out = _mm_set1_ps(1.f);
   const __m128 min_out = _mm_set1_ps(-1.f);
   __m128 X2, num, den;
   X2 = _mm_mul_ps(X, X);
   num = _mm_fmadd_ps(_mm_fmadd_ps(N2, X2, N1), X2, N0);
   den = _mm_fmadd_ps(_mm_fmadd_ps(D2, X2, D1), X2, D0);
   num = _mm_mul_ps(num, X);
   den = _mm_rcp_ps(den);
   num = _mm_mul_ps(num, den);
   return _mm_max_ps(min_out, _mm_min_ps(max_out, num));
}

static inline __m128 sigmoid4_approx(__m128 X)
{
   const __m128 N0 = _mm_set1_ps(238.13200378f);
   const __m128 N1 = _mm_set1_ps(6.02452230f);
   const __m128 N2 = _mm_set1_ps(0.00950985f);
   const __m128 D0 = _mm_set1_ps(952.72399902f);
   const __m128 D1 = _mm_set1_ps(103.34200287f);
   const __m128 D2 = _mm_set1_ps(0.74287558f);
   const __m128 half = _mm_set1_ps(0.5);
   const __m128 max_out = _mm_set1_ps(1.f);
   const __m128 min_out = _mm_set1_ps(0.f);
   __m128 X2, num, den;
   X2 = _mm_mul_ps(X, X);
   num = _mm_fmadd_ps(_mm_fmadd_ps(N2, X2, N1), X2, N0);
   den = _mm_fmadd_ps(_mm_fmadd_ps(D2, X2, D1), X2, D0);
   num = _mm_mul_ps(num, X);
   den = _mm_rcp_ps(den);
   num = _mm_fmadd_ps(num, den, half);
   return _mm_max_ps(min_out, _mm_min_ps(max_out, num));
}

static inline float tanh_approx(float x)
{
   float out[4];
   __m128 X, Y;
   X = _mm_set1_ps(x);
   Y = tanh4_approx(X);
   _mm_storeu_ps(out, Y);
   return out[0];
}

static inline float sigmoid_approx(float x)
{
   float out[4];
   __m128 X, Y;
   X = _mm_set1_ps(x);
   Y = sigmoid4_approx(X);
   _mm_storeu_ps(out, Y);
   return out[0];
}

#endif

static inline float celt_exp(float x)
{
   float out[8];
   __m256 X, Y;
   X = _mm256_set1_ps(x);
   Y = exp8_approx(X);
   _mm256_storeu_ps(out, Y);
   return out[0];
}

static inline void softmax(float *y, const float *x, int N)
{
    int i;
    for (i=0;i<N-7;i+=8)
    {
        __m256 X, Y;
        X = _mm256_loadu_ps(&x[i]);
        Y = exp8_approx(X);
        _mm256_storeu_ps(&y[i], Y);
    }
    for (;i<N;i++)
        y[i] = celt_exp(x[i]);
}

#ifdef __AVX__
static inline void vec_tanh(float *y, const float *x, int N)
{
    int i;
    for (i=0;i<N-7;i+=8)
    {
        __m256 X, Y;
        X = _mm256_loadu_ps(&x[i]);
        Y = tanh8_approx(X);
        _mm256_storeu_ps(&y[i], Y);
    }
    for (;i<N;i++)
    {
        y[i] = tanh_approx(x[i]);
    }
}

static inline void vec_sigmoid(float *y, const float *x, int N)
{
    int i;
    for (i=0;i<N-7;i+=8)
    {
        __m256 X, Y;
        X = _mm256_loadu_ps(&x[i]);
        Y = sigmoid8_approx(X);
        _mm256_storeu_ps(&y[i], Y);
    }
    for (;i<N;i++)
    {
        y[i] = sigmoid_approx(x[i]);
    }
}
#else
static inline void vec_tanh(float *y, const float *x, int N)
{
    int i;
    for (i=0;i<N-3;i+=4)
    {
        __m128 X, Y;
        X = _mm_loadu_ps(&x[i]);
        Y = tanh4_approx(X);
        _mm_storeu_ps(&y[i], Y);
    }
    for (;i<N;i++)
    {
        y[i] = tanh_approx(x[i]);
    }
}

static inline void vec_sigmoid(float *y, const float *x, int N)
{
    int i;
    for (i=0;i<N-3;i+=4)
    {
        __m128 X, Y;
        X = _mm_loadu_ps(&x[i]);
        Y = sigmoid4_approx(X);
        _mm_storeu_ps(&y[i], Y);
    }
    for (;i<N;i++)
    {
        y[i] = sigmoid_approx(x[i]);
    }
}

#endif

static inline void sgemv_accum16(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
   int i, j;
   for (i=0;i<rows;i+=16)
   {
      float * y;
      __m256 vy0, vy8;
      y = &out[i];
      vy0 = _mm256_loadu_ps(&y[0]);
      vy8 = _mm256_loadu_ps(&y[8]);
      for (j=0;j<cols;j++)
      {
         __m256 vxj;
         __m256 vw;
         vxj = _mm256_broadcast_ss(&x[j]);

         vw = _mm256_loadu_ps(&weights[j*col_stride + i]);
         vy0 = _mm256_fmadd_ps(vw, vxj, vy0);

         vw = _mm256_loadu_ps(&weights[j*col_stride + i + 8]);
         vy8 = _mm256_fmadd_ps(vw, vxj, vy8);
      }
      _mm256_storeu_ps (&y[0], vy0);
      _mm256_storeu_ps (&y[8], vy8);
   }
}
static inline void sparse_sgemv_accum16(float *out, const float *weights, int rows, const int *idx, const float *x)
{
   int i, j;
   for (i=0;i<rows;i+=16)
   {
      float *  y;
      int cols;
      __m256 vy0, vy8;
      y = &out[i];
      vy0 = _mm256_loadu_ps(&y[0]);
      vy8 = _mm256_loadu_ps(&y[8]);
      cols = *idx++;
      for (j=0;j<cols;j++)
      {
         int id;
         __m256 vxj;
         __m256 vw;
         id = *idx++;
         vxj = _mm256_broadcast_ss(&x[id]);

         vw = _mm256_loadu_ps(&weights[0]);
         vy0 = _mm256_fmadd_ps(vw, vxj, vy0);

         vw = _mm256_loadu_ps(&weights[8]);
         vy8 = _mm256_fmadd_ps(vw, vxj, vy8);
         weights += 16;
      }
      _mm256_storeu_ps (&y[0], vy0);
      _mm256_storeu_ps (&y[8], vy8);
   }
}

#ifdef DOT_PROD
#define USE_SU_BIAS

typedef signed char qweight;


#define MAX_INPUTS (2048)
#define MAX_OUTPUTS (8192)


#define SCALE (128.f*127.f)
#define SCALE_1 (1.f/128.f/127.f)

#if 1
static inline void sgemv_accum8x4(float *_out, const qweight *w, int rows, int cols, int col_stride, const float *_x)
{
   __m256i ones;
   int i, j;
   unsigned char x[MAX_INPUTS];
   (void)col_stride;
   ones = _mm256_set1_epi16(1);
   //for (i=0;i<cols;i++) x[i] = 127+floor(.5+127*_x[i]);
   vector_ps_to_epi8(x, _x, cols);
   for (i=0;i<rows;i+=8)
   {
      __m256i vy0;
      __m256 vout;
      vout = _mm256_loadu_ps(&_out[i]);
      vout = _mm256_mul_ps(vout, _mm256_set1_ps(SCALE));
      vy0 = _mm256_cvtps_epi32(vout);
      j=0;
#if 1 /* Unrolling by 4 gives some gain, comment out if it does not. */
      for (;j<cols-12;j+=16)
      {
         __m256i tmp;
         __m256i vxj;
         __m256i vw;
         vxj = _mm256_set1_epi32(*(int*)&x[j]);
         vw = _mm256_loadu_si256((const __m256i *)w); //_mm256_lddqu_si256?
         tmp = _mm256_maddubs_epi16(vxj, vw); //swap?
         tmp = _mm256_madd_epi16(tmp, ones);
         vy0 = _mm256_add_epi32(vy0, tmp);
         w += 32;
         vxj = _mm256_set1_epi32(*(int*)&x[j+4]);
         vw = _mm256_loadu_si256((const __m256i *)w); //_mm256_lddqu_si256?
         tmp = _mm256_maddubs_epi16(vxj, vw); //swap?
         tmp = _mm256_madd_epi16(tmp, ones);
         vy0 = _mm256_add_epi32(vy0, tmp);
         w += 32;
         vxj = _mm256_set1_epi32(*(int*)&x[j+8]);
         vw = _mm256_loadu_si256((const __m256i *)w); //_mm256_lddqu_si256?
         tmp = _mm256_maddubs_epi16(vxj, vw); //swap?
         tmp = _mm256_madd_epi16(tmp, ones);
         vy0 = _mm256_add_epi32(vy0, tmp);
         w += 32;
         vxj = _mm256_set1_epi32(*(int*)&x[j+12]);
         vw = _mm256_loadu_si256((const __m256i *)w); //_mm256_lddqu_si256?
         tmp = _mm256_maddubs_epi16(vxj, vw); //swap?
         tmp = _mm256_madd_epi16(tmp, ones);
         vy0 = _mm256_add_epi32(vy0, tmp);
         w += 32;
      }
#endif
      for (;j<cols;j+=4)
      {
         __m256i tmp;
         __m256i vxj;
         __m256i vw;
         vxj = _mm256_set1_epi32(*(int*)&x[j]);
         vw = _mm256_loadu_si256((const __m256i *)w); //_mm256_lddqu_si256?
         tmp = _mm256_maddubs_epi16(vxj, vw); //swap?
         tmp = _mm256_madd_epi16(tmp, ones);
         vy0 = _mm256_add_epi32(vy0, tmp);
         w += 32;
      }
      vout = _mm256_cvtepi32_ps(vy0);
      vout = _mm256_mul_ps(vout, _mm256_set1_ps(SCALE_1));
      _mm256_storeu_ps(&_out[i], vout);
   }
}
#else
static inline void sgemv_accum8x4(float *out, const qweight *w, int rows, int cols, int col_stride, const float *_x)
{
   int i, j;
   unsigned char x[MAX_INPUTS];
   (void)col_stride;
   for (i=0;i<rows;i++) out[i] *= SCALE;
   for (i=0;i<cols;i++) x[i] = 127+(int)floor(.5+127*_x[i]);
   for (i=0;i<rows;i+=8)
   {
      for (j=0;j<cols;j+=4)
      {
         float *  y;
         float xj0, xj1, xj2, xj3;
         xj0 = x[j+0];
         xj1 = x[j+1];
         xj2 = x[j+2];
         xj3 = x[j+3];
         y = &out[i];
         y[0] += (w[0]*xj0+w[1]*xj1+w[2]*xj2+w[3]*xj3);
         y[1] += (w[4]*xj0+w[5]*xj1+w[6]*xj2+w[7]*xj3);
         y[2] += (w[8]*xj0+w[9]*xj1+w[10]*xj2+w[11]*xj3);
         y[3] += (w[12]*xj0+w[13]*xj1+w[14]*xj2+w[15]*xj3);
         y[4] += (w[16]*xj0+w[17]*xj1+w[18]*xj2+w[19]*xj3);
         y[5] += (w[20]*xj0+w[21]*xj1+w[22]*xj2+w[23]*xj3);
         y[6] += (w[24]*xj0+w[25]*xj1+w[26]*xj2+w[27]*xj3);
         y[7] += (w[28]*xj0+w[29]*xj1+w[30]*xj2+w[31]*xj3);
         w += 32;
      }
   }
   for (i=0;i<rows;i++) out[i] *= SCALE_1;
}
#endif

#define AMX_USE 

#ifdef AMX_USE

#define SIZE 128 
#define AMX_M 8
#define AMX_N 1
#define AMX_K 16
#define AMX_K4 4

#include "gemm_amx.h"
#include<stdio.h>

static int gA[AMX_M][AMX_K], gB[AMX_K][AMX_N], gC[AMX_M][AMX_N], gD[AMX_M][AMX_N];
static int gA1[AMX_M][AMX_K4], gB1[AMX_K4][AMX_N], gC1[AMX_M][AMX_N], gD1[AMX_M][AMX_N];

#endif

static int cnter=0;

#define INTEL_DEBUG

static inline void save_fp32_to_file(char *filename, float *out, int size){
    #ifndef INTEL_DEBUG
    return;
    #endif
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "int %s_size = %d;\n", filename, size);
    fprintf(fp, "float %s[] = {", filename);
    for(int i=0;i< size;i++){
        fprintf(fp, "%f, ", out[i]);
        if((i+1) % 10 ==0) fprintf(fp, "\n");
    }
    fprintf(fp, "\n};");
    fclose(fp);
}

static inline void save_u8_to_file(char *filename, unsigned char *out, int size){
    #ifndef INTEL_DEBUG
    return;
    #endif
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "int %s_size = %d;\n", filename, size);
    fprintf(fp, "unsigned char %s[] = {", filename);
    for(int i=0;i< size;i++){
        fprintf(fp, "%d, ", out[i]);
        if((i+1) % 10 ==0) fprintf(fp, "\n");
    }
    fprintf(fp, "\n};");
    fclose(fp);
}

static inline void save_s8_to_file(char *filename, char *out, int size){
    #ifndef INTEL_DEBUG
    return;
    #endif
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "int %s_size = %d;\n", filename, size);
    fprintf(fp, "char %s[] = {", filename);
    for(int i=0;i< size;i++){
        fprintf(fp, "%d, ", out[i]);
        if((i+1) % 10 ==0) fprintf(fp, "\n");
    }
    fprintf(fp, "\n};");
    fclose(fp);
}

static inline void p_s8(const char *filename, const signed char *out, int size){
    #ifndef INTEL_DEBUG
    return;
    #endif
    printf("%s size=%d\n", filename, size);
    
    for(int i=0;i< size;i++){
        printf("%d ", out[i]);
        if((i+1) % 10 ==0) printf("\n");
    }
    printf("\n");    
}

static inline void p_u8(const char *filename, const unsigned char *out, int size){
    #ifndef INTEL_DEBUG
    return;
    #endif
    printf("%s size=%d\n", filename, size);
    
    for(int i=0;i< size;i++){
        printf("%d ", out[i]);
        if((i+1) % 10 ==0) printf("\n");
    }
    printf("\n");    
}

static inline void p_fp32(const char *filename, const float *out, int size){
    #ifndef INTEL_DEBUG
    return;
    #endif
    printf("%s size=%d\n", filename, size);
    
    for(int i=0;i< size;i++){
        printf("%f ", out[i]);
        if((i+1) % 10 ==0) printf("\n");
    }
    printf("\n");    
}

#define ALLOW_PRINT if(cnter >= 105600)
//#define ALLOW_PRINT if(0)
#define NOT_PRINT if(0)
    

static inline void sparse_sgemv_accum8x4_avx2(float *_out, const qweight *w, int rows, int cols, const int *idx, const float *_x)
{
    printf("rows=%d, cols=%d\n", rows, cols);
    
   __m256i ones;
   int i, j;
   unsigned char x[MAX_INPUTS];
   ones = _mm256_set1_epi16(1);
   //for (i=0;i<cols;i++) x[i] = 127+floor(.5+127*_x[i]);
   vector_ps_to_epi8(x, _x, cols);
   
   save_fp32_to_file("avx2/out0.h", _out, rows);
   save_fp32_to_file("avx2/x0_fp32.h", _x, cols);
   save_u8_to_file("avx2/x0.h", x, cols);
   save_s8_to_file("avx2/w0.h", w, 44224);
   
   for (i=0;i<rows;i+=8)
   {
      int colblocks;
      __m256i vy0;
      __m256 vout;
      colblocks = *idx++;
      vout = _mm256_loadu_ps(&_out[i]);
      vout = _mm256_mul_ps(vout, _mm256_set1_ps(SCALE));
      vy0 = _mm256_cvtps_epi32(vout);
      j=0;
#if 1 /* Unrolling by 4 gives some gain, comment out if it does not. */
      for (;j<colblocks-3;j+=4)
      {
         __m256i tmp;
         __m256i vxj;
         __m256i vw;
         vxj = _mm256_set1_epi32(*(int*)&x[*idx++]);
         vw = _mm256_loadu_si256((const __m256i *)w); //_mm256_lddqu_si256?
         tmp = _mm256_maddubs_epi16(vxj, vw); //swap?
         tmp = _mm256_madd_epi16(tmp, ones);
         vy0 = _mm256_add_epi32(vy0, tmp);
         w += 32;
         vxj = _mm256_set1_epi32(*(int*)&x[*idx++]);
         vw = _mm256_loadu_si256((const __m256i *)w); //_mm256_lddqu_si256?
         tmp = _mm256_maddubs_epi16(vxj, vw); //swap?
         tmp = _mm256_madd_epi16(tmp, ones);
         vy0 = _mm256_add_epi32(vy0, tmp);
         w += 32;
         vxj = _mm256_set1_epi32(*(int*)&x[*idx++]);
         vw = _mm256_loadu_si256((const __m256i *)w); //_mm256_lddqu_si256?
         tmp = _mm256_maddubs_epi16(vxj, vw); //swap?
         tmp = _mm256_madd_epi16(tmp, ones);
         vy0 = _mm256_add_epi32(vy0, tmp);
         w += 32;
         vxj = _mm256_set1_epi32(*(int*)&x[*idx++]);
         vw = _mm256_loadu_si256((const __m256i *)w); //_mm256_lddqu_si256?
         tmp = _mm256_maddubs_epi16(vxj, vw); //swap?
         tmp = _mm256_madd_epi16(tmp, ones);
         vy0 = _mm256_add_epi32(vy0, tmp);
         w += 32;
      }
#endif
      for (;j<colblocks;j++)
      {
         __m256i tmp;
         __m256i vxj;
         __m256i vw;
         int pos;
         pos = (*idx++);
         vxj = _mm256_set1_epi32(*(int*)&x[pos]);
         vw = _mm256_loadu_si256((const __m256i *)w); //_mm256_lddqu_si256?
         tmp = _mm256_maddubs_epi16(vxj, vw); //swap?
         tmp = _mm256_madd_epi16(tmp, ones);
         vy0 = _mm256_add_epi32(vy0, tmp);
         w += 32;
      }
      vout = _mm256_cvtepi32_ps(vy0);
      vout = _mm256_mul_ps(vout, _mm256_set1_ps(SCALE_1));
      _mm256_storeu_ps(&_out[i], vout);
      char tmp[1024];
      //sprintf(tmp, "avx2/out01_%d.h", i);
      //printf("tmp %s\n", tmp);
      //save_fp32_to_file(tmp, _out, rows);
   }
   save_fp32_to_file("avx2/out1.h", _out, rows);
   
}

static inline void sparse_sgemv_accum8x4_c(float *out, const qweight *w, int rows, int cols, const int *idx, const float *_x)
{
    //int gA[AMX_M][AMX_K], gB[AMX_K][AMX_N], gC[AMX_M][AMX_N], gD[AMX_M][AMX_N];
    
   printf("rows=%d, cols=%d\n", rows, cols);
   int i, j;
   unsigned char x[MAX_INPUTS];
   save_fp32_to_file("c/out0.h", out, rows);
   for (i=0;i<rows;i++) out[i] *= SCALE;
   for (i=0;i<cols;i++) x[i] = 127+floor(.5+127*_x[i]);
   int cnt=0;
   save_fp32_to_file("c/x0_fp32.h", _x, cols);
   save_u8_to_file("c/x0.h", x, cols);
   save_s8_to_file("c/w0.h", w, 44224);
   
   for (i=0;i<rows;i+=8)
   {
      int colblocks;
      colblocks = *idx++;
      //printf("colblocks=%d\n", colblocks);
      j=0;
      for (;j<colblocks-3;j+=4)
      {
         int pos;
         float * y;
         int xj0, xj1, xj2, xj3;
         pos = (*idx++);
         ALLOW_PRINT printf("i=%d, pos=%d\n", i, pos);
         
         //gA
         xj0 = x[pos+0];
         xj1 = x[pos+1];
         xj2 = x[pos+2];
         xj3 = x[pos+3];
         y = &out[i];
         ALLOW_PRINT {
             p_fp32("y0", y, 8);             
         }
         
         y[0] += (w[0]*xj0+w[1]*xj1+w[2]*xj2+w[3]*xj3);
         y[1] += (w[4]*xj0+w[5]*xj1+w[6]*xj2+w[7]*xj3);
         y[2] += (w[8]*xj0+w[9]*xj1+w[10]*xj2+w[11]*xj3);
         y[3] += (w[12]*xj0+w[13]*xj1+w[14]*xj2+w[15]*xj3);
         y[4] += (w[16]*xj0+w[17]*xj1+w[18]*xj2+w[19]*xj3);
         y[5] += (w[20]*xj0+w[21]*xj1+w[22]*xj2+w[23]*xj3);
         y[6] += (w[24]*xj0+w[25]*xj1+w[26]*xj2+w[27]*xj3);
         y[7] += (w[28]*xj0+w[29]*xj1+w[30]*xj2+w[31]*xj3);
         w += 32;
         cnt += 32;
         
         pos = (*idx++);
         ALLOW_PRINT printf("i=%d, pos=%d\n", i, pos);
         xj0 = x[pos+0];
         xj1 = x[pos+1];
         xj2 = x[pos+2];
         xj3 = x[pos+3];
         y = &out[i];
         y[0] += (w[0]*xj0+w[1]*xj1+w[2]*xj2+w[3]*xj3);
         y[1] += (w[4]*xj0+w[5]*xj1+w[6]*xj2+w[7]*xj3);
         y[2] += (w[8]*xj0+w[9]*xj1+w[10]*xj2+w[11]*xj3);
         y[3] += (w[12]*xj0+w[13]*xj1+w[14]*xj2+w[15]*xj3);
         y[4] += (w[16]*xj0+w[17]*xj1+w[18]*xj2+w[19]*xj3);
         y[5] += (w[20]*xj0+w[21]*xj1+w[22]*xj2+w[23]*xj3);
         y[6] += (w[24]*xj0+w[25]*xj1+w[26]*xj2+w[27]*xj3);
         y[7] += (w[28]*xj0+w[29]*xj1+w[30]*xj2+w[31]*xj3);
         w += 32;
         cnt += 32;
         
         pos = (*idx++);
         ALLOW_PRINT printf("i=%d, pos=%d\n", i, pos);
         xj0 = x[pos+0];
         xj1 = x[pos+1];
         xj2 = x[pos+2];
         xj3 = x[pos+3];
         y = &out[i];
         y[0] += (w[0]*xj0+w[1]*xj1+w[2]*xj2+w[3]*xj3);
         y[1] += (w[4]*xj0+w[5]*xj1+w[6]*xj2+w[7]*xj3);
         y[2] += (w[8]*xj0+w[9]*xj1+w[10]*xj2+w[11]*xj3);
         y[3] += (w[12]*xj0+w[13]*xj1+w[14]*xj2+w[15]*xj3);
         y[4] += (w[16]*xj0+w[17]*xj1+w[18]*xj2+w[19]*xj3);
         y[5] += (w[20]*xj0+w[21]*xj1+w[22]*xj2+w[23]*xj3);
         y[6] += (w[24]*xj0+w[25]*xj1+w[26]*xj2+w[27]*xj3);
         y[7] += (w[28]*xj0+w[29]*xj1+w[30]*xj2+w[31]*xj3);
         w += 32;
         cnt += 32;
         
         pos = (*idx++);
         ALLOW_PRINT printf("i=%d, pos=%d\n", i, pos);
         xj0 = x[pos+0];
         xj1 = x[pos+1];
         xj2 = x[pos+2];
         xj3 = x[pos+3];
         y = &out[i];
         y[0] += (w[0]*xj0+w[1]*xj1+w[2]*xj2+w[3]*xj3);
         y[1] += (w[4]*xj0+w[5]*xj1+w[6]*xj2+w[7]*xj3);
         y[2] += (w[8]*xj0+w[9]*xj1+w[10]*xj2+w[11]*xj3);
         y[3] += (w[12]*xj0+w[13]*xj1+w[14]*xj2+w[15]*xj3);
         y[4] += (w[16]*xj0+w[17]*xj1+w[18]*xj2+w[19]*xj3);
         y[5] += (w[20]*xj0+w[21]*xj1+w[22]*xj2+w[23]*xj3);
         y[6] += (w[24]*xj0+w[25]*xj1+w[26]*xj2+w[27]*xj3);
         y[7] += (w[28]*xj0+w[29]*xj1+w[30]*xj2+w[31]*xj3);
         w += 32;
         cnt += 32;
         
         ALLOW_PRINT {
             p_fp32("y1", y, 8);             
         }
         
      }
      for (;j<colblocks;j++)
      {
         //printf("handle rest j = %d\n", j);
         int pos;
         float * y;
         int xj0, xj1, xj2, xj3;
         pos = (*idx++);
         ALLOW_PRINT printf("i=%d, pos=%d\n", i, pos);
         xj0 = x[pos+0];
         xj1 = x[pos+1];
         xj2 = x[pos+2];
         xj3 = x[pos+3];
         y = &out[i];
         ALLOW_PRINT {
             p_fp32("y0_j", y, 8);    
             p_u8("x0_j",x+pos, 4);
             p_s8("w0_j",w, 32);
         }
         y[0] += (w[0]*xj0+w[1]*xj1+w[2]*xj2+w[3]*xj3);
         y[1] += (w[4]*xj0+w[5]*xj1+w[6]*xj2+w[7]*xj3);
         y[2] += (w[8]*xj0+w[9]*xj1+w[10]*xj2+w[11]*xj3);
         y[3] += (w[12]*xj0+w[13]*xj1+w[14]*xj2+w[15]*xj3);
         y[4] += (w[16]*xj0+w[17]*xj1+w[18]*xj2+w[19]*xj3);
         y[5] += (w[20]*xj0+w[21]*xj1+w[22]*xj2+w[23]*xj3);
         y[6] += (w[24]*xj0+w[25]*xj1+w[26]*xj2+w[27]*xj3);
         y[7] += (w[28]*xj0+w[29]*xj1+w[30]*xj2+w[31]*xj3);
         w += 32;
         cnt += 32;
         ALLOW_PRINT {
             p_fp32("y1_j", y, 8);             
         }
      }
      char tmp[1024];
      sprintf(tmp, "c/out01_%d.h", i);
      //printf("tmp %s\n", tmp);
      //save_fp32_to_file(tmp, out, rows);
      
   }
   //printf("w size=%d\n", cnt);
   
   for (i=0;i<rows;i++) out[i] *= SCALE_1;
   //p_fp32("out1", out, rows);
   save_fp32_to_file("c/out1.h", out, rows);
}


#define INTEL_OPT1

static inline void sparse_sgemv_accum8x4_amx(float *out, const qweight *w, int rows, int cols, const int *idx, const float *_x)
{
    //int gA[AMX_M][AMX_K], gB[AMX_K][AMX_N], gC[AMX_M][AMX_N], gD[AMX_M][AMX_N];
   //printf("rows=%d, cols=%d\n", rows, cols);
   //rows=1152, cols=384
   int i, j;
   unsigned char x[MAX_INPUTS];
   //save_fp32_to_file("amx/out0.h", out, rows);
   
    #ifdef INTEL_OPT 
      __m256i vy0; //8*int32
      __m256 vout; //8*float32     
    #else
    for (i=0;i<rows;i++) out[i] *= SCALE;
    #endif  
   /*
   32510 ms
   32414 ms
   */
      
   #ifdef INTEL_OPT
   vector_ps_to_epi8(x, _x, cols);
   #else
   //for (i=0;i<cols;i++) x[i] = 127+floor(.5+127*_x[i]);
   vector_ps_to_epi8(x, _x, cols); 
   #endif
   
   int cnt=0;
   
   //save_fp32_to_file("amx/out0_scale.h", out, rows);
   
   //save_fp32_to_file("amx/x0_fp32.h", _x, cols);
   //save_u8_to_file("amx/x0.h", x, cols);
   //save_s8_to_file("amx/w0.h", w, 44224);
   for (i=0;i<rows;i+=8)
   {
      //printf("i %d\n", i);
      int colblocks;
      colblocks = *idx++;
      NOT_PRINT printf("colblocks=%d\n", colblocks);
      j=0;
      float * y;
      
      #ifdef INTEL_OPT 
         vout = _mm256_loadu_ps(&out[i]);
         vout = _mm256_mul_ps(vout, _mm256_set1_ps(SCALE));
         vy0 = _mm256_cvtps_epi32(vout);       
         _mm256_store_epi32(gC, vy0);
      #else   
          y=&out[i];
          for(int k=0;k<8;k++){             
             gC[k][0]=y[k];             
          }
      #endif
      
      for (;j<colblocks-3;j+=4)
      {
         int pos;
    
         ALLOW_PRINT {
             p_fp32("y0", y, 8);             
         }
         
         /*
         reorder base
         for(int k=0;k<4;k++){
            for(int m=0;m<8;m++){
                for(int n=0;n<4;n++){
                        gA[m][k*4+n]=w[n+m*4+k*32];                 
                }
            }
            cnt += 32;
         }
         */
         //memcpy(gA, w, 4*8*4*sizeof(int))
         for(int k=0;k<8;k++){
            for(int m=0;m<16;m++){
                gA[k][m]=w[k*16+m];
                //ALLOW_PRINT printf("%d ", w[])
            }
            cnt += 32;
         }
         
         for(int k=0;k<4;k++){
             pos = (*idx++);
             NOT_PRINT printf("i=%d, pos=%d\n", i, pos);
             for(int m=0;m<4;m++){
                 gB[m+k*4][0]=x[pos+m];                 
             }
         }
         
        
         /*
         for(int k=0;k<8;k++){             
             gC[k][0]=y[k];             
         }
         */        
        
        //ALLOW_PRINT { 
        NOT_PRINT {       
            printf("gA\n");
            for(int d=0;d<8;d++){
                for(int e=0;e<16;e++){
                 printf("%d ", gA[d][e]);
                }
                printf("\n");
            }
            printf("gB\n");
            for(int d=0;d<16;d++){            
                printf("%d ", gB[d][0]);            
            } 
            printf("\n");
             
            printf("gC\n");
            for(int d=0;d<8;d++){            
                printf("%d ", gC[d][0]);            
            } 
            printf("\n");
        }
        //inner_product_ref((int *)gA, (int *)gB, (int *)gC, AMX_M, AMX_N, AMX_K);
        inner_product((int *)gA, (int *)gB, (int *)gC, AMX_M, AMX_N, AMX_K);
        
        
        /*
        for(int k=0;k<8;k++){             
            y[k]=(float)gC[k][0]; 
        }
        */
        
        
        ALLOW_PRINT {    
            /*
            printf("gC_1\n");
            for(int d=0;d<8;d++){            
                printf("%d ", gC[d][0]);            
            } 
            */
            printf("\n");        
             {
                 p_fp32("y1", y, 8);             
             }
        }
        w+=128;
      }
      
      //p_fp32("out01", out, rows);
      
      
      for (;j<colblocks;j++)
      {
         //printf("handle rest j=%d colblocks=%d\n", j, colblocks);
         ALLOW_PRINT printf("handle rest j = %d i =%d\n", j, i);
         int pos;
         //float * y=&out[i];  
         
                 
         ALLOW_PRINT {
             p_fp32("y0_j", y, 8);             
         }
         
         pos = (*idx++);
         NOT_PRINT printf("i=%d, pos=%d\n", i, pos);
         /*
         xj0 = x[pos+0];
         xj1 = x[pos+1];
         xj2 = x[pos+2];
         xj3 = x[pos+3];
         y = &out[i];
         y[0] += (w[0]*xj0+w[1]*xj1+w[2]*xj2+w[3]*xj3);
         y[1] += (w[4]*xj0+w[5]*xj1+w[6]*xj2+w[7]*xj3);
         y[2] += (w[8]*xj0+w[9]*xj1+w[10]*xj2+w[11]*xj3);
         y[3] += (w[12]*xj0+w[13]*xj1+w[14]*xj2+w[15]*xj3);
         y[4] += (w[16]*xj0+w[17]*xj1+w[18]*xj2+w[19]*xj3);
         y[5] += (w[20]*xj0+w[21]*xj1+w[22]*xj2+w[23]*xj3);
         y[6] += (w[24]*xj0+w[25]*xj1+w[26]*xj2+w[27]*xj3);
         y[7] += (w[28]*xj0+w[29]*xj1+w[30]*xj2+w[31]*xj3);
         */
         
         for(int k=0;k<1;k++){
            for(int m=0;m<8;m++){
                for(int n=0;n<4;n++){
                        gA1[m][k*4+n]=w[n+m*4+k*32];                 
                }
            }
            cnt += 32;
         }
         
         for(int k=0;k<1;k++){
             //pos = (*idx++);
             //printf("pos=%d\n", pos);
             for(int m=0;m<4;m++){
                 gB1[m+k*4][0]=x[pos+m]; 
                 //printf("gB1 x[pos+m]=%d %d\n", x[pos+m], pos+m);
             }
         }
         
        
         /*    
         for(int k=0;k<8;k++){             
             gC[k][0]=y[k];             
         }
         */
        
         
         
        //ALLOW_PRINT {
        NOT_PRINT {
            printf("gA1\n");
            for(int d=0;d<8;d++){
                for(int e=0;e<4;e++){
                 printf("%d ", gA1[d][e]);
                }
                printf("\n");
            }
            printf("gB1\n");
            for(int d=0;d<4;d++){            
                printf("%d ", gB1[d][0]);            
            } 
            printf("\n");
             
            printf("gC\n");
            for(int d=0;d<8;d++){            
                printf("%d ", gC[d][0]);            
            } 
            printf("\n");
        }
        
        inner_product((int *)gA1, (int *)gB1, (int *)gC, AMX_M, AMX_N, AMX_K4);
        //inner_product_ref((int *)gA1, (int *)gB1, (int *)gC, AMX_M, AMX_N, AMX_K4);
        
        
/*            
        for(int k=0;k<8;k++){             
            y[k]=(float)gC[k][0]; 
        }         
*/
       
        
        w += 32;
        
        ALLOW_PRINT {
            p_fp32("y2_j", y, 8);            
        }        
      }      
        
      #ifdef INTEL_OPT       
      vy0 = _mm256_load_epi32(gC);      
      vout = _mm256_cvtepi32_ps(vy0);
      vout = _mm256_mul_ps(vout, _mm256_set1_ps(SCALE_1));
      _mm256_storeu_ps(&out[i], vout);
      #else
      for(int k=0;k<8;k++){             
            y[k]=(float)gC[k][0]; 
      }    
      #endif
      char tmp[1024];
      //sprintf(tmp, "amx/out01_%d.h", i);
      //printf("tmp %s\n", tmp);
      //save_fp32_to_file(tmp, out, rows);
   }
   //ALLOW_PRINT printf("w size=%d\n", cnt);
   
   
   //#ifndef INTEL_OPT     
   for (i=0;i<rows;i++) out[i] *= SCALE_1;
   //#endif
   ALLOW_PRINT p_fp32("out", out, 8);
   //save_fp32_to_file("amx/out1.h", out, rows);
}


#define COMPARE_BASE

#ifdef COMPARE_BASE

static int compare_out(float *out, int rows){    
#include <math.h>
#include "out_first.h" 

    for(int i=0;i<rows;i++){
        if (fabs(amx_out1_h[i]-out[i])>0.0001){
            printf("diff %d current %f - expected %f\n", i, out[i], amx_out1_h[i]);
            return 1;
        }
        
    }
    return 0;
}

#endif

static inline void sparse_sgemv_accum8x4(float *out, const qweight *w, int rows, int cols, const int *idx, const float *_x){    
  cnter++;
   if (cnter < 0 ){
        //#ifdef INTEL_DEBUG
        return;
        //#endif       
   }
    //printf("avx2\n"); sparse_sgemv_accum8x4_avx2(out, w, rows, cols, idx, _x);
    //printf("amx cnter = %d\n", cnter); 
    sparse_sgemv_accum8x4_amx(out, w, rows, cols, idx, _x);
    //printf("c\n");sparse_sgemv_accum8x4_c(out, w, rows, cols, idx, _x);
    
    #ifdef COMPARE_BASE1
        #ifdef INTEL_DEBUG
        if (cnt ==1){
            int res = compare_out(out, rows);
            if (res==0) printf("-----------------result is passed\n");
            else printf("-----------------result is wrong\n");
        }
        #endif 
    #endif
}

#else /*DOT_PROD*/
typedef float qweight;
#define sgemv_accum8x4 sgemv_accum

static inline void sparse_sgemv_accum8x4(float *out, const qweight *weights, int rows, int ignore, const int *idx, const float *x)
{
   printf("rows=%d, cols=%d\n", rows, cols);
   int i, j;
   (void)ignore;
   for (i=0;i<rows;i+=8)
   {
      float *  y;
      int cols;
      __m256 vy0;
      y = &out[i];
      vy0 = _mm256_loadu_ps(&y[0]);
      cols = *idx++;
      for (j=0;j<cols;j++)
      {
         int id;
         __m256 vxj;
         __m256 vw;
         id = *idx++;
         vxj = _mm256_broadcast_ss(&x[id]);
         vw = _mm256_loadu_ps(&weights[0]);
         vy0 = _mm256_fmadd_ps(vw, vxj, vy0);

         vxj = _mm256_broadcast_ss(&x[id+1]);
         vw = _mm256_loadu_ps(&weights[8]);
         vy0 = _mm256_fmadd_ps(vw, vxj, vy0);

         vxj = _mm256_broadcast_ss(&x[id+2]);
         vw = _mm256_loadu_ps(&weights[16]);
         vy0 = _mm256_fmadd_ps(vw, vxj, vy0);

         vxj = _mm256_broadcast_ss(&x[id+3]);
         vw = _mm256_loadu_ps(&weights[24]);
         vy0 = _mm256_fmadd_ps(vw, vxj, vy0);

         weights += 32;
      }
      _mm256_storeu_ps (&y[0], vy0);
   }
}
#endif /*DOT_PROD*/

#endif /*VEC_AVX_H*/
