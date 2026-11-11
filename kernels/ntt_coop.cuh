/*
 * kernels/ntt_coop.cuh  ---  NTT/NTT Inverse : 다항식 1개를 스레드 128개가 함께 변환
 */
#ifndef MLDSA87_NTT_COOP_CUH
#define MLDSA87_NTT_COOP_CUH

#include "params.h"
#include "reduce.h"
#include <stdint.h>

#define NTT_COOP_THREADS 128 /* 다항식 1개를 맡는 스레드 수 = 한 레이어의 버터플라이 수 */

extern __constant__ int32_t zetas[N];

/* zeta 표를 상수메모리에서 공유메모리로 복사 */
__device__ __forceinline__ void zetas_to_shared(int32_t *z, unsigned int tid) {
    for (unsigned int i = tid; i < (unsigned)N; i += blockDim.x) { z[i] = zetas[i]; }
}

/*
 * NTT
 * a      : 공유메모리 다항식
 * z      : 공유메모리 zeta 표
 * tid    : 스레드 번호 : [0...127]
 * active : false 면 계산 없이 동기화만
 */
__device__ __forceinline__ void ntt_coop(int32_t *a, const int32_t *z, unsigned int tid, bool active = true) {
    for (unsigned int len = 128; len > 0; len >>= 1) {                    /* 8개 레이어: 128, 64, ... , 1 */
        if (active) {
            unsigned int g = tid / len;                                   /* 버터플라이 그룹 */
            unsigned int j = 2 * len * g + (tid % len);                   /* 담당 계수 위치 */
            int32_t zeta = z[(128 / len) + g];
            int32_t t = PQCLEAN_MLDSA87_CLEAN_montgomery_reduce((int64_t)zeta * a[j + len]);
            a[j + len] = a[j] - t;
            a[j] = a[j] + t;
        }
        __syncthreads();                                                  /* 다음 레이어 전 동기화 */
    }
}

/* 
 * NTT Inverse
 * a      : 공유메모리 다항식
 * z      : 공유메모리 zeta 표
 * tid    : 스레드 번호 : [0...127]
 * active : false 면 계산 없이 동기화만
 * 
 * 마지막에 모든 계수에 f = mont^2/256 = 41978을 곱함
 */
__device__ __forceinline__ void invntt_tomont_coop(int32_t *a, const int32_t *z, unsigned int tid, bool active = true) {
    const int32_t f = 41978;                                              /* mont^2 / 256 */
    for (unsigned int len = 1; len < (unsigned)N; len <<= 1) {
        if (active) {
            unsigned int G = 128 / len;                                   /* 이 레이어의 그룹 수 */
            unsigned int g = tid / len;
            unsigned int j = 2 * len * g + (tid % len);
            int32_t zeta = -z[2 * G - 1 - g];
            int32_t t = a[j];
            a[j] = t + a[j + len];
            a[j + len] = t - a[j + len];
            a[j + len] = PQCLEAN_MLDSA87_CLEAN_montgomery_reduce((int64_t)zeta * a[j + len]);
        }
        __syncthreads();
    }
    if (active) {
        a[tid] = PQCLEAN_MLDSA87_CLEAN_montgomery_reduce((int64_t)f * a[tid]);
        a[tid + NTT_COOP_THREADS] = PQCLEAN_MLDSA87_CLEAN_montgomery_reduce((int64_t)f * a[tid + NTT_COOP_THREADS]);
    }
    __syncthreads();
}

#endif
