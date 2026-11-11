/*
 *   - KeyGen/Sign/Verify 각각을 커널 하나로 실행
 */
#ifndef MLDSA87_MULTI_COMMON_CUH
#define MLDSA87_MULTI_COMMON_CUH

#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "packing.h"
#include "symmetric.h"
#include "fips202.h"
#include "reduce.h"
#include "rounding.h"
#include "ntt_coop.cuh"
#include <cooperative_groups.h>
#include <stdint.h>

#define MT_THREADS 256                         /* 블록당 스레드 수 */
#define MT_GROUPS  (MT_THREADS / 128)          /* 블록당 128스레드 묶음 수 = 2 */

/* 공유메모리 배열 선언 */
#ifndef KSHARED
#define KSHARED(T, name, ...) __shared__ T name __VA_ARGS__
#endif

/* GPU 전역 타이머(ns) */
__device__ __forceinline__ unsigned long long mt_gtimer(void) {
#ifdef __CUDA_ARCH__
    unsigned long long v; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v)); return v;
#endif
}

#define MT_MR(a, b) PQCLEAN_MLDSA87_CLEAN_montgomery_reduce((int64_t)(a) * (b))

__device__ __forceinline__ int mt_over(int32_t a, int32_t B) {
    int32_t t = a >> 31;
    t = a - (t & 2 * a);
    return t >= B;
}

/* 작업 영역 0...n-1 할당 */
#define MT_FOR_TASKS(n, task, act)                                                       \
    for (unsigned int base_ = blockIdx.x * MT_GROUPS; base_ < (unsigned)(n);             \
         base_ += gridDim.x * MT_GROUPS)                                                 \
        for (unsigned int task = base_ + threadIdx.x / 128, once_ = 1; once_; once_ = 0) \
            for (bool act = task < (unsigned)(n); once_; once_ = 0)

#endif
