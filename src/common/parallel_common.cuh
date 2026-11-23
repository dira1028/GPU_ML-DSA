/* Shared parallel arithmetic dependencies. */
#ifndef MLDSA87_MULTI_COMMON_CUH
#define MLDSA87_MULTI_COMMON_CUH

#include <cuda_runtime.h>
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "reduce.h"
#include "rounding.h"
#include "ntt_coop.cuh"
#include <stdint.h>

/**************************************************
 * 이름: MT_MR
 *
 * 설명: 계수 두 개의 Montgomery 곱 a·b·2^-32 mod q 계산
 *
 * 인수: - a, b: 곱할 계수 (int32)
 **************************************************/
#define MT_MR(a, b) PQCLEAN_MLDSA87_CLEAN_montgomery_reduce((int64_t)(a) * (b))

__device__ int mt_over(int32_t a, int32_t B);


#endif
