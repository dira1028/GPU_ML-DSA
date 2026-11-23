/*
 * kernels/ntt_coop.cuh  ---  NTT/NTT Inverse : 다항식 1개를 스레드 128개가 함께 변환
 */
#ifndef MLDSA87_NTT_COOP_CUH
#define MLDSA87_NTT_COOP_CUH

#include <cuda_runtime.h>
#include "params.h"
#include <stdint.h>

#define NTT_COOP_THREADS 128 /* 다항식 1개를 맡는 스레드 수 = 한 레이어의 버터플라이 수 */

extern __constant__ int32_t zetas[N];

/* zeta 표를 상수메모리에서 공유메모리로 복사 */
__device__ void zetas_to_shared(int32_t *z, unsigned int tid);

/*
 * NTT
 * a      : 공유메모리 다항식
 * z      : 공유메모리 zeta 표
 * tid    : 스레드 번호 : [0...127]
 * active : false 면 계산 없이 동기화만
 */
__device__ void ntt_coop(int32_t *a, const int32_t *z, unsigned int tid, bool active = true);

/* 
 * NTT Inverse
 * a      : 공유메모리 다항식
 * z      : 공유메모리 zeta 표
 * tid    : 스레드 번호 : [0...127]
 * active : false 면 계산 없이 동기화만
 * 
 * 마지막에 모든 계수에 f = mont^2/256 = 41978을 곱함
 */
__device__ void invntt_tomont_coop(int32_t *a, const int32_t *z, unsigned int tid, bool active = true);

#endif
