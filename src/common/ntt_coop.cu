#include "ntt_coop.cuh"
#include "reduce.h"


/**************************************************
 * 이름: zetas_to_shared
 *
 * 설명: zeta 표 (상수메모리 zetas[256]) 를 공유메모리로 복사
 *       협력 NTT 는 스레드마다 다른 zeta 를 읽으므로 공유메모리 배치가 유리
 *       블록의 모든 스레드가 나눠 복사하며, 호출 뒤 __syncthreads() 필요
 *
 * 인수: - z: 공유메모리 zeta 표 (int32 256개)
 *       - tid: 블록 안 스레드 번호
 **************************************************/
__device__ void zetas_to_shared(int32_t *z, unsigned int tid) {
    for (unsigned int i = tid; i < (unsigned)N; i += blockDim.x) { z[i] = zetas[i]; }
}

/**************************************************
 * 이름: ntt_coop
 *
 * 설명: 순방향 NTT. 다항식 1개를 128스레드가 변환
 *       레이어 길이 len 에서 스레드 tid 는 그룹 g = tid/len, 위치 j = 2·len·g + tid%len 의
 *       버터플라이 (a[j], a[j+len]) 를 담당하고, zeta 는 z[128/len + g] 사용
 *       레이어마다 __syncthreads() 를 하므로 블록의 모든 스레드가 같은 횟수로 호출 필요
 *
 * 인수: - a: 공유메모리 다항식 (제자리 변환)
 *       - z: 공유메모리 zeta 표
 *       - tid: 스레드 번호 0..127
 *       - active: false 면 계산 없이 동기화에만 참여
 **************************************************/
__device__ void ntt_coop(int32_t *a, const int32_t *z, unsigned int tid, bool active) {
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
    
/**************************************************
 * 이름: invntt_tomont_coop
 *
 * 설명: 역NTT + Montgomery 보정
 *       레이어 len = 1 → 128 순서로 버터플라이를 계산하고 zeta 는 −z[2·(128/len) − 1 − g] 사용
 *       마지막에 모든 계수에 f = mont²/256 = 41978 곱셈 (스레드당 계수 2개)
 *
 * 인수: - a: 공유메모리 다항식
 *       - z: 공유메모리 zeta 표
 *       - tid: 스레드 번호 0..127
 *       - active: false 면 계산 없이 동기화에만 참여
 **************************************************/
__device__ void invntt_tomont_coop(int32_t *a, const int32_t *z, unsigned int tid, bool active) {
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
