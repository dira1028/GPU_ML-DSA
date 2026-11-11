#include "ntt_coop.cuh"


__device__ void zetas_to_shared(int32_t *z, unsigned int tid) {
    for (unsigned int i = tid; i < (unsigned)N; i += blockDim.x) { z[i] = zetas[i]; }
}

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
