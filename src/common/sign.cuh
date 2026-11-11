/* Shared sign workspace and arithmetic helpers for parallel ML-DSA. */
#ifndef MLDSA87_SIGN_MULTI_CUH
#define MLDSA87_SIGN_MULTI_CUH

#include "parallel_common.cuh"

#define SG_SKB  PQCLEAN_MLDSA87_CLEAN_CRYPTO_SECRETKEYBYTES
#define SG_SIGB PQCLEAN_MLDSA87_CLEAN_CRYPTO_BYTES
#define SG_MAXM 256                              /* 메시지 최대 길이*/

/* GPU 작업 공간 */
typedef struct {
    int cdp_error; // First device-runtime error; host must check after completion.
    uint16_t cdp_nonce;

    uint8_t  sk[SG_SKB];                       /* 입력: 비밀키 */
    uint8_t  msg[SG_MAXM];                     /* 입력: 메시지 */
    int      mlen;
    uint8_t  mu[CRHBYTES], rhoprime[CRHBYTES];
    polyvecl mat[K];
    polyvecl s1hat, y, yhat, z;
    polyveck s2hat, t0hat, w0, w1, h;
    poly     cp, cphat;
    uint8_t  w1pack[K * POLYW1_PACKEDBYTES];
    uint8_t  sig[SG_SIGB];                     /* 출력: 서명 */
    int      flag[2];                          /* 반복 판정 */
    unsigned int nh_row[2][K];
    int      iters;                            /* 출력: 반복 횟수 */
} sgm_ws;

__device__ void sgm_unpack8(const uint8_t *sk, unsigned int tau, unsigned int g, int32_t *a);

#endif
