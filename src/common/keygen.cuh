/* Shared keygen workspace and arithmetic helpers for parallel ML-DSA. */
#ifndef MLDSA87_KEYGEN_MULTI_CUH
#define MLDSA87_KEYGEN_MULTI_CUH

#include "parallel_common.cuh"

#define KGM_PKB PQCLEAN_MLDSA87_CLEAN_CRYPTO_PUBLICKEYBYTES
#define KGM_SKB PQCLEAN_MLDSA87_CLEAN_CRYPTO_SECRETKEYBYTES

#define KGM_SK_TR  (2 * SEEDBYTES)
#define KGM_SK_ETA (2 * SEEDBYTES + TRBYTES)
#define KGM_SK_T0  (KGM_SK_ETA + (L + K) * POLYETA_PACKEDBYTES)

/* GPU 작업 공간 */
typedef struct {
    int cdp_error; // First device-runtime error; host must check after completion.

    uint8_t  seed[SEEDBYTES];                        /* 입력 */
    uint8_t  rho[SEEDBYTES], rhoprime[CRHBYTES], key[SEEDBYTES];
    uint8_t  pk[KGM_PKB], sk[KGM_SKB];               /* 출력 */
    polyvecl mat[K];                                 /* A_hat (NTT 영역) */
    polyvecl s1, s1hat;
    polyveck s2;
} kgm_ws;

__device__ void kgm_finish8(kgm_ws *w, unsigned int k, unsigned int g, const int32_t *a);

#endif
