/* Shared verify workspace and arithmetic helpers for parallel ML-DSA. */
#ifndef MLDSA87_VERIFY_MULTI_CUH
#define MLDSA87_VERIFY_MULTI_CUH

#include "parallel_common.cuh"

#define VF_PKB  PQCLEAN_MLDSA87_CLEAN_CRYPTO_PUBLICKEYBYTES
#define VF_SIGB PQCLEAN_MLDSA87_CLEAN_CRYPTO_BYTES
#define VF_MAXM 256

/* GPU 작업 공간 */
typedef struct {
    int cdp_error; // First device-runtime error; host must check after completion.

    uint8_t  pk[VF_PKB];                       /* 입력: 공개키 */
    uint8_t  sig[VF_SIGB];                     /* 입력: 서명 */
    uint8_t  msg[VF_MAXM];                     /* 입력: 메시지 */
    int      mlen;
    uint8_t  mu[CRHBYTES];
    polyvecl mat[K];
    polyvecl z, zhat;
    polyveck t1hat, h;
    poly     cp, cphat;
    uint8_t  w1pack[K * POLYW1_PACKEDBYTES];
    int      bad;                              /* 형식 오류·노름 초과 수 */
    int      result;                           /* 0 = 통과, -1 = 거부 */
} vfm_ws;

/* unpack_sig 의 힌트 계산 */
__device__ int vfm_decode_hint(const uint8_t *hb, polyveck *h);

#endif
