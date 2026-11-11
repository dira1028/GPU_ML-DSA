/*
 * single/single_kernels.cuh  ---  단일 스레드 구현의 커널
 */
#ifndef MLDSA87_SINGLE_KERNELS_CUH
#define MLDSA87_SINGLE_KERNELS_CUH

#include "../cuda/single_core.h"

/* KeyGen: seed → pk, sk */
__global__ void single_keygen_kernel(const uint8_t *seed, uint8_t *pk, uint8_t *sk) {
    mldsa87_keygen_core(seed, pk, sk);
}
/* Sign: (m, sk) -> sig.  ws: 전역 메모리 작업 공간 */
__global__ void single_sign_kernel(uint8_t *sig, size_t *siglen, const uint8_t *m, size_t mlen,
                                   const uint8_t *sk, sign_ws *ws) {
    mldsa87_sign_core(sig, siglen, m, mlen, sk, ws);
}
/* Verify: (sig, m, pk) -> *ret (0 통과 / -1 거부) */
__global__ void single_verify_kernel(const uint8_t *sig, const uint8_t *m, size_t mlen,
                                     const uint8_t *pk, verify_ws *ws, int *ret) {
    *ret = mldsa87_verify_core(sig, PQCLEAN_MLDSA87_CLEAN_CRYPTO_BYTES, m, mlen, pk, ws);
}

#endif
