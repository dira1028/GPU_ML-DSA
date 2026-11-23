/*
 * src/single/single_core.h  ---  단일 스레드 GPU 구현의 공용 선언
 */
#ifndef MLDSA87_SINGLE_CORE_H
#define MLDSA87_SINGLE_CORE_H

#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include <stddef.h>
#include <stdint.h>

/* Sign 작업 공간 */
typedef struct {
    polyvecl mat[K];
    polyvecl s1, y, z;
    polyveck t0, s2, w1, w0, h;
    poly cp;
} sign_ws;

/* Verify 작업 공간 */
typedef struct {
    polyvecl mat[K];
    polyvecl z;
    polyveck t1, w1, h;
    poly cp;
} verify_ws;

__device__ void mldsa87_keygen_core(const uint8_t seed_in[SEEDBYTES], uint8_t *pk, uint8_t *sk);

__device__ void mldsa87_sign_core(uint8_t *sig, size_t *siglen, const uint8_t *m, size_t mlen,
                                  const uint8_t *sk, sign_ws *ws);
                                  
__device__ int mldsa87_verify_core(const uint8_t *sig, size_t siglen, const uint8_t *m, size_t mlen,
                                   const uint8_t *pk, verify_ws *ws);

#endif
