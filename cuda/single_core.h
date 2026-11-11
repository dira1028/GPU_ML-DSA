/*
 * cuda/single_core.h  ---  단일 스레드 GPU 구현의 공용 선언
 *
 * 단일 스레드 구현 = PQClean 의 KeyGen / Sign / Verify 함수 본문을 그대로 GPU 장치 함수로 옮기고,
 * GPU 스레드 1개(<<<1, 1>>>)가 연산 전체를 순서대로 실행하는 구현. 다중 스레드 구현과 비교하는 기준이다.
 *
 * 원본(ref/sign.c)과 다른 점 (결과에는 영향 없음)
 *   R1  난수: 장치에서는 OS 난수를 쓸 수 없으므로
 *       - KeyGen 은 randombytes(seed) 대신 32바이트 시드를 인자로 받는다
 *       - Sign 은 rnd 를 0 으로 둔다 (결정론적 서명. 원본에서 randombytes 가 0 을 돌려준 것과 같음)
 *   R2  메모리: Sign·Verify 의 큰 지역 배열(행렬 A 등, 약 90~120KB)을 스레드 스택에 두면 GPU 스택이
 *       모자라 실패하므로 전역 메모리 작업 공간(sign_ws, verify_ws)으로 옮겼다. KeyGen 은 원본대로 지역 배열.
 *   R3  문맥 문자열(ctx)은 쓰지 않는다 (ctx = 없음, ctxlen = 0). PQClean 의 crypto_sign_signature /
 *       crypto_sign_verify 가 내부에서 ctx 없이 호출하는 것과 같다.
 */
#ifndef MLDSA87_SINGLE_CORE_H
#define MLDSA87_SINGLE_CORE_H

#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include <stddef.h>
#include <stdint.h>

/* Sign 작업 공간 (R2). 원본 crypto_sign_signature_ctx 의 지역 변수 mat, s1, y, z, t0, s2, w1, w0, h, cp */
typedef struct {
    polyvecl mat[K];
    polyvecl s1, y, z;
    polyveck t0, s2, w1, w0, h;
    poly cp;
} sign_ws;

/* Verify 작업 공간 (R2). 원본 crypto_sign_verify_ctx 의 지역 변수 mat, z, t1, w1, h, cp */
typedef struct {
    polyvecl mat[K];
    polyvecl z;
    polyveck t1, w1, h;
    poly cp;
} verify_ws;

/* KeyGen: seed(32바이트) → pk(2592), sk(4896).  원본 crypto_sign_keypair 와 같은 결과 */
__device__ void mldsa87_keygen_core(const uint8_t seed_in[SEEDBYTES], uint8_t *pk, uint8_t *sk);

/* Sign: 결정론적 서명(rnd = 0). sig(4627), *siglen 에 길이.  원본 crypto_sign_signature 와 같은 결과 */
__device__ void mldsa87_sign_core(uint8_t *sig, size_t *siglen, const uint8_t *m, size_t mlen,
                                  const uint8_t *sk, sign_ws *ws);

/* Verify: 0 = 통과, -1 = 거부.  원본 crypto_sign_verify 와 같은 값 */
__device__ int mldsa87_verify_core(const uint8_t *sig, size_t siglen, const uint8_t *m, size_t mlen,
                                   const uint8_t *pk, verify_ws *ws);

#endif
