/*
 * cuda/sign_core.cu  ---  단일 스레드 Sign (GPU 장치 함수)
 *
 * 원본 ref/sign.c 의 PQCLEAN_MLDSA87_CLEAN_crypto_sign_signature_ctx 본문을 그대로 옮겼다. 원본과 다른 점은 cuda/single_core.h 의 R1, R2, R3 뿐이며,
 * 해당 줄에 [R1]/[R2]/[R3] 표시를 달았다. 호출하는 함수들은 cuda/ 의 다른 .cu 이식본(원본과 같은 로직).
 * 이 함수를 GPU 스레드 1개가 실행하면 "단일 스레드 구현"이 된다 (single/single_kernels.cuh).
 */
#include "fips202.h"
#include "packing.h"
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "symmetric.h"
#include "single_core.h"
#include <stdint.h>

__device__ void mldsa87_sign_core(uint8_t *sig, size_t *siglen,
                                  const uint8_t *m, size_t mlen,
                                  const uint8_t *sk, sign_ws *ws) {
    unsigned int n;
    uint8_t seedbuf[2 * SEEDBYTES + TRBYTES + RNDBYTES + 2 * CRHBYTES];
    uint8_t *rho, *tr, *key, *mu, *rhoprime, *rnd;
    uint16_t nonce = 0;
    shake256incctx state;
    const uint8_t *ctx = nullptr; size_t ctxlen = 0;          /* [R3] 문맥 문자열 없음 */

    rho = seedbuf; tr = rho + SEEDBYTES; key = tr + TRBYTES;
    rnd = key + SEEDBYTES; mu = rnd + RNDBYTES; rhoprime = mu + CRHBYTES;
    /* [R2] 원본의 지역 변수 mat, s1, y, z, t0, s2, w1, w0, h, cp 는 ws->... (전역 메모리) */
    PQCLEAN_MLDSA87_CLEAN_unpack_sk(rho, tr, key, &ws->t0, &ws->s1, &ws->s2, sk);

    mu[0] = 0; mu[1] = (uint8_t)ctxlen;
    shake256_inc_init(&state);
    shake256_inc_absorb(&state, tr, TRBYTES);
    shake256_inc_absorb(&state, mu, 2);
    shake256_inc_absorb(&state, ctx, ctxlen);
    shake256_inc_absorb(&state, m, mlen);
    shake256_inc_finalize(&state);
    shake256_inc_squeeze(mu, CRHBYTES, &state);
    shake256_inc_ctx_release(&state);

    for (int i = 0; i < RNDBYTES; ++i) { rnd[i] = 0; }        /* [R1] 원본: randombytes(rnd, RNDBYTES) → 0 (결정론적) */
    shake256(rhoprime, CRHBYTES, key, SEEDBYTES + RNDBYTES + CRHBYTES);

    PQCLEAN_MLDSA87_CLEAN_polyvec_matrix_expand(ws->mat, rho);
    PQCLEAN_MLDSA87_CLEAN_polyvecl_ntt(&ws->s1);
    PQCLEAN_MLDSA87_CLEAN_polyveck_ntt(&ws->s2);
    PQCLEAN_MLDSA87_CLEAN_polyveck_ntt(&ws->t0);

    while (1) {                                               /* 원본의 rej: 레이블 + goto rej 와 같은 반복 */
        PQCLEAN_MLDSA87_CLEAN_polyvecl_uniform_gamma1(&ws->y, rhoprime, nonce++);
        ws->z = ws->y;
        PQCLEAN_MLDSA87_CLEAN_polyvecl_ntt(&ws->z);
        PQCLEAN_MLDSA87_CLEAN_polyvec_matrix_pointwise_montgomery(&ws->w1, ws->mat, &ws->z);
        PQCLEAN_MLDSA87_CLEAN_polyveck_reduce(&ws->w1);
        PQCLEAN_MLDSA87_CLEAN_polyveck_invntt_tomont(&ws->w1);

        PQCLEAN_MLDSA87_CLEAN_polyveck_caddq(&ws->w1);
        PQCLEAN_MLDSA87_CLEAN_polyveck_decompose(&ws->w1, &ws->w0, &ws->w1);
        PQCLEAN_MLDSA87_CLEAN_polyveck_pack_w1(sig, &ws->w1);

        shake256_inc_init(&state);
        shake256_inc_absorb(&state, mu, CRHBYTES);
        shake256_inc_absorb(&state, sig, K * POLYW1_PACKEDBYTES);
        shake256_inc_finalize(&state);
        shake256_inc_squeeze(sig, CTILDEBYTES, &state);
        shake256_inc_ctx_release(&state);
        PQCLEAN_MLDSA87_CLEAN_poly_challenge(&ws->cp, sig);
        PQCLEAN_MLDSA87_CLEAN_poly_ntt(&ws->cp);

        PQCLEAN_MLDSA87_CLEAN_polyvecl_pointwise_poly_montgomery(&ws->z, &ws->cp, &ws->s1);
        PQCLEAN_MLDSA87_CLEAN_polyvecl_invntt_tomont(&ws->z);
        PQCLEAN_MLDSA87_CLEAN_polyvecl_add(&ws->z, &ws->z, &ws->y);
        PQCLEAN_MLDSA87_CLEAN_polyvecl_reduce(&ws->z);
        if (PQCLEAN_MLDSA87_CLEAN_polyvecl_chknorm(&ws->z, GAMMA1 - BETA)) { continue; }

        PQCLEAN_MLDSA87_CLEAN_polyveck_pointwise_poly_montgomery(&ws->h, &ws->cp, &ws->s2);
        PQCLEAN_MLDSA87_CLEAN_polyveck_invntt_tomont(&ws->h);
        PQCLEAN_MLDSA87_CLEAN_polyveck_sub(&ws->w0, &ws->w0, &ws->h);
        PQCLEAN_MLDSA87_CLEAN_polyveck_reduce(&ws->w0);
        if (PQCLEAN_MLDSA87_CLEAN_polyveck_chknorm(&ws->w0, GAMMA2 - BETA)) { continue; }

        PQCLEAN_MLDSA87_CLEAN_polyveck_pointwise_poly_montgomery(&ws->h, &ws->cp, &ws->t0);
        PQCLEAN_MLDSA87_CLEAN_polyveck_invntt_tomont(&ws->h);
        PQCLEAN_MLDSA87_CLEAN_polyveck_reduce(&ws->h);
        if (PQCLEAN_MLDSA87_CLEAN_polyveck_chknorm(&ws->h, GAMMA2)) { continue; }

        PQCLEAN_MLDSA87_CLEAN_polyveck_add(&ws->w0, &ws->w0, &ws->h);
        n = PQCLEAN_MLDSA87_CLEAN_polyveck_make_hint(&ws->h, &ws->w0, &ws->w1);
        if (n > OMEGA) { continue; }
        break;
    }

    PQCLEAN_MLDSA87_CLEAN_pack_sig(sig, sig, &ws->z, &ws->h);
    *siglen = PQCLEAN_MLDSA87_CLEAN_CRYPTO_BYTES;
}
