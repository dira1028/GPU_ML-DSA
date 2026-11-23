/*
 * src/single/verify_core.cu  ---  단일 스레드 Verify (GPU 장치 함수)
 */
#include "fips202.h"
#include "packing.h"
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "single_core.h"
#include <stdint.h>

/**************************************************
 * 이름: mldsa87_verify_core
 *
 * 설명: 단일 스레드 Verify
 *
 * 인수: - sig: 서명
 *      - siglen: 서명 길이
 *      - m: 메시지
 *      - mlen: 메시지 길이
 *      - pk: 공개키
 *      - ws: 전역 메모리 작업 공간 (행렬 A, z, t1 등)
 *
 * 통과면 0, 거부면 −1 을 반환
 **************************************************/
__device__ int mldsa87_verify_core(const uint8_t *sig, size_t siglen,
                                   const uint8_t *m, size_t mlen,
                                   const uint8_t *pk, verify_ws *ws) {
    unsigned int i;
    uint8_t buf[K * POLYW1_PACKEDBYTES];
    uint8_t rho[SEEDBYTES];
    uint8_t mu[CRHBYTES];
    uint8_t c[CTILDEBYTES];
    uint8_t c2[CTILDEBYTES];
    shake256incctx state;
    const uint8_t *ctx = 0;
    size_t ctxlen = 0;

    if (ctxlen > 255 || siglen != PQCLEAN_MLDSA87_CLEAN_CRYPTO_BYTES) {
        return -1;
    }

    PQCLEAN_MLDSA87_CLEAN_unpack_pk(rho, &ws->t1, pk);
    if (PQCLEAN_MLDSA87_CLEAN_unpack_sig(c, &ws->z, &ws->h, sig)) {
        return -1;
    }
    if (PQCLEAN_MLDSA87_CLEAN_polyvecl_chknorm(&ws->z, GAMMA1 - BETA)) {
        return -1;
    }

    /* mu = CRH(H(pk), 0, ctxlen, ctx, msg) */
    shake256(mu, TRBYTES, pk, PQCLEAN_MLDSA87_CLEAN_CRYPTO_PUBLICKEYBYTES);
    shake256_inc_init(&state);
    shake256_inc_absorb(&state, mu, TRBYTES);
    mu[0] = 0;
    mu[1] = (uint8_t)ctxlen;
    shake256_inc_absorb(&state, mu, 2);
    shake256_inc_absorb(&state, ctx, ctxlen);
    shake256_inc_absorb(&state, m, mlen);
    shake256_inc_finalize(&state);
    shake256_inc_squeeze(mu, CRHBYTES, &state);
    shake256_inc_ctx_release(&state);

    /* Az - c*2^d*t1 */
    PQCLEAN_MLDSA87_CLEAN_poly_challenge(&ws->cp, c);
    PQCLEAN_MLDSA87_CLEAN_polyvec_matrix_expand(ws->mat, rho);

    PQCLEAN_MLDSA87_CLEAN_polyvecl_ntt(&ws->z);
    PQCLEAN_MLDSA87_CLEAN_polyvec_matrix_pointwise_montgomery(&ws->w1, ws->mat, &ws->z);

    PQCLEAN_MLDSA87_CLEAN_poly_ntt(&ws->cp);
    PQCLEAN_MLDSA87_CLEAN_polyveck_shiftl(&ws->t1);
    PQCLEAN_MLDSA87_CLEAN_polyveck_ntt(&ws->t1);
    PQCLEAN_MLDSA87_CLEAN_polyveck_pointwise_poly_montgomery(&ws->t1, &ws->cp, &ws->t1);

    PQCLEAN_MLDSA87_CLEAN_polyveck_sub(&ws->w1, &ws->w1, &ws->t1);
    PQCLEAN_MLDSA87_CLEAN_polyveck_reduce(&ws->w1);
    PQCLEAN_MLDSA87_CLEAN_polyveck_invntt_tomont(&ws->w1);

    /* w1 복원 */
    PQCLEAN_MLDSA87_CLEAN_polyveck_caddq(&ws->w1);
    PQCLEAN_MLDSA87_CLEAN_polyveck_use_hint(&ws->w1, &ws->w1, &ws->h);
    PQCLEAN_MLDSA87_CLEAN_polyveck_pack_w1(buf, &ws->w1);

    /* c2 = H(mu || w1) 후 c 와 비교 */
    shake256_inc_init(&state);
    shake256_inc_absorb(&state, mu, CRHBYTES);
    shake256_inc_absorb(&state, buf, K * POLYW1_PACKEDBYTES);
    shake256_inc_finalize(&state);
    shake256_inc_squeeze(c2, CTILDEBYTES, &state);
    shake256_inc_ctx_release(&state);
    for (i = 0; i < CTILDEBYTES; ++i) {
        if (c[i] != c2[i]) {
            return -1;
        }
    }
    return 0;
}
