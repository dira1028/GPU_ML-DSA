/*
 * src/single/keygen_core.cu  ---  단일 스레드 KeyGen (GPU 장치 함수)
 */
#include "fips202.h"
#include "packing.h"
#include "params.h"
#include "polyvec.h"
#include "single_core.h"
#include <stdint.h>

/**************************************************
 * 이름: mldsa87_keygen_core
 *
 * 설명: 단일 스레드 KeyGen. 원본 crypto_sign_keypair 본문을 그대로 옮기고, 난수 대신 시드를 인자로 입력 (R1)
 *       GPU 스레드 1개가 시드 확장, ExpandA, ExpandS, NTT, 행렬 곱, power2round, 패킹, H(pk) 를 순서대로 실행
 *
 * 인수: - seed_in: 32바이트 시드
 *      - pk: 출력 공개키 (2592바이트)
 *      - sk: 출력 비밀키 (4896바이트)
 **************************************************/
__device__ void mldsa87_keygen_core(const uint8_t seed_in[SEEDBYTES],
                                    uint8_t *pk, uint8_t *sk) {
    uint8_t seedbuf[2 * SEEDBYTES + CRHBYTES];
    uint8_t tr[TRBYTES];
    const uint8_t *rho, *rhoprime, *key;
    polyvecl mat[K];
    polyvecl s1, s1hat;
    polyveck s2, t1, t0;

    for (int i = 0; i < SEEDBYTES; ++i) {
        seedbuf[i] = seed_in[i];
    }
    /* --- 이하 crypto_sign_keypair 와 동일 --- */
    seedbuf[SEEDBYTES + 0] = K;
    seedbuf[SEEDBYTES + 1] = L;
    shake256(seedbuf, 2 * SEEDBYTES + CRHBYTES, seedbuf, SEEDBYTES + 2);
    rho = seedbuf;
    rhoprime = rho + SEEDBYTES;
    key = rhoprime + CRHBYTES;

    /* Expand matrix */
    PQCLEAN_MLDSA87_CLEAN_polyvec_matrix_expand(mat, rho);

    /* Sample short vectors s1 and s2 */
    PQCLEAN_MLDSA87_CLEAN_polyvecl_uniform_eta(&s1, rhoprime, 0);
    PQCLEAN_MLDSA87_CLEAN_polyveck_uniform_eta(&s2, rhoprime, L);

    /* Matrix-vector multiplication */
    s1hat = s1;
    PQCLEAN_MLDSA87_CLEAN_polyvecl_ntt(&s1hat);
    PQCLEAN_MLDSA87_CLEAN_polyvec_matrix_pointwise_montgomery(&t1, mat, &s1hat);
    PQCLEAN_MLDSA87_CLEAN_polyveck_reduce(&t1);
    PQCLEAN_MLDSA87_CLEAN_polyveck_invntt_tomont(&t1);

    /* Add error vector s2 */
    PQCLEAN_MLDSA87_CLEAN_polyveck_add(&t1, &t1, &s2);

    /* Extract t1 and write public key */
    PQCLEAN_MLDSA87_CLEAN_polyveck_caddq(&t1);
    PQCLEAN_MLDSA87_CLEAN_polyveck_power2round(&t1, &t0, &t1);
    PQCLEAN_MLDSA87_CLEAN_pack_pk(pk, rho, &t1);

    /* Compute H(rho, t1) and write secret key */
    shake256(tr, TRBYTES, pk, PQCLEAN_MLDSA87_CLEAN_CRYPTO_PUBLICKEYBYTES);
    PQCLEAN_MLDSA87_CLEAN_pack_sk(sk, rho, tr, key, &t0, &s1, &s2);
}
