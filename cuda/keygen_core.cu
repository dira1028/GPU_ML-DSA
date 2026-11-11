/*
 * cuda/keygen_core.cu  ---  단일 스레드 KeyGen (GPU 장치 함수)
 *
 * 원본 ref/sign.c 의 PQCLEAN_MLDSA87_CLEAN_crypto_sign_keypair 본문을 그대로 옮겼다. 원본과 다른 점은 cuda/single_core.h 의 R1 뿐이며,
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

__device__ void mldsa87_keygen_core(const uint8_t seed_in[SEEDBYTES],
                                    uint8_t *pk, uint8_t *sk) {
    uint8_t seedbuf[2 * SEEDBYTES + CRHBYTES];
    uint8_t tr[TRBYTES];
    const uint8_t *rho, *rhoprime, *key;
    polyvecl mat[K];
    polyvecl s1, s1hat;
    polyveck s2, t1, t0;

    /* [R1] 원본: randombytes(seedbuf, SEEDBYTES).  장치에서는 시드를 인자로 받아 복사 */
    for (int i = 0; i < SEEDBYTES; ++i) {
        seedbuf[i] = seed_in[i];
    }
    /* --- 이하 원본 crypto_sign_keypair 와 동일 --- */
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
