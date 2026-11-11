/*
 * test/make_vectors.c  ---  CPU 원본(PQClean ML-DSA-87 clean, ref/)으로 정답 벡터 파일 생성
 *
 *   ./make_vectors <출력파일> [시드 수=20] [키 수=3] [메시지 수=100]
 *
 * - KeyGen : 시드 s 마다 crypto_sign_keypair -> pk, sk
 * - Sign   : 키 k(= 시드 k 의 키)로 메시지 r 마다 crypto_sign_signature (결정론적: rnd = 0)
 * - Verify : 키 k, 메시지 r 의 서명에 test/verify_cases.h 의 케이스 c 를 적용해 crypto_sign_verify 결과 기록
 */
#include "params.h"
#include "sign.h"
#include "../test/verify_cases.h"
#include "../test/vectors.h"
#include <stdio.h>
#include <stdlib.h>

static int g_seed = -1;                       /* >= 0 이면 다음 randombytes 호출에 시드 g_seed 를 돌려줌 */
int PQCLEAN_randombytes(uint8_t *out, size_t n) {
    uint8_t s[32];
    if (g_seed >= 0) { vec_seed(g_seed, s); }
    for (size_t i = 0; i < n; i++) { out[i] = g_seed >= 0 ? s[i % 32] : 0; }
    g_seed = -1;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "사용: %s <출력파일> [시드 수] [키 수] [메시지 수]\n", argv[0]); return 2; }
    vec_set v;
    v.nseed = argc > 2 ? atoi(argv[2]) : 20;
    v.nkey  = argc > 3 ? atoi(argv[3]) : 3;
    v.nmsg  = argc > 4 ? atoi(argv[4]) : 100;
    v.mlen  = VC_MLEN;
    v.ncase = VC_NCASE;
    if (v.nkey > v.nseed) { v.nseed = v.nkey; }
    vec_alloc(&v);

    for (int s = 0; s < v.nseed; s++) {                                  /* KeyGen */
        g_seed = s;
        PQCLEAN_MLDSA87_CLEAN_crypto_sign_keypair(vec_pk(&v, s), vec_sk(&v, s));
    }
    uint8_t m[VC_MLEN];
    for (int k = 0; k < v.nkey; k++) {                                   /* Sign */
        for (int r = 0; r < v.nmsg; r++) {
            size_t siglen = 0;
            vc_message(m, r);
            g_seed = -1;
            PQCLEAN_MLDSA87_CLEAN_crypto_sign_signature(vec_sig(&v, k, r), &siglen, m, VC_MLEN, vec_sk(&v, k));
        }
    }
    static uint8_t pk[VEC_PKB], sig[VEC_SIGB];
    long npass = 0;
    for (int k = 0; k < v.nkey; k++) {                                   /* Verify */
        for (int r = 0; r < v.nmsg; r++) {
            for (int c = 0; c < v.ncase; c++) {
                vc_apply(c, r, vec_pk(&v, k), vec_sig(&v, k, r), pk, m, sig, VEC_PKB, VEC_SIGB);
                int ret = PQCLEAN_MLDSA87_CLEAN_crypto_sign_verify(sig, VEC_SIGB, m, VC_MLEN, pk);
                v.vres[((size_t)k * v.nmsg + r) * v.ncase + c] = (int8_t)ret;
                npass += (ret == 0);
            }
        }
    }
    if (vec_write(argv[1], &v)) { perror(argv[1]); return 1; }
    fprintf(stderr, "[make_vectors] %s: KeyGen 시드 %d개, Sign 키 %d개 x 메시지 %d개, Verify 판정 %d건(통과 %ld건)\n",
            argv[1], v.nseed, v.nkey, v.nmsg, v.nkey * v.nmsg * v.ncase, npass);
    return 0;
}
