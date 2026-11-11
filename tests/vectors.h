/*
 * test/vectors.h  ---  기준 벡터 파일 형식
 *
 * test/make_vectors.c 가 PQClean 원본(ref/)으로 만들고, bench/mldsa87_gpu.cu 가 읽어 GPU 결과와 비교한다.
 *
 *   헤더   : "MLDV" | nseed | nkey | nmsg | mlen | ncase     (int32 × 5)
 *   KeyGen : 시드 s = 0..nseed-1 마다  pk | sk                 (시드 s 의 값은 vec_seed())
 *   Sign   : 키 k = 0..nkey-1 (= 시드 k 의 키), 메시지 r 마다  sig     (결정론적 서명, rnd = 0)
 *   Verify : 키 k, 메시지 r, 케이스 c (test/verify_cases.h) 마다 원본 결과 (int8: 0 / -1)
 */
#ifndef MLDSA87_VECTORS_H
#define MLDSA87_VECTORS_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define VEC_PKB  2592
#define VEC_SKB  4896
#define VEC_SIGB 4627

/* 시드 s: s = 0 이면 0x00, 0x01, …, 0x1f */
static inline void vec_seed(int s, uint8_t seed[32]) {
    for (int i = 0; i < 32; i++) { seed[i] = (uint8_t)(i + s * (2 * i + 1) * 13); }
}

typedef struct {
    int nseed, nkey, nmsg, mlen, ncase;
    uint8_t *pk, *sk;        /* [nseed][PKB], [nseed][SKB] */
    uint8_t *sig;            /* [nkey][nmsg][SIGB] */
    int8_t  *vres;           /* [nkey][nmsg][ncase] */
} vec_set;

static inline uint8_t *vec_pk(const vec_set *v, int s)  { return v->pk + (size_t)s * VEC_PKB; }
static inline uint8_t *vec_sk(const vec_set *v, int s)  { return v->sk + (size_t)s * VEC_SKB; }
static inline uint8_t *vec_sig(const vec_set *v, int k, int r) { return v->sig + ((size_t)k * v->nmsg + r) * VEC_SIGB; }
static inline int vec_vres(const vec_set *v, int k, int r, int c) { return v->vres[((size_t)k * v->nmsg + r) * v->ncase + c]; }

static inline void vec_alloc(vec_set *v) {
    v->pk = (uint8_t *)malloc((size_t)v->nseed * VEC_PKB);
    v->sk = (uint8_t *)malloc((size_t)v->nseed * VEC_SKB);
    v->sig = (uint8_t *)malloc((size_t)v->nkey * v->nmsg * VEC_SIGB);
    v->vres = (int8_t *)malloc((size_t)v->nkey * v->nmsg * v->ncase);
}

static inline int vec_write(const char *path, const vec_set *v) {
    FILE *f = fopen(path, "wb");
    if (!f) { return 1; }
    int hdr[5] = { v->nseed, v->nkey, v->nmsg, v->mlen, v->ncase };
    fwrite("MLDV", 1, 4, f);
    fwrite(hdr, sizeof(int), 5, f);
    fwrite(v->pk, 1, (size_t)v->nseed * VEC_PKB, f);
    fwrite(v->sk, 1, (size_t)v->nseed * VEC_SKB, f);
    fwrite(v->sig, 1, (size_t)v->nkey * v->nmsg * VEC_SIGB, f);
    fwrite(v->vres, 1, (size_t)v->nkey * v->nmsg * v->ncase, f);
    return fclose(f) != 0;
}

static inline int vec_read(const char *path, vec_set *v) {
    FILE *f = fopen(path, "rb");
    char magic[4];
    int hdr[5];
    if (!f) { return 1; }
    if (fread(magic, 1, 4, f) != 4 || magic[0] != 'M' || magic[1] != 'L' || magic[2] != 'D' || magic[3] != 'V' ||
        fread(hdr, sizeof(int), 5, f) != 5) { fclose(f); return 1; }
    v->nseed = hdr[0]; v->nkey = hdr[1]; v->nmsg = hdr[2]; v->mlen = hdr[3]; v->ncase = hdr[4];
    if (v->nkey > v->nseed) { fclose(f); return 1; }
    vec_alloc(v);
    size_t a = (size_t)v->nseed * VEC_PKB, b = (size_t)v->nseed * VEC_SKB;
    size_t c = (size_t)v->nkey * v->nmsg * VEC_SIGB, d = (size_t)v->nkey * v->nmsg * v->ncase;
    int bad = fread(v->pk, 1, a, f) != a || fread(v->sk, 1, b, f) != b ||
              fread(v->sig, 1, c, f) != c || fread(v->vres, 1, d, f) != d;
    fclose(f);
    return bad;
}

#endif
