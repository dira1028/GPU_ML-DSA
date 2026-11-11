/*
 * multi/keygen_multi.cuh  ---  다중 스레드 KeyGen
 *
 * [입력/출력]  작업 공간 kgm_ws 의 seed(32바이트) → pk(2592), sk(4896)
 * [실행]      keygen_multi_kernel<<<nb, 256>>>
 */
#ifndef MLDSA87_KEYGEN_MULTI_CUH
#define MLDSA87_KEYGEN_MULTI_CUH

#include "../kernels/multi_common.cuh"

#define KGM_PKB PQCLEAN_MLDSA87_CLEAN_CRYPTO_PUBLICKEYBYTES
#define KGM_SKB PQCLEAN_MLDSA87_CLEAN_CRYPTO_SECRETKEYBYTES

#define KGM_SK_TR  (2 * SEEDBYTES)
#define KGM_SK_ETA (2 * SEEDBYTES + TRBYTES)
#define KGM_SK_T0  (KGM_SK_ETA + (L + K) * POLYETA_PACKEDBYTES)

/* 단계별 시간 기록 칸 (블록 0 스레드 0 이 각 단계 끝의 grid.sync 직후 기록, ns) */
enum { KGM_P0 = 0, KGM_P1, KGM_P2, KGM_P3, KGM_P4, KGM_NPH };
__attribute__((unused)) static const char *KGM_PHASE[KGM_NPH] = {
    "P0 seed expand (SHAKE256 x1)",
    "P1 ExpandA(56) || ExpandS(15)",
    "P2 NTT(s1) x7",
    "P3 rows: A*s1, invNTT, finish",
    "P4 tr = H(pk) (SHAKE256 x1)"
};

/* GPU 작업 공간 */
typedef struct {
    int cdp_error; // First device-runtime error; host must check after completion.

    uint8_t  seed[SEEDBYTES];                        /* 입력 */
    uint8_t  rho[SEEDBYTES], rhoprime[CRHBYTES], key[SEEDBYTES];
    uint8_t  pk[KGM_PKB], sk[KGM_SKB];               /* 출력 */
    polyvecl mat[K];                                 /* A_hat (NTT 영역) */
    polyvecl s1, s1hat;
    polyveck s2;
    unsigned long long ph[KGM_NPH];                  /* 단계별 시간(ns) */
} kgm_ws;

#if !defined(MLDSA_CDP) || !MLDSA_CDP
__device__ void kgm_seedexpand(kgm_ws *w) {
    uint8_t buf[2 * SEEDBYTES + CRHBYTES];
    for (int i = 0; i < SEEDBYTES; ++i) { buf[i] = w->seed[i]; }
    buf[SEEDBYTES + 0] = K; buf[SEEDBYTES + 1] = L;
    shake256(buf, 2 * SEEDBYTES + CRHBYTES, buf, SEEDBYTES + 2);
    for (int i = 0; i < SEEDBYTES; ++i) { w->rho[i] = buf[i]; }
    for (int i = 0; i < CRHBYTES;  ++i) { w->rhoprime[i] = buf[SEEDBYTES + i]; }
    for (int i = 0; i < SEEDBYTES; ++i) { w->key[i] = buf[SEEDBYTES + CRHBYTES + i]; }
}

#endif

__device__ __forceinline__ void kgm_finish8(kgm_ws *w, unsigned int k, unsigned int g, const int32_t *a) {
    int32_t v1[8], v0[8];
    const int32_t *s2 = w->s2.vec[k].coeffs + 8 * g;
    for (int j = 0; j < 8; ++j) {
        int32_t x = PQCLEAN_MLDSA87_CLEAN_caddq(a[8 * g + j] + s2[j]);
        v1[j] = PQCLEAN_MLDSA87_CLEAN_power2round(&v0[j], x);
    }
    uint8_t *r1 = w->pk + SEEDBYTES + k * POLYT1_PACKEDBYTES + 10 * g;
    for (int h = 0; h < 2; ++h) {
        const int32_t *c = v1 + 4 * h; uint8_t *r = r1 + 5 * h;
        r[0] = (uint8_t) (c[0] >> 0);
        r[1] = (uint8_t) ((c[0] >> 8) | (c[1] << 2));
        r[2] = (uint8_t) ((c[1] >> 6) | (c[2] << 4));
        r[3] = (uint8_t) ((c[2] >> 4) | (c[3] << 6));
        r[4] = (uint8_t) (c[3] >> 2);
    }
    uint32_t t[8];
    for (int j = 0; j < 8; ++j) { t[j] = (1 << (D - 1)) - v0[j]; }
    uint8_t *r = w->sk + KGM_SK_T0 + k * POLYT0_PACKEDBYTES + 13 * g;
    r[0]  =  (uint8_t) t[0];
    r[1]  =  (uint8_t) (t[0] >>  8);
    r[1] |=  (uint8_t) (t[1] <<  5);
    r[2]  =  (uint8_t) (t[1] >>  3);
    r[3]  =  (uint8_t) (t[1] >> 11);
    r[3] |=  (uint8_t) (t[2] <<  2);
    r[4]  =  (uint8_t) (t[2] >>  6);
    r[4] |=  (uint8_t) (t[3] <<  7);
    r[5]  =  (uint8_t) (t[3] >>  1);
    r[6]  =  (uint8_t) (t[3] >>  9);
    r[6] |=  (uint8_t) (t[4] <<  4);
    r[7]  =  (uint8_t) (t[4] >>  4);
    r[8]  =  (uint8_t) (t[4] >> 12);
    r[8] |=  (uint8_t) (t[5] <<  1);
    r[9]  =  (uint8_t) (t[5] >>  7);
    r[9] |=  (uint8_t) (t[6] <<  6);
    r[10] =  (uint8_t) (t[6] >>  2);
    r[11] =  (uint8_t) (t[6] >> 10);
    r[11] |= (uint8_t) (t[7] <<  3);
    r[12] =  (uint8_t) (t[7] >>  5);
}

#if !defined(MLDSA_CDP) || !MLDSA_CDP
__global__ void __launch_bounds__(MT_THREADS) keygen_multi_kernel(kgm_ws *w) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    KSHARED(int32_t, s_z, [N]);
    KSHARED(int32_t, s_a, [MT_GROUPS * N]);
    const unsigned int t = threadIdx.x, b = blockIdx.x, nb = gridDim.x;
    const unsigned int grp = t / 128, lt = t % 128;    /* 묶음 번호, 묶음 안 스레드 번호 */
    int32_t *a = s_a + grp * N;
    const bool rec = (b == 0 && t == 0);               /* 시간 기록 */
    unsigned long long tm = 0;
#define KGM_MARK(slot) do { if (rec) { unsigned long long n_ = mt_gtimer(); w->ph[slot] = n_ - tm; tm = n_; } } while (0)

    if (rec) { tm = mt_gtimer(); }
    zetas_to_shared(s_z, t);

    /* ---- P0: seed 확장 (호출 1개 → 스레드 1개) ---- */
    if (rec) { kgm_seedexpand(w); }
    grid.sync();
    KGM_MARK(KGM_P0);

    /* ---- P1: ExpandA 56개 || ExpandS 15개 ---- */
    if (t < 128) {                                     /* 앞쪽 128스레드: ExpandA. 원소 i -> 블록 i mod nb */
        for (unsigned int i = b + nb * t; i < (unsigned)(K * L); i += nb * 128) {
            const unsigned int r = i / L, c = i % L;
            PQCLEAN_MLDSA87_CLEAN_poly_uniform(&w->mat[r].vec[c], w->rho, (uint16_t)((r << 8) + c));
        }
    } else {                                           /* 뒤쪽 128스레드: ExpandS (s1: 0..6, s2: 7..14) */
        for (unsigned int j = b + nb * (t - 128); j < (unsigned)(L + K); j += nb * 128) {
            poly *p = (j < (unsigned)L) ? &w->s1.vec[j] : &w->s2.vec[j - L];
            PQCLEAN_MLDSA87_CLEAN_poly_uniform_eta(p, w->rhoprime, (uint16_t)j);
            PQCLEAN_MLDSA87_CLEAN_polyeta_pack(w->sk + KGM_SK_ETA + j * POLYETA_PACKEDBYTES, p);
        }
    }
    if (b == 0) {                                      /* pk, sk 앞부분: ρ, K */
        for (unsigned int i = t; i < (unsigned)SEEDBYTES; i += blockDim.x) {
            w->pk[i] = w->rho[i]; w->sk[i] = w->rho[i]; w->sk[SEEDBYTES + i] = w->key[i];
        }
    }
    grid.sync();
    KGM_MARK(KGM_P1);

    /* ---- P2: s_hat1 = NTT(s1) ---- */
    MT_FOR_TASKS(L, l, act) {
        if (act) { a[lt] = w->s1.vec[l].coeffs[lt]; a[lt + 128] = w->s1.vec[l].coeffs[lt + 128]; }
        __syncthreads();
        ntt_coop(a, s_z, lt, act);
        if (act) { w->s1hat.vec[l].coeffs[lt] = a[lt]; w->s1hat.vec[l].coeffs[lt + 128] = a[lt + 128]; }
        __syncthreads();
    }
    grid.sync();
    KGM_MARK(KGM_P2);

    /* ---- P3: 행 k = 0..7 을 배분, 행마다 matvec -> reduce -> NTT_Inverse ---- */
    MT_FOR_TASKS(K, k, act) {
        if (act) {
            for (unsigned int c = lt; c < (unsigned)N; c += 128) {
                int32_t acc = MT_MR(w->mat[k].vec[0].coeffs[c], w->s1hat.vec[0].coeffs[c]);
                for (int l = 1; l < L; ++l) { acc += MT_MR(w->mat[k].vec[l].coeffs[c], w->s1hat.vec[l].coeffs[c]); }
                a[c] = PQCLEAN_MLDSA87_CLEAN_reduce32(acc);
            }
        }
        __syncthreads();
        invntt_tomont_coop(a, s_z, lt, act);
        if (act && lt < (unsigned)(N / 8)) { kgm_finish8(w, k, lt, a); }   /* 32스레드 × 8계수 */
        __syncthreads();
    }
    grid.sync();
    KGM_MARK(KGM_P3);

    /* ---- P4: tr = H(pk) -> sk ---- */
    if (rec) {
        shake256(w->sk + KGM_SK_TR, TRBYTES, w->pk, KGM_PKB);
        KGM_MARK(KGM_P4);
    }
#undef KGM_MARK
}

#endif

#endif
