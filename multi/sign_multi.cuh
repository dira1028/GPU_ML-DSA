/*
 * multi/sign_multi.cuh  ---  다중 스레드 Sign
 *
 * [입력/출력]  작업 공간 sgm_ws 의 sk(4896), msg(mlen) → sig(4627), iters(반복 횟수).
 * [실행]      sign_multi_kernel<<<nb, 256>>>
*/
#ifndef MLDSA87_SIGN_MULTI_CUH
#define MLDSA87_SIGN_MULTI_CUH

#include "../kernels/multi_common.cuh"

#define SG_SKB  PQCLEAN_MLDSA87_CLEAN_CRYPTO_SECRETKEYBYTES
#define SG_SIGB PQCLEAN_MLDSA87_CLEAN_CRYPTO_BYTES
#define SG_MAXM 256                              /* 메시지 최대 길이*/

/* 단계별 시간 칸 */
enum { SGT_PRE1 = 0, SGT_PRE2, SGT_L1, SGT_L2, SGT_L3, SGT_L4, SGT_PACK, SGT_N };
__attribute__((unused)) static const char *SGT_NAME[SGT_N] = {
    "P1 ExpandA || mu,rho''", "P2 unpack sk + NTT(s1,s2,t0)",
    "L1 ExpandMask + NTT(y) *", "L2 A*y, invNTT, decompose, w1 *",
    "L3 H(mu||w1), challenge, NTT(c) *", "L4 z/w0/t0 check + hint *", "P3 pack sig"
};
__attribute__((unused)) static const int SGT_IN_LOOP[SGT_N] = { 0, 0, 1, 1, 1, 1, 0 };

/* GPU 작업 공간 */
typedef struct {
    int cdp_error; // First device-runtime error; host must check after completion.
    uint16_t cdp_nonce;

    uint8_t  sk[SG_SKB];                       /* 입력: 비밀키 */
    uint8_t  msg[SG_MAXM];                     /* 입력: 메시지 */
    int      mlen;
    uint8_t  mu[CRHBYTES], rhoprime[CRHBYTES];
    polyvecl mat[K];
    polyvecl s1hat, y, yhat, z;
    polyveck s2hat, t0hat, w0, w1, h;
    poly     cp, cphat;
    uint8_t  w1pack[K * POLYW1_PACKEDBYTES];
    uint8_t  sig[SG_SIGB];                     /* 출력: 서명 */
    int      flag[2];                          /* 반복 판정 */
    unsigned int nh_row[2][K];
    int      iters;                            /* 출력: 반복 횟수 */
    unsigned long long ph[SGT_N];              /* 단계별 누적 ns */
} sgm_ws;

__device__ __forceinline__ void sgm_unpack8(const uint8_t *sk, unsigned int tau, unsigned int g, int32_t *a) {
    const uint8_t *base = sk + 2 * SEEDBYTES + TRBYTES;
    int32_t *r = a + 8 * g;
    if (tau < (unsigned)(L + K)) {                                     /* eta (s1, s2) */
        const uint8_t *p = base + tau * POLYETA_PACKEDBYTES + 3 * g;
        r[0] =  (p[0] >> 0) & 7;
        r[1] =  (p[0] >> 3) & 7;
        r[2] = ((p[0] >> 6) | (p[1] << 2)) & 7;
        r[3] =  (p[1] >> 1) & 7;
        r[4] =  (p[1] >> 4) & 7;
        r[5] = ((p[1] >> 7) | (p[2] << 1)) & 7;
        r[6] =  (p[2] >> 2) & 7;
        r[7] =  (p[2] >> 5) & 7;
        for (int j = 0; j < 8; ++j) { r[j] = ETA - r[j]; }
    } else {                                                           /* t0 */
        const uint8_t *p = base + (L + K) * POLYETA_PACKEDBYTES + (tau - L - K) * POLYT0_PACKEDBYTES + 13 * g;
        r[0]  = p[0];       r[0] |= (uint32_t)p[1]  << 8;                                r[0] &= 0x1FFF;
        r[1]  = p[1] >> 5;  r[1] |= (uint32_t)p[2]  << 3; r[1] |= (uint32_t)p[3] << 11;  r[1] &= 0x1FFF;
        r[2]  = p[3] >> 2;  r[2] |= (uint32_t)p[4]  << 6;                                r[2] &= 0x1FFF;
        r[3]  = p[4] >> 7;  r[3] |= (uint32_t)p[5]  << 1; r[3] |= (uint32_t)p[6] << 9;   r[3] &= 0x1FFF;
        r[4]  = p[6] >> 4;  r[4] |= (uint32_t)p[7]  << 4; r[4] |= (uint32_t)p[8] << 12;  r[4] &= 0x1FFF;
        r[5]  = p[8] >> 1;  r[5] |= (uint32_t)p[9]  << 7;                                r[5] &= 0x1FFF;
        r[6]  = p[9] >> 6;  r[6] |= (uint32_t)p[10] << 2; r[6] |= (uint32_t)p[11] << 10; r[6] &= 0x1FFF;
        r[7]  = p[11] >> 3; r[7] |= (uint32_t)p[12] << 5;                                r[7] &= 0x1FFF;
        for (int j = 0; j < 8; ++j) { r[j] = (1 << (D - 1)) - r[j]; }
    }
}

#if !defined(MLDSA_CDP) || !MLDSA_CDP
__global__ void __launch_bounds__(MT_THREADS) sign_multi_kernel(sgm_ws *w) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    KSHARED(int32_t, s_z, [N]);           
    KSHARED(int32_t, s_a, [MT_GROUPS * N]);
    const unsigned int t = threadIdx.x, b = blockIdx.x, nb = gridDim.x;
    const unsigned int grp = t / 128, lt = t % 128;    /* 묶음 번호, 묶음 안 스레드 번호 */
    int32_t *a = s_a + grp * N;
    const bool rec = (b == 0 && t == 0);               /* 시간 기록·반복 판정 칸 초기화 */
    const uint8_t *rho = w->sk, *key = w->sk + SEEDBYTES, *tr = w->sk + 2 * SEEDBYTES;
    unsigned long long tm = 0;
    /* 단계 시간 누적 */
#define SG_MARK(slot) do { if (rec) { unsigned long long n_ = mt_gtimer(); w->ph[slot] += n_ - tm; tm = n_; } } while (0)

    if (rec) { for (int i = 0; i < SGT_N; i++) { w->ph[i] = 0; } tm = mt_gtimer(); }
    zetas_to_shared(s_z, t);

    /* ---- P1: ExpandA 56개 ---- */
    if (t < 128) {
        for (unsigned int i = b + nb * t; i < (unsigned)(K * L); i += nb * 128) {
            const unsigned int r = i / L, c = i % L;
            PQCLEAN_MLDSA87_CLEAN_poly_uniform(&w->mat[r].vec[c], rho, (uint16_t)((r << 8) + c));
        }
    } else if (b == 0 && t == 128) {
        shake256incctx st;
        uint8_t pre[2] = { 0, 0 };
        shake256_inc_init(&st);
        shake256_inc_absorb(&st, tr, TRBYTES);
        shake256_inc_absorb(&st, pre, 2);
        shake256_inc_absorb(&st, w->msg, (size_t)w->mlen);
        shake256_inc_finalize(&st);
        shake256_inc_squeeze(w->mu, CRHBYTES, &st);
        shake256_inc_ctx_release(&st);
        uint8_t kb[SEEDBYTES + RNDBYTES + CRHBYTES];
        for (int i = 0; i < SEEDBYTES; i++) { kb[i] = key[i]; }
        for (int i = 0; i < RNDBYTES; i++) { kb[SEEDBYTES + i] = 0; }
        for (int i = 0; i < CRHBYTES; i++) { kb[SEEDBYTES + RNDBYTES + i] = w->mu[i]; }
        shake256(w->rhoprime, CRHBYTES, kb, sizeof kb);
    }
    grid.sync();
    SG_MARK(SGT_PRE1);

    /* ---- P2: sk unpack + NTT ---- */
    MT_FOR_TASKS(L + 2 * K, tau, act) {
        if (act && lt < (unsigned)(N / 8)) { sgm_unpack8(w->sk, tau, lt, a); }
        __syncthreads();
        ntt_coop(a, s_z, lt, act);
        if (act) {
            int32_t *dst = (tau < (unsigned)L) ? w->s1hat.vec[tau].coeffs
                         : (tau < (unsigned)(L + K)) ? w->s2hat.vec[tau - L].coeffs : w->t0hat.vec[tau - L - K].coeffs;
            dst[lt] = a[lt]; dst[lt + 128] = a[lt + 128];
        }
        __syncthreads();
    }
    grid.sync();
    SG_MARK(SGT_PRE2);

    uint16_t nonce = 0;
    int it = 0;
    for (;;) {
        const int par = it & 1;
        it++;
        /* ---- L1: y = ExpandMask, y_hat = NTT(y) ---- */
        if (rec) { w->flag[par] = 0; for (int k = 0; k < K; k++) { w->nh_row[par][k] = 0; } }
        MT_FOR_TASKS(L, l, act) {
            if (act && lt == 0) {
                PQCLEAN_MLDSA87_CLEAN_poly_uniform_gamma1(&w->y.vec[l], w->rhoprime, (uint16_t)(L * nonce + l));
            }
            __syncthreads();
            if (act) { a[lt] = w->y.vec[l].coeffs[lt]; a[lt + 128] = w->y.vec[l].coeffs[lt + 128]; }
            __syncthreads();
            ntt_coop(a, s_z, lt, act);
            if (act) { w->yhat.vec[l].coeffs[lt] = a[lt]; w->yhat.vec[l].coeffs[lt + 128] = a[lt + 128]; }
            __syncthreads();
        }
        nonce++;
        grid.sync();
        SG_MARK(SGT_L1);

        /* ---- L2: 행 k: w = NTT Inverse(reduce(A_hat_k·y_hat)) → caddq → decompose → w1 packing ---- */
        MT_FOR_TASKS(K, k, act) {
            if (act) {
                for (unsigned int c = lt; c < (unsigned)N; c += 128) {
                    int32_t acc = MT_MR(w->mat[k].vec[0].coeffs[c], w->yhat.vec[0].coeffs[c]);
                    for (int l = 1; l < L; ++l) { acc += MT_MR(w->mat[k].vec[l].coeffs[c], w->yhat.vec[l].coeffs[c]); }
                    a[c] = PQCLEAN_MLDSA87_CLEAN_reduce32(acc);
                }
            }
            __syncthreads();
            invntt_tomont_coop(a, s_z, lt, act);
            if (act) {
                int32_t h0, h1, l0, l1;
                h0 = PQCLEAN_MLDSA87_CLEAN_decompose(&l0, PQCLEAN_MLDSA87_CLEAN_caddq(a[2 * lt]));
                h1 = PQCLEAN_MLDSA87_CLEAN_decompose(&l1, PQCLEAN_MLDSA87_CLEAN_caddq(a[2 * lt + 1]));
                w->w1.vec[k].coeffs[2 * lt] = h0; w->w1.vec[k].coeffs[2 * lt + 1] = h1;
                w->w0.vec[k].coeffs[2 * lt] = l0; w->w0.vec[k].coeffs[2 * lt + 1] = l1;
                w->w1pack[k * POLYW1_PACKEDBYTES + lt] = (uint8_t)(h0 | (h1 << 4));
            }
            __syncthreads();
        }
        grid.sync();
        SG_MARK(SGT_L2);

        /* ---- L3: c_tild = H(mu‖w1), c = SampleInBall(c_tild), c_hat = NTT(c) ---- */
        if (b == 0) {
            if (t == 0) {
                shake256incctx st;
                shake256_inc_init(&st);
                shake256_inc_absorb(&st, w->mu, CRHBYTES);
                shake256_inc_absorb(&st, w->w1pack, K * POLYW1_PACKEDBYTES);
                shake256_inc_finalize(&st);
                shake256_inc_squeeze(w->sig, CTILDEBYTES, &st);
                shake256_inc_ctx_release(&st);
                PQCLEAN_MLDSA87_CLEAN_poly_challenge(&w->cp, w->sig);
            }
            __syncthreads();
            const bool act = (grp == 0);
            if (act) { a[lt] = w->cp.coeffs[lt]; a[lt + 128] = w->cp.coeffs[lt + 128]; }
            __syncthreads();
            ntt_coop(a, s_z, lt, act);
            if (act) { w->cphat.coeffs[lt] = a[lt]; w->cphat.coeffs[lt + 128] = a[lt + 128]; }
        }
        grid.sync();
        SG_MARK(SGT_L3);

        /* ---- L4: 작업 15개 = z 검사 7개 + 행 검사·힌트 8개 ---- */
        MT_FOR_TASKS(L + K, tau, act) {
            const bool isz = tau < (unsigned)L;
            const unsigned int k = isz ? 0 : tau - L;
            int bad = 0;
            unsigned int cnt = 0;
            int32_t w0p[2] = { 0, 0 };
            /* 1단계: z = c_hat·s1_hat[l], 행 = c_hat·s2_hat[k] */
            if (act) {
                const int32_t *s = isz ? w->s1hat.vec[tau].coeffs : w->s2hat.vec[k].coeffs;
                a[lt] = MT_MR(w->cphat.coeffs[lt], s[lt]);
                a[lt + 128] = MT_MR(w->cphat.coeffs[lt + 128], s[lt + 128]);
            }
            __syncthreads();
            invntt_tomont_coop(a, s_z, lt, act);
            if (act) {
                for (int q = 0; q < 2; ++q) {
                    const unsigned int c = lt + 128 * q;
                    if (isz) {                                         /* z = reduce(NTT Inverse + y), ||z|| < gamma1−beta */
                        int32_t v = PQCLEAN_MLDSA87_CLEAN_reduce32(a[c] + w->y.vec[tau].coeffs[c]);
                        w->z.vec[tau].coeffs[c] = v;
                        bad |= mt_over(v, GAMMA1 - BETA);
                    } else {                                           /* w0 − cs2 = reduce, ||.|| < gamma2−beta */
                        int32_t v = PQCLEAN_MLDSA87_CLEAN_reduce32(w->w0.vec[k].coeffs[c] - a[c]);
                        w0p[q] = v;
                        bad |= mt_over(v, GAMMA2 - BETA);
                    }
                }
            }
            __syncthreads();
            /* 2단계: ct0 = reduce(NTT Inverse(c_hat·t0_hat[k])), ||ct0|| < gamma2, 힌트 = make_hint(w0−cs2+ct0, w1) */
            const bool act2 = act && !isz;
            if (act2) {
                a[lt] = MT_MR(w->cphat.coeffs[lt], w->t0hat.vec[k].coeffs[lt]);
                a[lt + 128] = MT_MR(w->cphat.coeffs[lt + 128], w->t0hat.vec[k].coeffs[lt + 128]);
            }
            __syncthreads();
            invntt_tomont_coop(a, s_z, lt, act2);
            if (act2) {
                for (int q = 0; q < 2; ++q) {
                    const unsigned int c = lt + 128 * q;
                    int32_t hv = PQCLEAN_MLDSA87_CLEAN_reduce32(a[c]);
                    bad |= mt_over(hv, GAMMA2);
                    unsigned int hb = PQCLEAN_MLDSA87_CLEAN_make_hint(w0p[q] + hv, w->w1.vec[k].coeffs[c]);
                    w->h.vec[k].coeffs[c] = (int32_t)hb;
                    cnt += hb;
                }
                if (cnt) { atomicAdd(&w->nh_row[par][k], cnt); }
            }
            if (bad) { atomicAdd(&w->flag[par], 1); }
            __syncthreads();
        }
        grid.sync();
        SG_MARK(SGT_L4);
        unsigned int tot = 0;
        for (int k = 0; k < K; k++) { tot += w->nh_row[par][k]; }
        if (w->flag[par] == 0 && tot <= (unsigned)OMEGA) { break; }
    }

    /* ---- P3: 서명 packing ---- */
    {
        const int par = (it - 1) & 1;
        uint8_t *zb = w->sig + CTILDEBYTES;
        uint8_t *hb = w->sig + CTILDEBYTES + L * POLYZ_PACKEDBYTES;
        const unsigned int gt = b * blockDim.x + t, nt = nb * blockDim.x;
        for (unsigned int i = gt; i < (unsigned)(L * N / 2); i += nt) {
            const unsigned int l = i / (N / 2), j = i % (N / 2);
            uint32_t t0v = GAMMA1 - w->z.vec[l].coeffs[2 * j], t1v = GAMMA1 - w->z.vec[l].coeffs[2 * j + 1];
            uint8_t *r = zb + l * POLYZ_PACKEDBYTES + 5 * j;
            r[0] = (uint8_t)(t0v);
            r[1] = (uint8_t)(t0v >> 8);
            r[2] = (uint8_t)(t0v >> 16);
            r[2] |= (uint8_t)(t1v << 4);
            r[3] = (uint8_t)(t1v >> 4);
            r[4] = (uint8_t)(t1v >> 12);
        }
        unsigned int tot = 0;
        for (int k = 0; k < K; k++) { tot += w->nh_row[par][k]; }
        for (unsigned int i = tot + gt; i < (unsigned)OMEGA; i += nt) { hb[i] = 0; }
        MT_FOR_TASKS(K, k, act) {
            int32_t *cntv = a;
            unsigned int c2 = 0;
            if (act) {
                c2 = (w->h.vec[k].coeffs[2 * lt] != 0) + (w->h.vec[k].coeffs[2 * lt + 1] != 0);
                cntv[lt] = (int32_t)c2;
            }
            __syncthreads();
            if (act) {
                unsigned int pos = 0;
                for (unsigned int i = 0; i < (unsigned)k; i++) { pos += w->nh_row[par][i]; }
                const unsigned int rowbase = pos;
                for (unsigned int i = 0; i < lt; i++) { pos += (unsigned)cntv[i]; }
                if (w->h.vec[k].coeffs[2 * lt] != 0)     { hb[pos++] = (uint8_t)(2 * lt); }
                if (w->h.vec[k].coeffs[2 * lt + 1] != 0) { hb[pos]   = (uint8_t)(2 * lt + 1); }
                if (lt == 0) { hb[OMEGA + k] = (uint8_t)(rowbase + w->nh_row[par][k]); }
            }
            __syncthreads();
        }
        if (rec) { w->iters = it; }
    }
    grid.sync();
    SG_MARK(SGT_PACK);
#undef SG_MARK
}

#endif

#endif
