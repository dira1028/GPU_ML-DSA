/*
 * multi/verify_multi.cuh  ---  다중 스레드 Verify (커널 1개)
 *
 * [입력/출력]  작업 공간 vfm_ws 의 pk(2592), sig(4627), msg(mlen) -> result (0 통과 / -1 거부).
 * [실행]      verify_multi_kernel<<<nb, 256>>>
 */
#ifndef MLDSA87_VERIFY_MULTI_CUH
#define MLDSA87_VERIFY_MULTI_CUH

#include "../kernels/multi_common.cuh"

#define VF_PKB  PQCLEAN_MLDSA87_CLEAN_CRYPTO_PUBLICKEYBYTES
#define VF_SIGB PQCLEAN_MLDSA87_CLEAN_CRYPTO_BYTES
#define VF_MAXM 256

/* 단계별 시간 칸 */
enum { VFT_P1 = 0, VFT_P2, VFT_P3, VFT_P4, VFT_N };
__attribute__((unused)) static const char *VFT_NAME[VFT_N] = {
    "P1 ExpandA|H(pk),mu|chal.|unpack",
    "P2 NTT z(7), t1*2^d(8), c(1)",
    "P3 A*z-c*t1, invNTT, use_hint",
    "P4 H(mu||w1) + compare"
};

/* GPU 작업 공간 */
typedef struct {
    int cdp_error; // First device-runtime error; host must check after completion.

    uint8_t  pk[VF_PKB];                       /* 입력: 공개키 */
    uint8_t  sig[VF_SIGB];                     /* 입력: 서명 */
    uint8_t  msg[VF_MAXM];                     /* 입력: 메시지 */
    int      mlen;
    uint8_t  mu[CRHBYTES];
    polyvecl mat[K];
    polyvecl z, zhat;
    polyveck t1hat, h;
    poly     cp, cphat;
    uint8_t  w1pack[K * POLYW1_PACKEDBYTES];
    int      bad;                              /* 형식 오류·노름 초과 수 */
    int      result;                           /* 0 = 통과, -1 = 거부 */
    unsigned long long ph[VFT_N];
} vfm_ws;

/* unpack_sig 의 힌트 계산 */
__device__ int vfm_decode_hint(const uint8_t *hb, polyveck *h) {
    unsigned int i, j, k = 0;
    int err = 0;
    for (i = 0; i < (unsigned)K; ++i) {
        for (j = 0; j < (unsigned)N; ++j) { h->vec[i].coeffs[j] = 0; }
        if (!err) {
            if (hb[OMEGA + i] < k || hb[OMEGA + i] > OMEGA) { err = 1; continue; }
            for (j = k; j < hb[OMEGA + i]; ++j) {
                if (j > k && hb[j] <= hb[j - 1]) { err = 1; break; }
                h->vec[i].coeffs[hb[j]] = 1;
            }
            if (!err) { k = hb[OMEGA + i]; }
        }
    }
    if (!err) {
        for (j = k; j < (unsigned)OMEGA; ++j) { if (hb[j]) { err = 1; break; } }
    }
    return err;
}

#if !defined(MLDSA_CDP) || !MLDSA_CDP
__global__ void __launch_bounds__(MT_THREADS) verify_multi_kernel(vfm_ws *w) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    KSHARED(int32_t, s_z, [N]);
    KSHARED(int32_t, s_a, [MT_GROUPS * N]);
    const unsigned int t = threadIdx.x, b = blockIdx.x, nb = gridDim.x;
    const unsigned int grp = t / 128, lt = t % 128;
    int32_t *a = s_a + grp * N;
    const bool rec = (b == 0 && t == 0);
    const uint8_t *rho = w->pk;
    unsigned long long tm = 0;
#define VF_MARK(slot) do { if (rec) { unsigned long long n_ = mt_gtimer(); w->ph[slot] = n_ - tm; tm = n_; } } while (0)

    if (rec) { tm = mt_gtimer(); w->bad = 0; }
    zetas_to_shared(s_z, t);
    grid.sync();
    if (rec) { tm = mt_gtimer(); }

    /* ---- P1 ---- */
    if (t < 128) {                                                      /* ExpandA 56개 */
        for (unsigned int i = b + nb * t; i < (unsigned)(K * L); i += nb * 128) {
            const unsigned int r = i / L, c = i % L;
            PQCLEAN_MLDSA87_CLEAN_poly_uniform(&w->mat[r].vec[c], rho, (uint16_t)((r << 8) + c));
        }
    } else if (b == 0 && t == 128) {                                    /* mu = H(H(pk) || 0 || 0 || m) */
        uint8_t tr[TRBYTES];
        shake256(tr, TRBYTES, w->pk, VF_PKB);
        shake256incctx st;
        uint8_t pre[2] = { 0, 0 };
        shake256_inc_init(&st);
        shake256_inc_absorb(&st, tr, TRBYTES);
        shake256_inc_absorb(&st, pre, 2);
        shake256_inc_absorb(&st, w->msg, (size_t)w->mlen);
        shake256_inc_finalize(&st);
        shake256_inc_squeeze(w->mu, CRHBYTES, &st);
        shake256_inc_ctx_release(&st);
    } else if (b == (1 % nb) && t == 160) {                             /* c = SampleInBall(c_tild) */
        PQCLEAN_MLDSA87_CLEAN_poly_challenge(&w->cp, w->sig);
    } else if (b == (2 % nb) && t == 192) {                             /* 힌트 검사 */
        if (vfm_decode_hint(w->sig + CTILDEBYTES + L * POLYZ_PACKEDBYTES, &w->h)) { atomicAdd(&w->bad, 1); }
    } else if (t >= 224) {                                              /* z unpack + 검사 */
        int bad = 0;
        for (unsigned int i = b * 32 + (t - 224); i < (unsigned)(L * N / 2); i += nb * 32) {
            const unsigned int l = i / (N / 2), j = i % (N / 2);
            const uint8_t *p = w->sig + CTILDEBYTES + l * POLYZ_PACKEDBYTES + 5 * j;
            int32_t c0, c1;
            c0  = p[0];
            c0 |= (uint32_t)p[1] << 8;
            c0 |= (uint32_t)p[2] << 16;
            c0 &= 0xFFFFF;
            c1  = p[2] >> 4;
            c1 |= (uint32_t)p[3] << 4;
            c1 |= (uint32_t)p[4] << 12;
            c0 = GAMMA1 - c0; c1 = GAMMA1 - c1;
            w->z.vec[l].coeffs[2 * j] = c0; w->z.vec[l].coeffs[2 * j + 1] = c1;
            bad |= mt_over(c0, GAMMA1 - BETA) | mt_over(c1, GAMMA1 - BETA);
        }
        if (bad) { atomicAdd(&w->bad, 1); }
    }
    grid.sync();
    VF_MARK(VFT_P1);

    /* ---- P2: NTT 16개 = z_hat 7개, t1_hat 8개(pk unpack -> ·2^d), c_hat 1개 ---- */
    MT_FOR_TASKS(L + K + 1, tau, act) {
        if (act) {
            if (tau < (unsigned)L) {
                a[lt] = w->z.vec[tau].coeffs[lt]; a[lt + 128] = w->z.vec[tau].coeffs[lt + 128];
            } else if (tau < (unsigned)(L + K)) {
                if (lt < (unsigned)(N / 4)) {
                    const uint8_t *p = w->pk + SEEDBYTES + (tau - L) * POLYT1_PACKEDBYTES + 5 * lt;
                    a[4 * lt + 0] = (int32_t)(((p[0] >> 0) | ((uint32_t)p[1] << 8)) & 0x3FF) << D;
                    a[4 * lt + 1] = (int32_t)(((p[1] >> 2) | ((uint32_t)p[2] << 6)) & 0x3FF) << D;
                    a[4 * lt + 2] = (int32_t)(((p[2] >> 4) | ((uint32_t)p[3] << 4)) & 0x3FF) << D;
                    a[4 * lt + 3] = (int32_t)(((p[3] >> 6) | ((uint32_t)p[4] << 2)) & 0x3FF) << D;
                }
            } else {
                a[lt] = w->cp.coeffs[lt]; a[lt + 128] = w->cp.coeffs[lt + 128];
            }
        }
        __syncthreads();
        ntt_coop(a, s_z, lt, act);
        if (act) {
            int32_t *dst = (tau < (unsigned)L) ? w->zhat.vec[tau].coeffs
                         : (tau < (unsigned)(L + K)) ? w->t1hat.vec[tau - L].coeffs : w->cphat.coeffs;
            dst[lt] = a[lt]; dst[lt + 128] = a[lt + 128];
        }
        __syncthreads();
    }
    grid.sync();
    VF_MARK(VFT_P2);

    /* ---- P3: 행 k: A_hat_k·z_hat − c_hat·t1_hat_k → reduce → NTT Inverse → caddq → use_hint → w1 packing ---- */
    MT_FOR_TASKS(K, k, act) {
        if (act) {
            for (unsigned int c = lt; c < (unsigned)N; c += 128) {
                int32_t acc = MT_MR(w->mat[k].vec[0].coeffs[c], w->zhat.vec[0].coeffs[c]);
                for (int l = 1; l < L; ++l) { acc += MT_MR(w->mat[k].vec[l].coeffs[c], w->zhat.vec[l].coeffs[c]); }
                acc -= MT_MR(w->cphat.coeffs[c], w->t1hat.vec[k].coeffs[c]);
                a[c] = PQCLEAN_MLDSA87_CLEAN_reduce32(acc);
            }
        }
        __syncthreads();
        invntt_tomont_coop(a, s_z, lt, act);
        if (act) {
            int32_t u0 = PQCLEAN_MLDSA87_CLEAN_use_hint(PQCLEAN_MLDSA87_CLEAN_caddq(a[2 * lt]), (unsigned)w->h.vec[k].coeffs[2 * lt]);
            int32_t u1 = PQCLEAN_MLDSA87_CLEAN_use_hint(PQCLEAN_MLDSA87_CLEAN_caddq(a[2 * lt + 1]), (unsigned)w->h.vec[k].coeffs[2 * lt + 1]);
            w->w1pack[k * POLYW1_PACKEDBYTES + lt] = (uint8_t)(u0 | (u1 << 4));
        }
        __syncthreads();
    }
    grid.sync();
    VF_MARK(VFT_P3);

    /* ---- P4: c_tile` = H(mu||w1), 비교 ---- */
    if (rec) {
        uint8_t c2[CTILDEBYTES];
        shake256incctx st;
        shake256_inc_init(&st);
        shake256_inc_absorb(&st, w->mu, CRHBYTES);
        shake256_inc_absorb(&st, w->w1pack, K * POLYW1_PACKEDBYTES);
        shake256_inc_finalize(&st);
        shake256_inc_squeeze(c2, CTILDEBYTES, &st);
        shake256_inc_ctx_release(&st);
        int ok = (w->bad == 0);
        for (int i = 0; i < CTILDEBYTES; ++i) { if (w->sig[i] != c2[i]) { ok = 0; } }
        w->result = ok ? 0 : -1;
        VF_MARK(VFT_P4);
    }
#undef VF_MARK
}

#endif

#endif
