#pragma once
// Arithmetic bodies retained from multi/*.cuh; one 128-thread group per block.
// No grid barriers: each stage is a separate child grid in an ordered stream.
#undef MT_FOR_TASKS
#define MT_FOR_TASKS(n, task, act) \
    for (unsigned task=blockIdx.x; task<unsigned(n); task+=gridDim.x) \
        for (bool act=true, once=true; once; once=false)
namespace cdp {
__global__ void kg_ntt(kgm_ws* w) {
    __shared__ int32_t s_z[N], s_a[N];
    const unsigned t=threadIdx.x, b=blockIdx.x, nb=gridDim.x;
    const unsigned grp=0, lt=t;
    int32_t* a=s_a;
    zetas_to_shared(s_z,t);
    __syncthreads();


    MT_FOR_TASKS(L, l, act) {
        if (act) { a[lt] = w->s1.vec[l].coeffs[lt]; a[lt + 128] = w->s1.vec[l].coeffs[lt + 128]; }
        __syncthreads();
        ntt_coop(a, s_z, lt, act);
        if (act) { w->s1hat.vec[l].coeffs[lt] = a[lt]; w->s1hat.vec[l].coeffs[lt + 128] = a[lt + 128]; }
        __syncthreads();
    }

}
__global__ void kg_rows(kgm_ws* w) {
    __shared__ int32_t s_z[N], s_a[N];
    const unsigned t=threadIdx.x, b=blockIdx.x, nb=gridDim.x;
    const unsigned grp=0, lt=t;
    int32_t* a=s_a;
    zetas_to_shared(s_z,t);
    __syncthreads();


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

}
__global__ void sg_unpack(sgm_ws* w) {
    __shared__ int32_t s_z[N], s_a[N];
    const unsigned t=threadIdx.x, b=blockIdx.x, nb=gridDim.x;
    const unsigned grp=0, lt=t;
    int32_t* a=s_a;
    zetas_to_shared(s_z,t);
    __syncthreads();


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

}
__global__ void sg_rows(sgm_ws* w) {
    __shared__ int32_t s_z[N], s_a[N];
    const unsigned t=threadIdx.x, b=blockIdx.x, nb=gridDim.x;
    const unsigned grp=0, lt=t;
    int32_t* a=s_a;
    zetas_to_shared(s_z,t);
    __syncthreads();


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

}
__global__ void sg_checks(sgm_ws* w) {
    __shared__ int32_t s_z[N], s_a[N];
    const unsigned t=threadIdx.x, b=blockIdx.x, nb=gridDim.x;
    const unsigned grp=0, lt=t;
    int32_t* a=s_a;
    zetas_to_shared(s_z,t);
    __syncthreads();
    const int par=(w->iters-1)&1;

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

}
__global__ void sg_pack(sgm_ws* w) {
    __shared__ int32_t s_z[N], s_a[N];
    const unsigned t=threadIdx.x, b=blockIdx.x, nb=gridDim.x;
    const unsigned grp=0, lt=t;
    int32_t* a=s_a;
    zetas_to_shared(s_z,t);
    __syncthreads();


    {
        const int par = (w->iters - 1) & 1;
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

    }

}
__global__ void vf_ntt(vfm_ws* w) {
    __shared__ int32_t s_z[N], s_a[N];
    const unsigned t=threadIdx.x, b=blockIdx.x, nb=gridDim.x;
    const unsigned grp=0, lt=t;
    int32_t* a=s_a;
    zetas_to_shared(s_z,t);
    __syncthreads();


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

}
__global__ void vf_rows(vfm_ws* w) {
    __shared__ int32_t s_z[N], s_a[N];
    const unsigned t=threadIdx.x, b=blockIdx.x, nb=gridDim.x;
    const unsigned grp=0, lt=t;
    int32_t* a=s_a;
    zetas_to_shared(s_z,t);
    __syncthreads();


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

}
__global__ void ntt_vector(poly* dst, const poly* src) {
    __shared__ int32_t s_z[N], s_a[N];
    const unsigned t=threadIdx.x, b=blockIdx.x, nb=gridDim.x;
    const unsigned grp=0, lt=t;
    int32_t* a=s_a;
    zetas_to_shared(s_z,t);
    __syncthreads();

    a[lt]=src[blockIdx.x].coeffs[lt];
    a[lt+128]=src[blockIdx.x].coeffs[lt+128];
    __syncthreads();
    ntt_coop(a,s_z,lt);
    dst[blockIdx.x].coeffs[lt]=a[lt];
    dst[blockIdx.x].coeffs[lt+128]=a[lt+128];
}
__global__ void kg_pack_secrets(kgm_ws* w) {
    unsigned j=blockIdx.x;
    if (threadIdx.x==0) {
        const poly* p=j<L ? &w->s1.vec[j] : &w->s2.vec[j-L];
        PQCLEAN_MLDSA87_CLEAN_polyeta_pack(w->sk+KGM_SK_ETA+j*POLYETA_PACKEDBYTES,p);
    }
    if(j==0 && threadIdx.x<SEEDBYTES) {
        unsigned t=threadIdx.x;
        w->pk[t]=w->rho[t]; w->sk[t]=w->rho[t]; w->sk[SEEDBYTES+t]=w->key[t];
    }
}
__global__ void vf_unpack(vfm_ws* w) {
    const unsigned t=threadIdx.x;
    if(blockIdx.x==0 && t==0 && vfm_decode_hint(w->sig+CTILDEBYTES+L*POLYZ_PACKEDBYTES,&w->h)) atomicAdd(&w->bad,1);
    unsigned l=blockIdx.x;
    const uint8_t* p=w->sig+CTILDEBYTES+l*POLYZ_PACKEDBYTES+5*t;
    int32_t a=(p[0] | (uint32_t(p[1])<<8) | (uint32_t(p[2])<<16)) & 0xfffff;
    int32_t b=(p[2]>>4) | (uint32_t(p[3])<<4) | (uint32_t(p[4])<<12);
    a=GAMMA1-a; b=GAMMA1-b;
    w->z.vec[l].coeffs[2*t]=a; w->z.vec[l].coeffs[2*t+1]=b;
    if(mt_over(a,GAMMA1-BETA) | mt_over(b,GAMMA1-BETA)) atomicAdd(&w->bad,1);
}
} // namespace cdp
#undef MT_FOR_TASKS
