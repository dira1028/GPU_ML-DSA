#pragma once
namespace cdp
{
    namespace S = SHA3_CUDA;
    // All helpers below are collective: exactly the same 25 lanes and lengths.
    __device__ void hash256(S::shakeincctx *s, uint8_t *out, size_t outlen,
                            const uint8_t *in, size_t inlen)
    {
        S::shake256_inc_init(s);
        S::shake256_inc_absorb(s, in, inlen);
        S::shake256_inc_finalize(s);
        S::shake256_inc_squeeze(out, outlen, s);
    }
    __global__ void kg_seed(kgm_ws *w)
    {
        __shared__ S::shakeincctx state;
        __shared__ uint8_t buf[2 * SEEDBYTES + CRHBYTES];
        for (unsigned i = threadIdx.x; i < SEEDBYTES; i += 25)
            buf[i] = w->seed[i];
        if (threadIdx.x == 0)
        {
            buf[SEEDBYTES] = K;
            buf[SEEDBYTES + 1] = L;
        }
        __syncthreads();
        hash256(&state, buf, sizeof(buf), buf, SEEDBYTES + 2);
        for (unsigned i = threadIdx.x; i < SEEDBYTES; i += 25)
        {
            w->rho[i] = buf[i];
            w->key[i] = buf[SEEDBYTES + CRHBYTES + i];
        }
        for (unsigned i = threadIdx.x; i < CRHBYTES; i += 25)
            w->rhoprime[i] = buf[SEEDBYTES + i];
    }
    __global__ void kg_tr(kgm_ws *w)
    {
        __shared__ S::shakeincctx state;
        hash256(&state, w->sk + KGM_SK_TR, TRBYTES, w->pk, KGM_PKB);
    }
    __global__ void sg_hash_prepare(sgm_ws *w)
    {
        __shared__ S::shakeincctx state;
        __shared__ uint8_t zeros[RNDBYTES];
        for (unsigned i = threadIdx.x; i < RNDBYTES; i += 25)
            zeros[i] = 0;
        __syncthreads();
        S::shake256_inc_init(&state);
        S::shake256_inc_absorb(&state, w->sk + 2 * SEEDBYTES, TRBYTES);
        S::shake256_inc_absorb(&state, zeros, 2);
        S::shake256_inc_absorb(&state, w->msg, size_t(w->mlen));
        S::shake256_inc_finalize(&state);
        S::shake256_inc_squeeze(w->mu, CRHBYTES, &state);
        S::shake256_inc_init(&state);
        S::shake256_inc_absorb(&state, w->sk + SEEDBYTES, SEEDBYTES);
        S::shake256_inc_absorb(&state, zeros, RNDBYTES);
        S::shake256_inc_absorb(&state, w->mu, CRHBYTES);
        S::shake256_inc_finalize(&state);
        S::shake256_inc_squeeze(w->rhoprime, CRHBYTES, &state);
    }
    __global__ void sg_hash_challenge(sgm_ws *w)
    {
        __shared__ S::shakeincctx state;
        S::shake256_inc_init(&state);
        S::shake256_inc_absorb(&state, w->mu, CRHBYTES);
        S::shake256_inc_absorb(&state, w->w1pack, K * POLYW1_PACKEDBYTES);
        S::shake256_inc_finalize(&state);
        S::shake256_inc_squeeze(w->sig, CTILDEBYTES, &state);
    }

    __global__ void vf_hash_prepare(vfm_ws *w)
    {
        __shared__ S::shakeincctx state;
        __shared__ uint8_t tr[TRBYTES], pre[2];
        if (threadIdx.x == 0)
        {
            pre[0] = 0;
            pre[1] = 0;
        }
        __syncthreads();
        hash256(&state, tr, TRBYTES, w->pk, VF_PKB);
        S::shake256_inc_init(&state);
        S::shake256_inc_absorb(&state, tr, TRBYTES);
        S::shake256_inc_absorb(&state, pre, 2);
        S::shake256_inc_absorb(&state, w->msg, size_t(w->mlen));
        S::shake256_inc_finalize(&state);
        S::shake256_inc_squeeze(w->mu, CRHBYTES, &state);
    }

    __global__ void vf_hash_compare(vfm_ws *w)
    {
        __shared__ S::shakeincctx state;
        __shared__ uint8_t c2[CTILDEBYTES];
        S::shake256_inc_init(&state);
        S::shake256_inc_absorb(&state, w->mu, CRHBYTES);
        S::shake256_inc_absorb(&state, w->w1pack, K * POLYW1_PACKEDBYTES);
        S::shake256_inc_finalize(&state);
        S::shake256_inc_squeeze(c2, CTILDEBYTES, &state);
        if (threadIdx.x == 0)
        {
            int ok = w->bad == 0;
            for (unsigned i = 0; i < CTILDEBYTES; ++i)
                if (c2[i] != w->sig[i])
                    ok = 0;
            w->result = ok ? 0 : -1;
        }
    }
} // namespace cdp
