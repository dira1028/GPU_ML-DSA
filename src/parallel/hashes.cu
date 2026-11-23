#include "hashes.cuh"

namespace cdp { namespace S = SHA3_CUDA; }

namespace cdp {
/**************************************************
 * 이름: hash256
 *
 * 설명: SHAKE256 한 번 계산: init → absorb(in) → finalize → squeeze(out). 25스레드 공동 호출
 *
 * 인수: - s: 공유메모리 SHAKE 상태
 *      - out: 출력
 *      - outlen: 출력 길이
 *      - in: 입력
 *      - inlen: 입력 길이
 **************************************************/
__device__ void hash256(S::shakeincctx *s, uint8_t *out, size_t outlen,
                            const uint8_t *in, size_t inlen) {
        S::shake256_inc_init(s);
        S::shake256_inc_absorb(s, in, inlen);
        S::shake256_inc_finalize(s);
        S::shake256_inc_squeeze(out, outlen, s);
    }

/**************************************************
 * 이름: kg_seed
 *
 * 설명: 자식 커널 <<<1, 25>>>. KeyGen 시드 확장
 *      SHAKE256(seed ‖ K ‖ L) 128바이트를 ρ (32), ρ' (64), K (32) 로 나눠 작업 공간에 기록
 *
 * 인수: - w: KeyGen 작업 공간 (seed 읽기, rho·rhoprime·key 쓰기)
 **************************************************/
__global__ void kg_seed(kgm_ws *w) {
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


/**************************************************
 * 이름: kg_tr
 *
 * 설명: 자식 커널 <<<1, 25>>>. tr = SHAKE256(pk) 64바이트를 sk 의 tr 자리에 기록
 *
 * 인수: - w: KeyGen 작업 공간 (pk 읽기, sk 쓰기)
 **************************************************/
    //sk 저장 순서
    //rho(32Byte) || K(32Byte) || tr(64Byte) || s1 || s2 || t0
    //tr위치에 공개키 해시를 저장
__global__ void kg_tr(kgm_ws *w) {
        __shared__ S::shakeincctx state;
        hash256(&state, w->sk + KGM_SK_TR, TRBYTES, w->pk, KGM_PKB);
    }

/**************************************************
 * 이름: sg_hash_prepare
 *
 * 설명: 자식 커널 <<<1, 25>>>. μ = SHAKE256(tr ‖ 0x00 ‖ 0x00 ‖ msg)과
 *      ρ'' = SHAKE256(K ‖ rnd ‖ μ) (rnd = 0 × 32) 계산
 *
 * 인수: - w: Sign 작업 공간 (sk·msg·mlen 읽기, mu·rhoprime 쓰기)
 **************************************************/
__global__ void sg_hash_prepare(sgm_ws *w) {
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

/**************************************************
 * 이름: sg_hash_challenge
 *
 * 설명: 자식 커널 <<<1, 25>>>. c̃ = SHAKE256(μ ‖ w1 패킹) 64바이트를 서명의 앞부분에 기록
 *
 * 인수: - w: Sign 작업 공간 (mu·w1pack 읽기, sig[0..63] 쓰기)
 **************************************************/
__global__ void sg_hash_challenge(sgm_ws *w) {
        __shared__ S::shakeincctx state;
        S::shake256_inc_init(&state);
        S::shake256_inc_absorb(&state, w->mu, CRHBYTES);
        S::shake256_inc_absorb(&state, w->w1pack, K * POLYW1_PACKEDBYTES);
        S::shake256_inc_finalize(&state);
        S::shake256_inc_squeeze(w->sig, CTILDEBYTES, &state);
    }

/**************************************************
 * 이름: vf_hash_prepare
 *
 * 설명: 자식 커널 <<<1, 25>>>. tr = SHAKE256(pk), μ = SHAKE256(tr ‖ 0x00 ‖ 0x00 ‖ msg) 계산
 *
 * 인수: - w: Verify 작업 공간 (pk·msg·mlen 읽기, mu 쓰기)
 **************************************************/
__global__ void vf_hash_prepare(vfm_ws *w) {
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

/**************************************************
 * 이름: vf_hash_compare
 *
 * 설명: 자식 커널 <<<1, 25>>>. c̃' = SHAKE256(μ ‖ w1') 을 계산해 서명의 c̃ 와 비교
 *      앞 단계의 형식·노름 오류 수(bad) 까지 반영해 result = 0 (통과) / −1 (거부) 기록
 *
 * 인수: - w: Verify 작업 공간 (mu·w1pack·sig·bad 읽기, result 쓰기)
 **************************************************/
__global__ void vf_hash_compare(vfm_ws *w) {
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
