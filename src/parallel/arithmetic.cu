#include "arithmetic.cuh"

#undef MT_FOR_TASKS
/**************************************************
 * 이름: MT_FOR_TASKS
 *
 * 설명: 작업 0..n−1 을 블록에 나눠 주는 반복문. 블록 b 가 작업 b, b+gridDim.x, … 담당
 *      CDP 자식 커널은 블록 하나가 다항식 하나를 맡으므로 act 는 항상 true
 *
 * 인수: - n: 작업 수
 *       - task: 작업 번호 변수 이름
 *       - act: 활성 여부 변수 이름
 **************************************************/
#define MT_FOR_TASKS(n, task, act)                                          \
    for (unsigned task = blockIdx.x; task < unsigned(n); task += gridDim.x) \
        for (bool act = true, once = true; once; once = false)

namespace cdp
{
    /**************************************************
     * 이름: kg_ntt
     *
     * 설명: 자식 커널 <<<L, 128>>>. ŝ1 = NTT(s1). 블록 하나가 다항식 하나를 협력 NTT 로 변환
     *
     * 인수: - w: KeyGen 작업 공간 (s1 읽기, s1hat 쓰기)
     **************************************************/
    // 128개 Thread가 Poly에 대한 NTT 수행
    __global__ void kg_ntt(kgm_ws *w)
    {
        // 각 블록에서 처리할 Poly를 공유메모리에 올림.
        // 동일한 블록에서만 공유메모리 접근이 가능(한 블록당 Poly 하나씩 작업하는 형태)
        __shared__ int32_t s_z[N], s_a[N];
        const unsigned t = threadIdx.x, b = blockIdx.x, nb = gridDim.x;
        const unsigned grp = 0, lt = t;
        int32_t *a = s_a;
        /*
            상수메모리에 저장된 zetas배열을 공유 메모리로 옮기는 작업.
            상수 메모리를 사용하는 것보다 해당 방식이 더 빠름
        */
        zetas_to_shared(s_z, t);
        __syncthreads();

        // 블록마다 처리할 작업을 나눠주는 매크로
        MT_FOR_TASKS(L, l, act)
        {
            // 작업 블록에 해당하는 Poly를 공유메모리에 복사
            if (act)
            {
                a[lt] = w->s1.vec[l].coeffs[lt];
                a[lt + 128] = w->s1.vec[l].coeffs[lt + 128];
            }
            __syncthreads();
            ntt_coop(a, s_z, lt, act);
            // ntt 결과를 저장
            if (act)
            {
                w->s1hat.vec[l].coeffs[lt] = a[lt];
                w->s1hat.vec[l].coeffs[lt + 128] = a[lt + 128];
            }
            __syncthreads();
        }
    }

    /**************************************************
     * 이름: kg_rows
     *
     * 설명: 자식 커널 <<<K, 128>>>. 행 k (= 블록) 마다 t = Σ_l Â[k][l] ∘ ŝ1[l] → reduce32 → 역NTT 후,
     *      kgm_finish8 로 s2 덧셈, power2round, t1 (pk)·t0 (sk) 패킹
     *
     * 인수: - w: KeyGen 작업 공간
     **************************************************/
    __global__ void kg_rows(kgm_ws *w)
    {
        __shared__ int32_t s_z[N], s_a[N];
        const unsigned t = threadIdx.x, b = blockIdx.x, nb = gridDim.x;
        const unsigned grp = 0, lt = t;
        int32_t *a = s_a;
        zetas_to_shared(s_z, t);
        __syncthreads();

        //현재 코드에서는 사용하지 않는 매크로
        //추후 커널들을 합치면 블록에 따라 작업을 분배하기 위해 사용해야 함
        MT_FOR_TASKS(K, k, act)
        {
            //A * s1
            /*
                블록 인덱스에 따라 A[k] * s1 계산
            */
            if (act)
            {
                for (unsigned int c = lt; c < (unsigned)N; c += 128)
                {
                    int32_t acc = MT_MR(w->mat[k].vec[0].coeffs[c], w->s1hat.vec[0].coeffs[c]);
                    for (int l = 1; l < L; ++l)
                    {
                        acc += MT_MR(w->mat[k].vec[l].coeffs[c], w->s1hat.vec[l].coeffs[c]);
                    }
                    a[c] = PQCLEAN_MLDSA87_CLEAN_reduce32(acc);
                }
            }
            __syncthreads();
            invntt_tomont_coop(a, s_z, lt, act);
            if (act && lt < (unsigned)(N / 8))
            {
                //(A * s1) + s2 계산
                kgm_finish8(w, k, lt, a);
            } /* 32스레드 × 8계수 */
            __syncthreads();
        }
    }

    /**************************************************
     * 이름: sg_unpack
     *
     * 설명: 자식 커널 <<<L+2K = 23, 128>>>. 블록 tau 가 sk 에서 s1, s2, t0 중 다항식 하나를 풀어
     *      (sgm_unpack8) NTT 한 뒤 s1hat, s2hat, t0hat 에 기록
     *
     * 인수: - w: Sign 작업 공간 (sk 읽기)
     **************************************************/
    __global__ void sg_unpack(sgm_ws *w)
    {
        __shared__ int32_t s_z[N], s_a[N];
        const unsigned t = threadIdx.x, b = blockIdx.x, nb = gridDim.x;
        const unsigned grp = 0, lt = t;
        int32_t *a = s_a;
        zetas_to_shared(s_z, t);
        __syncthreads();

        MT_FOR_TASKS(L + 2 * K, tau, act)
        {
            if (act && lt < (unsigned)(N / 8))
            {
                sgm_unpack8(w->sk, tau, lt, a);
            }
            __syncthreads();
            ntt_coop(a, s_z, lt, act);
            if (act)
            {
                int32_t *dst = (tau < (unsigned)L)         ? w->s1hat.vec[tau].coeffs
                               : (tau < (unsigned)(L + K)) ? w->s2hat.vec[tau - L].coeffs
                                                           : w->t0hat.vec[tau - L - K].coeffs;
                dst[lt] = a[lt];
                dst[lt + 128] = a[lt + 128];
            }
            __syncthreads();
        }
    }

    /**************************************************
     * 이름: sg_rows
     *
     * 설명: 자식 커널 <<<K, 128>>>. 행 k 마다 w = 역NTT(reduce32(Σ_l Â[k][l] ∘ ŷ[l])) → caddq → decompose 로
     *      (w1, w0) 생성 후 w1 패킹 (스레드당 계수 2개 → 1바이트)
     *
     * 인수: - w: Sign 작업 공간 (mat·yhat 읽기, w0·w1·w1pack 쓰기)
     **************************************************/
    __global__ void sg_rows(sgm_ws *w)
    {
        __shared__ int32_t s_z[N], s_a[N];
        const unsigned t = threadIdx.x, b = blockIdx.x, nb = gridDim.x;
        const unsigned grp = 0, lt = t;
        int32_t *a = s_a;
        zetas_to_shared(s_z, t);
        __syncthreads();

        MT_FOR_TASKS(K, k, act)
        {
            if (act)
            {
                for (unsigned int c = lt; c < (unsigned)N; c += 128)
                {
                    int32_t acc = MT_MR(w->mat[k].vec[0].coeffs[c], w->yhat.vec[0].coeffs[c]);
                    for (int l = 1; l < L; ++l)
                    {
                        acc += MT_MR(w->mat[k].vec[l].coeffs[c], w->yhat.vec[l].coeffs[c]);
                    }
                    a[c] = PQCLEAN_MLDSA87_CLEAN_reduce32(acc);
                }
            }
            __syncthreads();
            invntt_tomont_coop(a, s_z, lt, act);
            if (act)
            {
                int32_t h0, h1, l0, l1;
                h0 = PQCLEAN_MLDSA87_CLEAN_decompose(&l0, PQCLEAN_MLDSA87_CLEAN_caddq(a[2 * lt]));
                h1 = PQCLEAN_MLDSA87_CLEAN_decompose(&l1, PQCLEAN_MLDSA87_CLEAN_caddq(a[2 * lt + 1]));
                w->w1.vec[k].coeffs[2 * lt] = h0;
                w->w1.vec[k].coeffs[2 * lt + 1] = h1;
                w->w0.vec[k].coeffs[2 * lt] = l0;
                w->w0.vec[k].coeffs[2 * lt + 1] = l1;
                w->w1pack[k * POLYW1_PACKEDBYTES + lt] = (uint8_t)(h0 | (h1 << 4));
            }
            __syncthreads();
        }
    }

    /**************************************************
     * 이름: sg_checks
     *
     * 설명: 자식 커널 <<<L+K = 15, 128>>>
     *       작업 0..L−1: z = y + c·s1 생성과 ‖z‖ < γ1−β 검사
     *       작업 L..L+K−1: ‖w0 − c·s2‖ < γ2−β, ‖c·t0‖ < γ2 검사와 힌트 생성 (make_hint)
     *       실패 수는 flag[par], 행별 힌트 수는 nh_row[par][k] 에 원자적 누적 (par = 시도 번호의 홀짝)
     *
     * 인수: - w: Sign 작업 공간
     **************************************************/
    __global__ void sg_checks(sgm_ws *w)
    {
        __shared__ int32_t s_z[N], s_a[N];
        const unsigned t = threadIdx.x, b = blockIdx.x, nb = gridDim.x;
        const unsigned grp = 0, lt = t;
        int32_t *a = s_a;
        zetas_to_shared(s_z, t);
        __syncthreads();
        const int par = (w->iters - 1) & 1;

        MT_FOR_TASKS(L + K, tau, act)
        {
            const bool isz = tau < (unsigned)L;
            const unsigned int k = isz ? 0 : tau - L;
            int bad = 0;
            unsigned int cnt = 0;
            int32_t w0p[2] = {0, 0};
            /* 1단계: z = c_hat·s1_hat[l], 행 = c_hat·s2_hat[k] */
            if (act)
            {
                const int32_t *s = isz ? w->s1hat.vec[tau].coeffs : w->s2hat.vec[k].coeffs;
                a[lt] = MT_MR(w->cphat.coeffs[lt], s[lt]);
                a[lt + 128] = MT_MR(w->cphat.coeffs[lt + 128], s[lt + 128]);
            }
            __syncthreads();
            invntt_tomont_coop(a, s_z, lt, act);
            if (act)
            {
                for (int q = 0; q < 2; ++q)
                {
                    const unsigned int c = lt + 128 * q;
                    if (isz)
                    { /* z = reduce(NTT Inverse + y), ||z|| < gamma1−beta */
                        int32_t v = PQCLEAN_MLDSA87_CLEAN_reduce32(a[c] + w->y.vec[tau].coeffs[c]);
                        w->z.vec[tau].coeffs[c] = v;
                        bad |= mt_over(v, GAMMA1 - BETA);
                    }
                    else
                    { /* w0 − cs2 = reduce, ||.|| < gamma2−beta */
                        int32_t v = PQCLEAN_MLDSA87_CLEAN_reduce32(w->w0.vec[k].coeffs[c] - a[c]);
                        w0p[q] = v;
                        bad |= mt_over(v, GAMMA2 - BETA);
                    }
                }
            }
            __syncthreads();
            /* 2단계: ct0 = reduce(NTT Inverse(c_hat·t0_hat[k])), ||ct0|| < gamma2, 힌트 = make_hint(w0−cs2+ct0, w1) */
            const bool act2 = act && !isz;
            if (act2)
            {
                a[lt] = MT_MR(w->cphat.coeffs[lt], w->t0hat.vec[k].coeffs[lt]);
                a[lt + 128] = MT_MR(w->cphat.coeffs[lt + 128], w->t0hat.vec[k].coeffs[lt + 128]);
            }
            __syncthreads();
            invntt_tomont_coop(a, s_z, lt, act2);
            if (act2)
            {
                for (int q = 0; q < 2; ++q)
                {
                    const unsigned int c = lt + 128 * q;
                    int32_t hv = PQCLEAN_MLDSA87_CLEAN_reduce32(a[c]);
                    bad |= mt_over(hv, GAMMA2);
                    unsigned int hb = PQCLEAN_MLDSA87_CLEAN_make_hint(w0p[q] + hv, w->w1.vec[k].coeffs[c]);
                    w->h.vec[k].coeffs[c] = (int32_t)hb;
                    cnt += hb;
                }
                if (cnt)
                {
                    atomicAdd(&w->nh_row[par][k], cnt);
                }
            }
            if (bad)
            {
                atomicAdd(&w->flag[par], 1);
            }
            __syncthreads();
        }
    }

    /**************************************************
     * 이름: sg_pack
     *
     * 설명: 자식 커널 <<<K, 128>>>. 채택된 시도의 z 를 서명에 패킹 (2계수 → 5바이트) 하고,
     *      힌트를 pack_sig 형식(행별 인덱스 목록 + 행 끝 누적 개수)으로 기록
     *      행 안의 쓰기 위치는 스레드별 힌트 수의 앞쪽 합으로 결정
     *      c̃ 는 sg_hash_challenge 가 미리 기록
     *
     * 인수: - w: Sign 작업 공간 (z·h·nh_row 읽기, sig 쓰기)
     **************************************************/
    __global__ void sg_pack(sgm_ws *w)
    {
        __shared__ int32_t s_z[N], s_a[N];
        const unsigned t = threadIdx.x, b = blockIdx.x, nb = gridDim.x;
        const unsigned grp = 0, lt = t;
        int32_t *a = s_a;
        zetas_to_shared(s_z, t);
        __syncthreads();

        {
            const int par = (w->iters - 1) & 1;
            uint8_t *zb = w->sig + CTILDEBYTES;
            uint8_t *hb = w->sig + CTILDEBYTES + L * POLYZ_PACKEDBYTES;
            const unsigned int gt = b * blockDim.x + t, nt = nb * blockDim.x;
            for (unsigned int i = gt; i < (unsigned)(L * N / 2); i += nt)
            {
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
            for (int k = 0; k < K; k++)
            {
                tot += w->nh_row[par][k];
            }
            for (unsigned int i = tot + gt; i < (unsigned)OMEGA; i += nt)
            {
                hb[i] = 0;
            }
            MT_FOR_TASKS(K, k, act)
            {
                int32_t *cntv = a;
                unsigned int c2 = 0;
                if (act)
                {
                    c2 = (w->h.vec[k].coeffs[2 * lt] != 0) + (w->h.vec[k].coeffs[2 * lt + 1] != 0);
                    cntv[lt] = (int32_t)c2;
                }
                __syncthreads();
                if (act)
                {
                    unsigned int pos = 0;
                    for (unsigned int i = 0; i < (unsigned)k; i++)
                    {
                        pos += w->nh_row[par][i];
                    }
                    const unsigned int rowbase = pos;
                    for (unsigned int i = 0; i < lt; i++)
                    {
                        pos += (unsigned)cntv[i];
                    }
                    if (w->h.vec[k].coeffs[2 * lt] != 0)
                    {
                        hb[pos++] = (uint8_t)(2 * lt);
                    }
                    if (w->h.vec[k].coeffs[2 * lt + 1] != 0)
                    {
                        hb[pos] = (uint8_t)(2 * lt + 1);
                    }
                    if (lt == 0)
                    {
                        hb[OMEGA + k] = (uint8_t)(rowbase + w->nh_row[par][k]);
                    }
                }
                __syncthreads();
            }
        }
    }

    /**************************************************
     * 이름: vf_ntt
     *
     * 설명: 자식 커널 <<<L+K+1 = 16, 128>>>
     *      작업 0..L−1: ẑ = NTT(z) / L..L+K−1: pk 에서 t1 을 풀어 2^D 를 곱한 뒤 NTT / 마지막 작업: ĉ = NTT(c)
     *
     * 인수: - w: Verify 작업 공간
     **************************************************/
    __global__ void vf_ntt(vfm_ws *w)
    {
        __shared__ int32_t s_z[N], s_a[N];
        const unsigned t = threadIdx.x, b = blockIdx.x, nb = gridDim.x;
        const unsigned grp = 0, lt = t;
        int32_t *a = s_a;
        zetas_to_shared(s_z, t);
        __syncthreads();

        MT_FOR_TASKS(L + K + 1, tau, act)
        {
            if (act)
            {
                if (tau < (unsigned)L)
                {
                    a[lt] = w->z.vec[tau].coeffs[lt];
                    a[lt + 128] = w->z.vec[tau].coeffs[lt + 128];
                }
                else if (tau < (unsigned)(L + K))
                {
                    if (lt < (unsigned)(N / 4))
                    {
                        const uint8_t *p = w->pk + SEEDBYTES + (tau - L) * POLYT1_PACKEDBYTES + 5 * lt;
                        a[4 * lt + 0] = (int32_t)(((p[0] >> 0) | ((uint32_t)p[1] << 8)) & 0x3FF) << D;
                        a[4 * lt + 1] = (int32_t)(((p[1] >> 2) | ((uint32_t)p[2] << 6)) & 0x3FF) << D;
                        a[4 * lt + 2] = (int32_t)(((p[2] >> 4) | ((uint32_t)p[3] << 4)) & 0x3FF) << D;
                        a[4 * lt + 3] = (int32_t)(((p[3] >> 6) | ((uint32_t)p[4] << 2)) & 0x3FF) << D;
                    }
                }
                else
                {
                    a[lt] = w->cp.coeffs[lt];
                    a[lt + 128] = w->cp.coeffs[lt + 128];
                }
            }
            __syncthreads();
            ntt_coop(a, s_z, lt, act);
            if (act)
            {
                int32_t *dst = (tau < (unsigned)L)         ? w->zhat.vec[tau].coeffs
                               : (tau < (unsigned)(L + K)) ? w->t1hat.vec[tau - L].coeffs
                                                           : w->cphat.coeffs;
                dst[lt] = a[lt];
                dst[lt + 128] = a[lt + 128];
            }
            __syncthreads();
        }
    }

    /**************************************************
     * 이름: vf_rows
     *
     * 설명: 자식 커널 <<<K, 128>>>. 행 k 마다 w' = 역NTT(reduce32(Σ_l Â[k][l] ∘ ẑ[l] − ĉ ∘ t̂1[k]))
     *      → caddq → use_hint 로 w1' 생성 후 패킹
     *
     * 인수: - w: Verify 작업 공간 (mat·zhat·cphat·t1hat·h 읽기, w1pack 쓰기)
     **************************************************/
    __global__ void vf_rows(vfm_ws *w)
    {
        __shared__ int32_t s_z[N], s_a[N];
        const unsigned t = threadIdx.x, b = blockIdx.x, nb = gridDim.x;
        const unsigned grp = 0, lt = t;
        int32_t *a = s_a;
        zetas_to_shared(s_z, t);
        __syncthreads();

        MT_FOR_TASKS(K, k, act)
        {
            if (act)
            {
                for (unsigned int c = lt; c < (unsigned)N; c += 128)
                {
                    int32_t acc = MT_MR(w->mat[k].vec[0].coeffs[c], w->zhat.vec[0].coeffs[c]);
                    for (int l = 1; l < L; ++l)
                    {
                        acc += MT_MR(w->mat[k].vec[l].coeffs[c], w->zhat.vec[l].coeffs[c]);
                    }
                    acc -= MT_MR(w->cphat.coeffs[c], w->t1hat.vec[k].coeffs[c]);
                    a[c] = PQCLEAN_MLDSA87_CLEAN_reduce32(acc);
                }
            }
            __syncthreads();
            invntt_tomont_coop(a, s_z, lt, act);
            if (act)
            {
                int32_t u0 = PQCLEAN_MLDSA87_CLEAN_use_hint(PQCLEAN_MLDSA87_CLEAN_caddq(a[2 * lt]), (unsigned)w->h.vec[k].coeffs[2 * lt]);
                int32_t u1 = PQCLEAN_MLDSA87_CLEAN_use_hint(PQCLEAN_MLDSA87_CLEAN_caddq(a[2 * lt + 1]), (unsigned)w->h.vec[k].coeffs[2 * lt + 1]);
                w->w1pack[k * POLYW1_PACKEDBYTES + lt] = (uint8_t)(u0 | (u1 << 4));
            }
            __syncthreads();
        }
    }

    /**************************************************
     * 이름: ntt_vector
     *
     * 설명: 자식 커널 <<<다항식 수, 128>>>. 블록 b 가 src[b] 를 협력 NTT 해 dst[b] 에 기록
     *
     * 인수: - dst: 출력 다항식 배열
     *      - src: 입력 다항식 배열
     **************************************************/
    __global__ void ntt_vector(poly *dst, const poly *src)
    {
        __shared__ int32_t s_z[N], s_a[N];
        const unsigned t = threadIdx.x, b = blockIdx.x, nb = gridDim.x;
        const unsigned grp = 0, lt = t;
        int32_t *a = s_a;
        zetas_to_shared(s_z, t);
        __syncthreads();

        a[lt] = src[blockIdx.x].coeffs[lt];
        a[lt + 128] = src[blockIdx.x].coeffs[lt + 128];
        __syncthreads();
        ntt_coop(a, s_z, lt);
        dst[blockIdx.x].coeffs[lt] = a[lt];
        dst[blockIdx.x].coeffs[lt + 128] = a[lt + 128];
    }

    /**************************************************
     * 이름: kg_pack_secrets
     *
     * 설명: 자식 커널 <<<L+K, 128>>>. 블록 j 의 스레드 0 이 s1/s2 다항식 j 를 sk 의 η 자리에 패킹
     *      (polyeta_pack). 블록 0 은 pk·sk 앞부분에 ρ 를, sk 에 K 를 복사
     *
     * 인수: - w: KeyGen 작업 공간
     **************************************************/
    __global__ void kg_pack_secrets(kgm_ws *w)
    {
        unsigned j = blockIdx.x;

        // PQCLEAN_MLDSA87_CLEAN_polyeta_pack함수 추후 32 스레드 병렬화 필요
        if (threadIdx.x == 0)
        {
            const poly *p = j < L ? &w->s1.vec[j] : &w->s2.vec[j - L];
            PQCLEAN_MLDSA87_CLEAN_polyeta_pack(w->sk + KGM_SK_ETA + j * POLYETA_PACKEDBYTES, p);
        }

        if (j == 0 && threadIdx.x < SEEDBYTES)
        {
            unsigned t = threadIdx.x;
            w->pk[t] = w->rho[t];
            w->sk[t] = w->rho[t];
            w->sk[SEEDBYTES + t] = w->key[t];
        }
    }

    /**************************************************
     * 이름: vf_unpack
     *
     * 설명: 자식 커널 <<<L, 128>>>. 블록 l 이 서명의 z[l] 을 풀고 (스레드당 2계수) ‖z‖ < γ1−β 검사
     *      블록 0 스레드 0 은 힌트 해독 (vfm_decode_hint). 오류는 bad 에 누적
     *
     * 인수: - w: Verify 작업 공간 (sig 읽기, z·h·bad 쓰기)
     **************************************************/
    __global__ void vf_unpack(vfm_ws *w)
    {
        const unsigned t = threadIdx.x;
        if (blockIdx.x == 0 && t == 0 && vfm_decode_hint(w->sig + CTILDEBYTES + L * POLYZ_PACKEDBYTES, &w->h))
            atomicAdd(&w->bad, 1);
        unsigned l = blockIdx.x;
        const uint8_t *p = w->sig + CTILDEBYTES + l * POLYZ_PACKEDBYTES + 5 * t;
        int32_t a = (p[0] | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16)) & 0xfffff;
        int32_t b = (p[2] >> 4) | (uint32_t(p[3]) << 4) | (uint32_t(p[4]) << 12);
        a = GAMMA1 - a;
        b = GAMMA1 - b;
        w->z.vec[l].coeffs[2 * t] = a;
        w->z.vec[l].coeffs[2 * t + 1] = b;
        if (mt_over(a, GAMMA1 - BETA) | mt_over(b, GAMMA1 - BETA))
            atomicAdd(&w->bad, 1);
    }
} // namespace cdp
