#include "parent.cuh"
#include "sampling.cuh"
#include "hashes.cuh"
#include "arithmetic.cuh"

#if CUDART_VERSION < 12000
#error "The CDP path requires CUDA 12+ (CDP2 tail launches)."
#endif
#ifdef CUDA_FORCE_CDP1_IF_SUPPORTED
#error "The CDP path requires CDP2; remove CUDA_FORCE_CDP1_IF_SUPPORTED."
#endif
/**************************************************
 * 이름: CDP_CHECK
 *
 * 설명: 직전 자식 실행의 오류 확인
 *      오류면 cdp_error 에 기록하고 스트림을 정리한 뒤 부모 함수 종료 (이후 자식은 미등록)
 *      이미 등록된 자식은 그대로 실행되지만 호스트가 cdp_error 를 보고 결과를 폐기
 *
 * 인수: 없음
 **************************************************/
#define CDP_CHECK()                                           \
    do                                                        \
    {                                                         \
        if (!record(&w->cdp_error, cudaGetLastError()))       \
        {                                                     \
            record(&w->cdp_error, cudaStreamDestroy(stream)); \
            return;                                           \
        }                                                     \
    } while (0)

/**************************************************
 * 이름: CDP_STREAM
 *
 * 설명: 장치 스트림 stream 생성 (cudaStreamNonBlocking)
 *      실패하면 cdp_error 에 기록하고 부모 함수 종료
 *
 * 인수: 없음
 **************************************************/
#define CDP_STREAM()                                                                       \
    cudaStream_t stream;                                                                   \
    if (!record(&w->cdp_error, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))) \
    return

namespace cdp
{
    /**************************************************
     * 이름: record
     *
     * 설명: CUDA API 결과를 검사해 실패면 오류 코드를 error 에 기록 (atomicCAS 로 첫 오류만 보존)
     *
     * 인수: - error: 작업 공간의 cdp_error
     *      - status: 검사할 CUDA API 결과
     *
     * 성공이면 true 를 반환
     **************************************************/
    __device__ bool record(int *error, cudaError_t status)
    {
        if (status == cudaSuccess)
            return true;
        atomicCAS(error, 0, int(status));
        return false;
    }

    /**************************************************
     * 이름: keygen_parent
     *
     * 설명: KeyGen 부모 커널 (호스트가 <<<1, 1>>> 로 실행). 장치 스트림에 자식 커널을 순서대로 등록:
     *      kg_seed → expand_a → expand_s → kg_pack_secrets → kg_ntt → kg_rows → kg_tr
     *      같은 스트림이므로 순서대로 실행되며, 부모는 자식이 모두 끝나야 완료
     *
     * 인수: - w: KeyGen 작업 공간 (seed 입력, pk·sk 출력)
     **************************************************/
    __global__ void keygen_parent(kgm_ws *w)
    {
        if (blockIdx.x || threadIdx.x)
            return;
        w->cdp_error = 0;
        CDP_STREAM();
        /*
         * kg_seed: 시드를 확장해 rho, rhoprime, key 생성.
         * 블록 1개: 시드 하나를 처리.
         * 블록당 25스레드: Keccak의 25개 lane을 하나씩 담당.
         */
        kg_seed<<<1, 25, 0, stream>>>(w);
        CDP_CHECK();

        /*
         * expand_a: rho로 행렬 A 샘플링.
         * 블록 K*L개: 행렬의 다항식 하나씩 담당.
         * 블록당 25스레드: SHAKE의 25개 lane을 하나씩 담당.
         */
        expand_a<<<K * L, 25, 0, stream>>>(w->mat, w->rho);
        CDP_CHECK();

        /*
         * expand_s: rhoprime으로 비밀 벡터 s1, s2 샘플링.
         * 블록 L+K개: s1, s2의 다항식 하나씩 담당.
         * 블록당 25스레드: SHAKE의 25개 lane을 하나씩 담당.
         */
        expand_s<<<L + K, 25, 0, stream>>>(&w->s1, &w->s2, w->rhoprime);
        CDP_CHECK();

        /*
         * kg_pack_secrets: s1, s2를 패킹하고 rho, key 저장.
         * 블록 L+K개: s1, s2의 다항식 하나씩 담당.
         * 블록당 32스레드: 32바이트 시드 복사에 맞춤. 다항식 패킹은 스레드 0이 처리.
         */
        kg_pack_secrets<<<L + K, 32, 0, stream>>>(w);
        CDP_CHECK();

        /*
         * kg_ntt: s1을 NTT로 변환해 s1hat에 저장.
         * 블록 L개: s1의 다항식 하나씩 담당.
         * 블록당 128스레드: NTT 단계마다 128개 butterfly를 하나씩 담당.
         */
        kg_ntt<<<L, 128, 0, stream>>>(w);
        CDP_CHECK();

        /*
         * kg_rows: A*s1+s2를 계산하고 t1, t0로 분리해 패킹.
         * 블록 K개: 행렬의 행 하나씩 담당.
         * 블록당 128스레드: 계수 2개씩 곱셈을 처리하고 invNTT의 butterfly를 하나씩 담당.
         */

        kg_rows<<<K, 128, 0, stream>>>(w);
        CDP_CHECK();

        /*
         * kg_tr: 공개키의 해시 tr을 계산해 비밀키에 저장.
         * 블록 1개: 공개키 하나의 해시를 계산.
         * 블록당 25스레드: SHAKE의 25개 lane을 하나씩 담당.
         */

        kg_tr<<<1, 25, 0, stream>>>(w);
        CDP_CHECK();

        record(&w->cdp_error, cudaStreamDestroy(stream));
    }

    /**************************************************
     * 이름: sign_decide
     *
     * 설명: Sign 판정 커널 (sign_attempt 가 tail launch)
     *      이번 시도의 실패 수 (flag) 와 힌트 총수 (≤ OMEGA) 로 판정해,
     *      실패하면 다음 시도 (sign_attempt) 를 tail launch, 통과면 sg_pack 으로 서명 패킹
     *
     * 인수: - w: Sign 작업 공간
     **************************************************/
    __global__ void sign_decide(sgm_ws *w)
    {
        if (blockIdx.x || threadIdx.x || w->cdp_error)
            return;
        const int par = (w->iters - 1) & 1;
        unsigned count = 0;

        // Hint가 1인 계수의 개수 합산
        for (int k = 0; k < K; ++k)
            count += w->nh_row[par][k];
        // flag[par] -> 서명 과정의 Norm 검사에서 rejection
        // count > OMEGA -> Hint과정에서 rejection
        if (w->flag[par] || count > OMEGA)
        {
            /*
             * sign_attempt: 새 nonce로 서명 후보 생성 시작.
             * 블록 1개, 스레드 1개: 상태 갱신과 자식 커널 호출을 한 번만 수행.
             * tail launch로 앞선 작업이 끝난 뒤 실행.
             */

            sign_attempt<<<1, 1, 0, cudaStreamTailLaunch>>>(w);
            record(&w->cdp_error, cudaGetLastError());
        }
        else
        {
            CDP_STREAM();
            /*
             * sg_pack: z와 h를 패킹해 서명 완성.
             * 블록 K개: 힌트 다항식 하나씩 담당. z 패킹은 전체 스레드가 분담.
             * 블록당 128스레드: 힌트 계수 2개씩 처리.
             */
            sg_pack<<<K, 128, 0, stream>>>(w);
            CDP_CHECK();
            record(&w->cdp_error, cudaStreamDestroy(stream));
        }
    }

    /**************************************************
     * 이름: sign_attempt
     *
     * 설명: Sign 시도 1회 (sign_parent 또는 sign_decide 가 tail launch)
     *      판정 칸 초기화, iters·nonce 증가 후 자식 등록:
     *      expand_y → ntt_vector(y) → sg_rows → sg_hash_challenge → challenge → ntt_vector(c) → sg_checks
     *      마지막에 sign_decide 를 tail launch (이 커널과 자식이 모두 끝난 뒤 실행)
     *
     * 인수: - w: Sign 작업 공간
     **************************************************/
    __global__ void sign_attempt(sgm_ws *w)
    {
        if (blockIdx.x || threadIdx.x || w->cdp_error)
            return;
        const int par = w->iters & 1;
        w->flag[par] = 0;
        for (int k = 0; k < K; ++k)
            w->nh_row[par][k] = 0;
        ++w->iters;
        const uint16_t nonce = w->cdp_nonce++;
        CDP_STREAM();
        /*
         * expand_y: rhoprime과 nonce로 마스킹 벡터 y 샘플링.
         * 블록 L개: y의 다항식 하나씩 담당.
         * 블록당 25스레드: SHAKE의 25개 lane을 하나씩 담당.
         */
        expand_y<<<L, 25, 0, stream>>>(&w->y, w->rhoprime, nonce);
        CDP_CHECK();
        /*
         * ntt_vector: y를 NTT로 변환해 yhat에 저장.
         * 블록 L개: y의 다항식 하나씩 담당.
         * 블록당 128스레드: NTT 단계마다 128개 butterfly를 하나씩 담당.
         */
        ntt_vector<<<L, 128, 0, stream>>>(w->yhat.vec, w->y.vec);
        CDP_CHECK();
        /*
         * sg_rows: A*y를 계산하고 w1, w0로 분리한 뒤 w1 인코딩.
         * 블록 K개: 행렬의 행 하나씩 담당.
         * 블록당 128스레드: 계수 2개씩 곱셈을 처리하고 invNTT의 butterfly를 하나씩 담당.
         */
        sg_rows<<<K, 128, 0, stream>>>(w);
        CDP_CHECK();
        /*
         * sg_hash_challenge: mu와 인코딩된 w1을 해시해 c_tilde 생성.
         * 블록 1개: 전체 입력을 하나의 해시로 계산.
         * 블록당 25스레드: SHAKE의 25개 lane을 하나씩 담당.
         */
        sg_hash_challenge<<<1, 25, 0, stream>>>(w);
        CDP_CHECK();
        /*
         * challenge: c_tilde로 희소 챌린지 다항식 c 생성.
         * 블록 1개: 챌린지 다항식 하나를 생성.
         * 블록당 25스레드: SHAKE를 분담하고, 계수 선택은 스레드 0이 처리.
         */
        challenge<<<1, 25, 0, stream>>>(&w->cp, w->sig);
        CDP_CHECK();
        /*
         * ntt_vector: c를 NTT로 변환해 cphat에 저장.
         * 블록 1개: 챌린지 다항식 하나를 변환.
         * 블록당 128스레드: NTT 단계마다 128개 butterfly를 하나씩 담당.
         */
        ntt_vector<<<1, 128, 0, stream>>>(&w->cphat, &w->cp);
        CDP_CHECK();
        /*
         * sg_checks: z, w0-c*s2, c*t0의 노름을 검사하고 힌트와 기각 정보 생성.
         * 블록 L+K개: z의 L개 다항식과 힌트의 K개 행을 각각 담당.
         * 블록당 128스레드: 계수 2개씩 처리하고 invNTT의 butterfly를 하나씩 담당.
         */
        sg_checks<<<L + K, 128, 0, stream>>>(w);
        CDP_CHECK();
        record(&w->cdp_error, cudaStreamDestroy(stream));
        /*
         * sign_decide: 검사 결과와 힌트 수를 보고 재시도 또는 서명 패킹 결정.
         * 블록 1개, 스레드 1개: 힌트 수 합산과 분기 결정을 한 번만 수행.
         * tail launch로 앞선 검사가 끝난 뒤 실행.
         */
        sign_decide<<<1, 1, 0, cudaStreamTailLaunch>>>(w);
        record(&w->cdp_error, cudaGetLastError());
    }

    /**************************************************
     * 이름: sign_parent
     *
     * 설명: Sign 부모 커널 (호스트가 <<<1, 1>>> 로 실행)
     *      메시지 길이 검사 후 sg_hash_prepare → expand_a → sg_unpack 을 등록하고,
     *      첫 시도 (sign_attempt) 를 tail launch
     *
     * 인수: - w: Sign 작업 공간 (sk·msg·mlen 입력, sig·iters 출력)
     **************************************************/
    __global__ void sign_parent(sgm_ws *w)
    {
        if (blockIdx.x || threadIdx.x)
            return;
        w->cdp_error = 0;
        w->iters = 0;
        w->cdp_nonce = 0;
        if (w->mlen < 0 || w->mlen > SG_MAXM)
        {
            w->cdp_error = int(cudaErrorInvalidValue);
            return;
        }
        CDP_STREAM();
        /*
         * sg_hash_prepare: tr과 메시지로 mu를, key와 rnd, mu로 rhoprime을 계산.
         * 블록 1개: 두 해시를 순서대로 계산.
         * 블록당 25스레드: SHAKE의 25개 lane을 하나씩 담당.
         */
        sg_hash_prepare<<<1, 25, 0, stream>>>(w);
        CDP_CHECK();
        /*
         * expand_a: rho로 행렬 A 샘플링.
         * 블록 K*L개: 행렬의 다항식 하나씩 담당.
         * 블록당 25스레드: SHAKE의 25개 lane을 하나씩 담당.
         */
        expand_a<<<K * L, 25, 0, stream>>>(w->mat, w->sk);
        CDP_CHECK();
        /*
         * sg_unpack: 비밀키에서 s1, s2, t0를 복원하고 NTT 변환.
         * 블록 L+2*K개: s1, s2, t0의 다항식 하나씩 담당.
         * 블록당 128스레드: NTT 단계마다 128개 butterfly를 하나씩 담당.
         */
        sg_unpack<<<L + 2 * K, 128, 0, stream>>>(w);
        CDP_CHECK();
        record(&w->cdp_error, cudaStreamDestroy(stream));
        /*
         * sign_attempt: 새 nonce로 서명 후보 생성 시작.
         * 블록 1개, 스레드 1개: 상태 갱신과 자식 커널 호출을 한 번만 수행.
         * tail launch로 앞선 작업이 끝난 뒤 실행.
         */
        sign_attempt<<<1, 1, 0, cudaStreamTailLaunch>>>(w);
        record(&w->cdp_error, cudaGetLastError());
    }

    /**************************************************
     * 이름: verify_parent
     *
     * 설명: Verify 부모 커널 (호스트가 <<<1, 1>>> 로 실행). 결과를 −1 로 둔 뒤 자식을 순서대로 등록:
     *      expand_a → vf_hash_prepare → challenge → vf_unpack → vf_ntt → vf_rows → vf_hash_compare
     *
     * 인수: - w: Verify 작업 공간 (pk·sig·msg·mlen 입력, result 출력)
     **************************************************/
    __global__ void verify_parent(vfm_ws *w)
    {
        if (blockIdx.x || threadIdx.x)
            return;
        w->cdp_error = 0;
        w->bad = 0;
        w->result = -1;
        if (w->mlen < 0 || w->mlen > VF_MAXM)
        {
            w->cdp_error = int(cudaErrorInvalidValue);
            return;
        }
        CDP_STREAM();
        /*
         * expand_a: rho로 행렬 A 샘플링.
         * 블록 K*L개: 행렬의 다항식 하나씩 담당.
         * 블록당 25스레드: SHAKE의 25개 lane을 하나씩 담당.
         */
        expand_a<<<K * L, 25, 0, stream>>>(w->mat, w->pk);
        CDP_CHECK();
        /*
         * vf_hash_prepare: 공개키로 tr을, tr과 메시지로 mu를 계산.
         * 블록 1개: 두 해시를 순서대로 계산.
         * 블록당 25스레드: SHAKE의 25개 lane을 하나씩 담당.
         */
        vf_hash_prepare<<<1, 25, 0, stream>>>(w);
        CDP_CHECK();
        /*
         * challenge: c_tilde로 희소 챌린지 다항식 c 생성.
         * 블록 1개: 챌린지 다항식 하나를 생성.
         * 블록당 25스레드: SHAKE를 분담하고, 계수 선택은 스레드 0이 처리.
         */
        challenge<<<1, 25, 0, stream>>>(&w->cp, w->sig);
        CDP_CHECK();
        /*
         * vf_unpack: 서명에서 z, h를 복원하고 노름과 힌트 형식 검사.
         * 블록 L개: z의 다항식 하나씩 담당. 힌트는 첫 블록의 스레드 0이 처리.
         * 블록당 128스레드: 256개 계수를 2개씩 복원하고 검사.
         */
        vf_unpack<<<L, 128, 0, stream>>>(w);
        CDP_CHECK();
        /*
         * vf_ntt: z, t1*2^D, c를 NTT로 변환.
         * 블록 L+K+1개: z, t1, c의 다항식 하나씩 담당.
         * 블록당 128스레드: NTT 단계마다 128개 butterfly를 하나씩 담당.
         */
        vf_ntt<<<L + K + 1, 128, 0, stream>>>(w);
        CDP_CHECK();
        /*
         * vf_rows: A*z-c*(t1*2^D)를 계산하고 힌트로 w1을 복원해 인코딩.
         * 블록 K개: 행렬의 행 하나씩 담당.
         * 블록당 128스레드: 계수 2개씩 처리하고 invNTT의 butterfly를 하나씩 담당.
         */
        vf_rows<<<K, 128, 0, stream>>>(w);
        CDP_CHECK();
        /*
         * vf_hash_compare: mu와 복원한 w1의 해시를 c_tilde와 비교해 검증 결과 저장.
         * 블록 1개: 챌린지 해시 하나를 계산.
         * 블록당 25스레드: SHAKE를 분담하고, 최종 비교는 스레드 0이 처리.
         */
        vf_hash_compare<<<1, 25, 0, stream>>>(w);
        CDP_CHECK();
        record(&w->cdp_error, cudaStreamDestroy(stream));
    }
} // namespace cdp
