#include "SHA3.cuh"
#include <cuda_runtime.h>

namespace SHA3_CUDA
{
    __device__ __constant__ Word ROUND_CONSTANTS[ROUNDS] = {
        0x0000000000000001ULL, 0x0000000000008082ULL,
        0x800000000000808aULL, 0x8000000080008000ULL,
        0x000000000000808bULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL,
        0x000000000000008aULL, 0x0000000000000088ULL,
        0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL,
        0x8000000000008089ULL, 0x8000000000008003ULL,
        0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800aULL, 0x800000008000000aULL,
        0x8000000080008081ULL, 0x8000000000008080ULL,
        0x0000000080000001ULL, 0x8000000080008008ULL};

    __device__ __constant__ int ROTATIONS[25] = {
        0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43,
        25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14};

    __device__ __constant__ int PI_INV[25] = {
        0, 6, 12, 18, 24, 3, 9, 10, 16, 22, 1, 7, 13,
        19, 20, 4, 5, 11, 17, 23, 2, 8, 14, 15, 21};

    /**************************************************
     * 이름: rotl64
     *
     * 설명: 64비트 값의 왼쪽 n 비트 회전
     *      n = 0 일 때도 정의되도록 오른쪽 이동량을 (64−n)&63 으로 계산
     *
     * 인수: - a: 64비트 lane 값
     *      - n: 회전량 0..63
     *
     * 회전한 값을 반환
     **************************************************/
    __device__ __forceinline__ Word rotl64(Word a, int n)
    {
        return (a << n) | (a >> ((64 - n) & 63));
    }
    struct LaneIndices
    {
        int left, right, s1, s2, s3, s4, chi1, chi2, rho, pi;

        /**************************************************
         * 이름: LaneIndices::LaneIndices
         *
         * 설명: 스레드 t 가 맡는 lane (x, y) = (t%5, t/5) 에 대해 Keccak 라운드에서 필요한
         *      상대 lane 번호와 회전량의 사전 계산
         *      left/right: θ 의 열 x−1, x+1 / s1..s4: 같은 열의 다른 4행 /
         *      chi1, chi2: χ 의 (x+1, y), (x+2, y) / rho: ρ 회전량 / pi: π 에서 값을 가져올 lane
         *
         * 인수: - t: lane 번호 0..24 (= threadIdx.x)
         **************************************************/
        // [GPU 생성자] 라운드에서 사용할 상대 스레드 번호와 회전량을 한 번 계산
        __device__ explicit LaneIndices(int t)
        {
            const int x = t % 5, y = t / 5;
            left = (x + 4) % 5;
            right = (x + 1) % 5;
            s1 = x + 5 * ((y + 1) % 5);
            s2 = x + 5 * ((y + 2) % 5);
            s3 = x + 5 * ((y + 3) % 5);
            s4 = x + 5 * ((y + 4) % 5);
            chi1 = 5 * y + (x + 1) % 5;
            chi2 = 5 * y + (x + 2) % 5;
            rho = ROTATIONS[t];
            pi = PI_INV[t];
        }
    };

    /**************************************************
     * 이름: keccak_f
     *
     * 설명: Keccak-f[1600] 순열 24라운드
     *       25스레드가 lane 하나씩 레지스터에 들고 __shfl_sync 로 θ (열 패리티), ρ (회전), π (lane 이동),
     *       χ, ι (라운드 상수, lane 0) 를 계산한 뒤 결과를 s[t] 에 기록
     *       블록의 25스레드 공동 호출 필수
     *
     * 인수: - s: 공유메모리 상태 (lane 25개)
     **************************************************/
    __device__ __forceinline__ void keccak_f(volatile Word *s)
    {
        const int t = threadIdx.x;
        const LaneIndices p(t);
        Word a = s[t];
        Word column_parity;
#pragma unroll 24
        for (int round = 0; round < ROUNDS; ++round)
        {
            column_parity = a ^ __shfl_sync(LANE_MASK, a, p.s1);
            column_parity ^= __shfl_sync(LANE_MASK, a, p.s2);
            column_parity ^= __shfl_sync(LANE_MASK, a, p.s3);
            column_parity ^= __shfl_sync(LANE_MASK, a, p.s4);
            const Word l = __shfl_sync(LANE_MASK, column_parity, p.left);
            const Word r = __shfl_sync(LANE_MASK, column_parity, p.right);

            a = a ^ l ^ rotl64(r, 1);

            a = rotl64(a, p.rho);
            a = __shfl_sync(0x01FFFFFF, a, p.pi);

            const Word b = __shfl_sync(LANE_MASK, a, p.chi1);
            const Word c = __shfl_sync(LANE_MASK, a, p.chi2);
            a = a ^ ((~b) & c);
            if (t == 0)
                a ^= ROUND_CONSTANTS[round];
        }
        s[t] = a;
    }

    /**************************************************
     * 이름: keccak_inc_init
     *
     * 설명: 증분 SHAKE 상태를 0 으로, pos 를 0 으로 초기화
     *      25스레드 공동 호출
     *
     * 인수: - state: 공유메모리 상태 (lane 25개)
     *      - pos: 현재 블록 안 바이트 위치
     **************************************************/
    // [GPU 초기화] shared state[25]와 pos를 초기화한다. 25스레드 공동 호출
    __device__ void keccak_inc_init(Word *state, uint32_t *pos)
    {
        int tid = threadIdx.x;
        state[tid] = 0;
        if (tid == 0)
            *pos = 0;
        __syncthreads();
    }

    /**************************************************
     * 이름: keccak_inc_absorb
     *
     * 설명: 입력 m 을 상태에 흡수
     *      각 스레드는 자기 lane 이 맡는 바이트 범위 [8t, 8t+8) 와 이번 구간이 겹치는 부분만 XOR
     *      rate 를 다 채우면 keccak_f 실행
     *      25스레드 공동 호출
     *
     * 인수: - state: 공유메모리 상태
     *      - rate: 블록 크기 (SHAKE128 168, SHAKE256 136 바이트)
     *      - m: 입력
     *      - mlen: 입력 길이
     *      - pos: 현재 블록 안 바이트 위치 (갱신)
     **************************************************/
    // [GPU 25-lane 함수] keccak_inc_absorb; 상태/입출력과 길이를 받아 갱신
    __device__ void keccak_inc_absorb(
        Word *state,
        uint32_t rate,
        const uint8_t *m,
        size_t mlen,
        uint32_t *pos)
    {
        const int tid = threadIdx.x;

        while (mlen > 0)
        {
            uint32_t cur_pos = *pos;
            size_t len = rate - cur_pos;
            if (mlen < len)
            {
                len = mlen;
            }

            /* --------------------------------------------------------
             * 25개 스레드가 각자 담당하는 레인(state[tid])의 바이트를 병렬 처리
             * 각 레인 tid는 바이트 오프셋 [tid * 8, tid * 8 + 7]을 담당
             * -------------------------------------------------------- */
            const uint32_t lane_byte_start = tid * 8;
            const uint32_t lane_byte_end = lane_byte_start + 8;

            const uint32_t start = (cur_pos > lane_byte_start) ? cur_pos : lane_byte_start;
            const uint32_t end = (cur_pos + (uint32_t)len < lane_byte_end) ? (cur_pos + (uint32_t)len) : lane_byte_end;

            if (start < end)
            {
                Word lane_val = 0;
#pragma unroll
                for (uint32_t b = start; b < end; b++)
                {
                    lane_val |= ((Word)m[b - cur_pos]) << (8 * (b & 0x07));
                }
                state[tid] ^= lane_val;
            }

            // 모든 스레드가 이전 pos 읽기와 lane 작업을 마친 후 pos를 갱신한다.
            __syncthreads();
            if (tid == 0)
            {
                *pos = cur_pos + (uint32_t)len;
            }
            __syncthreads();

            m += len;
            mlen -= len;

            /* rate를 모두 채웠으면 Keccak-f 실행 */
            if (*pos == rate)
            {
                keccak_f(state);
                __syncthreads();

                if (tid == 0)
                {
                    *pos = 0;
                }
                __syncthreads();
            }
        }
    }

    /**************************************************
     * 이름: keccak_inc_finalize
     *
     * 설명: 패딩 추가와 흡수 종료
     *      현재 위치에 도메인 바이트 p, rate 의 마지막 바이트에 0x80 을 XOR 하고,
     *      다음 squeeze 가 순열부터 실행되도록 pos = rate 로 설정
     *      스레드 0 이 기록한 뒤 전체 동기화
     *
     * 인수: - state: 공유메모리 상태
     *      - rate: 블록 크기
     *      - p: 도메인 구분 바이트 (SHAKE 는 0x1F)
     *      - pos: 현재 블록 안 바이트 위치 (갱신)
     **************************************************/
    // [GPU 25-lane 함수] keccak_inc_finalize; 상태/입출력과 길이를 받아 갱신한다. 25스레드 공동 호출, 단일 SHAKE 실행에 포함.
    __device__ void keccak_inc_finalize(
        Word *state,
        uint32_t rate,
        uint8_t p,
        uint32_t *pos)
    {
        if (threadIdx.x == 0)
        {
            state[*pos >> 3] ^= ((Word)p) << (8 * (*pos & 0x07));
            state[(rate - 1) >> 3] ^= ((Word)0x80) << (8 * ((rate - 1) & 0x07));

            *pos = rate;
        }
        __syncthreads();
    }

    /**************************************************
     * 이름: keccak_inc_squeeze
     *
     * 설명: 출력 outlen 바이트 추출
     *       pos == rate 이면 keccak_f 로 새 블록을 만들고, 각 스레드는 자기 lane 의 바이트를 h 에 기록
     *       이어서 호출하면 앞 출력의 다음 바이트부터 출력. 25스레드 공동 호출
     *
     * 인수: - h: 출력 버퍼
     *      - outlen: 출력 길이
     *      - state: 공유메모리 상태
     *      - rate: 블록 크기
     *      - pos: 현재 블록 안 바이트 위치 (갱신)
     **************************************************/
    // [GPU 병렬 출력] 25스레드가 각 lane의 바이트를 h에 기록한다. 상태/pos를 갱신하며 전체 SHAKE 시간에 포함한다.
    __device__ void keccak_inc_squeeze(
        uint8_t *h,
        size_t outlen,
        Word *state,
        uint32_t rate,
        uint32_t *pos)
    {
        const int tid = threadIdx.x;

        while (outlen > 0)
        {
            if (*pos == rate)
            {
                keccak_f(state);
                __syncthreads();
                if (tid == 0)
                {
                    *pos = 0;
                }
                __syncthreads();
            }

            uint32_t cur_pos = *pos;
            size_t available = rate - cur_pos;
            // ext : 이번 while문에서 추출할 바이트 수.
            // 추출 길이와, rate중 남은 값 비교하여 작은값으로.
            size_t ext = (outlen < available) ? outlen : available;

            /* --------------------------------------------------------
             * 25개 스레드가 각자 담당하는 레인(state[tid])에서
             * 출력 배열 h로 바이트를 병렬 추출 (Absorb와 동일한 구조)
             * -------------------------------------------------------- */

            const uint32_t lane_byte_start = tid * 8;
            const uint32_t lane_byte_end = lane_byte_start + 8;

            const uint32_t start = (cur_pos > lane_byte_start) ? cur_pos : lane_byte_start;
            const uint32_t end = (cur_pos + (uint32_t)ext < lane_byte_end) ? (cur_pos + (uint32_t)ext) : lane_byte_end;
            // 현재 추출 구간과 lane의 교집합만 기록하므로 마지막 부분 lane도 처리한다.

            if (start < end)
            {
#pragma unroll
                for (uint32_t b = start; b < end; b++)
                {
                    // b - cur_pos는 현재 루프에서 h 배열의 저장할 상대적 인덱스
                    // SHAKE128 출력은 rate=168 범위이므로 활성 바이트 위치는 0~167이다.
                    // b의 하위 3비트는 64비트 lane 안의 바이트 위치(0~7)이다.
                    h[b - cur_pos] = (uint8_t)(state[tid] >> (8 * (b & 0x07)));
                }
            }

            /* 0번 스레드가 전역 pos 포인터 갱신 */
            // 모든 스레드가 이전 pos 읽기와 lane 작업을 마친 후 pos를 갱신한다.
            __syncthreads();
            if (tid == 0)
            {
                *pos = cur_pos + (uint32_t)ext;
            }
            __syncthreads();

            h += ext;
            outlen -= ext;
        }
    }

    /**************************************************
     * 이름: shake128_inc_init
     *
     * 설명: SHAKE128 증분 상태 초기화 (keccak_inc_init). 25스레드 공동 호출
     *
     * 인수: - state: SHAKE 상태 (공유메모리)
     **************************************************/
    __device__ void shake128_inc_init(shakeincctx *state) { keccak_inc_init(state->ctx, &state->pos); }

    /**************************************************
     * 이름: shake128_inc_absorb
     *
     * 설명: SHAKE128 (rate 168) 입력 흡수. 25스레드 공동 호출
     *
     * 인수: - state: SHAKE 상태
     *      - input: 입력
     *      - inlen: 입력 길이
     **************************************************/
    __device__ void shake128_inc_absorb(shakeincctx *state, const uint8_t *input, size_t inlen) { keccak_inc_absorb(state->ctx, SHAKE128_BLOCK_BYTES, input, inlen, &state->pos); }

    /**************************************************
     * 이름: shake128_inc_finalize
     *
     * 설명: SHAKE128 패딩 (0x1F … 0x80) 추가와 흡수 종료. 25스레드 공동 호출
     *
     * 인수: - state: SHAKE 상태
     **************************************************/
    __device__ void shake128_inc_finalize(shakeincctx *state) { keccak_inc_finalize(state->ctx, SHAKE128_BLOCK_BYTES, 0x1F, &state->pos); }

    /**************************************************
     * 이름: shake128_inc_squeeze
     *
     * 설명: SHAKE128 출력 outlen 바이트 생성. 25스레드 공동 호출
     *
     * 인수: - output: 출력 버퍼
     *      - outlen: 출력 길이
     *      - state: SHAKE 상태
     **************************************************/
    __device__ void shake128_inc_squeeze(uint8_t *output, size_t outlen, shakeincctx *state) { keccak_inc_squeeze(output, outlen, state->ctx, SHAKE128_BLOCK_BYTES, &state->pos); }

    /**************************************************
     * 이름: shake256_inc_init
     *
     * 설명: SHAKE256 증분 상태 초기화 (keccak_inc_init). 25스레드 공동 호출
     *
     * 인수: - state: SHAKE 상태 (공유메모리)
     **************************************************/
    __device__ void shake256_inc_init(shakeincctx *state) { keccak_inc_init(state->ctx, &state->pos); }

    /**************************************************
     * 이름: shake256_inc_absorb
     *
     * 설명: SHAKE256 (rate 136) 입력 흡수. 25스레드 공동 호출
     *
     * 인수: - state: SHAKE 상태
     *      - input: 입력
     *      - inlen: 입력 길이
     **************************************************/
    __device__ void shake256_inc_absorb(shakeincctx *state, const uint8_t *input, size_t inlen) { keccak_inc_absorb(state->ctx, SHAKE256_BLOCK_BYTES, input, inlen, &state->pos); }

    /**************************************************
     * 이름: shake256_inc_finalize
     *
     * 설명: SHAKE256 패딩 (0x1F … 0x80) 추가와 흡수 종료. 25스레드 공동 호출
     *
     * 인수: - state: SHAKE 상태
     **************************************************/
    __device__ void shake256_inc_finalize(shakeincctx *state) { keccak_inc_finalize(state->ctx, SHAKE256_BLOCK_BYTES, 0x1F, &state->pos); }

    /**************************************************
     * 이름: shake256_inc_squeeze
     *
     * 설명: SHAKE256 출력 outlen 바이트 생성. 25스레드 공동 호출
     *
     * 인수: - output: 출력 버퍼
     *      - outlen: 출력 길이
     *      - state: SHAKE 상태
     **************************************************/
    __device__ void shake256_inc_squeeze(uint8_t *output, size_t outlen, shakeincctx *state) { keccak_inc_squeeze(output, outlen, state->ctx, SHAKE256_BLOCK_BYTES, &state->pos); }

    /**************************************************
     * 이름: lane25_stream128_init
     *
     * 설명: ExpandA 용 SHAKE128 스트림 시작
     *      seed ‖ nonce 하위 바이트 ‖ 상위 바이트 흡수 후 finalize
     *      25스레드 공동 호출
     *
     * 인수: - state: SHAKE 상태
     *      - seed: ρ (32바이트)
     *      - nonce: (행 << 8) + 열
     **************************************************/
    __device__ void lane25_stream128_init(shakeincctx *state, const uint8_t seed[SEEDBYTES], uint16_t nonce)
    {
        uint8_t t[2];
        t[0] = (uint8_t)nonce;
        t[1] = (uint8_t)(nonce >> 8);

        shake128_inc_init(state);
        shake128_inc_absorb(state, seed, SEEDBYTES);
        shake128_inc_absorb(state, t, 2);
        shake128_inc_finalize(state);
    }

    /**************************************************
     * 이름: lane25_stream128_squeezeblocks
     *
     * 설명: SHAKE128 블록 (168바이트) outblocks 개 출력. 25스레드 공동 호출
     *
     * 인수: - output: 출력 버퍼 (168·outblocks 바이트)
     *      - outblocks: 블록 수
     *      - state: SHAKE 상태
     **************************************************/
    __device__ void lane25_stream128_squeezeblocks(uint8_t *output, size_t outblocks, shakeincctx *state)
    {
        shake128_inc_squeeze(output, outblocks * SHAKE128_BLOCK_BYTES, state);
    }

    __device__ void lane25_stream128_inc_ctx_release(shakeincctx *)
    {
    }

    /**************************************************
     * 이름: lane25_stream256_init
     *
     * 설명: ExpandS / ExpandMask 용 SHAKE256 스트림 시작
     *       seed ‖ nonce (2바이트, 하위 바이트 먼저) 흡수 후 finalize
     *       25스레드 공동 호출
     *
     * 인수: - state: SHAKE 상태
     *      - seed: ρ' 또는 ρ'' (64바이트)
     *      - nonce: ExpandS 는 다항식 번호, ExpandMask 는 L·nonce + l
     **************************************************/
    __device__ void lane25_stream256_init(shakeincctx *state, const uint8_t seed[CRHBYTES], uint16_t nonce)
    {
        uint8_t t[2];
        t[0] = (uint8_t)nonce;
        t[1] = (uint8_t)(nonce >> 8);

        shake256_inc_init(state);
        shake256_inc_absorb(state, seed, CRHBYTES);
        shake256_inc_absorb(state, t, 2);
        shake256_inc_finalize(state);
    }

    /**************************************************
     * 이름: lane25_stream256_squeezeblocks
     *
     * 설명: SHAKE256 블록 (136바이트) outblocks 개 출력. 25스레드 공동 호출
     *
     * 인수: - output: 출력 버퍼 (136·outblocks 바이트)
     *      - outblocks: 블록 수
     *      - state: SHAKE 상태
     **************************************************/
    __device__ void lane25_stream256_squeezeblocks(uint8_t *output, size_t outblocks, shakeincctx *state)
    {
        shake256_inc_squeeze(output, outblocks * SHAKE256_BLOCK_BYTES, state);
    }

    __device__ void lane25_stream256_inc_ctx_release(shakeincctx *)
    {
    }

}