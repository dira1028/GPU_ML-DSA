#include "SHA3.cuh"
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>
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

    __device__ __forceinline__ Word rotl64(Word a, int n)
    {
        return (a << n) | (a >> ((64 - n) & 63));
    }
    struct LaneIndices
    {
        int left, right, s1, s2, s3, s4, chi1, chi2, rho, pi;
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

    // [GPU 초기화] shared state[25]와 pos를 초기화한다. 25스레드 공동 호출
    __device__ void keccak_inc_init(Word *state, uint32_t *pos)
    {
        int tid = threadIdx.x;
        state[tid] = 0;
        if (tid == 0)
            *pos = 0;
        __syncthreads();
    }

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

    __device__ void shake128_inc_init(shakeincctx *state) { keccak_inc_init(state->ctx, &state->pos); }
    __device__ void shake128_inc_absorb(shakeincctx *state, const uint8_t *input, size_t inlen) { keccak_inc_absorb(state->ctx, SHAKE128_BLOCK_BYTES, input, inlen, &state->pos); }
    __device__ void shake128_inc_finalize(shakeincctx *state) { keccak_inc_finalize(state->ctx, SHAKE128_BLOCK_BYTES, 0x1F, &state->pos); }
    __device__ void shake128_inc_squeeze(uint8_t *output, size_t outlen, shakeincctx *state) { keccak_inc_squeeze(output, outlen, state->ctx, SHAKE128_BLOCK_BYTES, &state->pos); }

    __device__ void shake256_inc_init(shakeincctx *state) { keccak_inc_init(state->ctx, &state->pos); }
    __device__ void shake256_inc_absorb(shakeincctx *state, const uint8_t *input, size_t inlen) { keccak_inc_absorb(state->ctx, SHAKE256_BLOCK_BYTES, input, inlen, &state->pos); }
    __device__ void shake256_inc_finalize(shakeincctx *state) { keccak_inc_finalize(state->ctx, SHAKE256_BLOCK_BYTES, 0x1F, &state->pos); }
    __device__ void shake256_inc_squeeze(uint8_t *output, size_t outlen, shakeincctx *state) { keccak_inc_squeeze(output, outlen, state->ctx, SHAKE256_BLOCK_BYTES, &state->pos); }

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

    __device__ void lane25_stream128_squeezeblocks(uint8_t *output, size_t outblocks, shakeincctx *state)
    {
        shake128_inc_squeeze(output, outblocks * SHAKE128_BLOCK_BYTES, state);
    }

    __device__ void lane25_stream128_inc_ctx_release(shakeincctx *)
    {
    }

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

    __device__ void lane25_stream256_squeezeblocks(uint8_t *output, size_t outblocks, shakeincctx *state)
    {
        shake256_inc_squeeze(output, outblocks * SHAKE256_BLOCK_BYTES, state);
    }

    __device__ void lane25_stream256_inc_ctx_release(shakeincctx *)
    {
    }

}
