#include "sampling.cuh"
#include "SHA3.cuh"

namespace cdp
{
    /**************************************************
     * 이름: reject_uniform
     *
     * 설명: 3바이트씩 23비트 값을 읽어 Q 미만이면 계수로 채택
     *
     * 인수: - a: 출력 계수 배열
     *      - n: 최대로 채울 계수 수
     *      - p: SHAKE128 출력 바이트
     *      - len: p 의 길이
     *
     * 채택한 계수 개수를 반환
     **************************************************/
    __device__ unsigned reject_uniform(int32_t *a, unsigned n, const uint8_t *p, unsigned len)
    {
        unsigned count = 0;
        for (unsigned j = 0; j + 3 <= len && count < n; j += 3)
        {
            uint32_t v = (p[j] | (uint32_t(p[j + 1]) << 8) | (uint32_t(p[j + 2]) << 16)) & 0x7fffff;
            if (v < Q)
                a[count++] = v;
        }
        return count;
    }

    /**************************************************
     * 이름: reject_eta
     *
     * 설명: 바이트마다 4비트 값 두 개를 읽어 15 미만이면 2 − (x mod 5) 를 계수로 채택
     *
     * 인수: - a: 출력 계수 배열
     *      - n: 최대로 채울 계수 수
     *      - p: SHAKE256 출력 바이트
     *      - len: p 의 길이
     *
     * 채택한 계수 개수를 반환
     **************************************************/
    __device__ unsigned reject_eta(int32_t *a, unsigned n, const uint8_t *p, unsigned len)
    {
        unsigned count = 0;
        for (unsigned j = 0; j < len && count < n; ++j)
        {
            unsigned x = p[j] & 15, y = p[j] >> 4;
            if (x < 15)
                a[count++] = 2 - int(x - (205 * x >> 10) * 5);
            if (y < 15 && count < n)
                a[count++] = 2 - int(y - (205 * y >> 10) * 5);
        }
        return count;
    }

    /**************************************************
     * 이름: expand_a
     *
     * 설명: 자식 커널 <<<K·L = 56, 25>>>. 블록 b 가 행렬 원소 Â[r][c] (r = b/L, c = b%L) 하나를 생성
     *      SHAKE128(ρ ‖ nonce = (r<<8)+c) 에서 840바이트를 먼저 추출하고, 부족하면 168바이트씩 추가 추출
     *      남은 1~2바이트를 앞으로 옮겨 연결. 기각 샘플링은 스레드 0 담당
     *
     * 인수: - mat: 출력 행렬 Â (K × L 다항식, NTT 영역)
     *      - rho: ρ (32바이트. KeyGen 은 작업 공간의 rho, Sign 은 sk, Verify 는 pk 의 앞 32바이트)
     **************************************************/
    __global__ void expand_a(polyvecl *mat, const uint8_t *rho)
    {
        const unsigned r = blockIdx.x / L, c = blockIdx.x % L;
        __shared__ SHA3_CUDA::shakeincctx state;
        __shared__ uint8_t bytes[5 * 168 + 2];
        __shared__ unsigned ctr, off, buflen;

        // SHA3는 25개 스레드가 병렬로 해시
        // 나머지 파트는 Thread 0이 수행
        SHA3_CUDA::lane25_stream128_init(&state, rho, uint16_t((r << 8) + c));
        SHA3_CUDA::lane25_stream128_squeezeblocks(bytes, 5, &state);
        if (threadIdx.x == 0)
        {
            buflen = 5 * 168;
            ctr = reject_uniform(mat[r].vec[c].coeffs, N, bytes, buflen);
        }
        __syncthreads();
        while (ctr < N)
        {
            if (threadIdx.x == 0)
            {
                off = buflen % 3;
                for (unsigned j = 0; j < off; ++j)
                    bytes[j] = bytes[buflen - off + j];
            }
            __syncthreads();
            SHA3_CUDA::lane25_stream128_squeezeblocks(bytes + off, 1, &state);
            if (threadIdx.x == 0)
            {
                buflen = 168 + off;
                ctr += reject_uniform(mat[r].vec[c].coeffs + ctr, N - ctr, bytes, buflen);
            }
            __syncthreads();
        }
    }

    /**************************************************
     * 이름: expand_s
     *
     * 설명: 자식 커널 <<<L+K = 15, 25>>>. 블록 j 가 비밀 다항식 하나를 생성 (j < L: s1[j], 그 외 s2[j−L])
     *      SHAKE256(ρ' ‖ nonce = j) 에서 136바이트씩 추출하며 η 기각 샘플링
     *
     * 인수: - s1: 출력 s1 (L개)
     *      - s2: 출력 s2 (K개)
     *      - seed: ρ' (64바이트)
     **************************************************/
    __global__ void expand_s(polyvecl *s1, polyveck *s2, const uint8_t *seed)
    {
        unsigned j = blockIdx.x;
        // s1, s2생성
        poly *p = j < L ? &s1->vec[j] : &s2->vec[j - L];
        __shared__ SHA3_CUDA::shakeincctx state;
        __shared__ uint8_t bytes[136];
        __shared__ unsigned count;
        SHA3_CUDA::lane25_stream256_init(&state, seed, uint16_t(j));
        if (threadIdx.x == 0)
            count = 0;
        __syncthreads();
        do
        {
            SHA3_CUDA::lane25_stream256_squeezeblocks(bytes, 1, &state);
            if (threadIdx.x == 0)
                count += reject_eta(p->coeffs + count, N - count, bytes, 136);
            __syncthreads();
        } while (count < N);
    }

    /**************************************************
     * 이름: expand_y
     *
     * 설명: 자식 커널 <<<L = 7, 25>>>. Sign 의 마스크 다항식 y[l] (l = 블록 번호) 생성
     *       SHAKE256(ρ'' ‖ nonce = L·nonce + l) 에서 680바이트를 추출하고 앞 640바이트를 polyz_unpack
     *
     * 인수: - y: 출력 y (L개)
     *      - seed: ρ'' (64바이트)
     *      - nonce: 시도 번호 (0, 1, 2, …)
     **************************************************/
    __global__ void expand_y(polyvecl *y, const uint8_t *seed, uint16_t nonce)
    {
        __shared__ SHA3_CUDA::shakeincctx state;
        __shared__ uint8_t bytes[5 * 136];
        SHA3_CUDA::lane25_stream256_init(&state, seed, uint16_t(L * nonce + blockIdx.x));
        SHA3_CUDA::lane25_stream256_squeezeblocks(bytes, 5, &state);
        if (threadIdx.x == 0)
            PQCLEAN_MLDSA87_CLEAN_polyz_unpack(&y->vec[blockIdx.x], bytes);
    }

    /**************************************************
     * 이름: challenge
     *
     * 설명: 자식 커널 <<<1, 25>>>. SampleInBall
     *       SHAKE256(c̃) 의 첫 8바이트로 부호 비트를, 이후 바이트로 위치를 골라 ±1 계수가 TAU 개인 다항식 c 생성
     *       위치 선택은 스레드 0 이 원본 순서대로 처리
     *
     * 인수: - c: 출력 다항식
     *      - seed: c̃ (CTILDEBYTES = 64바이트)
     **************************************************/
    __global__ void challenge(poly *c, const uint8_t *seed)
    {
        __shared__ SHA3_CUDA::shakeincctx state;
        __shared__ uint8_t bytes[136];
        __shared__ unsigned pos, index;
        __shared__ uint64_t signs;
        SHA3_CUDA::shake256_inc_init(&state);
        SHA3_CUDA::shake256_inc_absorb(&state, seed, CTILDEBYTES);
        SHA3_CUDA::shake256_inc_finalize(&state);
        SHA3_CUDA::shake256_inc_squeeze(bytes, 136, &state);
        if (threadIdx.x == 0)
        {
            signs = 0;
            for (unsigned j = 0; j < 8; ++j)
                signs |= uint64_t(bytes[j]) << (8 * j);
            for (unsigned j = 0; j < N; ++j)
                c->coeffs[j] = 0;
            pos = 8;
            index = N - TAU;
        }
        __syncthreads();
        while (index < N)
        {
            if (pos == 136)
            {
                SHA3_CUDA::shake256_inc_squeeze(bytes, 136, &state);
                if (threadIdx.x == 0)
                    pos = 0;
                __syncthreads();
            }
            if (threadIdx.x == 0)
            {
                unsigned b = bytes[pos++];
                if (b <= index)
                {
                    c->coeffs[index] = c->coeffs[b];
                    c->coeffs[b] = 1 - 2 * int(signs & 1);
                    signs >>= 1;
                    ++index;
                }
            }
            __syncthreads();
        }
    }
} // namespace cdp
