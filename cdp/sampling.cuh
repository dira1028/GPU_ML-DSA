#pragma once
#include "SHA3.cuh"
#include "poly.h"

// Exactly 25 threads per block. Every lane participates in every SHAKE call.
// Rejection remains serial to preserve the original byte/acceptance ordering.
namespace cdp {
__device__ unsigned reject_uniform(int32_t* a, unsigned n, const uint8_t* p, unsigned len) {
    unsigned count = 0;
    for (unsigned j = 0; j + 3 <= len && count < n; j += 3) {
        uint32_t v = (p[j] | (uint32_t(p[j+1]) << 8) | (uint32_t(p[j+2]) << 16)) & 0x7fffff;
        if (v < Q) a[count++] = v;
    }
    return count;
}
__device__ unsigned reject_eta(int32_t* a, unsigned n, const uint8_t* p, unsigned len) {
    unsigned count = 0;
    for (unsigned j = 0; j < len && count < n; ++j) {
        unsigned x = p[j] & 15, y = p[j] >> 4;
        if (x < 15) a[count++] = 2 - int(x - (205*x >> 10)*5);
        if (y < 15 && count < n) a[count++] = 2 - int(y - (205*y >> 10)*5);
    }
    return count;
}
__global__ void expand_a(polyvecl* mat, const uint8_t* rho) {
    const unsigned r = blockIdx.x / L, c = blockIdx.x % L;
    __shared__ SHA3_CUDA::shakeincctx state;
    __shared__ uint8_t bytes[5*168 + 2];
    __shared__ unsigned count, off, len;
    SHA3_CUDA::lane25_stream128_init(&state, rho, uint16_t((r << 8) + c));
    SHA3_CUDA::lane25_stream128_squeezeblocks(bytes, 5, &state);
    if (threadIdx.x == 0) {
        len = 5*168;
        count = reject_uniform(mat[r].vec[c].coeffs, N, bytes, len);
    }
    __syncthreads();
    while (count < N) {
        if (threadIdx.x == 0) {
            off = len % 3;
            for (unsigned j = 0; j < off; ++j) bytes[j] = bytes[len-off+j];
        }
        __syncthreads();
        SHA3_CUDA::lane25_stream128_squeezeblocks(bytes + off, 1, &state);
        if (threadIdx.x == 0) {
            len = 168 + off;
            count += reject_uniform(mat[r].vec[c].coeffs + count, N-count, bytes, len);
        }
        __syncthreads();
    }
}
__global__ void expand_s(polyvecl* s1, polyveck* s2, const uint8_t* seed) {
    unsigned j = blockIdx.x;
    poly* p = j < L ? &s1->vec[j] : &s2->vec[j-L];
    __shared__ SHA3_CUDA::shakeincctx state;
    __shared__ uint8_t bytes[136];
    __shared__ unsigned count;
    SHA3_CUDA::lane25_stream256_init(&state, seed, uint16_t(j));
    if (threadIdx.x == 0) count = 0;
    __syncthreads();
    do {
        SHA3_CUDA::lane25_stream256_squeezeblocks(bytes, 1, &state);
        if (threadIdx.x == 0) count += reject_eta(p->coeffs + count, N-count, bytes, 136);
        __syncthreads();
    } while (count < N);
}
__global__ void expand_y(polyvecl* y, const uint8_t* seed, uint16_t nonce) {
    __shared__ SHA3_CUDA::shakeincctx state;
    __shared__ uint8_t bytes[5*136];
    SHA3_CUDA::lane25_stream256_init(&state, seed, uint16_t(L*nonce + blockIdx.x));
    SHA3_CUDA::lane25_stream256_squeezeblocks(bytes, 5, &state);
    if (threadIdx.x == 0) PQCLEAN_MLDSA87_CLEAN_polyz_unpack(&y->vec[blockIdx.x], bytes);
}
__global__ void challenge(poly* c, const uint8_t* seed) {
    __shared__ SHA3_CUDA::shakeincctx state;
    __shared__ uint8_t bytes[136];
    __shared__ unsigned pos, index;
    __shared__ uint64_t signs;
    SHA3_CUDA::shake256_inc_init(&state);
    SHA3_CUDA::shake256_inc_absorb(&state, seed, CTILDEBYTES);
    SHA3_CUDA::shake256_inc_finalize(&state);
    SHA3_CUDA::shake256_inc_squeeze(bytes, 136, &state);
    if (threadIdx.x == 0) {
        signs = 0;
        for (unsigned j = 0; j < 8; ++j) signs |= uint64_t(bytes[j]) << (8*j);
        for (unsigned j = 0; j < N; ++j) c->coeffs[j] = 0;
        pos = 8; index = N-TAU;
    }
    __syncthreads();
    while (index < N) {
        if (pos == 136) {
            SHA3_CUDA::shake256_inc_squeeze(bytes, 136, &state);
            if (threadIdx.x == 0) pos = 0;
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            unsigned b = bytes[pos++];
            if (b <= index) {
                c->coeffs[index] = c->coeffs[b];
                c->coeffs[b] = 1 - 2*int(signs & 1);
                signs >>= 1; ++index;
            }
        }
        __syncthreads();
    }
}
} // namespace cdp
