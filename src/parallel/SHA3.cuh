#pragma once
#include "params.h"
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

namespace SHA3_CUDA {
using Byte = uint8_t;
using Word = unsigned long long;
static_assert(sizeof(Word) == 8, "64-bit lanes required");
constexpr unsigned LANE_COUNT = 25;
constexpr unsigned LANE_MASK = (1u << LANE_COUNT) - 1;
constexpr int ROUNDS = 24;
constexpr uint32_t SHAKE128_BLOCK_BYTES = 168;
constexpr uint32_t SHAKE256_BLOCK_BYTES = 136;

// Call collectively from a block of 25 threads, using the same state and buffers.
struct shakeincctx { Word ctx[LANE_COUNT]; uint32_t pos; };

__device__ void shake128_inc_init(shakeincctx* state);
__device__ void shake128_inc_absorb(shakeincctx* state,const Byte* input,size_t inlen);
__device__ void shake128_inc_finalize(shakeincctx* state);
__device__ void shake128_inc_squeeze(Byte* output,size_t outlen,shakeincctx* state);

__device__ void shake256_inc_init(shakeincctx *state);
__device__ void shake256_inc_absorb(shakeincctx *state, const Byte *input, size_t inlen);
__device__ void shake256_inc_finalize(shakeincctx *state);
__device__ void shake256_inc_squeeze(Byte *output, size_t outlen, shakeincctx *state);

__device__ void lane25_stream128_init(shakeincctx *state, const Byte seed[SEEDBYTES], uint16_t nonce);
__device__ void lane25_stream128_squeezeblocks(Byte *output, size_t outblocks, shakeincctx *state);
__device__ void lane25_stream256_init(shakeincctx *state, const Byte seed[CRHBYTES], uint16_t nonce);
__device__ void lane25_stream256_squeezeblocks(Byte *output, size_t outblocks, shakeincctx *state);
} // namespace SHA3_CUDA
