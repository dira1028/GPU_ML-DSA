#pragma once
#include <cuda_runtime.h>
#include "SHA3.cuh"
#include "../common/keygen.cuh"
#include "../common/sign.cuh"
#include "../common/verify.cuh"

namespace cdp {
__device__ void hash256(SHA3_CUDA::shakeincctx *s, uint8_t *out, size_t outlen,
                            const uint8_t *in, size_t inlen);

__global__ void kg_seed(kgm_ws *w);

__global__ void kg_tr(kgm_ws *w);

__global__ void sg_hash_prepare(sgm_ws *w);

__global__ void sg_hash_challenge(sgm_ws *w);

__global__ void vf_hash_prepare(vfm_ws *w);

__global__ void vf_hash_compare(vfm_ws *w);
} // namespace cdp
