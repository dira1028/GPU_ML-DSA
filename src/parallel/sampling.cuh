#pragma once
#include <cuda_runtime.h>
#include "polyvec.h"

namespace cdp {
__device__ unsigned reject_uniform(int32_t* a, unsigned n, const uint8_t* p, unsigned len);

__device__ unsigned reject_eta(int32_t* a, unsigned n, const uint8_t* p, unsigned len);

__global__ void expand_a(polyvecl* mat, const uint8_t* rho);

__global__ void expand_s(polyvecl* s1, polyveck* s2, const uint8_t* seed);

__global__ void expand_y(polyvecl* y, const uint8_t* seed, uint16_t nonce);

__global__ void challenge(poly* c, const uint8_t* seed);
} // namespace cdp
