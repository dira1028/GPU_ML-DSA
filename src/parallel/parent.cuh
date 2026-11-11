#pragma once
#include <cuda_runtime.h>
#include "../common/keygen.cuh"
#include "../common/sign.cuh"
#include "../common/verify.cuh"

namespace cdp {
__device__ bool record(int* error, cudaError_t status);

__global__ void keygen_parent(kgm_ws* w);

__global__ void sign_decide(sgm_ws* w);

__global__ void sign_attempt(sgm_ws* w);

__global__ void sign_parent(sgm_ws* w);

__global__ void verify_parent(vfm_ws* w);
} // namespace cdp
