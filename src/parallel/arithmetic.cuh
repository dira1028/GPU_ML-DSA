#pragma once
#include <cuda_runtime.h>
#include "../common/keygen.cuh"
#include "../common/sign.cuh"
#include "../common/verify.cuh"

namespace cdp {
__global__ void kg_ntt(kgm_ws* w);

__global__ void kg_rows(kgm_ws* w);

__global__ void sg_unpack(sgm_ws* w);

__global__ void sg_rows(sgm_ws* w);

__global__ void sg_checks(sgm_ws* w);

__global__ void sg_pack(sgm_ws* w);

__global__ void vf_ntt(vfm_ws* w);

__global__ void vf_rows(vfm_ws* w);

__global__ void ntt_vector(poly* dst, const poly* src);

__global__ void kg_pack_secrets(kgm_ws* w);

__global__ void vf_unpack(vfm_ws* w);
} // namespace cdp
