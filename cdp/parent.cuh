//빌드 

/*
//빌드 :
/*
sed -i 's/-std=c++14/-std=c++17/' Makefile
make -j2 CDP=1 ARCH=-arch=sm_87 check
*/
*/

#pragma once
#include <cuda_runtime.h>
#if CUDART_VERSION < 12000
#error "The CDP path requires CUDA 12+ (CDP2 tail launches). Use CDP=0 for legacy."
#endif
#ifdef CUDA_FORCE_CDP1_IF_SUPPORTED
#error "The CDP path requires CDP2; remove CUDA_FORCE_CDP1_IF_SUPPORTED."
#endif
#include "../multi/keygen_multi.cuh"
#include "../multi/sign_multi.cuh"
#include "../multi/verify_multi.cuh"
#include "sampling.cuh"
#include "hashes.cuh"
#include "arithmetic.cuh"

namespace cdp {
__device__ bool record(int* error, cudaError_t status) {
    if(status==cudaSuccess) return true;
    atomicCAS(error,0,int(status));
    return false;
}
// Stop enqueueing after a launch failure. Already queued work completes normally;
// cdp_error makes its partial output unusable to the host.
#define CDP_CHECK() do { if(!record(&w->cdp_error,cudaGetLastError())) { \
    record(&w->cdp_error,cudaStreamDestroy(stream)); return; } } while(0)
#define CDP_STREAM() cudaStream_t stream; \
    if(!record(&w->cdp_error,cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking))) return

__global__ void keygen_parent(kgm_ws* w) {
    if(blockIdx.x || threadIdx.x) return;
    w->cdp_error=0;
    for(int i=0;i<KGM_NPH;++i) w->ph[i]=0;
    CDP_STREAM();
    kg_seed<<<1,25,0,stream>>>(w); CDP_CHECK();
    expand_a<<<K*L,25,0,stream>>>(w->mat,w->rho); CDP_CHECK();
    expand_s<<<L+K,25,0,stream>>>(&w->s1,&w->s2,w->rhoprime); CDP_CHECK();
    kg_pack_secrets<<<L+K,128,0,stream>>>(w); CDP_CHECK();
    kg_ntt<<<L,128,0,stream>>>(w); CDP_CHECK();
    kg_rows<<<K,128,0,stream>>>(w); CDP_CHECK();
    kg_tr<<<1,25,0,stream>>>(w); CDP_CHECK();
    record(&w->cdp_error,cudaStreamDestroy(stream));
}

__global__ void sign_attempt(sgm_ws* w);

__global__ void sign_decide(sgm_ws* w) {
    if(blockIdx.x || threadIdx.x || w->cdp_error) return;
    const int par=(w->iters-1)&1;
    unsigned count=0;
    for(int k=0;k<K;++k) count+=w->nh_row[par][k];
    if(w->flag[par] || count>OMEGA) {
        // Continuation, not a parent-side read of unfinished child results.
        sign_attempt<<<1,1,0,cudaStreamTailLaunch>>>(w);
        record(&w->cdp_error,cudaGetLastError());
    } else {
        CDP_STREAM();
        sg_pack<<<K,128,0,stream>>>(w); CDP_CHECK();
        record(&w->cdp_error,cudaStreamDestroy(stream));
    }
}
__global__ void sign_attempt(sgm_ws* w) {
    if(blockIdx.x || threadIdx.x || w->cdp_error) return;
    const int par=w->iters&1;
    w->flag[par]=0;
    for(int k=0;k<K;++k) w->nh_row[par][k]=0;
    ++w->iters;
    const uint16_t nonce=w->cdp_nonce++;
    CDP_STREAM();
    expand_y<<<L,25,0,stream>>>(&w->y,w->rhoprime,nonce); CDP_CHECK();
    ntt_vector<<<L,128,0,stream>>>(w->yhat.vec,w->y.vec); CDP_CHECK();
    sg_rows<<<K,128,0,stream>>>(w); CDP_CHECK();
    sg_hash_challenge<<<1,25,0,stream>>>(w); CDP_CHECK();
    challenge<<<1,25,0,stream>>>(&w->cp,w->sig); CDP_CHECK();
    ntt_vector<<<1,128,0,stream>>>(&w->cphat,&w->cp); CDP_CHECK();
    sg_checks<<<L+K,128,0,stream>>>(w); CDP_CHECK();
    record(&w->cdp_error,cudaStreamDestroy(stream));
    sign_decide<<<1,1,0,cudaStreamTailLaunch>>>(w);
    record(&w->cdp_error,cudaGetLastError());
}
__global__ void sign_parent(sgm_ws* w) {
    if(blockIdx.x || threadIdx.x) return;
    w->cdp_error=0; w->iters=0; w->cdp_nonce=0;
    for(int i=0;i<SGT_N;++i) w->ph[i]=0;
    if(w->mlen<0 || w->mlen>SG_MAXM) { w->cdp_error=int(cudaErrorInvalidValue); return; }
    CDP_STREAM();
    sg_hash_prepare<<<1,25,0,stream>>>(w); CDP_CHECK();
    expand_a<<<K*L,25,0,stream>>>(w->mat,w->sk); CDP_CHECK();
    sg_unpack<<<L+2*K,128,0,stream>>>(w); CDP_CHECK();
    record(&w->cdp_error,cudaStreamDestroy(stream));
    sign_attempt<<<1,1,0,cudaStreamTailLaunch>>>(w);
    record(&w->cdp_error,cudaGetLastError());
}
__global__ void verify_parent(vfm_ws* w) {
    if(blockIdx.x || threadIdx.x) return;
    w->cdp_error=0; w->bad=0; w->result=-1;
    for(int i=0;i<VFT_N;++i) w->ph[i]=0;
    if(w->mlen<0 || w->mlen>VF_MAXM) { w->cdp_error=int(cudaErrorInvalidValue); return; }
    CDP_STREAM();
    expand_a<<<K*L,25,0,stream>>>(w->mat,w->pk); CDP_CHECK();
    vf_hash_prepare<<<1,25,0,stream>>>(w); CDP_CHECK();
    challenge<<<1,25,0,stream>>>(&w->cp,w->sig); CDP_CHECK();
    vf_unpack<<<L,128,0,stream>>>(w); CDP_CHECK();
    vf_ntt<<<L+K+1,128,0,stream>>>(w); CDP_CHECK();
    vf_rows<<<K,128,0,stream>>>(w); CDP_CHECK();
    vf_hash_compare<<<1,25,0,stream>>>(w); CDP_CHECK();
    record(&w->cdp_error,cudaStreamDestroy(stream));
}
#undef CDP_CHECK
#undef CDP_STREAM
} // namespace cdp
