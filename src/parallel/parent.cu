#include "parent.cuh"
#include "sampling.cuh"
#include "hashes.cuh"
#include "arithmetic.cuh"

#if CUDART_VERSION < 12000
#error "The CDP path requires CUDA 12+ (CDP2 tail launches)."
#endif
#ifdef CUDA_FORCE_CDP1_IF_SUPPORTED
#error "The CDP path requires CDP2; remove CUDA_FORCE_CDP1_IF_SUPPORTED."
#endif
#define CDP_CHECK() do { if(!record(&w->cdp_error,cudaGetLastError())) { \
    record(&w->cdp_error,cudaStreamDestroy(stream)); return; } } while(0)
#define CDP_STREAM() cudaStream_t stream; \
    if(!record(&w->cdp_error,cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking))) return


namespace cdp {
__device__ bool record(int* error, cudaError_t status) {
    if(status==cudaSuccess) return true;
    atomicCAS(error,0,int(status));
    return false;
}

__global__ void keygen_parent(kgm_ws* w) {
    if(blockIdx.x || threadIdx.x) return;
    w->cdp_error=0;
    CDP_STREAM();
    /*  
        *시드 확장. H(seed || K || L)에서 rho, rhoprime, key 생성
        1개 블록
        블록 당 25개 Thread

        해시함수 - 25개 스레드 병렬 수행
    */
    kg_seed<<<1,25,0,stream>>>(w); CDP_CHECK();


    /*  
        *A 샘플링
        
        K * L개 블록 (s1, s2의 Poly개수 합)
        블록 당 25개 Thread
        
        각 블록이 Poly 샘플링을 담당함
        샘플링에 사용되는 해시함수 - 25개 스레드 병렬 수행
    */
    expand_a<<<K*L,25,0,stream>>>(w->mat,w->rho); CDP_CHECK();
    // 비밀 벡터 샘플링: rhoprime으로 s1, s2 생성 (ExpandS / polyvecl_uniform_eta, polyveck_uniform_eta).

    /*  
        *s1, s2 샘플링

        K * L개 블록 (s1, s2의 Poly개수 합)
        블록 당 25개 Thread
        
        각 블록이 Poly 샘플링을 담당함
        샘플링에 사용되는 해시함수 - 25개 스레드 병렬 수행
    */
    expand_s<<<L+K,25,0,stream>>>(&w->s1,&w->s2,w->rhoprime); CDP_CHECK();


    /*
        *s1, s2 먼저 인코딩 후 저장

        K + L개 블록 (s1, s2의 Poly개수 합)
        블록 당 32개 스레드

        각 블록이 s1, s2 내부의 Poly 하나를 담당함
        Thread 0 : 다항식 Pack
        Thread 0 ~ 31 : 각 스레드가 배열에 한 바이트씩 저장
        */
    kg_pack_secrets<<<L+K,32,0,stream>>>(w); CDP_CHECK();


    // s1_hat = NTT(s1): A*s1 계산을 위한 비밀 벡터 변환 (polyvecl_ntt).
    /*
        *NTT(s1)
        L개 블록 (s1의 Poly 개수)
        
        블록 당 128개 스레드

        각 블록이 s1 내부의 Poly 하나를 담당함
        Thread128개가 NTT Butterfly연산을 병렬로 처리함
    */
    kg_ntt<<<L,128,0,stream>>>(w); CDP_CHECK();


    // t = A*s1 + s2 계산 후 (t1,t0) = Power2Round(t); pk의 t1, sk의 t0 패킹.
    kg_rows<<<K,128,0,stream>>>(w); CDP_CHECK();
    // tr = H(pk) 계산 후 sk에 저장 (keypair의 공개키 해시 / pack_sk의 tr 부분).
    kg_tr<<<1,25,0,stream>>>(w); CDP_CHECK();
    record(&w->cdp_error,cudaStreamDestroy(stream));
}

__global__ void sign_decide(sgm_ws* w) {
    if(blockIdx.x || threadIdx.x || w->cdp_error) return;
    const int par=(w->iters-1)&1;
    unsigned count=0;
    for(int k=0;k<K;++k) count+=w->nh_row[par][k];
    if(w->flag[par] || count>OMEGA) {
        // Continuation, not a parent-side read of unfinished child results.
        // 서명 후보 기각: 다음 nonce로 y를 다시 샘플링 (reference/sign.c의 goto rej).
        sign_attempt<<<1,1,0,cudaStreamTailLaunch>>>(w);
        record(&w->cdp_error,cudaGetLastError());
    } else {
        CDP_STREAM();
        // 서명 인코딩: 이미 저장된 c_tilde에 z, h를 패킹하여 sig = (c_tilde,z,h) 완성 (pack_sig).
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
    // y 샘플링: rhoprime과 시도 nonce로 마스킹 벡터 생성 (ExpandMask / polyvecl_uniform_gamma1).
    expand_y<<<L,25,0,stream>>>(&w->y,w->rhoprime,nonce); CDP_CHECK();
    // y_hat = NTT(y): A*y 계산 준비 (signature_ctx의 polyvecl_ntt).
    ntt_vector<<<L,128,0,stream>>>(w->yhat.vec,w->y.vec); CDP_CHECK();
    // w = A*y 계산 후 (w1,w0) = Decompose(w); w1 인코딩 (행렬 곱 / polyveck_decompose / pack_w1).
    sg_rows<<<K,128,0,stream>>>(w); CDP_CHECK();
    // c_tilde = H(mu || Encode(w1)): 서명 챌린지 해시 생성 (signature_ctx의 SHAKE256).
    sg_hash_challenge<<<1,25,0,stream>>>(w); CDP_CHECK();
    // c = SampleInBall(c_tilde): 해시에서 희소 챌린지 다항식 생성 (poly_challenge).
    challenge<<<1,25,0,stream>>>(&w->cp,w->sig); CDP_CHECK();
    // c_hat = NTT(c): c*s1, c*s2, c*t0 계산 준비 (poly_ntt).
    ntt_vector<<<1,128,0,stream>>>(&w->cphat,&w->cp); CDP_CHECK();
    // z = y+c*s1, w0-c*s2, c*t0의 노름 검사 및 h 생성 (chknorm / make_hint); 기각 플래그와 힌트 수 기록.
    sg_checks<<<L+K,128,0,stream>>>(w); CDP_CHECK();
    record(&w->cdp_error,cudaStreamDestroy(stream));
    // 기각 판정: 노름 검사 실패 또는 힌트 수 > OMEGA이면 재시도, 통과하면 서명 패킹 (goto rej / pack_sig 분기).
    sign_decide<<<1,1,0,cudaStreamTailLaunch>>>(w);
    record(&w->cdp_error,cudaGetLastError());
}

__global__ void sign_parent(sgm_ws* w) {
    if(blockIdx.x || threadIdx.x) return;
    w->cdp_error=0; w->iters=0; w->cdp_nonce=0;
    if(w->mlen<0 || w->mlen>SG_MAXM) { w->cdp_error=int(cudaErrorInvalidValue); return; } 
    CDP_STREAM();
    // mu = H(tr || 0x00 || 0x00 || msg), rhoprime = H(key || rnd || mu) (빈 context, rnd=0).
    sg_hash_prepare<<<1,25,0,stream>>>(w); CDP_CHECK();
    // A 샘플링: rho로 NTT 영역 행렬 A_hat 생성 (ExpandA / polyvec_matrix_expand).
    expand_a<<<K*L,25,0,stream>>>(w->mat,w->sk); CDP_CHECK();
    // 비밀키에서 s1,s2,t0 복원 후 각각 NTT 변환 (unpack_sk의 계수 복원 / polyvecl_ntt / polyveck_ntt).
    sg_unpack<<<L+2*K,128,0,stream>>>(w); CDP_CHECK();
    record(&w->cdp_error,cudaStreamDestroy(stream));
    // 서명 후보 생성 시작: reference/sign.c의 rej 루프 진입 (y 샘플링부터 시작).
    sign_attempt<<<1,1,0,cudaStreamTailLaunch>>>(w);
    record(&w->cdp_error,cudaGetLastError());
}

__global__ void verify_parent(vfm_ws* w) {
    if(blockIdx.x || threadIdx.x) return;
    w->cdp_error=0; w->bad=0; w->result=-1;
    if(w->mlen<0 || w->mlen>VF_MAXM) { w->cdp_error=int(cudaErrorInvalidValue); return; }
    CDP_STREAM();
    // A 샘플링: rho로 NTT 영역 행렬 A_hat 생성 (ExpandA / polyvec_matrix_expand).
    expand_a<<<K*L,25,0,stream>>>(w->mat,w->pk); CDP_CHECK();
    // tr = H(pk), mu = H(tr || 0x00 || 0x00 || msg) 재계산 (verify_ctx의 메시지 해시, 빈 context).
    vf_hash_prepare<<<1,25,0,stream>>>(w); CDP_CHECK();
    // c = SampleInBall(c_tilde): 해시에서 희소 챌린지 다항식 생성 (poly_challenge).
    challenge<<<1,25,0,stream>>>(&w->cp,w->sig); CDP_CHECK();
    // 서명에서 z,h 복원 및 힌트 인코딩·z 노름 검사 (unpack_sig / polyvecl_chknorm).
    vf_unpack<<<L,128,0,stream>>>(w); CDP_CHECK();
    // z_hat = NTT(z), t1_hat = NTT(t1*2^D), c_hat = NTT(c) (공개키 t1 복원 / shiftl / NTT).
    vf_ntt<<<L+K+1,128,0,stream>>>(w); CDP_CHECK();
    // A*z-c*(t1*2^D) 계산 후 UseHint(h, ...)로 w1 복원 및 인코딩 (행렬 곱 / polyveck_use_hint / pack_w1).
    vf_rows<<<K,128,0,stream>>>(w); CDP_CHECK();
    // c_tilde 재계산: H(mu || Encode(w1))를 서명의 c_tilde와 비교; 형식·노름 검사도 통과해야 승인.
    vf_hash_compare<<<1,25,0,stream>>>(w); CDP_CHECK();
    record(&w->cdp_error,cudaStreamDestroy(stream));
}
} // namespace cdp
