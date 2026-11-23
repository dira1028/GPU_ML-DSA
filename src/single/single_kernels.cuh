/*
 * src/single/single_kernels.cuh  ---  단일 스레드 구현의 커널
 */
#ifndef MLDSA87_SINGLE_KERNELS_CUH
#define MLDSA87_SINGLE_KERNELS_CUH

#include "single_core.h"

/**************************************************
 * 이름: single_keygen_kernel
 *
 * 설명: 단일 스레드 KeyGen 커널 (호스트가 <<<1, 1>>> 로 실행). mldsa87_keygen_core 호출
 *
 * 인수: - seed: 32바이트 시드
 *      - pk: 출력 공개키
 *      - sk: 출력 비밀키
 **************************************************/
__global__ void single_keygen_kernel(const uint8_t *seed, uint8_t *pk, uint8_t *sk) {
    mldsa87_keygen_core(seed, pk, sk);
}

/**************************************************
 * 이름: single_sign_kernel
 *
 * 설명: 단일 스레드 Sign 커널 (<<<1, 1>>>). mldsa87_sign_core 호출
 *
 * 인수: - sig: 출력 서명
 *      - siglen: 출력 서명 길이
 *      - m: 메시지
 *      - mlen: 메시지 길이
 *      - sk: 비밀키
 *      - ws: 전역 메모리 작업 공간
 **************************************************/
__global__ void single_sign_kernel(uint8_t *sig, size_t *siglen, const uint8_t *m, size_t mlen,
                                   const uint8_t *sk, sign_ws *ws) {
    mldsa87_sign_core(sig, siglen, m, mlen, sk, ws);
}

/**************************************************
 * 이름: single_verify_kernel
 *
 * 설명: 단일 스레드 Verify 커널 (<<<1, 1>>>). mldsa87_verify_core 의 결과 (0 통과 / −1 거부) 를 *ret 에 기록
 *
 * 인수: - sig: 서명
 *      - m: 메시지
 *      - mlen: 메시지 길이
 *      - pk: 공개키
 *      - ws: 전역 메모리 작업 공간
 *      - ret: 출력 결과
 **************************************************/
__global__ void single_verify_kernel(const uint8_t *sig, const uint8_t *m, size_t mlen,
                                     const uint8_t *pk, verify_ws *ws, int *ret) {
    *ret = mldsa87_verify_core(sig, PQCLEAN_MLDSA87_CLEAN_CRYPTO_BYTES, m, mlen, pk, ws);
}

#endif
