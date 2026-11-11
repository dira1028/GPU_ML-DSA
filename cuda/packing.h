/*
 * [GPU 이식본] packing.h
 *   원본: ref/packing.h (PQClean crypto_sign/ml-dsa-87/clean, 수정 없음)
 *   이식 방법: 원본 코드와 주석을 그대로 두고, GPU 에서 실행되도록 아래만 바꿨다.
 *   - 모든 함수 선언에 __device__ 를 붙여 GPU 장치 함수로 만듦 (호스트에서는 호출하지 않음)
 *   계산 로직·호출 순서는 원본과 같으므로, 같은 입력에 대해 원본과 바이트 단위로 같은 결과를 낸다.
 *   (이 파일의 함수들은 단일 스레드 구현과 다중 스레드 구현이 함께 사용한다)
 */
#ifndef PQCLEAN_MLDSA87_CLEAN_PACKING_H
#define PQCLEAN_MLDSA87_CLEAN_PACKING_H
#include "params.h"
#include "polyvec.h"
#include <stdint.h>

__device__ void PQCLEAN_MLDSA87_CLEAN_pack_pk(uint8_t pk[PQCLEAN_MLDSA87_CLEAN_CRYPTO_PUBLICKEYBYTES], const uint8_t rho[SEEDBYTES], const polyveck *t1);

__device__ void PQCLEAN_MLDSA87_CLEAN_pack_sk(uint8_t sk[PQCLEAN_MLDSA87_CLEAN_CRYPTO_SECRETKEYBYTES],
                                   const uint8_t rho[SEEDBYTES],
                                   const uint8_t tr[TRBYTES],
                                   const uint8_t key[SEEDBYTES],
                                   const polyveck *t0,
                                   const polyvecl *s1,
                                   const polyveck *s2);

__device__ void PQCLEAN_MLDSA87_CLEAN_pack_sig(uint8_t sig[PQCLEAN_MLDSA87_CLEAN_CRYPTO_BYTES], const uint8_t c[CTILDEBYTES], const polyvecl *z, const polyveck *h);

__device__ void PQCLEAN_MLDSA87_CLEAN_unpack_pk(uint8_t rho[SEEDBYTES], polyveck *t1, const uint8_t pk[PQCLEAN_MLDSA87_CLEAN_CRYPTO_PUBLICKEYBYTES]);

__device__ void PQCLEAN_MLDSA87_CLEAN_unpack_sk(uint8_t rho[SEEDBYTES],
                                     uint8_t tr[TRBYTES],
                                     uint8_t key[SEEDBYTES],
                                     polyveck *t0,
                                     polyvecl *s1,
                                     polyveck *s2,
                                     const uint8_t sk[PQCLEAN_MLDSA87_CLEAN_CRYPTO_SECRETKEYBYTES]);

__device__ int PQCLEAN_MLDSA87_CLEAN_unpack_sig(uint8_t c[CTILDEBYTES], polyvecl *z, polyveck *h, const uint8_t sig[PQCLEAN_MLDSA87_CLEAN_CRYPTO_BYTES]);

#endif
