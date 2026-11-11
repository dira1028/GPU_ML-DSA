#ifndef PQCLEAN_MLDSA87_CLEAN_NTT_H
#define PQCLEAN_MLDSA87_CLEAN_NTT_H
#include "params.h"
#include <stdint.h>

__device__ void PQCLEAN_MLDSA87_CLEAN_ntt(int32_t a[N]);

__device__ void PQCLEAN_MLDSA87_CLEAN_invntt_tomont(int32_t a[N]);

#endif
