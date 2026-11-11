#ifndef PQCLEAN_MLDSA87_CLEAN_ROUNDING_H
#define PQCLEAN_MLDSA87_CLEAN_ROUNDING_H
#include "params.h"
#include <stdint.h>

__device__ int32_t PQCLEAN_MLDSA87_CLEAN_power2round(int32_t *a0, int32_t a);

__device__ int32_t PQCLEAN_MLDSA87_CLEAN_decompose(int32_t *a0, int32_t a);

__device__ unsigned int PQCLEAN_MLDSA87_CLEAN_make_hint(int32_t a0, int32_t a1);

__device__ int32_t PQCLEAN_MLDSA87_CLEAN_use_hint(int32_t a, unsigned int hint);

#endif
