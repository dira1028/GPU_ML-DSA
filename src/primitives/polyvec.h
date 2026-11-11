#ifndef PQCLEAN_MLDSA87_CLEAN_POLYVEC_H
#define PQCLEAN_MLDSA87_CLEAN_POLYVEC_H
#include "params.h"
#include "poly.h"
#include <stdint.h>

/* Vectors of polynomials of length L */
typedef struct {
    poly vec[L];
} polyvecl;

/* Vectors of polynomials of length K */
typedef struct {
    poly vec[K];
} polyveck;

#endif
