#include "ntt.h"
#include "params.h"
#include "poly.h"
#include "reduce.h"
#include "rounding.h"
#include <stdint.h>

#define DBENCH_START()
#define DBENCH_STOP(t)

/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_poly_reduce
*
* Description: Inplace reduction of all coefficients of polynomial to
*              representative in [-6283008,6283008].
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
__device__ void PQCLEAN_MLDSA87_CLEAN_poly_reduce(poly *a) {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N; ++i) {
        a->coeffs[i] = PQCLEAN_MLDSA87_CLEAN_reduce32(a->coeffs[i]);
    }

    DBENCH_STOP(*tred);
}

/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_poly_caddq
*
* Description: For all coefficients of in/out polynomial add Q if
*              coefficient is negative.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
__device__ void PQCLEAN_MLDSA87_CLEAN_poly_caddq(poly *a) {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N; ++i) {
        a->coeffs[i] = PQCLEAN_MLDSA87_CLEAN_caddq(a->coeffs[i]);
    }

    DBENCH_STOP(*tred);
}

/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_poly_add
*
* Description: Add polynomials. No modular reduction is performed.
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const poly *a: pointer to first summand
*              - const poly *b: pointer to second summand
**************************************************/
__device__ void PQCLEAN_MLDSA87_CLEAN_poly_add(poly *c, const poly *a, const poly *b)  {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N; ++i) {
        c->coeffs[i] = a->coeffs[i] + b->coeffs[i];
    }

    DBENCH_STOP(*tadd);
}

/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_poly_sub
*
* Description: Subtract polynomials. No modular reduction is
*              performed.
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const poly *a: pointer to first input polynomial
*              - const poly *b: pointer to second input polynomial to be
*                               subtraced from first input polynomial
**************************************************/
__device__ void PQCLEAN_MLDSA87_CLEAN_poly_sub(poly *c, const poly *a, const poly *b) {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N; ++i) {
        c->coeffs[i] = a->coeffs[i] - b->coeffs[i];
    }

    DBENCH_STOP(*tadd);
}

/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_poly_shiftl
*
* Description: Multiply polynomial by 2^D without modular reduction. Assumes
*              input coefficients to be less than 2^{31-D} in absolute value.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
__device__ void PQCLEAN_MLDSA87_CLEAN_poly_shiftl(poly *a) {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N; ++i) {
        a->coeffs[i] <<= D;
    }

    DBENCH_STOP(*tmul);
}

/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_poly_ntt
*
* Description: Inplace forward NTT. Coefficients can grow by
*              8*Q in absolute value.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
__device__ void PQCLEAN_MLDSA87_CLEAN_poly_ntt(poly *a) {
    DBENCH_START();

    PQCLEAN_MLDSA87_CLEAN_ntt(a->coeffs);

    DBENCH_STOP(*tmul);
}

/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_poly_invntt_tomont
*
* Description: Inplace inverse NTT and multiplication by 2^{32}.
*              Input coefficients need to be less than Q in absolute
*              value and output coefficients are again bounded by Q.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
__device__ void PQCLEAN_MLDSA87_CLEAN_poly_invntt_tomont(poly *a) {
    DBENCH_START();

    PQCLEAN_MLDSA87_CLEAN_invntt_tomont(a->coeffs);

    DBENCH_STOP(*tmul);
}

/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_poly_pointwise_montgomery
*
* Description: Pointwise multiplication of polynomials in NTT domain
*              representation and multiplication of resulting polynomial
*              by 2^{-32}.
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const poly *a: pointer to first input polynomial
*              - const poly *b: pointer to second input polynomial
**************************************************/
__device__ void PQCLEAN_MLDSA87_CLEAN_poly_pointwise_montgomery(poly *c, const poly *a, const poly *b) {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N; ++i) {
        c->coeffs[i] = PQCLEAN_MLDSA87_CLEAN_montgomery_reduce((int64_t)a->coeffs[i] * b->coeffs[i]);
    }

    DBENCH_STOP(*tmul);
}

/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_poly_power2round
*
* Description: For all coefficients c of the input polynomial,
*              compute c0, c1 such that c mod Q = c1*2^D + c0
*              with -2^{D-1} < c0 <= 2^{D-1}. Assumes coefficients to be
*              standard representatives.
*
* Arguments:   - poly *a1: pointer to output polynomial with coefficients c1
*              - poly *a0: pointer to output polynomial with coefficients c0
*              - const poly *a: pointer to input polynomial
**************************************************/
__device__ void PQCLEAN_MLDSA87_CLEAN_poly_power2round(poly *a1, poly *a0, const poly *a) {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N; ++i) {
        a1->coeffs[i] = PQCLEAN_MLDSA87_CLEAN_power2round(&a0->coeffs[i], a->coeffs[i]);
    }

    DBENCH_STOP(*tround);
}

/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_poly_decompose
*
* Description: For all coefficients c of the input polynomial,
*              compute high and low bits c0, c1 such c mod Q = c1*ALPHA + c0
*              with -ALPHA/2 < c0 <= ALPHA/2 except c1 = (Q-1)/ALPHA where we
*              set c1 = 0 and -ALPHA/2 <= c0 = c mod Q - Q < 0.
*              Assumes coefficients to be standard representatives.
*
* Arguments:   - poly *a1: pointer to output polynomial with coefficients c1
*              - poly *a0: pointer to output polynomial with coefficients c0
*              - const poly *a: pointer to input polynomial
**************************************************/
__device__ void PQCLEAN_MLDSA87_CLEAN_poly_decompose(poly *a1, poly *a0, const poly *a) {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N; ++i) {
        a1->coeffs[i] = PQCLEAN_MLDSA87_CLEAN_decompose(&a0->coeffs[i], a->coeffs[i]);
    }

    DBENCH_STOP(*tround);
}

/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_poly_make_hint
*
* Description: Compute hint polynomial. The coefficients of which indicate
*              whether the low bits of the corresponding coefficient of
*              the input polynomial overflow into the high bits.
*
* Arguments:   - poly *h: pointer to output hint polynomial
*              - const poly *a0: pointer to low part of input polynomial
*              - const poly *a1: pointer to high part of input polynomial
*
* Returns number of 1 bits.
**************************************************/
__device__ unsigned int PQCLEAN_MLDSA87_CLEAN_poly_make_hint(poly *h, const poly *a0, const poly *a1) {
    unsigned int i, s = 0;
    DBENCH_START();

    for (i = 0; i < N; ++i) {
        h->coeffs[i] = PQCLEAN_MLDSA87_CLEAN_make_hint(a0->coeffs[i], a1->coeffs[i]);
        s += h->coeffs[i];
    }

    DBENCH_STOP(*tround);
    return s;
}

/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_poly_use_hint
*
* Description: Use hint polynomial to correct the high bits of a polynomial.
*
* Arguments:   - poly *b: pointer to output polynomial with corrected high bits
*              - const poly *a: pointer to input polynomial
*              - const poly *h: pointer to input hint polynomial
**************************************************/
__device__ void PQCLEAN_MLDSA87_CLEAN_poly_use_hint(poly *b, const poly *a, const poly *h) {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N; ++i) {
        b->coeffs[i] = PQCLEAN_MLDSA87_CLEAN_use_hint(a->coeffs[i], h->coeffs[i]);
    }

    DBENCH_STOP(*tround);
}

/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_poly_chknorm
*
* Description: Check infinity norm of polynomial against given bound.
*              Assumes input coefficients were reduced by PQCLEAN_MLDSA87_CLEAN_reduce32().
*
* Arguments:   - const poly *a: pointer to polynomial
*              - int32_t B: norm bound
*
* Returns 0 if norm is strictly smaller than B <= (Q-1)/8 and 1 otherwise.
**************************************************/
__device__ int PQCLEAN_MLDSA87_CLEAN_poly_chknorm(const poly *a, int32_t B) {
    unsigned int i;
    int32_t t;
    DBENCH_START();

    if (B > (Q - 1) / 8) {
        return 1;
    }

    /* It is ok to leak which coefficient violates the bound since
       the probability for each coefficient is independent of secret
       data but we must not leak the sign of the centralized representative. */
    for (i = 0; i < N; ++i) {
        /* Absolute value */
        t = a->coeffs[i] >> 31;
        t = a->coeffs[i] - (t & 2 * a->coeffs[i]);

        if (t >= B) {
            DBENCH_STOP(*tsample);
            return 1;
        }
    }

    DBENCH_STOP(*tsample);
    return 0;
}


/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_polyeta_pack
*
* Description: Bit-pack polynomial with coefficients in [-ETA,ETA].
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYETA_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
__device__ void PQCLEAN_MLDSA87_CLEAN_polyeta_pack(uint8_t *r, const poly *a) {
    unsigned int i;
    uint8_t t[8];
    DBENCH_START();

    for (i = 0; i < N / 8; ++i) {
        t[0] = (uint8_t) (ETA - a->coeffs[8 * i + 0]);
        t[1] = (uint8_t) (ETA - a->coeffs[8 * i + 1]);
        t[2] = (uint8_t) (ETA - a->coeffs[8 * i + 2]);
        t[3] = (uint8_t) (ETA - a->coeffs[8 * i + 3]);
        t[4] = (uint8_t) (ETA - a->coeffs[8 * i + 4]);
        t[5] = (uint8_t) (ETA - a->coeffs[8 * i + 5]);
        t[6] = (uint8_t) (ETA - a->coeffs[8 * i + 6]);
        t[7] = (uint8_t) (ETA - a->coeffs[8 * i + 7]);

        r[3 * i + 0]  = (t[0] >> 0) | (t[1] << 3) | (t[2] << 6);
        r[3 * i + 1]  = (t[2] >> 2) | (t[3] << 1) | (t[4] << 4) | (t[5] << 7);
        r[3 * i + 2]  = (t[5] >> 1) | (t[6] << 2) | (t[7] << 5);
    }

    DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_polyeta_unpack
*
* Description: Unpack polynomial with coefficients in [-ETA,ETA].
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
__device__ void PQCLEAN_MLDSA87_CLEAN_polyeta_unpack(poly *r, const uint8_t *a) {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N / 8; ++i) {
        r->coeffs[8 * i + 0] =  (a[3 * i + 0] >> 0) & 7;
        r->coeffs[8 * i + 1] =  (a[3 * i + 0] >> 3) & 7;
        r->coeffs[8 * i + 2] = ((a[3 * i + 0] >> 6) | (a[3 * i + 1] << 2)) & 7;
        r->coeffs[8 * i + 3] =  (a[3 * i + 1] >> 1) & 7;
        r->coeffs[8 * i + 4] =  (a[3 * i + 1] >> 4) & 7;
        r->coeffs[8 * i + 5] = ((a[3 * i + 1] >> 7) | (a[3 * i + 2] << 1)) & 7;
        r->coeffs[8 * i + 6] =  (a[3 * i + 2] >> 2) & 7;
        r->coeffs[8 * i + 7] =  (a[3 * i + 2] >> 5) & 7;

        r->coeffs[8 * i + 0] = ETA - r->coeffs[8 * i + 0];
        r->coeffs[8 * i + 1] = ETA - r->coeffs[8 * i + 1];
        r->coeffs[8 * i + 2] = ETA - r->coeffs[8 * i + 2];
        r->coeffs[8 * i + 3] = ETA - r->coeffs[8 * i + 3];
        r->coeffs[8 * i + 4] = ETA - r->coeffs[8 * i + 4];
        r->coeffs[8 * i + 5] = ETA - r->coeffs[8 * i + 5];
        r->coeffs[8 * i + 6] = ETA - r->coeffs[8 * i + 6];
        r->coeffs[8 * i + 7] = ETA - r->coeffs[8 * i + 7];
    }

    DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_polyt1_pack
*
* Description: Bit-pack polynomial t1 with coefficients fitting in 10 bits.
*              Input coefficients are assumed to be standard representatives.
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYT1_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
__device__ void PQCLEAN_MLDSA87_CLEAN_polyt1_pack(uint8_t *r, const poly *a) {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N / 4; ++i) {
        r[5 * i + 0] = (uint8_t) (a->coeffs[4 * i + 0] >> 0);
        r[5 * i + 1] = (uint8_t) ((a->coeffs[4 * i + 0] >> 8) | (a->coeffs[4 * i + 1] << 2));
        r[5 * i + 2] = (uint8_t) ((a->coeffs[4 * i + 1] >> 6) | (a->coeffs[4 * i + 2] << 4));
        r[5 * i + 3] = (uint8_t) ((a->coeffs[4 * i + 2] >> 4) | (a->coeffs[4 * i + 3] << 6));
        r[5 * i + 4] = (uint8_t) (a->coeffs[4 * i + 3] >> 2);
    }

    DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_polyt1_unpack
*
* Description: Unpack polynomial t1 with 10-bit coefficients.
*              Output coefficients are standard representatives.
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
__device__ void PQCLEAN_MLDSA87_CLEAN_polyt1_unpack(poly *r, const uint8_t *a) {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N / 4; ++i) {
        r->coeffs[4 * i + 0] = ((a[5 * i + 0] >> 0) | ((uint32_t)a[5 * i + 1] << 8)) & 0x3FF;
        r->coeffs[4 * i + 1] = ((a[5 * i + 1] >> 2) | ((uint32_t)a[5 * i + 2] << 6)) & 0x3FF;
        r->coeffs[4 * i + 2] = ((a[5 * i + 2] >> 4) | ((uint32_t)a[5 * i + 3] << 4)) & 0x3FF;
        r->coeffs[4 * i + 3] = ((a[5 * i + 3] >> 6) | ((uint32_t)a[5 * i + 4] << 2)) & 0x3FF;
    }

    DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_polyt0_pack
*
* Description: Bit-pack polynomial t0 with coefficients in ]-2^{D-1}, 2^{D-1}].
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYT0_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
__device__ void PQCLEAN_MLDSA87_CLEAN_polyt0_pack(uint8_t *r, const poly *a) {
    unsigned int i;
    uint32_t t[8];
    DBENCH_START();

    for (i = 0; i < N / 8; ++i) {
        t[0] = (1 << (D - 1)) - a->coeffs[8 * i + 0];
        t[1] = (1 << (D - 1)) - a->coeffs[8 * i + 1];
        t[2] = (1 << (D - 1)) - a->coeffs[8 * i + 2];
        t[3] = (1 << (D - 1)) - a->coeffs[8 * i + 3];
        t[4] = (1 << (D - 1)) - a->coeffs[8 * i + 4];
        t[5] = (1 << (D - 1)) - a->coeffs[8 * i + 5];
        t[6] = (1 << (D - 1)) - a->coeffs[8 * i + 6];
        t[7] = (1 << (D - 1)) - a->coeffs[8 * i + 7];

        r[13 * i + 0]  =  (uint8_t) t[0];
        r[13 * i + 1]  =  (uint8_t) (t[0] >>  8);
        r[13 * i + 1] |=  (uint8_t) (t[1] <<  5);
        r[13 * i + 2]  =  (uint8_t) (t[1] >>  3);
        r[13 * i + 3]  =  (uint8_t) (t[1] >> 11);
        r[13 * i + 3] |=  (uint8_t) (t[2] <<  2);
        r[13 * i + 4]  =  (uint8_t) (t[2] >>  6);
        r[13 * i + 4] |=  (uint8_t) (t[3] <<  7);
        r[13 * i + 5]  =  (uint8_t) (t[3] >>  1);
        r[13 * i + 6]  =  (uint8_t) (t[3] >>  9);
        r[13 * i + 6] |=  (uint8_t) (t[4] <<  4);
        r[13 * i + 7]  =  (uint8_t) (t[4] >>  4);
        r[13 * i + 8]  =  (uint8_t) (t[4] >> 12);
        r[13 * i + 8] |=  (uint8_t) (t[5] <<  1);
        r[13 * i + 9]  =  (uint8_t) (t[5] >>  7);
        r[13 * i + 9] |=  (uint8_t) (t[6] <<  6);
        r[13 * i + 10]  =  (uint8_t) (t[6] >>  2);
        r[13 * i + 11]  =  (uint8_t) (t[6] >> 10);
        r[13 * i + 11] |=  (uint8_t) (t[7] <<  3);
        r[13 * i + 12]  =  (uint8_t) (t[7] >>  5);
    }

    DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_polyt0_unpack
*
* Description: Unpack polynomial t0 with coefficients in ]-2^{D-1}, 2^{D-1}].
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
__device__ void PQCLEAN_MLDSA87_CLEAN_polyt0_unpack(poly *r, const uint8_t *a) {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N / 8; ++i) {
        r->coeffs[8 * i + 0]  = a[13 * i + 0];
        r->coeffs[8 * i + 0] |= (uint32_t)a[13 * i + 1] << 8;
        r->coeffs[8 * i + 0] &= 0x1FFF;

        r->coeffs[8 * i + 1]  = a[13 * i + 1] >> 5;
        r->coeffs[8 * i + 1] |= (uint32_t)a[13 * i + 2] << 3;
        r->coeffs[8 * i + 1] |= (uint32_t)a[13 * i + 3] << 11;
        r->coeffs[8 * i + 1] &= 0x1FFF;

        r->coeffs[8 * i + 2]  = a[13 * i + 3] >> 2;
        r->coeffs[8 * i + 2] |= (uint32_t)a[13 * i + 4] << 6;
        r->coeffs[8 * i + 2] &= 0x1FFF;

        r->coeffs[8 * i + 3]  = a[13 * i + 4] >> 7;
        r->coeffs[8 * i + 3] |= (uint32_t)a[13 * i + 5] << 1;
        r->coeffs[8 * i + 3] |= (uint32_t)a[13 * i + 6] << 9;
        r->coeffs[8 * i + 3] &= 0x1FFF;

        r->coeffs[8 * i + 4]  = a[13 * i + 6] >> 4;
        r->coeffs[8 * i + 4] |= (uint32_t)a[13 * i + 7] << 4;
        r->coeffs[8 * i + 4] |= (uint32_t)a[13 * i + 8] << 12;
        r->coeffs[8 * i + 4] &= 0x1FFF;

        r->coeffs[8 * i + 5]  = a[13 * i + 8] >> 1;
        r->coeffs[8 * i + 5] |= (uint32_t)a[13 * i + 9] << 7;
        r->coeffs[8 * i + 5] &= 0x1FFF;

        r->coeffs[8 * i + 6]  = a[13 * i + 9] >> 6;
        r->coeffs[8 * i + 6] |= (uint32_t)a[13 * i + 10] << 2;
        r->coeffs[8 * i + 6] |= (uint32_t)a[13 * i + 11] << 10;
        r->coeffs[8 * i + 6] &= 0x1FFF;

        r->coeffs[8 * i + 7]  = a[13 * i + 11] >> 3;
        r->coeffs[8 * i + 7] |= (uint32_t)a[13 * i + 12] << 5;
        r->coeffs[8 * i + 7] &= 0x1FFF;

        r->coeffs[8 * i + 0] = (1 << (D - 1)) - r->coeffs[8 * i + 0];
        r->coeffs[8 * i + 1] = (1 << (D - 1)) - r->coeffs[8 * i + 1];
        r->coeffs[8 * i + 2] = (1 << (D - 1)) - r->coeffs[8 * i + 2];
        r->coeffs[8 * i + 3] = (1 << (D - 1)) - r->coeffs[8 * i + 3];
        r->coeffs[8 * i + 4] = (1 << (D - 1)) - r->coeffs[8 * i + 4];
        r->coeffs[8 * i + 5] = (1 << (D - 1)) - r->coeffs[8 * i + 5];
        r->coeffs[8 * i + 6] = (1 << (D - 1)) - r->coeffs[8 * i + 6];
        r->coeffs[8 * i + 7] = (1 << (D - 1)) - r->coeffs[8 * i + 7];
    }

    DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_polyz_pack
*
* Description: Bit-pack polynomial with coefficients
*              in [-(GAMMA1 - 1), GAMMA1].
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYZ_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
__device__ void PQCLEAN_MLDSA87_CLEAN_polyz_pack(uint8_t *r, const poly *a) {
    unsigned int i;
    uint32_t t[4];
    DBENCH_START();

    for (i = 0; i < N / 2; ++i) {
        t[0] = GAMMA1 - a->coeffs[2 * i + 0];
        t[1] = GAMMA1 - a->coeffs[2 * i + 1];

        r[5 * i + 0]  = (uint8_t) (t[0]);
        r[5 * i + 1]  = (uint8_t) (t[0] >> 8);
        r[5 * i + 2]  = (uint8_t) (t[0] >> 16);
        r[5 * i + 2] |= (uint8_t) (t[1] << 4);
        r[5 * i + 3]  = (uint8_t) (t[1] >> 4);
        r[5 * i + 4]  = (uint8_t) (t[1] >> 12);
    }

    DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_polyz_unpack
*
* Description: Unpack polynomial z with coefficients
*              in [-(GAMMA1 - 1), GAMMA1].
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
__device__ void PQCLEAN_MLDSA87_CLEAN_polyz_unpack(poly *r, const uint8_t *a) {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N / 2; ++i) {
        r->coeffs[2 * i + 0]  = a[5 * i + 0];
        r->coeffs[2 * i + 0] |= (uint32_t)a[5 * i + 1] << 8;
        r->coeffs[2 * i + 0] |= (uint32_t)a[5 * i + 2] << 16;
        r->coeffs[2 * i + 0] &= 0xFFFFF;

        r->coeffs[2 * i + 1]  = a[5 * i + 2] >> 4;
        r->coeffs[2 * i + 1] |= (uint32_t)a[5 * i + 3] << 4;
        r->coeffs[2 * i + 1] |= (uint32_t)a[5 * i + 4] << 12;
        /* r->coeffs[2*i+1] &= 0xFFFFF; */ /* No effect, since we're anyway at 20 bits */

        r->coeffs[2 * i + 0] = GAMMA1 - r->coeffs[2 * i + 0];
        r->coeffs[2 * i + 1] = GAMMA1 - r->coeffs[2 * i + 1];
    }

    DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        PQCLEAN_MLDSA87_CLEAN_polyw1_pack
*
* Description: Bit-pack polynomial w1 with coefficients in [0,15] or [0,43].
*              Input coefficients are assumed to be standard representatives.
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYW1_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
__device__ void PQCLEAN_MLDSA87_CLEAN_polyw1_pack(uint8_t *r, const poly *a) {
    unsigned int i;
    DBENCH_START();

    for (i = 0; i < N / 2; ++i) {
        r[i] = (uint8_t) (a->coeffs[2 * i + 0] | (a->coeffs[2 * i + 1] << 4));
    }

    DBENCH_STOP(*tpack);
}
