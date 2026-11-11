#include "keygen.cuh"


__device__ void kgm_finish8(kgm_ws *w, unsigned int k, unsigned int g, const int32_t *a) {
    int32_t v1[8], v0[8];
    const int32_t *s2 = w->s2.vec[k].coeffs + 8 * g;
    for (int j = 0; j < 8; ++j) {
        int32_t x = PQCLEAN_MLDSA87_CLEAN_caddq(a[8 * g + j] + s2[j]);
        v1[j] = PQCLEAN_MLDSA87_CLEAN_power2round(&v0[j], x);
    }
    uint8_t *r1 = w->pk + SEEDBYTES + k * POLYT1_PACKEDBYTES + 10 * g;
    for (int h = 0; h < 2; ++h) {
        const int32_t *c = v1 + 4 * h; uint8_t *r = r1 + 5 * h;
        r[0] = (uint8_t) (c[0] >> 0);
        r[1] = (uint8_t) ((c[0] >> 8) | (c[1] << 2));
        r[2] = (uint8_t) ((c[1] >> 6) | (c[2] << 4));
        r[3] = (uint8_t) ((c[2] >> 4) | (c[3] << 6));
        r[4] = (uint8_t) (c[3] >> 2);
    }
    uint32_t t[8];
    for (int j = 0; j < 8; ++j) { t[j] = (1 << (D - 1)) - v0[j]; }
    uint8_t *r = w->sk + KGM_SK_T0 + k * POLYT0_PACKEDBYTES + 13 * g;
    r[0]  =  (uint8_t) t[0];
    r[1]  =  (uint8_t) (t[0] >>  8);
    r[1] |=  (uint8_t) (t[1] <<  5);
    r[2]  =  (uint8_t) (t[1] >>  3);
    r[3]  =  (uint8_t) (t[1] >> 11);
    r[3] |=  (uint8_t) (t[2] <<  2);
    r[4]  =  (uint8_t) (t[2] >>  6);
    r[4] |=  (uint8_t) (t[3] <<  7);
    r[5]  =  (uint8_t) (t[3] >>  1);
    r[6]  =  (uint8_t) (t[3] >>  9);
    r[6] |=  (uint8_t) (t[4] <<  4);
    r[7]  =  (uint8_t) (t[4] >>  4);
    r[8]  =  (uint8_t) (t[4] >> 12);
    r[8] |=  (uint8_t) (t[5] <<  1);
    r[9]  =  (uint8_t) (t[5] >>  7);
    r[9] |=  (uint8_t) (t[6] <<  6);
    r[10] =  (uint8_t) (t[6] >>  2);
    r[11] =  (uint8_t) (t[6] >> 10);
    r[11] |= (uint8_t) (t[7] <<  3);
    r[12] =  (uint8_t) (t[7] >>  5);
}
