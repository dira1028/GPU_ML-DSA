#include "sign.cuh"


__device__ void sgm_unpack8(const uint8_t *sk, unsigned int tau, unsigned int g, int32_t *a) {
    const uint8_t *base = sk + 2 * SEEDBYTES + TRBYTES;
    int32_t *r = a + 8 * g;
    if (tau < (unsigned)(L + K)) {                                     /* eta (s1, s2) */
        const uint8_t *p = base + tau * POLYETA_PACKEDBYTES + 3 * g;
        r[0] =  (p[0] >> 0) & 7;
        r[1] =  (p[0] >> 3) & 7;
        r[2] = ((p[0] >> 6) | (p[1] << 2)) & 7;
        r[3] =  (p[1] >> 1) & 7;
        r[4] =  (p[1] >> 4) & 7;
        r[5] = ((p[1] >> 7) | (p[2] << 1)) & 7;
        r[6] =  (p[2] >> 2) & 7;
        r[7] =  (p[2] >> 5) & 7;
        for (int j = 0; j < 8; ++j) { r[j] = ETA - r[j]; }
    } else {                                                           /* t0 */
        const uint8_t *p = base + (L + K) * POLYETA_PACKEDBYTES + (tau - L - K) * POLYT0_PACKEDBYTES + 13 * g;
        r[0]  = p[0];       r[0] |= (uint32_t)p[1]  << 8;                                r[0] &= 0x1FFF;
        r[1]  = p[1] >> 5;  r[1] |= (uint32_t)p[2]  << 3; r[1] |= (uint32_t)p[3] << 11;  r[1] &= 0x1FFF;
        r[2]  = p[3] >> 2;  r[2] |= (uint32_t)p[4]  << 6;                                r[2] &= 0x1FFF;
        r[3]  = p[4] >> 7;  r[3] |= (uint32_t)p[5]  << 1; r[3] |= (uint32_t)p[6] << 9;   r[3] &= 0x1FFF;
        r[4]  = p[6] >> 4;  r[4] |= (uint32_t)p[7]  << 4; r[4] |= (uint32_t)p[8] << 12;  r[4] &= 0x1FFF;
        r[5]  = p[8] >> 1;  r[5] |= (uint32_t)p[9]  << 7;                                r[5] &= 0x1FFF;
        r[6]  = p[9] >> 6;  r[6] |= (uint32_t)p[10] << 2; r[6] |= (uint32_t)p[11] << 10; r[6] &= 0x1FFF;
        r[7]  = p[11] >> 3; r[7] |= (uint32_t)p[12] << 5;                                r[7] &= 0x1FFF;
        for (int j = 0; j < 8; ++j) { r[j] = (1 << (D - 1)) - r[j]; }
    }
}
