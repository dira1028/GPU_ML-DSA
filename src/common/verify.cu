#include "verify.cuh"


__device__ int vfm_decode_hint(const uint8_t *hb, polyveck *h) {
    unsigned int i, j, k = 0;
    int err = 0;
    for (i = 0; i < (unsigned)K; ++i) {
        for (j = 0; j < (unsigned)N; ++j) { h->vec[i].coeffs[j] = 0; }
        if (!err) {
            if (hb[OMEGA + i] < k || hb[OMEGA + i] > OMEGA) { err = 1; continue; }
            for (j = k; j < hb[OMEGA + i]; ++j) {
                if (j > k && hb[j] <= hb[j - 1]) { err = 1; break; }
                h->vec[i].coeffs[hb[j]] = 1;
            }
            if (!err) { k = hb[OMEGA + i]; }
        }
    }
    if (!err) {
        for (j = k; j < (unsigned)OMEGA; ++j) { if (hb[j]) { err = 1; break; } }
    }
    return err;
}
