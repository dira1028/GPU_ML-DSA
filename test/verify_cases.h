/*
 * test/verify_cases.h  ---  Verify 정확성 검사용 입력 변형
 */
#ifndef MLDSA87_VERIFY_CASES_H
#define MLDSA87_VERIFY_CASES_H

#include <stdint.h>
#include <string.h>

#define VC_NCASE 12
#define VC_MLEN  32

/* 메시지 r: 32바이트, m[0] = r 의 하위 바이트, m[1] = 상위 바이트, 나머지 0 */
static inline void vc_message(uint8_t *m, int r) {
    memset(m, 0, VC_MLEN);
    m[0] = (uint8_t)(r & 0xff);
    m[1] = (uint8_t)((r >> 8) & 0xff);
}

static inline void vc_apply(int c, int r, const uint8_t *pk0, const uint8_t *sig0,
                            uint8_t *pk, uint8_t *msg, uint8_t *sig, size_t pklen, size_t siglen) {
    memcpy(pk, pk0, pklen);
    vc_message(msg, r);
    memcpy(sig, sig0, siglen);
    switch (c) {
    case 1: sig[0] ^= 0x01; break;
    case 2: sig[CTILDEBYTES + 100] ^= 0x04; break;
    case 3: sig[CTILDEBYTES + L * POLYZ_PACKEDBYTES + OMEGA + 0] = (uint8_t)(OMEGA + 1); break;
    case 4: msg[VC_MLEN - 1] ^= 0x80; break;
    case 5: pk[SEEDBYTES + 10] ^= 0x01; break;
    case 6: pk[0] ^= 0x01; break;
    default:
        if (c >= 7) {
            const int fz = c - 7;
            sig[CTILDEBYTES + L * POLYZ_PACKEDBYTES + ((r * 7 + fz * 13) % (OMEGA + K))] = (uint8_t)(r * 31 + fz * 17 + 5);
        }
        break;
    }
}

#endif
