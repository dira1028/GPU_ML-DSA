#include "verify.cuh"


/**************************************************
 * 이름: vfm_decode_hint
 *
 * 설명: 서명의 힌트 영역을 풀어 h 에 0/1 로 기록하고, unpack_sig 와 같은 형식 검사 수행
 *
 * 인수: - hb: 서명의 힌트 영역 시작 (OMEGA + K 바이트)
 *       - h: 출력 힌트 다항식 벡터
 *
 * 형식 오류면 1, 정상이면 0 을 반환
 **************************************************/
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
