#include "keygen.cuh"

/**************************************************
 * 이름: kgm_finish8
 *
 * 설명: KeyGen 행 k 의 계수 8g..8g+7 마무리
 *       역NTT 를 마친 A·s1 에 s2 를 더하고 caddq → power2round 로 (t1, t0) 분리 후,
 *       t1 은 pk 에 (4계수 → 5바이트, 2묶음), t0 은 sk 에 (8계수 → 13바이트) 패킹
 *       원본 polyveck_add, polyveck_caddq, polyveck_power2round, polyt1_pack, polyt0_pack 을
 *       계수 8개 단위로 합친 함수
 *
 * 인수: - w: KeyGen 작업 공간 (s2 읽기, pk·sk 쓰기)
 *       - k: 행 번호 0..K-1
 *       - g: 8계수 묶음 번호 0..31
 *       - a: 행 k 의 역NTT 결과 (공유메모리, 계수 256개)
 **************************************************/
//현재 코드는 ML-DSA 87에 맞춰져 있음
//다른 버전에 사용하기 위해 패킹 코드 수정 필요
__device__ void kgm_finish8(kgm_ws *w, unsigned int k, unsigned int g, const int32_t *a)
{
    int32_t v1[8], v0[8];
    //32개 스레드가 연산을 담당할 계수를 정함
    /*
        Input : (A * s1)
        (A * s1) + s2 계산

        32개 스레드가 계수 8개씩 맡아서 계산함

    */
    const int32_t *s2 = w->s2.vec[k].coeffs + 8 * g;
    for (int j = 0; j < 8; ++j)
    {
        int32_t x = PQCLEAN_MLDSA87_CLEAN_caddq(a[8 * g + j] + s2[j]);
        v1[j] = PQCLEAN_MLDSA87_CLEAN_power2round(&v0[j], x);
    }
    //pk 저장 순서
    //rho(32Byte) || t1[0](320Byte) || t1[1] || ... || t1[K-1]
    //t1 bitPack 과정
    uint8_t *r1 = w->pk + SEEDBYTES + k * POLYT1_PACKEDBYTES + 10 * g;
    for (int h = 0; h < 2; ++h)
    {
        const int32_t *c = v1 + 4 * h;
        uint8_t *r = r1 + 5 * h;
        r[0] = (uint8_t)(c[0] >> 0);
        r[1] = (uint8_t)((c[0] >> 8) | (c[1] << 2));
        r[2] = (uint8_t)((c[1] >> 6) | (c[2] << 4));
        r[3] = (uint8_t)((c[2] >> 4) | (c[3] << 6));
        r[4] = (uint8_t)(c[3] >> 2);
    }

    //sk 저장 순서
    //rho(32Byte) || K(32Byte) || tr(64Byte) || s1 || s2 || t0[0](416Byte) || t0[1] || ... || t0[K-1]
    //t0 bitPack 과정
    uint32_t t[8];
    for (int j = 0; j < 8; ++j)
    {
        t[j] = (1 << (D - 1)) - v0[j];
    }
    uint8_t *r = w->sk + KGM_SK_T0 + k * POLYT0_PACKEDBYTES + 13 * g;
    r[0] = (uint8_t)t[0];
    r[1] = (uint8_t)(t[0] >> 8);
    r[1] |= (uint8_t)(t[1] << 5);
    r[2] = (uint8_t)(t[1] >> 3);
    r[3] = (uint8_t)(t[1] >> 11);
    r[3] |= (uint8_t)(t[2] << 2);
    r[4] = (uint8_t)(t[2] >> 6);
    r[4] |= (uint8_t)(t[3] << 7);
    r[5] = (uint8_t)(t[3] >> 1);
    r[6] = (uint8_t)(t[3] >> 9);
    r[6] |= (uint8_t)(t[4] << 4);
    r[7] = (uint8_t)(t[4] >> 4);
    r[8] = (uint8_t)(t[4] >> 12);
    r[8] |= (uint8_t)(t[5] << 1);
    r[9] = (uint8_t)(t[5] >> 7);
    r[9] |= (uint8_t)(t[6] << 6);
    r[10] = (uint8_t)(t[6] >> 2);
    r[11] = (uint8_t)(t[6] >> 10);
    r[11] |= (uint8_t)(t[7] << 3);
    r[12] = (uint8_t)(t[7] >> 5);
}
