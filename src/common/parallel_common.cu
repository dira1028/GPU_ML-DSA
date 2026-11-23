#include "parallel_common.cuh"


/**************************************************
 * 이름: mt_over
 *
 * 설명: poly_chknorm 의 계수 1개 판정
 *
 * 인수: - a: 검사할 계수 (중심 표현)
 *      - B: 노름 경계
 *
 * |a| >= B 이면 1, 아니면 0 을 반환
 **************************************************/
__device__ int mt_over(int32_t a, int32_t B) {
    int32_t t = a >> 31;
    t = a - (t & 2 * a);
    return t >= B;
}
