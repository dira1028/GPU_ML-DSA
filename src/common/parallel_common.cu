#include "parallel_common.cuh"


__device__ int mt_over(int32_t a, int32_t B) {
    int32_t t = a >> 31;
    t = a - (t & 2 * a);
    return t >= B;
}
