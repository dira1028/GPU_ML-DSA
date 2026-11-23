/* Parallel ML-DSA-87 GPU benchmark: KeyGen / Sign / Verify 커널 평균 시간만 출력 */
#include "../src/parallel/parent.cuh"
#include "../tests/verify_cases.h"
#include "../tests/vectors.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

#define PKB  PQCLEAN_MLDSA87_CLEAN_CRYPTO_PUBLICKEYBYTES
#define SKB  PQCLEAN_MLDSA87_CLEAN_CRYPTO_SECRETKEYBYTES
#define SIGB PQCLEAN_MLDSA87_CLEAN_CRYPTO_BYTES

enum { OP_KG = 0, OP_SG, OP_VF, NOP };
static const char *OP_NAME[NOP] = { "KeyGen", "Sign", "Verify" };

/* GPU 스트림과 작업 공간 */
static cudaStream_t s0;
static kgm_ws *d_kg;
static sgm_ws *d_sg;
static vfm_ws *d_vf;

/* 키 설정 (Sign: sk, Verify: pk) */
static void set_key(int op, const vec_set *v) {
    if (op == OP_SG) { CK(cudaMemcpy(d_sg->sk, vec_sk(v, 0), SKB, cudaMemcpyHostToDevice)); }
    if (op == OP_VF) { CK(cudaMemcpy(d_vf->pk, vec_pk(v, 0), PKB, cudaMemcpyHostToDevice)); }
}

/* 입력 업로드 (KeyGen: 시드 / Sign: 메시지 / Verify: 서명 + 메시지) */
static void upload(int op, const uint8_t *seed, const uint8_t *msg, int mlen, const uint8_t *sig) {
    if (op == OP_KG) {
        CK(cudaMemcpy(d_kg->seed, seed, SEEDBYTES, cudaMemcpyHostToDevice));
    } else if (op == OP_SG) {
        CK(cudaMemcpy(d_sg->msg, msg, (size_t)mlen, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(&d_sg->mlen, &mlen, sizeof(int), cudaMemcpyHostToDevice));
    } else {
        CK(cudaMemcpy(d_vf->sig, sig, SIGB, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_vf->msg, msg, (size_t)mlen, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(&d_vf->mlen, &mlen, sizeof(int), cudaMemcpyHostToDevice));
    }
}

/* 부모 커널 실행 */
static void run(int op) {
    if (op == OP_KG)      { cdp::keygen_parent<<<1, 1, 0, s0>>>(d_kg); }
    else if (op == OP_SG) { cdp::sign_parent<<<1, 1, 0, s0>>>(d_sg); }
    else                  { cdp::verify_parent<<<1, 1, 0, s0>>>(d_vf); }
}

/* 자식 커널 오류 확인 (부모·자식 커널 완료 후 호출) */
static void check_child_error(int op) {
    int error = 0;
    const int *p = op == OP_KG ? &d_kg->cdp_error : op == OP_SG ? &d_sg->cdp_error : &d_vf->cdp_error;
    CK(cudaMemcpy(&error, p, sizeof(error), cudaMemcpyDeviceToHost));
    CK(static_cast<cudaError_t>(error));
}

/* 결과 다운로드 후 정답 벡터와 비교 (KeyGen: pk + sk / Sign: 서명 / Verify: 통과 여부) */
static int output_ok(int op, const vec_set *v, int r) {
    static uint8_t pk[PKB], sk[SKB], sig[SIGB];
    int res = -1;
    check_child_error(op);
    if (op == OP_KG) {
        CK(cudaMemcpy(pk, d_kg->pk, PKB, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(sk, d_kg->sk, SKB, cudaMemcpyDeviceToHost));
        return !memcmp(pk, vec_pk(v, 0), PKB) && !memcmp(sk, vec_sk(v, 0), SKB);
    }
    if (op == OP_SG) {
        CK(cudaMemcpy(sig, d_sg->sig, SIGB, cudaMemcpyDeviceToHost));
        return !memcmp(sig, vec_sig(v, 0, r), SIGB);
    }
    CK(cudaMemcpy(&res, &d_vf->result, sizeof(int), cudaMemcpyDeviceToHost));
    return res == 0;
}

/* 연산 op 를 n 회 측정 (워밍업 5회 제외). 커널 평균 µs 반환, 정답 불일치 수는 *bad, Sign 평균 반복 횟수는 *iters */
static double measure(int op, const vec_set *v, int n, int *bad, double *iters) {
    uint8_t seed[SEEDBYTES], msg[VC_MLEN];
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    double sum = 0;
    long long it_sum = 0;
    *bad = 0;
    set_key(op, v);
    vec_seed(0, seed);
    for (int i = -5; i < n; i++) {
        const int r = (i + 5) % v->nmsg;
        vc_message(msg, r);
        upload(op, seed, msg, VC_MLEN, vec_sig(v, 0, r));
        CK(cudaEventRecord(e0, s0));
        run(op);
        CK(cudaEventRecord(e1, s0));
        CK(cudaEventSynchronize(e1));
        CK(cudaGetLastError());
        const int ok = output_ok(op, v, r);
        if (i < 0) { continue; }                         /* 워밍업 */
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        sum += ms * 1000.0;
        *bad += !ok;
        if (op == OP_SG) {
            int it = 0;
            CK(cudaMemcpy(&it, &d_sg->iters, sizeof(int), cudaMemcpyDeviceToHost));
            it_sum += it;
        }
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    *iters = (double)it_sum / n;
    return sum / n;
}

/* "a,b,c" 형식의 양의 정수 3개 변환 */
static bool parse3(const char *s, int *a) {
    char tail;
    return sscanf(s, "%d,%d,%d%c", &a[0], &a[1], &a[2], &tail) == 3 && a[0] > 0 && a[1] > 0 && a[2] > 0;
}

int main(int argc, char **argv) {
    int n[NOP] = { 200, 1000, 1000 };
    if (argc < 3 || strcmp(argv[1], "bench")) {
        fprintf(stderr, "사용: %s bench <벡터파일> [--multi=KG,SG,VF]\n", argv[0]);
        return 2;
    }
    for (int i = 3; i < argc; i++) {
        if (strncmp(argv[i], "--multi=", 8) || !parse3(argv[i] + 8, n)) {
            fprintf(stderr, "--multi=KG,SG,VF 형식 필요\n");
            return 2;
        }
    }
    vec_set v;
    if (vec_read(argv[2], &v) || v.mlen != VC_MLEN) { fprintf(stderr, "벡터 파일을 읽을 수 없음: %s\n", argv[2]); return 1; }

    CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    CK(cudaMalloc(&d_kg, sizeof(kgm_ws))); CK(cudaMalloc(&d_sg, sizeof(sgm_ws))); CK(cudaMalloc(&d_vf, sizeof(vfm_ws)));
    CK(cudaMemset(d_kg, 0, sizeof(kgm_ws))); CK(cudaMemset(d_sg, 0, sizeof(sgm_ws))); CK(cudaMemset(d_vf, 0, sizeof(vfm_ws)));

    printf("ML-DSA-87 parallel GPU (커널 평균)\n");
    int total_bad = 0;
    for (int op = 0; op < NOP; op++) {
        int bad;
        double iters;
        const double us = measure(op, &v, n[op], &bad, &iters);
        printf("%-7s %10.1f us  (n = %d", OP_NAME[op], us, n[op]);
        if (op == OP_SG) { printf(", 평균 반복 %.2f", iters); }
        printf(")\n");
        total_bad += bad;
    }
    if (total_bad) { printf("error: 정답 벡터와 불일치 %d건\n", total_bad); }
    return total_bad ? 1 : 0;
}
