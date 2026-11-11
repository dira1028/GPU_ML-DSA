/* Parallel ML-DSA-87 correctness checks and CUDA event benchmark. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "../src/parallel/parent.cuh"
#include "../tests/verify_cases.h"
#include "../tests/vectors.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda_runtime.h>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

#define PKB  PQCLEAN_MLDSA87_CLEAN_CRYPTO_PUBLICKEYBYTES
#define SKB  PQCLEAN_MLDSA87_CLEAN_CRYPTO_SECRETKEYBYTES
#define SIGB PQCLEAN_MLDSA87_CLEAN_CRYPTO_BYTES

enum { OP_KG = 0, OP_SG, OP_VF, NOP };
static const char *OP_NAME[NOP] = { "KeyGen", "Sign", "Verify" };

/* GPU 버퍼 */
static cudaStream_t s0;
/* GPU workspaces */
static kgm_ws *d_kg;
static sgm_ws *d_sg;
static vfm_ws *d_vf;

static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }

/* 연산별 입력 · 실행 · 출력 */
typedef struct {
    int op;
    const vec_set *v;
} job;

/* 키 설정 */
static void set_key(const job *j, int k) {
    if (j->op == OP_SG) {
        CK(cudaMemcpy(d_sg->sk, vec_sk(j->v, k), SKB, cudaMemcpyHostToDevice));
    } else if (j->op == OP_VF) {
        CK(cudaMemcpy(d_vf->pk, vec_pk(j->v, k), PKB, cudaMemcpyHostToDevice));
    }
}

static void set_pk_bytes(const uint8_t *pk) {
    CK(cudaMemcpy(d_vf->pk, pk, PKB, cudaMemcpyHostToDevice));
}

/* 입력 업로드 -> KeyGen: seed / Sign: 메시지 / Verify: 서명 + 메시지 */
static void upload(const job *j, const uint8_t *seed, const uint8_t *msg, int mlen, const uint8_t *sig) {
    if (j->op == OP_KG) {
        CK(cudaMemcpy(d_kg->seed, seed, SEEDBYTES, cudaMemcpyHostToDevice));
    } else if (j->op == OP_SG) {
        CK(cudaMemcpy(d_sg->msg, msg, (size_t)mlen, cudaMemcpyHostToDevice));
        { CK(cudaMemcpy(&d_sg->mlen, &mlen, sizeof(int), cudaMemcpyHostToDevice)); }
    } else {
        CK(cudaMemcpy(d_vf->sig, sig, SIGB, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_vf->msg, msg, (size_t)mlen, cudaMemcpyHostToDevice));
        { CK(cudaMemcpy(&d_vf->mlen, &mlen, sizeof(int), cudaMemcpyHostToDevice)); }
    }
}

/* 실행 */
static void run(const job *j) {
    if (j->op == OP_KG)      { cdp::keygen_parent<<<1,1,0,s0>>>(d_kg); }
    else if (j->op == OP_SG) { cdp::sign_parent<<<1,1,0,s0>>>(d_sg); }
    else                     { cdp::verify_parent<<<1,1,0,s0>>>(d_vf); }
}

/* Call only after the parent and descendants have completed. */
static void check_child_error(const job *j) {
    int error=0;
    const int* p=j->op==OP_KG ? &d_kg->cdp_error : j->op==OP_SG ? &d_sg->cdp_error : &d_vf->cdp_error;
    CK(cudaMemcpy(&error,p,sizeof(error),cudaMemcpyDeviceToHost));
    CK(static_cast<cudaError_t>(error));
}

/* 출력 다운로드 -> KeyGen: pk + sk / Sign: sig / Verify: 결과 값 */
static void download(const job *j, uint8_t *pk, uint8_t *sk, uint8_t *sig, int *res) {
    check_child_error(j);
    if (j->op == OP_KG) {
        CK(cudaMemcpy(pk, d_kg->pk, PKB, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(sk, d_kg->sk, SKB, cudaMemcpyDeviceToHost));
    } else if (j->op == OP_SG) {
        CK(cudaMemcpy(sig, d_sg->sig, SIGB, cudaMemcpyDeviceToHost));
    } else {
        CK(cudaMemcpy(res, &d_vf->result, sizeof(int), cudaMemcpyDeviceToHost));
    }
}

/* 정답과 비교 */
static int run_and_check(const job *j, const uint8_t *seed, const uint8_t *msg, const uint8_t *sig_in,
                         const uint8_t *exp_pk, const uint8_t *exp_sk, const uint8_t *exp_sig, int exp_res) {
    static uint8_t pk[PKB], sk[SKB], sig[SIGB];
    int res = 12345;
    upload(j, seed, msg, VC_MLEN, sig_in);
    run(j);
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(s0));
    download(j, pk, sk, sig, &res);
    if (j->op == OP_KG) { return !memcmp(pk, exp_pk, PKB) && !memcmp(sk, exp_sk, SKB); }
    if (j->op == OP_SG) { return !memcmp(sig, exp_sig, SIGB); }
    return res == exp_res;
}

static void bench_input(const job *j, int i, uint8_t *seed, uint8_t *msg, const uint8_t **sig_in,
                        const uint8_t **ep, const uint8_t **es, const uint8_t **esig) {
    const int r = i % j->v->nmsg;
    vec_seed(0, seed);
    vc_message(msg, r);
    *sig_in = vec_sig(j->v, 0, r);
    *ep = vec_pk(j->v, 0); *es = vec_sk(j->v, 0); *esig = vec_sig(j->v, 0, r);
}

/* ------------------------------------------------------------------ 측정 */
typedef struct { double kmean, kmed, emean, emed, iters; int n, bad; } stats;

static int cmp_d(const void *x, const void *y) { double a = *(const double *)x, b = *(const double *)y; return (a > b) - (a < b); }

/* 연산 j 를 n 회 측정. 매 회: (1) 커널 구간  (2) end-to-end  (3) 정답 비교 */
static stats measure(const job *j, int n) {
    stats st; memset(&st, 0, sizeof st);
    double *ker = (double *)malloc(sizeof(double) * n), *e2e = (double *)malloc(sizeof(double) * n);
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    uint8_t seed[SEEDBYTES], msg[VC_MLEN];
    static uint8_t pk[PKB], sk[SKB], sig[SIGB];
    const uint8_t *sig_in, *ep, *es, *esig;
    int res = 0;
    long long iters = 0;
    set_key(j, 0);

    for (int w = 0; w < 5; w++) {                                           /* 워밍업 */
        bench_input(j, w, seed, msg, &sig_in, &ep, &es, &esig);
        upload(j, seed, msg, VC_MLEN, sig_in); run(j); CK(cudaStreamSynchronize(s0));
        download(j, pk, sk, sig, &res);
    }
    for (int i = 0; i < n; i++) {
        bench_input(j, i, seed, msg, &sig_in, &ep, &es, &esig);
        /* (1) 커널 구간 */
        upload(j, seed, msg, VC_MLEN, sig_in);
        CK(cudaEventRecord(e0, s0));
        run(j);
        CK(cudaEventRecord(e1, s0));
        CK(cudaEventSynchronize(e1));
        CK(cudaGetLastError());
        check_child_error(j);
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        ker[i] = ms * 1000.0;
        if (j->op == OP_SG) {
            int it = 0;
            CK(cudaMemcpy(&it, &d_sg->iters, sizeof(int), cudaMemcpyDeviceToHost));
            iters += it;
        }
        /* (2) end-to-end: 업로드 + 커널 + 다운로드 */
        double a = now_us();
        upload(j, seed, msg, VC_MLEN, sig_in);
        run(j);
        CK(cudaStreamSynchronize(s0));
        download(j, pk, sk, sig, &res);
        e2e[i] = now_us() - a;
        /* (3) 정답 비교 */
        int ok = (j->op == OP_KG) ? (!memcmp(pk, ep, PKB) && !memcmp(sk, es, SKB))
               : (j->op == OP_SG) ? !memcmp(sig, esig, SIGB) : (res == 0);
        st.bad += !ok;
    }
    for (int i = 0; i < n; i++) { st.kmean += ker[i]; st.emean += e2e[i]; }
    st.kmean /= n; st.emean /= n;
    qsort(ker, n, sizeof(double), cmp_d); qsort(e2e, n, sizeof(double), cmp_d);
    st.kmed = ker[n / 2]; st.emed = e2e[n / 2];
    st.n = n;
    st.iters = (double)iters / n;
    free(ker); free(e2e);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return st;
}

/* ------------------------------------------------------------------ check: 정확성 */
static int do_check(const vec_set *v) {
    int total_bad = 0;
    uint8_t seed[SEEDBYTES], msg[VC_MLEN];
    static uint8_t pk[PKB], sig[SIGB];

    printf("# 정확성 검사: GPU 출력 = CPU 원본(PQClean) ?\n");
    printf("%-8s %-28s %8s %8s\n", "op", "implementation", "tests", "mismatch");
    for (int op = 0; op < NOP; op++) {
        int tests = 0, bad = 0;
        job j = { op, v };
        if (op == OP_KG) {
            const int ns = v->nseed;
            for (int s = 0; s < ns; s++) {
                vec_seed(s, seed);
                bad += !run_and_check(&j, seed, nullptr, nullptr, vec_pk(v, s), vec_sk(v, s), nullptr, 0); tests++;
            }
        } else if (op == OP_SG) {
            const int nk = v->nkey, nm = v->nmsg;
            for (int k = 0; k < nk; k++) {
                set_key(&j, k);
                for (int r = 0; r < nm; r++) {
                    vc_message(msg, r);
                    bad += !run_and_check(&j, nullptr, msg, nullptr, nullptr, nullptr, vec_sig(v, k, r), 0); tests++;
                }
            }
        } else {
            const int nk = v->nkey, nm = v->nmsg;
            for (int k = 0; k < nk; k++) {
                for (int r = 0; r < nm; r++) {
                    for (int c = 0; c < v->ncase; c++) {
                        vc_apply(c, r, vec_pk(v, k), vec_sig(v, k, r), pk, msg, sig, PKB, SIGB);
                        set_pk_bytes(pk);
                        bad += !run_and_check(&j, nullptr, msg, sig, nullptr, nullptr, nullptr, vec_vres(v, k, r, c)); tests++;
                    }
                }
            }
        }
        const char *name = "parallel ML-DSA";
        printf("%-8s %-28s %8d %8d\n", OP_NAME[op], name, tests, bad);
        total_bad += bad;
    }
    if (total_bad) { printf("결과: 불일치 %d건 ✗ — 성능 값을 신뢰하지 마세요\n", total_bad); }
    else { printf("결과: 모두 일치 (정확성 OK)\n"); }
    return total_bad ? 1 : 0;
}

/* ------------------------------------------------------------------ bench: 통합 병렬 구현 */
static bool parse3(const char *s, int *a) {
    char tail;
    return sscanf(s, "%d,%d,%d%c", &a[0], &a[1], &a[2], &tail) == 3
        && a[0] > 0 && a[1] > 0 && a[2] > 0;
}

static int do_bench(const vec_set *v, const int *n_multi) {
    printf("# %s (µs, 1회당)\n", "Parallel ML-DSA GPU");
    printf("# 커널 = CUDA events, e2e = 입력 업로드 + 커널 + 결과 다운로드. 입력: KeyGen 시드 0 / Sign·Verify 키 0, 메시지 %d개 순환\n", v->nmsg);
    printf("%-8s %-20s %6s %6s %12s %12s %12s %12s %6s\n", "op", "implementation", "parents", "n",
           "kernel mean", "kernel med", "e2e mean", "e2e med", "bad");
    stats r[NOP];
    int bad = 0;
    for (int op = 0; op < NOP; op++) {
        job j = { op, v };
        r[op] = measure(&j, n_multi[op]);
        const stats &s = r[op];
        bad += s.bad;
        printf("%-8s %-20s %6d %6d %12.1f %12.1f %12.1f %12.1f %6d\n", OP_NAME[op],
               "parallel ML-DSA", 1, s.n, s.kmean, s.kmed, s.emean, s.emed, s.bad);
    }
    printf("(다중 스레드 Sign 평균 반복 횟수 %.3f.  측정 중 정답 불일치 %d건, 0 이어야 정상)\n", r[OP_SG].iters, bad);
    return bad ? 1 : 0;
}

/* Command line */
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "사용: %s check|bench <벡터파일> [--multi=KG,SG,VF]\n", argv[0]);
        return 2;
    }
    int n_multi[3] = { 200, 1000, 1000 };
    for (int i = 3; i < argc; i++) {
        if (strncmp(argv[i], "--multi=", 8) || !parse3(argv[i] + 8, n_multi)) {
            fprintf(stderr, "Expected --multi=KG,SG,VF with positive counts\n"); return 2;
        }
    }
    if (strcmp(argv[1], "check") && strcmp(argv[1], "bench")) { fprintf(stderr, "Unknown command\n"); return 2; }
    vec_set v;
    if (vec_read(argv[2], &v) || v.mlen != VC_MLEN || v.ncase != VC_NCASE) { fprintf(stderr, "벡터 파일을 읽을 수 없음: %s\n", argv[2]); return 1; }

    /* Device information */
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 3 || (prop.major==3 && prop.minor<5)) {
        fprintf(stderr,"Dynamic parallelism requires compute capability >= 3.5\n"); return 1;
    }
    fprintf(stderr,"[CDP2] %s: one parent; SHAKE 25 threads/block, arithmetic 128.\n",prop.name);

    CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    CK(cudaMalloc(&d_kg, sizeof(kgm_ws))); CK(cudaMalloc(&d_sg, sizeof(sgm_ws))); CK(cudaMalloc(&d_vf, sizeof(vfm_ws)));
    CK(cudaMemset(d_kg, 0, sizeof(kgm_ws))); CK(cudaMemset(d_sg, 0, sizeof(sgm_ws))); CK(cudaMemset(d_vf, 0, sizeof(vfm_ws)));
    CK(cudaDeviceSynchronize());

    int rc;
    if (!strcmp(argv[1], "check")) { rc = do_check(&v); }
    else { rc = do_bench(&v, n_multi); }
    return rc;
}
