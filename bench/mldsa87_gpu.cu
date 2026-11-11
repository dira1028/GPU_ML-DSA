/*
 * bench/mldsa87_gpu.cu  ---  단일 스레드 대 다중 스레드 GPU 구현
 *
 * 사용법
 *   ./mldsa87_gpu check <벡터파일>                정확성: GPU 출력이 CPU 원본(벡터 파일)과 같은지
 *   ./mldsa87_gpu bench <벡터파일> [옵션]          성능: 같은 입력·같은 측정 방식으로 단일 대 다중 비교
 *   ./mldsa87_gpu sweep <벡터파일> [옵션]          다중 스레드의 블록 수(1·2·4·8)별 시간과 단계별 시간
 *   옵션:  --nb=KG,SG,VF        bench 에서 쓸 다중 스레드 블록 수 (기본 4,8,8)
 *          --single=KG,SG,VF    단일 스레드 측정 횟수 (기본 50,100,100)
 *          --multi=KG,SG,VF     다중 스레드 측정 횟수 (기본 200,1000,1000)
 */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef MLDSA_CDP
#define MLDSA_CDP 1
#endif
#if MLDSA_CDP
#include "../cdp/parent.cuh"
#else
#include "../single/single_kernels.cuh"
#include "../multi/keygen_multi.cuh"
#include "../multi/sign_multi.cuh"
#include "../multi/verify_multi.cuh"
#endif
#include "../test/verify_cases.h"
#include "../test/vectors.h"
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
enum { IMPL_SINGLE = 0, IMPL_MULTI };
static constexpr int FIRST_IMPL = MLDSA_CDP ? IMPL_MULTI : IMPL_SINGLE;

/* GPU 버퍼 */
static cudaStream_t s0;
/* 단일 스레드용 */
static uint8_t *d_seed, *d_pk, *d_sk, *d_m, *d_sig;
static size_t *d_siglen;
static int *d_ret;
#if !MLDSA_CDP
static sign_ws *d_sws;
static verify_ws *d_vws;
#endif
/* 다중 스레드용 작업 공간 */
static kgm_ws *d_kg;
static sgm_ws *d_sg;
static vfm_ws *d_vf;
static int g_nbmax = 1;

static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }

/* 협력 커널 실행  */
template <class... P>
static void coop_launch(void (*kernel)(P...), int nb, P... args) {
    void *argv[] = { (void *)&args... };
    CK(cudaLaunchCooperativeKernel((void *)kernel, dim3(nb), dim3(MT_THREADS), argv, 0, s0));
}

/* 연산별 입력 · 실행 · 출력 */
typedef struct {
    int op, impl, nb;
    const vec_set *v;
} job;

/* 키 설정 */
static void set_key(const job *j, int k) {
    if (j->op == OP_SG) {
        if (j->impl == IMPL_SINGLE) { CK(cudaMemcpy(d_sk, vec_sk(j->v, k), SKB, cudaMemcpyHostToDevice)); }
        else { CK(cudaMemcpy(d_sg->sk, vec_sk(j->v, k), SKB, cudaMemcpyHostToDevice)); }
    } else if (j->op == OP_VF) {
        if (j->impl == IMPL_SINGLE) { CK(cudaMemcpy(d_pk, vec_pk(j->v, k), PKB, cudaMemcpyHostToDevice)); }
        else { CK(cudaMemcpy(d_vf->pk, vec_pk(j->v, k), PKB, cudaMemcpyHostToDevice)); }
    }
}

static void set_pk_bytes(const job *j, const uint8_t *pk) {
    if (j->impl == IMPL_SINGLE) { CK(cudaMemcpy(d_pk, pk, PKB, cudaMemcpyHostToDevice)); }
    else { CK(cudaMemcpy(d_vf->pk, pk, PKB, cudaMemcpyHostToDevice)); }
}

/* 입력 업로드 -> KeyGen: seed / Sign: 메시지 / Verify: 서명 + 메시지 */
static void upload(const job *j, const uint8_t *seed, const uint8_t *msg, int mlen, const uint8_t *sig) {
    const bool s = (j->impl == IMPL_SINGLE);
    if (j->op == OP_KG) {
        CK(cudaMemcpy(s ? d_seed : d_kg->seed, seed, SEEDBYTES, cudaMemcpyHostToDevice));
    } else if (j->op == OP_SG) {
        CK(cudaMemcpy(s ? d_m : d_sg->msg, msg, (size_t)mlen, cudaMemcpyHostToDevice));
        if (!s) { CK(cudaMemcpy(&d_sg->mlen, &mlen, sizeof(int), cudaMemcpyHostToDevice)); }
    } else {
        CK(cudaMemcpy(s ? d_sig : d_vf->sig, sig, SIGB, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(s ? d_m : d_vf->msg, msg, (size_t)mlen, cudaMemcpyHostToDevice));
        if (!s) { CK(cudaMemcpy(&d_vf->mlen, &mlen, sizeof(int), cudaMemcpyHostToDevice)); }
    }
}

/* 실행 */
static void run(const job *j, int mlen) {
#if MLDSA_CDP
    if (j->op == OP_KG)      { cdp::keygen_parent<<<1,1,0,s0>>>(d_kg); }
    else if (j->op == OP_SG) { cdp::sign_parent<<<1,1,0,s0>>>(d_sg); }
    else                     { cdp::verify_parent<<<1,1,0,s0>>>(d_vf); }
#else
    if (j->impl == IMPL_SINGLE) {
        if (j->op == OP_KG)      { single_keygen_kernel<<<1, 1, 0, s0>>>(d_seed, d_pk, d_sk); }
        else if (j->op == OP_SG) { single_sign_kernel<<<1, 1, 0, s0>>>(d_sig, d_siglen, d_m, (size_t)mlen, d_sk, d_sws); }
        else                     { single_verify_kernel<<<1, 1, 0, s0>>>(d_sig, d_m, (size_t)mlen, d_pk, d_vws, d_ret); }
    } else {
        if (j->op == OP_KG)      { coop_launch(keygen_multi_kernel, j->nb, d_kg); }
        else if (j->op == OP_SG) { coop_launch(sign_multi_kernel, j->nb, d_sg); }
        else                     { coop_launch(verify_multi_kernel, j->nb, d_vf); }
    }
#endif
}

/* Call only after the parent and descendants have completed. */
static void check_child_error(const job *j) {
#if MLDSA_CDP
    if (j->impl == IMPL_MULTI) {
        int error=0;
        const int* p=j->op==OP_KG ? &d_kg->cdp_error : j->op==OP_SG ? &d_sg->cdp_error : &d_vf->cdp_error;
        CK(cudaMemcpy(&error,p,sizeof(error),cudaMemcpyDeviceToHost));
        CK(static_cast<cudaError_t>(error));
    }
#endif
}

/* 출력 다운로드 -> KeyGen: pk + sk / Sign: sig / Verify: 결과 값 */
static void download(const job *j, uint8_t *pk, uint8_t *sk, uint8_t *sig, int *res) {
    const bool s = (j->impl == IMPL_SINGLE);
    check_child_error(j);
    if (j->op == OP_KG) {
        CK(cudaMemcpy(pk, s ? d_pk : d_kg->pk, PKB, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(sk, s ? d_sk : d_kg->sk, SKB, cudaMemcpyDeviceToHost));
    } else if (j->op == OP_SG) {
        CK(cudaMemcpy(sig, s ? d_sig : d_sg->sig, SIGB, cudaMemcpyDeviceToHost));
    } else {
        CK(cudaMemcpy(res, s ? d_ret : &d_vf->result, sizeof(int), cudaMemcpyDeviceToHost));
    }
}

/* 정답과 비교 */
static int run_and_check(const job *j, const uint8_t *seed, const uint8_t *msg, const uint8_t *sig_in,
                         const uint8_t *exp_pk, const uint8_t *exp_sk, const uint8_t *exp_sig, int exp_res) {
    static uint8_t pk[PKB], sk[SKB], sig[SIGB];
    int res = 12345;
    upload(j, seed, msg, VC_MLEN, sig_in);
    run(j, VC_MLEN);
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
typedef struct { double kmean, kmed, emean, emed, iters, ph[8]; int n, bad, nph; } stats;

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
        upload(j, seed, msg, VC_MLEN, sig_in); run(j, VC_MLEN); CK(cudaStreamSynchronize(s0));
        download(j, pk, sk, sig, &res);
    }
    for (int i = 0; i < n; i++) {
        bench_input(j, i, seed, msg, &sig_in, &ep, &es, &esig);
        /* (1) 커널 구간 */
        upload(j, seed, msg, VC_MLEN, sig_in);
        CK(cudaEventRecord(e0, s0));
        run(j, VC_MLEN);
        CK(cudaEventRecord(e1, s0));
        CK(cudaEventSynchronize(e1));
        CK(cudaGetLastError());
        check_child_error(j);
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        ker[i] = ms * 1000.0;
        /* 다중 스레드 */
        if (j->impl == IMPL_MULTI) {
            unsigned long long ph[8] = { 0 };
            if (j->op == OP_KG) { st.nph = KGM_NPH; CK(cudaMemcpy(ph, d_kg->ph, sizeof(unsigned long long) * KGM_NPH, cudaMemcpyDeviceToHost)); }
            if (j->op == OP_SG) {
                int it = 0;
                st.nph = SGT_N; CK(cudaMemcpy(ph, d_sg->ph, sizeof(unsigned long long) * SGT_N, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&it, &d_sg->iters, sizeof(int), cudaMemcpyDeviceToHost)); iters += it;
            }
            if (j->op == OP_VF) { st.nph = VFT_N; CK(cudaMemcpy(ph, d_vf->ph, sizeof(unsigned long long) * VFT_N, cudaMemcpyDeviceToHost)); }
#if MLDSA_CDP
            st.nph=0; // Legacy in-kernel phase counters do not measure CDP child grids.
#endif
            for (int p = 0; p < st.nph; p++) { st.ph[p] += ph[p] / 1000.0; }
        }
        /* (2) end-to-end: 업로드 + 커널 + 다운로드 */
        double a = now_us();
        upload(j, seed, msg, VC_MLEN, sig_in);
        run(j, VC_MLEN);
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
    for (int p = 0; p < st.nph; p++) { st.ph[p] /= n; }
    free(ker); free(e2e);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return st;
}

/* ------------------------------------------------------------------ check: 정확성 */
static int nb_list(int *nbs) {
#if MLDSA_CDP
    nbs[0]=1; return 1; // One parent; child grids have task-specific fixed sizes.
#endif
 int n = 0; for (int nb = 1; nb <= 8; nb *= 2) { if (nb <= g_nbmax) { nbs[n++] = nb; } } return n; }

static int do_check(const vec_set *v) {
    int nbs[4], nn = nb_list(nbs), total_bad = 0;
    uint8_t seed[SEEDBYTES], msg[VC_MLEN];
    static uint8_t pk[PKB], sig[SIGB];
    const int s_seed = v->nseed < 3 ? v->nseed : 3, s_msg = v->nmsg < 10 ? v->nmsg : 10;

    printf("# 정확성 검사: GPU 출력 = CPU 원본(PQClean) ?\n");
    printf("%-8s %-28s %8s %8s\n", "op", "implementation", "tests", "mismatch");
    for (int impl = FIRST_IMPL; impl <= IMPL_MULTI; impl++) {
        const int nimpl = (impl == IMPL_SINGLE) ? 1 : nn;
        for (int op = 0; op < NOP; op++) {
            int tests = 0, bad = 0;
            for (int q = 0; q < nimpl; q++) {
                job j = { op, impl, impl == IMPL_SINGLE ? 1 : nbs[q], v };
                if (op == OP_KG) {
                    const int ns = impl == IMPL_SINGLE ? s_seed : v->nseed;
                    for (int s = 0; s < ns; s++) {
                        vec_seed(s, seed);
                        bad += !run_and_check(&j, seed, nullptr, nullptr, vec_pk(v, s), vec_sk(v, s), nullptr, 0); tests++;
                    }
                } else if (op == OP_SG) {
                    const int nk = impl == IMPL_SINGLE ? 1 : v->nkey, nm = impl == IMPL_SINGLE ? s_msg : v->nmsg;
                    for (int k = 0; k < nk; k++) {
                        set_key(&j, k);
                        for (int r = 0; r < nm; r++) {
                            vc_message(msg, r);
                            bad += !run_and_check(&j, nullptr, msg, nullptr, nullptr, nullptr, vec_sig(v, k, r), 0); tests++;
                        }
                    }
                } else {
                    const int nk = impl == IMPL_SINGLE ? 1 : v->nkey, nm = impl == IMPL_SINGLE ? s_msg : v->nmsg;
                    for (int k = 0; k < nk; k++) {
                        for (int r = 0; r < nm; r++) {
                            for (int c = 0; c < v->ncase; c++) {
                                vc_apply(c, r, vec_pk(v, k), vec_sig(v, k, r), pk, msg, sig, PKB, SIGB);
                                set_pk_bytes(&j, pk);
                                bad += !run_and_check(&j, nullptr, msg, sig, nullptr, nullptr, nullptr, vec_vres(v, k, r, c)); tests++;
                            }
                        }
                    }
                }
            }
            char name[64];
            if (impl == IMPL_SINGLE) { snprintf(name, sizeof name, "single (1 thread)"); }
            else if (MLDSA_CDP) { snprintf(name, sizeof name, "CDP parent"); }
            else { snprintf(name, sizeof name, "multi (blocks 1..%d, %d kinds)", nbs[nn - 1], nn); }
            printf("%-8s %-28s %8d %8d\n", OP_NAME[op], name, tests, bad);
            total_bad += bad;
        }
    }
    if (total_bad) { printf("결과: 불일치 %d건 ✗ — 성능 값을 신뢰하지 마세요\n", total_bad); }
    else { printf("결과: 모두 일치 (정확성 OK)\n"); }
    return total_bad ? 1 : 0;
}

/* ------------------------------------------------------------------ bench: 단일 대 다중 */
static void parse3(const char *s, int *a) { sscanf(s, "%d,%d,%d", &a[0], &a[1], &a[2]); }

static int do_bench(const vec_set *v, const int *nb, const int *n_single, const int *n_multi) {
    printf("# %s (µs, 1회당)\n", MLDSA_CDP ? "CDP2 GPU" : "단일 스레드 대 다중 스레드 GPU 구현");
    printf("# 커널 = CUDA events, e2e = 입력 업로드 + 커널 + 결과 다운로드. 입력: KeyGen 시드 0 / Sign·Verify 키 0, 메시지 %d개 순환\n", v->nmsg);
    printf("%-8s %-20s %6s %6s %12s %12s %12s %12s %6s\n", "op", "implementation", "blocks", "n",
           "kernel mean", "kernel med", "e2e mean", "e2e med", "bad");
    stats r[NOP][2];
    int bad = 0;
    for (int op = 0; op < NOP; op++) {
        for (int impl = FIRST_IMPL; impl <= IMPL_MULTI; impl++) {
            job j = { op, impl, impl == IMPL_SINGLE ? 1 : nb[op], v };
            r[op][impl] = measure(&j, impl == IMPL_SINGLE ? n_single[op] : n_multi[op]);
            const stats &s = r[op][impl];
            bad += s.bad;
            printf("%-8s %-20s %6d %6d %12.1f %12.1f %12.1f %12.1f %6d\n", OP_NAME[op],
                   impl == IMPL_SINGLE ? "single (1 thread)" : (MLDSA_CDP ? "CDP parent" : "multi"), j.nb, s.n, s.kmean, s.kmed, s.emean, s.emed, s.bad);
        }
    }
#if !MLDSA_CDP
    printf("\n# 속도 향상 (단일 / 다중)\n%-8s %12s %12s %12s %12s\n", "op", "kernel mean", "kernel med", "e2e mean", "e2e med");
    for (int op = 0; op < NOP; op++) {
        const stats &a = r[op][0], &b = r[op][1];
        printf("%-8s %11.1fx %11.1fx %11.1fx %11.1fx\n", OP_NAME[op], a.kmean / b.kmean, a.kmed / b.kmed, a.emean / b.emean, a.emed / b.emed);
    }
#endif
    printf("(다중 스레드 Sign 평균 반복 횟수 %.3f.  측정 중 정답 불일치 %d건, 0 이어야 정상)\n", r[OP_SG][1].iters, bad);
    return bad ? 1 : 0;
}

/* ------------------------------------------------------------------ sweep: 블록 수별 */
static int do_sweep(const vec_set *v, const int *n_multi) {
    int nbs[4], nn = nb_list(nbs), bad = 0;
    for (int op = 0; op < NOP; op++) {
        stats r[4];
        for (int q = 0; q < nn; q++) { job j = { op, IMPL_MULTI, nbs[q], v }; r[q] = measure(&j, n_multi[op]); bad += r[q].bad; }
        printf("# %s 다중 스레드: 블록 수별 (µs, %d회)\n%-36s", OP_NAME[op], n_multi[op], "");
        for (int q = 0; q < nn; q++) { printf(" %9db", nbs[q]); }
        printf("\n%-36s", "kernel mean");
        for (int q = 0; q < nn; q++) { printf(" %10.1f", r[q].kmean); }
        printf("\n%-36s", "kernel median");
        for (int q = 0; q < nn; q++) { printf(" %10.1f", r[q].kmed); }
        printf("\n");
        const char *const *names = op == OP_KG ? KGM_PHASE : op == OP_SG ? SGT_NAME : VFT_NAME;
        for (int p = 0; p < r[0].nph; p++) {
            printf("  %-34s", names[p]);
            for (int q = 0; q < nn; q++) { printf(" %10.1f", r[q].ph[p]); }
            printf("\n");
        }
        if (op == OP_SG) { printf("  (* = 기각 반복 루프 안, 반복 합.  평균 반복 %.3f)\n", r[0].iters); }
        printf("\n");
    }
    printf("(측정 중 정답 불일치 %d건, 0 이어야 정상)\n", bad);
    return bad ? 1 : 0;
}

/* ------------------------------------------------------------------ main */
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "사용: %s check|bench|sweep <벡터파일> [--nb=KG,SG,VF] [--single=KG,SG,VF] [--multi=KG,SG,VF]\n", argv[0]);
        return 2;
    }
    int nb[3] = { 4, 8, 8 }, n_single[3] = { 50, 100, 100 }, n_multi[3] = { 200, 1000, 1000 };
    for (int i = 3; i < argc; i++) {
        if (!strncmp(argv[i], "--nb=", 5)) { parse3(argv[i] + 5, nb); }
        else if (!strncmp(argv[i], "--single=", 9)) { parse3(argv[i] + 9, n_single); }
        else if (!strncmp(argv[i], "--multi=", 8)) { parse3(argv[i] + 8, n_multi); }
    }
    vec_set v;
    if (vec_read(argv[2], &v) || v.mlen != VC_MLEN || v.ncase != VC_NCASE) { fprintf(stderr, "벡터 파일을 읽을 수 없음: %s\n", argv[2]); return 1; }

    /* 장치 정보와 협력 실행 가능한 블록 수 (세 커널 모두 SM 당 블록 수 × SM 수) */
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
#if MLDSA_CDP
    if (prop.major < 3 || (prop.major==3 && prop.minor<5)) {
        fprintf(stderr,"Dynamic parallelism requires compute capability >= 3.5\n"); return 1;
    }
    for (int i=0;i<3;++i) nb[i]=1;
    g_nbmax=1;
    fprintf(stderr,"[CDP2] %s: one parent; SHAKE 25 threads/block, arithmetic 128. --nb ignored; phase sweep unavailable.\n",prop.name);
#else
    int pk_ = 0, ps_ = 0, pv_ = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&pk_, keygen_multi_kernel, MT_THREADS, 0));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ps_, sign_multi_kernel, MT_THREADS, 0));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&pv_, verify_multi_kernel, MT_THREADS, 0));
    int per = pk_ < ps_ ? pk_ : ps_; per = per < pv_ ? per : pv_;
    g_nbmax = per * prop.multiProcessorCount;
    if (!prop.cooperativeLaunch || g_nbmax < 1) { fprintf(stderr, "이 GPU 는 협력 실행(cooperative launch)을 지원하지 않음\n"); return 1; }
    for (int i = 0; i < 3; i++) { if (nb[i] > g_nbmax) { nb[i] = g_nbmax; } }
    fprintf(stderr, "[device] %s, SM %d개, 다중 스레드 커널 SM당 최대 %d블록 → 블록 수 최대 %d\n",
            prop.name, prop.multiProcessorCount, per, g_nbmax);
#endif

    CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
#if !MLDSA_CDP
    CK(cudaMalloc(&d_seed, SEEDBYTES)); CK(cudaMalloc(&d_pk, PKB)); CK(cudaMalloc(&d_sk, SKB));
    CK(cudaMalloc(&d_m, VC_MLEN)); CK(cudaMalloc(&d_sig, SIGB)); CK(cudaMalloc(&d_siglen, sizeof(size_t)));
    CK(cudaMalloc(&d_ret, sizeof(int))); CK(cudaMalloc(&d_sws, sizeof(sign_ws))); CK(cudaMalloc(&d_vws, sizeof(verify_ws)));
#endif
    CK(cudaMalloc(&d_kg, sizeof(kgm_ws))); CK(cudaMalloc(&d_sg, sizeof(sgm_ws))); CK(cudaMalloc(&d_vf, sizeof(vfm_ws)));
    CK(cudaMemset(d_kg, 0, sizeof(kgm_ws))); CK(cudaMemset(d_sg, 0, sizeof(sgm_ws))); CK(cudaMemset(d_vf, 0, sizeof(vfm_ws)));
    CK(cudaDeviceSynchronize());

    int rc;
    if (!strcmp(argv[1], "check")) { rc = do_check(&v); }
    else if (!strcmp(argv[1], "sweep")) {
#if MLDSA_CDP
        fprintf(stderr,"CDP uses fixed child grids: use bench, or make CDP=0 sweep for legacy block sweeps.\n"); rc=2;
#else
        rc = do_sweep(&v, n_multi);
#endif
    }
    else { rc = do_bench(&v, nb, n_single, n_multi); }
    return rc;
}
