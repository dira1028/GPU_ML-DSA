# ML-DSA-87 GPU: 단일 스레드 대 다중 스레드 구현 비교
#
#   make            CDP2 빌드 (build/cdp/), CUDA 12+ 필요
#   make CDP=0      기존 협력 커널 빌드 (build/legacy/)
#   make logical-check   CUDA 없이 Node.js 수학적 대조
#   make check      CDP 출력이 CPU 원본(PQClean)과 같은지 비교
#   make bench      CDP 성능 측정 (CDP=0에서는 단일 대 다중 비교)
#   make CDP=0 sweep  기존 다중 스레드의 블록 수별 시간과 단계별 시간
# 	make res        커널별 레지스터·스택·공유메모리 사용량
# 	make clean
#
# 변수:  ARCH (기본 -arch=sm_87 = Orin),  NB (bench 의 다중 스레드 블록 수 KeyGen,Sign,Verify, 기본 4,8,8)
#        SINGLE / MULTI (bench 측정 횟수, 기본 50,100,100 / 200,1000,1000)
#

NVCC  ?= nvcc
ARCH  ?= -arch=sm_87
CC    ?= gcc
NB     ?= 4,8,8
SINGLE ?= 50,100,100
MULTI  ?= 200,1000,1000

CDP ?= 1
B = build/$(if $(filter 1,$(CDP)),cdp,legacy)
NVCCFLAGS = -O2 -std=c++14 -rdc=true $(ARCH) -Icuda -Ikernels -DMLDSA_CDP=$(CDP)
COMMON_GPU = cuda/reduce.cu cuda/ntt.cu cuda/rounding.cu cuda/poly.cu
LEGACY_GPU = $(COMMON_GPU) cuda/fips202.cu cuda/polyvec.cu cuda/packing.cu \
             cuda/symmetric-shake.cu cuda/keygen_core.cu cuda/sign_core.cu cuda/verify_core.cu
ifeq ($(CDP),1)
GPU_SRC = $(COMMON_GPU) cdp/SHA3.cu
CDPLIBS = -lcudadevrt
else
GPU_SRC = $(LEGACY_GPU)
endif
HDRS  = $(wildcard cuda/*.h kernels/*.cuh multi/*.cuh single/*.cuh test/*.h cdp/*.cuh)
REFSRC = ref/ntt.c ref/reduce.c ref/rounding.c ref/poly.c ref/polyvec.c ref/packing.c \
         ref/symmetric-shake.c ref/sign.c ref/fips202.c

all: $(B)/mldsa87_gpu $(B)/make_vectors

$(B)/mldsa87_gpu: bench/mldsa87_gpu.cu $(GPU_SRC) $(HDRS) Makefile | $(B)
	$(NVCC) $(NVCCFLAGS) $(GPU_SRC) bench/mldsa87_gpu.cu $(CDPLIBS) -o $@

$(B)/make_vectors: test/make_vectors.c $(REFSRC) test/verify_cases.h test/vectors.h | $(B)
	$(CC) -O2 -std=c99 -Iref $(REFSRC) test/make_vectors.c -o $@

# 정답 벡터: KeyGen 시드 20개, Sign 키 3개 x 메시지 100개, Verify 판정 3,600건
$(B)/vectors.bin: $(B)/make_vectors
	$(B)/make_vectors $@ 20 3 100

$(B):
	mkdir -p $(B)

check: $(B)/mldsa87_gpu $(B)/vectors.bin
	$(B)/mldsa87_gpu check $(B)/vectors.bin

bench: $(B)/mldsa87_gpu $(B)/vectors.bin
	$(B)/mldsa87_gpu bench $(B)/vectors.bin --nb=$(NB) --single=$(SINGLE) --multi=$(MULTI)

sweep: $(B)/mldsa87_gpu $(B)/vectors.bin
	$(B)/mldsa87_gpu sweep $(B)/vectors.bin --multi=$(MULTI)

res: $(B)/mldsa87_gpu
	@cuobjdump -res-usage $(B)/mldsa87_gpu

emu-check: $(B)/make_vectors
	@test -f tools/cpu_emulator/emu_build.sh || (echo "CPU emulator is not included; use make logical-check"; exit 1)
	tools/cpu_emulator/emu_build.sh $(B)/mldsa87_gpu_emu -DMLDSA_CDP=0 -Icuda -Ikernels $(LEGACY_GPU) bench/mldsa87_gpu.cu
	$(B)/make_vectors $(B)/vectors_small.bin 4 2 4
	$(B)/mldsa87_gpu_emu check $(B)/vectors_small.bin

clean:
	rm -rf $(B)

.PHONY: logical-check
logical-check:
	node test/cdp_logic.cjs
