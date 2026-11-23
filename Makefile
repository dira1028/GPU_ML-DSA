# Integrated parallel ML-DSA-87 (CUDA 12+ CDP2 and 25-lane SHAKE)
NVCC ?= nvcc
CC ?= gcc
ARCH ?= -arch=sm_87
MULTI ?= 200,1000,1000
B = build/parallel
NVCCFLAGS = -O2 -std=c++14 -rdc=true $(ARCH) -Isrc/primitives -Isrc/common -Isrc/single
GPU_SRC = $(wildcard src/primitives/*.cu src/common/*.cu src/parallel/*.cu src/single/*.cu)
HDRS = $(wildcard src/primitives/*.h src/common/*.cuh src/parallel/*.cuh src/single/*.h src/single/*.cuh tests/*.h)
REFSRC = $(wildcard reference/*.c)
.PHONY: all bench res clean logical-check
all: $(B)/mldsa87_gpu $(B)/make_vectors
$(B)/mldsa87_gpu: bench/mldsa87_gpu.cu $(GPU_SRC) $(HDRS) Makefile | $(B)
	$(NVCC) $(NVCCFLAGS) $(GPU_SRC) $< -lcudadevrt -o $@
$(B)/make_vectors: tests/make_vectors.c $(REFSRC) $(wildcard reference/*.h tests/*.h) Makefile | $(B)
	$(CC) -O2 -std=c99 -Ireference $(REFSRC) $< -o $@
$(B)/vectors.bin: $(B)/make_vectors
	$< $@ 20 3 100
$(B):
	mkdir -p $(B)
bench: $(B)/mldsa87_gpu $(B)/vectors.bin
	$(B)/mldsa87_gpu bench $(B)/vectors.bin --multi=$(MULTI)
res: $(B)/mldsa87_gpu
	cuobjdump -res-usage $<
logical-check:
	node tests/cdp_logic.cjs
clean:
	rm -rf $(B)
