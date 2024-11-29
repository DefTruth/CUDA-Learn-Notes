INCLUDE_DIRS=-I ./utils -I ../../third-party/cutlass/include -I ../../third-party/cutlass/tools/util/include
ARCHS=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89
ARCHS_89=-gencode arch=compute_89,code=sm_89
DEFAULT_FLAGS=-O2 $(ARCHS) -std=c++17 $(INCLUDE_DIRS) --expt-relaxed-constexpr -lcublas
DEFAULT_FLAGS_89=-O2 $(ARCHS_89) -std=c++17 $(INCLUDE_DIRS) --expt-relaxed-constexpr -lcublas
default:
	nvcc cutlass/hgemm_mma_stage_tn_cute.cu -o hgemm_cute.bin $(DEFAULT_FLAGS)
	nvcc cublas/hgemm_cublas.cu -o hgemm_cublas.bin $(DEFAULT_FLAGS)
	nvcc mma/hgemm_mma_stage.cu -o hgemm_mma_stage.bin $(DEFAULT_FLAGS)
cute_89:
	nvcc cutlass/hgemm_mma_stage_tn_cute.cu -o hgemm_cute.89.bin $(DEFAULT_FLAGS_89)
cute_89_debug:
	nvcc cutlass/hgemm_mma_stage_tn_cute.cu -o hgemm_cute.89.debug.bin $(DEFAULT_FLAGS_89) -DCUTE_HGEMM_DEBUG
mma_89:
	nvcc mma/hgemm_mma_stage.cu -o hgemm_mma_stage.89.bin $(DEFAULT_FLAGS_89)
clean:
	rm -rf *.bin
