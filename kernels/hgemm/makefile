INCLUDE_DIRS=-I ./utils -I ../../third-party/cutlass/include -I ../../third-party/cutlass/tools/util/include
ARCHS=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89
default:
	nvcc cutlass/hgemm_mma_stage_tn_cute.cu -o hgemm_cute.bin -O2 $(ARCHS) -std=c++17 $(INCLUDE_DIRS) --expt-relaxed-constexpr -lcublas
	nvcc cublas/hgemm_cublas.cu -o hgemm_cublas.bin -O2 $(ARCHS) -std=c++17 $(INCLUDE_DIRS) --expt-relaxed-constexpr -lcublas
	nvcc mma/hgemm_mma_stage.cu -o hgemm_mma_stage.bin -O2 $(ARCHS) -std=c++17 $(INCLUDE_DIRS) --expt-relaxed-constexpr -lcublas

