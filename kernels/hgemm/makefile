INCLUDE_DIRS=-I ./utils -I ../../third-party/cutlass/include -I ../../third-party/cutlass/tools/util/include
default:
	nvcc cutlass/hgemm_mma_stage_tn_cute.cu -o hgemm_cute.bin -O2 -arch=sm_89 -std=c++17 $(INCLUDE_DIRS) --expt-relaxed-constexpr -lcublas
	nvcc cublas/hgemm_cublas.cu -o hgemm_cublas.bin -O2 -arch=sm_89 -std=c++17 $(INCLUDE_DIRS) --expt-relaxed-constexpr -lcublas
	nvcc mma/hgemm_mma_stage.cu -o hgemm_mma_stage.bin -O2 -arch=sm_89 -std=c++17 $(INCLUDE_DIRS) --expt-relaxed-constexpr -lcublas

