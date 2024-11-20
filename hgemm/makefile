INCLUDE_DIRS=-I ../ -I ../third-party/cutlass/include -I ../third-party/cutlass/tools/util/include
default:
	nvcc hgemm_mma_stage_tn_cute.cu -o hgemm_cute.bin -O2 -arch=sm_89 -std=c++17 $(INCLUDE_DIRS) --expt-relaxed-constexpr -lcublas
	nvcc hgemm_cublas.cu -o hgemm_cublas.bin -O2 -arch=sm_89 -std=c++17 $(INCLUDE_DIRS) --expt-relaxed-constexpr -lcublas
	nvcc hgemm_mma_stage.cu -o hgemm_mma_stage.bin -O2 -arch=sm_89 -std=c++17 $(INCLUDE_DIRS) --expt-relaxed-constexpr -lcublas

