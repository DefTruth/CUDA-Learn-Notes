INCLUDE_DIRS=-I ./utils -I ../../third-party/cutlass/include -I ../../third-party/cutlass/tools/util/include
ARCHS=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89
ARCHS_80=-gencode arch=compute_80,code=sm_80
ARCHS_89=-gencode arch=compute_89,code=sm_89
DEFAULT_FLAGS=-O2 $(ARCHS) -std=c++17 $(INCLUDE_DIRS) --expt-relaxed-constexpr -lcublas
DEFAULT_FLAGS_89=-O2 $(ARCHS_89) -std=c++17 $(INCLUDE_DIRS) --expt-relaxed-constexpr -lcublas
DEFAULT_FLAGS_80=-O2 $(ARCHS_80) -std=c++17 $(INCLUDE_DIRS) --expt-relaxed-constexpr -lcublas

# Default
default:
	nvcc cutlass/hgemm_mma_stage_tn_cute.cu -o hgemm_cute.bin $(DEFAULT_FLAGS)
	nvcc cublas/hgemm_cublas.cu -o hgemm_cublas.bin $(DEFAULT_FLAGS)
	nvcc mma/basic/hgemm_mma_stage.cu -o hgemm_mma_stage.bin $(DEFAULT_FLAGS)
	nvcc mma/basic/hgemm_mma_stage_tn.cu -o hgemm_mma_stage_tn.bin $(DEFAULT_FLAGS)
	nvcc mma/swizzle/hgemm_mma_stage_swizzle.cu -o hgemm_mma_stage_swizzle.bin $(DEFAULT_FLAGS)
	nvcc mma/swizzle/hgemm_mma_stage_tn_swizzle.cu -o hgemm_mma_stage_tn_swizzle.bin $(DEFAULT_FLAGS)
	nvcc mma/swizzle/hgemm_mma_stage_tn_swizzle_x4.cu -o hgemm_mma_stage_tn_swizzle_x4.bin $(DEFAULT_FLAGS)

# SM 89
cute_89:
	nvcc cutlass/hgemm_mma_stage_tn_cute.cu -o hgemm_cute.89.bin $(DEFAULT_FLAGS_89)
cute_89_debug:
	nvcc cutlass/hgemm_mma_stage_tn_cute.cu -o hgemm_cute.89.debug.bin $(DEFAULT_FLAGS_89) -DCUTE_HGEMM_DEBUG -Xcompiler "-Wno-format"
# SM 89 NN debug
mma_89:
	nvcc mma/basic/hgemm_mma_stage.cu -o hgemm_mma_stage.89.bin $(DEFAULT_FLAGS_89)
mma_89_debug:
	nvcc mma/basic/hgemm_mma_stage.cu -o hgemm_mma_stage.89.debug.bin $(DEFAULT_FLAGS_89) -DHGEMM_MMA_DEBUG
mma_89_swizzle:
	nvcc mma/swizzle/hgemm_mma_stage_swizzle.cu -o hgemm_mma_stage_swizzle.89.bin $(DEFAULT_FLAGS_89)
mma_89_swizzle_debug:
	nvcc mma/swizzle/hgemm_mma_stage_swizzle.cu -o hgemm_mma_stage_swizzle.89.debug.bin $(DEFAULT_FLAGS_89) -DHGEMM_MMA_DEBUG
# SM 89  TN debug
mma_tn_89:
	nvcc mma/basic/hgemm_mma_stage_tn.cu -o hgemm_mma_stage_tn.89.bin $(DEFAULT_FLAGS_89)
mma_tn_89_debug:
	nvcc mma/basic/hgemm_mma_stage_tn.cu -o hgemm_mma_stage_tn.89.debug.bin $(DEFAULT_FLAGS_89) -DHGEMM_MMA_DEBUG
mma_tn_89_swizzle:
	nvcc mma/swizzle/hgemm_mma_stage_tn_swizzle.cu -o hgemm_mma_stage_tn_swizzle.89.bin $(DEFAULT_FLAGS_89)
mma_tn_89_swizzle_debug:
	nvcc mma/swizzle/hgemm_mma_stage_tn_swizzle.cu -o hgemm_mma_stage_tn_swizzle.89.debug.bin $(DEFAULT_FLAGS_89) -DHGEMM_MMA_DEBUG
mma_tn_89_swizzle_x2:
	nvcc mma/swizzle/hgemm_mma_stage_tn_swizzle_x2.cu -o hgemm_mma_stage_tn_swizzle_x2.89.bin $(DEFAULT_FLAGS_89)
mma_tn_89_swizzle_x2_debug:
	nvcc mma/swizzle/hgemm_mma_stage_tn_swizzle_x2.cu -o hgemm_mma_stage_tn_swizzle_x2.89.debug.bin $(DEFAULT_FLAGS_89) -DHGEMM_MMA_DEBUG
mma_tn_89_swizzle_x4:
	nvcc mma/swizzle/hgemm_mma_stage_tn_swizzle_x4.cu -o hgemm_mma_stage_tn_swizzle_x4.89.bin $(DEFAULT_FLAGS_89)
mma_tn_89_swizzle_x4_debug:
	nvcc mma/swizzle/hgemm_mma_stage_tn_swizzle_x4.cu -o hgemm_mma_stage_tn_swizzle_x4.89.debug.bin $(DEFAULT_FLAGS_89) -DHGEMM_MMA_DEBUG

# SM 80
cute_80:
	nvcc cutlass/hgemm_mma_stage_tn_cute.cu -o hgemm_cute.80.bin $(DEFAULT_FLAGS_80)
cute_80_debug:
	nvcc cutlass/hgemm_mma_stage_tn_cute.cu -o hgemm_cute.80.debug.bin $(DEFAULT_FLAGS_80) -DCUTE_HGEMM_DEBUG -Xcompiler "-Wno-format"
# SM 80 TN debug
mma_80:
	nvcc mma/basic/hgemm_mma_stage.cu -o hgemm_mma_stage.80.bin $(DEFAULT_FLAGS_80)
mma_80_debug:
	nvcc mma/basic/hgemm_mma_stage.cu -o hgemm_mma_stage.80.debug.bin $(DEFAULT_FLAGS_80) -DHGEMM_MMA_DEBUG
mma_80_swizzle:
	nvcc mma/swizzle/hgemm_mma_stage_swizzle.cu -o hgemm_mma_stage_swizzle.80.bin $(DEFAULT_FLAGS_80)
mma_80_swizzle_debug:
	nvcc mma/swizzle/hgemm_mma_stage_swizzle.cu -o hgemm_mma_stage_swizzle.80.debug.bin $(DEFAULT_FLAGS_80) -DHGEMM_MMA_DEBUG
# SM 80 TN debug
mma_tn_80:
	nvcc mma/basic/hgemm_mma_stage_tn.cu -o hgemm_mma_stage_tn.80.bin $(DEFAULT_FLAGS_80)
mma_tn_80_debug:
	nvcc mma/basic/hgemm_mma_stage_tn.cu -o hgemm_mma_stage_tn.80.debug.bin $(DEFAULT_FLAGS_80) -DHGEMM_MMA_DEBUG
mma_tn_80_swizzle:
	nvcc mma/swizzle/hgemm_mma_stage_tn_swizzle.cu -o hgemm_mma_stage_tn_swizzle.80.bin $(DEFAULT_FLAGS_80)
mma_tn_80_swizzle_debug:
	nvcc mma/swizzle/hgemm_mma_stage_tn_swizzle.cu -o hgemm_mma_stage_tn_swizzle.80.debug.bin $(DEFAULT_FLAGS_80) -DHGEMM_MMA_DEBUG
mma_tn_80_swizzle_x4:
	nvcc mma/swizzle/hgemm_mma_stage_tn_swizzle_x4.cu -o hgemm_mma_stage_tn_swizzle_x4.80.bin $(DEFAULT_FLAGS_80)
mma_tn_80_swizzle_x4_debug:
	nvcc mma/swizzle/hgemm_mma_stage_tn_swizzle_x4.cu -o hgemm_mma_stage_tn_swizzle_x4.80.debug.bin $(DEFAULT_FLAGS_80) -DHGEMM_MMA_DEBUG

clean:
	rm -rf *.bin & rm -rf ./bin
