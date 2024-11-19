default:
	nvcc hgemm_mma_stage_tn_cute.cu -o hgemm_cute.bin -O2 -arch=sm_89 -std=c++17  -I ../ -I ../third-party/cutlass/include -I ../third-party/cutlass/tools/util/include --expt-relaxed-constexpr -lcublas
