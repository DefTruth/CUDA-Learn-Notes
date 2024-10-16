import torch
import time 
from torch.utils.cpp_extension import load
from functools import partial
from typing import Optional

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(name='hgemm_lib', 
           sources=['hgemm.cu', 'hgemm_async.cu', 'hgemm_wmma.cu', 
                    'hgemm_wmma_stage.cu', 'hgemm_cublas.cu'], 
           extra_cuda_cflags=[
               "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math"
            ], 
           extra_cflags=['-std=c++17'])

MAX_TFLOPS = -1

def run_benchmark(perf_func: callable, 
                  a: torch.Tensor, b: torch.Tensor,
                  tag: str, out: Optional[torch.Tensor] = None, 
                  stages: int = -1, swizzle: bool = False,
                  swizzle_stride: int = 1,
                  warmup: int = 5, iters: int = 20,
                  show_all: bool = False):
    global MAX_TFLOPS

    M = a.size(0)
    K = a.size(1)
    N = b.size(1)
    if (a.size(0) > 1024 or a.size(1) >= 1024 
        or b.size(1) > 1024):
        iters = 10
    
    if swizzle:
        # make swizzle stride as N/4 and multiples of 256
        swizzle_stride = int((int(N / 4) // 256) * 256)
        swizzle_stride = swizzle_stride if swizzle_stride >= 256 else 1
        swizzle = swizzle if swizzle_stride >= 256 else False
    else:
        swizzle_stride = 1 # means no thread block swizzle
    
    if stages:
        assert swizzle_stride is not None

    if out is not None: 
        out.fill_(0)      
    if out is not None:
        for i in range(warmup):
            if stages > 1:
                perf_func(a, b, out, stages, swizzle, swizzle_stride)
            else:
                perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b) 
    
    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            if stages > 1:
                perf_func(a, b, out, stages, swizzle, swizzle_stride)
            else:
                perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b) 
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000 # ms
    mean_time = total_time / iters
    out_info = f"{tag}"
    out_val = out.flatten()[:2].detach().cpu().numpy().tolist()
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}"[:10] for v in out_val]
    TFLOPS = (2 * M * N * K) * 1e-9 / (mean_time)
    mean_time = str(f"{mean_time:<12}")[:8]
    swizzle_stride = 'NOOP' if swizzle_stride == 1 else swizzle_stride

    # caculate TFLOPS improved.
    if TFLOPS > MAX_TFLOPS:
        if MAX_TFLOPS > 0:
            improve = ((TFLOPS - MAX_TFLOPS) / MAX_TFLOPS) * 100
            improve = round(improve, 2)
        else:
            improve = 0
        MAX_TFLOPS = TFLOPS
        print(f"{out_info:>40}: {out_val}, time:{mean_time}ms, "
              f"swizzle: {swizzle_stride:<4}, TFLOPS: {TFLOPS:<6.2f}(+{improve:.2f}%)")
    else:
        print(f"{out_info:>40}: {out_val}, time:{mean_time}ms, "
              f"swizzle: {swizzle_stride:<4}, TFLOPS: {TFLOPS:<6.2f}")
    if show_all: print(out)
    time.sleep(0.05)
    return out, mean_time


Ms = [4096, 8192, 16384]
Ns = [4096, 8192, 16384]
Ks = [2048, 4096, 8192]
MAX_M, MAX_N, MAX_K = 16384, 16384, 8192
# pre allocate for fast profiling.
A = torch.randn((MAX_M, MAX_K), dtype=torch.half).cuda()
B = torch.randn((MAX_K, MAX_N), dtype=torch.half).cuda()
C = torch.randn((MAX_M, MAX_N), dtype=torch.half).cuda()
torch.cuda.synchronize()

MNKs = [(M, N, K) for M in Ms for N in Ns for K in Ks]
for (M, N, K) in MNKs:
    MAX_TFLOPS = -1
    print("-" * 130)
    print(" " * 55 + f"M={M}, N={N}, K={K}")
    a = A[:M, :K].contiguous()
    b = B[:K, :N].contiguous()
    c = C[:M, :N].contiguous()
    torch.cuda.synchronize()

    # CUDA Cores FP16
    # run_benchmark(lib.hgemm_naive_f16, a, b, "f16(naive)",  c)
    run_benchmark(lib.hgemm_t_8x8_sliced_k_f16x8_pack_bcf, a, b, "f16x8pack(t8x8+bcf)", c)
    run_benchmark(lib.hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf, a, b, "f16x8pack(t8x8+bcf+dbuf)", c)
    run_benchmark(lib.hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf, a, b, "f16x8pack(t8x8+k16+dbuf)", c)

    print("-" * 68 + "WMMA" + "-" * 58)
    # run_benchmark(lib.hgemm_wmma_m16n16k16_naive, a, b, "f16wmma(naive)", c)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2, a, b, "f16wmma(mma4x2)", c)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4, a, b, "f16wmma(mma4x2+warp2x4)", c)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_offset, a, b, "f16wmma(mma2x4+warp2x4+dbuf)", c)

    # Stages, dsmem
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages, a, b, "f16wmma(mma2x4+warp2x4+stage4)", c, stages=4)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages, a, b, "f16wmma(mma2x4+warp2x4+stage3)", c, stages=3)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages, a, b, "f16wmma(mma2x4+warp2x4+stage2)", c, stages=2)

    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem, a, b, "f16wmma(mma2x4+...+stage4+dsmem)", c, stages=4)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem, a, b, "f16wmma(mma2x4+...+stage3+dsmem)", c, stages=3)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem, a, b, "f16wmma(mma2x4+...+stage2+dsmem)", c, stages=2)

    # run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem, a, b, "f16wmma(mma4x4+...+stage4+dsmem)", c, stages=4)
    # run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem, a, b, "f16wmma(mma4x4+...+stage3+dsmem)", c, stages=3)
    # run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem, a, b, "f16wmma(mma4x4+...+stage2+dsmem)", c, stages=2)
    
    # Thread block swizzle
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages, a, b, "f16wmma(mma2x4+...+stage4+swizzle)", c, stages=4, swizzle=True)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages, a, b, "f16wmma(mma2x4+...+stage3+swizzle)", c, stages=3, swizzle=True)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages, a, b, "f16wmma(mma2x4+...+stage2+swizzle)", c, stages=2, swizzle=True)

    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem, a, b, "f16wmma(...+stage4+dsmem+swizzle)", c, stages=4, swizzle=True)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem, a, b, "f16wmma(...+stage3+dsmem+swizzle)", c, stages=3, swizzle=True)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem, a, b, "f16wmma(...+stage2+dsmem+swizzle)", c, stages=2, swizzle=True)

    # run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem, a, b, "f16wmma(mma4x4+stage4+dsmem+swizzle)", c, stages=4, swizzle=True)
    # run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem, a, b, "f16wmma(mma4x4+stage3+dsmem+swizzle)", c, stages=3, swizzle=True)
    # run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem, a, b, "f16wmma(mma4x4+stage2+dsmem+swizzle)", c, stages=2, swizzle=True)
    
    run_benchmark(lib.hgemm_cublas_tensor_op, a, b, "f16(cublas)", c)
    run_benchmark(partial(torch.matmul, out=c), a, b, "f16_th")
    torch.cuda.synchronize()
    print("-" * 130)
