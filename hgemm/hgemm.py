import torch
import time 
from torch.utils.cpp_extension import load
from functools import partial
from typing import Optional

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(name='hgemm_lib', 
           sources=['hgemm.cu', 'hgemm_async.cu', 'hgemm_wmma.cu'], 
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


def run_benchmark(perf_func: callable, 
                  a: torch.Tensor, b: torch.Tensor,
                  tag: str, out: Optional[torch.Tensor] = None, 
                  warmup: int = 5, iters: int = 20,
                  show_all: bool = False):
    
    if (a.size(0) > 1024 or a.size(1) >= 1024 
        or b.size(1) > 1024):
        iters = 50

    if out is not None: 
        out.fill_(0)      
    if out is not None:
        for i in range(warmup):
            perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b) 
    
    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b) 
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000 # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>40}: {out_val}, time:{mean_time:.6f}ms")
    if show_all: print(out)
    return out.clone(), mean_time


Ms = [1024, 2048, 4096]
Ns = [1024, 2048, 4096]
Ks = [256,  512,  1024]
MNKs = [(M, N, K) for M in Ms for N in Ns for K in Ks]
for (M, N, K) in MNKs:
    print("-" * 110)
    print(" " * 45 + f"M={M}, N={N}, K={K}")
    a = torch.randn((M, K)).cuda().half().contiguous() 
    b = torch.randn((K, N)).cuda().half().contiguous() 
    c = torch.randn((M, N)).cuda().half().contiguous() 
    run_benchmark(lib.hgemm_naive_f16,                                     
                  a, b, "f16",                               c)
    run_benchmark(lib.hgemm_sliced_k_f16,                                  
                  a, b, "f16(sk)",                           c)
    run_benchmark(lib.hgemm_t_4x4_sliced_k_f16x4_pack_bcf,                 
                  a, b, "f16x4pack(t4x4bcf)",                c)
    run_benchmark(lib.hgemm_t_4x4_sliced_k_f16x4_pack_bcf_offset,          
                  a, b, "f16x4pack(t4x4offset)",             c)
    run_benchmark(lib.hgemm_t_8x8_sliced_k_f16x4,                          
                  a, b, "f16x4(t8x8sk)",                     c)
    run_benchmark(lib.hgemm_t_8x8_sliced_k_f16x4_bcf,                      
                  a, b, "f16x4(t8x8bcf)",                    c)
    run_benchmark(lib.hgemm_t_8x8_sliced_k_f16x4_pack,                     
                  a, b, "f16x4pack(t8x8sk)",                 c)
    run_benchmark(lib.hgemm_t_8x8_sliced_k_f16x4_pack_bcf,                 
                  a, b, "f16x4pack(bcf)",                    c)
    run_benchmark(lib.hgemm_t_8x8_sliced_k_f16x4_pack_bcf_offset,          
                  a, b, "f16x4pack(bcf+offset)",             c)
    run_benchmark(lib.hgemm_t_8x8_sliced_k_f16x8_pack_bcf,                 
                  a, b, "f16x8pack(bcf)",                    c)
    run_benchmark(lib.hgemm_t_8x8_sliced_k_f16x8_pack_bcf_offset,          
                  a, b, "f16x8pack(bcf+offset)",             c)
    run_benchmark(lib.hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf,            
                  a, b, "f16x8pack(bcf+dbuf)",               c)
    print("-" * 52 + "Async" + "-" * 53)
    run_benchmark(lib.hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf,              
                  a, b, "f16x8pack(k16+dbuf)",               c)
    run_benchmark(lib.hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_offset,       
                  a, b, "f16x8pack(k16+dbuf+offset)",        c)
    run_benchmark(lib.hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async,              
                  a, b, "f16x8pack(k16+dbuf+async)",         c)
    run_benchmark(lib.hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf,              
                  a, b, "f16x8pack(k32+dbuf)",               c)
    run_benchmark(lib.hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async,              
                  a, b, "f16x8pack(k32+dbuf+async)",         c)
    run_benchmark(lib.hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf,              
                  a, b, "f16x8pack(k32+dbuf+t16x8)",         c)
    run_benchmark(lib.hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async,              
                  a, b, "f16x8pack(k32+dbuf+t16x8+async)",   c)
    print("-" * 53 + "WMMA" + "-" * 53)
    run_benchmark(lib.hgemm_wmma_m16n16k16_naive,              
                  a, b, "f16wmma(+naive)",                      c)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2,              
                  a, b, "f16wmma(mma4x2)",                      c)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4,              
                  a, b, "f16wmma(mma4x2+warp2x4)",              c)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_async,              
                  a, b, "f16wmma(mma4x2+warp2x4+async)",        c)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_async_offset,              
                  a, b, "f16wmma(mma4x2+warp2x4+async+offset)", c)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async,              
                  a, b, "f16wmma(mma4x2+warp2x4+dbuf)", c)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_offset,              
                  a, b, "f16wmma(mma4x2+warp2x4+dbuf+offset)", c)
    run_benchmark(partial(torch.matmul, out=c),
                  a, b, "f16_th")
    print("-" * 110)
