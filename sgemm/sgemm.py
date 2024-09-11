import torch
import time 
from torch.utils.cpp_extension import load
from functools import partial
from typing import Optional

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(name='sgemm_lib', 
           sources=['sgemm.cu'], 
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
                  warmup: int = 10, iters: int = 200,
                  show_all: bool = False):
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
    print(f"{out_info:>17}: {out_val}, time:{mean_time:.8f}ms")
    if show_all: print(out)
    return out.clone(), mean_time


print("-" * 80)
M, N, K = 2048, 1024, 128
a = torch.randn((M, K)).cuda().float().contiguous() 
b = torch.randn((K, N)).cuda().float().contiguous() 
c = torch.randn((M, N)).cuda().float().contiguous() 
run_benchmark(lib.sgemm_sliced_k_f32, a, b, "f32(sk)", c)
run_benchmark(lib.sgemm_t_8x8_sliced_k_f32x4, a, b, "f32x4(t8x8sk)", c)
run_benchmark(partial(torch.matmul, out=c), a, b, "f32_th")
print("-" * 80)

