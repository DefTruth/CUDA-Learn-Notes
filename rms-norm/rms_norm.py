import torch
import time 
from torch.utils.cpp_extension import load
from functools import partial
from typing import Optional

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(name='rms_norm_lib', 
           sources=['rms_norm.cu'], 
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


# un-fused naive rms norm
def naive_rms_norm(x: torch.Tensor, g: float):
    # y'=x/rms(x) 1/rms(x) = rsqrtf(sum(x^2)/K)
    s_rms = torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True))
    y = (x * s_rms) * g
    return y


def run_benchmark(perf_func: callable, x: torch.Tensor, 
                  tag: str, out: Optional[torch.Tensor] = None, 
                  warmup: int = 10, iters: int = 1000,
                  show_all: bool = False):
    g = 1.0
    if out is not None: 
        out.fill_(0)      
    if out is not None:
        for i in range(warmup):
            perf_func(x, out, g)
    else:
        for i in range(warmup):
            _ = perf_func(x, g) 
    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(x, out, g)
    else:
        for i in range(iters):
            out = perf_func(x, g) 
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000 # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>12}: {out_val}, time:{mean_time:.8f}ms")
    if show_all: print(out)
    return out, mean_time


print("-" * 80)
N, K = 4096, 512
x = torch.randn((N, K)).cuda().float().contiguous()
out = torch.zeros_like(x).cuda().float().contiguous()
run_benchmark(lib.rms_norm_f32,   x, "f32",   out)
run_benchmark(lib.rms_norm_f32x4, x, "f32x4", out)
run_benchmark(naive_rms_norm,     x, "f32_th")
print("-" * 80)
