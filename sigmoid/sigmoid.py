import torch
import time 
from torch.utils.cpp_extension import load
from typing import Optional
from functools import partial

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(name='sigmoid_lib', 
           sources=['sigmoid.cu'], 
           extra_cuda_cflags=[
               "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
            ], 
           extra_cflags=['-std=c++17'])


def run_benchmark(perf_func: callable, x: torch.Tensor, tag: str, 
                  out: Optional[torch.Tensor] = None, warmup: int = 10, 
                  iters: int = 1000, show_all: bool = False):
    if out is not None: 
        out.fill_(0)    
    # warmup
    if out is not None:
        for i in range(warmup):
            perf_func(x, out)
    else:
        for i in range(warmup):
            _ = perf_func(x) 
    torch.cuda.synchronize()

    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(x, out)
    else:
        for i in range(iters):
            out = perf_func(x) 
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000 # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>15}: {out_val}, time:{mean_time:.8f}ms")
    if show_all: print(out)
    return out, mean_time


print("-" * 80)
N_ELEMENTS = 256*92*16
x = torch.randn((N_ELEMENTS)).cuda().float()
run_benchmark(lib.sigmoid_f32,   x, "f32")
run_benchmark(lib.sigmoid_f32x4, x, "f32x4")
run_benchmark(torch.sigmoid, x , "f32_th")

print("-" * 80)
# v2: no copy of y Tensor
y = torch.zeros_like(x).cuda().float()
run_benchmark(lib.sigmoid_f32_v2,   x, "f32(v2)", y)
run_benchmark(lib.sigmoid_f32x4_v2, x, "f32x4(v2)", y)
run_benchmark(partial(torch.sigmoid, out=y), x , "f32_th")
print("-" * 80)
