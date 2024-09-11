import torch
import time 
from torch.utils.cpp_extension import load
from typing import Optional
from functools import partial

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(name='elementwise_lib', 
           sources=['elementwise.cu'], 
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


def run_benchmark(perf_func: callable, a: torch.Tensor, b: torch.Tensor, tag: str, 
                  out: Optional[torch.Tensor] = None, warmup: int = 10, 
                  iters: int = 1000, show_all: bool = False):
    # torch.dot vs custom dot_prod kernel
    if out is not None: 
        out.fill_(0)    
    # warmup
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
    out_val = out.detach().cpu().numpy().tolist()[:2]
    print(f"{out_info:>14}: {out_val}, time:{mean_time:.8f}ms")
    if show_all: print(out)
    return out, mean_time


print("-" * 80)
N_ELEMENTS = 256*92*16
a = torch.randn((N_ELEMENTS)).cuda().float()
b = torch.randn((N_ELEMENTS)).cuda().float()
run_benchmark(lib.elementwise_add_f32,   a, b, "f32")
run_benchmark(lib.elementwise_add_f32x4, a, b, "f32x4")
run_benchmark(torch.add, a, b, "f32_th")

print("-" * 80)
a_f16 = a.half()
b_f16 = b.half()
run_benchmark(lib.elementwise_add_f16,   a_f16, b_f16, "f16")
run_benchmark(lib.elementwise_add_f16x2, a_f16, b_f16, "f16x2")
run_benchmark(torch.add, a_f16, b_f16, "f16_th")

print("-" * 80)
# v2: no copy of c Tensor
c = torch.zeros_like(a).cuda().float()
run_benchmark(lib.elementwise_add_f32_v2,   a, b, "f32(v2)",   c)
run_benchmark(lib.elementwise_add_f32x4_v2, a, b, "f32x4(v2)", c)
run_benchmark(partial(torch.add, out=c),    a, b, "f32_th")

print("-" * 80)
# v2: no copy of c Tensor
c_f16 = torch.zeros_like(a_f16).cuda().half()
run_benchmark(lib.elementwise_add_f16_v2,    a_f16, b_f16, "f16(v2)",   c_f16)
run_benchmark(lib.elementwise_add_f16x2_v2,  a_f16, b_f16, "f16x2(v2)", c_f16)
run_benchmark(partial(torch.add, out=c_f16), a_f16, b_f16, "f16_th")

print("-" * 80)
