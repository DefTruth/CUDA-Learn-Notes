import torch
import time 
from torch.utils.cpp_extension import load
from functools import partial
from typing import Optional

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(name='softmax_lib', 
           sources=['softmax.cu'], 
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


def run_benchmark(perf_func: callable, x: torch.Tensor, 
                  tag: str, out: Optional[torch.Tensor] = None, 
                  warmup: int = 10, iters: int = 1000,
                  show_all: bool = False):
    if out is not None: 
        out.fill_(0)      
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
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>20}: {out_val}, time:{mean_time:.8f}ms")
    if show_all: print(out)
    return out, mean_time


print("-" * 80)
N_ELEMENTS = 256*48
x = torch.randn((N_ELEMENTS)).cuda().float()
run_benchmark(lib.softmax_f32,               x, "f32")
run_benchmark(lib.softmax_f32x4,             x, "f32x4")
run_benchmark(partial(torch.softmax, dim=0), x, "f32_th")

print("-" * 80)
# v2: no copy for out Tensor
out = torch.zeros_like(x).cuda().float().contiguous()
run_benchmark(lib.softmax_f32_v2,                     x, "f32(v2)",   out)
run_benchmark(lib.softmax_f32x4_v2,                   x, "f32x4(v2)", out)
run_benchmark(partial(torch.softmax, dim=0, out=out), x, "f32_th(v2)")

print("-" * 80)
S, H = 1024, 512
x = torch.randn((S, H)).cuda().float().contiguous()
run_benchmark(lib.softmax_f32_per_token,        x, "f32(per)")
run_benchmark(lib.softmax_f32x4_per_token,      x, "f32x4(per)")
run_benchmark(lib.safe_softmax_f32_per_token,   x, "f32(safe)")
run_benchmark(lib.safe_softmax_f32x4_per_token, x, "f32x4(safe)")
run_benchmark(partial(torch.softmax, dim=1),    x, "f32_th(per)")

print("-" * 80)
# v2: no copy for out Tensor
out = torch.zeros_like(x).cuda().float().contiguous()
run_benchmark(lib.softmax_f32_per_token_v2,           x, "f32(per v2)",    out)
run_benchmark(lib.softmax_f32x4_per_token_v2,         x, "f32x4(per v2)",  out)
run_benchmark(lib.safe_softmax_f32_per_token_v2,      x, "f32(safe v2)",   out) 
run_benchmark(lib.safe_softmax_f32x4_per_token_v2,    x, "f32x4(safe v2)", out) 
run_benchmark(partial(torch.softmax, dim=1, out=out), x, "f32_th(per v2)")
print("-" * 80)
