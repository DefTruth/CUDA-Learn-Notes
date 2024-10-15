import torch
import time
import math
from torch.utils.cpp_extension import load
from functools import partial
from typing import Optional
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F
torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="rope",
    sources=["rope.cu"],
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
    extra_cflags=["-std=c++17"],
)


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 2,
    iters: int = 20,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            perf_func(a, out)
    else:
        for i in range(warmup):
            _ = perf_func(a)

    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(a, out)
    else:
        for i in range(iters):
            out = perf_func(a)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>20}: {out_val}, time:{mean_time:.6f}ms")
    if show_all:
        print(out)
    return out.clone(), mean_time


def naive_rope(
    x: torch.Tensor,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = x.shape[-1]
    seq_len = x.shape[-2]
    # get the shape of x (ignore the head dimension). 
    # x: [batch_size, seq_len, dim]
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    # x_: [batch_size, seq_len, dim//2, 2]
    x_ = torch.view_as_complex(x_)
    # pack neibored element into a complex
    # x_: [batch_size, seq_len, dim//2, 1]. eg: tensor([(1.6116-0.5772j), ...]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len , device=freqs.device)
    freqs = torch.outer(t, freqs).float().cuda()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    # get rotate angle
    xq_out = torch.view_as_real(x_ * freqs_cis).flatten(1)
    # do rotate
    return xq_out.type_as(x)

print("-" * 100)
M = [4096, 8192]
N = [512, 1024]
MN = [[m, n] for m in M for n in N]
for M,N in MN:
    print(" " * 40 + f"M={M}, N={N}")
    print("-" * 100)
    x = torch.randn((M, N)).cuda().float().contiguous()
    out = torch.zeros_like(x).cuda().float().contiguous()
    run_benchmark(lib.rope_f32,          x, "f32",          out)
    run_benchmark(lib.rope_f32x4_pack,   x, "f32x4_pack",   out)
    run_benchmark(naive_rope,            x, "f32_th")
    print("-" * 100)     
