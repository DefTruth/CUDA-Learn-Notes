import math
import time
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from functools import partial
from typing import Optional

torch.set_grad_enabled(False)
# Load the CUDA kernel as a python module
lib = load(name='flash_attn_lib', 
           sources=[
               './naive/flash_attn_cuda.cu',
               './mma/flash_attn_mma_old.cu',
                 'pybind/flash_attn.cc'], 
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

# un-fused naive attn
def naive_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y


def run_benchmark(perf_func: callable, 
                  q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  tag: str, out: Optional[torch.Tensor] = None, 
                  warmup: int = 5, iters: int = 10,
                  show_all: bool = False):
    if out is not None: 
        out.fill_(0)      
    if out is not None:
        for i in range(warmup):
            perf_func(q, k, v, out)
    else:
        for i in range(warmup):
            _ = perf_func(q, k, v)
    
    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(q, k, v, out)
    else:
        for i in range(iters):
            out = perf_func(q, k, v)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000 # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>20}: {out_val}, time:{mean_time:.6f}ms")
    if show_all: print(out[0, 0, 0, :])
    return out.clone(), mean_time

Bs = [8, 16]
Hs = [8, 16]
Ns = [1024, 2048, 4096]
Ds = [64, 128] # only support [64, 128] now
# batch_size, n_head, seq_len, head_dim (B,nh,N,d)
BHNDs = [(B, H, N, D) for B in Bs for H in Hs for N in Ns for D in Ds]

print("-" * 100)
print(" "* 25 + "B: batch_size, H: n_head, N: seq_len, D: head_dim")
for (B, H, N, D) in BHNDs:
    print("-" * 100)
    print(" " * 40 + f"B={B}, H={H}, N={N}, D={D}")
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    o = torch.randn(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    torch.cuda.synchronize()
  
    # using fp16 Tesor Core MMA instruction
    run_benchmark(lib.flash_attn_2_fwd_f16_mma_m16n8k16, q, k, v, "FA2MMAf16", o)
    run_benchmark(naive_attn, q, k, v, "f16_th(naive)")
    print("-" * 100)
