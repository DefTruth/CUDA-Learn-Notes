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
           sources=['flash_attn.cu', 'flash_attn_mma.cu', 'flash_attn.cc'], 
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
                  warmup: int = 10, iters: int = 100,
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
Ns = [256, 512, 1024]
Ds = [64, 128] # only support [64, 128] now
# batch_size, n_head, seq_len, head_dim (B,nh,N,d)
BHNDs = [(B, H, N, D) for B in Bs for H in Hs for N in Ns for D in Ds]

print("-" * 100)
print(" "* 25 + "B: batch_size, H: n_head, N: seq_len, D: head_dim")
for (B, H, N, D) in BHNDs:
    print("-" * 100)
    print(" " * 40 + f"B={B}, H={H}, N={N}, D={D}")
    q = torch.randn(B, H, N, D).float().cuda().contiguous()
    k = torch.randn(B, H, N, D).float().cuda().contiguous()
    v = torch.randn(B, H, N, D).float().cuda().contiguous()
    o = torch.randn(B, H, N, D).float().cuda().contiguous()
    if D <= 64:
        run_benchmark(lib.flash_attn_1_fwd_f32, 
                      q, k, v, "FA1f32", o)
        run_benchmark(naive_attn,                  
                      q, k, v, "f32_th(naive)")
    
    if D in (64, 128):
        print("-" * 100)
        # using fp16 Tesor Core MMA instruction
        q_f16 = q.half().contiguous()
        k_f16 = k.half().contiguous()
        v_f16 = v.half().contiguous()
        o_f16 = o.half().contiguous()
        run_benchmark(lib.flash_attn_2_fwd_f16_mma_m16n8k16, 
                      q_f16, k_f16, v_f16, "FA2MMAf16", o_f16)
        run_benchmark(naive_attn, 
                      q_f16, k_f16, v_f16, "f16_th(naive)")
    print("-" * 100)
