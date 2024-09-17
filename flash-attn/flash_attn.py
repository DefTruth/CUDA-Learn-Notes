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
           sources=['flash_attn_1_fwd_f32.cu', 
                    'flash_attn_2_fwd_f16_mma_m16n8k16.cu',
                    'flash_attn.cc'], 
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
                  warmup: int = 10, iters: int = 200,
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
    print(f"{out_info:>17}: {out_val}, time:{mean_time:.8f}ms")
    if show_all: print(out[0, 0, 0, :])
    return out.clone(), mean_time


print("-" * 80)
# batch_size, n_head, seq_len, head_dim (B,nh,N,d)
B, nh, N, d = 16, 12, 64, 64
q = torch.randn(B, nh, N, d).float().cuda().contiguous()
k = torch.randn(B, nh, N, d).float().cuda().contiguous()
v = torch.randn(B, nh, N, d).float().cuda().contiguous()
o = torch.randn(B, nh, N, d).float().cuda().contiguous()
q.requires_grad = False
k.requires_grad = False
v.requires_grad = False
o.requires_grad = False
run_benchmark(lib.flash_attn_1_fwd_f32,    q, k, v, "fa1f32")
run_benchmark(lib.flash_attn_1_fwd_f32_v2, q, k, v, "fa1f32(v2)", o)
run_benchmark(naive_attn,                  q, k, v, "attnf32_th")

print("-" * 80)
# using fp16 Tesor Core mma instruction
q_f16 = q.half()
k_f16 = k.half()
v_f16 = v.half()
o_f16 = o.half()
run_benchmark(lib.flash_attn_2_fwd_f16_mma_m16n8k16,    q_f16, k_f16, v_f16, "fa2mmaf16")
run_benchmark(lib.flash_attn_2_fwd_f16_mma_m16n8k16_v2, q_f16, k_f16, v_f16, "fa2mmaf16(v2)", o_f16)
run_benchmark(naive_attn,                               q_f16, k_f16, v_f16, "attnf16_th")
print("-" * 80)
