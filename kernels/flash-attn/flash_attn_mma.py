import os
import math
import time
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from typing import Optional
from flash_attn import flash_attn_func
import argparse
import random
import numpy as np

torch.set_grad_enabled(False)
torch.set_printoptions(precision=6, threshold=8, edgeitems=3, 
                       linewidth=120, sci_mode=False)


def set_rand_seed(seed:int=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_project_dir():
    return os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))


def pretty_print_line(m: str = "", sep: str = "-", width: int = 130):
    res_len = width - len(m)
    left_len = int(res_len / 2)
    right_len = res_len - left_len
    pretty_line = sep * left_len + m + sep * right_len
    print(pretty_line)


project_dir = get_project_dir()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rand-q", '--no-rq', action="store_true")
    parser.add_argument("--no-rand-k", '--no-rk', action="store_true")
    parser.add_argument("--no-rand-v", '--no-rv', action="store_true")
    parser.add_argument("--no-rand-qkv", '--no-rqkv', action="store_true")
    parser.add_argument("--run-torch-unfused", '--torch', action="store_true")
    parser.add_argument("--run-torch-sdpa", '--sdpa', action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-all", action="store_true")
    parser.add_argument("--show-all", '--show', action="store_true")
    parser.add_argument("--show-matrix", action="store_true")
    parser.add_argument("--only-flops-matmul", "--flops-mm", action="store_true")
    parser.add_argument("--B", type=int, default=None)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--D", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", '--v', action="store_true")
    parser.add_argument("--warmup", "--w", type=int, default=1)
    parser.add_argument("--iters", "--i", type=int, default=5)
    parser.add_argument("--range-k", '--gk', action="store_true")
    return parser.parse_args()


args = get_args()
pretty_print_line()
print(args)
pretty_print_line()

# Load the CUDA kernel as a python module
lib = load(name='flash_attn_lib', 
           sources=[
               './mma/flash_attn_mma_split_kv.cu',
               './mma/flash_attn_mma_split_q.cu',
               './mma/flash_attn_mma_share_kv.cu',
               './mma/flash_attn_mma_share_qkv.cu',
               './pybind/flash_attn.cc'
            ], 
           extra_cuda_cflags=[
               "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "-Xptxas -v",
                "-diag-suppress 177",
                f"-I {project_dir}/kernels/flash-attn/utils",
                "-DFLASH_ATTN_MMA_DEBUG" if args.debug else ""
            ], 
           extra_cflags=['-std=c++17'],
           verbose=args.verbose)


def get_mha_tflops(B: int, H: int, N: int, D: int, secs: float=1.0, 
                   only_matmul: bool = False):
    # Q @ K^T FLOPs
    flops_qk = B * H * N * N * (2 * D - 1)
    
    # Scaling FLOPs
    flops_scaling = B * H * N * N
    
    # Safe_Softmax FLOPs
    flops_row_max = B * H * N * (N - 1)   # row max
    flops_subtract_max = B * H * N * N    # sub max
    flops_exp = B * H * N * N             # pointwise exp
    flops_row_sum = B * H * N * (N - 1)   # row sum
    flops_normalization = B * H * N * N   # normalization
    
    flops_safe_softmax = (flops_row_max + flops_subtract_max + flops_exp 
                          + flops_row_sum + flops_normalization)
    
    # P @ V FLOPs
    flops_pv = B * H * N * D * (2 * N - 1)
    
    # Total FLOPs
    total_flops = flops_qk + flops_scaling + flops_safe_softmax + flops_pv
    if only_matmul:
        total_flops = flops_qk + flops_pv
    
    # Convert to TFLOPS
    # 1 TFLOPS = 10^12 FLOPS
    # ref: https://imgtec.eetrend.com/blog/2021/100062210.html.
    tflops = total_flops * 1e-12 / (secs)
    
    return tflops


MAX_TFLOPS = -1
STATIS_INFO: dict[str, list[float]] = {}
STATIS_INFO["MNK"] = []
TOATL_TFLOPS: dict[str, float] = {}


def run_benchmark(perf_func: callable, 
                  q: torch.Tensor, 
                  k: torch.Tensor, 
                  v: torch.Tensor,
                  tag: str, 
                  out: Optional[torch.Tensor] = None, 
                  s: Optional[torch.Tensor] = None, # DEBUG
                  stages: int = -1,
                  warmup: int = args.warmup, 
                  iters: int = args.iters,
                  show_matrix: bool = args.show_matrix,
                  only_show_improved: bool = not args.show_all):
    
    global MAX_TFLOPS

    if out is not None: 
        out.fill_(0)
    if s is not None:
        s.fill_(0)      
    if out is not None:
        for i in range(warmup):
            if stages >= 1:
                if s is not None:
                    perf_func(q, k, v, out, s, stages)
                else:
                    perf_func(q, k, v, out, stages)
            else:
                perf_func(q, k, v, out)
    else:
        for i in range(warmup):
            _ = perf_func(q, k, v)
    
    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            if stages >= 1:
                if s is not None:
                    perf_func(q, k, v, out, s, stages)
                else:
                    perf_func(q, k, v, out, stages)
            else:
                perf_func(q, k, v, out)
    else:
        for i in range(iters):
            out = perf_func(q, k, v)
    torch.cuda.synchronize()
    end = time.time()
    total_secs = (end - start)
    total_time = (end - start) * 1000 # ms
    mean_time = total_time / iters
    mean_secs = total_secs / iters
    B, H, N, D = q.size()
    if "flash" in tag:
        B, N, H, D = q.size()
    TFLOPS = get_mha_tflops(B, H, N, D, mean_secs, 
                            only_matmul=args.only_flops_matmul)
    out_info = f"{tag}"
    out_val_first = out.flatten()[:3].detach().cpu().numpy().tolist()
    out_val_last = out.flatten()[-3:].detach().cpu().numpy().tolist()
    out_val_first = [round(v, 8) for v in out_val_first]
    out_val_last = [round(v, 8) for v in out_val_last]
    out_val = out_val_first[:2]
    out_val.append(out_val_last[-1])
    out_val = [f"{v:<12}" for v in out_val]

    # caculate TFLOPS improved.
    if TFLOPS > MAX_TFLOPS:
        if MAX_TFLOPS > 0:
            improve = ((TFLOPS - MAX_TFLOPS) / MAX_TFLOPS) * 100
            improve = round(improve, 2)
        else:
            improve = 0
        MAX_TFLOPS = TFLOPS
        print(f"{out_info:>32}: {out_val}, time:{mean_time:<.6f}ms, "
              f"TFLOPS:{TFLOPS:<6.2f}(+{improve:.2f}%)")
    else:
        if not only_show_improved or "flash" in tag:
            print(f"{out_info:>32}: {out_val}, time:{mean_time:<.6f}ms, "
                  f"TFLOPS:{TFLOPS:<6.2f}")
            
    if show_matrix: print(out)
    time.sleep(args.sleep)
    torch.cuda.synchronize()
    return out.clone(), mean_time


def get_qkvo(B, H, N, D):
    if not (args.no_rand_q or args.no_rand_qkv):
        q = torch.randn((B, H, N, D), dtype=torch.half, device="cuda")
    else:
        q = torch.ones(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    if not (args.no_rand_k or args.no_rand_qkv):
        k = torch.randn((B, H, N, D), dtype=torch.half, device="cuda")
    else:
        k = torch.ones(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
        if args.range_k:
            for i in range(N):
                k[:, :, i, :] = (i + 1) / N
            k = k.cuda().half().contiguous()
    if not (args.no_rand_v or args.no_rand_qkv):
        v = torch.randn((B, H, N, D), dtype=torch.half, device="cuda")
    else:
        v = torch.ones(B, H, N, D, device="cuda", dtype=torch.half).contiguous()

    o = torch.zeros(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    # transpose (H,N) -> (N,H) for FA2.
    fq = q.transpose(1,   2).contiguous()
    fk = k.transpose(1,   2).contiguous()
    fv = v.transpose(1,   2).contiguous()

    return q, k, v, o, fq, fk, fv


# un-fused naive attn
def unfused_standard_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y


def check_all_close(out_flash: torch.Tensor, out_mma: torch.Tensor, 
                    tag: str = "out_mma", check_all: bool = False):
    out_flash = out_flash.transpose(1, 2)
    if check_all:
        for i in range(int(N/8)):
            if i < 4:
                pretty_print_line()
                print(f"out_flash[:, :,  {(i*8)}:{(i+1)*8}, :]:\n")
                print(out_flash[:, :,  (i*8):(i+1)*8, :].float())
                print(f"{tag}[:, :, {(i*8)}:{(i+1)*8}, :]:\n")
                print(out_mma[:, :, (i*8):(i+1)*8, :].float())
        pretty_print_line()
    diff = torch.abs(out_flash.float() - out_mma.float())
    all_close = str(torch.allclose(out_flash.float(), out_mma.float(), atol=1e-2))
    pretty_print_line(
        f"out_flash vs {tag:<18}, all close: {all_close:<6}, "
        f"max diff: {diff.max().item():.6f}, min diff: {diff.min().item():.6f}, "
        f"mean diff: {diff.mean().item():.6f}"
    )


Bs = [1, 2, 4] if not args.B else [args.B]
Hs = [1, 4, 8] if not args.H else [args.H]
Ns = [1024, 2048, 4096] if not args.N else [args.N]
Ds = [64, 128] if not args.D else [args.D] 
# batch_size, n_head, seq_len, head_dim (B,H,N,D)
BHNDs = [(B, H, N, D) for B in Bs for H in Hs for N in Ns for D in Ds]

seed = args.seed if args.seed else random.choice(range(10000))
set_rand_seed(seed)
pretty_print_line()
pretty_print_line(f"B: batch_size, H: n_head, N: seq_len, D: head_dim, "
                  f"seed: {seed}, Warmup: {args.warmup}, Iters: {args.iters}")

for (B, H, N, D) in BHNDs:
    MAX_TFLOPS = -1
    pretty_print_line()
    pretty_print_line(f"B={B}, H={H}, N={N}, D={D}, Warmup: {args.warmup}, Iters: {args.iters}")
    q, k, v, o, fq, fk, fv = get_qkvo(B, H, N, D)
    torch.cuda.synchronize()
    
    if args.run_torch_unfused:
        out_unfused,    _ = run_benchmark(unfused_standard_attn, q, k, v, "torch(unfused)")
    out_mma_split_kv1,  _ = run_benchmark(lib.flash_attn_mma_stages_split_kv, q, k, v, "mma(split-kv+stage1)", o, stages=1)
    out_mma_split_kv2,  _ = run_benchmark(lib.flash_attn_mma_stages_split_kv, q, k, v, "mma(split-kv+stage2)", o, stages=2)
    out_mma_split_q1,   _ = run_benchmark(lib.flash_attn_mma_stages_split_q,  q, k, v, "mma(split-q+stage1)",  o, stages=1)
    out_mma_split_q2,   _ = run_benchmark(lib.flash_attn_mma_stages_split_q,  q, k, v, "mma(split-q+stage2)",  o, stages=2)
    out_mma_share_qkv1, _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_qkv, q, k, v, "mma(split-q+share-qkv+stage1)", o, stages=1)
    out_mma_share_qkv2, _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_qkv, q, k, v, "mma(split-q+share-qkv+stage2)", o, stages=2)
    out_mma_share_kv1,  _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_kv,  q, k, v, "mma(split-q+share-kv+stage1)",  o, stages=1)
    out_mma_share_kv2,  _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_kv,  q, k, v, "mma(split-q+share-kv+stage2)",  o, stages=2)
    out_flash,          _ = run_benchmark(flash_attn_func, fq, fk, fv, "(flash)")
    if args.run_torch_sdpa:
        out_sdpa,       _ = run_benchmark(F.scaled_dot_product_attention, q, k, v, "(sdpa)")
    pretty_print_line()
    
    torch.cuda.synchronize()
    if args.check:
        pretty_print_line()
        check_all_close(out_flash, out_mma_split_kv1,  "out_mma_split_kv1",  args.check_all)
        check_all_close(out_flash, out_mma_split_q1,   "out_mma_split_q1",   args.check_all)
        check_all_close(out_flash, out_mma_share_kv1,  "out_mma_share_kv1",  args.check_all)
        check_all_close(out_flash, out_mma_share_qkv1, "out_mma_share_qkv1", args.check_all)
        check_all_close(out_flash, out_mma_split_kv2,  "out_mma_split_kv2",  args.check_all)
        check_all_close(out_flash, out_mma_split_q2,   "out_mma_split_q2",   args.check_all)
        check_all_close(out_flash, out_mma_share_kv2,  "out_mma_share_kv2",  args.check_all)
        check_all_close(out_flash, out_mma_share_qkv2, "out_mma_share_qkv2", args.check_all)
        pretty_print_line()
