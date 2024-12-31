import os
import math
import time
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from typing import Optional
from torch.nn.attention import sdpa_kernel, SDPBackend
from flash_attn import flash_attn_func
from functools import partial
import argparse
import random
import numpy as np

torch.set_grad_enabled(False)
torch.set_printoptions(precision=6, threshold=8, edgeitems=3, 
                       linewidth=120, sci_mode=False)


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
    parser.add_argument("--run-acc-f32", "--acc-f32", "--f32", action="store_true")
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
    parser.add_argument("--build-others", '--others', action="store_true")
    parser.add_argument("--tag-hints", '--tags', '--hints', type=str, default=None)
    return parser.parse_args()


args = get_args()


def set_rand_seed(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device_name():
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    # since we will run GPU on WSL2, so add WSL2 tag.
    if "Laptop" in device_name:
        device_name += " WSL2"
    return device_name


def get_device_capability():
    return torch.cuda.get_device_capability(torch.cuda.current_device())


def get_build_sources():
    build_sources = []
    # Basic
    build_sources.append('./mma/basic/flash_attn_mma_split_kv.cu')
    build_sources.append('./mma/basic/flash_attn_mma_split_q.cu')
    build_sources.append('./mma/basic/flash_attn_mma_share_kv.cu')
    build_sources.append('./mma/basic/flash_attn_mma_share_qkv.cu')
    build_sources.append('./mma/basic/flash_attn_mma_tiling_qk.cu')
    build_sources.append('./mma/basic/flash_attn_mma_share_qkv_F32F16F16F32.cu')
    # Swizzle
    build_sources.append('./mma/swizzle/flash_attn_mma_share_kv_swizzle_q.cu')
    build_sources.append('./mma/swizzle/flash_attn_mma_share_kv_swizzle_qk.cu')
    build_sources.append('./mma/swizzle/flash_attn_mma_share_kv_swizzle_qkv.cu')
    build_sources.append('./mma/swizzle/flash_attn_mma_share_qkv_swizzle_q.cu')
    build_sources.append('./mma/swizzle/flash_attn_mma_share_qkv_swizzle_qk.cu')
    build_sources.append('./mma/swizzle/flash_attn_mma_share_qkv_swizzle_qkv.cu')
    build_sources.append('./mma/swizzle/flash_attn_mma_tiling_qk_swizzle_q.cu')
    build_sources.append('./mma/swizzle/flash_attn_mma_tiling_qk_swizzle_qk.cu')
    build_sources.append('./mma/swizzle/flash_attn_mma_tiling_qk_swizzle_qkv.cu')
    # Others
    if args.build_others:
        build_sources.append('./mma/others/flash_attn_mma_share_qkv_s2g_o.cu')
        build_sources.append('./mma/others/flash_attn_mma_share_qkv_F32F16F16F32_rr.cu')
    # Pybind
    build_sources.append('./pybind/flash_attn.cc')
    return build_sources


def get_project_dir():
    return os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))


project_dir = get_project_dir()


def get_build_cuda_cflags(build_pkg: bool = False):
    device_name = get_device_name()
    project_dir = get_project_dir()
    extra_cuda_cflags = []
    extra_cuda_cflags.append("-O3")
    extra_cuda_cflags.append("-std=c++17")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF_OPERATORS__")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF_CONVERSIONS__")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF2_OPERATORS__")
    extra_cuda_cflags.append("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
    extra_cuda_cflags.append("--expt-relaxed-constexpr")
    extra_cuda_cflags.append("--expt-extended-lambda")
    extra_cuda_cflags.append("--use_fast_math")
    extra_cuda_cflags.append("-DFLASH_ATTN_MMA_DEBUG" if args.debug else "")
    extra_cuda_cflags.append("-DBUILD_FLASH_ATTN_MMA_OTHERS" if args.build_others else "")
    extra_cuda_cflags.append("-DBUILD_FLASH_ATTN_MMA_L20"  if "L20"  in device_name else "")
    extra_cuda_cflags.append("-DBUILD_FLASH_ATTN_MMA_4090" if "4090" in device_name else "")
    extra_cuda_cflags.append("-DBUILD_FLASH_ATTN_MMA_3080" if "3080" in device_name else "")
    extra_cuda_cflags.append("-diag-suppress 177" if not build_pkg else "--ptxas-options=-v")
    extra_cuda_cflags.append("-Xptxas -v" if not build_pkg else "--ptxas-options=-O3")
    extra_cuda_cflags.append(f'-I {project_dir}/kernels/flash-attn')
    extra_cuda_cflags.append(f'-I {project_dir}/kernels/flash-attn/utils')
    extra_cuda_cflags.append(f'-I {project_dir}/kernels/flash-attn/mma')
    extra_cuda_cflags.append(f'-I {project_dir}/kernels/flash-attn/mma/basic')
    extra_cuda_cflags.append(f'-I {project_dir}/kernels/flash-attn/mma/swizzle')
    extra_cuda_cflags.append(f'-I {project_dir}/kernels/flash-attn/mma/others')
    extra_cuda_cflags.append(f'-I {project_dir}/kernels/flash-attn/cutlass')
    extra_cuda_cflags.append(f'-I {project_dir}/kernels/flash-attn/pybind')
    extra_cuda_cflags.append(f'-I {project_dir}/third-party/cutlass/include')
    extra_cuda_cflags.append(f'-I {project_dir}/third-party/cutlass/tools/util/include')
    return extra_cuda_cflags


def get_build_cflags():
    extra_cflags = []
    extra_cflags.append("-std=c++17")
    extra_cflags.append("-DBUILD_FLASH_ATTN_MMA_OTHERS" if args.build_others else "")
    return extra_cflags


def pretty_print_line(m: str = "", sep: str = "-", width: int = 150):
    res_len = width - len(m)
    left_len = int(res_len / 2)
    right_len = res_len - left_len
    pretty_line = sep * left_len + m + sep * right_len
    print(pretty_line)



if args.D and args.D > 256:
    args.run_torch_sdpa = True
pretty_print_line()
print(args)
pretty_print_line()


# Load the CUDA kernel as a python module
lib = load(name='flash_attn_lib', 
           sources=get_build_sources(), 
           extra_cuda_cflags=get_build_cuda_cflags(), 
           extra_cflags=get_build_cflags(),
           verbose=args.verbose)


if not args.build_others:
    fake_fa_func = lambda q, k, v, o, s: o # fake FA func
    setattr(lib, "flash_attn_mma_stages_split_q_shared_qkv_s2g_o", fake_fa_func)
    setattr(lib, "flash_attn_mma_stages_split_q_shared_qkv_acc_f32_rr", fake_fa_func)


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
    global MAX_HEADDIM_CFG

    tag_hints: str = args.tag_hints # e.g "share-qkv,tiling-kv,swizzle"
    if tag_hints:
        tag_hints: list = tag_hints.strip().split(",")
        tag_hints.append("flash")
        tag_hints.append("sdpa")
        tag_hints.append("unfused")
        hit_hints = False
        for hint in tag_hints:
            if hint in tag:
                hit_hints = True
        if not hit_hints:
            return None, None
    
    if not args.build_others:
        others_tags = ["s2g-o", "rr"]
        for o_tag in others_tags:
            if o_tag in tag:
                return None, None

    if "sdpa" in tag and (not args.run_torch_sdpa):
        return None, None
    if "unfused" in tag and (not args.run_torch_unfused):
        return None, None
    if "acc-f32" in tag and (not args.run_acc_f32):
        return None, None
    
    B, H, N, D = q.size()
    if "flash" in tag:
        B, N, H, D = q.size()

    max_supported_D = MAX_HEADDIM_CFG.get(tag, None)
    # skip if headdim not supported.
    if max_supported_D is not None:
        if D > max_supported_D:
            return None, None

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
        print(f"{out_info:>50}: {out_val}, time:{str(mean_time)[:8]}ms, "
              f"TFLOPS:{TFLOPS:<6.2f}(+{improve:.2f}%)")
    else:
        if (not only_show_improved) or (("flash" in tag) or ("sdpa" in tag)):
            print(f"{out_info:>50}: {out_val}, time:{str(mean_time)[:8]}ms, "
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
    # transpose (N,D) -> (D,N) for V smem swizzle.
    tk = k.transpose(-2, -1).contiguous() # [B,H,N,D] -> [B,H,D,N]
    tv = v.transpose(-2, -1).contiguous() # [B,H,N,D] -> [B,H,D,N]

    return q, k, v, o, fq, fk, fv, tk, tv


# un-fused naive attn
def unfused_standard_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y


def sdpa(q: Tensor, k: Tensor, v: Tensor, use_flash: bool = False):
    if not use_flash:
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            out: Tensor = F.scaled_dot_product_attention(q, k, v)
    else:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out: Tensor = F.scaled_dot_product_attention(q, k, v)
    return out


def check_all_close(out_flash_or_sdpa: torch.Tensor, out_mma: torch.Tensor, 
                    tag: str = "out_mma", check_all: bool = False, 
                    is_flash: bool = True):
    if any((out_flash_or_sdpa is None, out_mma is None)):
        return
    if is_flash:
        true_tag = "out_flash"
        out_flash_or_sdpa = out_flash_or_sdpa.transpose(1, 2)
    else:
        true_tag = "out_sdpa"
    if check_all:
        for i in range(int(N/8)):
            if i < 4:
                pretty_print_line()
                print(f"{true_tag}[:, :,  {(i*8)}:{(i+1)*8}, :]:\n")
                print(out_flash_or_sdpa[:, :,  (i*8):(i+1)*8, :].float())
                print(f"{tag}[:, :, {(i*8)}:{(i+1)*8}, :]:\n")
                print(out_mma[:, :, (i*8):(i+1)*8, :].float())
        pretty_print_line()
    diff = torch.abs(out_flash_or_sdpa.float() - out_mma.float())
    all_close = str(torch.allclose(out_flash_or_sdpa.float(), out_mma.float(), atol=1e-2))
    pretty_print_line(
        f"{true_tag} vs {tag:<25}, all close: {all_close:<6}, "
        f"max diff: {diff.max().item():.6f}, min diff: {diff.min().item():.6f}, "
        f"mean diff: {diff.mean().item():.6f}"
    )


Bs = [1, 4, 8] if not args.B else [args.B]
Hs = [1, 4, 8] if not args.H else [args.H]
Ns = [1024, 2048, 4096, 8192] if not args.N else [args.N]
Ds = [64, 128, 256, 512] if not args.D else [args.D] 
# batch_size, n_head, seq_len, head_dim (B,H,N,D)
BHNDs = [(B, H, N, D) for B in Bs for H in Hs for N in Ns for D in Ds]
# max headdim supported for different methods. skip if D > max_D.
MAX_HEADDIM_CFG: dict[str, int] = {
    # FA2, SDPA, Naive MHA.
    "(flash)":                                      256, 
    "(sdpa)":                                       4096, # may no limit
    "(unfused)":                                    4096, # may no limit
    # Split-KV
    "mma(split-kv+stage1)":                         128,
    "mma(split-kv+stage2)":                         128,
    # Split-Q
    "mma(split-q+stage1)":                          128,
    "mma(split-q+stage2)":                          128,
    # Split-Q + Shared KV SMEM
    "mma(split-q+share-kv+stage1)":                 256,
    "mma(split-q+share-kv+stage2)":                 128,
    "mma(split-q+share-kv+swizzle-q+stage1)":       256,
    "mma(split-q+share-kv+swizzle-q+stage2)":       128,
    "mma(split-q+share-kv+swizzle-qk+stage1)":      256,
    "mma(split-q+share-kv+swizzle-qk+stage2)":      128,
    "mma(split-q+share-kv+swizzle-qkv+stage1)":     256,
    "mma(split-q+share-kv+swizzle-qkv+stage2)":     128,
    "mma(split-q+share-qkv+acc-f32+stage1)":        256,
    "mma(split-q+share-qkv+acc-f32+stage2)":        128,
    # Split-Q + Fully Shared QKV SMEM
    "mma(split-q+share-qkv+stage1)":                256, 
    "mma(split-q+share-qkv+stage2)":                128, 
    "mma(split-q+share-qkv+swizzle-q+stage1)":      256,
    "mma(split-q+share-qkv+swizzle-q+stage2)":      128,
    "mma(split-q+share-qkv+swizzle-qk+stage1)":     256,
    "mma(split-q+share-qkv+swizzle-qk+stage2)":     128,
    "mma(split-q+share-qkv+swizzle-qkv+stage1)":    256,
    "mma(split-q+share-qkv+swizzle-qkv+stage2)":    128,
    # Split-Q + QK Fine-grained Tiling
    "mma(split-q+tiling-qk+stage1)":                1024,
    "mma(split-q+tiling-qk+stage2)":                1024,
    "mma(split-q+tiling-qk+swizzle-q+stage1)":      1024,
    "mma(split-q+tiling-qk+swizzle-q+stage2)":      1024,
    "mma(split-q+tiling-qk+swizzle-qk+stage1)":     1024,
    "mma(split-q+tiling-qk+swizzle-qk+stage2)":     1024,
    "mma(split-q+tiling-qk+swizzle-qkv+stage1)":    256,
    "mma(split-q+tiling-qk+swizzle-qkv+stage2)":    256,
    # Others, O s2g, etc.
    "mma(split-q+share-qkv+s2g-o+stage1)":          256,
    "mma(split-q+share-qkv+s2g-o+stage2)":          128,
    "mma(split-q+share-qkv+acc-f32+rr+stage1)":     256,
    "mma(split-q+share-qkv+acc-f32+rr+stage2)":     256,
}

seed = args.seed if args.seed else random.choice(range(10000))
set_rand_seed(seed)
pretty_print_line()
pretty_print_line(f"B: batch_size, H: n_head, N: seq_len, D: head_dim, "
                  f"seed: {seed}, Warmup: {args.warmup}, Iters: {args.iters}")

run_torch_sdpa = args.run_torch_sdpa
for (B, H, N, D) in BHNDs:
    MAX_TFLOPS = -1
    q, k, v, o, fq, fk, fv, tk, tv = get_qkvo(B, H, N, D)
    if D > 256:
        args.run_torch_sdpa = True
    else:
        args.run_torch_sdpa = run_torch_sdpa
    torch.cuda.synchronize()
    pretty_print_line()
    pretty_print_line(f"B={B}, H={H}, N={N}, D={D}, Warmup: {args.warmup}, Iters: {args.iters}")
    # Naive MHA.
    out_unfused,               _ = run_benchmark(unfused_standard_attn, q, k, v, "(unfused)")
    # Split-KV
    out_mma_split_kv1,         _ = run_benchmark(lib.flash_attn_mma_stages_split_kv, q, k, v, "mma(split-kv+stage1)", o, stages=1)
    out_mma_split_kv2,         _ = run_benchmark(lib.flash_attn_mma_stages_split_kv, q, k, v, "mma(split-kv+stage2)", o, stages=2)
    # Split-Q
    out_mma_split_q1,          _ = run_benchmark(lib.flash_attn_mma_stages_split_q, q, k, v, "mma(split-q+stage1)",  o, stages=1)
    out_mma_split_q2,          _ = run_benchmark(lib.flash_attn_mma_stages_split_q, q, k, v, "mma(split-q+stage2)",  o, stages=2)
    # Split-Q + Shared KV SMEM + Swizzle
    out_mma_share_kv1,         _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_kv, q, k, v, "mma(split-q+share-kv+stage1)",  o, stages=1)
    out_mma_share_kv2,         _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_kv, q, k, v, "mma(split-q+share-kv+stage2)",  o, stages=2)
    out_mma_share_kv_sq1,      _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_kv_swizzle_q, q, k, v, "mma(split-q+share-kv+swizzle-q+stage1)",  o, stages=1)
    out_mma_share_kv_sq2,      _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_kv_swizzle_q, q, k, v, "mma(split-q+share-kv+swizzle-q+stage2)",  o, stages=2)
    out_mma_share_kv_sqk1,     _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_kv_swizzle_qk, q, k, v, "mma(split-q+share-kv+swizzle-qk+stage1)",  o, stages=1)
    out_mma_share_kv_sqk2,     _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_kv_swizzle_qk, q, k, v, "mma(split-q+share-kv+swizzle-qk+stage2)",  o, stages=2)
    out_mma_share_kv_sqkv1,    _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_kv_swizzle_qkv, q, k, tv, "mma(split-q+share-kv+swizzle-qkv+stage1)",  o, stages=1)
    out_mma_share_kv_sqkv2,    _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_kv_swizzle_qkv, q, k, tv, "mma(split-q+share-kv+swizzle-qkv+stage2)",  o, stages=2)
    # Split-Q + Fully Shared QKV SMEM + Swizzle
    out_mma_share_qkv1,        _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_qkv, q, k, v, "mma(split-q+share-qkv+stage1)", o, stages=1)
    out_mma_share_qkv2,        _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_qkv, q, k, v, "mma(split-q+share-qkv+stage2)", o, stages=2)
    out_mma_share_qkv_f321,    _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_qkv_acc_f32, q, k, v, "mma(split-q+share-qkv+acc-f32+stage1)", o, stages=1)
    out_mma_share_qkv_f322,    _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_qkv_acc_f32, q, k, v, "mma(split-q+share-qkv+acc-f32+stage2)", o, stages=2)
    out_mma_share_qkv_sq1,     _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_qkv_swizzle_q, q, k, v, "mma(split-q+share-qkv+swizzle-q+stage1)", o, stages=1)
    out_mma_share_qkv_sq2,     _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_qkv_swizzle_q, q, k, v, "mma(split-q+share-qkv+swizzle-q+stage2)", o, stages=2)
    out_mma_share_qkv_sqk1,    _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_qkv_swizzle_qk, q, k, v, "mma(split-q+share-qkv+swizzle-qk+stage1)", o, stages=1)
    out_mma_share_qkv_sqk2,    _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_qkv_swizzle_qk, q, k, v, "mma(split-q+share-qkv+swizzle-qk+stage2)", o, stages=2)
    out_mma_share_qkv_sqkv1,   _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_qkv_swizzle_qkv, q, k, tv, "mma(split-q+share-qkv+swizzle-qkv+stage1)", o, stages=1)
    out_mma_share_qkv_sqkv2,   _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_qkv_swizzle_qkv, q, k, tv, "mma(split-q+share-qkv+swizzle-qkv+stage2)", o, stages=2)
    # Split-Q + QK Fine-grained Tiling + Swizzle
    out_mma_tiling_qk1,        _ = run_benchmark(lib.flash_attn_mma_stages_split_q_tiling_qk, q, k, v, "mma(split-q+tiling-qk+stage1)",  o, stages=1)
    out_mma_tiling_qk2,        _ = run_benchmark(lib.flash_attn_mma_stages_split_q_tiling_qk, q, k, v, "mma(split-q+tiling-qk+stage2)",  o, stages=2)
    out_mma_tiling_qk_sq1,     _ = run_benchmark(lib.flash_attn_mma_stages_split_q_tiling_qk_swizzle_q, q, k, v, "mma(split-q+tiling-qk+swizzle-q+stage1)",  o, stages=1)
    out_mma_tiling_qk_sq2,     _ = run_benchmark(lib.flash_attn_mma_stages_split_q_tiling_qk_swizzle_q, q, k, v, "mma(split-q+tiling-qk+swizzle-q+stage2)",  o, stages=2)
    out_mma_tiling_qk_sqk1,    _ = run_benchmark(lib.flash_attn_mma_stages_split_q_tiling_qk_swizzle_qk, q, k, v, "mma(split-q+tiling-qk+swizzle-qk+stage1)",  o, stages=1)
    out_mma_tiling_qk_sqk2,    _ = run_benchmark(lib.flash_attn_mma_stages_split_q_tiling_qk_swizzle_qk, q, k, v, "mma(split-q+tiling-qk+swizzle-qk+stage2)",  o, stages=2)
    out_mma_tiling_qk_sqkv1,   _ = run_benchmark(lib.flash_attn_mma_stages_split_q_tiling_qk_swizzle_qkv, q, k, tv, "mma(split-q+tiling-qk+swizzle-qkv+stage1)",  o, stages=1)
    out_mma_tiling_qk_sqkv2,   _ = run_benchmark(lib.flash_attn_mma_stages_split_q_tiling_qk_swizzle_qkv, q, k, tv, "mma(split-q+tiling-qk+swizzle-qkv+stage2)",  o, stages=2)
    # Others, O s2g, etc.
    out_mma_share_qkv_s2g1,    _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_qkv_s2g_o, q, k, v, "mma(split-q+share-qkv+s2g-o+stage1)", o, stages=1)
    out_mma_share_qkv_s2g2,    _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_qkv_s2g_o, q, k, v, "mma(split-q+share-qkv+s2g-o+stage2)", o, stages=2)
    out_mma_share_qkv_rr1,     _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_qkv_acc_f32_rr, q, k, v, "mma(split-q+share-qkv+acc-f32+rr+stage1)", o, stages=1)
    out_mma_share_qkv_rr2,     _ = run_benchmark(lib.flash_attn_mma_stages_split_q_shared_qkv_acc_f32_rr, q, k, v, "mma(split-q+share-qkv+acc-f32+rr+stage2)", o, stages=2)
    # FA2, SDPA official
    out_flash,                 _ = run_benchmark(flash_attn_func, fq, fk, fv, "(flash)")
    out_sdpa,                  _ = run_benchmark(partial(sdpa, use_flash=(D<=256)), q, k, v, "(sdpa)")
    pretty_print_line()
    
    torch.cuda.synchronize()
    if args.check:
        if D <= 256:
            pretty_print_line()
            # Split-KV
            check_all_close(out_flash, out_mma_split_kv1,         "out_mma_split_kv1",        args.check_all)
            check_all_close(out_flash, out_mma_split_kv2,         "out_mma_split_kv2",        args.check_all)
            # Split-Q
            check_all_close(out_flash, out_mma_split_q1,          "out_mma_split_q1",         args.check_all)
            check_all_close(out_flash, out_mma_split_q2,          "out_mma_split_q2",         args.check_all)
            # Split-Q + Shared KV SMEM
            check_all_close(out_flash, out_mma_share_kv1,         "out_mma_share_kv1",        args.check_all)
            check_all_close(out_flash, out_mma_share_kv2,         "out_mma_share_kv2",        args.check_all)
            check_all_close(out_flash, out_mma_share_kv_sq1,      "out_mma_share_kv_sq1",     args.check_all)
            check_all_close(out_flash, out_mma_share_kv_sq2,      "out_mma_share_kv_sq2",     args.check_all)
            check_all_close(out_flash, out_mma_share_kv_sqk1,     "out_mma_share_kv_sqk1",    args.check_all)
            check_all_close(out_flash, out_mma_share_kv_sqk2,     "out_mma_share_kv_sqk2",    args.check_all)
            check_all_close(out_flash, out_mma_share_kv_sqkv1,    "out_mma_share_kv_sqkv1",   args.check_all)
            check_all_close(out_flash, out_mma_share_kv_sqkv2,    "out_mma_share_kv_sqkv2",   args.check_all)
            # Split-Q + Fully Shared QKV SMEM
            check_all_close(out_flash, out_mma_share_qkv1,        "out_mma_share_qkv1",       args.check_all)
            check_all_close(out_flash, out_mma_share_qkv2,        "out_mma_share_qkv2",       args.check_all)
            check_all_close(out_flash, out_mma_share_qkv_f321,    "out_mma_share_qkv_f321",   args.check_all)
            check_all_close(out_flash, out_mma_share_qkv_f322,    "out_mma_share_qkv_f322",   args.check_all)
            check_all_close(out_flash, out_mma_share_qkv_sq1,     "out_mma_share_qkv_sq1",    args.check_all)
            check_all_close(out_flash, out_mma_share_qkv_sq2,     "out_mma_share_qkv_sq2",    args.check_all)
            check_all_close(out_flash, out_mma_share_qkv_sqk1,    "out_mma_share_qkv_sqk1",   args.check_all)
            check_all_close(out_flash, out_mma_share_qkv_sqk2,    "out_mma_share_qkv_sqk2",   args.check_all)
            check_all_close(out_flash, out_mma_share_qkv_sqkv1,   "out_mma_share_qkv_sqkv1",  args.check_all)
            check_all_close(out_flash, out_mma_share_qkv_sqkv2,   "out_mma_share_qkv_sqkv2",  args.check_all)
            # Split-Q + QK Fine-grained Tiling
            check_all_close(out_flash, out_mma_tiling_qk1,        "out_mma_tiling_qk1",       args.check_all)
            check_all_close(out_flash, out_mma_tiling_qk2,        "out_mma_tiling_qk2",       args.check_all)
            check_all_close(out_flash, out_mma_tiling_qk_sq1,     "out_mma_tiling_qk_sq1",    args.check_all)
            check_all_close(out_flash, out_mma_tiling_qk_sq2,     "out_mma_tiling_qk_sq2",    args.check_all)
            check_all_close(out_flash, out_mma_tiling_qk_sqk1,    "out_mma_tiling_qk_sqk1",   args.check_all)
            check_all_close(out_flash, out_mma_tiling_qk_sqk2,    "out_mma_tiling_qk_sqk2",   args.check_all)
            check_all_close(out_flash, out_mma_tiling_qk_sqkv1,   "out_mma_tiling_qk_sqkv1",  args.check_all)
            check_all_close(out_flash, out_mma_tiling_qk_sqkv2,   "out_mma_tiling_qk_sqkv2",  args.check_all)
            # Others, O s2g, etc.
            check_all_close(out_flash, out_mma_share_qkv_s2g1,    "out_mma_share_qkv_s2g1",   args.check_all)
            check_all_close(out_flash, out_mma_share_qkv_s2g2,    "out_mma_share_qkv_s2g2",   args.check_all)
            check_all_close(out_flash, out_mma_share_qkv_rr1,     "out_mma_share_qkv_rr1",    args.check_all)
            check_all_close(out_flash, out_mma_share_qkv_rr2,     "out_mma_share_qkv_rr2",    args.check_all)
            pretty_print_line()
        elif args.run_torch_sdpa:
            pretty_print_line()
            # Split-Q + Fully Shared QKV SMEM
            check_all_close(out_sdpa, out_mma_share_qkv1,         "out_mma_share_qkv1",       args.check_all, False)
            check_all_close(out_sdpa, out_mma_share_qkv2,         "out_mma_share_qkv2",       args.check_all, False)
            check_all_close(out_sdpa, out_mma_share_qkv_f321,     "out_mma_share_qkv_f321",   args.check_all, False)
            check_all_close(out_sdpa, out_mma_share_qkv_f322,     "out_mma_share_qkv_f322",   args.check_all, False)
            # Split-Q + QK Fine-grained Tiling
            check_all_close(out_sdpa, out_mma_tiling_qk1,         "out_mma_tiling_qk1",       args.check_all, False)
            check_all_close(out_sdpa, out_mma_tiling_qk2,         "out_mma_tiling_qk2",       args.check_all, False)
            check_all_close(out_sdpa, out_mma_tiling_qk_sq1,      "out_mma_tiling_qk_sq1",    args.check_all, False)
            check_all_close(out_sdpa, out_mma_tiling_qk_sq2,      "out_mma_tiling_qk_sq2",    args.check_all, False)
            check_all_close(out_sdpa, out_mma_tiling_qk_sqk1,     "out_mma_tiling_qk_sqk1",   args.check_all, False)
            check_all_close(out_sdpa, out_mma_tiling_qk_sqk2,     "out_mma_tiling_qk_sqk2",   args.check_all, False)
            check_all_close(out_sdpa, out_mma_tiling_qk_sqkv1,    "out_mma_tiling_qk_sqkv1",  args.check_all, False)
            check_all_close(out_sdpa, out_mma_tiling_qk_sqkv2,    "out_mma_tiling_qk_sqkv2",  args.check_all, False)
            # Others, O s2g, etc.
            check_all_close(out_sdpa, out_mma_share_qkv_rr1,      "out_mma_share_qkv_rr1",    args.check_all, False)
            check_all_close(out_sdpa, out_mma_share_qkv_rr2,      "out_mma_share_qkv_rr2",    args.check_all, False)
            pretty_print_line()
