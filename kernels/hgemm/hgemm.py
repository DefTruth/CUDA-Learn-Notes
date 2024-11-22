import os
import gc
import torch
import time 
from functools import partial
from typing import Optional
import argparse
from tools.utils import (get_device_name, 
                         pretty_print_line,
                         try_load_hgemm_library,
                         as_col_major)

torch.set_grad_enabled(False)


def get_args():
    parser = argparse.ArgumentParser(description="hgemm benchmark")
    parser.add_argument("--M", type=int, default=None, help="Matrix M size")
    parser.add_argument("--N", type=int, default=None, help="Matrix N size")
    parser.add_argument("--K", type=int, default=None, help="Matrix K size")
    parser.add_argument("--MNK", type=int, default=None, help="Matrix M=N=K size")
    parser.add_argument("--MMNK", type=int, default=12800, help="Matrix MAX M=M=N=K size")
    parser.add_argument("--SEP", '--sep', type=int, default=256, help="Matrix SEP M=M=N=K size")
    parser.add_argument("--warmup", "--w", type=int, default=2, help="Warmup iters")
    parser.add_argument("--iters", "--i", type=int, default=10, help="Benchmark iters")
    parser.add_argument("--verbose", "--v", action="store_true", help="Verbose")
    parser.add_argument("--show-matrix", "--show-m", action="store_true", help="Show output matrix values")
    parser.add_argument("--show-all-info", "--show-a", action="store_true", help="Show all the profile info")
    parser.add_argument("--show-memory", "--show-mm", action="store_true", help="Show gpu memory info")
    parser.add_argument("--enable-mma", "--mma", action="store_true", help="Enable MMA kernel tests")
    parser.add_argument("--enable-mma-tn", "--mma-tn", action="store_true", help="Enable TN MMA kernel tests")
    parser.add_argument("--enable-wmma", "--wmma", action="store_true", help="Enable WMMA kernel tests")
    parser.add_argument("--enable-cuda", "--cuda", action="store_true", help="Enable CUDA kernel tests")
    parser.add_argument("--enable-mma-all", "--mma-all", action="store_true", help="Enable all MMA kernel tests")
    parser.add_argument("--enable-wmma-all", "--wmma-all", action="store_true", help="Enable all WMMA kernel tests")
    parser.add_argument("--enable-cuda-all", "--cuda-all", action="store_true", help="Enable all CUDA kernel tests")
    parser.add_argument("--enable-torch", "--torch", action="store_true", help="Enable torch matmul")
    parser.add_argument("--enable-cute-tn", "--cute-tn", action="store_true", help="Enable cute hgemm matmul")
    parser.add_argument("--enable-cute", "--cute", action="store_true", help="Enable cute hgemm matmul")
    parser.add_argument("--disable-cublas", "--no-cublas", action="store_true", help="Disable cublas hgemm")
    parser.add_argument("--disable-cublas-tn", "--no-cublas-tn", action="store_true", help="Disable cublas TN hgemm")
    parser.add_argument("--sleep-duration", "--sleep", type=float, default=0.1, help="Sleep duration")
    parser.add_argument("--swizzle-factor", "--swizzle", type=float, default=None, help="Swizzle factor")
    parser.add_argument("--no-default", action="store_true", help="Disable default tests")
    parser.add_argument("--plot-flops", "--plot", action="store_true", help="Plot TFLOPS")
    parser.add_argument("--plot-topk", "--topk", type=int, default=8, help="Plot top k TFLOPS")
    parser.add_argument("--no-plot-best", "--no-best", action="store_true", help="Not Plot best TFLOPS")
    parser.add_argument("--exclude-tags", "--exclude", type=str, default=None, help="Exclude tag for plot, sperated by comma")
    parser.add_argument("--save-dir", "--dir", type=str, default="./", help="Save dir for plot")
    parser.add_argument("--save-tag", "--tag", type=str, default=None, help="Save name for plot")
    parser.add_argument("--force-build", "--build", action="store_true", help="Force build from sources")
    return parser.parse_args()


args = get_args()
pretty_print_line()
print(args)
pretty_print_line()


hgemm = try_load_hgemm_library(force_build=args.force_build, 
                               verbose=args.verbose)

MAX_TFLOPS = -1
STATIS_INFO: dict[str, list[float]] = {}
STATIS_INFO["MNK"] = []
TOATL_TFLOPS: dict[str, float] = {}
CUBLAS_TOTAL_TFLOPS = 0
CUBLAS_TN_TOTAL_TFLOPS = 0


def make_block_swizzle_stride(N: int, K: int):
    # make swizzle stride as N/8,N/4,N/2 and multiples of 256
    if args.swizzle_factor is None:
        swizzle_factor = 0.5 if N <= 4096 else 0.25
        if all((N >= 14848, K > 8192, N % 8 == 0)):
            swizzle_factor = 0.125
    else:
        swizzle_factor = args.swizzle_factor

    swizzle_stride = int(N * swizzle_factor)
    swizzle_stride = swizzle_stride if swizzle_stride >= 256 else 1

    return swizzle_stride


@torch.no_grad
def run_benchmark(perf_func: callable, 
                  a: torch.Tensor, b: torch.Tensor,
                  tag: str, out: Optional[torch.Tensor] = None, 
                  stages: int = -1, swizzle: bool = False,
                  swizzle_stride: int = 1,
                  warmup: int = args.warmup, 
                  iters: int = args.iters,
                  show_matrix: bool = args.show_matrix,
                  only_show_improved: bool = not args.show_all_info):
    global MAX_TFLOPS

    M = a.size(0)
    K = a.size(1)
    N = b.size(1) # TN still has shape [K,N]
    if swizzle:
        swizzle_stride = make_block_swizzle_stride(N, K)
        swizzle = swizzle if swizzle_stride >= 256 else False
    else:
        swizzle_stride = 1 # means no thread block swizzle
    
    if stages:
        assert swizzle_stride is not None

    if out is not None: 
        out.fill_(0)      
    if out is not None:
        for i in range(warmup):
            if stages > 1:
                perf_func(a, b, out, stages, swizzle, swizzle_stride)
            else:
                perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b) 
    
    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            if stages > 1:
                perf_func(a, b, out, stages, swizzle, swizzle_stride)
            else:
                perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b) 
    torch.cuda.synchronize()

    end = time.time()
    total_time = (end - start) * 1000 # ms
    mean_time = total_time / iters
    out_info = f"{tag}"
    out_flat = out.flatten()
    out_val_first = out_flat[:2].detach().cpu().numpy().tolist()
    out_val_last = out_flat[-2:].detach().cpu().numpy().tolist()
    out_val = [out_val_first[0], out_val_last[-1]]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}"[:10] for v in out_val]
    TFLOPS = (2 * M * N * K) * 1e-9 / (mean_time)
    mean_time = str(f"{mean_time:<12}")[:8]
    swizzle_stride = 'NOOP' if swizzle_stride == 1 else swizzle_stride

    # caculate TFLOPS improved.
    if TFLOPS > MAX_TFLOPS:
        if MAX_TFLOPS > 0:
            improve = ((TFLOPS - MAX_TFLOPS) / MAX_TFLOPS) * 100
            improve = round(improve, 2)
        else:
            improve = 0
        MAX_TFLOPS = TFLOPS
        print(f"{out_info:>50}: {out_val}, time:{mean_time}ms, "
              f"swizzle<block>: {swizzle_stride:<4}, TFLOPS: {TFLOPS:<6.2f}(+{improve:.2f}%)")
    else:
        if not only_show_improved or "cublas" in tag:
            print(f"{out_info:>50}: {out_val}, time:{mean_time}ms, "
                  f"swizzle<block>: {swizzle_stride:<4}, TFLOPS: {TFLOPS:<6.2f}")
    if show_matrix: print(out)
    if args.plot_flops:
        STATIS_INFO[tag] = STATIS_INFO.get(tag, [])
        STATIS_INFO[tag].append(TFLOPS)
        if "cublas" not in tag:
            TOATL_TFLOPS[tag] = TOATL_TFLOPS.get(tag, 0) + TFLOPS
        else:
            global CUBLAS_TOTAL_TFLOPS
            global CUBLAS_TN_TOTAL_TFLOPS
            if tag == "tn(cublas)":
                CUBLAS_TN_TOTAL_TFLOPS += TFLOPS
            else:
                CUBLAS_TOTAL_TFLOPS += TFLOPS

    torch.cuda.synchronize()
    del out_flat
    out_flat = None
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(args.sleep_duration)
    return out, mean_time


def get_topk_tflops():
    topk_tflops = sorted(TOATL_TFLOPS.items(), key=lambda x: x[1], 
                         reverse=True)
    pretty_print_line()
    pretty_print_line(f"THE TOTAL TFLOPS OF {len(topk_tflops)} HGEMM ALGO ON {get_device_name()} DEVICE", " ")
    pretty_print_line()
    for tag, tflops in list(topk_tflops)[::-1]:
        print(f"{tag:>50}: {tflops:>20.2f} TFLOPS")
    if CUBLAS_TN_TOTAL_TFLOPS > 1:
        print(f"{'tn(cublas)':>50}: {CUBLAS_TN_TOTAL_TFLOPS:>20.2f} TFLOPS")    
    if CUBLAS_TOTAL_TFLOPS > 1:
        print(f"{'(cublas)':>50}: {CUBLAS_TOTAL_TFLOPS:>20.2f} TFLOPS")    
    pretty_print_line()
    return list(dict(topk_tflops[:args.plot_topk]).keys())


@torch.no_grad
def get_best_tflops():
    all_tflops = []
    for tag, tflops in STATIS_INFO.items():
        if "cublas" not in tag and "MNK" not in tag:
            all_tflops.append(tflops)
    # [N, NUM_MNK], reduce max on N dim
    all_tflops = torch.tensor(all_tflops, dtype=torch.float)
    best_tflops = torch.max(all_tflops, dim=0, keepdim=False)[0].tolist()
    return best_tflops


def plot_tflops():
    import matplotlib.pyplot as plt
    import numpy as np
    ax: plt.Axes = plt.subplots(figsize=(16, 9))[1] # fig, axs
    plt.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.05)
    ax.set_title(f"My HGEMM vs cuBLAS, {get_device_name()}, Warmup={args.warmup}, Iters={args.iters}")
    ax.set_xlabel("M=N=K")
    ax.set_ylabel("TFLOPS")
    ax.grid(True)
    ax.set_xticks(np.arange(0, len(STATIS_INFO["MNK"]), 1))
    ax.set_xticklabels(STATIS_INFO["MNK"], rotation=45, ha='right')
    exclude_tags = args.exclude_tags.split(",") if args.exclude_tags else []
    exclude_tags.append("MNK")
    exclude_tags = set(exclude_tags)

    topk_tflops = get_topk_tflops()
    STATIS_INFO["(best)"] = get_best_tflops()
    draw_tags = topk_tflops
    draw_tags.append("(cublas)")
    draw_tags.append("tn(cublas)")
    draw_tags.append("(best)")

    def skip_it(tag: str) -> bool:
        for etag in exclude_tags:
            if etag in tag:
                return True
        if tag not in draw_tags:
            return True
        return False
    
    for tag, tflops in STATIS_INFO.items():
        if skip_it(tag): 
            continue
        if tag == "(cublas)":
            ax.plot(tflops, label=tag, linewidth=3, color='orange')
        elif tag == "tn(cublas)":
            ax.plot(tflops, label=tag, linewidth=3, color='green')
        else:
            if "best" in tag and not args.no_plot_best:
                ax.plot(tflops, label=tag, linewidth=4, color='blue')
            else:
                ax.plot(tflops, label=tag, linestyle='--')

    ax.legend()
    device_name = get_device_name().replace(" ", "_")
    if args.save_tag:
        save_path = f"{args.save_dir}/{device_name}_{args.save_tag}.png"
    else:
        save_path = f"{args.save_dir}/{device_name}.png"
    plt.savefig(save_path, dpi=300)
    pretty_print_line(f"plot hgemm TFLOPS done, saved as {save_path}")


def get_mnk(sep: int = args.SEP):
    Ms = list(range(sep, args.MMNK + sep, sep))
    Ns = list(range(sep, args.MMNK + sep, sep))
    Ks = list(range(sep, args.MMNK + sep, sep))
    return Ms, Ns, Ks


Ms, Ns, Ks = get_mnk()
STATIS_INFO["MNK"] = Ms
if args.MNK:
    Ms = [args.MNK]
    Ns = [args.MNK]
    Ks = [args.MNK]
# prefer different M, N, K
if args.M and args.N and args.K:
    Ms = [args.M]
    Ns = [args.N]
    Ks = [args.K]
MAX_M, MAX_N, MAX_K = max(Ms), max(Ns), max(Ks)
# pre allocate for fast profiling.
torch.cuda.synchronize()
start = time.time()
pretty_print_line(f"Allocate buffers for fast profiling start, MAX_M={MAX_M}, MAX_N={MAX_N}, MAX_K={MAX_K}")
A = torch.randn((MAX_M, MAX_K), dtype=torch.half, device="cuda").cuda()
B = torch.randn((MAX_K, MAX_N), dtype=torch.half, device="cuda").cuda()
C = torch.randn((MAX_M, MAX_N), dtype=torch.half, device="cuda").cuda()
torch.cuda.synchronize()
end = time.time()
pretty_print_line(f"Allocate buffers for fast profiling done, time: {(end - start)} s")

PERF_COUNT = 0
for (M, N, K) in zip(Ms, Ns, Ks):
    MAX_TFLOPS = -1
    PERF_COUNT += 1
    pretty_print_line()
    pretty_print_line(f"M={M}, N={N}, K={K}, Warmup={args.warmup}, Iters={args.iters}, {PERF_COUNT}/{len(Ms)}", sep=" ")
    pretty_print_line()
    a = A[:M, :K].contiguous()
    b = B[:K, :N].contiguous()
    c = C[:M, :N].contiguous()
    b_col_major = as_col_major(b)
    torch.cuda.synchronize()
    # CUDA Cores FP16, NN
    if args.enable_cuda_all: # more cuda cores kernel tests
        run_benchmark(hgemm.hgemm_naive_f16, a, b, "(naive)",  c)
        run_benchmark(hgemm.hgemm_t_8x8_sliced_k_f16x8_pack_bcf, a, b, "(f16x8pack+t8x8+bcf)", c)
    if (args.enable_cuda or args.enable_cuda_all) and (not args.no_default):
        run_benchmark(hgemm.hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf, a, b, "(f16x8pack+t8x8+dbuf)", c)
        run_benchmark(hgemm.hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf, a, b, "(f16x8pack+t8x8+k16+dbuf)", c)
    # WMMA API, stages, dsmem, swizzle, NN
    if (args.enable_wmma or args.enable_wmma_all) and (not args.no_default):
        pretty_print_line("WMMA")
        run_benchmark(hgemm.hgemm_wmma_m16n16k16_mma4x2, a, b, "(wmma4x2)", c)
        run_benchmark(hgemm.hgemm_wmma_m16n16k16_mma4x2_warp2x4, a, b, "(wmma4x2+warp2x4)", c)
        run_benchmark(hgemm.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem, a, b, "(wmma4x2+warp2x4+stage3+dsmem)", c, stages=3)
        run_benchmark(hgemm.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem, a, b, "(wmma4x2+warp2x4+stage2+dsmem)", c, stages=2)
        run_benchmark(hgemm.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem, a, b, "(wmma4x2+warp2x4+stage3+dsmem+swizzle<block>)", c, stages=3, swizzle=True)
        run_benchmark(hgemm.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem, a, b, "(wmma4x2+warp2x4+stage2+dsmem+swizzle<block>)", c, stages=2, swizzle=True)
    if args.enable_wmma_all: # more wmma kernel tests.
        run_benchmark(hgemm.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages, a, b, "(wmma4x2+warp2x4+stage3)", c, stages=3)
        run_benchmark(hgemm.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages, a, b, "(wmma4x2+warp2x4+stage2)", c, stages=2)
        run_benchmark(hgemm.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages, a, b, "(wmma4x2+warp2x4+stage3+swizzle<block>)", c, stages=3, swizzle=True)
        run_benchmark(hgemm.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages, a, b, "(wmma4x2+warp2x4+stage2+swizzle<block>)", c, stages=2, swizzle=True)
        # Prefer on NVIDIA TRX 3080 Laptop 16GB GDDR6 device.
        run_benchmark(hgemm.hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem, a, b, "(wmma4x4+warp4x4+stage3+dsmem)", c, stages=3)
        run_benchmark(hgemm.hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem, a, b, "(wmma4x4+warp4x4+stage2+dsmem)", c, stages=2)
        run_benchmark(hgemm.hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem, a, b, "(wmma4x2+warp4x4+stage3+dsmem)", c, stages=3)
        run_benchmark(hgemm.hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem, a, b, "(wmma4x2+warp4x4+stage2+dsmem)", c, stages=2)
        run_benchmark(hgemm.hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem, a, b, "(wmma4x4+warp4x4+stage3+dsmem+swizzle<block>)", c, stages=3, swizzle=True)
        run_benchmark(hgemm.hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem, a, b, "(wmma4x4+warp4x4+stage2+dsmem+swizzle<block>)", c, stages=2, swizzle=True)
        run_benchmark(hgemm.hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem, a, b, "(wmma4x2+warp4x4+stage3+dsmem+swizzle<block>)", c, stages=3, swizzle=True)
        run_benchmark(hgemm.hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem, a, b, "(wmma4x2+warp4x4+stage2+dsmem+swizzle<block>)", c, stages=2, swizzle=True)
    # MMA API, stages, dsmem, swizzle, NN
    if (args.enable_mma or args.enable_mma_all) and (not args.no_default):
        pretty_print_line("MMA")
    if args.enable_mma_all: # more mma kernel tests.
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4, a, b, "(mma2x4+warp4x4)", c)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4_stages, a, b, "(mma2x4+warp4x4+stage3)", c, stages=3)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4_stages, a, b, "(mma2x4+warp4x4+stage2)", c, stages=2)
    if (args.enable_mma or args.enable_mma_all) and (not args.no_default):
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem, a, b, "(mma2x4+warp4x4+stage3+dsmem)", c, stages=3)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem, a, b, "(mma2x4+warp4x4+stage2+dsmem)", c, stages=2)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem, a, b, "(mma2x4+warp4x4x2+stage4+dsmem)", c, stages=4)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem, a, b, "(mma2x4+warp4x4x2+stage3+dsmem)", c, stages=3)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem, a, b, "(mma2x4+warp4x4x2+stage2+dsmem)", c, stages=2)
    if args.enable_mma_all: # more mma kernel tests.
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr, a, b, "(mma2x4+warp4x4x2+stage4+dsmem+rr)", c, stages=4)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr, a, b, "(mma2x4+warp4x4x2+stage3+dsmem+rr)", c, stages=3)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr, a, b, "(mma2x4+warp4x4x2+stage2+dsmem+rr)", c, stages=2)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4, a, b, "(mma2x4+warp4x4x2+stage4+dsmem+x4)", c, stages=4)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4, a, b, "(mma2x4+warp4x4x2+stage3+dsmem+x4)", c, stages=3)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4, a, b, "(mma2x4+warp4x4x2+stage2+dsmem+x4)", c, stages=2)
    if (args.enable_mma or args.enable_mma_all) and (not args.no_default):
        # Thread block swizzle
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem, a, b, "(mma2x4+warp4x4+stage3+dsmem+swizzle<block>)", c, stages=3, swizzle=True)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem, a, b, "(mma2x4+warp4x4+stage2+dsmem+swizzle<block>)", c, stages=2, swizzle=True)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem, a, b, "(mma2x4+warp4x4x2+stage4+dsmem+swizzle<block>)", c, stages=4, swizzle=True)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem, a, b, "(mma2x4+warp4x4x2+stage3+dsmem+swizzle<block>)", c, stages=3, swizzle=True)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem, a, b, "(mma2x4+warp4x4x2+stage2+dsmem+swizzle<block>)", c, stages=2, swizzle=True)
    if args.enable_mma_all:
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4_stages, a, b, "(mma2x4+warp4x4+stage3+swizzle<block>)", c, stages=3, swizzle=True)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4_stages, a, b, "(mma2x4+warp4x4+stage2+swizzle<block>)", c, stages=2, swizzle=True)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr, a, b, "(mma2x4+warp4x4x2+stage4+dsmem+swizzle<block>+rr)", c, stages=4, swizzle=True)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr, a, b, "(mma2x4+warp4x4x2+stage3+dsmem+swizzle<block>+rr)", c, stages=3, swizzle=True)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr, a, b, "(mma2x4+warp4x4x2+stage2+dsmem+swizzle<block>+rr)", c, stages=2, swizzle=True)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4, a, b, "(mma2x4+warp4x4x2+stage4+dsmem+swizzle<block>+x4)", c, stages=4, swizzle=True)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4, a, b, "(mma2x4+warp4x4x2+stage3+dsmem+swizzle<block>+x4)", c, stages=3, swizzle=True)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4, a, b, "(mma2x4+warp4x4x2+stage2+dsmem+swizzle<block>+x4)", c, stages=2, swizzle=True)
    # TN(MMA/CuTe), TN layout: A row major with shape [M,K], B col major with shape [K,N]
    if any((args.enable_mma_tn, args.enable_cute_tn)):
        pretty_print_line("TN(MMA/CuTe)")
    if args.enable_mma_tn:
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn, a, b_col_major, "tn(mma2x4+warp4x4+stage3+dsmem)", c, stages=3)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn, a, b_col_major, "tn(mma2x4+warp4x4+stage2+dsmem)", c, stages=2)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn, a, b_col_major, "tn(mma2x4+warp4x4+stage3+dsmem+swizzle<block>)", c, stages=3, swizzle=True)
        run_benchmark(hgemm.hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn, a, b_col_major, "tn(mma2x4+warp4x4+stage2+dsmem+swizzle<block>)", c, stages=2, swizzle=True)
    if args.enable_cute_tn:
        run_benchmark(hgemm.hgemm_mma_stages_tn_cute, a, b_col_major, "tn(cute+stage4+swizzle<smem>)", c, stages=4)
        run_benchmark(hgemm.hgemm_mma_stages_tn_cute, a, b_col_major, "tn(cute+stage3+swizzle<smem>)", c, stages=3)
        run_benchmark(hgemm.hgemm_mma_stages_tn_cute, a, b_col_major, "tn(cute+stage2+swizzle<smem>)", c, stages=2)
        run_benchmark(hgemm.hgemm_mma_stages_tn_cute, a, b_col_major, "tn(cute+stage4+swizzle<smem+block>)", c, stages=4, swizzle=True)
        run_benchmark(hgemm.hgemm_mma_stages_tn_cute, a, b_col_major, "tn(cute+stage3+swizzle<smem+block>)", c, stages=3, swizzle=True)
        run_benchmark(hgemm.hgemm_mma_stages_tn_cute, a, b_col_major, "tn(cute+stage2+swizzle<smem+block>)", c, stages=2, swizzle=True)
    # TN layout: cublas
    if not args.disable_cublas_tn and any((args.enable_mma_tn, args.enable_cute_tn)):
        hgemm.init_cublas_handle()
        run_benchmark(hgemm.hgemm_cublas_tensor_op_tn, a, b_col_major, "tn(cublas)", c)
        hgemm.destroy_cublas_handle()
    # NN layout: cublas/torch
    if (not args.disable_cublas) and any((
        args.enable_mma, args.enable_mma_all, args.enable_wmma, args.enable_wmma_all, 
        args.enable_cuda, args.enable_cuda_all, args.enable_torch)):
        hgemm.init_cublas_handle()
        run_benchmark(hgemm.hgemm_cublas_tensor_op_nn, a, b, "(cublas)", c)
        hgemm.destroy_cublas_handle()
    if args.enable_torch:
        run_benchmark(partial(torch.matmul, out=c), a, b, "(torch)")
    torch.cuda.synchronize()
    # Avoid OOM
    del a; a = None
    del b; b = None
    del c; c = None
    del b_col_major; 
    b_col_major = None
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    pretty_print_line()

if args.show_memory:
    pretty_print_line()
    print(torch.cuda.memory_summary())
    pretty_print_line()

if args.plot_flops:
    plot_tflops()
