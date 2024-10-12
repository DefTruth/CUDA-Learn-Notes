import torch
import time
from torch.utils.cpp_extension import load
from functools import partial
from typing import Optional
from torch.nn.functional import embedding

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="embedding",
    sources=["embedding.cu"],
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
    b: torch.Tensor,
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
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>23}: {out_val}, time:{mean_time:.6f}ms")
    if show_all:
        print(out)
    return out.clone(), mean_time


Ms = [1024, 4096]  # max value of token_ids
Ns = [2048, 4096]  # seqlen
Ks = [512, 1024]  # embedding size
MNKs = [(M, N, K) for M in Ms for N in Ns for K in Ks]
for M, N, K in MNKs:
    print("-" * 110)
    print(" " * 45 + f"MaxV={M}, SeqLen={N}, EmbSize={K}")
    i = torch.randint(0, M, size=(N,)).cuda().int().contiguous()
    weight = torch.randn((M, K)).float().cuda().contiguous()
    o = torch.zeros((N, K)).float().cuda().contiguous()

    run_benchmark(lib.embedding_f32, i, weight, "f32", o)
    run_benchmark(lib.embedding_f32x4, i, weight, "f32x4", o)
    run_benchmark(lib.embedding_f32x4_pack, i, weight, "f32x4_pack", o)
    run_benchmark(partial(embedding), i, weight, "f32_th")

    print("-" * 110)
    weight_f16 = torch.randn((M, K)).half().cuda().contiguous()
    o_f16 = torch.zeros((N, K)).half().cuda().contiguous()
    run_benchmark(lib.embedding_f16, i, weight_f16, "f16", o_f16)
    run_benchmark(lib.embedding_f16x8, i, weight_f16, "f16x8", o_f16)
    run_benchmark(lib.embedding_f16x8_pack, i, weight_f16, "f16x8_pack", o_f16)
    run_benchmark(partial(embedding), i, weight_f16, "f16_th")
    print("-" * 110)
