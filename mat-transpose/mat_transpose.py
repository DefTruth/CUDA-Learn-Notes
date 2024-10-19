import torch
import time
from torch.utils.cpp_extension import load
from typing import Optional
from functools import partial

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="mat_transpose_lib",
    sources=["mat_transpose.cu"],
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
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    # warmup
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
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    real_t = f"{out.T.equal(x)}"
    out_val = out[:2, :2].flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>35}: {out_val}, validate {real_t:<5}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time


Ms = [1024, 2048, 4096]
Ns = [1024, 2048, 4096]
MNs = [(M, N) for M in Ms for N in Ns]
copy_x = lambda x: x
# show the three elements x[0][0], x[0][1], x[1][0]
for M, N in MNs:
    print("-" * 130)
    print(" " * 55 + f"M={M}, N={N}")
    x = torch.randn((M, N)).cuda().float().contiguous()
    y = torch.randn((N, M)).cuda().float().contiguous()
    run_benchmark(partial(copy_x), x, "original")
    run_benchmark(lib.mat_transpose_f32_col2row, x, "f32_col2row", y)
    run_benchmark(lib.mat_transpose_f32_row2col, x, "f32_row2col", y)
    run_benchmark(lib.mat_transpose_f32_col2row2d, x, "f32_col2row(2d)", y)
    run_benchmark(lib.mat_transpose_f32_row2col2d, x, "f32_row2col(2d)", y)
    if M == N:
        run_benchmark(lib.mat_transpose_f32_diagonal2d, x, "f32_diagnonal", y)
    run_benchmark(lib.mat_transpose_f32x4_col2row, x, "f32x4_col2row", y)
    run_benchmark(lib.mat_transpose_f32x4_row2col, x, "f32x4_row2col", y)
    run_benchmark(lib.mat_transpose_f32x4_col2row2d, x, "f32x4_col2row(2d)", y)
    run_benchmark(lib.mat_transpose_f32x4_row2col2d, x, "f32x4_row2col(2d)", y)
    run_benchmark(lib.mat_transpose_f32x4_shared_col2row2d, x, "f32x4_shared_col2row(2d)", y)
    run_benchmark(lib.mat_transpose_f32x4_shared_row2col2d, x, "f32x4_shared_row2col(2d)", y)
    run_benchmark(lib.mat_transpose_f32x4_shared_bcf_col2row2d, x, "f32x4_shared_bcf_col2row(2d)", y)
    run_benchmark(lib.mat_transpose_f32x4_shared_bcf_row2col2d, x, "f32x4_shared_bcf_row2col(2d)", y)
    run_benchmark(partial(torch.transpose_copy, dim0=0, dim1=1, out=y), x, "f32_th")
    print("-" * 130)
