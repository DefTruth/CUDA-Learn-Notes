import torch
import time 
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(name='block_all_reduce_lib', 
           sources=['block_all_reduce.cu'], 
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
           extra_cflags=['-std=c++17'])


def run_benchmark(perf_func: callable, values: torch.Tensor, tag: str, 
                  warmup: int = 10, iters: int = 1000):
    # if perf_func.__name__ == torch.sum.__name__:
    #     values = values.float() # for precision
    for i in range(warmup):
        out = perf_func(values) # warmup
    torch.cuda.synchronize()
    start = time.time()
    for i in range(iters):
        out = perf_func(values)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000 # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.item()
    if tag.startswith("i8"):
        print(f"{out_info:>25}: {out_val:<15}, time:{mean_time:.8f}ms")
    else:
        print(f"{out_info:>25}: {out_val:<15.8f}, time:{mean_time:.8f}ms")
    return out, mean_time


Ss = [1024, 2048, 4096]
Ks = [1024, 2048, 4096]
SKs = [(S, K) for S in Ss for K in Ks]

for (S, K) in SKs:
    print("-" * 80)
    print(" " * 40 + f"S={S}, K={K}")
    values = torch.randn((S, K)).cuda().float()
    run_benchmark(lib.block_all_reduce_sum_f32_f32,   values, "f32f32")
    run_benchmark(lib.block_all_reduce_sum_f32x4_f32, values, "f32x4f32")
    run_benchmark(torch.sum,                          values, "f32f32_th")

    print("-" * 80)
    values_half = values.half()
    run_benchmark(lib.block_all_reduce_sum_f16_f16,        values_half, "f16f16")
    run_benchmark(lib.block_all_reduce_sum_f16_f32,        values_half, "f16f32")
    run_benchmark(lib.block_all_reduce_sum_f16x2_f32,      values_half, "f16x2f32")
    run_benchmark(lib.block_all_reduce_sum_f16x2_f16,      values_half, "f16x2f16")
    run_benchmark(lib.block_all_reduce_sum_f16x8_pack_f16, values_half, "f16x8packf16")
    run_benchmark(lib.block_all_reduce_sum_f16x8_pack_f32, values_half, "f16x8packf32")
    run_benchmark(torch.sum,                               values_half, "f16f16_th")

    print("-" * 80)
    values_bf16 = values.bfloat16()
    run_benchmark(lib.block_all_reduce_sum_bf16_bf16,        values_bf16, "bf16bf16")
    run_benchmark(lib.block_all_reduce_sum_bf16_f32,         values_bf16, "bf16f32")
    run_benchmark(lib.block_all_reduce_sum_bf16x2_f32,       values_bf16, "bf16x2f32")
    run_benchmark(lib.block_all_reduce_sum_bf16x2_bf16,      values_bf16, "bf16x2bf16")
    run_benchmark(lib.block_all_reduce_sum_bf16x8_pack_f32,  values_bf16, "bf16x8packf32")
    run_benchmark(lib.block_all_reduce_sum_bf16x8_pack_bf16, values_bf16, "bf16x8packbf16")
    run_benchmark(torch.sum,                                 values_bf16, "bf16bf16_th")

    print("-" * 80)
    values_f8e4m3 = values.to(dtype=torch.float8_e4m3fn)
    run_benchmark(lib.block_all_reduce_sum_fp8_e4m3_f16,         values_f8e4m3,        "f8e4m3f16")
    run_benchmark(lib.block_all_reduce_sum_fp8_e4m3x16_pack_f16, values_f8e4m3,        "f8e4m3x16packf16")
    run_benchmark(torch.sum,                                     values_f8e4m3.half(), "f8e4m3f16_th") # torch.sum not support fp8

    print("-" * 80)
    values_f8e5m2 = values.to(dtype=torch.float8_e5m2)
    run_benchmark(lib.block_all_reduce_sum_fp8_e5m2_f16,         values_f8e5m2,        "f8e5m2f16")
    run_benchmark(lib.block_all_reduce_sum_fp8_e5m2x16_pack_f16, values_f8e5m2,        "f8e5m2x16packf16")
    run_benchmark(torch.sum,                                     values_f8e5m2.half(), "f8e5m2f16_th") # torch.sum not support fp8

    print("-" * 80)
    values_i8 = values.to(dtype=torch.int8)
    run_benchmark(lib.block_all_reduce_sum_i8_i32,         values_i8, "i8i32")
    run_benchmark(lib.block_all_reduce_sum_i8x16_pack_i32, values_i8, "i8x16packi32")
    run_benchmark(torch.sum,                               values_i8, "i8i32_th")
    print("-" * 80)
