import torch
import time 
from torch.utils.cpp_extension import load
from functools import partial
from typing import Optional

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(name='softmax_lib', 
           sources=['softmax.cu'], 
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


def run_benchmark(perf_func: callable, x: torch.Tensor, 
                  tag: str, out: Optional[torch.Tensor] = None, 
                  warmup: int = 10, iters: int = 1000,
                  show_all: bool = False):
    if out is not None: 
        out.fill_(0)      
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
    total_time = (end - start) * 1000 # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>24}: {out_val}, time:{mean_time:.8f}ms")
    if show_all: print(out)
    return out, mean_time

# grid memory fence
print("-" * 100)
N = 128 * 128
print(" " * 45 + f"N={N}")
print("-" * 100)
x = torch.randn((N)).cuda().float()
out = torch.zeros_like(x).cuda().float().contiguous()
run_benchmark(lib.softmax_f32,                        x, "f32(fence)",   out)
run_benchmark(lib.softmax_f32x4,                      x, "f32x4(fence)", out)
run_benchmark(partial(torch.softmax, dim=0, out=out), x, "f32_th")

# per token softmax
print("-" * 100)
S, H = 4096, 256
print(" " * 45 + f"S={S}, H={H}")
print("-" * 100)
x = torch.randn((S, H)).cuda().float().contiguous()
out = torch.zeros_like(x).cuda().float().contiguous()
run_benchmark(lib.softmax_f32_per_token,              x, "f32(per)",    out)
run_benchmark(lib.softmax_f32x4_per_token,            x, "f32x4(per)",  out)
run_benchmark(lib.safe_softmax_f32_per_token,         x, "f32(safe)",   out) 
run_benchmark(lib.online_softmax_f32_per_token,       x, "f32(online)", out)
run_benchmark(lib.safe_softmax_f32x4_per_token,       x, "f32x4(safe)", out) 
run_benchmark(lib.online_softmax_f32_per_token,       x, "f32(online)",    out)
run_benchmark(partial(torch.softmax, dim=1, out=out), x, "f32_th(per)")

print("-" * 100)
x_f16 = x.half().contiguous()
out_f16 = out.half().contiguous()
run_benchmark(lib.safe_softmax_f16_f32_per_token,         x_f16, "f16f32(safe)",       out_f16) 
run_benchmark(lib.safe_softmax_f16x2_f32_per_token,       x_f16, "f16x2f32(safe)",     out_f16) 
run_benchmark(lib.safe_softmax_f16x8_pack_f32_per_token,  x_f16, "f16x8packf32(safe)", out_f16) 
run_benchmark(partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)")
print("-" * 100)

# per token softmax
print("-" * 100)
S, H = 4096, 512
print(" " * 45 + f"S={S}, H={H}")
print("-" * 100)
x = torch.randn((S, H)).cuda().float().contiguous()
out = torch.zeros_like(x).cuda().float().contiguous()
run_benchmark(lib.softmax_f32_per_token,              x, "f32(per)",    out)
run_benchmark(lib.softmax_f32x4_per_token,            x, "f32x4(per)",  out)
run_benchmark(lib.safe_softmax_f32_per_token,         x, "f32(safe)",   out) 
run_benchmark(lib.online_softmax_f32_per_token,       x, "f32(online)", out)
run_benchmark(lib.safe_softmax_f32x4_per_token,       x, "f32x4(safe)", out) 
run_benchmark(lib.online_softmax_f32_per_token,       x, "f32(online)", out)
run_benchmark(partial(torch.softmax, dim=1, out=out), x, "f32_th(per)")

print("-" * 100)
x_f16 = x.half().contiguous()
out_f16 = out.half().contiguous()
run_benchmark(lib.safe_softmax_f16_f32_per_token,         x_f16, "f16f32(safe)",       out_f16) 
run_benchmark(lib.safe_softmax_f16x2_f32_per_token,       x_f16, "f16x2f32(safe)",     out_f16) 
run_benchmark(lib.safe_softmax_f16x8_pack_f32_per_token,  x_f16, "f16x8packf32(safe)", out_f16) 
run_benchmark(partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)")
print("-" * 100)

# per token softmax
print("-" * 100)
S, H = 4096, 1024
print(" " * 45 + f"S={S}, H={H}")
print("-" * 100)
x = torch.randn((S, H)).cuda().float().contiguous()
out = torch.zeros_like(x).cuda().float().contiguous()
run_benchmark(lib.softmax_f32_per_token,              x, "f32(per)",    out)
run_benchmark(lib.softmax_f32x4_per_token,            x, "f32x4(per)",  out)
run_benchmark(lib.safe_softmax_f32_per_token,         x, "f32(safe)",   out) 
run_benchmark(lib.online_softmax_f32_per_token,       x, "f32(online)", out)
run_benchmark(lib.safe_softmax_f32x4_per_token,       x, "f32x4(safe)", out) 
run_benchmark(lib.online_softmax_f32_per_token,       x, "f32(online)", out)
run_benchmark(partial(torch.softmax, dim=1, out=out), x, "f32_th(per)")

print("-" * 100)
x_f16 = x.half().contiguous()
out_f16 = out.half().contiguous()
run_benchmark(lib.safe_softmax_f16_f32_per_token,         x_f16, "f16f32(safe)",       out_f16) 
run_benchmark(lib.safe_softmax_f16x2_f32_per_token,       x_f16, "f16x2f32(safe)",     out_f16) 
run_benchmark(lib.safe_softmax_f16x8_pack_f32_per_token,  x_f16, "f16x8packf32(safe)", out_f16) 
run_benchmark(partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)")
print("-" * 100)

# per token softmax
print("-" * 100)
S, H = 4096, 2048
print(" " * 45 + f"S={S}, H={H}")
print("-" * 100)
x = torch.randn((S, H)).cuda().float().contiguous()
out = torch.zeros_like(x).cuda().float().contiguous()
run_benchmark(lib.softmax_f32x4_per_token,            x, "f32x4(per)",  out)
run_benchmark(lib.safe_softmax_f32x4_per_token,       x, "f32x4(safe)", out) 
run_benchmark(partial(torch.softmax, dim=1, out=out), x, "f32_th(per)")

print("-" * 100)
x_f16 = x.half().contiguous()
out_f16 = out.half().contiguous()
run_benchmark(lib.safe_softmax_f16x2_f32_per_token,       x_f16, "f16x2f32(safe)",     out_f16) 
run_benchmark(lib.safe_softmax_f16x8_pack_f32_per_token,  x_f16, "f16x8packf32(safe)", out_f16) 
run_benchmark(partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)")
print("-" * 100)

# per token softmax
print("-" * 100)
S, H = 4096, 4096
print(" " * 45 + f"S={S}, H={H}")
print("-" * 100)
x = torch.randn((S, H)).cuda().float().contiguous()
out = torch.zeros_like(x).cuda().float().contiguous()
run_benchmark(lib.softmax_f32x4_per_token,            x, "f32x4(per)",  out)
run_benchmark(lib.safe_softmax_f32x4_per_token,       x, "f32x4(safe)", out) 
run_benchmark(partial(torch.softmax, dim=1, out=out), x, "f32_th(per)")

print("-" * 100)
x_f16 = x.half().contiguous()
out_f16 = out.half().contiguous()
run_benchmark(lib.safe_softmax_f16x8_pack_f32_per_token,  x_f16, "f16x8packf32(safe)", out_f16) 
run_benchmark(partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)")
print("-" * 100)

# per token softmax
print("-" * 100)
S, H = 4096, 8192
print(" " * 45 + f"S={S}, H={H}")
print("-" * 100)
x = torch.randn((S, H)).cuda().float().contiguous()
out = torch.zeros_like(x).cuda().float().contiguous()
x_f16 = x.half().contiguous()
out_f16 = out.half().contiguous()
run_benchmark(lib.safe_softmax_f16x8_pack_f32_per_token,  x_f16, "f16x8packf32(safe)", out_f16) 
run_benchmark(partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)")

# per token softmax
print("-" * 100)
S, H = 8192, 8192
print(" " * 45 + f"S={S}, H={H}")
print("-" * 100)
x = torch.randn((S, H)).cuda().float().contiguous()
out = torch.zeros_like(x).cuda().float().contiguous()
x_f16 = x.half().contiguous()
out_f16 = out.half().contiguous()
run_benchmark(lib.safe_softmax_f16x8_pack_f32_per_token,  x_f16, "f16x8packf32(safe)", out_f16) 
run_benchmark(partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)")
print("-" * 100)
