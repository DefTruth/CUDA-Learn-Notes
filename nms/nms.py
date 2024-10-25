import torch
import time
from torch.utils.cpp_extension import load
from typing import Optional
from functools import partial
from torchvision.ops import nms
torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="nms_lib",
    sources=["nms.cu"],
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


def generate_random_data(Nboxes):
    boxes = torch.rand(Nboxes, 4)
    for i in range(Nboxes):
        if boxes[i, 0] > boxes[i, 2]:
            boxes[i, 0], boxes[i, 2] = boxes[i, 2], boxes[i, 0]
        if boxes[i, 1] > boxes[i, 3]:
            boxes[i, 1], boxes[i, 3] = boxes[i, 3], boxes[i, 1]
    scores = torch.rand(Nboxes)
    return boxes, scores


def run_benchmark(
    perf_func: callable,
    scores: torch.Tensor,
    boxes: torch.Tensor,
    thresholds: float,
    tag: str,
    warmup: int = 10,
    iters: int = 100,
    show_all: bool = False,
):
    # warmup
    for i in range(warmup):
        out = perf_func(scores, boxes, thresholds)
    torch.cuda.synchronize()

    start = time.time()
    # iters
    for i in range(iters):
        out = perf_func(scores, boxes, thresholds)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"{tag}"
    out_val = sorted(out.flatten().detach().cpu().numpy().tolist())
    len_val = len(out_val)
    out_val = out_val[-min(3, len_val) :]
    out_val = [f"{v:<5}" for v in out_val]
    print(f"{out_info:>14}: {out_val}, len of keep: {len_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time


Nboxes = [1024, 2048, 4096, 8192]
thresholds = 0.5


for nboxes in Nboxes:
    print("-" * 85)
    print(" " * 40 + f"nboxes={nboxes}")
    boxes, scores = generate_random_data(nboxes)
    boxes = boxes.cuda().float().contiguous()
    scores = scores.cuda().float().contiguous()
    run_benchmark(lib.nms, boxes, scores, thresholds, "nms")
    run_benchmark(nms, boxes, scores, thresholds, "nms_th")
    print("-" * 85)
