# Modified from https://github.com/tspeterkim/flash-attention-minimal/blob/main/bench.py
import math
import time
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)
# Load the CUDA kernel as a python module
custom_flash_attn = load(name='custom_flash_attn', 
                         sources=[
                            'flash_attn.cc',
                            'flash_attn_1_fwd_f32.cu',
                            'flash_attn_2_fwd_f32.cu'
                         ], 
                         extra_cuda_cflags=['-O2'])

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 16
n_head = 12
seq_len = 64
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd).float().cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).float().cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).float().cuda()
q.requires_grad = False
k.requires_grad = False
v.requires_grad = False
print('=== profiling manual attention ===')

def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

for _ in range(2): 
    manual_result = manual_attn(q, k, v) # warmup

torch.cuda.synchronize()
with torch.autograd.profiler.profile(use_cuda=True, with_flops=True) as prof:
    with torch.autograd.profiler.record_function("manual_attn"):
        manual_result = manual_attn(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

for _ in range(2): 
    custom_result = custom_flash_attn.flash_attn_1_fwd_f32(q, k, v) # warmup
print('=== profiling flash_attn_1_fwd_f32 attention === ')
with torch.autograd.profiler.profile(use_cuda=True, with_flops=True) as prof:
     with torch.autograd.profiler.record_function("flash_attn_1_fwd_f32"):
        custom_result = custom_flash_attn.flash_attn_1_fwd_f32(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
print('attn values sanity check:', torch.allclose(custom_result, manual_result, rtol=0, atol=1e-02))

# Why custom flash attn is slow than naive attn in for loop test ?
REPEAT = 10
manual_result = manual_attn(q, k, v) # warmup
st = time.time()
for _ in range(REPEAT):
    manual_result = manual_attn(q, k, v)
    torch.cuda.synchronize()
print(f"manual attention mean time(ms): {((time.time() - st) * 1000) / REPEAT}")
custom_result = custom_flash_attn.flash_attn_1_fwd_f32(q, k, v)  # warmup
st = time.time()
for _ in range(REPEAT):
    custom_result = custom_flash_attn.flash_attn_1_fwd_f32(q, k, v)
    torch.cuda.synchronize()
print(f"flash_attn_1_fwd_f32 mean time(ms): {((time.time() - st) * 1000) / REPEAT}")

