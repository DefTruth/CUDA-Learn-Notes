# Modified from https://github.com/tspeterkim/flash-attention-minimal/blob/main/bench.py
import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

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

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

print('=== profiling manual attention ===')

def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for _ in range(10):
        manual_result = manual_attn(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

torch.cuda.synchronize()
torch.cuda.empty_cache()
print('=== profiling flash_attn_1_fwd_f32 attention === ')
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for _ in range(10):
        custom_result = custom_flash_attn.flash_attn_1_fwd_f32(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('attn values sanity check:', torch.allclose(custom_result, manual_result, rtol=0, atol=1e-02))
