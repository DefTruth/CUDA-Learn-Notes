import torch
import time 
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(name='hist_lib', 
           sources=['histogram.cu'], 
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

a = torch.tensor(list(range(10))*1000, dtype=torch.int32).cuda()
h_i32 = lib.histogram_i32(a)
print("-" * 80)
for i in range(h_i32.shape[0]):
    print(f"h_i32   {i}: {h_i32[i]}")
    
print("-" * 80)
h_i32x4 = lib.histogram_i32x4(a)
for i in range(h_i32x4.shape[0]):
    print(f"h_i32x4 {i}: {h_i32x4[i]}")
print("-" * 80)
