# Reduce

## 0x00 说明

包含以下内容：

- [X] warp_reduce_fp32/fp16/bf16_kernel
- [X] block_reduce_fp32_kernel
- [X] block_all_reduce_sum_f32_acc_with_f32_kernel
- [X] block_all_reduce_sum_f32x4_acc_with_f32_kernel(float4向量化版本)
- [X] block_all_reduce_sum_f16_acc_with_f16_kernel(fp16版本，使用fp16 acc)
- [X] block_all_reduce_sum_f16_acc_with_f32_kernel(fp16版本，使用fp32 acc)
- [X] block_all_reduce_sum_f16x2_acc_with_f16_kernel(fp16向量化版本，使用fp16 acc)
- [X] block_all_reduce_sum_f16x2_acc_with_f32_kernel(fp16向量化版本，使用fp32 acc)
- [X] block_all_reduce_sum_bf16_acc_with_bf16_kernel(bf16版本，使用bf16 acc)
- [X] block_all_reduce_sum_bf16_acc_with_f32_kernel(bf16版本，使用fp32 acc)
- [X] block_all_reduce_sum_bf16x2_acc_with_bf16_kernel(bf16向量化版本，使用bf16 acc)
- [X] block_all_reduce_sum_bf16x2_acc_with_f32_kernel(bf16向量化版本，使用fp32 acc)
- [X] block_all_reduce_sum_fp8_e4m3_acc_with_f16_kernel(fp8_e4m3版本，使用fp16 acc)
- [X] PyTorch bindings for block reduce **fp32/fp16/bf16/fp8** kernels

所有支持的block all reduce kernel:

```c++
// packed_type, acc_type, th_type, element_type, n_elements_per_pack
TORCH_BINDING_BLOCK_ALL_REDUCE(f32,      f32,  torch::kFloat32,       float,              1)
TORCH_BINDING_BLOCK_ALL_REDUCE(f32x4,    f32,  torch::kFloat32,       float,              4)
TORCH_BINDING_BLOCK_ALL_REDUCE(f16,      f16,  torch::kHalf,          half,               1)
TORCH_BINDING_BLOCK_ALL_REDUCE(f16,      f32,  torch::kHalf,          half,               1)
TORCH_BINDING_BLOCK_ALL_REDUCE(f16x2,    f16,  torch::kHalf,          half,               2)
TORCH_BINDING_BLOCK_ALL_REDUCE(f16x2,    f32,  torch::kHalf,          half,               2)
TORCH_BINDING_BLOCK_ALL_REDUCE(bf16,     bf16, torch::kBFloat16,      __nv_bfloat16,      1)
TORCH_BINDING_BLOCK_ALL_REDUCE(bf16,     f32,  torch::kBFloat16,      __nv_bfloat16,      1)
TORCH_BINDING_BLOCK_ALL_REDUCE(bf16x2,   bf16, torch::kBFloat16,      __nv_bfloat16,      2)
TORCH_BINDING_BLOCK_ALL_REDUCE(bf16x2,   f32,  torch::kBFloat16,      __nv_bfloat16,      2)
TORCH_BINDING_BLOCK_ALL_REDUCE(fp8_e4m3, f16,  torch::kFloat8_e4m3fn, __nv_fp8_storage_t, 1)
```

## 测试

```bash
TORCH_CUDA_ARCH_LIST=Ada # 只测试Ada架构 不指定默认编译所有架构 耗时较长
python3 block_all_reduce.py
```

输出:

```bash
--------------------------------------------------------------------------------
       out_f32f32: 123.12593842 , time:0.01150918ms
     out_f32x4f32: 123.12657928 , time:0.01162839ms
    out_f32f32_th: 123.12606812 , time:0.01259112ms
--------------------------------------------------------------------------------
       out_f16f16: 123.17724609 , time:0.01111102ms
       out_f16f32: 123.10200500 , time:0.01112914ms
     out_f16x2f32: 122.77922058 , time:0.01101422ms
     out_f16x2f16: 122.53564453 , time:0.01100302ms
    out_f16f16_th: 123.12500000 , time:0.01260138ms
--------------------------------------------------------------------------------
     out_bf16bf16: 126.17968750 , time:0.01109409ms
      out_bf16f32: 122.96487427 , time:0.01116443ms
    out_bf16x2f32: 122.93243408 , time:0.01112390ms
   out_bf16x2bf16: 120.63281250 , time:0.01102233ms
  out_bf16bf16_th: 123.00000000 , time:0.01253748ms
--------------------------------------------------------------------------------
    out_f8e4m3f16: 123.31835938 , time:0.01106477ms
 out_f8e4m3f16_th: 123.68750000 , time:0.01271629ms
--------------------------------------------------------------------------------
```
