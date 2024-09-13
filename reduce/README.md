# Reduce

## 0x00 说明

包含以下内容：

- [X] warp_reduce_fp32/fp16/bf16_kernel
- [X] block_reduce_fp32_kernel
- [X] block_all_reduce_sum_f32_f32_kernel
- [X] block_all_reduce_sum_f32x4_f32_kernel(float4向量化版本)
- [X] block_all_reduce_sum_f16_f16_kernel(fp16版本，使用fp16 acc)
- [X] block_all_reduce_sum_f16_f32_kernel(fp16版本，使用fp32 acc)
- [X] block_all_reduce_sum_f16x2_f16_kernel(fp16向量化版本，使用fp16 acc)
- [X] block_all_reduce_sum_f16x2_f32_kernel(fp16向量化版本，使用fp32 acc)
- [X] block_all_reduce_sum_bf16_bf16_kernel(bf16版本，使用bf16 acc)
- [X] block_all_reduce_sum_bf16_f32_kernel(bf16版本，使用fp32 acc)
- [X] block_all_reduce_sum_bf16x2_bf16_kernel(bf16向量化版本，使用bf16 acc)
- [X] block_all_reduce_sum_bf16x2_f32_kernel(bf16向量化版本，使用fp32 acc)
- [X] block_all_reduce_sum_fp8_e4m3_f16_kernel(fp8_e4m3版本，使用fp16 acc)
- [X] block_all_reduce_sum_fp8_e5m2_f16_kernel(fp8_e5m2版本，使用fp16 acc)
- [X] block_all_reduce_sum_i8_i32_kernel(i8版本，使用i32 acc)
- [X] PyTorch bindings for block reduce **fp32/fp16/bf16/fp8/i8** kernels

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
TORCH_BINDING_BLOCK_ALL_REDUCE(fp8_e5m2, f16,  torch::kFloat8_e5m2,   __nv_fp8_storage_t, 1)
TORCH_BINDING_BLOCK_ALL_REDUCE_I(i8,     i32,  torch::kInt8,          int8_t,             1)
```

## 测试

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长
export TORCH_CUDA_ARCH_LIST=Ada 
python3 block_all_reduce.py
```

输出:

```bash
--------------------------------------------------------------------------------
       out_f32f32: -596.47686768  , time:0.01075578ms
     out_f32x4f32: -596.47680664  , time:0.01164222ms
    out_f32f32_th: -596.47698975  , time:0.01194477ms
--------------------------------------------------------------------------------
       out_f16f16: -596.67187500  , time:0.01060605ms
       out_f16f32: -596.70013428  , time:0.01052594ms
     out_f16x2f32: -596.76770020  , time:0.01054215ms
     out_f16x2f16: -596.66699219  , time:0.01048732ms
    out_f16f16_th: -596.50000000  , time:0.01203275ms
--------------------------------------------------------------------------------
     out_bf16bf16: -595.89062500  , time:0.01056409ms
      out_bf16f32: -594.54827881  , time:0.01535106ms
    out_bf16x2f32: -593.80480957  , time:0.01053929ms
   out_bf16x2bf16: -594.53515625  , time:0.01057482ms
  out_bf16bf16_th: -596.00000000  , time:0.01200724ms
--------------------------------------------------------------------------------
    out_f8e4m3f16: -607.60742188  , time:0.01056290ms
 out_f8e4m3f16_th: -608.00000000  , time:0.01213408ms
--------------------------------------------------------------------------------
    out_f8e5m2f16: -582.93847656  , time:0.01059389ms
 out_f8e5m2f16_th: -583.00000000  , time:0.01211667ms
--------------------------------------------------------------------------------
        out_i8i32: 376832.00000000, time:0.01021981ms
     out_i8i32_th: 376832.00000000, time:0.01966381ms
--------------------------------------------------------------------------------
```
