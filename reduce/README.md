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
- [X] block_all_reduce_sum_f16x8_pack_f16_kernel(fp16向量化版本，使用fp16 acc, pack)
- [X] block_all_reduce_sum_f16x8_pack_f32_kernel(fp16向量化版本，使用fp32 acc, pack)
- [X] block_all_reduce_sum_bf16_bf16_kernel(bf16版本，使用bf16 acc)
- [X] block_all_reduce_sum_bf16_f32_kernel(bf16版本，使用fp32 acc)
- [X] block_all_reduce_sum_bf16x8_pack_bf16_kernel(bf16版本，使用bf16 acc, pack)
- [X] block_all_reduce_sum_bf16x8_pack_f32_kernel(bf16版本，使用fp32 acc, pack)
- [X] block_all_reduce_sum_bf16x2_bf16_kernel(bf16向量化版本，使用bf16 acc)
- [X] block_all_reduce_sum_bf16x2_f32_kernel(bf16向量化版本，使用fp32 acc)
- [X] block_all_reduce_sum_fp8_e4m3_f16_kernel(fp8_e4m3版本，使用fp16 acc)
- [X] block_all_reduce_sum_fp8_e5m2_f16_kernel(fp8_e5m2版本，使用fp16 acc)
- [X] block_all_reduce_sum_fp8_e4m3x16_pack_f16_kernel(fp8_e4m3版本，使用fp16 acc, pack)
- [X] block_all_reduce_sum_fp8_e5m2x16_pack_f16_kernel(fp8_e5m2版本，使用fp16 acc, pack)
- [X] block_all_reduce_sum_i8_i32_kernel(i8版本，使用i32 acc)
- [X] block_all_reduce_sum_i8x16_pack_i32_kernel(i8版本，使用i32 acc, pack)
- [X] PyTorch bindings for block reduce **fp32/fp16/bf16/fp8/i8** kernels

所有支持的block all reduce kernel:

```c++
// packed_type, acc_type, th_type, element_type, n_elements_per_pack, out_type
TORCH_BINDING_REDUCE(f32,              f32,  torch::kFloat32,       float,              1,  float)
TORCH_BINDING_REDUCE(f32x4,            f32,  torch::kFloat32,       float,              4,  float)
TORCH_BINDING_REDUCE(f16,              f16,  torch::kHalf,          half,               1,  float)
TORCH_BINDING_REDUCE(f16,              f32,  torch::kHalf,          half,               1,  float)
TORCH_BINDING_REDUCE(f16x2,            f16,  torch::kHalf,          half,               2,  float)
TORCH_BINDING_REDUCE(f16x2,            f32,  torch::kHalf,          half,               2,  float)
TORCH_BINDING_REDUCE(f16x8_pack,       f16,  torch::kHalf,          half,               8,  float)
TORCH_BINDING_REDUCE(f16x8_pack,       f32,  torch::kHalf,          half,               8,  float)
TORCH_BINDING_REDUCE(bf16,             bf16, torch::kBFloat16,      __nv_bfloat16,      1,  float)
TORCH_BINDING_REDUCE(bf16,             f32,  torch::kBFloat16,      __nv_bfloat16,      1,  float)
TORCH_BINDING_REDUCE(bf16x2,           bf16, torch::kBFloat16,      __nv_bfloat16,      2,  float)
TORCH_BINDING_REDUCE(bf16x2,           f32,  torch::kBFloat16,      __nv_bfloat16,      2,  float)
TORCH_BINDING_REDUCE(bf16x8_pack,      bf16, torch::kBFloat16,      __nv_bfloat16,      8,  float)
TORCH_BINDING_REDUCE(bf16x8_pack,      f32,  torch::kBFloat16,      __nv_bfloat16,      8,  float)
TORCH_BINDING_REDUCE(fp8_e4m3,         f16,  torch::kFloat8_e4m3fn, __nv_fp8_storage_t, 1,  float)
TORCH_BINDING_REDUCE(fp8_e4m3x16_pack, f16,  torch::kFloat8_e4m3fn, __nv_fp8_storage_t, 16, float)
TORCH_BINDING_REDUCE(fp8_e5m2,         f16,  torch::kFloat8_e5m2,   __nv_fp8_storage_t, 1,  float)
TORCH_BINDING_REDUCE(fp8_e5m2x16_pack, f16,  torch::kFloat8_e5m2,   __nv_fp8_storage_t, 16, float)
TORCH_BINDING_REDUCE(i8,               i32,  torch::kInt8,          int8_t,             1,  int32_t)
TORCH_BINDING_REDUCE(i8x16_pack,       i32,  torch::kInt8,          int8_t,             16, int32_t)
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
               out_f32f32: 3560.86938477  , time:0.05275035ms
             out_f32x4f32: 3560.86572266  , time:0.01305699ms
            out_f32f32_th: 3560.86328125  , time:0.01432800ms
--------------------------------------------------------------------------------
               out_f16f16: 3560.96826172  , time:0.05209661ms
               out_f16f32: 3559.97070312  , time:0.05207825ms
             out_f16x2f32: 3560.90795898  , time:0.02845407ms
             out_f16x2f16: 3559.59863281  , time:0.02829432ms
         out_f16x8packf16: 3559.52539062  , time:0.01028347ms
         out_f16x8packf32: 3559.96630859  , time:0.01034307ms
            out_f16f16_th: 3560.00000000  , time:0.01241636ms
--------------------------------------------------------------------------------
             out_bf16bf16: 3534.50000000  , time:0.05398655ms
              out_bf16f32: 3562.39233398  , time:0.05214715ms
            out_bf16x2f32: 3566.61621094  , time:0.02825093ms
           out_bf16x2bf16: 3564.35937500  , time:0.02951026ms
        out_bf16x8packf32: 3555.31005859  , time:0.01030946ms
       out_bf16x8packbf16: 3556.28125000  , time:0.01031280ms
          out_bf16bf16_th: 3568.00000000  , time:0.01231003ms
--------------------------------------------------------------------------------
            out_f8e4m3f16: 3596.70507812  , time:0.05316186ms
     out_f8e4m3x16packf16: 3595.40234375  , time:0.01038933ms
         out_f8e4m3f16_th: 3596.00000000  , time:0.01244307ms
--------------------------------------------------------------------------------
            out_f8e5m2f16: 3647.05859375  , time:0.05315685ms
     out_f8e5m2x16packf16: 3647.43554688  , time:0.01037335ms
         out_f8e5m2f16_th: 3646.00000000  , time:0.01245165ms
--------------------------------------------------------------------------------
                out_i8i32: 3476           , time:0.05345321ms
         out_i8x16packi32: 3476           , time:0.01041198ms
             out_i8i32_th: 3476           , time:0.05220509ms
--------------------------------------------------------------------------------
```
