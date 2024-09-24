# Dot Product

## 0x00 说明

包含以下内容：

- [X] dot_prod_f32_f32_kernel
- [X] dot_prod_f32x4_f32_kernel(float4向量化版本)
- [X] dot_prod_f16_f32_kernel(fp16版本，使用fp32 acc)
- [X] dot_prod_f16x2_f32_kernel(fp16向量化版本，使用fp32 acc)
- [X] dot_prod_f16x8_pack_f32_kernel(fp16向量化版本，使用fp32 acc, pack)
- [X] PyTorch bindings

## 测试

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长
export TORCH_CUDA_ARCH_LIST=Ada 
python3 dot_product.py
```

输出:

```bash
--------------------------------------------------------------------------------
       out_f32f32: -1534.59301758 , time:0.17350578ms
     out_f32x4f32: -1534.61364746 , time:0.18058038ms
    out_f32f32_th: -1534.61157227 , time:0.18307972ms
--------------------------------------------------------------------------------
       out_f16f32: -1538.26318359 , time:0.10106802ms
     out_f16x2f32: -1537.58288574 , time:0.05217433ms
 out_f16x8packf32: -1536.44006348 , time:0.02096844ms
    out_f16f16_th: -1536.00000000 , time:0.02491832ms
--------------------------------------------------------------------------------
```
