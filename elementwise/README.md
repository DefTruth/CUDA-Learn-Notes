# Elementwise

## 0x00 说明

包含以下内容：

- [X] elementwise_add_f32_kernel
- [X] elementwise_add_f32x4_kernel(float4向量化版本)
- [X] elementwise_add_f16_kernel(fp16版本)
- [X] elementwise_add_f16x2_kernel(fp16向量化版本)
- [X] elementwise_add_f16x8_kernel(fp16向量化版本)
- [X] elementwise_add_f16x8_pack_kernel(fp16向量化版本, pack)
- [X] PyTorch bindings


## 测试

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长
export TORCH_CUDA_ARCH_LIST=Ada 
python3 elementwise.py
```

输出:

```bash
--------------------------------------------------------------------------------
           out_f32: [-1.53079593, 0.52963573], time:0.28430200ms
         out_f32x4: [-1.53079593, 0.52963573], time:0.29020834ms
        out_f32_th: [-1.53079593, 0.52963573], time:0.29701710ms
--------------------------------------------------------------------------------
           out_f16: [-1.53027344, 0.52929688], time:0.05925465ms
         out_f16x2: [-1.53027344, 0.52929688], time:0.04892802ms
         out_f16x8: [-1.53027344, 0.52929688], time:0.04291439ms
     out_f16x8pack: [-1.53027344, 0.52929688], time:0.03846574ms
        out_f16_th: [-1.53027344, 0.52929688], time:0.04044223ms
--------------------------------------------------------------------------------
```
