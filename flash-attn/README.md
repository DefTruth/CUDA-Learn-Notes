# FlashAttention

## 0x00 说明

包含以下内容：

- [X] flash_attn_1_fwd_f32_kernel 
- [ ] flash_attn_2_fwd_f32_kernel
- [ ] flash_attn_2_fwd_f16_kernel
- [x] flash_attn_2_fwd_f16_mma_m16n8k16_kernel
- [X] PyTorch bindings

### 运行测试   
```bash
python3 flash_attn.py
```
日志如下：
```bash
--------------------------------------------------------------------------------
    out_fa1fwdf32: [0.11064263, 0.08648866, -0.07250906], time:2.32403278ms
out_fa1fwdf32(v2): [0.11064263, 0.08648866, -0.07250906], time:2.22899675ms
   out_attnf32_th: [0.11064263, 0.08648865, -0.07250906], time:0.11474848ms
--------------------------------------------------------------------------------
```
