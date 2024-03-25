## FlashAttention 测试  

### 前置依赖  
- PyTorch >= 2.2.1  
- CUDA >= 12.2

```bash
python3 -m pip install torch
```

### 运行测试   
```bash
python3 flash_attn.py
```
日志如下：（RTX 3080 Ti）
```bash
python3 flash_attn.py
=== profiling manual attention ===
STAGE:2024-03-25 08:47:18 3818250:3818250 ActivityProfilerController.cpp:314] Completed Stage: Warm Up
STAGE:2024-03-25 08:47:18 3818250:3818250 ActivityProfilerController.cpp:320] Completed Stage: Collection
STAGE:2024-03-25 08:47:18 3818250:3818250 ActivityProfilerController.cpp:324] Completed Stage: Post Processing
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                     Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  Total KFLOPs
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
              manual_attn        45.60%     513.000us        98.31%       1.106ms       1.106ms     489.000us        43.82%       1.116ms       1.116ms             1            --
             aten::matmul        14.31%     161.000us        40.27%     453.000us     226.500us     131.000us        11.74%     496.000us     248.000us             2            --
                aten::bmm         5.78%      65.000us         7.82%      88.000us      44.000us     166.000us        14.87%     166.000us      83.000us             2    201326.592
            aten::reshape         4.98%      56.000us         7.38%      83.000us      20.750us      74.000us         6.63%     105.000us      26.250us             4            --
             aten::expand         4.62%      52.000us         6.13%      69.000us      17.250us      65.000us         5.82%      90.000us      22.500us             4            --
          aten::transpose         3.47%      39.000us         4.27%      48.000us      48.000us      44.000us         3.94%      54.000us      54.000us             1            --
            aten::softmax         1.16%      13.000us         3.47%      39.000us      39.000us      17.000us         1.52%      44.000us      44.000us             1            --
         aten::as_strided         0.53%       6.000us         0.53%       6.000us       1.200us      35.000us         3.14%      35.000us       7.000us             5            --
                aten::mul         2.22%      25.000us         2.84%      32.000us      32.000us      33.000us         2.96%      33.000us      33.000us             1       786.432
           aten::_softmax         1.42%      16.000us         1.96%      22.000us      22.000us      27.000us         2.42%      27.000us      27.000us             1            --
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 1.125ms
Self CUDA time total: 1.116ms

=== profiling flash_attn_1_fwd_f32 attention ===
STAGE:2024-03-25 08:47:18 3818250:3818250 ActivityProfilerController.cpp:314] Completed Stage: Warm Up
STAGE:2024-03-25 08:47:18 3818250:3818250 ActivityProfilerController.cpp:320] Completed Stage: Collection
STAGE:2024-03-25 08:47:18 3818250:3818250 ActivityProfilerController.cpp:324] Completed Stage: Post Processing
--------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
--------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
      flash_attn_1_fwd_f32         5.76%     148.000us        15.72%     404.000us     404.000us       1.804ms        96.37%       1.872ms       1.872ms             1
          aten::zeros_like         1.21%      31.000us         5.21%     134.000us     134.000us       8.000us         0.43%      31.000us      31.000us             1
               aten::zero_         1.44%      37.000us         2.96%      76.000us      38.000us      11.000us         0.59%      25.000us      12.500us             2
               aten::zeros         0.78%      20.000us         2.41%      62.000us      62.000us       8.000us         0.43%      21.000us      21.000us             1
               aten::fill_         0.89%      23.000us         1.60%      41.000us      13.667us      19.000us         1.01%      19.000us       6.333us             3
                aten::full         0.74%      19.000us         1.71%      44.000us      44.000us       9.000us         0.48%      16.000us      16.000us             1
          aten::empty_like         1.01%      26.000us         1.71%      44.000us      44.000us       6.000us         0.32%       8.000us       8.000us             1
               aten::empty         0.62%      16.000us         0.62%      16.000us       8.000us       5.000us         0.27%       5.000us       2.500us             2
       aten::empty_strided         0.54%      14.000us         0.54%      14.000us      14.000us       2.000us         0.11%       2.000us       2.000us             1
           cudaEventRecord         2.18%      56.000us         2.18%      56.000us       2.154us       0.000us         0.00%       0.000us       0.000us            26
--------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 2.570ms
Self CUDA time total: 1.872ms

attn values sanity check: True
```
