## FlashAttention 测试  

### 前置依赖  
- PyTorch >= 2.2.1  
- CUDA >= 12.2  

### 测试   
```bash
python3 flash_attn.py
```
日志如下：（RTX 3080 Ti）
```bash
python3 flash_attn.py
=== profiling manual attention ===
STAGE:2024-03-24 11:29:47 15422:15422 ActivityProfilerController.cpp:314] Completed Stage: Warm Up
STAGE:2024-03-24 11:29:48 15422:15422 ActivityProfilerController.cpp:320] Completed Stage: Collection
STAGE:2024-03-24 11:29:48 15422:15422 ActivityProfilerController.cpp:324] Completed Stage: Post Processing
-------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                     aten::matmul         0.02%     142.000us        99.79%     837.362ms     418.681ms     111.000us         0.01%     837.407ms     418.704ms             2
                                        aten::bmm        97.58%     818.847ms        99.75%     837.018ms     418.509ms     837.055ms        99.93%     837.055ms     418.527ms             2
                                     aten::expand         0.00%      36.000us         0.01%      56.000us      14.000us      65.000us         0.01%     124.000us      31.000us             4
                                    aten::reshape         0.00%      40.000us         0.01%      76.000us      19.000us      95.000us         0.01%     115.000us      28.750us             4
                                  aten::transpose         0.01%      48.000us         0.01%      63.000us      63.000us      78.000us         0.01%      95.000us      95.000us             1
                                        aten::mul         0.01%      51.000us         0.01%      63.000us      63.000us      79.000us         0.01%      79.000us      79.000us             1
                                 aten::as_strided         0.00%       9.000us         0.00%       9.000us       1.800us      76.000us         0.01%      76.000us      15.200us             5
                                    aten::softmax         0.00%      11.000us         0.01%      43.000us      43.000us      12.000us         0.00%      58.000us      58.000us             1
                                   aten::_softmax         0.00%      16.000us         0.00%      26.000us      26.000us      46.000us         0.01%      46.000us      46.000us             1
                                       aten::view         0.00%      15.000us         0.00%      15.000us       5.000us      19.000us         0.00%      19.000us       6.333us             3
-------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 839.157ms
Self CUDA time total: 837.639ms

=== profiling flash_attn_1_fwd_f32 attention ===
STAGE:2024-03-24 11:29:48 15422:15422 ActivityProfilerController.cpp:314] Completed Stage: Warm Up
Max shared memory: 49152, requested shared memory: 28672 \nSTAGE:2024-03-24 11:29:48 15422:15422 ActivityProfilerController.cpp:320] Completed Stage: Collection
STAGE:2024-03-24 11:29:48 15422:15422 ActivityProfilerController.cpp:324] Completed Stage: Post Processing
--------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
--------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                  aten::to         0.47%      16.000us        21.73%     743.000us     371.500us     101.000us        11.50%     748.000us     374.000us             2
            aten::_to_copy         1.20%      41.000us        20.97%     717.000us     358.500us      30.000us         3.42%     647.000us     323.500us             2
       aten::empty_strided         1.52%      52.000us        12.81%     438.000us     146.000us     494.000us        56.26%     494.000us     164.667us             3
               aten::copy_         0.53%      18.000us         6.79%     232.000us     116.000us     125.000us        14.24%     125.000us      62.500us             2
               aten::zeros         0.38%      13.000us         1.20%      41.000us      41.000us      57.000us         6.49%      61.000us      61.000us             1
                aten::full         0.44%      15.000us         1.08%      37.000us      37.000us      20.000us         2.28%      39.000us      39.000us             1
          aten::zeros_like         0.58%      20.000us         4.24%     145.000us     145.000us      10.000us         1.14%      30.000us      30.000us             1
               aten::fill_         0.88%      30.000us         1.52%      52.000us      26.000us      21.000us         2.39%      21.000us      10.500us             2
               aten::zero_         0.88%      30.000us         2.25%      77.000us      38.500us       6.000us         0.68%      16.000us       8.000us             2
               aten::empty         0.23%       8.000us         0.23%       8.000us       4.000us      10.000us         1.14%      10.000us       5.000us             2
--------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 3.419ms
Self CUDA time total: 878.000us

attn values sanity check: True
```