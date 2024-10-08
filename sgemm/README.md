# SGEMM 

## 0x00 说明

包含以下内容：

- [X] sgemm_naive_f32_kernel (naive)
- [X] sgemm_sliced_k_f32_kernel (sliced_k with smem)
- [X] sgemm_t_8x8_sliced_k_f32x4_kernel (thread tile 8x8)
- [X] sgemm_t_8x8_sliced_k_f32x4_bcf_kernel (bank conflicts free)
- [X] sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel (bank conflicts free, double buffers)
- [X] sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_async_kernel (double buffers, naive copy async)
- [X] PyTorch bindings

## 共享内存 Bank Conflicts

含义：在访问shared memory时，因多个线程读写同一个Bank中的不同数据地址时，导致shared memory 并发读写 退化 成顺序读写的现象叫做Bank Conflict；

![](https://github.com/PaddleJitLab/CUDATutorial/blob/develop/docs/09_optimize_reduce/02_bank_conflict/images/ef322be7c3e5b6b9be69d2b90e88083f50569a58a97129f348e483b946ab4edf.png)

SM调度单位为一个warp（一个warp内32个Thread），shared_memory 可以 被一个warp中的所有（32个）线程进行访问，shared_memory 映射到大小相等的32个Bank上，Bank的数据读取带宽为32bit / cycle (4 bytes)，因此，主要需要考虑一个Warp内32线程的访问共享内存时的bank冲突。
对于多个线程读取同一个Bank数据时（不同地址），硬件把内存读写请求，拆分成 conflict-free requests，进行顺序读写，此时将会触发多次内存事务。特别地，当一个warp中的所有线程读写同一个地址时，会触发broadcast机制，此时不会退化成顺序读写。上面提到触发broadcast机制的条件是all threads acess same address，但在翻阅cuda-c-programming-guide以及最新版本的[NVProfGuide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) 时，发现只要是多个thread 读写就会触发broadcast（不需要All）。
  
- 多个线程读同一个数据时，仅有一个线程读，然后broadcast到其他线程
- 多个线程写同一个数据时，仅会有一个线程写成功

NVIDIA的[文章](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)中指出，我们还可以通过 `cudaDeviceSetSharedMemConfig()` 函数设置默认Bank Size（默认为4 bytes）来避免bank conflicts，可设置为cudaSharedMemBankSizeFourByte或者cudaSharedMemBankSizeEightByte。对于某些场景来说，设置cudaSharedMemBankSizeEightByte或许更加合适，比如使用double数据类型时。 

```C
cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
```

## 双缓冲 Double Buffers

本仓库实现的SGEMM Double Buffers策略如下：1）主循环从bk = 1 开始，第一次数据加载在主循环之前，最后一次计算在主循环之后，这是pipeline 的特点决定的；2）由于计算和下一次访存使用的Shared Memory不同，因此主循环中每次循环只需要一次__syncthreads()即可，对比非double buffers版本，总共节省了 ((K + BK - 1) / BK) - 1 次block内的同步操作。比如，bk=1时，FFMA计算使用的是s_a[0]和s_b[0]，因此，和s_a[1]和s_b[1]的加载是没有依赖关系的。FFMA计算，从global内存到s_a[1]和s_b[1]和HFMA计算可以并行。s_a[1]和s_b[1]用于加载下一块BK需要的数据到共享内存；3）由于GPU不能向CPU那样支持乱序执行，主循环中需要先将下一次循环计算需要的Gloabal Memory中的数据load 到寄存器，然后进行本次计算，之后再将load到寄存器中的数据写到Shared Memory，这样在LDG指令向Global Memory做load时，不会影响后续HFMA及其它运算指令的 launch 执行，也就达到了Double Buffers的目的。

```C
  // 1）主循环从bk = 1 开始，第一次数据加载在主循环之前，最后一次计算在主循环之后，这是pipeline 的特点决定的；
  // 2）由于计算和下一次访存使用的Shared Memory不同，因此主循环中每次循环只需要一次__syncthreads()即可
  // 3）由于GPU不能向CPU那样支持乱序执行，主循环中需要先将下一次循环计算需要的Gloabal Memory中的数据load 
  // 到寄存器，然后进行本次计算，之后再将load到寄存器中的数据写到Shared Memory，这样在LDG指令向Global 
  // Memory做load时，不会影响后续FFMA及其它运算指令的 launch 执行，也就达到了Double Buffering的目的。
  
  // bk = 0 is loading here, buffer 0

  {
    int load_a_gmem_k = load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
    FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

    s_a[0][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
    s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
  }
  // Without this synchronization, accuracy may occasionally be abnormal.
  __syncthreads(); 

  // bk start from 1，需要注意的是，虽然 bk 从 1 开始，但实际上 bk=1时，使用的是
  // 第0块BK中的数据（已经加载到共享内存s_a[0]和s_b[0]）；bk=2时，实际计算的是第1块
  // BK中的数据。其余以此类推，这个循环结束后，剩下最后一块BK大小的数据需要计算。
  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {

    int smem_sel = (bk - 1) & 1;
    int smem_sel_next = bk & 1;

    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
    FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2     ]);
      FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
      FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2     ]);
      FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

      #pragma unroll
      for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
          r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }
    
    // 对比非double buffers版本，此处不需要__syncthreads()，总共节省了
    // ((K + BK - 1) / BK) - 1 次block内的同步操作。比如，bk=1时，HFMA计算
    // 使用的是s_a[0]和s_b[0]，因此，和s_a[1]和s_b[1]的加载是没有依赖关系的。
    // 从global内存到s_a[1]和s_b[1]和HFMA计算可以并行。s_a[1]和s_b[1]用于
    // 加载下一块BK需要的数据到共享内存。
    s_a[smem_sel_next][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
    s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

    __syncthreads();
  }
  
  // 计算剩下最后一块BK
  #pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2     ]);
    FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
    FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2     ]);
    FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][tx * TN / 2 + BN / 2]);

    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
      #pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
        r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }
```

## 参考文献 

- [CUDA编程概念】一、什么是bank conflict？](https://zhuanlan.zhihu.com/p/659142274)
- [解决 bank conflict](https://github.com/PaddleJitLab/CUDATutorial/blob/develop/docs/09_optimize_reduce/02_bank_conflict/README.md)
- [Bank Conflict free 的几种方式](https://zhuanlan.zhihu.com/p/722286440)
- [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [CUDA（三）：通用矩阵乘法：从入门到熟练](https://zhuanlan.zhihu.com/p/657632577)
  
## 测试

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长: Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada 
python3 sgemm.py
```
输出:

```bash
--------------------------------------------------------------------------------------------------------------
                                             M=2048, N=2048, K=1024
                         out_f32: ['-48.24956894', '-25.17781448', '-33.17438889'], time:2.583649ms
                     out_f32(sk): ['-48.24956894', '-25.17781448', '-33.17438889'], time:1.838329ms
               out_f32x4(t8x8sk): ['-48.24956894', '-25.17781448', '-33.17438889'], time:0.324585ms
              out_f32x4(t8x8bcf): ['-48.24956894', '-25.17781448', '-33.17438889'], time:0.288281ms
       out_f32x4(t8x8bcf+offset): ['-48.24956894', '-25.17781448', '-33.17438889'], time:0.287428ms
             out_f32x4(t8x8dbuf): ['-48.24956894', '-25.17781448', '-33.17438889'], time:0.226986ms
      out_f32x4(t8x8dbuf+offset): ['-48.24956894', '-25.17781448', '-33.17438889'], time:0.228636ms
----------------------------------------------------Async-----------------------------------------------------
       out_f32x4(t8x8dbuf+async): ['-48.24956894', '-25.17781448', '-33.17438889'], time:0.296133ms
                      out_f32_th: ['-48.24959564', '-25.17781067', '-33.17438507'], time:0.251844ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=2048, N=2048, K=2048
                         out_f32: ['-29.81858635', '-80.56855011', '58.73820114 '], time:5.148268ms
                     out_f32(sk): ['-29.81858635', '-80.56855011', '58.73820114 '], time:3.652341ms
               out_f32x4(t8x8sk): ['-29.81858635', '-80.56855011', '58.73820114 '], time:0.637898ms
              out_f32x4(t8x8bcf): ['-29.81858635', '-80.56855011', '58.73820114 '], time:0.572691ms
       out_f32x4(t8x8bcf+offset): ['-29.81858635', '-80.56855011', '58.73820114 '], time:0.571003ms
             out_f32x4(t8x8dbuf): ['-29.81858635', '-80.56855011', '58.73820114 '], time:0.452697ms
      out_f32x4(t8x8dbuf+offset): ['-29.81858635', '-80.56855011', '58.73820114 '], time:0.461376ms
----------------------------------------------------Async-----------------------------------------------------
       out_f32x4(t8x8dbuf+async): ['-29.81858635', '-80.56855011', '58.73820114 '], time:0.588417ms
                      out_f32_th: ['-29.81859779', '-80.56848145', '58.73811722 '], time:0.461845ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=2048, N=4096, K=1024
                         out_f32: ['2.2462709   ', '-11.99490261', '-4.03884935 '], time:5.121467ms
                     out_f32(sk): ['2.2462709   ', '-11.99490261', '-4.03884935 '], time:3.653898ms
               out_f32x4(t8x8sk): ['2.2462709   ', '-11.99490261', '-4.03884935 '], time:0.620434ms
              out_f32x4(t8x8bcf): ['2.2462709   ', '-11.99490261', '-4.03884935 '], time:0.525150ms
       out_f32x4(t8x8bcf+offset): ['2.2462709   ', '-11.99490261', '-4.03884935 '], time:0.523758ms
             out_f32x4(t8x8dbuf): ['2.2462709   ', '-11.99490261', '-4.03884935 '], time:0.459876ms
      out_f32x4(t8x8dbuf+offset): ['2.2462709   ', '-11.99490261', '-4.03884935 '], time:0.452015ms
----------------------------------------------------Async-----------------------------------------------------
       out_f32x4(t8x8dbuf+async): ['2.2462709   ', '-11.99490261', '-4.03884935 '], time:0.575559ms
                      out_f32_th: ['2.2462709   ', '-11.99490261', '-4.03884935 '], time:0.527534ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=2048, N=4096, K=2048
                         out_f32: ['-48.77085114', '-7.00706053 ', '35.51366806 '], time:10.209405ms
                     out_f32(sk): ['-48.77085114', '-7.00706053 ', '35.51366806 '], time:7.271879ms
               out_f32x4(t8x8sk): ['-48.77085114', '-7.00706053 ', '35.51366806 '], time:1.220059ms
              out_f32x4(t8x8bcf): ['-48.77085114', '-7.00706053 ', '35.51366806 '], time:1.057007ms
       out_f32x4(t8x8bcf+offset): ['-48.77085114', '-7.00706053 ', '35.51366806 '], time:1.050584ms
             out_f32x4(t8x8dbuf): ['-48.77085114', '-7.00706053 ', '35.51366806 '], time:0.916047ms
      out_f32x4(t8x8dbuf+offset): ['-48.77085114', '-7.00706053 ', '35.51366806 '], time:0.928414ms
----------------------------------------------------Async-----------------------------------------------------
       out_f32x4(t8x8dbuf+async): ['-48.77085114', '-7.00706053 ', '35.51366806 '], time:1.154635ms
                      out_f32_th: ['-48.77086639', '-7.0071063  ', '35.51367188 '], time:0.912690ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=4096, N=2048, K=1024
                         out_f32: ['51.90616226 ', '-92.83547974', '35.52999878 '], time:5.121930ms
                     out_f32(sk): ['51.90616226 ', '-92.83547974', '35.52999878 '], time:3.653657ms
               out_f32x4(t8x8sk): ['51.90616226 ', '-92.83547974', '35.52999878 '], time:0.620129ms
              out_f32x4(t8x8bcf): ['51.90616226 ', '-92.83547974', '35.52999878 '], time:0.525870ms
       out_f32x4(t8x8bcf+offset): ['51.90616226 ', '-92.83547974', '35.52999878 '], time:0.524361ms
             out_f32x4(t8x8dbuf): ['51.90616226 ', '-92.83547974', '35.52999878 '], time:0.448403ms
      out_f32x4(t8x8dbuf+offset): ['51.90616226 ', '-92.83547974', '35.52999878 '], time:0.450542ms
----------------------------------------------------Async-----------------------------------------------------
       out_f32x4(t8x8dbuf+async): ['51.90616226 ', '-92.83547974', '35.52999878 '], time:0.575702ms
                      out_f32_th: ['51.90616226 ', '-92.83547974', '35.52999878 '], time:0.519440ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=4096, N=2048, K=2048
                         out_f32: ['-6.06764698 ', '12.45317936 ', '-37.5351944 '], time:10.213797ms
                     out_f32(sk): ['-6.06764698 ', '12.45317936 ', '-37.5351944 '], time:7.270150ms
               out_f32x4(t8x8sk): ['-6.06764698 ', '12.45317936 ', '-37.5351944 '], time:1.222253ms
              out_f32x4(t8x8bcf): ['-6.06764698 ', '12.45317936 ', '-37.5351944 '], time:1.050663ms
       out_f32x4(t8x8bcf+offset): ['-6.06764698 ', '12.45317936 ', '-37.5351944 '], time:1.052284ms
             out_f32x4(t8x8dbuf): ['-6.06764698 ', '12.45317936 ', '-37.5351944 '], time:0.935955ms
      out_f32x4(t8x8dbuf+offset): ['-6.06764698 ', '12.45317936 ', '-37.5351944 '], time:0.925636ms
----------------------------------------------------Async-----------------------------------------------------
       out_f32x4(t8x8dbuf+async): ['-6.06764698 ', '12.45317936 ', '-37.5351944 '], time:1.151929ms
                      out_f32_th: ['-6.0676651  ', '12.45318413 ', '-37.53521729'], time:0.913348ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=4096, N=4096, K=1024
                         out_f32: ['-7.01906157 ', '23.74757385 ', '-15.73774815'], time:10.206554ms
                     out_f32(sk): ['-7.01906157 ', '23.74757385 ', '-15.73774815'], time:7.308741ms
               out_f32x4(t8x8sk): ['-7.01906157 ', '23.74757385 ', '-15.73774815'], time:1.236022ms
              out_f32x4(t8x8bcf): ['-7.01906157 ', '23.74757385 ', '-15.73774815'], time:1.083252ms
       out_f32x4(t8x8bcf+offset): ['-7.01906157 ', '23.74757385 ', '-15.73774815'], time:1.077623ms
             out_f32x4(t8x8dbuf): ['-7.01906157 ', '23.74757385 ', '-15.73774815'], time:0.961998ms
      out_f32x4(t8x8dbuf+offset): ['-7.01906157 ', '23.74757385 ', '-15.73774815'], time:0.993268ms
----------------------------------------------------Async-----------------------------------------------------
       out_f32x4(t8x8dbuf+async): ['-7.01906157 ', '23.74757385 ', '-15.73774815'], time:1.169443ms
                      out_f32_th: ['-7.01906157 ', '23.74757385 ', '-15.73774815'], time:0.897839ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=4096, N=4096, K=2048
                         out_f32: ['74.64017487 ', '45.85220718 ', '-51.95149231'], time:20.366695ms
                     out_f32(sk): ['74.64017487 ', '45.85220718 ', '-51.95149231'], time:14.591529ms
               out_f32x4(t8x8sk): ['74.64017487 ', '45.85220718 ', '-51.95149231'], time:2.502334ms
              out_f32x4(t8x8bcf): ['74.64017487 ', '45.85220718 ', '-51.95149231'], time:2.215445ms
       out_f32x4(t8x8bcf+offset): ['74.64017487 ', '45.85220718 ', '-51.95149231'], time:2.200658ms
             out_f32x4(t8x8dbuf): ['74.64017487 ', '45.85220718 ', '-51.95149231'], time:2.020864ms
      out_f32x4(t8x8dbuf+offset): ['74.64017487 ', '45.85220718 ', '-51.95149231'], time:2.060344ms
----------------------------------------------------Async-----------------------------------------------------
       out_f32x4(t8x8dbuf+async): ['74.64017487 ', '45.85220718 ', '-51.95149231'], time:2.395530ms
                      out_f32_th: ['74.64017487 ', '45.85220718 ', '-51.95149231'], time:1.833019ms
--------------------------------------------------------------------------------------------------------------
```
