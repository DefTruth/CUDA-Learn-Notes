# SGEMM 

## 0x00 说明

包含以下内容：

- [X] sgemm_naive_f32_kernel (naive)
- [X] sgemm_sliced_k_f32_kernel (sliced_k with smem)
- [X] sgemm_t_8x8_sliced_k_f32x4_kernel (thread tile 8x8)
- [X] sgemm_t_8x8_sliced_k_f32x4_bcf_kernel (bank conflicts free)
- [X] sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel (bank conflicts free, double buffers)
- [X] sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_kernel (double buffers, k16)
- [X] sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async_kernel (double buffers, k16, copy async)
- [X] sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages (WMMA, Tile MMA/Warp, Copy Async, Stage, Pad, Block swizzle)
- [X] PyTorch bindings

## 目前性能
目前在L20上，CUDA Cores FP32(L20 FP32/TF32理论算力为59.8 TFLOPS) 的实现能达到cuBLAS大概85%~90%左右的性能(TFLOPS)，部分size下会超过cuBLAS。已知问题为bank conflicts没有完全消除，目前通过padding的方式缓解bank conflicts会导致shared memory浪费，也会影响SM occupancy。而Tensor Cores TF32的实现，只能达到cuBLAS TF32大概80%左右的性能，尚有较大差距。目前未手工实现Warp swizzle(受限于WMMA API的灵活性以及本人的能力)，后续将会尝试通过MMA PTX实现warp swizzle。另外，当前TF32的实现依赖额外的FP32转TF32的kernel，对整体性能有影响。

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
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=2048
                  out_f32x4(t8x8sk): ['70.6019897', '26.1625347'], time:2.428984ms, swizzle: NOOP, TFLOPS: 28.29 (+0.00%)
                 out_f32x4(t8x8bcf): ['70.6019897', '26.1625347'], time:2.112817ms, swizzle: NOOP, TFLOPS: 32.53 (+14.96%)
                out_f32x4(t8x8dbuf): ['70.6019897', '26.1625347'], time:1.877713ms, swizzle: NOOP, TFLOPS: 36.60 (+12.52%)
                    out_f32(cublas): ['70.6019897', '26.1625347'], time:2.229022ms, swizzle: NOOP, TFLOPS: 30.83
                         out_f32_th: ['70.6019897', '26.1625347'], time:1.778435ms, swizzle: NOOP, TFLOPS: 38.64 (+5.58%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['70.5943985', '26.1725273'], time:2.035927ms, swizzle: NOOP, TFLOPS: 33.75
    out_tf32(mma2x4+warp2x4+stage2): ['70.5943985', '26.1725273'], time:1.670312ms, swizzle: NOOP, TFLOPS: 41.14 (+6.47%)
  out_tf32(mma2x4+...+stage3+dsmem): ['70.5943985', '26.1725273'], time:1.820373ms, swizzle: NOOP, TFLOPS: 37.75
  out_tf32(mma2x4+...+stage2+dsmem): ['70.5943985', '26.1725273'], time:1.646137ms, swizzle: NOOP, TFLOPS: 41.75 (+1.47%)
out_tf32(mma2x4+...+stage3+swizzle): ['70.5943985', '26.1725273'], time:2.027678ms, swizzle: 512 , TFLOPS: 33.89
out_tf32(mma2x4+...+stage2+swizzle): ['70.5943985', '26.1725273'], time:1.640319ms, swizzle: 512 , TFLOPS: 41.89 (+0.35%)
 out_tf32(...+stage3+dsmem+swizzle): ['70.5943985', '26.1725273'], time:1.807355ms, swizzle: 512 , TFLOPS: 38.02
 out_tf32(...+stage2+dsmem+swizzle): ['70.5943985', '26.1725273'], time:1.627850ms, swizzle: 512 , TFLOPS: 42.21 (+0.77%)
              out_tf32(cublas+tf32): ['70.5943985', '26.1725273'], time:7.086372ms, swizzle: NOOP, TFLOPS: 9.70
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=4096
                  out_f32x4(t8x8sk): ['151.780014', '4.5990448 '], time:4.822254ms, swizzle: NOOP, TFLOPS: 28.50 (+0.00%)
                 out_f32x4(t8x8bcf): ['151.780014', '4.5990448 '], time:4.319739ms, swizzle: NOOP, TFLOPS: 31.82 (+11.63%)
                out_f32x4(t8x8dbuf): ['151.780014', '4.5990448 '], time:3.906702ms, swizzle: NOOP, TFLOPS: 35.18 (+10.57%)
                    out_f32(cublas): ['151.780014', '4.5990448 '], time:4.850530ms, swizzle: NOOP, TFLOPS: 28.33
                         out_f32_th: ['151.780014', '4.5990448 '], time:3.584909ms, swizzle: NOOP, TFLOPS: 38.34 (+8.98%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['151.794143', '4.5965395 '], time:4.346919ms, swizzle: NOOP, TFLOPS: 31.62
    out_tf32(mma2x4+warp2x4+stage2): ['151.794143', '4.5965395 '], time:3.493309ms, swizzle: NOOP, TFLOPS: 39.34 (+2.62%)
  out_tf32(mma2x4+...+stage3+dsmem): ['151.794143', '4.5965395 '], time:3.765821ms, swizzle: NOOP, TFLOPS: 36.50
  out_tf32(mma2x4+...+stage2+dsmem): ['151.794143', '4.5965395 '], time:3.599095ms, swizzle: NOOP, TFLOPS: 38.19
out_tf32(mma2x4+...+stage3+swizzle): ['151.794143', '4.5965395 '], time:4.048442ms, swizzle: 512 , TFLOPS: 33.95
out_tf32(mma2x4+...+stage2+swizzle): ['151.794143', '4.5965395 '], time:3.320336ms, swizzle: 512 , TFLOPS: 41.39 (+5.21%)
 out_tf32(...+stage3+dsmem+swizzle): ['151.794143', '4.5965395 '], time:3.658032ms, swizzle: 512 , TFLOPS: 37.57
 out_tf32(...+stage2+dsmem+swizzle): ['151.794143', '4.5965395 '], time:3.310155ms, swizzle: 512 , TFLOPS: 41.52 (+0.31%)
              out_tf32(cublas+tf32): ['151.794143', '4.5965395 '], time:2.807903ms, swizzle: NOOP, TFLOPS: 48.95 (+17.89%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=8192
                  out_f32x4(t8x8sk): ['118.496635', '44.2837791'], time:9.974384ms, swizzle: NOOP, TFLOPS: 27.56 (+0.00%)
                 out_f32x4(t8x8bcf): ['118.496635', '44.2837791'], time:8.764767ms, swizzle: NOOP, TFLOPS: 31.36 (+13.80%)
                out_f32x4(t8x8dbuf): ['118.496635', '44.2837791'], time:8.941769ms, swizzle: NOOP, TFLOPS: 30.74
                    out_f32(cublas): ['118.496635', '44.2837791'], time:7.849812ms, swizzle: NOOP, TFLOPS: 35.02 (+11.66%)
                         out_f32_th: ['118.496635', '44.2837791'], time:7.393693ms, swizzle: NOOP, TFLOPS: 37.18 (+6.17%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['118.526184', '44.2716636'], time:8.627605ms, swizzle: NOOP, TFLOPS: 31.86
    out_tf32(mma2x4+warp2x4+stage2): ['118.526184', '44.2716636'], time:6.934285ms, swizzle: NOOP, TFLOPS: 39.64 (+6.63%)
  out_tf32(mma2x4+...+stage3+dsmem): ['118.526184', '44.2716636'], time:7.462024ms, swizzle: NOOP, TFLOPS: 36.84
  out_tf32(mma2x4+...+stage2+dsmem): ['118.526184', '44.2716636'], time:6.970906ms, swizzle: NOOP, TFLOPS: 39.43
out_tf32(mma2x4+...+stage3+swizzle): ['118.526184', '44.2716636'], time:8.261394ms, swizzle: 512 , TFLOPS: 33.27
out_tf32(mma2x4+...+stage2+swizzle): ['118.526184', '44.2716636'], time:6.864094ms, swizzle: 512 , TFLOPS: 40.05 (+1.02%)
 out_tf32(...+stage3+dsmem+swizzle): ['118.526184', '44.2716636'], time:7.449316ms, swizzle: 512 , TFLOPS: 36.90
 out_tf32(...+stage2+dsmem+swizzle): ['118.526184', '44.2716636'], time:6.867933ms, swizzle: 512 , TFLOPS: 40.02
              out_tf32(cublas+tf32): ['118.526184', '44.2716636'], time:5.459380ms, swizzle: NOOP, TFLOPS: 50.35 (+25.73%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=2048
                  out_f32x4(t8x8sk): ['70.5972366', '26.1622695'], time:4.638457ms, swizzle: NOOP, TFLOPS: 29.63 (+0.00%)
                 out_f32x4(t8x8bcf): ['70.5972366', '26.1622695'], time:4.083228ms, swizzle: NOOP, TFLOPS: 33.66 (+13.60%)
                out_f32x4(t8x8dbuf): ['70.5972366', '26.1622695'], time:3.705859ms, swizzle: NOOP, TFLOPS: 37.09 (+10.18%)
                    out_f32(cublas): ['70.5972366', '26.1622695'], time:4.071259ms, swizzle: NOOP, TFLOPS: 33.76
                         out_f32_th: ['70.5972366', '26.1622695'], time:3.648686ms, swizzle: NOOP, TFLOPS: 37.67 (+1.57%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['70.5943985', '26.1725273'], time:3.987336ms, swizzle: NOOP, TFLOPS: 34.47
    out_tf32(mma2x4+warp2x4+stage2): ['70.5943985', '26.1725273'], time:3.204703ms, swizzle: NOOP, TFLOPS: 42.89 (+13.85%)
  out_tf32(mma2x4+...+stage3+dsmem): ['70.5943985', '26.1725273'], time:3.465056ms, swizzle: NOOP, TFLOPS: 39.66
  out_tf32(mma2x4+...+stage2+dsmem): ['70.5943985', '26.1725273'], time:3.179168ms, swizzle: NOOP, TFLOPS: 43.23 (+0.80%)
out_tf32(mma2x4+...+stage3+swizzle): ['70.5943985', '26.1725273'], time:3.828763ms, swizzle: 1024, TFLOPS: 35.90
out_tf32(mma2x4+...+stage2+swizzle): ['70.5943985', '26.1725273'], time:3.141665ms, swizzle: 1024, TFLOPS: 43.75 (+1.19%)
 out_tf32(...+stage3+dsmem+swizzle): ['70.5943985', '26.1725273'], time:3.441977ms, swizzle: 1024, TFLOPS: 39.93
 out_tf32(...+stage2+dsmem+swizzle): ['70.5943985', '26.1725273'], time:3.152799ms, swizzle: 1024, TFLOPS: 43.59
              out_tf32(cublas+tf32): ['70.5943985', '26.1725273'], time:2.859544ms, swizzle: NOOP, TFLOPS: 48.06 (+9.87%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=4096
                  out_f32x4(t8x8sk): ['151.801406', '4.59161139'], time:9.912538ms, swizzle: NOOP, TFLOPS: 27.73 (+0.00%)
                 out_f32x4(t8x8bcf): ['151.801406', '4.59161139'], time:8.917999ms, swizzle: NOOP, TFLOPS: 30.82 (+11.15%)
                out_f32x4(t8x8dbuf): ['151.801406', '4.59161139'], time:8.958077ms, swizzle: NOOP, TFLOPS: 30.68
                    out_f32(cublas): ['151.801406', '4.59161139'], time:7.909870ms, swizzle: NOOP, TFLOPS: 34.75 (+12.75%)
                         out_f32_th: ['151.801406', '4.59161139'], time:7.236218ms, swizzle: NOOP, TFLOPS: 37.99 (+9.31%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['151.794143', '4.5965395 '], time:7.893776ms, swizzle: NOOP, TFLOPS: 34.82
    out_tf32(mma2x4+warp2x4+stage2): ['151.794143', '4.5965395 '], time:6.559514ms, swizzle: NOOP, TFLOPS: 41.91 (+10.32%)
  out_tf32(mma2x4+...+stage3+dsmem): ['151.794143', '4.5965395 '], time:6.930255ms, swizzle: NOOP, TFLOPS: 39.66
  out_tf32(mma2x4+...+stage2+dsmem): ['151.794143', '4.5965395 '], time:6.577444ms, swizzle: NOOP, TFLOPS: 41.79
out_tf32(mma2x4+...+stage3+swizzle): ['151.794143', '4.5965395 '], time:7.675647ms, swizzle: 1024, TFLOPS: 35.81
out_tf32(mma2x4+...+stage2+swizzle): ['151.794143', '4.5965395 '], time:6.308770ms, swizzle: 1024, TFLOPS: 43.57 (+3.97%)
 out_tf32(...+stage3+dsmem+swizzle): ['151.794143', '4.5965395 '], time:6.884336ms, swizzle: 1024, TFLOPS: 39.93
 out_tf32(...+stage2+dsmem+swizzle): ['151.794143', '4.5965395 '], time:6.305503ms, swizzle: 1024, TFLOPS: 43.59 (+0.05%)
              out_tf32(cublas+tf32): ['151.794143', '4.5965395 '], time:5.328726ms, swizzle: NOOP, TFLOPS: 51.58 (+18.33%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=8192
                  out_f32x4(t8x8sk): ['118.518661', '44.2836265'], time:20.20986ms, swizzle: NOOP, TFLOPS: 27.20 (+0.00%)
                 out_f32x4(t8x8bcf): ['118.518661', '44.2836265'], time:18.03719ms, swizzle: NOOP, TFLOPS: 30.48 (+12.05%)
                out_f32x4(t8x8dbuf): ['118.518661', '44.2836265'], time:18.61379ms, swizzle: NOOP, TFLOPS: 29.53
                    out_f32(cublas): ['118.518661', '44.2836265'], time:15.54746ms, swizzle: NOOP, TFLOPS: 35.36 (+16.01%)
                         out_f32_th: ['118.518661', '44.2836265'], time:15.30375ms, swizzle: NOOP, TFLOPS: 35.92 (+1.59%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['118.526184', '44.2716636'], time:15.66731ms, swizzle: NOOP, TFLOPS: 35.09
    out_tf32(mma2x4+warp2x4+stage2): ['118.526184', '44.2716636'], time:13.19141ms, swizzle: NOOP, TFLOPS: 41.68 (+16.01%)
  out_tf32(mma2x4+...+stage3+dsmem): ['118.526184', '44.2716636'], time:13.83848ms, swizzle: NOOP, TFLOPS: 39.73
  out_tf32(mma2x4+...+stage2+dsmem): ['118.526184', '44.2716636'], time:13.15524ms, swizzle: NOOP, TFLOPS: 41.79 (+0.27%)
out_tf32(mma2x4+...+stage3+swizzle): ['118.526184', '44.2716636'], time:15.49148ms, swizzle: 1024, TFLOPS: 35.49
out_tf32(mma2x4+...+stage2+swizzle): ['118.526184', '44.2716636'], time:12.80868ms, swizzle: 1024, TFLOPS: 42.92 (+2.71%)
 out_tf32(...+stage3+dsmem+swizzle): ['118.526184', '44.2716636'], time:13.90929ms, swizzle: 1024, TFLOPS: 39.52
 out_tf32(...+stage2+dsmem+swizzle): ['118.526184', '44.2716636'], time:12.78388ms, swizzle: 1024, TFLOPS: 43.00 (+0.19%)
              out_tf32(cublas+tf32): ['118.526184', '44.2716636'], time:10.33768ms, swizzle: NOOP, TFLOPS: 53.18 (+23.66%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=2048
                  out_f32x4(t8x8sk): ['70.5972366', '26.1622695'], time:9.941315ms, swizzle: NOOP, TFLOPS: 27.65 (+0.00%)
                 out_f32x4(t8x8bcf): ['70.5972366', '26.1622695'], time:9.267258ms, swizzle: NOOP, TFLOPS: 29.66 (+7.27%)
                out_f32x4(t8x8dbuf): ['70.5972366', '26.1622695'], time:9.232449ms, swizzle: NOOP, TFLOPS: 29.77 (+0.38%)
                    out_f32(cublas): ['70.5972366', '26.1622695'], time:7.846927ms, swizzle: NOOP, TFLOPS: 35.03 (+17.66%)
                         out_f32_th: ['70.5972366', '26.1622695'], time:7.085800ms, swizzle: NOOP, TFLOPS: 38.79 (+10.74%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['70.5943985', '26.1725273'], time:7.701039ms, swizzle: NOOP, TFLOPS: 35.69
    out_tf32(mma2x4+warp2x4+stage2): ['70.5943985', '26.1725273'], time:6.537389ms, swizzle: NOOP, TFLOPS: 42.05 (+8.39%)
  out_tf32(mma2x4+...+stage3+dsmem): ['70.5943985', '26.1725273'], time:6.712508ms, swizzle: NOOP, TFLOPS: 40.95
  out_tf32(mma2x4+...+stage2+dsmem): ['70.5943985', '26.1725273'], time:6.550049ms, swizzle: NOOP, TFLOPS: 41.97
out_tf32(mma2x4+...+stage3+swizzle): ['70.5943985', '26.1725273'], time:7.554650ms, swizzle: 2048, TFLOPS: 36.39
out_tf32(mma2x4+...+stage2+swizzle): ['70.5943985', '26.1725273'], time:6.168079ms, swizzle: 2048, TFLOPS: 44.56 (+5.99%)
 out_tf32(...+stage3+dsmem+swizzle): ['70.5943985', '26.1725273'], time:6.722187ms, swizzle: 2048, TFLOPS: 40.89
 out_tf32(...+stage2+dsmem+swizzle): ['70.5943985', '26.1725273'], time:6.171321ms, swizzle: 2048, TFLOPS: 44.54
              out_tf32(cublas+tf32): ['70.5943985', '26.1725273'], time:5.131006ms, swizzle: NOOP, TFLOPS: 53.57 (+20.21%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=4096
                  out_f32x4(t8x8sk): ['151.799118', '4.6021018 '], time:20.19996ms, swizzle: NOOP, TFLOPS: 27.22 (+0.00%)
                 out_f32x4(t8x8bcf): ['151.799118', '4.6021018 '], time:18.53487ms, swizzle: NOOP, TFLOPS: 29.66 (+8.98%)
                out_f32x4(t8x8dbuf): ['151.799118', '4.6021018 '], time:18.93479ms, swizzle: NOOP, TFLOPS: 29.03
                    out_f32(cublas): ['151.799118', '4.6021018 '], time:14.90321ms, swizzle: NOOP, TFLOPS: 36.89 (+24.37%)
                         out_f32_th: ['151.799118', '4.6021018 '], time:14.38026ms, swizzle: NOOP, TFLOPS: 38.23 (+3.64%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['151.794143', '4.5965395 '], time:15.34090ms, swizzle: NOOP, TFLOPS: 35.84
    out_tf32(mma2x4+warp2x4+stage2): ['151.794143', '4.5965395 '], time:12.95042ms, swizzle: NOOP, TFLOPS: 42.45 (+11.04%)
  out_tf32(mma2x4+...+stage3+dsmem): ['151.794143', '4.5965395 '], time:13.73360ms, swizzle: NOOP, TFLOPS: 40.03
  out_tf32(mma2x4+...+stage2+dsmem): ['151.794143', '4.5965395 '], time:12.93442ms, swizzle: NOOP, TFLOPS: 42.50 (+0.12%)
out_tf32(mma2x4+...+stage3+swizzle): ['151.794143', '4.5965395 '], time:15.03224ms, swizzle: 2048, TFLOPS: 36.57
out_tf32(mma2x4+...+stage2+swizzle): ['151.794143', '4.5965395 '], time:12.34993ms, swizzle: 2048, TFLOPS: 44.51 (+4.73%)
 out_tf32(...+stage3+dsmem+swizzle): ['151.794143', '4.5965395 '], time:13.40029ms, swizzle: 2048, TFLOPS: 41.03
 out_tf32(...+stage2+dsmem+swizzle): ['151.794143', '4.5965395 '], time:12.32724ms, swizzle: 2048, TFLOPS: 44.60 (+0.18%)
              out_tf32(cublas+tf32): ['151.794143', '4.5965395 '], time:9.960341ms, swizzle: NOOP, TFLOPS: 55.19 (+23.76%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=8192
                  out_f32x4(t8x8sk): ['118.513626', '44.2889137'], time:40.22870ms, swizzle: NOOP, TFLOPS: 27.33 (+0.00%)
                 out_f32x4(t8x8bcf): ['118.513626', '44.2889137'], time:39.04280ms, swizzle: NOOP, TFLOPS: 28.16 (+3.04%)
                out_f32x4(t8x8dbuf): ['118.513626', '44.2889137'], time:39.80977ms, swizzle: NOOP, TFLOPS: 27.62
                    out_f32(cublas): ['118.513626', '44.2889137'], time:28.38425ms, swizzle: NOOP, TFLOPS: 38.74 (+37.55%)
                         out_f32_th: ['118.513626', '44.2889137'], time:29.08875ms, swizzle: NOOP, TFLOPS: 37.80
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['118.526184', '44.2716636'], time:30.07037ms, swizzle: NOOP, TFLOPS: 36.56
    out_tf32(mma2x4+warp2x4+stage2): ['118.526184', '44.2716636'], time:26.02388ms, swizzle: NOOP, TFLOPS: 42.25 (+9.07%)
  out_tf32(mma2x4+...+stage3+dsmem): ['118.526184', '44.2716636'], time:27.45041ms, swizzle: NOOP, TFLOPS: 40.05
  out_tf32(mma2x4+...+stage2+dsmem): ['118.526184', '44.2716636'], time:26.32236ms, swizzle: NOOP, TFLOPS: 41.77
out_tf32(mma2x4+...+stage3+swizzle): ['118.526184', '44.2716636'], time:30.09891ms, swizzle: 2048, TFLOPS: 36.53
out_tf32(mma2x4+...+stage2+swizzle): ['118.526184', '44.2716636'], time:24.76131ms, swizzle: 2048, TFLOPS: 44.40 (+5.10%)
 out_tf32(...+stage3+dsmem+swizzle): ['118.526184', '44.2716636'], time:26.82106ms, swizzle: 2048, TFLOPS: 40.99
 out_tf32(...+stage2+dsmem+swizzle): ['118.526184', '44.2716636'], time:24.67982ms, swizzle: 2048, TFLOPS: 44.55 (+0.33%)
              out_tf32(cublas+tf32): ['118.526184', '44.2716636'], time:19.58444ms, swizzle: NOOP, TFLOPS: 56.14 (+26.02%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=2048
                  out_f32x4(t8x8sk): ['70.5949554', '26.1727619'], time:4.644012ms, swizzle: NOOP, TFLOPS: 29.59 (+0.00%)
                 out_f32x4(t8x8bcf): ['70.5949554', '26.1727619'], time:4.165029ms, swizzle: NOOP, TFLOPS: 33.00 (+11.50%)
                out_f32x4(t8x8dbuf): ['70.5949554', '26.1727619'], time:3.532195ms, swizzle: NOOP, TFLOPS: 38.91 (+17.92%)
                    out_f32(cublas): ['70.5949554', '26.1727619'], time:4.056715ms, swizzle: NOOP, TFLOPS: 33.88
                         out_f32_th: ['70.5949554', '26.1727619'], time:3.668260ms, swizzle: NOOP, TFLOPS: 37.47
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['70.5943985', '26.1725273'], time:4.008388ms, swizzle: NOOP, TFLOPS: 34.29
    out_tf32(mma2x4+warp2x4+stage2): ['70.5943985', '26.1725273'], time:3.218698ms, swizzle: NOOP, TFLOPS: 42.70 (+9.74%)
  out_tf32(mma2x4+...+stage3+dsmem): ['70.5943985', '26.1725273'], time:3.489041ms, swizzle: NOOP, TFLOPS: 39.39
  out_tf32(mma2x4+...+stage2+dsmem): ['70.5943985', '26.1725273'], time:3.196096ms, swizzle: NOOP, TFLOPS: 43.00 (+0.71%)
out_tf32(mma2x4+...+stage3+swizzle): ['70.5943985', '26.1725273'], time:3.782248ms, swizzle: 512 , TFLOPS: 36.34
out_tf32(mma2x4+...+stage2+swizzle): ['70.5943985', '26.1725273'], time:3.096580ms, swizzle: 512 , TFLOPS: 44.38 (+3.21%)
 out_tf32(...+stage3+dsmem+swizzle): ['70.5943985', '26.1725273'], time:3.394317ms, swizzle: 512 , TFLOPS: 40.49
 out_tf32(...+stage2+dsmem+swizzle): ['70.5943985', '26.1725273'], time:3.095269ms, swizzle: 512 , TFLOPS: 44.40 (+0.04%)
              out_tf32(cublas+tf32): ['70.5943985', '26.1725273'], time:11.76311ms, swizzle: NOOP, TFLOPS: 11.68
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=4096
                  out_f32x4(t8x8sk): ['151.796371', '4.59689951'], time:9.283566ms, swizzle: NOOP, TFLOPS: 29.61 (+0.00%)
                 out_f32x4(t8x8bcf): ['151.796371', '4.59689951'], time:8.359241ms, swizzle: NOOP, TFLOPS: 32.88 (+11.06%)
                out_f32x4(t8x8dbuf): ['151.796371', '4.59689951'], time:7.493996ms, swizzle: NOOP, TFLOPS: 36.68 (+11.55%)
                    out_f32(cublas): ['151.796371', '4.59689951'], time:7.483124ms, swizzle: NOOP, TFLOPS: 36.73 (+0.15%)
                         out_f32_th: ['151.796371', '4.59689951'], time:7.139444ms, swizzle: NOOP, TFLOPS: 38.50 (+4.81%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['151.794143', '4.5965395 '], time:7.942914ms, swizzle: NOOP, TFLOPS: 34.61
    out_tf32(mma2x4+warp2x4+stage2): ['151.794143', '4.5965395 '], time:6.454420ms, swizzle: NOOP, TFLOPS: 42.59 (+10.61%)
  out_tf32(mma2x4+...+stage3+dsmem): ['151.794143', '4.5965395 '], time:7.018256ms, swizzle: NOOP, TFLOPS: 39.17
  out_tf32(mma2x4+...+stage2+dsmem): ['151.794143', '4.5965395 '], time:6.443977ms, swizzle: NOOP, TFLOPS: 42.66 (+0.16%)
out_tf32(mma2x4+...+stage3+swizzle): ['151.794143', '4.5965395 '], time:7.723641ms, swizzle: 512 , TFLOPS: 35.59
out_tf32(mma2x4+...+stage2+swizzle): ['151.794143', '4.5965395 '], time:6.369042ms, swizzle: 512 , TFLOPS: 43.16 (+1.18%)
 out_tf32(...+stage3+dsmem+swizzle): ['151.794143', '4.5965395 '], time:6.931543ms, swizzle: 512 , TFLOPS: 39.66
 out_tf32(...+stage2+dsmem+swizzle): ['151.794143', '4.5965395 '], time:6.361842ms, swizzle: 512 , TFLOPS: 43.21 (+0.11%)
              out_tf32(cublas+tf32): ['151.794143', '4.5965395 '], time:5.284237ms, swizzle: NOOP, TFLOPS: 52.02 (+20.39%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=8192
                  out_f32x4(t8x8sk): ['118.532104', '44.2729606'], time:19.66500ms, swizzle: NOOP, TFLOPS: 27.96 (+0.00%)
                 out_f32x4(t8x8bcf): ['118.532104', '44.2729606'], time:17.24970ms, swizzle: NOOP, TFLOPS: 31.87 (+14.00%)
                out_f32x4(t8x8dbuf): ['118.532104', '44.2729606'], time:17.30856ms, swizzle: NOOP, TFLOPS: 31.76
                    out_f32(cublas): ['118.532104', '44.2729606'], time:15.01247ms, swizzle: NOOP, TFLOPS: 36.62 (+14.90%)
                         out_f32_th: ['118.532104', '44.2729606'], time:14.77088ms, swizzle: NOOP, TFLOPS: 37.22 (+1.64%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['118.526184', '44.2716636'], time:15.61958ms, swizzle: NOOP, TFLOPS: 35.20
    out_tf32(mma2x4+warp2x4+stage2): ['118.526184', '44.2716636'], time:13.11204ms, swizzle: NOOP, TFLOPS: 41.93 (+12.65%)
  out_tf32(mma2x4+...+stage3+dsmem): ['118.526184', '44.2716636'], time:13.86370ms, swizzle: NOOP, TFLOPS: 39.65
  out_tf32(mma2x4+...+stage2+dsmem): ['118.526184', '44.2716636'], time:13.01887ms, swizzle: NOOP, TFLOPS: 42.23 (+0.72%)
out_tf32(mma2x4+...+stage3+swizzle): ['118.526184', '44.2716636'], time:15.49036ms, swizzle: 512 , TFLOPS: 35.49
out_tf32(mma2x4+...+stage2+swizzle): ['118.526184', '44.2716636'], time:12.93551ms, swizzle: 512 , TFLOPS: 42.50 (+0.64%)
 out_tf32(...+stage3+dsmem+swizzle): ['118.526184', '44.2716636'], time:13.91084ms, swizzle: 512 , TFLOPS: 39.52
 out_tf32(...+stage2+dsmem+swizzle): ['118.526184', '44.2716636'], time:12.87522ms, swizzle: 512 , TFLOPS: 42.70 (+0.47%)
              out_tf32(cublas+tf32): ['118.526184', '44.2716636'], time:10.32779ms, swizzle: NOOP, TFLOPS: 53.23 (+24.67%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=2048
                  out_f32x4(t8x8sk): ['70.5949554', '26.1727619'], time:9.005260ms, swizzle: NOOP, TFLOPS: 30.52 (+0.00%)
                 out_f32x4(t8x8bcf): ['70.5949554', '26.1727619'], time:8.109664ms, swizzle: NOOP, TFLOPS: 33.90 (+11.04%)
                out_f32x4(t8x8dbuf): ['70.5949554', '26.1727619'], time:7.237076ms, swizzle: NOOP, TFLOPS: 37.98 (+12.06%)
                    out_f32(cublas): ['70.5949554', '26.1727619'], time:7.283616ms, swizzle: NOOP, TFLOPS: 37.74
                         out_f32_th: ['70.5949554', '26.1727619'], time:7.025599ms, swizzle: NOOP, TFLOPS: 39.13 (+3.01%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['70.5943985', '26.1725273'], time:7.638692ms, swizzle: NOOP, TFLOPS: 35.98
    out_tf32(mma2x4+warp2x4+stage2): ['70.5943985', '26.1725273'], time:6.153583ms, swizzle: NOOP, TFLOPS: 44.67 (+14.17%)
  out_tf32(mma2x4+...+stage3+dsmem): ['70.5943985', '26.1725273'], time:6.675100ms, swizzle: NOOP, TFLOPS: 41.18
  out_tf32(mma2x4+...+stage2+dsmem): ['70.5943985', '26.1725273'], time:6.140279ms, swizzle: NOOP, TFLOPS: 44.77 (+0.22%)
out_tf32(mma2x4+...+stage3+swizzle): ['70.5943985', '26.1725273'], time:7.350254ms, swizzle: 1024, TFLOPS: 37.40
out_tf32(mma2x4+...+stage2+swizzle): ['70.5943985', '26.1725273'], time:6.009721ms, swizzle: 1024, TFLOPS: 45.74 (+2.17%)
 out_tf32(...+stage3+dsmem+swizzle): ['70.5943985', '26.1725273'], time:6.560659ms, swizzle: 1024, TFLOPS: 41.90
 out_tf32(...+stage2+dsmem+swizzle): ['70.5943985', '26.1725273'], time:6.008577ms, swizzle: 1024, TFLOPS: 45.75 (+0.02%)
              out_tf32(cublas+tf32): ['70.5943985', '26.1725273'], time:5.121445ms, swizzle: NOOP, TFLOPS: 53.67 (+17.32%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=4096
                  out_f32x4(t8x8sk): ['151.796371', '4.59689951'], time:19.40293ms, swizzle: NOOP, TFLOPS: 28.33 (+0.00%)
                 out_f32x4(t8x8bcf): ['151.796371', '4.59689951'], time:17.21770ms, swizzle: NOOP, TFLOPS: 31.93 (+12.69%)
                out_f32x4(t8x8dbuf): ['151.796371', '4.59689951'], time:17.95308ms, swizzle: NOOP, TFLOPS: 30.62
                    out_f32(cublas): ['151.796371', '4.59689951'], time:14.42518ms, swizzle: NOOP, TFLOPS: 38.11 (+19.36%)
                         out_f32_th: ['151.796371', '4.59689951'], time:14.29438ms, swizzle: NOOP, TFLOPS: 38.46 (+0.92%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['151.794143', '4.5965395 '], time:14.90476ms, swizzle: NOOP, TFLOPS: 36.88
    out_tf32(mma2x4+warp2x4+stage2): ['151.794143', '4.5965395 '], time:12.51502ms, swizzle: NOOP, TFLOPS: 43.93 (+14.22%)
  out_tf32(mma2x4+...+stage3+dsmem): ['151.794143', '4.5965395 '], time:13.19789ms, swizzle: NOOP, TFLOPS: 41.65
  out_tf32(mma2x4+...+stage2+dsmem): ['151.794143', '4.5965395 '], time:12.53654ms, swizzle: NOOP, TFLOPS: 43.85
out_tf32(mma2x4+...+stage3+swizzle): ['151.794143', '4.5965395 '], time:14.80431ms, swizzle: 1024, TFLOPS: 37.13
out_tf32(mma2x4+...+stage2+swizzle): ['151.794143', '4.5965395 '], time:12.12592ms, swizzle: 1024, TFLOPS: 45.34 (+3.21%)
 out_tf32(...+stage3+dsmem+swizzle): ['151.794143', '4.5965395 '], time:13.21063ms, swizzle: 1024, TFLOPS: 41.61
 out_tf32(...+stage2+dsmem+swizzle): ['151.794143', '4.5965395 '], time:12.12511ms, swizzle: 1024, TFLOPS: 45.34 (+0.01%)
              out_tf32(cublas+tf32): ['151.794143', '4.5965395 '], time:10.02106ms, swizzle: NOOP, TFLOPS: 54.86 (+21.00%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=8192
                  out_f32x4(t8x8sk): ['118.532104', '44.2729606'], time:39.05200ms, swizzle: NOOP, TFLOPS: 28.16 (+0.00%)
                 out_f32x4(t8x8bcf): ['118.532104', '44.2729606'], time:36.05434ms, swizzle: NOOP, TFLOPS: 30.50 (+8.31%)
                out_f32x4(t8x8dbuf): ['118.532104', '44.2729606'], time:36.42346ms, swizzle: NOOP, TFLOPS: 30.19
                    out_f32(cublas): ['118.532104', '44.2729606'], time:28.22470ms, swizzle: NOOP, TFLOPS: 38.96 (+27.74%)
                         out_f32_th: ['118.532104', '44.2729606'], time:28.45404ms, swizzle: NOOP, TFLOPS: 38.64
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['118.526184', '44.2716636'], time:29.65857ms, swizzle: NOOP, TFLOPS: 37.07
    out_tf32(mma2x4+warp2x4+stage2): ['118.526184', '44.2716636'], time:25.09703ms, swizzle: NOOP, TFLOPS: 43.81 (+12.46%)
  out_tf32(mma2x4+...+stage3+dsmem): ['118.526184', '44.2716636'], time:26.67160ms, swizzle: NOOP, TFLOPS: 41.22
  out_tf32(mma2x4+...+stage2+dsmem): ['118.526184', '44.2716636'], time:25.22740ms, swizzle: NOOP, TFLOPS: 43.58
out_tf32(mma2x4+...+stage3+swizzle): ['118.526184', '44.2716636'], time:29.67340ms, swizzle: 1024, TFLOPS: 37.05
out_tf32(mma2x4+...+stage2+swizzle): ['118.526184', '44.2716636'], time:24.31735ms, swizzle: 1024, TFLOPS: 45.22 (+3.21%)
 out_tf32(...+stage3+dsmem+swizzle): ['118.526184', '44.2716636'], time:26.41408ms, swizzle: 1024, TFLOPS: 41.63
 out_tf32(...+stage2+dsmem+swizzle): ['118.526184', '44.2716636'], time:24.30074ms, swizzle: 1024, TFLOPS: 45.25 (+0.07%)
              out_tf32(cublas+tf32): ['118.526184', '44.2716636'], time:19.56663ms, swizzle: NOOP, TFLOPS: 56.19 (+24.19%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=2048
                  out_f32x4(t8x8sk): ['70.5949554', '26.1727619'], time:19.93403ms, swizzle: NOOP, TFLOPS: 27.58 (+0.00%)
                 out_f32x4(t8x8bcf): ['70.5949554', '26.1727619'], time:17.85275ms, swizzle: NOOP, TFLOPS: 30.79 (+11.66%)
                out_f32x4(t8x8dbuf): ['70.5949554', '26.1727619'], time:17.60568ms, swizzle: NOOP, TFLOPS: 31.23 (+1.40%)
                    out_f32(cublas): ['70.5949554', '26.1727619'], time:14.66460ms, swizzle: NOOP, TFLOPS: 37.49 (+20.06%)
                         out_f32_th: ['70.5949554', '26.1727619'], time:14.66336ms, swizzle: NOOP, TFLOPS: 37.49 (+0.01%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['70.5943985', '26.1725273'], time:14.75033ms, swizzle: NOOP, TFLOPS: 37.27
    out_tf32(mma2x4+warp2x4+stage2): ['70.5943985', '26.1725273'], time:12.68918ms, swizzle: NOOP, TFLOPS: 43.32 (+15.56%)
  out_tf32(mma2x4+...+stage3+dsmem): ['70.5943985', '26.1725273'], time:13.28039ms, swizzle: NOOP, TFLOPS: 41.40
  out_tf32(mma2x4+...+stage2+dsmem): ['70.5943985', '26.1725273'], time:12.78223ms, swizzle: NOOP, TFLOPS: 43.01
out_tf32(mma2x4+...+stage3+swizzle): ['70.5943985', '26.1725273'], time:14.66119ms, swizzle: 2048, TFLOPS: 37.50
out_tf32(mma2x4+...+stage2+swizzle): ['70.5943985', '26.1725273'], time:11.99231ms, swizzle: 2048, TFLOPS: 45.84 (+5.81%)
 out_tf32(...+stage3+dsmem+swizzle): ['70.5943985', '26.1725273'], time:13.03169ms, swizzle: 2048, TFLOPS: 42.19
 out_tf32(...+stage2+dsmem+swizzle): ['70.5943985', '26.1725273'], time:11.96327ms, swizzle: 2048, TFLOPS: 45.95 (+0.24%)
              out_tf32(cublas+tf32): ['70.5943985', '26.1725273'], time:9.859824ms, swizzle: NOOP, TFLOPS: 55.76 (+21.33%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=4096
                  out_f32x4(t8x8sk): ['151.796371', '4.59689951'], time:40.03288ms, swizzle: NOOP, TFLOPS: 27.47 (+0.00%)
                 out_f32x4(t8x8bcf): ['151.796371', '4.59689951'], time:39.52372ms, swizzle: NOOP, TFLOPS: 27.82 (+1.29%)
                out_f32x4(t8x8dbuf): ['151.796371', '4.59689951'], time:37.59534ms, swizzle: NOOP, TFLOPS: 29.25 (+5.13%)
                    out_f32(cublas): ['151.796371', '4.59689951'], time:27.83019ms, swizzle: NOOP, TFLOPS: 39.51 (+35.09%)
                         out_f32_th: ['151.796371', '4.59689951'], time:27.95956ms, swizzle: NOOP, TFLOPS: 39.33
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['151.794143', '4.5965395 '], time:29.30724ms, swizzle: NOOP, TFLOPS: 37.52
    out_tf32(mma2x4+warp2x4+stage2): ['151.794143', '4.5965395 '], time:25.27904ms, swizzle: NOOP, TFLOPS: 43.49 (+10.09%)
  out_tf32(mma2x4+...+stage3+dsmem): ['151.794143', '4.5965395 '], time:27.31575ms, swizzle: NOOP, TFLOPS: 40.25
  out_tf32(mma2x4+...+stage2+dsmem): ['151.794143', '4.5965395 '], time:25.58822ms, swizzle: NOOP, TFLOPS: 42.97
out_tf32(mma2x4+...+stage3+swizzle): ['151.794143', '4.5965395 '], time:29.27069ms, swizzle: 2048, TFLOPS: 37.56
out_tf32(mma2x4+...+stage2+swizzle): ['151.794143', '4.5965395 '], time:23.81775ms, swizzle: 2048, TFLOPS: 46.16 (+6.14%)
 out_tf32(...+stage3+dsmem+swizzle): ['151.794143', '4.5965395 '], time:26.00069ms, swizzle: 2048, TFLOPS: 42.29
 out_tf32(...+stage2+dsmem+swizzle): ['151.794143', '4.5965395 '], time:23.87239ms, swizzle: 2048, TFLOPS: 46.06
              out_tf32(cublas+tf32): ['151.794143', '4.5965395 '], time:19.24333ms, swizzle: NOOP, TFLOPS: 57.14 (+23.77%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=8192
                  out_f32x4(t8x8sk): ['118.532104', '44.2729606'], time:81.30698ms, swizzle: NOOP, TFLOPS: 27.05 (+0.00%)
                 out_f32x4(t8x8bcf): ['118.532104', '44.2729606'], time:75.78270ms, swizzle: NOOP, TFLOPS: 29.02 (+7.29%)
                out_f32x4(t8x8dbuf): ['118.532104', '44.2729606'], time:75.56617ms, swizzle: NOOP, TFLOPS: 29.10 (+0.29%)
                    out_f32(cublas): ['118.532104', '44.2729606'], time:56.42166ms, swizzle: NOOP, TFLOPS: 38.97 (+33.93%)
                         out_f32_th: ['118.532104', '44.2729606'], time:57.50610ms, swizzle: NOOP, TFLOPS: 38.24
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['118.526184', '44.2716636'], time:58.45718ms, swizzle: NOOP, TFLOPS: 37.62
    out_tf32(mma2x4+warp2x4+stage2): ['118.526184', '44.2716636'], time:51.36411ms, swizzle: NOOP, TFLOPS: 42.81 (+9.85%)
  out_tf32(mma2x4+...+stage3+dsmem): ['118.526184', '44.2716636'], time:53.86862ms, swizzle: NOOP, TFLOPS: 40.82
  out_tf32(mma2x4+...+stage2+dsmem): ['118.526184', '44.2716636'], time:51.22380ms, swizzle: NOOP, TFLOPS: 42.93 (+0.27%)
out_tf32(mma2x4+...+stage3+swizzle): ['118.526184', '44.2716636'], time:58.32481ms, swizzle: 2048, TFLOPS: 37.70
out_tf32(mma2x4+...+stage2+swizzle): ['118.526184', '44.2716636'], time:47.85780ms, swizzle: 2048, TFLOPS: 45.95 (+7.03%)
 out_tf32(...+stage3+dsmem+swizzle): ['118.526184', '44.2716636'], time:51.81453ms, swizzle: 2048, TFLOPS: 42.44
 out_tf32(...+stage2+dsmem+swizzle): ['118.526184', '44.2716636'], time:47.76165ms, swizzle: 2048, TFLOPS: 46.04 (+0.20%)
              out_tf32(cublas+tf32): ['118.526184', '44.2716636'], time:38.08858ms, swizzle: NOOP, TFLOPS: 57.73 (+25.40%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=2048
                  out_f32x4(t8x8sk): ['70.5949554', '26.1727619'], time:9.190845ms, swizzle: NOOP, TFLOPS: 29.91 (+0.00%)
                 out_f32x4(t8x8bcf): ['70.5949554', '26.1727619'], time:8.345413ms, swizzle: NOOP, TFLOPS: 32.94 (+10.13%)
                out_f32x4(t8x8dbuf): ['70.5949554', '26.1727619'], time:7.679963ms, swizzle: NOOP, TFLOPS: 35.79 (+8.66%)
                    out_f32(cublas): ['70.5949554', '26.1727619'], time:7.500529ms, swizzle: NOOP, TFLOPS: 36.65 (+2.39%)
                         out_f32_th: ['70.5949554', '26.1727619'], time:7.146787ms, swizzle: NOOP, TFLOPS: 38.46 (+4.95%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['70.5943985', '26.1725273'], time:7.968235ms, swizzle: NOOP, TFLOPS: 34.50
    out_tf32(mma2x4+warp2x4+stage2): ['70.5943985', '26.1725273'], time:6.254506ms, swizzle: NOOP, TFLOPS: 43.95 (+14.27%)
  out_tf32(mma2x4+...+stage3+dsmem): ['70.5943985', '26.1725273'], time:6.782460ms, swizzle: NOOP, TFLOPS: 40.53
  out_tf32(mma2x4+...+stage2+dsmem): ['70.5943985', '26.1725273'], time:6.247973ms, swizzle: NOOP, TFLOPS: 43.99 (+0.10%)
out_tf32(mma2x4+...+stage3+swizzle): ['70.5943985', '26.1725273'], time:7.488203ms, swizzle: 512 , TFLOPS: 36.71
out_tf32(mma2x4+...+stage2+swizzle): ['70.5943985', '26.1725273'], time:6.200075ms, swizzle: 512 , TFLOPS: 44.33 (+0.77%)
 out_tf32(...+stage3+dsmem+swizzle): ['70.5943985', '26.1725273'], time:6.759619ms, swizzle: 512 , TFLOPS: 40.66
 out_tf32(...+stage2+dsmem+swizzle): ['70.5943985', '26.1725273'], time:6.231451ms, swizzle: 512 , TFLOPS: 44.11
              out_tf32(cublas+tf32): ['70.5943985', '26.1725273'], time:5.184912ms, swizzle: NOOP, TFLOPS: 53.01 (+19.58%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=4096
                  out_f32x4(t8x8sk): ['151.796371', '4.59689951'], time:18.67318ms, swizzle: NOOP, TFLOPS: 29.44 (+0.00%)
                 out_f32x4(t8x8bcf): ['151.796371', '4.59689951'], time:16.58837ms, swizzle: NOOP, TFLOPS: 33.14 (+12.57%)
                out_f32x4(t8x8dbuf): ['151.796371', '4.59689951'], time:16.44637ms, swizzle: NOOP, TFLOPS: 33.43 (+0.86%)
                    out_f32(cublas): ['151.796371', '4.59689951'], time:14.57281ms, swizzle: NOOP, TFLOPS: 37.72 (+12.86%)
                         out_f32_th: ['151.796371', '4.59689951'], time:14.51504ms, swizzle: NOOP, TFLOPS: 37.87 (+0.40%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['151.794143', '4.5965395 '], time:15.47667ms, swizzle: NOOP, TFLOPS: 35.52
    out_tf32(mma2x4+warp2x4+stage2): ['151.794143', '4.5965395 '], time:12.47291ms, swizzle: NOOP, TFLOPS: 44.08 (+16.37%)
  out_tf32(mma2x4+...+stage3+dsmem): ['151.794143', '4.5965395 '], time:13.44106ms, swizzle: NOOP, TFLOPS: 40.90
  out_tf32(mma2x4+...+stage2+dsmem): ['151.794143', '4.5965395 '], time:12.39275ms, swizzle: NOOP, TFLOPS: 44.36 (+0.65%)
out_tf32(mma2x4+...+stage3+swizzle): ['151.794143', '4.5965395 '], time:14.96281ms, swizzle: 512 , TFLOPS: 36.74
out_tf32(mma2x4+...+stage2+swizzle): ['151.794143', '4.5965395 '], time:12.40277ms, swizzle: 512 , TFLOPS: 44.33
 out_tf32(...+stage3+dsmem+swizzle): ['151.794143', '4.5965395 '], time:13.47801ms, swizzle: 512 , TFLOPS: 40.79
 out_tf32(...+stage2+dsmem+swizzle): ['151.794143', '4.5965395 '], time:12.43972ms, swizzle: 512 , TFLOPS: 44.19
              out_tf32(cublas+tf32): ['151.794143', '4.5965395 '], time:10.85467ms, swizzle: NOOP, TFLOPS: 50.65 (+14.17%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=8192
                  out_f32x4(t8x8sk): ['118.532104', '44.2729606'], time:38.72056ms, swizzle: NOOP, TFLOPS: 28.40 (+0.00%)
                 out_f32x4(t8x8bcf): ['118.532104', '44.2729606'], time:34.69905ms, swizzle: NOOP, TFLOPS: 31.69 (+11.59%)
                out_f32x4(t8x8dbuf): ['118.532104', '44.2729606'], time:36.12399ms, swizzle: NOOP, TFLOPS: 30.44
                    out_f32(cublas): ['118.532104', '44.2729606'], time:28.58903ms, swizzle: NOOP, TFLOPS: 38.46 (+21.37%)
                         out_f32_th: ['118.532104', '44.2729606'], time:28.67548ms, swizzle: NOOP, TFLOPS: 38.34
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['118.526184', '44.2716636'], time:30.18698ms, swizzle: NOOP, TFLOPS: 36.42
    out_tf32(mma2x4+warp2x4+stage2): ['118.526184', '44.2716636'], time:25.13649ms, swizzle: NOOP, TFLOPS: 43.74 (+13.74%)
  out_tf32(mma2x4+...+stage3+dsmem): ['118.526184', '44.2716636'], time:26.80773ms, swizzle: NOOP, TFLOPS: 41.01
  out_tf32(mma2x4+...+stage2+dsmem): ['118.526184', '44.2716636'], time:25.33824ms, swizzle: NOOP, TFLOPS: 43.39
out_tf32(mma2x4+...+stage3+swizzle): ['118.526184', '44.2716636'], time:30.06722ms, swizzle: 512 , TFLOPS: 36.57
out_tf32(mma2x4+...+stage2+swizzle): ['118.526184', '44.2716636'], time:25.04580ms, swizzle: 512 , TFLOPS: 43.90 (+0.36%)
 out_tf32(...+stage3+dsmem+swizzle): ['118.526184', '44.2716636'], time:26.84135ms, swizzle: 512 , TFLOPS: 40.96
 out_tf32(...+stage2+dsmem+swizzle): ['118.526184', '44.2716636'], time:24.95355ms, swizzle: 512 , TFLOPS: 44.06 (+0.37%)
              out_tf32(cublas+tf32): ['118.526184', '44.2716636'], time:19.74155ms, swizzle: NOOP, TFLOPS: 55.70 (+26.40%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=2048
                  out_f32x4(t8x8sk): ['70.5949554', '26.1727619'], time:18.36364ms, swizzle: NOOP, TFLOPS: 29.94 (+0.00%)
                 out_f32x4(t8x8bcf): ['70.5949554', '26.1727619'], time:16.34912ms, swizzle: NOOP, TFLOPS: 33.63 (+12.32%)
                out_f32x4(t8x8dbuf): ['70.5949554', '26.1727619'], time:14.82284ms, swizzle: NOOP, TFLOPS: 37.09 (+10.30%)
                    out_f32(cublas): ['70.5949554', '26.1727619'], time:14.45541ms, swizzle: NOOP, TFLOPS: 38.03 (+2.54%)
                         out_f32_th: ['70.5949554', '26.1727619'], time:14.56203ms, swizzle: NOOP, TFLOPS: 37.75
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['70.5943985', '26.1725273'], time:14.91312ms, swizzle: NOOP, TFLOPS: 36.86
    out_tf32(mma2x4+warp2x4+stage2): ['70.5943985', '26.1725273'], time:12.08066ms, swizzle: NOOP, TFLOPS: 45.51 (+19.66%)
  out_tf32(mma2x4+...+stage3+dsmem): ['70.5943985', '26.1725273'], time:13.07072ms, swizzle: NOOP, TFLOPS: 42.06
  out_tf32(mma2x4+...+stage2+dsmem): ['70.5943985', '26.1725273'], time:12.01016ms, swizzle: NOOP, TFLOPS: 45.77 (+0.59%)
out_tf32(mma2x4+...+stage3+swizzle): ['70.5943985', '26.1725273'], time:14.58058ms, swizzle: 1024, TFLOPS: 37.70
out_tf32(mma2x4+...+stage2+swizzle): ['70.5943985', '26.1725273'], time:12.01112ms, swizzle: 1024, TFLOPS: 45.77
 out_tf32(...+stage3+dsmem+swizzle): ['70.5943985', '26.1725273'], time:13.10825ms, swizzle: 1024, TFLOPS: 41.94
 out_tf32(...+stage2+dsmem+swizzle): ['70.5943985', '26.1725273'], time:12.03854ms, swizzle: 1024, TFLOPS: 45.67
              out_tf32(cublas+tf32): ['70.5943985', '26.1725273'], time:9.944319ms, swizzle: NOOP, TFLOPS: 55.28 (+20.77%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=4096
                  out_f32x4(t8x8sk): ['151.796371', '4.59689951'], time:39.44745ms, swizzle: NOOP, TFLOPS: 27.87 (+0.00%)
                 out_f32x4(t8x8bcf): ['151.796371', '4.59689951'], time:35.19003ms, swizzle: NOOP, TFLOPS: 31.24 (+12.10%)
                out_f32x4(t8x8dbuf): ['151.796371', '4.59689951'], time:36.57977ms, swizzle: NOOP, TFLOPS: 30.06
                    out_f32(cublas): ['151.796371', '4.59689951'], time:27.93822ms, swizzle: NOOP, TFLOPS: 39.36 (+25.96%)
                         out_f32_th: ['151.796371', '4.59689951'], time:27.93700ms, swizzle: NOOP, TFLOPS: 39.36 (+0.00%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['151.794143', '4.5965395 '], time:29.24573ms, swizzle: NOOP, TFLOPS: 37.60
    out_tf32(mma2x4+warp2x4+stage2): ['151.794143', '4.5965395 '], time:24.57020ms, swizzle: NOOP, TFLOPS: 44.75 (+13.70%)
  out_tf32(mma2x4+...+stage3+dsmem): ['151.794143', '4.5965395 '], time:26.55055ms, swizzle: NOOP, TFLOPS: 41.41
  out_tf32(mma2x4+...+stage2+dsmem): ['151.794143', '4.5965395 '], time:24.88572ms, swizzle: NOOP, TFLOPS: 44.18
out_tf32(mma2x4+...+stage3+swizzle): ['151.794143', '4.5965395 '], time:29.28466ms, swizzle: 1024, TFLOPS: 37.55
out_tf32(mma2x4+...+stage2+swizzle): ['151.794143', '4.5965395 '], time:23.89683ms, swizzle: 1024, TFLOPS: 46.01 (+2.82%)
 out_tf32(...+stage3+dsmem+swizzle): ['151.794143', '4.5965395 '], time:26.11415ms, swizzle: 1024, TFLOPS: 42.10
 out_tf32(...+stage2+dsmem+swizzle): ['151.794143', '4.5965395 '], time:23.87890ms, swizzle: 1024, TFLOPS: 46.05 (+0.08%)
              out_tf32(cublas+tf32): ['151.794143', '4.5965395 '], time:19.27731ms, swizzle: NOOP, TFLOPS: 57.04 (+23.87%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=8192
                  out_f32x4(t8x8sk): ['118.532104', '44.2729606'], time:79.11319ms, swizzle: NOOP, TFLOPS: 27.80 (+0.00%)
                 out_f32x4(t8x8bcf): ['118.532104', '44.2729606'], time:70.98405ms, swizzle: NOOP, TFLOPS: 30.98 (+11.45%)
                out_f32x4(t8x8dbuf): ['118.532104', '44.2729606'], time:71.76809ms, swizzle: NOOP, TFLOPS: 30.64
                    out_f32(cublas): ['118.532104', '44.2729606'], time:55.91969ms, swizzle: NOOP, TFLOPS: 39.32 (+26.94%)
                         out_f32_th: ['118.532104', '44.2729606'], time:56.78405ms, swizzle: NOOP, TFLOPS: 38.73
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['118.526184', '44.2716636'], time:58.23874ms, swizzle: NOOP, TFLOPS: 37.76
    out_tf32(mma2x4+warp2x4+stage2): ['118.526184', '44.2716636'], time:49.20217ms, swizzle: NOOP, TFLOPS: 44.69 (+13.65%)
  out_tf32(mma2x4+...+stage3+dsmem): ['118.526184', '44.2716636'], time:53.33271ms, swizzle: NOOP, TFLOPS: 41.23
  out_tf32(mma2x4+...+stage2+dsmem): ['118.526184', '44.2716636'], time:49.59840ms, swizzle: NOOP, TFLOPS: 44.34
out_tf32(mma2x4+...+stage3+swizzle): ['118.526184', '44.2716636'], time:58.33761ms, swizzle: 1024, TFLOPS: 37.69
out_tf32(mma2x4+...+stage2+swizzle): ['118.526184', '44.2716636'], time:47.81997ms, swizzle: 1024, TFLOPS: 45.99 (+2.89%)
 out_tf32(...+stage3+dsmem+swizzle): ['118.526184', '44.2716636'], time:51.88267ms, swizzle: 1024, TFLOPS: 42.38
 out_tf32(...+stage2+dsmem+swizzle): ['118.526184', '44.2716636'], time:47.80828ms, swizzle: 1024, TFLOPS: 46.00 (+0.02%)
              out_tf32(cublas+tf32): ['118.526184', '44.2716636'], time:38.11509ms, swizzle: NOOP, TFLOPS: 57.69 (+25.43%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=2048
                  out_f32x4(t8x8sk): ['70.5949554', '26.1727619'], time:40.08102ms, swizzle: NOOP, TFLOPS: 27.43 (+0.00%)
                 out_f32x4(t8x8bcf): ['70.5949554', '26.1727619'], time:39.66226ms, swizzle: NOOP, TFLOPS: 27.72 (+1.06%)
                out_f32x4(t8x8dbuf): ['70.5949554', '26.1727619'], time:36.46554ms, swizzle: NOOP, TFLOPS: 30.15 (+8.77%)
                    out_f32(cublas): ['70.5949554', '26.1727619'], time:28.34019ms, swizzle: NOOP, TFLOPS: 38.80 (+28.67%)
                         out_f32_th: ['70.5949554', '26.1727619'], time:28.30972ms, swizzle: NOOP, TFLOPS: 38.84 (+0.11%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['70.5943985', '26.1725273'], time:28.73399ms, swizzle: NOOP, TFLOPS: 38.27
    out_tf32(mma2x4+warp2x4+stage2): ['70.5943985', '26.1725273'], time:25.33073ms, swizzle: NOOP, TFLOPS: 43.41 (+11.76%)
  out_tf32(mma2x4+...+stage3+dsmem): ['70.5943985', '26.1725273'], time:26.69138ms, swizzle: NOOP, TFLOPS: 41.19
  out_tf32(mma2x4+...+stage2+dsmem): ['70.5943985', '26.1725273'], time:25.41232ms, swizzle: NOOP, TFLOPS: 43.27
out_tf32(mma2x4+...+stage3+swizzle): ['70.5943985', '26.1725273'], time:28.79602ms, swizzle: 2048, TFLOPS: 38.18
out_tf32(mma2x4+...+stage2+swizzle): ['70.5943985', '26.1725273'], time:23.39887ms, swizzle: 2048, TFLOPS: 46.99 (+8.26%)
 out_tf32(...+stage3+dsmem+swizzle): ['70.5943985', '26.1725273'], time:25.56235ms, swizzle: 2048, TFLOPS: 43.01
 out_tf32(...+stage2+dsmem+swizzle): ['70.5943985', '26.1725273'], time:23.46084ms, swizzle: 2048, TFLOPS: 46.87
              out_tf32(cublas+tf32): ['70.5943985', '26.1725273'], time:19.40128ms, swizzle: NOOP, TFLOPS: 56.67 (+20.60%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=4096
                  out_f32x4(t8x8sk): ['151.796371', '4.59689951'], time:81.40509ms, swizzle: NOOP, TFLOPS: 27.01 (+0.00%)
                 out_f32x4(t8x8bcf): ['151.796371', '4.59689951'], time:75.39424ms, swizzle: NOOP, TFLOPS: 29.17 (+7.97%)
                out_f32x4(t8x8dbuf): ['151.796371', '4.59689951'], time:75.67217ms, swizzle: NOOP, TFLOPS: 29.06
                    out_f32(cublas): ['151.796371', '4.59689951'], time:55.54578ms, swizzle: NOOP, TFLOPS: 39.59 (+35.73%)
                         out_f32_th: ['151.796371', '4.59689951'], time:56.35116ms, swizzle: NOOP, TFLOPS: 39.02
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['151.794143', '4.5965395 '], time:57.64467ms, swizzle: NOOP, TFLOPS: 38.15
    out_tf32(mma2x4+warp2x4+stage2): ['151.794143', '4.5965395 '], time:50.40433ms, swizzle: NOOP, TFLOPS: 43.63 (+10.20%)
  out_tf32(mma2x4+...+stage3+dsmem): ['151.794143', '4.5965395 '], time:53.50663ms, swizzle: NOOP, TFLOPS: 41.10
  out_tf32(mma2x4+...+stage2+dsmem): ['151.794143', '4.5965395 '], time:50.22649ms, swizzle: NOOP, TFLOPS: 43.78 (+0.35%)
out_tf32(mma2x4+...+stage3+swizzle): ['151.794143', '4.5965395 '], time:57.27660ms, swizzle: 2048, TFLOPS: 38.39
out_tf32(mma2x4+...+stage2+swizzle): ['151.794143', '4.5965395 '], time:46.61462ms, swizzle: 2048, TFLOPS: 47.17 (+7.75%)
 out_tf32(...+stage3+dsmem+swizzle): ['151.794143', '4.5965395 '], time:50.91807ms, swizzle: 2048, TFLOPS: 43.19
 out_tf32(...+stage2+dsmem+swizzle): ['151.794143', '4.5965395 '], time:46.73092ms, swizzle: 2048, TFLOPS: 47.06
              out_tf32(cublas+tf32): ['151.794143', '4.5965395 '], time:38.29209ms, swizzle: NOOP, TFLOPS: 57.43 (+21.73%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=8192
                  out_f32x4(t8x8sk): ['118.532104', '44.2729606'], time:162.8879ms, swizzle: NOOP, TFLOPS: 27.00 (+0.00%)
                 out_f32x4(t8x8bcf): ['118.532104', '44.2729606'], time:151.1848ms, swizzle: NOOP, TFLOPS: 29.09 (+7.74%)
                out_f32x4(t8x8dbuf): ['118.532104', '44.2729606'], time:151.3025ms, swizzle: NOOP, TFLOPS: 29.07
                    out_f32(cublas): ['118.532104', '44.2729606'], time:112.4181ms, swizzle: NOOP, TFLOPS: 39.12 (+34.48%)
                         out_f32_th: ['118.532104', '44.2729606'], time:112.4917ms, swizzle: NOOP, TFLOPS: 39.10
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['118.526184', '44.2716636'], time:115.7331ms, swizzle: NOOP, TFLOPS: 38.00
    out_tf32(mma2x4+warp2x4+stage2): ['118.526184', '44.2716636'], time:100.3637ms, swizzle: NOOP, TFLOPS: 43.82 (+12.01%)
  out_tf32(mma2x4+...+stage3+dsmem): ['118.526184', '44.2716636'], time:106.3712ms, swizzle: NOOP, TFLOPS: 41.35
  out_tf32(mma2x4+...+stage2+dsmem): ['118.526184', '44.2716636'], time:102.4972ms, swizzle: NOOP, TFLOPS: 42.91
out_tf32(mma2x4+...+stage3+swizzle): ['118.526184', '44.2716636'], time:114.2313ms, swizzle: 2048, TFLOPS: 38.50
out_tf32(mma2x4+...+stage2+swizzle): ['118.526184', '44.2716636'], time:93.91186ms, swizzle: 2048, TFLOPS: 46.83 (+6.87%)
 out_tf32(...+stage3+dsmem+swizzle): ['118.526184', '44.2716636'], time:101.5390ms, swizzle: 2048, TFLOPS: 43.31
 out_tf32(...+stage2+dsmem+swizzle): ['118.526184', '44.2716636'], time:93.69635ms, swizzle: 2048, TFLOPS: 46.94 (+0.23%)
              out_tf32(cublas+tf32): ['118.526184', '44.2716636'], time:75.96850ms, swizzle: NOOP, TFLOPS: 57.89 (+23.34%)
----------------------------------------------------------------------------------------------------------------------------------
```
