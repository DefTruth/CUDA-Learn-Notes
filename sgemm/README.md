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
                     out_f32(naive): ['-17.842391', '0.18722232'], time:20.40503ms, swizzle: NOOP, TFLOPS: 3.37  (+0.00%)
                  out_f32x4(t8x8sk): ['-17.842391', '0.18722232'], time:2.430105ms, swizzle: NOOP, TFLOPS: 28.28 (+739.68%)
                 out_f32x4(t8x8bcf): ['-17.842391', '0.18722232'], time:2.102661ms, swizzle: NOOP, TFLOPS: 32.68 (+15.57%)
                out_f32x4(t8x8dbuf): ['-17.842391', '0.18722232'], time:1.985645ms, swizzle: NOOP, TFLOPS: 34.61 (+5.89%)
                    out_f32(cublas): ['-17.842391', '0.18722232'], time:2.087247ms, swizzle: NOOP, TFLOPS: 32.92
                         out_f32_th: ['-17.842391', '0.18722232'], time:1.845526ms, swizzle: NOOP, TFLOPS: 37.24 (+7.59%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-17.849796', '0.19758633'], time:2.149534ms, swizzle: NOOP, TFLOPS: 31.97
    out_tf32(mma2x4+warp2x4+stage2): ['-17.849796', '0.19758633'], time:1.663148ms, swizzle: NOOP, TFLOPS: 41.32 (+10.97%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-17.849796', '0.19758633'], time:1.823186ms, swizzle: NOOP, TFLOPS: 37.69
  out_tf32(mma2x4+...+stage2+dsmem): ['-17.849796', '0.19758633'], time:1.662409ms, swizzle: NOOP, TFLOPS: 41.34 (+0.04%)
out_tf32(mma2x4+...+stage3+swizzle): ['-17.849796', '0.19758633'], time:2.001917ms, swizzle: 1024, TFLOPS: 34.33
out_tf32(mma2x4+...+stage2+swizzle): ['-17.849796', '0.19758633'], time:1.634597ms, swizzle: 1024, TFLOPS: 42.04 (+1.70%)
 out_tf32(...+stage3+dsmem+swizzle): ['-17.849796', '0.19758633'], time:1.806831ms, swizzle: 1024, TFLOPS: 38.03
 out_tf32(...+stage2+dsmem+swizzle): ['-17.849796', '0.19758633'], time:1.635658ms, swizzle: 1024, TFLOPS: 42.01
              out_tf32(cublas+tf32): ['-17.849796', '0.19758633'], time:1.539385ms, swizzle: NOOP, TFLOPS: 44.64 (+6.19%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=4096
                     out_f32(naive): ['-24.547933', '26.0833282'], time:40.70725ms, swizzle: NOOP, TFLOPS: 3.38  (+0.00%)
                  out_f32x4(t8x8sk): ['-24.547933', '26.0833282'], time:4.871845ms, swizzle: NOOP, TFLOPS: 28.21 (+735.56%)
                 out_f32x4(t8x8bcf): ['-24.547933', '26.0833282'], time:4.412031ms, swizzle: NOOP, TFLOPS: 31.15 (+10.42%)
                out_f32x4(t8x8dbuf): ['-24.547933', '26.0833282'], time:4.048168ms, swizzle: NOOP, TFLOPS: 33.95 (+8.99%)
                    out_f32(cublas): ['-24.547933', '26.0833282'], time:4.019129ms, swizzle: NOOP, TFLOPS: 34.20 (+0.72%)
                         out_f32_th: ['-24.547933', '26.0833282'], time:3.687226ms, swizzle: NOOP, TFLOPS: 37.27 (+9.00%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-24.539142', '26.0933208'], time:4.418766ms, swizzle: NOOP, TFLOPS: 31.10
    out_tf32(mma2x4+warp2x4+stage2): ['-24.539142', '26.0933208'], time:3.449714ms, swizzle: NOOP, TFLOPS: 39.84 (+6.88%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-24.539142', '26.0933208'], time:3.769552ms, swizzle: NOOP, TFLOPS: 36.46
  out_tf32(mma2x4+...+stage2+dsmem): ['-24.539142', '26.0933208'], time:3.442633ms, swizzle: NOOP, TFLOPS: 39.92 (+0.21%)
out_tf32(mma2x4+...+stage3+swizzle): ['-24.539142', '26.0933208'], time:4.073047ms, swizzle: 1024, TFLOPS: 33.74
out_tf32(mma2x4+...+stage2+swizzle): ['-24.539142', '26.0933208'], time:3.320538ms, swizzle: 1024, TFLOPS: 41.39 (+3.68%)
 out_tf32(...+stage3+dsmem+swizzle): ['-24.539142', '26.0933208'], time:3.651261ms, swizzle: 1024, TFLOPS: 37.64
 out_tf32(...+stage2+dsmem+swizzle): ['-24.539142', '26.0933208'], time:3.319275ms, swizzle: 1024, TFLOPS: 41.41 (+0.04%)
              out_tf32(cublas+tf32): ['-24.539142', '26.0933208'], time:2.791321ms, swizzle: NOOP, TFLOPS: 49.24 (+18.91%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=8192
                     out_f32(naive): ['47.3211364', '96.7818374'], time:124.1455ms, swizzle: NOOP, TFLOPS: 2.21  (+0.00%)
                  out_f32x4(t8x8sk): ['47.3211364', '96.7818374'], time:10.18377ms, swizzle: NOOP, TFLOPS: 26.99 (+1119.05%)
                 out_f32x4(t8x8bcf): ['47.3211364', '96.7818374'], time:8.965158ms, swizzle: NOOP, TFLOPS: 30.66 (+13.59%)
                out_f32x4(t8x8dbuf): ['47.3211364', '96.7818374'], time:9.146523ms, swizzle: NOOP, TFLOPS: 30.05
                    out_f32(cublas): ['47.3211364', '96.7818374'], time:7.824325ms, swizzle: NOOP, TFLOPS: 35.13 (+14.58%)
                         out_f32_th: ['47.3211364', '96.7818374'], time:7.979285ms, swizzle: NOOP, TFLOPS: 34.45
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['47.3002510', '96.8180007'], time:8.486199ms, swizzle: NOOP, TFLOPS: 32.39
    out_tf32(mma2x4+warp2x4+stage2): ['47.3002510', '96.8180007'], time:6.938815ms, swizzle: NOOP, TFLOPS: 39.61 (+12.76%)
  out_tf32(mma2x4+...+stage3+dsmem): ['47.3002510', '96.8180007'], time:7.443344ms, swizzle: NOOP, TFLOPS: 36.93
  out_tf32(mma2x4+...+stage2+dsmem): ['47.3002510', '96.8180007'], time:6.947302ms, swizzle: NOOP, TFLOPS: 39.57
out_tf32(mma2x4+...+stage3+swizzle): ['47.3002510', '96.8180007'], time:8.268499ms, swizzle: 1024, TFLOPS: 33.24
out_tf32(mma2x4+...+stage2+swizzle): ['47.3002510', '96.8180007'], time:6.863093ms, swizzle: 1024, TFLOPS: 40.05 (+1.10%)
 out_tf32(...+stage3+dsmem+swizzle): ['47.3002510', '96.8180007'], time:7.466220ms, swizzle: 1024, TFLOPS: 36.82
 out_tf32(...+stage2+dsmem+swizzle): ['47.3002510', '96.8180007'], time:6.836020ms, swizzle: 1024, TFLOPS: 40.21 (+0.40%)
              out_tf32(cublas+tf32): ['47.3002510', '96.8180007'], time:5.307173ms, swizzle: NOOP, TFLOPS: 51.79 (+28.81%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=2048
                     out_f32(naive): ['-17.835220', '0.19710006'], time:40.66953ms, swizzle: NOOP, TFLOPS: 3.38  (+0.00%)
                  out_f32x4(t8x8sk): ['-17.835220', '0.19710006'], time:4.663670ms, swizzle: NOOP, TFLOPS: 29.47 (+772.05%)
                 out_f32x4(t8x8bcf): ['-17.835220', '0.19710006'], time:4.213857ms, swizzle: NOOP, TFLOPS: 32.62 (+10.67%)
                out_f32x4(t8x8dbuf): ['-17.835220', '0.19710006'], time:3.852760ms, swizzle: NOOP, TFLOPS: 35.67 (+9.37%)
                    out_f32(cublas): ['-17.835220', '0.19710006'], time:3.993618ms, swizzle: NOOP, TFLOPS: 34.41
                         out_f32_th: ['-17.835220', '0.19710006'], time:3.633618ms, swizzle: NOOP, TFLOPS: 37.82 (+6.03%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-17.849796', '0.19758633'], time:3.958535ms, swizzle: NOOP, TFLOPS: 34.72
    out_tf32(mma2x4+warp2x4+stage2): ['-17.849796', '0.19758633'], time:3.184044ms, swizzle: NOOP, TFLOPS: 43.16 (+14.12%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-17.849796', '0.19758633'], time:3.465819ms, swizzle: NOOP, TFLOPS: 39.66
  out_tf32(mma2x4+...+stage2+dsmem): ['-17.849796', '0.19758633'], time:3.177452ms, swizzle: NOOP, TFLOPS: 43.25 (+0.21%)
out_tf32(mma2x4+...+stage3+swizzle): ['-17.849796', '0.19758633'], time:3.823959ms, swizzle: 2048, TFLOPS: 35.94
out_tf32(mma2x4+...+stage2+swizzle): ['-17.849796', '0.19758633'], time:3.122901ms, swizzle: 2048, TFLOPS: 44.01 (+1.75%)
 out_tf32(...+stage3+dsmem+swizzle): ['-17.849796', '0.19758633'], time:3.422784ms, swizzle: 2048, TFLOPS: 40.15
 out_tf32(...+stage2+dsmem+swizzle): ['-17.849796', '0.19758633'], time:3.124201ms, swizzle: 2048, TFLOPS: 43.99
              out_tf32(cublas+tf32): ['-17.849796', '0.19758633'], time:2.821981ms, swizzle: NOOP, TFLOPS: 48.70 (+10.66%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=4096
                     out_f32(naive): ['-24.541957', '26.1021537'], time:134.3554ms, swizzle: NOOP, TFLOPS: 2.05  (+0.00%)
                  out_f32x4(t8x8sk): ['-24.541957', '26.1021537'], time:9.993898ms, swizzle: NOOP, TFLOPS: 27.50 (+1244.37%)
                 out_f32x4(t8x8bcf): ['-24.541957', '26.1021537'], time:9.019649ms, swizzle: NOOP, TFLOPS: 30.48 (+10.80%)
                out_f32x4(t8x8dbuf): ['-24.541957', '26.1021537'], time:9.230816ms, swizzle: NOOP, TFLOPS: 29.78
                    out_f32(cublas): ['-24.541957', '26.1021537'], time:7.709038ms, swizzle: NOOP, TFLOPS: 35.66 (+17.00%)
                         out_f32_th: ['-24.541957', '26.1021537'], time:7.547247ms, swizzle: NOOP, TFLOPS: 36.42 (+2.14%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-24.539142', '26.0933208'], time:7.894265ms, swizzle: NOOP, TFLOPS: 34.82
    out_tf32(mma2x4+warp2x4+stage2): ['-24.539142', '26.0933208'], time:6.565976ms, swizzle: NOOP, TFLOPS: 41.86 (+14.94%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-24.539142', '26.0933208'], time:6.902301ms, swizzle: NOOP, TFLOPS: 39.82
  out_tf32(mma2x4+...+stage2+dsmem): ['-24.539142', '26.0933208'], time:6.546056ms, swizzle: NOOP, TFLOPS: 41.99 (+0.30%)
out_tf32(mma2x4+...+stage3+swizzle): ['-24.539142', '26.0933208'], time:7.725191ms, swizzle: 2048, TFLOPS: 35.58
out_tf32(mma2x4+...+stage2+swizzle): ['-24.539142', '26.0933208'], time:6.347680ms, swizzle: 2048, TFLOPS: 43.30 (+3.13%)
 out_tf32(...+stage3+dsmem+swizzle): ['-24.539142', '26.0933208'], time:6.931185ms, swizzle: 2048, TFLOPS: 39.66
 out_tf32(...+stage2+dsmem+swizzle): ['-24.539142', '26.0933208'], time:6.350684ms, swizzle: 2048, TFLOPS: 43.28
              out_tf32(cublas+tf32): ['-24.539142', '26.0933208'], time:5.310356ms, swizzle: NOOP, TFLOPS: 51.76 (+19.53%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=8192
                     out_f32(naive): ['47.3047409', '96.8061981'], time:275.4955ms, swizzle: NOOP, TFLOPS: 2.00  (+0.00%)
                  out_f32x4(t8x8sk): ['47.3047409', '96.8061981'], time:20.12772ms, swizzle: NOOP, TFLOPS: 27.31 (+1268.74%)
                 out_f32x4(t8x8bcf): ['47.3047409', '96.8061981'], time:18.67715ms, swizzle: NOOP, TFLOPS: 29.43 (+7.77%)
                out_f32x4(t8x8dbuf): ['47.3047409', '96.8061981'], time:20.63833ms, swizzle: NOOP, TFLOPS: 26.64
                    out_f32(cublas): ['47.3047409', '96.8061981'], time:15.28421ms, swizzle: NOOP, TFLOPS: 35.97 (+22.20%)
                         out_f32_th: ['47.3047409', '96.8061981'], time:14.75317ms, swizzle: NOOP, TFLOPS: 37.26 (+3.60%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['47.3002510', '96.8180007'], time:15.59357ms, swizzle: NOOP, TFLOPS: 35.26
    out_tf32(mma2x4+warp2x4+stage2): ['47.3002510', '96.8180007'], time:13.13108ms, swizzle: NOOP, TFLOPS: 41.87 (+12.35%)
  out_tf32(mma2x4+...+stage3+dsmem): ['47.3002510', '96.8180007'], time:13.90327ms, swizzle: NOOP, TFLOPS: 39.54
  out_tf32(mma2x4+...+stage2+dsmem): ['47.3002510', '96.8180007'], time:13.19309ms, swizzle: NOOP, TFLOPS: 41.67
out_tf32(mma2x4+...+stage3+swizzle): ['47.3002510', '96.8180007'], time:15.48467ms, swizzle: 2048, TFLOPS: 35.50
out_tf32(mma2x4+...+stage2+swizzle): ['47.3002510', '96.8180007'], time:12.81875ms, swizzle: 2048, TFLOPS: 42.89 (+2.44%)
 out_tf32(...+stage3+dsmem+swizzle): ['47.3002510', '96.8180007'], time:13.90204ms, swizzle: 2048, TFLOPS: 39.54
 out_tf32(...+stage2+dsmem+swizzle): ['47.3002510', '96.8180007'], time:12.77471ms, swizzle: 2048, TFLOPS: 43.03 (+0.34%)
              out_tf32(cublas+tf32): ['47.3002510', '96.8180007'], time:10.32357ms, swizzle: NOOP, TFLOPS: 53.25 (+23.74%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=2048
                     out_f32(naive): ['-17.835220', '0.19710006'], time:138.1242ms, swizzle: NOOP, TFLOPS: 1.99  (+0.00%)
                  out_f32x4(t8x8sk): ['-17.835220', '0.19710006'], time:10.01632ms, swizzle: NOOP, TFLOPS: 27.44 (+1278.99%)
                 out_f32x4(t8x8bcf): ['-17.835220', '0.19710006'], time:9.498941ms, swizzle: NOOP, TFLOPS: 28.94 (+5.45%)
                out_f32x4(t8x8dbuf): ['-17.835220', '0.19710006'], time:9.595859ms, swizzle: NOOP, TFLOPS: 28.65
                    out_f32(cublas): ['-17.835220', '0.19710006'], time:7.673835ms, swizzle: NOOP, TFLOPS: 35.82 (+23.78%)
                         out_f32_th: ['-17.835220', '0.19710006'], time:7.615864ms, swizzle: NOOP, TFLOPS: 36.09 (+0.76%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-17.849796', '0.19758633'], time:7.673478ms, swizzle: NOOP, TFLOPS: 35.82
    out_tf32(mma2x4+warp2x4+stage2): ['-17.849796', '0.19758633'], time:6.537437ms, swizzle: NOOP, TFLOPS: 42.05 (+16.50%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-17.849796', '0.19758633'], time:6.784737ms, swizzle: NOOP, TFLOPS: 40.51
  out_tf32(mma2x4+...+stage2+dsmem): ['-17.849796', '0.19758633'], time:6.545460ms, swizzle: NOOP, TFLOPS: 42.00
out_tf32(mma2x4+...+stage3+swizzle): ['-17.849796', '0.19758633'], time:7.543981ms, swizzle: 4096, TFLOPS: 36.44
out_tf32(mma2x4+...+stage2+swizzle): ['-17.849796', '0.19758633'], time:6.199836ms, swizzle: 4096, TFLOPS: 44.34 (+5.45%)
 out_tf32(...+stage3+dsmem+swizzle): ['-17.849796', '0.19758633'], time:6.745231ms, swizzle: 4096, TFLOPS: 40.75
 out_tf32(...+stage2+dsmem+swizzle): ['-17.849796', '0.19758633'], time:6.165897ms, swizzle: 4096, TFLOPS: 44.58 (+0.55%)
              out_tf32(cublas+tf32): ['-17.849796', '0.19758633'], time:5.148077ms, swizzle: NOOP, TFLOPS: 53.39 (+19.77%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=4096
                     out_f32(naive): ['-24.556724', '26.1026535'], time:275.8962ms, swizzle: NOOP, TFLOPS: 1.99  (+0.00%)
                  out_f32x4(t8x8sk): ['-24.556724', '26.1026535'], time:20.26476ms, swizzle: NOOP, TFLOPS: 27.13 (+1261.46%)
                 out_f32x4(t8x8bcf): ['-24.556724', '26.1026535'], time:19.30289ms, swizzle: NOOP, TFLOPS: 28.48 (+4.98%)
                out_f32x4(t8x8dbuf): ['-24.556724', '26.1026535'], time:20.73652ms, swizzle: NOOP, TFLOPS: 26.51
                    out_f32(cublas): ['-24.556724', '26.1026535'], time:14.44500ms, swizzle: NOOP, TFLOPS: 38.06 (+33.63%)
                         out_f32_th: ['-24.556724', '26.1026535'], time:14.17189ms, swizzle: NOOP, TFLOPS: 38.79 (+1.93%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-24.539142', '26.0933208'], time:15.15815ms, swizzle: NOOP, TFLOPS: 36.27
    out_tf32(mma2x4+warp2x4+stage2): ['-24.539142', '26.0933208'], time:13.11252ms, swizzle: NOOP, TFLOPS: 41.93 (+8.08%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-24.539142', '26.0933208'], time:13.83591ms, swizzle: NOOP, TFLOPS: 39.73
  out_tf32(mma2x4+...+stage2+dsmem): ['-24.539142', '26.0933208'], time:13.17880ms, swizzle: NOOP, TFLOPS: 41.72
out_tf32(mma2x4+...+stage3+swizzle): ['-24.539142', '26.0933208'], time:15.04755ms, swizzle: 4096, TFLOPS: 36.53
out_tf32(mma2x4+...+stage2+swizzle): ['-24.539142', '26.0933208'], time:12.35028ms, swizzle: 4096, TFLOPS: 44.51 (+6.17%)
 out_tf32(...+stage3+dsmem+swizzle): ['-24.539142', '26.0933208'], time:13.40752ms, swizzle: 4096, TFLOPS: 41.00
 out_tf32(...+stage2+dsmem+swizzle): ['-24.539142', '26.0933208'], time:12.35129ms, swizzle: 4096, TFLOPS: 44.51
              out_tf32(cublas+tf32): ['-24.539142', '26.0933208'], time:10.01133ms, swizzle: NOOP, TFLOPS: 54.91 (+23.36%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=8192
                     out_f32(naive): ['47.3072891', '96.7974395'], time:551.3394ms, swizzle: NOOP, TFLOPS: 1.99  (+0.00%)
                  out_f32x4(t8x8sk): ['47.3072891', '96.7974395'], time:41.22277ms, swizzle: NOOP, TFLOPS: 26.67 (+1237.46%)
                 out_f32x4(t8x8bcf): ['47.3072891', '96.7974395'], time:39.89914ms, swizzle: NOOP, TFLOPS: 27.56 (+3.32%)
                out_f32x4(t8x8dbuf): ['47.3072891', '96.7974395'], time:40.29097ms, swizzle: NOOP, TFLOPS: 27.29
                    out_f32(cublas): ['47.3072891', '96.7974395'], time:29.63916ms, swizzle: NOOP, TFLOPS: 37.10 (+34.62%)
                         out_f32_th: ['47.3072891', '96.7974395'], time:29.41981ms, swizzle: NOOP, TFLOPS: 37.37 (+0.75%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['47.3002510', '96.8180007'], time:30.10461ms, swizzle: NOOP, TFLOPS: 36.52
    out_tf32(mma2x4+warp2x4+stage2): ['47.3002510', '96.8180007'], time:26.53632ms, swizzle: NOOP, TFLOPS: 41.43 (+10.87%)
  out_tf32(mma2x4+...+stage3+dsmem): ['47.3002510', '96.8180007'], time:27.68478ms, swizzle: NOOP, TFLOPS: 39.72
  out_tf32(mma2x4+...+stage2+dsmem): ['47.3002510', '96.8180007'], time:26.57709ms, swizzle: NOOP, TFLOPS: 41.37
out_tf32(mma2x4+...+stage3+swizzle): ['47.3002510', '96.8180007'], time:30.01861ms, swizzle: 4096, TFLOPS: 36.63
out_tf32(mma2x4+...+stage2+swizzle): ['47.3002510', '96.8180007'], time:25.24836ms, swizzle: 4096, TFLOPS: 43.55 (+5.10%)
 out_tf32(...+stage3+dsmem+swizzle): ['47.3002510', '96.8180007'], time:27.08832ms, swizzle: 4096, TFLOPS: 40.59
 out_tf32(...+stage2+dsmem+swizzle): ['47.3002510', '96.8180007'], time:25.11584ms, swizzle: 4096, TFLOPS: 43.78 (+0.53%)
              out_tf32(cublas+tf32): ['47.3002510', '96.8180007'], time:19.69352ms, swizzle: NOOP, TFLOPS: 55.83 (+27.53%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=2048
                     out_f32(naive): ['-17.849985', '0.19760081'], time:40.67556ms, swizzle: NOOP, TFLOPS: 3.38  (+0.00%)
                  out_f32x4(t8x8sk): ['-17.849985', '0.19760081'], time:4.759192ms, swizzle: NOOP, TFLOPS: 28.88 (+754.67%)
                 out_f32x4(t8x8bcf): ['-17.849985', '0.19760081'], time:4.249489ms, swizzle: NOOP, TFLOPS: 32.34 (+11.99%)
                out_f32x4(t8x8dbuf): ['-17.849985', '0.19760081'], time:3.854548ms, swizzle: NOOP, TFLOPS: 35.66 (+10.25%)
                    out_f32(cublas): ['-17.849985', '0.19760081'], time:4.017460ms, swizzle: NOOP, TFLOPS: 34.21
                         out_f32_th: ['-17.849985', '0.19760081'], time:3.689861ms, swizzle: NOOP, TFLOPS: 37.25 (+4.46%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-17.849796', '0.19758633'], time:4.017901ms, swizzle: NOOP, TFLOPS: 34.21
    out_tf32(mma2x4+warp2x4+stage2): ['-17.849796', '0.19758633'], time:3.204596ms, swizzle: NOOP, TFLOPS: 42.89 (+15.14%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-17.849796', '0.19758633'], time:3.486108ms, swizzle: NOOP, TFLOPS: 39.42
  out_tf32(mma2x4+...+stage2+dsmem): ['-17.849796', '0.19758633'], time:3.196144ms, swizzle: NOOP, TFLOPS: 43.00 (+0.26%)
out_tf32(mma2x4+...+stage3+swizzle): ['-17.849796', '0.19758633'], time:3.813004ms, swizzle: 1024, TFLOPS: 36.04
out_tf32(mma2x4+...+stage2+swizzle): ['-17.849796', '0.19758633'], time:3.110313ms, swizzle: 1024, TFLOPS: 44.19 (+2.76%)
 out_tf32(...+stage3+dsmem+swizzle): ['-17.849796', '0.19758633'], time:3.408885ms, swizzle: 1024, TFLOPS: 40.32
 out_tf32(...+stage2+dsmem+swizzle): ['-17.849796', '0.19758633'], time:3.111791ms, swizzle: 1024, TFLOPS: 44.17
              out_tf32(cublas+tf32): ['-17.849796', '0.19758633'], time:2.788853ms, swizzle: NOOP, TFLOPS: 49.28 (+11.53%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=4096
                     out_f32(naive): ['-24.539411', '26.0933971'], time:81.24710ms, swizzle: NOOP, TFLOPS: 3.38  (+0.00%)
                  out_f32x4(t8x8sk): ['-24.539411', '26.0933971'], time:9.419822ms, swizzle: NOOP, TFLOPS: 29.18 (+762.51%)
                 out_f32x4(t8x8bcf): ['-24.539411', '26.0933971'], time:8.507907ms, swizzle: NOOP, TFLOPS: 32.31 (+10.72%)
                out_f32x4(t8x8dbuf): ['-24.539411', '26.0933971'], time:7.820093ms, swizzle: NOOP, TFLOPS: 35.15 (+8.80%)
                    out_f32(cublas): ['-24.539411', '26.0933971'], time:7.591652ms, swizzle: NOOP, TFLOPS: 36.21 (+3.01%)
                         out_f32_th: ['-24.539411', '26.0933971'], time:7.503700ms, swizzle: NOOP, TFLOPS: 36.63 (+1.17%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-24.539142', '26.0933208'], time:7.938122ms, swizzle: NOOP, TFLOPS: 34.63
    out_tf32(mma2x4+warp2x4+stage2): ['-24.539142', '26.0933208'], time:6.458127ms, swizzle: NOOP, TFLOPS: 42.56 (+16.19%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-24.539142', '26.0933208'], time:6.996679ms, swizzle: NOOP, TFLOPS: 39.29
  out_tf32(mma2x4+...+stage2+dsmem): ['-24.539142', '26.0933208'], time:6.418275ms, swizzle: NOOP, TFLOPS: 42.83 (+0.62%)
out_tf32(mma2x4+...+stage3+swizzle): ['-24.539142', '26.0933208'], time:7.730925ms, swizzle: 1024, TFLOPS: 35.56
out_tf32(mma2x4+...+stage2+swizzle): ['-24.539142', '26.0933208'], time:6.363022ms, swizzle: 1024, TFLOPS: 43.20 (+0.87%)
 out_tf32(...+stage3+dsmem+swizzle): ['-24.539142', '26.0933208'], time:6.948149ms, swizzle: 1024, TFLOPS: 39.56
 out_tf32(...+stage2+dsmem+swizzle): ['-24.539142', '26.0933208'], time:6.365275ms, swizzle: 1024, TFLOPS: 43.18
              out_tf32(cublas+tf32): ['-24.539142', '26.0933208'], time:5.335128ms, swizzle: NOOP, TFLOPS: 51.52 (+19.27%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=8192
                     out_f32(naive): ['47.2999496', '96.8197784'], time:247.6329ms, swizzle: NOOP, TFLOPS: 2.22  (+0.00%)
                  out_f32x4(t8x8sk): ['47.2999496', '96.8197784'], time:19.65559ms, swizzle: NOOP, TFLOPS: 27.97 (+1159.86%)
                 out_f32x4(t8x8bcf): ['47.2999496', '96.8197784'], time:17.52810ms, swizzle: NOOP, TFLOPS: 31.36 (+12.14%)
                out_f32x4(t8x8dbuf): ['47.2999496', '96.8197784'], time:18.90896ms, swizzle: NOOP, TFLOPS: 29.07
                    out_f32(cublas): ['47.2999496', '96.8197784'], time:15.03305ms, swizzle: NOOP, TFLOPS: 36.57 (+16.60%)
                         out_f32_th: ['47.2999496', '96.8197784'], time:14.72257ms, swizzle: NOOP, TFLOPS: 37.34 (+2.11%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['47.3002510', '96.8180007'], time:15.59674ms, swizzle: NOOP, TFLOPS: 35.25
    out_tf32(mma2x4+warp2x4+stage2): ['47.3002510', '96.8180007'], time:13.05602ms, swizzle: NOOP, TFLOPS: 42.11 (+12.76%)
  out_tf32(mma2x4+...+stage3+dsmem): ['47.3002510', '96.8180007'], time:13.85312ms, swizzle: NOOP, TFLOPS: 39.68
  out_tf32(mma2x4+...+stage2+dsmem): ['47.3002510', '96.8180007'], time:13.07342ms, swizzle: NOOP, TFLOPS: 42.05
out_tf32(mma2x4+...+stage3+swizzle): ['47.3002510', '96.8180007'], time:15.54510ms, swizzle: 1024, TFLOPS: 35.37
out_tf32(mma2x4+...+stage2+swizzle): ['47.3002510', '96.8180007'], time:12.80124ms, swizzle: 1024, TFLOPS: 42.95 (+1.99%)
 out_tf32(...+stage3+dsmem+swizzle): ['47.3002510', '96.8180007'], time:13.91153ms, swizzle: 1024, TFLOPS: 39.52
 out_tf32(...+stage2+dsmem+swizzle): ['47.3002510', '96.8180007'], time:12.78195ms, swizzle: 1024, TFLOPS: 43.01 (+0.15%)
              out_tf32(cublas+tf32): ['47.3002510', '96.8180007'], time:10.32341ms, swizzle: NOOP, TFLOPS: 53.25 (+23.82%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=2048
                     out_f32(naive): ['-17.849985', '0.19760081'], time:81.26928ms, swizzle: NOOP, TFLOPS: 3.38  (+0.00%)
                  out_f32x4(t8x8sk): ['-17.849985', '0.19760081'], time:9.237766ms, swizzle: NOOP, TFLOPS: 29.76 (+779.75%)
                 out_f32x4(t8x8bcf): ['-17.849985', '0.19760081'], time:8.254611ms, swizzle: NOOP, TFLOPS: 33.30 (+11.91%)
                out_f32x4(t8x8dbuf): ['-17.849985', '0.19760081'], time:7.502532ms, swizzle: NOOP, TFLOPS: 36.64 (+10.02%)
                    out_f32(cublas): ['-17.849985', '0.19760081'], time:8.107531ms, swizzle: NOOP, TFLOPS: 33.90
                         out_f32_th: ['-17.849985', '0.19760081'], time:7.478880ms, swizzle: NOOP, TFLOPS: 36.75 (+0.32%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-17.849796', '0.19758633'], time:7.743370ms, swizzle: NOOP, TFLOPS: 35.50
    out_tf32(mma2x4+warp2x4+stage2): ['-17.849796', '0.19758633'], time:6.154835ms, swizzle: NOOP, TFLOPS: 44.66 (+21.51%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-17.849796', '0.19758633'], time:6.668007ms, swizzle: NOOP, TFLOPS: 41.22
  out_tf32(mma2x4+...+stage2+dsmem): ['-17.849796', '0.19758633'], time:6.114292ms, swizzle: NOOP, TFLOPS: 44.96 (+0.66%)
out_tf32(mma2x4+...+stage3+swizzle): ['-17.849796', '0.19758633'], time:7.382285ms, swizzle: 2048, TFLOPS: 37.23
out_tf32(mma2x4+...+stage2+swizzle): ['-17.849796', '0.19758633'], time:6.063973ms, swizzle: 2048, TFLOPS: 45.33 (+0.83%)
 out_tf32(...+stage3+dsmem+swizzle): ['-17.849796', '0.19758633'], time:6.617772ms, swizzle: 2048, TFLOPS: 41.54
 out_tf32(...+stage2+dsmem+swizzle): ['-17.849796', '0.19758633'], time:6.061935ms, swizzle: 2048, TFLOPS: 45.34 (+0.03%)
              out_tf32(cublas+tf32): ['-17.849796', '0.19758633'], time:5.143392ms, swizzle: NOOP, TFLOPS: 53.44 (+17.86%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=4096
                     out_f32(naive): ['-24.539411', '26.0933971'], time:268.3647ms, swizzle: NOOP, TFLOPS: 2.05  (+0.00%)
                  out_f32x4(t8x8sk): ['-24.539411', '26.0933971'], time:19.59303ms, swizzle: NOOP, TFLOPS: 28.06 (+1269.69%)
                 out_f32x4(t8x8bcf): ['-24.539411', '26.0933971'], time:17.70466ms, swizzle: NOOP, TFLOPS: 31.05 (+10.67%)
                out_f32x4(t8x8dbuf): ['-24.539411', '26.0933971'], time:19.52338ms, swizzle: NOOP, TFLOPS: 28.16
                    out_f32(cublas): ['-24.539411', '26.0933971'], time:14.43643ms, swizzle: NOOP, TFLOPS: 38.08 (+22.64%)
                         out_f32_th: ['-24.539411', '26.0933971'], time:14.19519ms, swizzle: NOOP, TFLOPS: 38.73 (+1.70%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-24.539142', '26.0933208'], time:14.91366ms, swizzle: NOOP, TFLOPS: 36.86
    out_tf32(mma2x4+warp2x4+stage2): ['-24.539142', '26.0933208'], time:12.80041ms, swizzle: NOOP, TFLOPS: 42.95 (+10.90%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-24.539142', '26.0933208'], time:13.52418ms, swizzle: NOOP, TFLOPS: 40.65
  out_tf32(mma2x4+...+stage2+dsmem): ['-24.539142', '26.0933208'], time:12.89166ms, swizzle: NOOP, TFLOPS: 42.64
out_tf32(mma2x4+...+stage3+swizzle): ['-24.539142', '26.0933208'], time:14.86917ms, swizzle: 2048, TFLOPS: 36.97
out_tf32(mma2x4+...+stage2+swizzle): ['-24.539142', '26.0933208'], time:12.12794ms, swizzle: 2048, TFLOPS: 45.33 (+5.54%)
 out_tf32(...+stage3+dsmem+swizzle): ['-24.539142', '26.0933208'], time:13.22101ms, swizzle: 2048, TFLOPS: 41.58
 out_tf32(...+stage2+dsmem+swizzle): ['-24.539142', '26.0933208'], time:12.12646ms, swizzle: 2048, TFLOPS: 45.34 (+0.01%)
              out_tf32(cublas+tf32): ['-24.539142', '26.0933208'], time:9.958040ms, swizzle: NOOP, TFLOPS: 55.21 (+21.78%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=8192
                     out_f32(naive): ['47.2999496', '96.8197784'], time:550.5259ms, swizzle: NOOP, TFLOPS: 2.00  (+0.00%)
                  out_f32x4(t8x8sk): ['47.2999496', '96.8197784'], time:39.90060ms, swizzle: NOOP, TFLOPS: 27.56 (+1279.74%)
                 out_f32x4(t8x8bcf): ['47.2999496', '96.8197784'], time:36.95698ms, swizzle: NOOP, TFLOPS: 29.75 (+7.96%)
                out_f32x4(t8x8dbuf): ['47.2999496', '96.8197784'], time:38.06703ms, swizzle: NOOP, TFLOPS: 28.88
                    out_f32(cublas): ['47.2999496', '96.8197784'], time:28.85241ms, swizzle: NOOP, TFLOPS: 38.11 (+28.09%)
                         out_f32_th: ['47.2999496', '96.8197784'], time:29.13621ms, swizzle: NOOP, TFLOPS: 37.74
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['47.3002510', '96.8180007'], time:29.58662ms, swizzle: NOOP, TFLOPS: 37.16
    out_tf32(mma2x4+warp2x4+stage2): ['47.3002510', '96.8180007'], time:25.65428ms, swizzle: NOOP, TFLOPS: 42.86 (+12.47%)
  out_tf32(mma2x4+...+stage3+dsmem): ['47.3002510', '96.8180007'], time:27.24572ms, swizzle: NOOP, TFLOPS: 40.36
  out_tf32(mma2x4+...+stage2+dsmem): ['47.3002510', '96.8180007'], time:25.65881ms, swizzle: NOOP, TFLOPS: 42.85
out_tf32(mma2x4+...+stage3+swizzle): ['47.3002510', '96.8180007'], time:29.62752ms, swizzle: 2048, TFLOPS: 37.11
out_tf32(mma2x4+...+stage2+swizzle): ['47.3002510', '96.8180007'], time:24.43090ms, swizzle: 2048, TFLOPS: 45.00 (+5.01%)
 out_tf32(...+stage3+dsmem+swizzle): ['47.3002510', '96.8180007'], time:26.46713ms, swizzle: 2048, TFLOPS: 41.54
 out_tf32(...+stage2+dsmem+swizzle): ['47.3002510', '96.8180007'], time:24.40419ms, swizzle: 2048, TFLOPS: 45.05 (+0.11%)
              out_tf32(cublas+tf32): ['47.3002510', '96.8180007'], time:19.60293ms, swizzle: NOOP, TFLOPS: 56.09 (+24.49%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=2048
                     out_f32(naive): ['-17.849985', '0.19760081'], time:276.2140ms, swizzle: NOOP, TFLOPS: 1.99  (+0.00%)
                  out_f32x4(t8x8sk): ['-17.849985', '0.19760081'], time:20.09291ms, swizzle: NOOP, TFLOPS: 27.36 (+1274.68%)
                 out_f32x4(t8x8bcf): ['-17.849985', '0.19760081'], time:18.52561ms, swizzle: NOOP, TFLOPS: 29.68 (+8.46%)
                out_f32x4(t8x8dbuf): ['-17.849985', '0.19760081'], time:19.83964ms, swizzle: NOOP, TFLOPS: 27.71
                    out_f32(cublas): ['-17.849985', '0.19760081'], time:14.76491ms, swizzle: NOOP, TFLOPS: 37.23 (+25.47%)
                         out_f32_th: ['-17.849985', '0.19760081'], time:14.15612ms, swizzle: NOOP, TFLOPS: 38.84 (+4.30%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-17.849796', '0.19758633'], time:14.75453ms, swizzle: NOOP, TFLOPS: 37.26
    out_tf32(mma2x4+warp2x4+stage2): ['-17.849796', '0.19758633'], time:13.11275ms, swizzle: NOOP, TFLOPS: 41.93 (+7.96%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-17.849796', '0.19758633'], time:13.43187ms, swizzle: NOOP, TFLOPS: 40.93
  out_tf32(mma2x4+...+stage2+dsmem): ['-17.849796', '0.19758633'], time:13.04426ms, swizzle: NOOP, TFLOPS: 42.15 (+0.53%)
out_tf32(mma2x4+...+stage3+swizzle): ['-17.849796', '0.19758633'], time:14.68685ms, swizzle: 4096, TFLOPS: 37.43
out_tf32(mma2x4+...+stage2+swizzle): ['-17.849796', '0.19758633'], time:11.95122ms, swizzle: 4096, TFLOPS: 46.00 (+9.15%)
 out_tf32(...+stage3+dsmem+swizzle): ['-17.849796', '0.19758633'], time:13.05289ms, swizzle: 4096, TFLOPS: 42.12
 out_tf32(...+stage2+dsmem+swizzle): ['-17.849796', '0.19758633'], time:11.98161ms, swizzle: 4096, TFLOPS: 45.88
              out_tf32(cublas+tf32): ['-17.849796', '0.19758633'], time:9.834122ms, swizzle: NOOP, TFLOPS: 55.90 (+21.53%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=4096
                     out_f32(naive): ['-24.539411', '26.0933971'], time:551.5206ms, swizzle: NOOP, TFLOPS: 1.99  (+0.00%)
                  out_f32x4(t8x8sk): ['-24.539411', '26.0933971'], time:40.85628ms, swizzle: NOOP, TFLOPS: 26.91 (+1249.90%)
                 out_f32x4(t8x8bcf): ['-24.539411', '26.0933971'], time:38.69991ms, swizzle: NOOP, TFLOPS: 28.41 (+5.57%)
                out_f32x4(t8x8dbuf): ['-24.539411', '26.0933971'], time:39.29961ms, swizzle: NOOP, TFLOPS: 27.98
                    out_f32(cublas): ['-24.539411', '26.0933971'], time:28.43469ms, swizzle: NOOP, TFLOPS: 38.67 (+36.10%)
                         out_f32_th: ['-24.539411', '26.0933971'], time:28.36043ms, swizzle: NOOP, TFLOPS: 38.77 (+0.26%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-24.539142', '26.0933208'], time:29.58275ms, swizzle: NOOP, TFLOPS: 37.17
    out_tf32(mma2x4+warp2x4+stage2): ['-24.539142', '26.0933208'], time:25.94059ms, swizzle: NOOP, TFLOPS: 42.39 (+9.33%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-24.539142', '26.0933208'], time:27.37162ms, swizzle: NOOP, TFLOPS: 40.17
  out_tf32(mma2x4+...+stage2+dsmem): ['-24.539142', '26.0933208'], time:25.87283ms, swizzle: NOOP, TFLOPS: 42.50 (+0.26%)
out_tf32(mma2x4+...+stage3+swizzle): ['-24.539142', '26.0933208'], time:29.19225ms, swizzle: 4096, TFLOPS: 37.66
out_tf32(mma2x4+...+stage2+swizzle): ['-24.539142', '26.0933208'], time:23.91030ms, swizzle: 4096, TFLOPS: 45.98 (+8.21%)
 out_tf32(...+stage3+dsmem+swizzle): ['-24.539142', '26.0933208'], time:25.98127ms, swizzle: 4096, TFLOPS: 42.32
 out_tf32(...+stage2+dsmem+swizzle): ['-24.539142', '26.0933208'], time:23.91967ms, swizzle: 4096, TFLOPS: 45.97
              out_tf32(cublas+tf32): ['-24.539142', '26.0933208'], time:19.25871ms, swizzle: NOOP, TFLOPS: 57.09 (+24.15%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=8192
                     out_f32(naive): ['47.2999496', '96.8197784'], time:1102.202ms, swizzle: NOOP, TFLOPS: 2.00  (+0.00%)
                  out_f32x4(t8x8sk): ['47.2999496', '96.8197784'], time:82.22703ms, swizzle: NOOP, TFLOPS: 26.74 (+1240.44%)
                 out_f32x4(t8x8bcf): ['47.2999496', '96.8197784'], time:77.98941ms, swizzle: NOOP, TFLOPS: 28.20 (+5.43%)
                out_f32x4(t8x8dbuf): ['47.2999496', '96.8197784'], time:78.90355ms, swizzle: NOOP, TFLOPS: 27.87
                    out_f32(cublas): ['47.2999496', '96.8197784'], time:58.00436ms, swizzle: NOOP, TFLOPS: 37.91 (+34.45%)
                         out_f32_th: ['47.2999496', '96.8197784'], time:58.51061ms, swizzle: NOOP, TFLOPS: 37.58
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['47.3002510', '96.8180007'], time:59.37192ms, swizzle: NOOP, TFLOPS: 37.04
    out_tf32(mma2x4+warp2x4+stage2): ['47.3002510', '96.8180007'], time:51.85855ms, swizzle: NOOP, TFLOPS: 42.40 (+11.85%)
  out_tf32(mma2x4+...+stage3+dsmem): ['47.3002510', '96.8180007'], time:55.02256ms, swizzle: NOOP, TFLOPS: 39.97
  out_tf32(mma2x4+...+stage2+dsmem): ['47.3002510', '96.8180007'], time:52.14641ms, swizzle: NOOP, TFLOPS: 42.17
out_tf32(mma2x4+...+stage3+swizzle): ['47.3002510', '96.8180007'], time:58.22221ms, swizzle: 4096, TFLOPS: 37.77
out_tf32(mma2x4+...+stage2+swizzle): ['47.3002510', '96.8180007'], time:49.53384ms, swizzle: 4096, TFLOPS: 44.39 (+4.69%)
 out_tf32(...+stage3+dsmem+swizzle): ['47.3002510', '96.8180007'], time:53.26046ms, swizzle: 4096, TFLOPS: 41.29
 out_tf32(...+stage2+dsmem+swizzle): ['47.3002510', '96.8180007'], time:49.65736ms, swizzle: 4096, TFLOPS: 44.28
              out_tf32(cublas+tf32): ['47.3002510', '96.8180007'], time:38.19205ms, swizzle: NOOP, TFLOPS: 57.58 (+29.70%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=2048
                     out_f32(naive): ['-17.849985', '0.19760081'], time:81.29155ms, swizzle: NOOP, TFLOPS: 3.38  (+0.00%)
                  out_f32x4(t8x8sk): ['-17.849985', '0.19760081'], time:9.530687ms, swizzle: NOOP, TFLOPS: 28.84 (+752.95%)
                 out_f32x4(t8x8bcf): ['-17.849985', '0.19760081'], time:8.524906ms, swizzle: NOOP, TFLOPS: 32.24 (+11.80%)
                out_f32x4(t8x8dbuf): ['-17.849985', '0.19760081'], time:8.015465ms, swizzle: NOOP, TFLOPS: 34.29 (+6.36%)
                    out_f32(cublas): ['-17.849985', '0.19760081'], time:8.247447ms, swizzle: NOOP, TFLOPS: 33.33
                         out_f32_th: ['-17.849985', '0.19760081'], time:7.800579ms, swizzle: NOOP, TFLOPS: 35.24 (+2.75%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-17.849796', '0.19758633'], time:7.802319ms, swizzle: NOOP, TFLOPS: 35.23
    out_tf32(mma2x4+warp2x4+stage2): ['-17.849796', '0.19758633'], time:6.260669ms, swizzle: NOOP, TFLOPS: 43.91 (+24.60%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-17.849796', '0.19758633'], time:6.773519ms, swizzle: NOOP, TFLOPS: 40.58
  out_tf32(mma2x4+...+stage2+dsmem): ['-17.849796', '0.19758633'], time:6.221926ms, swizzle: NOOP, TFLOPS: 44.18 (+0.62%)
out_tf32(mma2x4+...+stage3+swizzle): ['-17.849796', '0.19758633'], time:7.493686ms, swizzle: 1024, TFLOPS: 36.68
out_tf32(mma2x4+...+stage2+swizzle): ['-17.849796', '0.19758633'], time:6.252801ms, swizzle: 1024, TFLOPS: 43.96
 out_tf32(...+stage3+dsmem+swizzle): ['-17.849796', '0.19758633'], time:6.773734ms, swizzle: 1024, TFLOPS: 40.58
 out_tf32(...+stage2+dsmem+swizzle): ['-17.849796', '0.19758633'], time:6.259787ms, swizzle: 1024, TFLOPS: 43.91
              out_tf32(cublas+tf32): ['-17.849796', '0.19758633'], time:5.200731ms, swizzle: NOOP, TFLOPS: 52.85 (+19.64%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=4096
                     out_f32(naive): ['-24.539411', '26.0933971'], time:162.3996ms, swizzle: NOOP, TFLOPS: 3.39  (+0.00%)
                  out_f32x4(t8x8sk): ['-24.539411', '26.0933971'], time:18.98237ms, swizzle: NOOP, TFLOPS: 28.96 (+755.53%)
                 out_f32x4(t8x8bcf): ['-24.539411', '26.0933971'], time:17.01517ms, swizzle: NOOP, TFLOPS: 32.31 (+11.56%)
                out_f32x4(t8x8dbuf): ['-24.539411', '26.0933971'], time:17.42953ms, swizzle: NOOP, TFLOPS: 31.54
                    out_f32(cublas): ['-24.539411', '26.0933971'], time:14.61760ms, swizzle: NOOP, TFLOPS: 37.61 (+16.40%)
                         out_f32_th: ['-24.539411', '26.0933971'], time:14.48466ms, swizzle: NOOP, TFLOPS: 37.95 (+0.92%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-24.539142', '26.0933208'], time:15.16782ms, swizzle: NOOP, TFLOPS: 36.24
    out_tf32(mma2x4+warp2x4+stage2): ['-24.539142', '26.0933208'], time:12.41897ms, swizzle: NOOP, TFLOPS: 44.27 (+16.63%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-24.539142', '26.0933208'], time:13.44157ms, swizzle: NOOP, TFLOPS: 40.90
  out_tf32(mma2x4+...+stage2+dsmem): ['-24.539142', '26.0933208'], time:12.39446ms, swizzle: NOOP, TFLOPS: 44.35 (+0.20%)
out_tf32(mma2x4+...+stage3+swizzle): ['-24.539142', '26.0933208'], time:14.96917ms, swizzle: 1024, TFLOPS: 36.73
out_tf32(mma2x4+...+stage2+swizzle): ['-24.539142', '26.0933208'], time:12.42594ms, swizzle: 1024, TFLOPS: 44.24
 out_tf32(...+stage3+dsmem+swizzle): ['-24.539142', '26.0933208'], time:13.48555ms, swizzle: 1024, TFLOPS: 40.77
 out_tf32(...+stage2+dsmem+swizzle): ['-24.539142', '26.0933208'], time:12.44277ms, swizzle: 1024, TFLOPS: 44.18
              out_tf32(cublas+tf32): ['-24.539142', '26.0933208'], time:10.07603ms, swizzle: NOOP, TFLOPS: 54.56 (+23.01%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=8192
                     out_f32(naive): ['47.2999496', '96.8197784'], time:494.3340ms, swizzle: NOOP, TFLOPS: 2.22  (+0.00%)
                  out_f32x4(t8x8sk): ['47.2999496', '96.8197784'], time:39.43489ms, swizzle: NOOP, TFLOPS: 27.88 (+1153.54%)
                 out_f32x4(t8x8bcf): ['47.2999496', '96.8197784'], time:35.69089ms, swizzle: NOOP, TFLOPS: 30.81 (+10.49%)
                out_f32x4(t8x8dbuf): ['47.2999496', '96.8197784'], time:37.27245ms, swizzle: NOOP, TFLOPS: 29.50
                    out_f32(cublas): ['47.2999496', '96.8197784'], time:29.58321ms, swizzle: NOOP, TFLOPS: 37.17 (+20.65%)
                         out_f32_th: ['47.2999496', '96.8197784'], time:29.77937ms, swizzle: NOOP, TFLOPS: 36.92
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['47.3002510', '96.8180007'], time:30.08049ms, swizzle: NOOP, TFLOPS: 36.55
    out_tf32(mma2x4+warp2x4+stage2): ['47.3002510', '96.8180007'], time:25.29913ms, swizzle: NOOP, TFLOPS: 43.46 (+16.93%)
  out_tf32(mma2x4+...+stage3+dsmem): ['47.3002510', '96.8180007'], time:27.23990ms, swizzle: NOOP, TFLOPS: 40.36
  out_tf32(mma2x4+...+stage2+dsmem): ['47.3002510', '96.8180007'], time:25.82558ms, swizzle: NOOP, TFLOPS: 42.57
out_tf32(mma2x4+...+stage3+swizzle): ['47.3002510', '96.8180007'], time:30.02738ms, swizzle: 1024, TFLOPS: 36.62
out_tf32(mma2x4+...+stage2+swizzle): ['47.3002510', '96.8180007'], time:24.85141ms, swizzle: 1024, TFLOPS: 44.24 (+1.80%)
 out_tf32(...+stage3+dsmem+swizzle): ['47.3002510', '96.8180007'], time:26.89465ms, swizzle: 1024, TFLOPS: 40.88
 out_tf32(...+stage2+dsmem+swizzle): ['47.3002510', '96.8180007'], time:24.83090ms, swizzle: 1024, TFLOPS: 44.28 (+0.08%)
              out_tf32(cublas+tf32): ['47.3002510', '96.8180007'], time:19.69830ms, swizzle: NOOP, TFLOPS: 55.82 (+26.06%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=2048
                     out_f32(naive): ['-17.849985', '0.19760081'], time:162.4853ms, swizzle: NOOP, TFLOPS: 3.38  (+0.00%)
                  out_f32x4(t8x8sk): ['-17.849985', '0.19760081'], time:18.61557ms, swizzle: NOOP, TFLOPS: 29.53 (+772.85%)
                 out_f32x4(t8x8bcf): ['-17.849985', '0.19760081'], time:16.65081ms, swizzle: NOOP, TFLOPS: 33.02 (+11.80%)
                out_f32x4(t8x8dbuf): ['-17.849985', '0.19760081'], time:16.60894ms, swizzle: NOOP, TFLOPS: 33.10 (+0.25%)
                    out_f32(cublas): ['-17.849985', '0.19760081'], time:14.59673ms, swizzle: NOOP, TFLOPS: 37.66 (+13.79%)
                         out_f32_th: ['-17.849985', '0.19760081'], time:14.25113ms, swizzle: NOOP, TFLOPS: 38.58 (+2.43%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-17.849796', '0.19758633'], time:14.85270ms, swizzle: NOOP, TFLOPS: 37.01
    out_tf32(mma2x4+warp2x4+stage2): ['-17.849796', '0.19758633'], time:12.02559ms, swizzle: NOOP, TFLOPS: 45.72 (+18.51%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-17.849796', '0.19758633'], time:13.04501ms, swizzle: NOOP, TFLOPS: 42.14
  out_tf32(mma2x4+...+stage2+dsmem): ['-17.849796', '0.19758633'], time:12.00811ms, swizzle: NOOP, TFLOPS: 45.78 (+0.15%)
out_tf32(mma2x4+...+stage3+swizzle): ['-17.849796', '0.19758633'], time:14.59468ms, swizzle: 2048, TFLOPS: 37.67
out_tf32(mma2x4+...+stage2+swizzle): ['-17.849796', '0.19758633'], time:12.04820ms, swizzle: 2048, TFLOPS: 45.63
 out_tf32(...+stage3+dsmem+swizzle): ['-17.849796', '0.19758633'], time:13.12594ms, swizzle: 2048, TFLOPS: 41.88
 out_tf32(...+stage2+dsmem+swizzle): ['-17.849796', '0.19758633'], time:12.04621ms, swizzle: 2048, TFLOPS: 45.64
              out_tf32(cublas+tf32): ['-17.849796', '0.19758633'], time:9.895133ms, swizzle: NOOP, TFLOPS: 55.56 (+21.35%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=4096
                     out_f32(naive): ['-24.539411', '26.0933971'], time:536.4258ms, swizzle: NOOP, TFLOPS: 2.05  (+0.00%)
                  out_f32x4(t8x8sk): ['-24.539411', '26.0933971'], time:40.05281ms, swizzle: NOOP, TFLOPS: 27.45 (+1239.30%)
                 out_f32x4(t8x8bcf): ['-24.539411', '26.0933971'], time:35.56506ms, swizzle: NOOP, TFLOPS: 30.92 (+12.62%)
                out_f32x4(t8x8dbuf): ['-24.539411', '26.0933971'], time:38.55193ms, swizzle: NOOP, TFLOPS: 28.52
                    out_f32(cublas): ['-24.539411', '26.0933971'], time:28.43211ms, swizzle: NOOP, TFLOPS: 38.67 (+25.09%)
                         out_f32_th: ['-24.539411', '26.0933971'], time:28.49795ms, swizzle: NOOP, TFLOPS: 38.58
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-24.539142', '26.0933208'], time:29.34467ms, swizzle: NOOP, TFLOPS: 37.47
    out_tf32(mma2x4+warp2x4+stage2): ['-24.539142', '26.0933208'], time:25.17921ms, swizzle: NOOP, TFLOPS: 43.67 (+12.92%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-24.539142', '26.0933208'], time:26.83230ms, swizzle: NOOP, TFLOPS: 40.98
  out_tf32(mma2x4+...+stage2+dsmem): ['-24.539142', '26.0933208'], time:25.15255ms, swizzle: NOOP, TFLOPS: 43.71 (+0.11%)
out_tf32(mma2x4+...+stage3+swizzle): ['-24.539142', '26.0933208'], time:29.20446ms, swizzle: 2048, TFLOPS: 37.65
out_tf32(mma2x4+...+stage2+swizzle): ['-24.539142', '26.0933208'], time:23.84365ms, swizzle: 2048, TFLOPS: 46.11 (+5.49%)
 out_tf32(...+stage3+dsmem+swizzle): ['-24.539142', '26.0933208'], time:26.03495ms, swizzle: 2048, TFLOPS: 42.23
 out_tf32(...+stage2+dsmem+swizzle): ['-24.539142', '26.0933208'], time:23.87169ms, swizzle: 2048, TFLOPS: 46.06
              out_tf32(cublas+tf32): ['-24.539142', '26.0933208'], time:19.30768ms, swizzle: NOOP, TFLOPS: 56.95 (+23.49%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=8192
                     out_f32(naive): ['47.2999496', '96.8197784'], time:1100.691ms, swizzle: NOOP, TFLOPS: 2.00  (+0.00%)
                  out_f32x4(t8x8sk): ['47.2999496', '96.8197784'], time:79.86506ms, swizzle: NOOP, TFLOPS: 27.53 (+1278.19%)
                 out_f32x4(t8x8bcf): ['47.2999496', '96.8197784'], time:74.10305ms, swizzle: NOOP, TFLOPS: 29.68 (+7.78%)
                out_f32x4(t8x8dbuf): ['47.2999496', '96.8197784'], time:74.76978ms, swizzle: NOOP, TFLOPS: 29.41
                    out_f32(cublas): ['47.2999496', '96.8197784'], time:57.91260ms, swizzle: NOOP, TFLOPS: 37.97 (+27.96%)
                         out_f32_th: ['47.2999496', '96.8197784'], time:58.26066ms, swizzle: NOOP, TFLOPS: 37.74
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['47.3002510', '96.8180007'], time:58.64821ms, swizzle: NOOP, TFLOPS: 37.50
    out_tf32(mma2x4+warp2x4+stage2): ['47.3002510', '96.8180007'], time:49.91295ms, swizzle: NOOP, TFLOPS: 44.06 (+16.03%)
  out_tf32(mma2x4+...+stage3+dsmem): ['47.3002510', '96.8180007'], time:54.05123ms, swizzle: NOOP, TFLOPS: 40.68
  out_tf32(mma2x4+...+stage2+dsmem): ['47.3002510', '96.8180007'], time:50.64512ms, swizzle: NOOP, TFLOPS: 43.42
out_tf32(mma2x4+...+stage3+swizzle): ['47.3002510', '96.8180007'], time:58.26839ms, swizzle: 2048, TFLOPS: 37.74
out_tf32(mma2x4+...+stage2+swizzle): ['47.3002510', '96.8180007'], time:48.46489ms, swizzle: 2048, TFLOPS: 45.37 (+2.99%)
 out_tf32(...+stage3+dsmem+swizzle): ['47.3002510', '96.8180007'], time:51.88893ms, swizzle: 2048, TFLOPS: 42.38
 out_tf32(...+stage2+dsmem+swizzle): ['47.3002510', '96.8180007'], time:48.42858ms, swizzle: 2048, TFLOPS: 45.41 (+0.07%)
              out_tf32(cublas+tf32): ['47.3002510', '96.8180007'], time:38.56168ms, swizzle: NOOP, TFLOPS: 57.03 (+25.59%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=2048
                     out_f32(naive): ['-17.849985', '0.19760081'], time:552.1761ms, swizzle: NOOP, TFLOPS: 1.99  (+0.00%)
                  out_f32x4(t8x8sk): ['-17.849985', '0.19760081'], time:41.13322ms, swizzle: NOOP, TFLOPS: 26.73 (+1242.41%)
                 out_f32x4(t8x8bcf): ['-17.849985', '0.19760081'], time:39.94113ms, swizzle: NOOP, TFLOPS: 27.53 (+2.98%)
                out_f32x4(t8x8dbuf): ['-17.849985', '0.19760081'], time:39.03629ms, swizzle: NOOP, TFLOPS: 28.17 (+2.32%)
                    out_f32(cublas): ['-17.849985', '0.19760081'], time:29.52983ms, swizzle: NOOP, TFLOPS: 37.23 (+32.19%)
                         out_f32_th: ['-17.849985', '0.19760081'], time:29.55764ms, swizzle: NOOP, TFLOPS: 37.20
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-17.849796', '0.19758633'], time:29.32828ms, swizzle: NOOP, TFLOPS: 37.49 (+0.69%)
    out_tf32(mma2x4+warp2x4+stage2): ['-17.849796', '0.19758633'], time:26.04321ms, swizzle: NOOP, TFLOPS: 42.22 (+12.61%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-17.849796', '0.19758633'], time:27.00719ms, swizzle: NOOP, TFLOPS: 40.71
  out_tf32(mma2x4+...+stage2+dsmem): ['-17.849796', '0.19758633'], time:26.04371ms, swizzle: NOOP, TFLOPS: 42.22
out_tf32(mma2x4+...+stage3+swizzle): ['-17.849796', '0.19758633'], time:28.73388ms, swizzle: 4096, TFLOPS: 38.27
out_tf32(mma2x4+...+stage2+swizzle): ['-17.849796', '0.19758633'], time:23.58162ms, swizzle: 4096, TFLOPS: 46.63 (+10.44%)
 out_tf32(...+stage3+dsmem+swizzle): ['-17.849796', '0.19758633'], time:25.63526ms, swizzle: 4096, TFLOPS: 42.89
 out_tf32(...+stage2+dsmem+swizzle): ['-17.849796', '0.19758633'], time:23.56603ms, swizzle: 4096, TFLOPS: 46.66 (+0.07%)
              out_tf32(cublas+tf32): ['-17.849796', '0.19758633'], time:19.51431ms, swizzle: NOOP, TFLOPS: 56.34 (+20.76%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=4096
                     out_f32(naive): ['-24.539411', '26.0933971'], time:1103.022ms, swizzle: NOOP, TFLOPS: 1.99  (+0.00%)
                  out_f32x4(t8x8sk): ['-24.539411', '26.0933971'], time:83.92670ms, swizzle: NOOP, TFLOPS: 26.20 (+1214.27%)
                 out_f32x4(t8x8bcf): ['-24.539411', '26.0933971'], time:80.32040ms, swizzle: NOOP, TFLOPS: 27.38 (+4.49%)
                out_f32x4(t8x8dbuf): ['-24.539411', '26.0933971'], time:81.26325ms, swizzle: NOOP, TFLOPS: 27.06
                    out_f32(cublas): ['-24.539411', '26.0933971'], time:58.41444ms, swizzle: NOOP, TFLOPS: 37.65 (+37.50%)
                         out_f32_th: ['-24.539411', '26.0933971'], time:58.71068ms, swizzle: NOOP, TFLOPS: 37.46
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-24.539142', '26.0933208'], time:59.40877ms, swizzle: NOOP, TFLOPS: 37.02
    out_tf32(mma2x4+warp2x4+stage2): ['-24.539142', '26.0933208'], time:51.33775ms, swizzle: NOOP, TFLOPS: 42.83 (+13.78%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-24.539142', '26.0933208'], time:54.76785ms, swizzle: NOOP, TFLOPS: 40.15
  out_tf32(mma2x4+...+stage2+dsmem): ['-24.539142', '26.0933208'], time:51.49431ms, swizzle: NOOP, TFLOPS: 42.70
out_tf32(mma2x4+...+stage3+swizzle): ['-24.539142', '26.0933208'], time:57.20764ms, swizzle: 4096, TFLOPS: 38.44
out_tf32(mma2x4+...+stage2+swizzle): ['-24.539142', '26.0933208'], time:47.47045ms, swizzle: 4096, TFLOPS: 46.32 (+8.15%)
 out_tf32(...+stage3+dsmem+swizzle): ['-24.539142', '26.0933208'], time:50.96282ms, swizzle: 4096, TFLOPS: 43.15
 out_tf32(...+stage2+dsmem+swizzle): ['-24.539142', '26.0933208'], time:47.44813ms, swizzle: 4096, TFLOPS: 46.35 (+0.05%)
              out_tf32(cublas+tf32): ['-24.539142', '26.0933208'], time:38.30858ms, swizzle: NOOP, TFLOPS: 57.40 (+23.86%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=8192
                     out_f32(naive): ['47.2999496', '96.8197784'], time:2203.968ms, swizzle: NOOP, TFLOPS: 2.00  (+0.00%)
                  out_f32x4(t8x8sk): ['47.2999496', '96.8197784'], time:164.6066ms, swizzle: NOOP, TFLOPS: 26.72 (+1238.93%)
                 out_f32x4(t8x8bcf): ['47.2999496', '96.8197784'], time:156.5503ms, swizzle: NOOP, TFLOPS: 28.09 (+5.15%)
                out_f32x4(t8x8dbuf): ['47.2999496', '96.8197784'], time:157.3980ms, swizzle: NOOP, TFLOPS: 27.94
                    out_f32(cublas): ['47.2999496', '96.8197784'], time:115.1302ms, swizzle: NOOP, TFLOPS: 38.20 (+35.98%)
                         out_f32_th: ['47.2999496', '96.8197784'], time:115.7525ms, swizzle: NOOP, TFLOPS: 38.00
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['47.3002510', '96.8180007'], time:117.6291ms, swizzle: NOOP, TFLOPS: 37.39
    out_tf32(mma2x4+warp2x4+stage2): ['47.3002510', '96.8180007'], time:102.4631ms, swizzle: NOOP, TFLOPS: 42.92 (+12.36%)
  out_tf32(mma2x4+...+stage3+dsmem): ['47.3002510', '96.8180007'], time:108.2939ms, swizzle: NOOP, TFLOPS: 40.61
  out_tf32(mma2x4+...+stage2+dsmem): ['47.3002510', '96.8180007'], time:104.3870ms, swizzle: NOOP, TFLOPS: 42.13
out_tf32(mma2x4+...+stage3+swizzle): ['47.3002510', '96.8180007'], time:114.1043ms, swizzle: 4096, TFLOPS: 38.54
out_tf32(mma2x4+...+stage2+swizzle): ['47.3002510', '96.8180007'], time:98.40396ms, swizzle: 4096, TFLOPS: 44.69 (+4.13%)
 out_tf32(...+stage3+dsmem+swizzle): ['47.3002510', '96.8180007'], time:106.3529ms, swizzle: 4096, TFLOPS: 41.35
 out_tf32(...+stage2+dsmem+swizzle): ['47.3002510', '96.8180007'], time:97.71902ms, swizzle: 4096, TFLOPS: 45.01 (+0.70%)
              out_tf32(cublas+tf32): ['47.3002510', '96.8180007'], time:75.97206ms, swizzle: NOOP, TFLOPS: 57.89 (+28.62%)
----------------------------------------------------------------------------------------------------------------------------------
```
