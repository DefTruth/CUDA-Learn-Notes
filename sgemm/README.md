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
- [X] sgemm_wmma_m16n16k8_mma4x2_warp2x4_stage2/3/4 (Tensor Cores, Tile MMA/Warp, Copy Async, Stage, Thread block swizzle)
- [X] PyTorch bindings

目前在L20上，CUDA Cores FP32的实现能达到cuBLAS大概90%~95%左右的性能(TFLOPS)，部分size下会超过cuBLAS。已知问题为bank conflicts没有完全消除，目前通过padding的方式缓解bank conflicts会导致shared memory浪费，也会影响SM occupancy。而Tensor Cores TF32的实现，只能达到cuBLAS TF32大概80%左右的性能，尚有较大差距。目前未手工实现Warp swizzle(受限于WMMA API的灵活性以及本人的能力)，后续将会尝试通过MMA PTX实现warp swizzle。

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
                                                       M=4096, N=8192, K=1024
                     out_f32(naive): ['-45.176471', '17.7125091'], time:20.41349ms, swizzle: NOOP, TFLOPS: 3.37  (+0.00%)
                  out_f32x4(t8x8sk): ['-45.176471', '17.7125091'], time:2.325522ms, swizzle: NOOP, TFLOPS: 29.55 (+777.80%)
                 out_f32x4(t8x8bcf): ['-45.176471', '17.7125091'], time:2.031362ms, swizzle: NOOP, TFLOPS: 33.83 (+14.48%)
                out_f32x4(t8x8dbuf): ['-45.176471', '17.7125091'], time:1.718842ms, swizzle: NOOP, TFLOPS: 39.98 (+18.18%)
                    out_f32(cublas): ['-45.176471', '17.7125091'], time:1.925849ms, swizzle: NOOP, TFLOPS: 35.68
                         out_f32_th: ['-45.176471', '17.7125091'], time:1.837718ms, swizzle: NOOP, TFLOPS: 37.39
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-45.176361', '17.7123947'], time:1.949667ms, swizzle: NOOP, TFLOPS: 35.25
    out_tf32(mma2x4+warp2x4+stage2): ['-45.176361', '17.7123947'], time:1.541721ms, swizzle: NOOP, TFLOPS: 44.57 (+11.49%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-45.176361', '17.7123947'], time:1.676261ms, swizzle: NOOP, TFLOPS: 41.00
  out_tf32(mma2x4+...+stage2+dsmem): ['-45.176361', '17.7123947'], time:1.525127ms, swizzle: NOOP, TFLOPS: 45.06 (+1.09%)
out_tf32(mma2x4+...+stage3+swizzle): ['-45.176361', '17.7123947'], time:1.883172ms, swizzle: 2048, TFLOPS: 36.49
out_tf32(mma2x4+...+stage2+swizzle): ['-45.176361', '17.7123947'], time:1.541399ms, swizzle: 2048, TFLOPS: 44.58
 out_tf32(...+stage3+dsmem+swizzle): ['-45.176361', '17.7123947'], time:1.697039ms, swizzle: 2048, TFLOPS: 40.49
 out_tf32(...+stage2+dsmem+swizzle): ['-45.176361', '17.7123947'], time:1.544988ms, swizzle: 2048, TFLOPS: 44.48
              out_tf32(cublas+tf32): ['-45.176361', '17.7123947'], time:1.331019ms, swizzle: NOOP, TFLOPS: 51.63 (+14.58%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=2048
                     out_f32(naive): ['-63.444438', '38.4295463'], time:40.65858ms, swizzle: NOOP, TFLOPS: 3.38  (+0.00%)
                  out_f32x4(t8x8sk): ['-63.444438', '38.4295463'], time:4.587244ms, swizzle: NOOP, TFLOPS: 29.96 (+786.34%)
                 out_f32x4(t8x8bcf): ['-63.444438', '38.4295463'], time:4.092347ms, swizzle: NOOP, TFLOPS: 33.58 (+12.09%)
                out_f32x4(t8x8dbuf): ['-63.444438', '38.4295463'], time:3.641486ms, swizzle: NOOP, TFLOPS: 37.74 (+12.38%)
                    out_f32(cublas): ['-63.444438', '38.4295463'], time:3.668510ms, swizzle: NOOP, TFLOPS: 37.46
                         out_f32_th: ['-63.444438', '38.4295463'], time:3.594660ms, swizzle: NOOP, TFLOPS: 38.23 (+1.30%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-63.444118', '38.4292221'], time:3.965806ms, swizzle: NOOP, TFLOPS: 34.66
    out_tf32(mma2x4+warp2x4+stage2): ['-63.444118', '38.4292221'], time:3.167927ms, swizzle: NOOP, TFLOPS: 43.38 (+13.47%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-63.444118', '38.4292221'], time:3.450572ms, swizzle: NOOP, TFLOPS: 39.83
  out_tf32(mma2x4+...+stage2+dsmem): ['-63.444118', '38.4292221'], time:3.164565ms, swizzle: NOOP, TFLOPS: 43.43 (+0.11%)
out_tf32(mma2x4+...+stage3+swizzle): ['-63.444118', '38.4292221'], time:3.796541ms, swizzle: 2048, TFLOPS: 36.20
out_tf32(mma2x4+...+stage2+swizzle): ['-63.444118', '38.4292221'], time:3.100752ms, swizzle: 2048, TFLOPS: 44.32 (+2.06%)
 out_tf32(...+stage3+dsmem+swizzle): ['-63.444118', '38.4292221'], time:3.401374ms, swizzle: 2048, TFLOPS: 40.41
 out_tf32(...+stage2+dsmem+swizzle): ['-63.444118', '38.4292221'], time:3.104865ms, swizzle: 2048, TFLOPS: 44.27
              out_tf32(cublas+tf32): ['-63.444118', '38.4292221'], time:2.581250ms, swizzle: NOOP, TFLOPS: 53.25 (+20.13%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=4096
                     out_f32(naive): ['-53.302734', '81.0102462'], time:134.4451ms, swizzle: NOOP, TFLOPS: 2.04  (+0.00%)
                  out_f32x4(t8x8sk): ['-53.302734', '81.0102462'], time:9.697830ms, swizzle: NOOP, TFLOPS: 28.34 (+1286.34%)
                 out_f32x4(t8x8bcf): ['-53.302734', '81.0102462'], time:8.738136ms, swizzle: NOOP, TFLOPS: 31.46 (+10.98%)
                out_f32x4(t8x8dbuf): ['-53.302734', '81.0102462'], time:8.403766ms, swizzle: NOOP, TFLOPS: 32.71 (+3.98%)
                    out_f32(cublas): ['-53.302734', '81.0102462'], time:7.346343ms, swizzle: NOOP, TFLOPS: 37.42 (+14.39%)
                         out_f32_th: ['-53.302734', '81.0102462'], time:7.214975ms, swizzle: NOOP, TFLOPS: 38.10 (+1.82%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-53.301563', '81.0088272'], time:7.857871ms, swizzle: NOOP, TFLOPS: 34.98
    out_tf32(mma2x4+warp2x4+stage2): ['-53.301563', '81.0088272'], time:6.543970ms, swizzle: NOOP, TFLOPS: 42.00 (+10.25%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-53.301563', '81.0088272'], time:6.888926ms, swizzle: NOOP, TFLOPS: 39.90
  out_tf32(mma2x4+...+stage2+dsmem): ['-53.301563', '81.0088272'], time:6.524801ms, swizzle: NOOP, TFLOPS: 42.13 (+0.29%)
out_tf32(mma2x4+...+stage3+swizzle): ['-53.301563', '81.0088272'], time:7.694256ms, swizzle: 2048, TFLOPS: 35.73
out_tf32(mma2x4+...+stage2+swizzle): ['-53.301563', '81.0088272'], time:6.326651ms, swizzle: 2048, TFLOPS: 43.45 (+3.13%)
 out_tf32(...+stage3+dsmem+swizzle): ['-53.301563', '81.0088272'], time:6.912016ms, swizzle: 2048, TFLOPS: 39.77
 out_tf32(...+stage2+dsmem+swizzle): ['-53.301563', '81.0088272'], time:6.328773ms, swizzle: 2048, TFLOPS: 43.43
              out_tf32(cublas+tf32): ['-53.301563', '81.0088272'], time:5.089497ms, swizzle: NOOP, TFLOPS: 54.01 (+24.31%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=1024, K=512
                     out_f32(naive): ['-38.139781', '21.1172447'], time:2.568578ms, swizzle: NOOP, TFLOPS: 3.34  (+0.00%)
                  out_f32x4(t8x8sk): ['-38.139781', '21.1172447'], time:0.320827ms, swizzle: NOOP, TFLOPS: 26.77 (+700.61%)
                 out_f32x4(t8x8bcf): ['-38.139781', '21.1172447'], time:0.265586ms, swizzle: NOOP, TFLOPS: 32.34 (+20.80%)
                out_f32x4(t8x8dbuf): ['-38.139781', '21.1172447'], time:0.228130ms, swizzle: NOOP, TFLOPS: 37.65 (+16.42%)
                    out_f32(cublas): ['-38.139781', '21.1172447'], time:0.279211ms, swizzle: NOOP, TFLOPS: 30.76
                         out_f32_th: ['-38.139781', '21.1172447'], time:0.259435ms, swizzle: NOOP, TFLOPS: 33.11
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-38.139759', '21.1171932'], time:0.258398ms, swizzle: NOOP, TFLOPS: 33.24
    out_tf32(mma2x4+warp2x4+stage2): ['-38.139759', '21.1171932'], time:0.216126ms, swizzle: NOOP, TFLOPS: 39.74 (+5.55%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-38.139759', '21.1171932'], time:0.241804ms, swizzle: NOOP, TFLOPS: 35.52
  out_tf32(mma2x4+...+stage2+dsmem): ['-38.139759', '21.1171932'], time:0.217437ms, swizzle: NOOP, TFLOPS: 39.51
out_tf32(mma2x4+...+stage3+swizzle): ['-38.139759', '21.1171932'], time:0.266206ms, swizzle: 256 , TFLOPS: 32.27
out_tf32(mma2x4+...+stage2+swizzle): ['-38.139759', '21.1171932'], time:0.213682ms, swizzle: 256 , TFLOPS: 40.20 (+1.14%)
 out_tf32(...+stage3+dsmem+swizzle): ['-38.139759', '21.1171932'], time:0.239634ms, swizzle: 256 , TFLOPS: 35.85
 out_tf32(...+stage2+dsmem+swizzle): ['-38.139759', '21.1171932'], time:0.214004ms, swizzle: 256 , TFLOPS: 40.14
              out_tf32(cublas+tf32): ['-38.139759', '21.1171932'], time:0.211155ms, swizzle: NOOP, TFLOPS: 40.68 (+1.20%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=1024, K=1024
                     out_f32(naive): ['-45.176471', '17.7125091'], time:5.143117ms, swizzle: NOOP, TFLOPS: 3.34  (+0.00%)
                  out_f32x4(t8x8sk): ['-45.176471', '17.7125091'], time:0.621640ms, swizzle: NOOP, TFLOPS: 27.64 (+727.35%)
                 out_f32x4(t8x8bcf): ['-45.176471', '17.7125091'], time:0.525009ms, swizzle: NOOP, TFLOPS: 32.72 (+18.41%)
                out_f32x4(t8x8dbuf): ['-45.176471', '17.7125091'], time:0.445735ms, swizzle: NOOP, TFLOPS: 38.54 (+17.79%)
                    out_f32(cublas): ['-45.176471', '17.7125091'], time:0.519812ms, swizzle: NOOP, TFLOPS: 33.05
                         out_f32_th: ['-45.176471', '17.7125091'], time:0.519371ms, swizzle: NOOP, TFLOPS: 33.08
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-45.176361', '17.7123947'], time:0.523340ms, swizzle: NOOP, TFLOPS: 32.83
    out_tf32(mma2x4+warp2x4+stage2): ['-45.176361', '17.7123947'], time:0.433433ms, swizzle: NOOP, TFLOPS: 39.64 (+2.84%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-45.176361', '17.7123947'], time:0.478589ms, swizzle: NOOP, TFLOPS: 35.90
  out_tf32(mma2x4+...+stage2+dsmem): ['-45.176361', '17.7123947'], time:0.422394ms, swizzle: NOOP, TFLOPS: 40.67 (+2.61%)
out_tf32(mma2x4+...+stage3+swizzle): ['-45.176361', '17.7123947'], time:0.516581ms, swizzle: 256 , TFLOPS: 33.26
out_tf32(mma2x4+...+stage2+swizzle): ['-45.176361', '17.7123947'], time:0.417721ms, swizzle: 256 , TFLOPS: 41.13 (+1.12%)
 out_tf32(...+stage3+dsmem+swizzle): ['-45.176361', '17.7123947'], time:0.464761ms, swizzle: 256 , TFLOPS: 36.96
 out_tf32(...+stage2+dsmem+swizzle): ['-45.176361', '17.7123947'], time:0.412452ms, swizzle: 256 , TFLOPS: 41.65 (+1.28%)
              out_tf32(cublas+tf32): ['-45.176361', '17.7123947'], time:0.352025ms, swizzle: NOOP, TFLOPS: 48.80 (+17.17%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=1024, K=2048
                     out_f32(naive): ['-63.444438', '38.4295463'], time:10.25099ms, swizzle: NOOP, TFLOPS: 3.35  (+0.00%)
                  out_f32x4(t8x8sk): ['-63.444438', '38.4295463'], time:1.252877ms, swizzle: NOOP, TFLOPS: 27.42 (+718.20%)
                 out_f32x4(t8x8bcf): ['-63.444438', '38.4295463'], time:1.111567ms, swizzle: NOOP, TFLOPS: 30.91 (+12.71%)
                out_f32x4(t8x8dbuf): ['-63.444438', '38.4295463'], time:0.930094ms, swizzle: NOOP, TFLOPS: 36.94 (+19.51%)
                    out_f32(cublas): ['-63.444450', '38.4295539'], time:0.992000ms, swizzle: NOOP, TFLOPS: 34.64
                         out_f32_th: ['-63.444450', '38.4295539'], time:0.966060ms, swizzle: NOOP, TFLOPS: 35.57
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-63.444118', '38.4292221'], time:1.119196ms, swizzle: NOOP, TFLOPS: 30.70
    out_tf32(mma2x4+warp2x4+stage2): ['-63.444118', '38.4292221'], time:0.890374ms, swizzle: NOOP, TFLOPS: 38.59 (+4.46%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-63.444118', '38.4292221'], time:0.972461ms, swizzle: NOOP, TFLOPS: 35.33
  out_tf32(mma2x4+...+stage2+dsmem): ['-63.444118', '38.4292221'], time:0.870144ms, swizzle: NOOP, TFLOPS: 39.49 (+2.32%)
out_tf32(mma2x4+...+stage3+swizzle): ['-63.444118', '38.4292221'], time:1.021003ms, swizzle: 256 , TFLOPS: 33.65
out_tf32(mma2x4+...+stage2+swizzle): ['-63.444118', '38.4292221'], time:0.827944ms, swizzle: 256 , TFLOPS: 41.50 (+5.10%)
 out_tf32(...+stage3+dsmem+swizzle): ['-63.444118', '38.4292221'], time:0.927448ms, swizzle: 256 , TFLOPS: 37.05
 out_tf32(...+stage2+dsmem+swizzle): ['-63.444118', '38.4292221'], time:0.825393ms, swizzle: 256 , TFLOPS: 41.63 (+0.31%)
              out_tf32(cublas+tf32): ['-63.444118', '38.4292221'], time:0.660848ms, swizzle: NOOP, TFLOPS: 51.99 (+24.90%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=1024, K=4096
                     out_f32(naive): ['-53.302734', '81.0102462'], time:20.42202ms, swizzle: NOOP, TFLOPS: 3.36  (+0.00%)
                  out_f32x4(t8x8sk): ['-53.302734', '81.0102462'], time:2.512419ms, swizzle: NOOP, TFLOPS: 27.35 (+712.84%)
                 out_f32x4(t8x8bcf): ['-53.302734', '81.0102462'], time:2.267622ms, swizzle: NOOP, TFLOPS: 30.30 (+10.80%)
                out_f32x4(t8x8dbuf): ['-53.302734', '81.0102462'], time:1.932835ms, swizzle: NOOP, TFLOPS: 35.55 (+17.32%)
                    out_f32(cublas): ['-53.302742', '81.0102233'], time:1.997232ms, swizzle: NOOP, TFLOPS: 34.41
                         out_f32_th: ['-53.302742', '81.0102233'], time:1.887249ms, swizzle: NOOP, TFLOPS: 36.41 (+2.42%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-53.301563', '81.0088272'], time:2.416503ms, swizzle: NOOP, TFLOPS: 28.44
    out_tf32(mma2x4+warp2x4+stage2): ['-53.301563', '81.0088272'], time:1.971995ms, swizzle: NOOP, TFLOPS: 34.85
  out_tf32(mma2x4+...+stage3+dsmem): ['-53.301563', '81.0088272'], time:2.134644ms, swizzle: NOOP, TFLOPS: 32.19
  out_tf32(mma2x4+...+stage2+dsmem): ['-53.301563', '81.0088272'], time:1.952743ms, swizzle: NOOP, TFLOPS: 35.19
out_tf32(mma2x4+...+stage3+swizzle): ['-53.301563', '81.0088272'], time:2.245724ms, swizzle: 256 , TFLOPS: 30.60
out_tf32(mma2x4+...+stage2+swizzle): ['-53.301563', '81.0088272'], time:1.889324ms, swizzle: 256 , TFLOPS: 36.37
 out_tf32(...+stage3+dsmem+swizzle): ['-53.301563', '81.0088272'], time:2.058339ms, swizzle: 256 , TFLOPS: 33.39
 out_tf32(...+stage2+dsmem+swizzle): ['-53.301563', '81.0088272'], time:1.884579ms, swizzle: 256 , TFLOPS: 36.46 (+0.14%)
              out_tf32(cublas+tf32): ['-53.301563', '81.0088272'], time:1.295614ms, swizzle: NOOP, TFLOPS: 53.04 (+45.46%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=2048, K=512
                     out_f32(naive): ['-38.139781', '21.1172447'], time:5.130171ms, swizzle: NOOP, TFLOPS: 3.35  (+0.00%)
                  out_f32x4(t8x8sk): ['-38.139781', '21.1172447'], time:0.624418ms, swizzle: NOOP, TFLOPS: 27.51 (+721.59%)
                 out_f32x4(t8x8bcf): ['-38.139781', '21.1172447'], time:0.527155ms, swizzle: NOOP, TFLOPS: 32.59 (+18.45%)
                out_f32x4(t8x8dbuf): ['-38.139781', '21.1172447'], time:0.452876ms, swizzle: NOOP, TFLOPS: 37.94 (+16.40%)
                    out_f32(cublas): ['-38.139781', '21.1172447'], time:0.455439ms, swizzle: NOOP, TFLOPS: 37.72
                         out_f32_th: ['-38.139781', '21.1172447'], time:0.445389ms, swizzle: NOOP, TFLOPS: 38.57 (+1.68%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-38.139759', '21.1171932'], time:0.492978ms, swizzle: NOOP, TFLOPS: 34.85
    out_tf32(mma2x4+warp2x4+stage2): ['-38.139759', '21.1171932'], time:0.407230ms, swizzle: NOOP, TFLOPS: 42.19 (+9.37%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-38.139759', '21.1171932'], time:0.444924ms, swizzle: NOOP, TFLOPS: 38.61
  out_tf32(mma2x4+...+stage2+dsmem): ['-38.139759', '21.1171932'], time:0.411081ms, swizzle: NOOP, TFLOPS: 41.79
out_tf32(mma2x4+...+stage3+swizzle): ['-38.139759', '21.1171932'], time:0.498151ms, swizzle: 512 , TFLOPS: 34.49
out_tf32(mma2x4+...+stage2+swizzle): ['-38.139759', '21.1171932'], time:0.407123ms, swizzle: 512 , TFLOPS: 42.20 (+0.03%)
 out_tf32(...+stage3+dsmem+swizzle): ['-38.139759', '21.1171932'], time:0.449836ms, swizzle: 512 , TFLOPS: 38.19
 out_tf32(...+stage2+dsmem+swizzle): ['-38.139759', '21.1171932'], time:0.407958ms, swizzle: 512 , TFLOPS: 42.11
              out_tf32(cublas+tf32): ['-38.139759', '21.1171932'], time:0.363135ms, swizzle: NOOP, TFLOPS: 47.31 (+12.11%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=2048, K=1024
                     out_f32(naive): ['-45.176471', '17.7125091'], time:10.25258ms, swizzle: NOOP, TFLOPS: 3.35  (+0.00%)
                  out_f32x4(t8x8sk): ['-45.176471', '17.7125091'], time:1.233506ms, swizzle: NOOP, TFLOPS: 27.86 (+731.17%)
                 out_f32x4(t8x8bcf): ['-45.176471', '17.7125091'], time:1.065564ms, swizzle: NOOP, TFLOPS: 32.25 (+15.76%)
                out_f32x4(t8x8dbuf): ['-45.176471', '17.7125091'], time:0.924682ms, swizzle: NOOP, TFLOPS: 37.16 (+15.24%)
                    out_f32(cublas): ['-45.176471', '17.7125091'], time:0.961315ms, swizzle: NOOP, TFLOPS: 35.74
                         out_f32_th: ['-45.176471', '17.7125091'], time:0.953686ms, swizzle: NOOP, TFLOPS: 36.03
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-45.176361', '17.7123947'], time:1.086962ms, swizzle: NOOP, TFLOPS: 31.61
    out_tf32(mma2x4+warp2x4+stage2): ['-45.176361', '17.7123947'], time:0.838124ms, swizzle: NOOP, TFLOPS: 41.00 (+10.33%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-45.176361', '17.7123947'], time:0.898563ms, swizzle: NOOP, TFLOPS: 38.24
  out_tf32(mma2x4+...+stage2+dsmem): ['-45.176361', '17.7123947'], time:0.816154ms, swizzle: NOOP, TFLOPS: 42.10 (+2.69%)
out_tf32(mma2x4+...+stage3+swizzle): ['-45.176361', '17.7123947'], time:0.986397ms, swizzle: 512 , TFLOPS: 34.83
out_tf32(mma2x4+...+stage2+swizzle): ['-45.176361', '17.7123947'], time:0.798583ms, swizzle: 512 , TFLOPS: 43.03 (+2.20%)
 out_tf32(...+stage3+dsmem+swizzle): ['-45.176361', '17.7123947'], time:0.885534ms, swizzle: 512 , TFLOPS: 38.80
 out_tf32(...+stage2+dsmem+swizzle): ['-45.176361', '17.7123947'], time:0.798654ms, swizzle: 512 , TFLOPS: 43.02
              out_tf32(cublas+tf32): ['-45.176361', '17.7123947'], time:0.679850ms, swizzle: NOOP, TFLOPS: 50.54 (+17.46%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=2048, K=2048
                     out_f32(naive): ['-63.444438', '38.4295463'], time:20.41318ms, swizzle: NOOP, TFLOPS: 3.37  (+0.00%)
                  out_f32x4(t8x8sk): ['-63.444438', '38.4295463'], time:2.422833ms, swizzle: NOOP, TFLOPS: 28.36 (+742.53%)
                 out_f32x4(t8x8bcf): ['-63.444438', '38.4295463'], time:2.085757ms, swizzle: NOOP, TFLOPS: 32.95 (+16.16%)
                out_f32x4(t8x8dbuf): ['-63.444438', '38.4295463'], time:1.884448ms, swizzle: NOOP, TFLOPS: 36.47 (+10.68%)
                    out_f32(cublas): ['-63.444438', '38.4295463'], time:1.839911ms, swizzle: NOOP, TFLOPS: 37.35 (+2.42%)
                         out_f32_th: ['-63.444438', '38.4295463'], time:1.811099ms, swizzle: NOOP, TFLOPS: 37.94 (+1.59%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-63.444118', '38.4292221'], time:2.120578ms, swizzle: NOOP, TFLOPS: 32.41
    out_tf32(mma2x4+warp2x4+stage2): ['-63.444118', '38.4292221'], time:1.714158ms, swizzle: NOOP, TFLOPS: 40.09 (+5.66%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-63.444118', '38.4292221'], time:1.859271ms, swizzle: NOOP, TFLOPS: 36.96
  out_tf32(mma2x4+...+stage2+dsmem): ['-63.444118', '38.4292221'], time:1.693952ms, swizzle: NOOP, TFLOPS: 40.57 (+1.19%)
out_tf32(mma2x4+...+stage3+swizzle): ['-63.444118', '38.4292221'], time:1.982808ms, swizzle: 512 , TFLOPS: 34.66
out_tf32(mma2x4+...+stage2+swizzle): ['-63.444118', '38.4292221'], time:1.616728ms, swizzle: 512 , TFLOPS: 42.51 (+4.78%)
 out_tf32(...+stage3+dsmem+swizzle): ['-63.444118', '38.4292221'], time:1.785743ms, swizzle: 512 , TFLOPS: 38.48
 out_tf32(...+stage2+dsmem+swizzle): ['-63.444118', '38.4292221'], time:1.616978ms, swizzle: 512 , TFLOPS: 42.50
              out_tf32(cublas+tf32): ['-63.444118', '38.4292221'], time:1.309895ms, swizzle: NOOP, TFLOPS: 52.46 (+23.42%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=2048, K=4096
                     out_f32(naive): ['-53.302734', '81.0102462'], time:40.70653ms, swizzle: NOOP, TFLOPS: 3.38  (+0.00%)
                  out_f32x4(t8x8sk): ['-53.302734', '81.0102462'], time:4.840099ms, swizzle: NOOP, TFLOPS: 28.40 (+741.03%)
                 out_f32x4(t8x8bcf): ['-53.302734', '81.0102462'], time:4.246413ms, swizzle: NOOP, TFLOPS: 32.37 (+13.98%)
                out_f32x4(t8x8dbuf): ['-53.302734', '81.0102462'], time:3.861558ms, swizzle: NOOP, TFLOPS: 35.59 (+9.97%)
                    out_f32(cublas): ['-53.302734', '81.0102462'], time:3.663694ms, swizzle: NOOP, TFLOPS: 37.51 (+5.40%)
                         out_f32_th: ['-53.302734', '81.0102462'], time:3.551971ms, swizzle: NOOP, TFLOPS: 38.69 (+3.15%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-53.301563', '81.0088272'], time:4.366672ms, swizzle: NOOP, TFLOPS: 31.47
    out_tf32(mma2x4+warp2x4+stage2): ['-53.301563', '81.0088272'], time:3.547680ms, swizzle: NOOP, TFLOPS: 38.74 (+0.12%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-53.301563', '81.0088272'], time:3.854322ms, swizzle: NOOP, TFLOPS: 35.66
  out_tf32(mma2x4+...+stage2+dsmem): ['-53.301563', '81.0088272'], time:3.531742ms, swizzle: NOOP, TFLOPS: 38.92 (+0.45%)
out_tf32(mma2x4+...+stage3+swizzle): ['-53.301563', '81.0088272'], time:4.192590ms, swizzle: 512 , TFLOPS: 32.78
out_tf32(mma2x4+...+stage2+swizzle): ['-53.301563', '81.0088272'], time:3.461444ms, swizzle: 512 , TFLOPS: 39.71 (+2.03%)
 out_tf32(...+stage3+dsmem+swizzle): ['-53.301563', '81.0088272'], time:3.778910ms, swizzle: 512 , TFLOPS: 36.37
 out_tf32(...+stage2+dsmem+swizzle): ['-53.301563', '81.0088272'], time:3.457832ms, swizzle: 512 , TFLOPS: 39.75 (+0.10%)
              out_tf32(cublas+tf32): ['-53.301563', '81.0088272'], time:2.559447ms, swizzle: NOOP, TFLOPS: 53.70 (+35.10%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=512
                     out_f32(naive): ['-38.139781', '21.1172447'], time:10.26971ms, swizzle: NOOP, TFLOPS: 3.35  (+0.00%)
                  out_f32x4(t8x8sk): ['-38.139781', '21.1172447'], time:1.201164ms, swizzle: NOOP, TFLOPS: 28.61 (+754.98%)
                 out_f32x4(t8x8bcf): ['-38.139781', '21.1172447'], time:1.043498ms, swizzle: NOOP, TFLOPS: 32.93 (+15.11%)
                out_f32x4(t8x8dbuf): ['-38.139781', '21.1172447'], time:0.908219ms, swizzle: NOOP, TFLOPS: 37.83 (+14.89%)
                    out_f32(cublas): ['-38.139781', '21.1172447'], time:0.960171ms, swizzle: NOOP, TFLOPS: 35.79
                         out_f32_th: ['-38.139781', '21.1172447'], time:0.959157ms, swizzle: NOOP, TFLOPS: 35.82
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-38.139759', '21.1171932'], time:1.014435ms, swizzle: NOOP, TFLOPS: 33.87
    out_tf32(mma2x4+warp2x4+stage2): ['-38.139759', '21.1171932'], time:0.840067ms, swizzle: NOOP, TFLOPS: 40.90 (+8.11%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-38.139759', '21.1171932'], time:0.912702ms, swizzle: NOOP, TFLOPS: 37.65
  out_tf32(mma2x4+...+stage2+dsmem): ['-38.139759', '21.1171932'], time:0.814163ms, swizzle: NOOP, TFLOPS: 42.20 (+3.18%)
out_tf32(mma2x4+...+stage3+swizzle): ['-38.139759', '21.1171932'], time:0.938665ms, swizzle: 1024, TFLOPS: 36.60
out_tf32(mma2x4+...+stage2+swizzle): ['-38.139759', '21.1171932'], time:0.772488ms, swizzle: 1024, TFLOPS: 44.48 (+5.39%)
 out_tf32(...+stage3+dsmem+swizzle): ['-38.139759', '21.1171932'], time:0.847816ms, swizzle: 1024, TFLOPS: 40.53
 out_tf32(...+stage2+dsmem+swizzle): ['-38.139759', '21.1171932'], time:0.774466ms, swizzle: 1024, TFLOPS: 44.37
              out_tf32(cublas+tf32): ['-38.139759', '21.1171932'], time:0.699150ms, swizzle: NOOP, TFLOPS: 49.14 (+10.49%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=1024
                     out_f32(naive): ['-45.176471', '17.7125091'], time:20.41501ms, swizzle: NOOP, TFLOPS: 3.37  (+0.00%)
                  out_f32x4(t8x8sk): ['-45.176471', '17.7125091'], time:2.332031ms, swizzle: NOOP, TFLOPS: 29.47 (+775.42%)
                 out_f32x4(t8x8bcf): ['-45.176471', '17.7125091'], time:2.066183ms, swizzle: NOOP, TFLOPS: 33.26 (+12.87%)
                out_f32x4(t8x8dbuf): ['-45.176471', '17.7125091'], time:1.836705ms, swizzle: NOOP, TFLOPS: 37.41 (+12.49%)
                    out_f32(cublas): ['-45.176471', '17.7125091'], time:1.938247ms, swizzle: NOOP, TFLOPS: 35.45
                         out_f32_th: ['-45.176471', '17.7125091'], time:1.861834ms, swizzle: NOOP, TFLOPS: 36.91
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-45.176361', '17.7123947'], time:2.018451ms, swizzle: NOOP, TFLOPS: 34.05
    out_tf32(mma2x4+warp2x4+stage2): ['-45.176361', '17.7123947'], time:1.617014ms, swizzle: NOOP, TFLOPS: 42.50 (+13.59%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-45.176361', '17.7123947'], time:1.688873ms, swizzle: NOOP, TFLOPS: 40.69
  out_tf32(mma2x4+...+stage2+dsmem): ['-45.176361', '17.7123947'], time:1.543068ms, swizzle: NOOP, TFLOPS: 44.53 (+4.79%)
out_tf32(mma2x4+...+stage3+swizzle): ['-45.176361', '17.7123947'], time:1.872932ms, swizzle: 1024, TFLOPS: 36.69
out_tf32(mma2x4+...+stage2+swizzle): ['-45.176361', '17.7123947'], time:1.532876ms, swizzle: 1024, TFLOPS: 44.83 (+0.66%)
 out_tf32(...+stage3+dsmem+swizzle): ['-45.176361', '17.7123947'], time:1.685547ms, swizzle: 1024, TFLOPS: 40.77
 out_tf32(...+stage2+dsmem+swizzle): ['-45.176361', '17.7123947'], time:1.534163ms, swizzle: 1024, TFLOPS: 44.79
              out_tf32(cublas+tf32): ['-45.176361', '17.7123947'], time:1.331567ms, swizzle: NOOP, TFLOPS: 51.61 (+15.12%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=2048
                     out_f32(naive): ['-63.444438', '38.4295463'], time:40.66463ms, swizzle: NOOP, TFLOPS: 3.38  (+0.00%)
                  out_f32x4(t8x8sk): ['-63.444438', '38.4295463'], time:4.637265ms, swizzle: NOOP, TFLOPS: 29.64 (+776.91%)
                 out_f32x4(t8x8bcf): ['-63.444438', '38.4295463'], time:4.159796ms, swizzle: NOOP, TFLOPS: 33.04 (+11.48%)
                out_f32x4(t8x8dbuf): ['-63.444438', '38.4295463'], time:3.692436ms, swizzle: NOOP, TFLOPS: 37.22 (+12.66%)
                    out_f32(cublas): ['-63.444438', '38.4295463'], time:3.658747ms, swizzle: NOOP, TFLOPS: 37.56 (+0.92%)
                         out_f32_th: ['-63.444438', '38.4295463'], time:3.592646ms, swizzle: NOOP, TFLOPS: 38.26 (+1.84%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-63.444118', '38.4292221'], time:3.974151ms, swizzle: NOOP, TFLOPS: 34.58
    out_tf32(mma2x4+warp2x4+stage2): ['-63.444118', '38.4292221'], time:3.176820ms, swizzle: NOOP, TFLOPS: 43.26 (+13.09%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-63.444118', '38.4292221'], time:3.462576ms, swizzle: NOOP, TFLOPS: 39.69
  out_tf32(mma2x4+...+stage2+dsmem): ['-63.444118', '38.4292221'], time:3.171265ms, swizzle: NOOP, TFLOPS: 43.34 (+0.18%)
out_tf32(mma2x4+...+stage3+swizzle): ['-63.444118', '38.4292221'], time:3.786218ms, swizzle: 1024, TFLOPS: 36.30
out_tf32(mma2x4+...+stage2+swizzle): ['-63.444118', '38.4292221'], time:3.086996ms, swizzle: 1024, TFLOPS: 44.52 (+2.73%)
 out_tf32(...+stage3+dsmem+swizzle): ['-63.444118', '38.4292221'], time:3.384017ms, swizzle: 1024, TFLOPS: 40.61
 out_tf32(...+stage2+dsmem+swizzle): ['-63.444118', '38.4292221'], time:3.088080ms, swizzle: 1024, TFLOPS: 44.51
              out_tf32(cublas+tf32): ['-63.444118', '38.4292221'], time:2.580893ms, swizzle: NOOP, TFLOPS: 53.25 (+19.61%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=4096
                     out_f32(naive): ['-53.302734', '81.0102462'], time:81.23998ms, swizzle: NOOP, TFLOPS: 3.38  (+0.00%)
                  out_f32x4(t8x8sk): ['-53.302734', '81.0102462'], time:9.287548ms, swizzle: NOOP, TFLOPS: 29.60 (+774.72%)
                 out_f32x4(t8x8bcf): ['-53.302734', '81.0102462'], time:8.298587ms, swizzle: NOOP, TFLOPS: 33.12 (+11.92%)
                out_f32x4(t8x8dbuf): ['-53.302734', '81.0102462'], time:7.456612ms, swizzle: NOOP, TFLOPS: 36.86 (+11.29%)
                    out_f32(cublas): ['-53.302734', '81.0102462'], time:7.171332ms, swizzle: NOOP, TFLOPS: 38.33 (+3.98%)
                         out_f32_th: ['-53.302734', '81.0102462'], time:7.176363ms, swizzle: NOOP, TFLOPS: 38.30
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-53.301563', '81.0088272'], time:7.860970ms, swizzle: NOOP, TFLOPS: 34.97
    out_tf32(mma2x4+warp2x4+stage2): ['-53.301563', '81.0088272'], time:6.430733ms, swizzle: NOOP, TFLOPS: 42.74 (+11.52%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-53.301563', '81.0088272'], time:6.960058ms, swizzle: NOOP, TFLOPS: 39.49
  out_tf32(mma2x4+...+stage2+dsmem): ['-53.301563', '81.0088272'], time:6.388366ms, swizzle: NOOP, TFLOPS: 43.03 (+0.66%)
out_tf32(mma2x4+...+stage3+swizzle): ['-53.301563', '81.0088272'], time:7.707548ms, swizzle: 1024, TFLOPS: 35.66
out_tf32(mma2x4+...+stage2+swizzle): ['-53.301563', '81.0088272'], time:6.338858ms, swizzle: 1024, TFLOPS: 43.36 (+0.78%)
 out_tf32(...+stage3+dsmem+swizzle): ['-53.301563', '81.0088272'], time:6.924736ms, swizzle: 1024, TFLOPS: 39.70
 out_tf32(...+stage2+dsmem+swizzle): ['-53.301563', '81.0088272'], time:6.340122ms, swizzle: 1024, TFLOPS: 43.36
              out_tf32(cublas+tf32): ['-53.301563', '81.0088272'], time:5.090487ms, swizzle: NOOP, TFLOPS: 54.00 (+24.52%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=512
                     out_f32(naive): ['-38.139781', '21.1172447'], time:20.61622ms, swizzle: NOOP, TFLOPS: 3.33  (+0.00%)
                  out_f32x4(t8x8sk): ['-38.139781', '21.1172447'], time:2.345955ms, swizzle: NOOP, TFLOPS: 29.29 (+778.80%)
                 out_f32x4(t8x8bcf): ['-38.139781', '21.1172447'], time:2.059459ms, swizzle: NOOP, TFLOPS: 33.37 (+13.91%)
                out_f32x4(t8x8dbuf): ['-38.139781', '21.1172447'], time:1.853263ms, swizzle: NOOP, TFLOPS: 37.08 (+11.13%)
                    out_f32(cublas): ['-38.139781', '21.1172447'], time:2.026152ms, swizzle: NOOP, TFLOPS: 33.92
                         out_f32_th: ['-38.139781', '21.1172447'], time:1.903581ms, swizzle: NOOP, TFLOPS: 36.10
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-38.139759', '21.1171932'], time:1.948964ms, swizzle: NOOP, TFLOPS: 35.26
    out_tf32(mma2x4+warp2x4+stage2): ['-38.139759', '21.1171932'], time:1.598036ms, swizzle: NOOP, TFLOPS: 43.00 (+15.97%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-38.139759', '21.1171932'], time:1.643598ms, swizzle: NOOP, TFLOPS: 41.81
  out_tf32(mma2x4+...+stage2+dsmem): ['-38.139759', '21.1171932'], time:1.511526ms, swizzle: NOOP, TFLOPS: 45.46 (+5.72%)
out_tf32(mma2x4+...+stage3+swizzle): ['-38.139759', '21.1171932'], time:1.839053ms, swizzle: 2048, TFLOPS: 37.37
out_tf32(mma2x4+...+stage2+swizzle): ['-38.139759', '21.1171932'], time:1.516401ms, swizzle: 2048, TFLOPS: 45.32
 out_tf32(...+stage3+dsmem+swizzle): ['-38.139759', '21.1171932'], time:1.657950ms, swizzle: 2048, TFLOPS: 41.45
 out_tf32(...+stage2+dsmem+swizzle): ['-38.139759', '21.1171932'], time:1.527225ms, swizzle: 2048, TFLOPS: 45.00
              out_tf32(cublas+tf32): ['-38.139759', '21.1171932'], time:1.322066ms, swizzle: NOOP, TFLOPS: 51.98 (+14.33%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=1024
                     out_f32(naive): ['-45.176471', '17.7125091'], time:40.72170ms, swizzle: NOOP, TFLOPS: 3.38  (+0.00%)
                  out_f32x4(t8x8sk): ['-45.176471', '17.7125091'], time:4.620969ms, swizzle: NOOP, TFLOPS: 29.74 (+781.24%)
                 out_f32x4(t8x8bcf): ['-45.176471', '17.7125091'], time:4.087328ms, swizzle: NOOP, TFLOPS: 33.63 (+13.06%)
                out_f32x4(t8x8dbuf): ['-45.176471', '17.7125091'], time:3.678190ms, swizzle: NOOP, TFLOPS: 37.37 (+11.12%)
                    out_f32(cublas): ['-45.176471', '17.7125091'], time:3.668308ms, swizzle: NOOP, TFLOPS: 37.47 (+0.27%)
                         out_f32_th: ['-45.176471', '17.7125091'], time:3.576636ms, swizzle: NOOP, TFLOPS: 38.43 (+2.56%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-45.176361', '17.7123947'], time:3.752338ms, swizzle: NOOP, TFLOPS: 36.63
    out_tf32(mma2x4+warp2x4+stage2): ['-45.176361', '17.7123947'], time:2.998983ms, swizzle: NOOP, TFLOPS: 45.83 (+19.26%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-45.176361', '17.7123947'], time:3.254044ms, swizzle: NOOP, TFLOPS: 42.24
  out_tf32(mma2x4+...+stage2+dsmem): ['-45.176361', '17.7123947'], time:2.984964ms, swizzle: NOOP, TFLOPS: 46.04 (+0.47%)
out_tf32(mma2x4+...+stage3+swizzle): ['-45.176361', '17.7123947'], time:3.664243ms, swizzle: 2048, TFLOPS: 37.51
out_tf32(mma2x4+...+stage2+swizzle): ['-45.176361', '17.7123947'], time:2.994239ms, swizzle: 2048, TFLOPS: 45.90
 out_tf32(...+stage3+dsmem+swizzle): ['-45.176361', '17.7123947'], time:3.278803ms, swizzle: 2048, TFLOPS: 41.92
 out_tf32(...+stage2+dsmem+swizzle): ['-45.176361', '17.7123947'], time:2.997756ms, swizzle: 2048, TFLOPS: 45.85
              out_tf32(cublas+tf32): ['-45.176361', '17.7123947'], time:2.519428ms, swizzle: NOOP, TFLOPS: 54.55 (+18.48%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=2048
                     out_f32(naive): ['-63.444438', '38.4295463'], time:81.26155ms, swizzle: NOOP, TFLOPS: 3.38  (+0.00%)
                  out_f32x4(t8x8sk): ['-63.444438', '38.4295463'], time:9.094738ms, swizzle: NOOP, TFLOPS: 30.22 (+793.50%)
                 out_f32x4(t8x8bcf): ['-63.444438', '38.4295463'], time:8.141028ms, swizzle: NOOP, TFLOPS: 33.76 (+11.71%)
                out_f32x4(t8x8dbuf): ['-63.444438', '38.4295463'], time:7.292842ms, swizzle: NOOP, TFLOPS: 37.69 (+11.63%)
                    out_f32(cublas): ['-63.444438', '38.4295463'], time:7.507431ms, swizzle: NOOP, TFLOPS: 36.61
                         out_f32_th: ['-63.444438', '38.4295463'], time:7.204532ms, swizzle: NOOP, TFLOPS: 38.15 (+1.23%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-63.444118', '38.4292221'], time:7.576847ms, swizzle: NOOP, TFLOPS: 36.28
    out_tf32(mma2x4+warp2x4+stage2): ['-63.444118', '38.4292221'], time:6.128752ms, swizzle: NOOP, TFLOPS: 44.85 (+17.55%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-63.444118', '38.4292221'], time:6.638467ms, swizzle: NOOP, TFLOPS: 41.41
  out_tf32(mma2x4+...+stage2+dsmem): ['-63.444118', '38.4292221'], time:6.089091ms, swizzle: NOOP, TFLOPS: 45.14 (+0.65%)
out_tf32(mma2x4+...+stage3+swizzle): ['-63.444118', '38.4292221'], time:7.358527ms, swizzle: 2048, TFLOPS: 37.36
out_tf32(mma2x4+...+stage2+swizzle): ['-63.444118', '38.4292221'], time:6.038904ms, swizzle: 2048, TFLOPS: 45.52 (+0.83%)
 out_tf32(...+stage3+dsmem+swizzle): ['-63.444118', '38.4292221'], time:6.595373ms, swizzle: 2048, TFLOPS: 41.68
 out_tf32(...+stage2+dsmem+swizzle): ['-63.444118', '38.4292221'], time:6.041479ms, swizzle: 2048, TFLOPS: 45.50
              out_tf32(cublas+tf32): ['-63.444118', '38.4292221'], time:4.927039ms, swizzle: NOOP, TFLOPS: 55.79 (+22.57%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=4096
                     out_f32(naive): ['-53.302734', '81.0102462'], time:268.5254ms, swizzle: NOOP, TFLOPS: 2.05  (+0.00%)
                  out_f32x4(t8x8sk): ['-53.302734', '81.0102462'], time:19.46458ms, swizzle: NOOP, TFLOPS: 28.24 (+1279.56%)
                 out_f32x4(t8x8bcf): ['-53.302734', '81.0102462'], time:17.27800ms, swizzle: NOOP, TFLOPS: 31.82 (+12.66%)
                out_f32x4(t8x8dbuf): ['-53.302734', '81.0102462'], time:19.06965ms, swizzle: NOOP, TFLOPS: 28.83
                    out_f32(cublas): ['-53.302734', '81.0102462'], time:14.07110ms, swizzle: NOOP, TFLOPS: 39.07 (+22.79%)
                         out_f32_th: ['-53.302734', '81.0102462'], time:13.88913ms, swizzle: NOOP, TFLOPS: 39.58 (+1.31%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-53.301563', '81.0088272'], time:14.84680ms, swizzle: NOOP, TFLOPS: 37.03
    out_tf32(mma2x4+warp2x4+stage2): ['-53.301563', '81.0088272'], time:12.48193ms, swizzle: NOOP, TFLOPS: 44.04 (+11.27%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-53.301563', '81.0088272'], time:13.23648ms, swizzle: NOOP, TFLOPS: 41.53
  out_tf32(mma2x4+...+stage2+dsmem): ['-53.301563', '81.0088272'], time:12.67541ms, swizzle: NOOP, TFLOPS: 43.37
out_tf32(mma2x4+...+stage3+swizzle): ['-53.301563', '81.0088272'], time:14.82864ms, swizzle: 2048, TFLOPS: 37.07
out_tf32(mma2x4+...+stage2+swizzle): ['-53.301563', '81.0088272'], time:12.10294ms, swizzle: 2048, TFLOPS: 45.42 (+3.13%)
 out_tf32(...+stage3+dsmem+swizzle): ['-53.301563', '81.0088272'], time:13.19457ms, swizzle: 2048, TFLOPS: 41.67
 out_tf32(...+stage2+dsmem+swizzle): ['-53.301563', '81.0088272'], time:12.10335ms, swizzle: 2048, TFLOPS: 45.42
              out_tf32(cublas+tf32): ['-53.301563', '81.0088272'], time:9.771990ms, swizzle: NOOP, TFLOPS: 56.26 (+23.85%)
----------------------------------------------------------------------------------------------------------------------------------
```
