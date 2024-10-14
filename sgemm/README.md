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
- [X] sgemm_wmma_m16n16k8_mma4x2_warp2x4_stage2/3 (Tensor Cores, Tile MMA/Warp, Copy Async, Stage)
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
                                             M=2048, N=2048, K=512
                            out_f32: ['-9.28823185 ', '1.97204959  ', '39.39577103 '], time:1.296747ms
                        out_f32(sk): ['-9.28823185 ', '1.97204959  ', '39.39577103 '], time:0.926423ms
                  out_f32x4(t8x8sk): ['-9.28823185 ', '1.97204959  ', '39.39577103 '], time:0.169945ms
                 out_f32x4(t8x8bcf): ['-9.28823185 ', '1.97204959  ', '39.39577103 '], time:0.148237ms
                out_f32x4(t8x8dbuf): ['-9.28823185 ', '1.97204959  ', '39.39577103 '], time:0.118232ms
----------------------------------------------------WMMA------------------------------------------------------
                    out_f32(cublas): ['-9.28823185 ', '1.97204959  ', '39.39577103 '], time:0.168777ms
          out_tf32(m16n16k8+stage2): ['-9.29609776 ', '1.98244298  ', '39.39843369 '], time:0.109994ms
              out_tf32(cublas+tf32): ['-9.29609776 ', '1.98244298  ', '39.39843369 '], time:0.192106ms
                         out_f32_th: ['-9.28823185 ', '1.97204959  ', '39.39577103 '], time:0.134206ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=2048, N=2048, K=1024
                            out_f32: ['-24.95363998', '43.15856934 ', '7.36148405  '], time:2.580559ms
                        out_f32(sk): ['-24.95363998', '43.15856934 ', '7.36148405  '], time:1.839375ms
                  out_f32x4(t8x8sk): ['-24.95363998', '43.15856934 ', '7.36148405  '], time:0.324976ms
                 out_f32x4(t8x8bcf): ['-24.95363998', '43.15856934 ', '7.36148405  '], time:0.288677ms
                out_f32x4(t8x8dbuf): ['-24.95363998', '43.15856934 ', '7.36148405  '], time:0.227904ms
----------------------------------------------------WMMA------------------------------------------------------
                    out_f32(cublas): ['-24.95365906', '43.15855026 ', '7.36147976  '], time:0.267315ms
          out_tf32(m16n16k8+stage2): ['-24.95110321', '43.14693451 ', '7.35476875  '], time:0.210178ms
              out_tf32(cublas+tf32): ['-24.95110321', '43.14693451 ', '7.35476875  '], time:0.179255ms
                         out_f32_th: ['-24.95365906', '43.15855026 ', '7.36147976  '], time:0.252545ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=2048, N=4096, K=512
                            out_f32: ['35.91652679 ', '-27.46051407', '36.37091446 '], time:2.582872ms
                        out_f32(sk): ['35.91652679 ', '-27.46051407', '36.37091446 '], time:1.841712ms
                  out_f32x4(t8x8sk): ['35.91652679 ', '-27.46051407', '36.37091446 '], time:0.321710ms
                 out_f32x4(t8x8bcf): ['35.91652679 ', '-27.46051407', '36.37091446 '], time:0.267148ms
                out_f32x4(t8x8dbuf): ['35.91652679 ', '-27.46051407', '36.37091446 '], time:0.230658ms
----------------------------------------------------WMMA------------------------------------------------------
                    out_f32(cublas): ['35.91652679 ', '-27.46051407', '36.37091446 '], time:0.277698ms
          out_tf32(m16n16k8+stage2): ['35.91730499 ', '-27.4563942 ', '36.36489487 '], time:0.209129ms
              out_tf32(cublas+tf32): ['35.91730499 ', '-27.4563942 ', '36.36489487 '], time:0.184381ms
                         out_f32_th: ['35.91652679 ', '-27.46051407', '36.37091446 '], time:0.266743ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=2048, N=4096, K=1024
                            out_f32: ['-37.3944397 ', '-6.64517021 ', '16.86618805 '], time:5.123031ms
                        out_f32(sk): ['-37.3944397 ', '-6.64517021 ', '16.86618805 '], time:3.653574ms
                  out_f32x4(t8x8sk): ['-37.3944397 ', '-6.64517021 ', '16.86618805 '], time:0.620103ms
                 out_f32x4(t8x8bcf): ['-37.3944397 ', '-6.64517021 ', '16.86618805 '], time:0.525570ms
                out_f32x4(t8x8dbuf): ['-37.3944397 ', '-6.64517021 ', '16.86618805 '], time:0.447142ms
----------------------------------------------------WMMA------------------------------------------------------
                    out_f32(cublas): ['-37.3944397 ', '-6.64517021 ', '16.86618805 '], time:0.528228ms
          out_tf32(m16n16k8+stage2): ['-37.38657761', '-6.64346695 ', '16.8794136  '], time:0.405896ms
              out_tf32(cublas+tf32): ['-37.38657761', '-6.64346695 ', '16.8794136  '], time:0.341558ms
                         out_f32_th: ['-37.3944397 ', '-6.64517021 ', '16.86618805 '], time:0.518692ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=4096, N=2048, K=512
                            out_f32: ['10.36220837 ', '7.20147228  ', '-13.39327431'], time:2.573860ms
                        out_f32(sk): ['10.36220837 ', '7.20147228  ', '-13.39327431'], time:1.841915ms
                  out_f32x4(t8x8sk): ['10.36220837 ', '7.20147228  ', '-13.39327431'], time:0.321782ms
                 out_f32x4(t8x8bcf): ['10.36220837 ', '7.20147228  ', '-13.39327431'], time:0.267649ms
                out_f32x4(t8x8dbuf): ['10.36220837 ', '7.20147228  ', '-13.39327431'], time:0.230742ms
----------------------------------------------------WMMA------------------------------------------------------
                    out_f32(cublas): ['10.36220837 ', '7.20147228  ', '-13.39327431'], time:0.275660ms
          out_tf32(m16n16k8+stage2): ['10.36374187 ', '7.20014429  ', '-13.3949852 '], time:0.209141ms
              out_tf32(cublas+tf32): ['10.36374187 ', '7.20014429  ', '-13.3949852 '], time:0.188208ms
                         out_f32_th: ['10.36220837 ', '7.20147228  ', '-13.39327431'], time:0.265503ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=4096, N=2048, K=1024
                            out_f32: ['-54.5629158 ', '4.83383846  ', '16.64455605 '], time:5.123162ms
                        out_f32(sk): ['-54.5629158 ', '4.83383846  ', '16.64455605 '], time:3.658152ms
                  out_f32x4(t8x8sk): ['-54.5629158 ', '4.83383846  ', '16.64455605 '], time:0.620127ms
                 out_f32x4(t8x8bcf): ['-54.5629158 ', '4.83383846  ', '16.64455605 '], time:0.526130ms
                out_f32x4(t8x8dbuf): ['-54.5629158 ', '4.83383846  ', '16.64455605 '], time:0.446749ms
----------------------------------------------------WMMA------------------------------------------------------
                    out_f32(cublas): ['-54.5629158 ', '4.83383846  ', '16.64455605 '], time:0.525975ms
          out_tf32(m16n16k8+stage2): ['-54.56904221', '4.82699633  ', '16.64771652 '], time:0.406063ms
              out_tf32(cublas+tf32): ['-54.56904221', '4.82699633  ', '16.64771652 '], time:0.341392ms
                         out_f32_th: ['-54.5629158 ', '4.83383846  ', '16.64455605 '], time:0.518477ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=4096, N=4096, K=512
                            out_f32: ['-21.22754669', '8.71378803  ', '0.50345194  '], time:5.148792ms
                        out_f32(sk): ['-21.22754669', '8.71378803  ', '0.50345194  '], time:3.675473ms
                  out_f32x4(t8x8sk): ['-21.22754669', '8.71378803  ', '0.50345194  '], time:0.629508ms
                 out_f32x4(t8x8bcf): ['-21.22754669', '8.71378803  ', '0.50345194  '], time:0.534534ms
                out_f32x4(t8x8dbuf): ['-21.22754669', '8.71378803  ', '0.50345194  '], time:0.462973ms
----------------------------------------------------WMMA------------------------------------------------------
                    out_f32(cublas): ['-21.22754669', '8.71378803  ', '0.50345194  '], time:0.458336ms
          out_tf32(m16n16k8+stage2): ['-21.22730255', '8.72151184  ', '0.5072608   '], time:0.406647ms
              out_tf32(cublas+tf32): ['-21.22730255', '8.72151184  ', '0.5072608   '], time:0.360131ms
                         out_f32_th: ['-21.22754669', '8.71378803  ', '0.50345194  '], time:0.449002ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=4096, N=4096, K=1024
                            out_f32: ['-36.18135071', '-70.96016693', '-23.16605759'], time:10.224795ms
                        out_f32(sk): ['-36.18135071', '-70.96016693', '-23.16605759'], time:7.320023ms
                  out_f32x4(t8x8sk): ['-36.18135071', '-70.96016693', '-23.16605759'], time:1.219559ms
                 out_f32x4(t8x8bcf): ['-36.18135071', '-70.96016693', '-23.16605759'], time:1.042867ms
                out_f32x4(t8x8dbuf): ['-36.18135071', '-70.96016693', '-23.16605759'], time:0.891960ms
----------------------------------------------------WMMA------------------------------------------------------
                    out_f32(cublas): ['-36.18135071', '-70.96016693', '-23.16605759'], time:0.867224ms
          out_tf32(m16n16k8+stage2): ['-36.17189789', '-70.95304108', '-23.14374542'], time:0.794697ms
              out_tf32(cublas+tf32): ['-36.17189789', '-70.95304108', '-23.14374542'], time:0.679755ms
                         out_f32_th: ['-36.18135071', '-70.96016693', '-23.16605759'], time:0.858045ms
--------------------------------------------------------------------------------------------------------------
```
