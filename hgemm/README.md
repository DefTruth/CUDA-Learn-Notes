# HGEMM 

## 0x00 说明

包含以下内容：

- [X] hgemm_sliced_k_f16_kernel 
- [X] hgemm_t_8x8_sliced_k_f16x4_kernel(unpack)
- [X] hgemm_t_8x8_sliced_k_f16x4_pack_kernel(pack 16x4)
- [X] hgemm_t_8x8_sliced_k_f16x4_bcf_kernel(bank conflicts reduce)
- [X] hgemm_t_8x8_sliced_k_f16x4_pack_bcf_kernel(bank conflicts reduce, pack)
- [X] hgemm_t_4x4_sliced_k_f16x4_pack_bcf_kernel(bank conflicts reduce, pack)
- [X] hgemm_t_8x8_sliced_k_f16x8_pack_bcf_kernel(bank conflicts reduce, pack)
- [X] hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf_kernel(bank conflicts reduce, pack, double buffers)
- [X] hgemm_t_8x8_sliced_k16/32_f16x8_pack_bcf_dbuf_kernel(pack, double buffers)
- [X] hgemm_t_8x8_sliced_k16/32_f16x8_pack_bcf_dbuf_async_kernel(pack, double buffers, copy async)
- [X] hgemm_wmma_m16n16k16_naive(WMMA) 
- [X] hgemm_wmma_m16n16k16_mma4x2(WMMA, Tile MMA) 
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4(TWMMA, Tile MMA/Warp, pack) 
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_async(WMMA, Tile MMA/Warp, Copy Async) 
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_async_offset(WMMA, Tile MMA/Warp, Copy Async, Pad)
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(WMMA, Tile MMA/Warp, Copy Async, Double Buffers, Pad)  
- [X] hgemm_wmma_m16n16k16_mma4x4_warp2x2x2_dbuf_async(WMMA, Tile MMA/Warp, Copy Async, Double Buffers, Pad)  
- [X] hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async(WMMA, Tile MMA/Warp, Copy Async, Double Buffers, Pad)
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_rbuf_async(WMMA, Tile MMA/Warp, Copy Async, Double/Reg Buffers, Pad)
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages(WMMA, Tile MMA/Warp, Copy Async, Stages, Pad, Block swizzle) 
- [X] hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages(WMMA, Tile MMA/Warp, Copy Async, Stages, Pad, Block swizzle) 
- [X] PyTorch bindings

## 目前性能  

目前最优的实现，在L20上（理论Tensor Cores FP16算力为 119.5 TFLOPS），能达到cuBLAS大概95%~98%左右的性能(105-110 TFLOPS vs 105-115 TFLOPS)，部分case会超越cuBLAS。已知问题为bank conflicts没有完全消除，目前通过padding的方式缓解bank conflicts会导致shared memory浪费，也会影响SM occupancy。并且尚未手工实现Warp swizzle(受限于WMMA API的灵活性以及本人的能力)，后续将会尝试通过MMA PTX实现warp swizzle。

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

本仓库实现的HGEMM Double Buffers策略如下：1）主循环从bk = 1 开始，第一次数据加载在主循环之前，最后一次计算在主循环之后，这是pipeline 的特点决定的；2）由于计算和下一次访存使用的Shared Memory不同，因此主循环中每次循环只需要一次__syncthreads()即可，对比非double buffers版本，总共节省了 ((K + BK - 1) / BK) - 1 次block内的同步操作。比如，bk=1时，HFMA计算使用的是s_a[0]和s_b[0]，因此，和s_a[1]和s_b[1]的加载是没有依赖关系的。HFMA计算，从global内存到s_a[1]和s_b[1]和HFMA计算可以并行。s_a[1]和s_b[1]用于加载下一块BK需要的数据到共享内存；3）由于GPU不能向CPU那样支持乱序执行，主循环中需要先将下一次循环计算需要的Gloabal Memory中的数据load 到寄存器，然后进行本次计算，之后再将load到寄存器中的数据写到Shared Memory，这样在LDG指令向Global Memory做load时，不会影响后续HFMA及其它运算指令的 launch 执行，也就达到了Double Buffers的目的。

```C
  // bk = 0 is loading here, buffer 0
  {
    int load_a_gmem_k = load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    LDST64BITS(r_load_a[0]) = LDST64BITS(a[load_a_gmem_addr]);
    LDST64BITS(r_load_b[0]) = LDST64BITS(b[load_b_gmem_addr]);

    s_a[0][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
    s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    LDST64BITS(s_b[0][load_b_smem_k][load_b_smem_n]) = LDST64BITS(r_load_b[0]);
  }
  // Without this synchronization, accuracy may occasionally be abnormal.
  __syncthreads(); 
  
  // bk start from 1，需要注意的是，虽然 bk 从 1 开始，但实际上 bk=1时，使用的是
  // 第0块BK中的数据（已经加载到共享内存s_a[0]和s_b[0]）；bk=2时，实际计算的是第1块
  // BK中的数据。其余以此类推，这个循环结束后，剩下最后一块BK大小的数据需要计算。
  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {

    int smem_sel = (bk - 1) & 1; // bk 1->0, bk 2->1, bk 3->0, ...
    int smem_sel_next = bk & 1;  // bk 1->1, bk 2->0, bk 3->1, ...

    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    LDST64BITS(r_load_a[0]) = LDST64BITS(a[load_a_gmem_addr]);
    LDST64BITS(r_load_b[0]) = LDST64BITS(b[load_b_gmem_addr]);
    
    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[smem_sel][tk][ty * TM]);
      LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[smem_sel][tk][tx * TN]);

      #pragma unroll
      for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
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
    LDST128BITS(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = LDST128BITS(r_load_b[0]);

    __syncthreads();
  }
  
  // 计算剩下最后一块BK
  #pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[1][tk][ty * TM]);
    LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[1][tk][tx * TN]);

    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
      #pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }

```

## PyTorch HGEMM Profile

在Ada架构下，PyTorch 2.4对FP16使用matmul时，会调用ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn kernel，内部实际使用HMMA(Tensor Cores)进行计算。

```bash
ncu -o hgemm.prof -f python3 prof.py
nsys profile --stats=true -t cuda,osrt,nvtx -o hgemm.prof --force-overwrite true python3 prof.py
```
- 日志

```bash
==PROF== Connected to process 367502 (/usr/bin/python3.10)
==PROF== Profiling "unrolled_elementwise_kernel" - 0: 0%....50%....100% - 8 passes
==PROF== Profiling "unrolled_elementwise_kernel" - 1: 0%....50%....100% - 8 passes
==PROF== Profiling "unrolled_elementwise_kernel" - 2: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 3: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 4: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 5: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 6: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 7: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 8: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 9: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 10: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 11: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 12: 0%....50%....100% - 8 passes
==PROF== Profiling "ampere_fp16_s1688gemm_fp16_12..." - 13: 0%....50%....100% - 8 passes
```

- SASS (L20)

```C
// ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn_kernel
310	00007f41 37d5b850	      LDSM.16.M88.4 R192, [R169+UR8+0x2000] 
311	00007f41 37d5b860	      LDSM.16.M88.4 R196, [R169+UR8+0x2800] 
312	00007f41 37d5b870	@!P0  BRA.U 0x7f4137d5c3f0 
313	00007f41 37d5b880	      HMMA.1688.F32 R0, R176, R192, R0 
314	00007f41 37d5b890	      LDSM.16.MT88.4 R184, [R167+UR8+0x400] 
315	00007f41 37d5b8a0	      HMMA.1688.F32 R32, R178, R192, R32 
316	00007f41 37d5b8b0	      LDSM.16.M88.4 R200, [R170+UR8+0x2000] 
317	00007f41 37d5b8c0	      HMMA.1688.F32 R64, R180, R192, R64 
318	00007f41 37d5b8d0	      LDSM.16.MT88.4 R188, [R168+UR8+0x400] 
319	00007f41 37d5b8e0	      HMMA.1688.F32 R96, R182, R192, R96 
320	00007f41 37d5b8f0	      LDSM.16.M88.4 R204, [R170+UR8+0x2800] 
321	00007f41 37d5b900	      HMMA.1688.F32 R100, R182, R193, R100 
322	00007f41 37d5b910	      HMMA.1688.F32 R68, R180, R193, R68 
323	00007f41 37d5b920	      HMMA.1688.F32 R36, R178, R193, R36 
324	00007f41 37d5b930	      HMMA.1688.F32 R4, R176, R193, R4 
325	00007f41 37d5b940	      HMMA.1688.F32 R8, R176, R194, R8 
326	00007f41 37d5b950	      HMMA.1688.F32 R40, R178, R194, R40 
327	00007f41 37d5b960	      HMMA.1688.F32 R72, R180, R194, R72 
328	00007f41 37d5b970	      HMMA.1688.F32 R104, R182, R194, R104 
329	00007f41 37d5b980	      HMMA.1688.F32 R108, R182, R195, R108 
330	00007f41 37d5b990	      HMMA.1688.F32 R76, R180, R195, R76 
331	00007f41 37d5b9a0	      HMMA.1688.F32 R44, R178, R195, R44 
332	00007f41 37d5b9b0	      HMMA.1688.F32 R12, R176, R195, R12 
333	00007f41 37d5b9c0	      HMMA.1688.F32 R16, R176, R196, R16 
334	00007f41 37d5b9d0	      HMMA.1688.F32 R48, R178, R196, R48 
335	00007f41 37d5b9e0	      HMMA.1688.F32 R80, R180, R196, R80 
336	00007f41 37d5b9f0	      HMMA.1688.F32 R112, R182, R196, R112 
337	00007f41 37d5ba00	      HMMA.1688.F32 R116, R182, R197, R116 
```
- SASS (RTX 3080)

```C
// sm80_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize96x64x32_stage3_warpsize2x2x1_tensor16x8x16_kernel
341	00000007 44ff6340	      HMMA.16816.F32 R12, R72, R80, R12 
342	00000007 44ff6350	      HMMA.16816.F32 R16, R72, R82, R16 
343	00000007 44ff6360	      HMMA.16816.F32 R20, R84, R76, R20 
344	00000007 44ff6370	      LDSM.16.M88.4 R52, [R92+UR8] 
345	00000007 44ff6380	      HMMA.16816.F32 R24, R84, R78, R24 
346	00000007 44ff6390	      LDSM.16.M88.4 R64, [R92+UR8+0x800] 
347	00000007 44ff63a0	      HMMA.16816.F32 R28, R84, R80, R28 
348	00000007 44ff63b0	      LDSM.16.M88.4 R68, [R92+UR8+0x1000] 
349	00000007 44ff63c0	      HMMA.16816.F32 R32, R84, R82, R32 
350	00000007 44ff63d0	      LDSM.16.MT88.4 R56, [R3+UR7+0x4800] 
351	00000007 44ff63e0	      HMMA.16816.F32 R36, R88, R76, R36 
352	00000007 44ff63f0	      LDSM.16.MT88.4 R60, [R106+UR7+0x4800] 
353	00000007 44ff6400	      HMMA.16816.F32 R40, R88, R78, R40 
354	00000007 44ff6410	      HMMA.16816.F32 R44, R88, R80, R44 
355	00000007 44ff6420	      HMMA.16816.F32 R48, R88, R82, R48 
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
python3 hgemm.py
```

输出:

- NVIDIA L20  
```bash
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=2048
                     f16x8pack(t8x8+bcf): ['-48.625   ', '-19.59375 '], time:1.423811ms, swizzle: NOOP, TFLOPS: 48.26 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-48.625   ', '-19.59375 '], time:1.406931ms, swizzle: NOOP, TFLOPS: 48.84 (+1.20%)
                f16x8pack(t8x8+k16+dbuf): ['-48.625   ', '-19.59375 '], time:1.330733ms, swizzle: NOOP, TFLOPS: 51.64 (+5.73%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-48.6875  ', '-19.71875 '], time:1.490950ms, swizzle: NOOP, TFLOPS: 46.09
                 f16wmma(mma4x2+warp2x4): ['-48.6875  ', '-19.71875 '], time:0.942635ms, swizzle: NOOP, TFLOPS: 72.90 (+41.17%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-48.6875  ', '-19.71875 '], time:0.703215ms, swizzle: NOOP, TFLOPS: 97.72 (+34.05%)
          f16wmma(mma2x4+warp2x4+stage4): ['-48.6875  ', '-19.71875 '], time:0.705647ms, swizzle: NOOP, TFLOPS: 97.38
          f16wmma(mma2x4+warp2x4+stage3): ['-48.6875  ', '-19.71875 '], time:0.695824ms, swizzle: NOOP, TFLOPS: 98.76 (+1.06%)
          f16wmma(mma2x4+warp2x4+stage2): ['-48.6875  ', '-19.71875 '], time:0.699400ms, swizzle: NOOP, TFLOPS: 98.25
        f16wmma(mma2x4+...+stage4+dsmem): ['-48.6875  ', '-19.71875 '], time:0.707101ms, swizzle: NOOP, TFLOPS: 97.18
        f16wmma(mma2x4+...+stage3+dsmem): ['-48.6875  ', '-19.71875 '], time:0.702190ms, swizzle: NOOP, TFLOPS: 97.86
        f16wmma(mma2x4+...+stage2+dsmem): ['-48.6875  ', '-19.71875 '], time:0.697612ms, swizzle: NOOP, TFLOPS: 98.51
      f16wmma(mma2x4+...+stage4+swizzle): ['-48.6875  ', '-19.71875 '], time:0.710916ms, swizzle: 1024, TFLOPS: 96.66
      f16wmma(mma2x4+...+stage3+swizzle): ['-48.6875  ', '-19.71875 '], time:0.694918ms, swizzle: 1024, TFLOPS: 98.89 (+0.13%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-48.6875  ', '-19.71875 '], time:0.686240ms, swizzle: 1024, TFLOPS: 100.14(+1.26%)
       f16wmma(...+stage4+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:0.713205ms, swizzle: 1024, TFLOPS: 96.35
       f16wmma(...+stage3+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:0.702953ms, swizzle: 1024, TFLOPS: 97.76
       f16wmma(...+stage2+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:0.686788ms, swizzle: 1024, TFLOPS: 100.06
                             f16(cublas): ['-48.6875  ', '-19.71875 '], time:0.854802ms, swizzle: NOOP, TFLOPS: 80.39
                                  f16_th: ['-48.75    ', '-19.765625'], time:0.660657ms, swizzle: NOOP, TFLOPS: 104.02(+3.87%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=4096
                     f16x8pack(t8x8+bcf): ['-7.390625 ', '-9.75     '], time:2.838373ms, swizzle: NOOP, TFLOPS: 48.42 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-7.390625 ', '-9.75     '], time:2.812886ms, swizzle: NOOP, TFLOPS: 48.86 (+0.91%)
                f16x8pack(t8x8+k16+dbuf): ['-7.390625 ', '-9.75     '], time:2.657032ms, swizzle: NOOP, TFLOPS: 51.73 (+5.87%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-7.765625 ', '-9.59375  '], time:2.967953ms, swizzle: NOOP, TFLOPS: 46.31
                 f16wmma(mma4x2+warp2x4): ['-7.765625 ', '-9.59375  '], time:1.837468ms, swizzle: NOOP, TFLOPS: 74.80 (+44.60%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-7.765625 ', '-9.59375  '], time:1.372146ms, swizzle: NOOP, TFLOPS: 100.16(+33.91%)
          f16wmma(mma2x4+warp2x4+stage4): ['-7.765625 ', '-9.59375  '], time:1.376533ms, swizzle: NOOP, TFLOPS: 99.84
          f16wmma(mma2x4+warp2x4+stage3): ['-7.765625 ', '-9.59375  '], time:1.353287ms, swizzle: NOOP, TFLOPS: 101.56(+1.39%)
          f16wmma(mma2x4+warp2x4+stage2): ['-7.765625 ', '-9.59375  '], time:1.358866ms, swizzle: NOOP, TFLOPS: 101.14
        f16wmma(mma2x4+...+stage4+dsmem): ['-7.765625 ', '-9.59375  '], time:1.368713ms, swizzle: NOOP, TFLOPS: 100.41
        f16wmma(mma2x4+...+stage3+dsmem): ['-7.765625 ', '-9.59375  '], time:1.362800ms, swizzle: NOOP, TFLOPS: 100.85
        f16wmma(mma2x4+...+stage2+dsmem): ['-7.765625 ', '-9.59375  '], time:1.357173ms, swizzle: NOOP, TFLOPS: 101.27
      f16wmma(mma2x4+...+stage4+swizzle): ['-7.765625 ', '-9.59375  '], time:1.377367ms, swizzle: 1024, TFLOPS: 99.78
      f16wmma(mma2x4+...+stage3+swizzle): ['-7.765625 ', '-9.59375  '], time:1.351928ms, swizzle: 1024, TFLOPS: 101.66(+0.10%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-7.765625 ', '-9.59375  '], time:1.335167ms, swizzle: 1024, TFLOPS: 102.94(+1.26%)
       f16wmma(...+stage4+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:1.376914ms, swizzle: 1024, TFLOPS: 99.82
       f16wmma(...+stage3+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:1.368808ms, swizzle: 1024, TFLOPS: 100.41
       f16wmma(...+stage2+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:1.335954ms, swizzle: 1024, TFLOPS: 102.88
                             f16(cublas): ['-7.765625 ', '-9.59375  '], time:1.504993ms, swizzle: NOOP, TFLOPS: 91.32
                                  f16_th: ['-7.9101562', '-9.703125 '], time:1.287364ms, swizzle: NOOP, TFLOPS: 106.76(+3.71%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=8192
                     f16x8pack(t8x8+bcf): ['-34.6875  ', '109.75    '], time:5.812764ms, swizzle: NOOP, TFLOPS: 47.29 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-34.6875  ', '109.75    '], time:5.647635ms, swizzle: NOOP, TFLOPS: 48.67 (+2.92%)
                f16x8pack(t8x8+k16+dbuf): ['-34.6875  ', '109.75    '], time:5.406832ms, swizzle: NOOP, TFLOPS: 50.84 (+4.45%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-34.6875  ', '108.625   '], time:5.936312ms, swizzle: NOOP, TFLOPS: 46.30
                 f16wmma(mma4x2+warp2x4): ['-34.6875  ', '108.625   '], time:3.657937ms, swizzle: NOOP, TFLOPS: 75.15 (+47.81%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-34.6875  ', '108.625   '], time:2.697873ms, swizzle: NOOP, TFLOPS: 101.89(+35.59%)
          f16wmma(mma2x4+warp2x4+stage4): ['-34.6875  ', '108.625   '], time:2.692198ms, swizzle: NOOP, TFLOPS: 102.10(+0.21%)
          f16wmma(mma2x4+warp2x4+stage3): ['-34.6875  ', '108.625   '], time:2.661299ms, swizzle: NOOP, TFLOPS: 103.29(+1.16%)
          f16wmma(mma2x4+warp2x4+stage2): ['-34.6875  ', '108.625   '], time:2.675151ms, swizzle: NOOP, TFLOPS: 102.75
        f16wmma(mma2x4+...+stage4+dsmem): ['-34.6875  ', '108.625   '], time:2.687406ms, swizzle: NOOP, TFLOPS: 102.28
        f16wmma(mma2x4+...+stage3+dsmem): ['-34.6875  ', '108.625   '], time:2.682614ms, swizzle: NOOP, TFLOPS: 102.47
        f16wmma(mma2x4+...+stage2+dsmem): ['-34.6875  ', '108.625   '], time:2.671098ms, swizzle: NOOP, TFLOPS: 102.91
      f16wmma(mma2x4+...+stage4+swizzle): ['-34.6875  ', '108.625   '], time:2.708673ms, swizzle: 1024, TFLOPS: 101.48
      f16wmma(mma2x4+...+stage3+swizzle): ['-34.6875  ', '108.625   '], time:2.665734ms, swizzle: 1024, TFLOPS: 103.12
      f16wmma(mma2x4+...+stage2+swizzle): ['-34.6875  ', '108.625   '], time:2.629923ms, swizzle: 1024, TFLOPS: 104.52(+1.19%)
       f16wmma(...+stage4+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:2.705121ms, swizzle: 1024, TFLOPS: 101.61
       f16wmma(...+stage3+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:2.694535ms, swizzle: 1024, TFLOPS: 102.01
       f16wmma(...+stage2+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:2.630662ms, swizzle: 1024, TFLOPS: 104.49
                             f16(cublas): ['-34.625   ', '109.0     '], time:2.632546ms, swizzle: NOOP, TFLOPS: 104.42
                                  f16_th: ['-34.90625 ', '108.5625  '], time:2.402138ms, swizzle: NOOP, TFLOPS: 114.43(+9.48%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=2048
                     f16x8pack(t8x8+bcf): ['-48.625   ', '-19.59375 '], time:2.712392ms, swizzle: NOOP, TFLOPS: 50.67 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-48.625   ', '-19.59375 '], time:2.710318ms, swizzle: NOOP, TFLOPS: 50.71 (+0.08%)
                f16x8pack(t8x8+k16+dbuf): ['-48.625   ', '-19.59375 '], time:2.539610ms, swizzle: NOOP, TFLOPS: 54.12 (+6.72%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-48.6875  ', '-19.71875 '], time:2.962994ms, swizzle: NOOP, TFLOPS: 46.39
                 f16wmma(mma4x2+warp2x4): ['-48.6875  ', '-19.71875 '], time:1.867127ms, swizzle: NOOP, TFLOPS: 73.61 (+36.02%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-48.6875  ', '-19.71875 '], time:1.337242ms, swizzle: NOOP, TFLOPS: 102.78(+39.63%)
          f16wmma(mma2x4+warp2x4+stage4): ['-48.6875  ', '-19.71875 '], time:1.343107ms, swizzle: NOOP, TFLOPS: 102.33
          f16wmma(mma2x4+warp2x4+stage3): ['-48.6875  ', '-19.71875 '], time:1.310682ms, swizzle: NOOP, TFLOPS: 104.86(+2.03%)
          f16wmma(mma2x4+warp2x4+stage2): ['-48.6875  ', '-19.71875 '], time:1.316022ms, swizzle: NOOP, TFLOPS: 104.44
        f16wmma(mma2x4+...+stage4+dsmem): ['-48.6875  ', '-19.71875 '], time:1.335048ms, swizzle: NOOP, TFLOPS: 102.95
        f16wmma(mma2x4+...+stage3+dsmem): ['-48.6875  ', '-19.71875 '], time:1.320838ms, swizzle: NOOP, TFLOPS: 104.05
        f16wmma(mma2x4+...+stage2+dsmem): ['-48.6875  ', '-19.71875 '], time:1.315379ms, swizzle: NOOP, TFLOPS: 104.49
      f16wmma(mma2x4+...+stage4+swizzle): ['-48.6875  ', '-19.71875 '], time:1.343488ms, swizzle: 2048, TFLOPS: 102.30
      f16wmma(mma2x4+...+stage3+swizzle): ['-48.6875  ', '-19.71875 '], time:1.307892ms, swizzle: 2048, TFLOPS: 105.08(+0.21%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-48.6875  ', '-19.71875 '], time:1.296949ms, swizzle: 2048, TFLOPS: 105.97(+0.84%)
       f16wmma(...+stage4+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:1.345443ms, swizzle: 2048, TFLOPS: 102.15
       f16wmma(...+stage3+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:1.321220ms, swizzle: 2048, TFLOPS: 104.02
       f16wmma(...+stage2+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:1.297736ms, swizzle: 2048, TFLOPS: 105.91
                             f16(cublas): ['-48.6875  ', '-19.71875 '], time:1.421713ms, swizzle: NOOP, TFLOPS: 96.67
                                  f16_th: ['-48.75    ', '-19.765625'], time:1.302385ms, swizzle: NOOP, TFLOPS: 105.53
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=4096
                     f16x8pack(t8x8+bcf): ['-7.390625 ', '-9.75     '], time:5.503845ms, swizzle: NOOP, TFLOPS: 49.94 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-7.390625 ', '-9.75     '], time:5.462884ms, swizzle: NOOP, TFLOPS: 50.32 (+0.75%)
                f16x8pack(t8x8+k16+dbuf): ['-7.390625 ', '-9.75     '], time:5.190730ms, swizzle: NOOP, TFLOPS: 52.96 (+5.24%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-7.765625 ', '-9.59375  '], time:5.923151ms, swizzle: NOOP, TFLOPS: 46.41
                 f16wmma(mma4x2+warp2x4): ['-7.765625 ', '-9.59375  '], time:3.649044ms, swizzle: NOOP, TFLOPS: 75.33 (+42.25%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-7.765625 ', '-9.59375  '], time:2.605748ms, swizzle: NOOP, TFLOPS: 105.49(+40.04%)
          f16wmma(mma2x4+warp2x4+stage4): ['-7.765625 ', '-9.59375  '], time:2.599406ms, swizzle: NOOP, TFLOPS: 105.75(+0.24%)
          f16wmma(mma2x4+warp2x4+stage3): ['-7.765625 ', '-9.59375  '], time:2.583003ms, swizzle: NOOP, TFLOPS: 106.42(+0.64%)
          f16wmma(mma2x4+warp2x4+stage2): ['-7.765625 ', '-9.59375  '], time:2.571964ms, swizzle: NOOP, TFLOPS: 106.87(+0.43%)
        f16wmma(mma2x4+...+stage4+dsmem): ['-7.765625 ', '-9.59375  '], time:2.590227ms, swizzle: NOOP, TFLOPS: 106.12
        f16wmma(mma2x4+...+stage3+dsmem): ['-7.765625 ', '-9.59375  '], time:2.579307ms, swizzle: NOOP, TFLOPS: 106.57
        f16wmma(mma2x4+...+stage2+dsmem): ['-7.765625 ', '-9.59375  '], time:2.568602ms, swizzle: NOOP, TFLOPS: 107.01(+0.13%)
      f16wmma(mma2x4+...+stage4+swizzle): ['-7.765625 ', '-9.59375  '], time:2.613925ms, swizzle: 2048, TFLOPS: 105.16
      f16wmma(mma2x4+...+stage3+swizzle): ['-7.765625 ', '-9.59375  '], time:2.559590ms, swizzle: 2048, TFLOPS: 107.39(+0.35%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-7.765625 ', '-9.59375  '], time:2.534532ms, swizzle: 2048, TFLOPS: 108.45(+0.99%)
       f16wmma(...+stage4+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:2.610015ms, swizzle: 2048, TFLOPS: 105.32
       f16wmma(...+stage3+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:2.585339ms, swizzle: 2048, TFLOPS: 106.32
       f16wmma(...+stage2+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:2.534747ms, swizzle: 2048, TFLOPS: 108.44
                             f16(cublas): ['-7.765625 ', '-9.59375  '], time:2.619481ms, swizzle: NOOP, TFLOPS: 104.94
                                  f16_th: ['-7.9101562', '-9.703125 '], time:2.551317ms, swizzle: NOOP, TFLOPS: 107.74
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=8192
                     f16x8pack(t8x8+bcf): ['-34.6875  ', '109.75    '], time:12.21077ms, swizzle: NOOP, TFLOPS: 45.02 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-34.6875  ', '109.75    '], time:11.94691ms, swizzle: NOOP, TFLOPS: 46.02 (+2.21%)
                f16x8pack(t8x8+k16+dbuf): ['-34.6875  ', '109.75    '], time:11.40398ms, swizzle: NOOP, TFLOPS: 48.21 (+4.76%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-34.6875  ', '108.625   '], time:12.47351ms, swizzle: NOOP, TFLOPS: 44.07
                 f16wmma(mma4x2+warp2x4): ['-34.6875  ', '108.625   '], time:7.357668ms, swizzle: NOOP, TFLOPS: 74.72 (+54.99%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-34.6875  ', '108.625   '], time:5.378651ms, swizzle: NOOP, TFLOPS: 102.21(+36.79%)
          f16wmma(mma2x4+warp2x4+stage4): ['-34.6875  ', '108.625   '], time:5.387425ms, swizzle: NOOP, TFLOPS: 102.04
          f16wmma(mma2x4+warp2x4+stage3): ['-34.6875  ', '108.625   '], time:5.188965ms, swizzle: NOOP, TFLOPS: 105.95(+3.66%)
          f16wmma(mma2x4+warp2x4+stage2): ['-34.6875  ', '108.625   '], time:5.350542ms, swizzle: NOOP, TFLOPS: 102.75
        f16wmma(mma2x4+...+stage4+dsmem): ['-34.6875  ', '108.625   '], time:5.344128ms, swizzle: NOOP, TFLOPS: 102.87
        f16wmma(mma2x4+...+stage3+dsmem): ['-34.6875  ', '108.625   '], time:5.347514ms, swizzle: NOOP, TFLOPS: 102.81
        f16wmma(mma2x4+...+stage2+dsmem): ['-34.6875  ', '108.625   '], time:5.389356ms, swizzle: NOOP, TFLOPS: 102.01
      f16wmma(mma2x4+...+stage4+swizzle): ['-34.6875  ', '108.625   '], time:5.190229ms, swizzle: 2048, TFLOPS: 105.92
      f16wmma(mma2x4+...+stage3+swizzle): ['-34.6875  ', '108.625   '], time:5.091047ms, swizzle: 2048, TFLOPS: 107.98(+1.92%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-34.6875  ', '108.625   '], time:5.043625ms, swizzle: 2048, TFLOPS: 109.00(+0.94%)
       f16wmma(...+stage4+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:5.185484ms, swizzle: 2048, TFLOPS: 106.02
       f16wmma(...+stage3+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:5.150270ms, swizzle: 2048, TFLOPS: 106.74
       f16wmma(...+stage2+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:5.045652ms, swizzle: 2048, TFLOPS: 108.96
                             f16(cublas): ['-34.6875  ', '108.625   '], time:5.080008ms, swizzle: NOOP, TFLOPS: 108.22
                                  f16_th: ['-34.9375  ', '108.625   '], time:4.943799ms, swizzle: NOOP, TFLOPS: 111.20(+2.02%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=2048
                     f16x8pack(t8x8+bcf): ['-48.625   ', '-19.59375 '], time:5.394196ms, swizzle: NOOP, TFLOPS: 50.96 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-48.625   ', '-19.59375 '], time:5.343627ms, swizzle: NOOP, TFLOPS: 51.44 (+0.95%)
                f16x8pack(t8x8+k16+dbuf): ['-48.625   ', '-19.59375 '], time:5.101990ms, swizzle: NOOP, TFLOPS: 53.88 (+4.74%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-48.6875  ', '-19.71875 '], time:5.898356ms, swizzle: NOOP, TFLOPS: 46.60
                 f16wmma(mma4x2+warp2x4): ['-48.6875  ', '-19.71875 '], time:3.552985ms, swizzle: NOOP, TFLOPS: 77.37 (+43.60%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-48.6875  ', '-19.71875 '], time:2.599167ms, swizzle: NOOP, TFLOPS: 105.76(+36.70%)
          f16wmma(mma2x4+warp2x4+stage4): ['-48.6875  ', '-19.71875 '], time:2.602124ms, swizzle: NOOP, TFLOPS: 105.64
          f16wmma(mma2x4+warp2x4+stage3): ['-48.6875  ', '-19.71875 '], time:2.547097ms, swizzle: NOOP, TFLOPS: 107.92(+2.04%)
          f16wmma(mma2x4+warp2x4+stage2): ['-48.6875  ', '-19.71875 '], time:2.559137ms, swizzle: NOOP, TFLOPS: 107.41
        f16wmma(mma2x4+...+stage4+dsmem): ['-48.6875  ', '-19.71875 '], time:2.596354ms, swizzle: NOOP, TFLOPS: 105.87
        f16wmma(mma2x4+...+stage3+dsmem): ['-48.6875  ', '-19.71875 '], time:2.562403ms, swizzle: NOOP, TFLOPS: 107.27
        f16wmma(mma2x4+...+stage2+dsmem): ['-48.6875  ', '-19.71875 '], time:2.556300ms, swizzle: NOOP, TFLOPS: 107.53
      f16wmma(mma2x4+...+stage4+swizzle): ['-48.6875  ', '-19.71875 '], time:2.617216ms, swizzle: 4096, TFLOPS: 105.03
      f16wmma(mma2x4+...+stage3+swizzle): ['-48.6875  ', '-19.71875 '], time:2.541565ms, swizzle: 4096, TFLOPS: 108.15(+0.22%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-48.6875  ', '-19.71875 '], time:2.533459ms, swizzle: 4096, TFLOPS: 108.50(+0.32%)
       f16wmma(...+stage4+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:2.629590ms, swizzle: 4096, TFLOPS: 104.53
       f16wmma(...+stage3+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:2.574324ms, swizzle: 4096, TFLOPS: 106.78
       f16wmma(...+stage2+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:2.524018ms, swizzle: 4096, TFLOPS: 108.90(+0.37%)
                             f16(cublas): ['-48.6875  ', '-19.71875 '], time:2.599477ms, swizzle: NOOP, TFLOPS: 105.74
                                  f16_th: ['-48.75    ', '-19.765625'], time:2.402782ms, swizzle: NOOP, TFLOPS: 114.40(+5.05%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=4096
                     f16x8pack(t8x8+bcf): ['-7.390625 ', '-9.75     '], time:12.03119ms, swizzle: NOOP, TFLOPS: 45.69 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-7.390625 ', '-9.75     '], time:11.66582ms, swizzle: NOOP, TFLOPS: 47.13 (+3.13%)
                f16x8pack(t8x8+k16+dbuf): ['-7.390625 ', '-9.75     '], time:11.37144ms, swizzle: NOOP, TFLOPS: 48.35 (+2.59%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-7.765625 ', '-9.59375  '], time:15.57343ms, swizzle: NOOP, TFLOPS: 35.30
                 f16wmma(mma4x2+warp2x4): ['-7.765625 ', '-9.59375  '], time:7.270383ms, swizzle: NOOP, TFLOPS: 75.62 (+56.41%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-7.765625 ', '-9.59375  '], time:5.333232ms, swizzle: NOOP, TFLOPS: 103.08(+36.32%)
          f16wmma(mma2x4+warp2x4+stage4): ['-7.765625 ', '-9.59375  '], time:5.406975ms, swizzle: NOOP, TFLOPS: 101.68
          f16wmma(mma2x4+warp2x4+stage3): ['-7.765625 ', '-9.59375  '], time:5.376362ms, swizzle: NOOP, TFLOPS: 102.25
          f16wmma(mma2x4+warp2x4+stage2): ['-7.765625 ', '-9.59375  '], time:5.213570ms, swizzle: NOOP, TFLOPS: 105.45(+2.30%)
        f16wmma(mma2x4+...+stage4+dsmem): ['-7.765625 ', '-9.59375  '], time:5.353713ms, swizzle: NOOP, TFLOPS: 102.69
        f16wmma(mma2x4+...+stage3+dsmem): ['-7.765625 ', '-9.59375  '], time:5.388331ms, swizzle: NOOP, TFLOPS: 102.03
        f16wmma(mma2x4+...+stage2+dsmem): ['-7.765625 ', '-9.59375  '], time:5.367493ms, swizzle: NOOP, TFLOPS: 102.42
      f16wmma(mma2x4+...+stage4+swizzle): ['-7.765625 ', '-9.59375  '], time:5.185770ms, swizzle: 4096, TFLOPS: 106.01(+0.54%)
      f16wmma(mma2x4+...+stage3+swizzle): ['-7.765625 ', '-9.59375  '], time:5.021572ms, swizzle: 4096, TFLOPS: 109.48(+3.27%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-7.765625 ', '-9.59375  '], time:4.987549ms, swizzle: 4096, TFLOPS: 110.23(+0.68%)
       f16wmma(...+stage4+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:5.108428ms, swizzle: 4096, TFLOPS: 107.62
       f16wmma(...+stage3+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:5.049347ms, swizzle: 4096, TFLOPS: 108.88
       f16wmma(...+stage2+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:4.974079ms, swizzle: 4096, TFLOPS: 110.52(+0.27%)
                             f16(cublas): ['-7.765625 ', '-9.59375  '], time:4.976677ms, swizzle: NOOP, TFLOPS: 110.47
                                  f16_th: ['-7.9101562', '-9.703125 '], time:4.902982ms, swizzle: NOOP, TFLOPS: 112.13(+1.45%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=8192
                     f16x8pack(t8x8+bcf): ['-34.6875  ', '109.75    '], time:24.99871ms, swizzle: NOOP, TFLOPS: 43.98 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-34.6875  ', '109.75    '], time:25.10454ms, swizzle: NOOP, TFLOPS: 43.80
                f16x8pack(t8x8+k16+dbuf): ['-34.6875  ', '109.75    '], time:24.46808ms, swizzle: NOOP, TFLOPS: 44.94 (+2.17%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-34.6875  ', '108.625   '], time:31.00190ms, swizzle: NOOP, TFLOPS: 35.47
                 f16wmma(mma4x2+warp2x4): ['-34.6875  ', '108.625   '], time:14.35732ms, swizzle: NOOP, TFLOPS: 76.58 (+70.42%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-34.6875  ', '108.625   '], time:12.56091ms, swizzle: NOOP, TFLOPS: 87.53 (+14.30%)
          f16wmma(mma2x4+warp2x4+stage4): ['-34.6875  ', '108.625   '], time:12.45114ms, swizzle: NOOP, TFLOPS: 88.31 (+0.88%)
          f16wmma(mma2x4+warp2x4+stage3): ['-34.6875  ', '108.625   '], time:12.46438ms, swizzle: NOOP, TFLOPS: 88.21
          f16wmma(mma2x4+warp2x4+stage2): ['-34.6875  ', '108.625   '], time:12.39051ms, swizzle: NOOP, TFLOPS: 88.74 (+0.49%)
        f16wmma(mma2x4+...+stage4+dsmem): ['-34.6875  ', '108.625   '], time:12.44428ms, swizzle: NOOP, TFLOPS: 88.35
        f16wmma(mma2x4+...+stage3+dsmem): ['-34.6875  ', '108.625   '], time:12.48049ms, swizzle: NOOP, TFLOPS: 88.10
        f16wmma(mma2x4+...+stage2+dsmem): ['-34.6875  ', '108.625   '], time:12.36248ms, swizzle: NOOP, TFLOPS: 88.94 (+0.23%)
      f16wmma(mma2x4+...+stage4+swizzle): ['-34.6875  ', '108.625   '], time:10.31520ms, swizzle: 4096, TFLOPS: 106.59(+19.85%)
      f16wmma(mma2x4+...+stage3+swizzle): ['-34.6875  ', '108.625   '], time:10.19382ms, swizzle: 4096, TFLOPS: 107.86(+1.19%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-34.6875  ', '108.625   '], time:10.03310ms, swizzle: 4096, TFLOPS: 109.59(+1.60%)
       f16wmma(...+stage4+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:10.31091ms, swizzle: 4096, TFLOPS: 106.64
       f16wmma(...+stage3+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:10.25998ms, swizzle: 4096, TFLOPS: 107.17
       f16wmma(...+stage2+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:9.964489ms, swizzle: 4096, TFLOPS: 110.34(+0.69%)
                             f16(cublas): ['-34.6875  ', '108.625   '], time:9.730339ms, swizzle: NOOP, TFLOPS: 113.00(+2.41%)
                                  f16_th: ['-34.9375  ', '108.625   '], time:9.733605ms, swizzle: NOOP, TFLOPS: 112.96
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=2048
                     f16x8pack(t8x8+bcf): ['-48.625   ', '-19.59375 '], time:2.748227ms, swizzle: NOOP, TFLOPS: 50.01 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-48.625   ', '-19.59375 '], time:2.679944ms, swizzle: NOOP, TFLOPS: 51.28 (+2.55%)
                f16x8pack(t8x8+k16+dbuf): ['-48.625   ', '-19.59375 '], time:2.553391ms, swizzle: NOOP, TFLOPS: 53.83 (+4.96%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-48.6875  ', '-19.71875 '], time:2.964806ms, swizzle: NOOP, TFLOPS: 46.36
                 f16wmma(mma4x2+warp2x4): ['-48.6875  ', '-19.71875 '], time:1.839733ms, swizzle: NOOP, TFLOPS: 74.71 (+38.79%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-48.6875  ', '-19.71875 '], time:1.338410ms, swizzle: NOOP, TFLOPS: 102.69(+37.46%)
          f16wmma(mma2x4+warp2x4+stage4): ['-48.6875  ', '-19.71875 '], time:1.342177ms, swizzle: NOOP, TFLOPS: 102.40
          f16wmma(mma2x4+warp2x4+stage3): ['-48.6875  ', '-19.71875 '], time:1.316523ms, swizzle: NOOP, TFLOPS: 104.40(+1.66%)
          f16wmma(mma2x4+warp2x4+stage2): ['-48.6875  ', '-19.71875 '], time:1.317405ms, swizzle: NOOP, TFLOPS: 104.33
        f16wmma(mma2x4+...+stage4+dsmem): ['-48.6875  ', '-19.71875 '], time:1.334977ms, swizzle: NOOP, TFLOPS: 102.95
        f16wmma(mma2x4+...+stage3+dsmem): ['-48.6875  ', '-19.71875 '], time:1.319456ms, swizzle: NOOP, TFLOPS: 104.16
        f16wmma(mma2x4+...+stage2+dsmem): ['-48.6875  ', '-19.71875 '], time:1.315546ms, swizzle: NOOP, TFLOPS: 104.47(+0.07%)
      f16wmma(mma2x4+...+stage4+swizzle): ['-48.6875  ', '-19.71875 '], time:1.343369ms, swizzle: 1024, TFLOPS: 102.31
      f16wmma(mma2x4+...+stage3+swizzle): ['-48.6875  ', '-19.71875 '], time:1.307725ms, swizzle: 1024, TFLOPS: 105.10(+0.60%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-48.6875  ', '-19.71875 '], time:1.297426ms, swizzle: 1024, TFLOPS: 105.93(+0.79%)
       f16wmma(...+stage4+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:1.345181ms, swizzle: 1024, TFLOPS: 102.17
       f16wmma(...+stage3+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:1.321530ms, swizzle: 1024, TFLOPS: 104.00
       f16wmma(...+stage2+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:1.297283ms, swizzle: 1024, TFLOPS: 105.94(+0.01%)
                             f16(cublas): ['-48.6875  ', '-19.71875 '], time:1.478171ms, swizzle: NOOP, TFLOPS: 92.98
                                  f16_th: ['-48.75    ', '-19.765625'], time:1.302719ms, swizzle: NOOP, TFLOPS: 105.50
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=4096
                     f16x8pack(t8x8+bcf): ['-7.390625 ', '-9.75     '], time:5.530977ms, swizzle: NOOP, TFLOPS: 49.70 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-7.390625 ', '-9.75     '], time:5.496430ms, swizzle: NOOP, TFLOPS: 50.01 (+0.63%)
                f16x8pack(t8x8+k16+dbuf): ['-7.390625 ', '-9.75     '], time:5.250906ms, swizzle: NOOP, TFLOPS: 52.35 (+4.68%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-7.765625 ', '-9.59375  '], time:5.908870ms, swizzle: NOOP, TFLOPS: 46.52
                 f16wmma(mma4x2+warp2x4): ['-7.765625 ', '-9.59375  '], time:3.609466ms, swizzle: NOOP, TFLOPS: 76.15 (+45.48%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-7.765625 ', '-9.59375  '], time:2.606987ms, swizzle: NOOP, TFLOPS: 105.44(+38.45%)
          f16wmma(mma2x4+warp2x4+stage4): ['-7.765625 ', '-9.59375  '], time:2.613306ms, swizzle: NOOP, TFLOPS: 105.18
          f16wmma(mma2x4+warp2x4+stage3): ['-7.765625 ', '-9.59375  '], time:2.563500ms, swizzle: NOOP, TFLOPS: 107.23(+1.70%)
          f16wmma(mma2x4+warp2x4+stage2): ['-7.765625 ', '-9.59375  '], time:2.575325ms, swizzle: NOOP, TFLOPS: 106.74
        f16wmma(mma2x4+...+stage4+dsmem): ['-7.765625 ', '-9.59375  '], time:2.595806ms, swizzle: NOOP, TFLOPS: 105.89
        f16wmma(mma2x4+...+stage3+dsmem): ['-7.765625 ', '-9.59375  '], time:2.583646ms, swizzle: NOOP, TFLOPS: 106.39
        f16wmma(mma2x4+...+stage2+dsmem): ['-7.765625 ', '-9.59375  '], time:2.573943ms, swizzle: NOOP, TFLOPS: 106.79
      f16wmma(mma2x4+...+stage4+swizzle): ['-7.765625 ', '-9.59375  '], time:2.617716ms, swizzle: 1024, TFLOPS: 105.01
      f16wmma(mma2x4+...+stage3+swizzle): ['-7.765625 ', '-9.59375  '], time:2.564501ms, swizzle: 1024, TFLOPS: 107.19
      f16wmma(mma2x4+...+stage2+swizzle): ['-7.765625 ', '-9.59375  '], time:2.539563ms, swizzle: 1024, TFLOPS: 108.24(+0.94%)
       f16wmma(...+stage4+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:2.615571ms, swizzle: 1024, TFLOPS: 105.09
       f16wmma(...+stage3+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:2.590703ms, swizzle: 1024, TFLOPS: 106.10
       f16wmma(...+stage2+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:2.539587ms, swizzle: 1024, TFLOPS: 108.24
                             f16(cublas): ['-7.765625 ', '-9.59375  '], time:2.676177ms, swizzle: NOOP, TFLOPS: 102.71
                                  f16_th: ['-7.9101562', '-9.703125 '], time:2.554368ms, swizzle: NOOP, TFLOPS: 107.61
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=8192
                     f16x8pack(t8x8+bcf): ['-34.6875  ', '109.75    '], time:11.29407ms, swizzle: NOOP, TFLOPS: 48.68 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-34.6875  ', '109.75    '], time:11.21764ms, swizzle: NOOP, TFLOPS: 49.01 (+0.68%)
                f16x8pack(t8x8+k16+dbuf): ['-34.6875  ', '109.75    '], time:10.83180ms, swizzle: NOOP, TFLOPS: 50.75 (+3.56%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-34.6875  ', '108.625   '], time:11.73310ms, swizzle: NOOP, TFLOPS: 46.86
                 f16wmma(mma4x2+warp2x4): ['-34.6875  ', '108.625   '], time:7.149004ms, swizzle: NOOP, TFLOPS: 76.90 (+51.51%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-34.6875  ', '108.625   '], time:5.116224ms, swizzle: NOOP, TFLOPS: 107.45(+39.73%)
          f16wmma(mma2x4+warp2x4+stage4): ['-34.6875  ', '108.625   '], time:5.128741ms, swizzle: NOOP, TFLOPS: 107.19
          f16wmma(mma2x4+warp2x4+stage3): ['-34.6875  ', '108.625   '], time:5.057311ms, swizzle: NOOP, TFLOPS: 108.71(+1.16%)
          f16wmma(mma2x4+warp2x4+stage2): ['-34.6875  ', '108.625   '], time:5.109739ms, swizzle: NOOP, TFLOPS: 107.59
        f16wmma(mma2x4+...+stage4+dsmem): ['-34.6875  ', '108.625   '], time:5.136680ms, swizzle: NOOP, TFLOPS: 107.03
        f16wmma(mma2x4+...+stage3+dsmem): ['-34.6875  ', '108.625   '], time:5.097627ms, swizzle: NOOP, TFLOPS: 107.85
        f16wmma(mma2x4+...+stage2+dsmem): ['-34.6875  ', '108.625   '], time:5.075955ms, swizzle: NOOP, TFLOPS: 108.31
      f16wmma(mma2x4+...+stage4+swizzle): ['-34.6875  ', '108.625   '], time:5.198264ms, swizzle: 1024, TFLOPS: 105.76
      f16wmma(mma2x4+...+stage3+swizzle): ['-34.6875  ', '108.625   '], time:5.154633ms, swizzle: 1024, TFLOPS: 106.65
      f16wmma(mma2x4+...+stage2+swizzle): ['-34.6875  ', '108.625   '], time:5.041408ms, swizzle: 1024, TFLOPS: 109.05(+0.32%)
       f16wmma(...+stage4+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:5.199956ms, swizzle: 1024, TFLOPS: 105.72
       f16wmma(...+stage3+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:5.206274ms, swizzle: 1024, TFLOPS: 105.59
       f16wmma(...+stage2+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:5.060267ms, swizzle: 1024, TFLOPS: 108.64
                             f16(cublas): ['-34.6875  ', '108.625   '], time:5.067610ms, swizzle: NOOP, TFLOPS: 108.48
                                  f16_th: ['-34.90625 ', '108.5625  '], time:4.848861ms, swizzle: NOOP, TFLOPS: 113.38(+3.97%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=2048
                     f16x8pack(t8x8+bcf): ['-48.625   ', '-19.59375 '], time:5.350041ms, swizzle: NOOP, TFLOPS: 51.38 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-48.625   ', '-19.59375 '], time:5.371356ms, swizzle: NOOP, TFLOPS: 51.17
                f16x8pack(t8x8+k16+dbuf): ['-48.625   ', '-19.59375 '], time:5.128169ms, swizzle: NOOP, TFLOPS: 53.60 (+4.33%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-48.6875  ', '-19.71875 '], time:5.927252ms, swizzle: NOOP, TFLOPS: 46.38
                 f16wmma(mma4x2+warp2x4): ['-48.6875  ', '-19.71875 '], time:3.571295ms, swizzle: NOOP, TFLOPS: 76.97 (+43.59%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-48.6875  ', '-19.71875 '], time:2.595305ms, swizzle: NOOP, TFLOPS: 105.91(+37.61%)
          f16wmma(mma2x4+warp2x4+stage4): ['-48.6875  ', '-19.71875 '], time:2.612948ms, swizzle: NOOP, TFLOPS: 105.20
          f16wmma(mma2x4+warp2x4+stage3): ['-48.6875  ', '-19.71875 '], time:2.542471ms, swizzle: NOOP, TFLOPS: 108.11(+2.08%)
          f16wmma(mma2x4+warp2x4+stage2): ['-48.6875  ', '-19.71875 '], time:2.558398ms, swizzle: NOOP, TFLOPS: 107.44
        f16wmma(mma2x4+...+stage4+dsmem): ['-48.6875  ', '-19.71875 '], time:2.595710ms, swizzle: NOOP, TFLOPS: 105.90
        f16wmma(mma2x4+...+stage3+dsmem): ['-48.6875  ', '-19.71875 '], time:2.560544ms, swizzle: NOOP, TFLOPS: 107.35
        f16wmma(mma2x4+...+stage2+dsmem): ['-48.6875  ', '-19.71875 '], time:2.550745ms, swizzle: NOOP, TFLOPS: 107.76
      f16wmma(mma2x4+...+stage4+swizzle): ['-48.6875  ', '-19.71875 '], time:2.618861ms, swizzle: 2048, TFLOPS: 104.96
      f16wmma(mma2x4+...+stage3+swizzle): ['-48.6875  ', '-19.71875 '], time:2.540159ms, swizzle: 2048, TFLOPS: 108.21(+0.09%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-48.6875  ', '-19.71875 '], time:2.523684ms, swizzle: 2048, TFLOPS: 108.92(+0.65%)
       f16wmma(...+stage4+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:2.618551ms, swizzle: 2048, TFLOPS: 104.97
       f16wmma(...+stage3+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:2.562999ms, swizzle: 2048, TFLOPS: 107.25
       f16wmma(...+stage2+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:2.522230ms, swizzle: 2048, TFLOPS: 108.98(+0.06%)
                             f16(cublas): ['-48.6875  ', '-19.71875 '], time:2.600812ms, swizzle: NOOP, TFLOPS: 105.69
                                  f16_th: ['-48.75    ', '-19.765625'], time:2.399682ms, swizzle: NOOP, TFLOPS: 114.55(+5.11%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=4096
                     f16x8pack(t8x8+bcf): ['-7.390625 ', '-9.75     '], time:10.91930ms, swizzle: NOOP, TFLOPS: 50.35 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-7.390625 ', '-9.75     '], time:10.92619ms, swizzle: NOOP, TFLOPS: 50.32
                f16x8pack(t8x8+k16+dbuf): ['-7.390625 ', '-9.75     '], time:10.35106ms, swizzle: NOOP, TFLOPS: 53.11 (+5.49%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-7.765625 ', '-9.59375  '], time:11.71412ms, swizzle: NOOP, TFLOPS: 46.93
                 f16wmma(mma4x2+warp2x4): ['-7.765625 ', '-9.59375  '], time:7.003331ms, swizzle: NOOP, TFLOPS: 78.50 (+47.80%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-7.765625 ', '-9.59375  '], time:5.050349ms, swizzle: NOOP, TFLOPS: 108.86(+38.67%)
          f16wmma(mma2x4+warp2x4+stage4): ['-7.765625 ', '-9.59375  '], time:5.069613ms, swizzle: NOOP, TFLOPS: 108.44
          f16wmma(mma2x4+warp2x4+stage3): ['-7.765625 ', '-9.59375  '], time:5.004692ms, swizzle: NOOP, TFLOPS: 109.85(+0.91%)
          f16wmma(mma2x4+warp2x4+stage2): ['-7.765625 ', '-9.59375  '], time:5.027818ms, swizzle: NOOP, TFLOPS: 109.34
        f16wmma(mma2x4+...+stage4+dsmem): ['-7.765625 ', '-9.59375  '], time:5.068135ms, swizzle: NOOP, TFLOPS: 108.47
        f16wmma(mma2x4+...+stage3+dsmem): ['-7.765625 ', '-9.59375  '], time:5.042505ms, swizzle: NOOP, TFLOPS: 109.02
        f16wmma(mma2x4+...+stage2+dsmem): ['-7.765625 ', '-9.59375  '], time:5.024003ms, swizzle: NOOP, TFLOPS: 109.43
      f16wmma(mma2x4+...+stage4+swizzle): ['-7.765625 ', '-9.59375  '], time:5.120134ms, swizzle: 2048, TFLOPS: 107.37
      f16wmma(mma2x4+...+stage3+swizzle): ['-7.765625 ', '-9.59375  '], time:5.047202ms, swizzle: 2048, TFLOPS: 108.92
      f16wmma(mma2x4+...+stage2+swizzle): ['-7.765625 ', '-9.59375  '], time:5.019378ms, swizzle: 2048, TFLOPS: 109.53
       f16wmma(...+stage4+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:5.148577ms, swizzle: 2048, TFLOPS: 106.78
       f16wmma(...+stage3+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:5.109333ms, swizzle: 2048, TFLOPS: 107.60
       f16wmma(...+stage2+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:4.964089ms, swizzle: 2048, TFLOPS: 110.75(+0.82%)
                             f16(cublas): ['-7.765625 ', '-9.59375  '], time:5.002450ms, swizzle: NOOP, TFLOPS: 109.90
                                  f16_th: ['-7.9101562', '-9.703125 '], time:4.903984ms, swizzle: NOOP, TFLOPS: 112.10(+1.23%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=8192
                     f16x8pack(t8x8+bcf): ['-34.6875  ', '109.75    '], time:24.32680ms, swizzle: NOOP, TFLOPS: 45.20 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-34.6875  ', '109.75    '], time:24.22029ms, swizzle: NOOP, TFLOPS: 45.40 (+0.44%)
                f16x8pack(t8x8+k16+dbuf): ['-34.6875  ', '109.75    '], time:23.78268ms, swizzle: NOOP, TFLOPS: 46.23 (+1.84%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-34.6875  ', '108.625   '], time:25.18262ms, swizzle: NOOP, TFLOPS: 43.66
                 f16wmma(mma4x2+warp2x4): ['-34.6875  ', '108.625   '], time:14.21759ms, swizzle: NOOP, TFLOPS: 77.33 (+67.28%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-34.6875  ', '108.625   '], time:10.99987ms, swizzle: NOOP, TFLOPS: 99.96 (+29.25%)
          f16wmma(mma2x4+warp2x4+stage4): ['-34.6875  ', '108.625   '], time:11.02399ms, swizzle: NOOP, TFLOPS: 99.74
          f16wmma(mma2x4+warp2x4+stage3): ['-34.6875  ', '108.625   '], time:11.08629ms, swizzle: NOOP, TFLOPS: 99.18
          f16wmma(mma2x4+warp2x4+stage2): ['-34.6875  ', '108.625   '], time:10.86187ms, swizzle: NOOP, TFLOPS: 101.23(+1.27%)
        f16wmma(mma2x4+...+stage4+dsmem): ['-34.6875  ', '108.625   '], time:11.01264ms, swizzle: NOOP, TFLOPS: 99.84
        f16wmma(mma2x4+...+stage3+dsmem): ['-34.6875  ', '108.625   '], time:10.96932ms, swizzle: NOOP, TFLOPS: 100.24
        f16wmma(mma2x4+...+stage2+dsmem): ['-34.6875  ', '108.625   '], time:10.93618ms, swizzle: NOOP, TFLOPS: 100.54
      f16wmma(mma2x4+...+stage4+swizzle): ['-34.6875  ', '108.625   '], time:10.27595ms, swizzle: 2048, TFLOPS: 107.00(+5.70%)
      f16wmma(mma2x4+...+stage3+swizzle): ['-34.6875  ', '108.625   '], time:10.13357ms, swizzle: 2048, TFLOPS: 108.50(+1.41%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-34.6875  ', '108.625   '], time:9.931540ms, swizzle: 2048, TFLOPS: 110.71(+2.03%)
       f16wmma(...+stage4+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:10.26072ms, swizzle: 2048, TFLOPS: 107.16
       f16wmma(...+stage3+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:10.19623ms, swizzle: 2048, TFLOPS: 107.84
       f16wmma(...+stage2+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:10.00247ms, swizzle: 2048, TFLOPS: 109.92
                             f16(cublas): ['-34.6875  ', '108.625   '], time:9.718918ms, swizzle: NOOP, TFLOPS: 113.13(+2.19%)
                                  f16_th: ['-34.9375  ', '108.625   '], time:9.733724ms, swizzle: NOOP, TFLOPS: 112.96
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=2048
                     f16x8pack(t8x8+bcf): ['-48.625   ', '-19.59375 '], time:11.07859ms, swizzle: NOOP, TFLOPS: 49.62 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-48.625   ', '-19.59375 '], time:10.94355ms, swizzle: NOOP, TFLOPS: 50.24 (+1.23%)
                f16x8pack(t8x8+k16+dbuf): ['-48.625   ', '-19.59375 '], time:10.53507ms, swizzle: NOOP, TFLOPS: 52.18 (+3.88%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-48.6875  ', '-19.71875 '], time:11.76457ms, swizzle: NOOP, TFLOPS: 46.73
                 f16wmma(mma4x2+warp2x4): ['-48.6875  ', '-19.71875 '], time:6.933164ms, swizzle: NOOP, TFLOPS: 79.29 (+51.95%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-48.6875  ', '-19.71875 '], time:5.144977ms, swizzle: NOOP, TFLOPS: 106.85(+34.76%)
          f16wmma(mma2x4+warp2x4+stage4): ['-48.6875  ', '-19.71875 '], time:5.183672ms, swizzle: NOOP, TFLOPS: 106.06
          f16wmma(mma2x4+warp2x4+stage3): ['-48.6875  ', '-19.71875 '], time:5.052256ms, swizzle: NOOP, TFLOPS: 108.81(+1.84%)
          f16wmma(mma2x4+warp2x4+stage2): ['-48.6875  ', '-19.71875 '], time:5.082416ms, swizzle: NOOP, TFLOPS: 108.17
        f16wmma(mma2x4+...+stage4+dsmem): ['-48.6875  ', '-19.71875 '], time:5.201864ms, swizzle: NOOP, TFLOPS: 105.68
        f16wmma(mma2x4+...+stage3+dsmem): ['-48.6875  ', '-19.71875 '], time:5.106210ms, swizzle: NOOP, TFLOPS: 107.66
        f16wmma(mma2x4+...+stage2+dsmem): ['-48.6875  ', '-19.71875 '], time:5.088162ms, swizzle: NOOP, TFLOPS: 108.05
      f16wmma(mma2x4+...+stage4+swizzle): ['-48.6875  ', '-19.71875 '], time:5.184173ms, swizzle: 4096, TFLOPS: 106.05
      f16wmma(mma2x4+...+stage3+swizzle): ['-48.6875  ', '-19.71875 '], time:5.041408ms, swizzle: 4096, TFLOPS: 109.05(+0.22%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-48.6875  ', '-19.71875 '], time:5.055046ms, swizzle: 4096, TFLOPS: 108.75
       f16wmma(...+stage4+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:5.221557ms, swizzle: 4096, TFLOPS: 105.29
       f16wmma(...+stage3+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:5.159568ms, swizzle: 4096, TFLOPS: 106.55
       f16wmma(...+stage2+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:5.045223ms, swizzle: 4096, TFLOPS: 108.97
                             f16(cublas): ['-48.6875  ', '-19.71875 '], time:5.022454ms, swizzle: NOOP, TFLOPS: 109.46(+0.38%)
                                  f16_th: ['-48.75    ', '-19.765625'], time:4.814910ms, swizzle: NOOP, TFLOPS: 114.18(+4.31%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=4096
                     f16x8pack(t8x8+bcf): ['-7.390625 ', '-9.75     '], time:24.28481ms, swizzle: NOOP, TFLOPS: 45.28 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-7.390625 ', '-9.75     '], time:24.54431ms, swizzle: NOOP, TFLOPS: 44.80
                f16x8pack(t8x8+k16+dbuf): ['-7.390625 ', '-9.75     '], time:23.84874ms, swizzle: NOOP, TFLOPS: 46.10 (+1.83%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-7.765625 ', '-9.59375  '], time:31.31301ms, swizzle: NOOP, TFLOPS: 35.11
                 f16wmma(mma4x2+warp2x4): ['-7.765625 ', '-9.59375  '], time:14.29131ms, swizzle: NOOP, TFLOPS: 76.94 (+66.88%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-7.765625 ', '-9.59375  '], time:11.20250ms, swizzle: NOOP, TFLOPS: 98.15 (+27.57%)
          f16wmma(mma2x4+warp2x4+stage4): ['-7.765625 ', '-9.59375  '], time:11.39490ms, swizzle: NOOP, TFLOPS: 96.49
          f16wmma(mma2x4+warp2x4+stage3): ['-7.765625 ', '-9.59375  '], time:11.49122ms, swizzle: NOOP, TFLOPS: 95.68
          f16wmma(mma2x4+warp2x4+stage2): ['-7.765625 ', '-9.59375  '], time:11.13839ms, swizzle: NOOP, TFLOPS: 98.71 (+0.58%)
        f16wmma(mma2x4+...+stage4+dsmem): ['-7.765625 ', '-9.59375  '], time:11.54367ms, swizzle: NOOP, TFLOPS: 95.25
        f16wmma(mma2x4+...+stage3+dsmem): ['-7.765625 ', '-9.59375  '], time:11.50746ms, swizzle: NOOP, TFLOPS: 95.55
        f16wmma(mma2x4+...+stage2+dsmem): ['-7.765625 ', '-9.59375  '], time:11.11392ms, swizzle: NOOP, TFLOPS: 98.93 (+0.22%)
      f16wmma(mma2x4+...+stage4+swizzle): ['-7.765625 ', '-9.59375  '], time:10.30538ms, swizzle: 4096, TFLOPS: 106.69(+7.85%)
      f16wmma(mma2x4+...+stage3+swizzle): ['-7.765625 ', '-9.59375  '], time:10.13453ms, swizzle: 4096, TFLOPS: 108.49(+1.69%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-7.765625 ', '-9.59375  '], time:10.02380ms, swizzle: 4096, TFLOPS: 109.69(+1.10%)
       f16wmma(...+stage4+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:10.28742ms, swizzle: 4096, TFLOPS: 106.88
       f16wmma(...+stage3+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:10.25483ms, swizzle: 4096, TFLOPS: 107.22
       f16wmma(...+stage2+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:10.08176ms, swizzle: 4096, TFLOPS: 109.06
                             f16(cublas): ['-7.765625 ', '-9.59375  '], time:9.751081ms, swizzle: NOOP, TFLOPS: 112.76(+2.80%)
                                  f16_th: ['-7.9101562', '-9.703125 '], time:9.577250ms, swizzle: NOOP, TFLOPS: 114.80(+1.82%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=8192
                     f16x8pack(t8x8+bcf): ['-34.6875  ', '109.75    '], time:50.14533ms, swizzle: NOOP, TFLOPS: 43.85 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-34.6875  ', '109.75    '], time:50.38545ms, swizzle: NOOP, TFLOPS: 43.64
                f16x8pack(t8x8+k16+dbuf): ['-34.6875  ', '109.75    '], time:49.45671ms, swizzle: NOOP, TFLOPS: 44.46 (+1.39%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-34.6875  ', '108.625   '], time:61.91453ms, swizzle: NOOP, TFLOPS: 35.52
                 f16wmma(mma4x2+warp2x4): ['-34.6875  ', '108.625   '], time:28.20057ms, swizzle: NOOP, TFLOPS: 77.98 (+75.37%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-34.6875  ', '108.625   '], time:25.29430ms, swizzle: NOOP, TFLOPS: 86.94 (+11.49%)
          f16wmma(mma2x4+warp2x4+stage4): ['-34.6875  ', '108.625   '], time:24.86543ms, swizzle: NOOP, TFLOPS: 88.44 (+1.72%)
          f16wmma(mma2x4+warp2x4+stage3): ['-34.6875  ', '108.625   '], time:24.93984ms, swizzle: NOOP, TFLOPS: 88.17
          f16wmma(mma2x4+warp2x4+stage2): ['-34.6875  ', '108.625   '], time:24.97000ms, swizzle: NOOP, TFLOPS: 88.07
        f16wmma(mma2x4+...+stage4+dsmem): ['-34.6875  ', '108.625   '], time:24.86479ms, swizzle: NOOP, TFLOPS: 88.44 (+0.00%)
        f16wmma(mma2x4+...+stage3+dsmem): ['-34.6875  ', '108.625   '], time:24.94227ms, swizzle: NOOP, TFLOPS: 88.16
        f16wmma(mma2x4+...+stage2+dsmem): ['-34.6875  ', '108.625   '], time:24.95753ms, swizzle: NOOP, TFLOPS: 88.11
      f16wmma(mma2x4+...+stage4+swizzle): ['-34.6875  ', '108.625   '], time:20.66848ms, swizzle: 4096, TFLOPS: 106.40(+20.30%)
      f16wmma(mma2x4+...+stage3+swizzle): ['-34.6875  ', '108.625   '], time:20.48211ms, swizzle: 4096, TFLOPS: 107.36(+0.91%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-34.6875  ', '108.625   '], time:20.18594ms, swizzle: 4096, TFLOPS: 108.94(+1.47%)
       f16wmma(...+stage4+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:20.59302ms, swizzle: 4096, TFLOPS: 106.78
       f16wmma(...+stage3+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:20.59633ms, swizzle: 4096, TFLOPS: 106.77
       f16wmma(...+stage2+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:20.18511ms, swizzle: 4096, TFLOPS: 108.94(+0.00%)
                             f16(cublas): ['-34.6875  ', '108.625   '], time:19.39537ms, swizzle: NOOP, TFLOPS: 113.38(+4.07%)
                                  f16_th: ['-34.9375  ', '108.625   '], time:19.39315ms, swizzle: NOOP, TFLOPS: 113.39(+0.01%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=2048
                     f16x8pack(t8x8+bcf): ['-48.625   ', '-19.59375 '], time:5.363512ms, swizzle: NOOP, TFLOPS: 51.25 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-48.625   ', '-19.59375 '], time:5.482912ms, swizzle: NOOP, TFLOPS: 50.13
                f16x8pack(t8x8+k16+dbuf): ['-48.625   ', '-19.59375 '], time:5.202841ms, swizzle: NOOP, TFLOPS: 52.83 (+3.09%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-48.6875  ', '-19.71875 '], time:5.953073ms, swizzle: NOOP, TFLOPS: 46.17
                 f16wmma(mma4x2+warp2x4): ['-48.6875  ', '-19.71875 '], time:3.572988ms, swizzle: NOOP, TFLOPS: 76.93 (+45.62%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-48.6875  ', '-19.71875 '], time:2.598595ms, swizzle: NOOP, TFLOPS: 105.78(+37.50%)
          f16wmma(mma2x4+warp2x4+stage4): ['-48.6875  ', '-19.71875 '], time:2.609777ms, swizzle: NOOP, TFLOPS: 105.33
          f16wmma(mma2x4+warp2x4+stage3): ['-48.6875  ', '-19.71875 '], time:2.544808ms, swizzle: NOOP, TFLOPS: 108.02(+2.11%)
          f16wmma(mma2x4+warp2x4+stage2): ['-48.6875  ', '-19.71875 '], time:2.558016ms, swizzle: NOOP, TFLOPS: 107.46
        f16wmma(mma2x4+...+stage4+dsmem): ['-48.6875  ', '-19.71875 '], time:2.594542ms, swizzle: NOOP, TFLOPS: 105.94
        f16wmma(mma2x4+...+stage3+dsmem): ['-48.6875  ', '-19.71875 '], time:2.561473ms, swizzle: NOOP, TFLOPS: 107.31
        f16wmma(mma2x4+...+stage2+dsmem): ['-48.6875  ', '-19.71875 '], time:2.553391ms, swizzle: NOOP, TFLOPS: 107.65
      f16wmma(mma2x4+...+stage4+swizzle): ['-48.6875  ', '-19.71875 '], time:2.616786ms, swizzle: 1024, TFLOPS: 105.04
      f16wmma(mma2x4+...+stage3+swizzle): ['-48.6875  ', '-19.71875 '], time:2.541470ms, swizzle: 1024, TFLOPS: 108.16(+0.13%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-48.6875  ', '-19.71875 '], time:2.533721ms, swizzle: 1024, TFLOPS: 108.49(+0.31%)
       f16wmma(...+stage4+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:2.628636ms, swizzle: 1024, TFLOPS: 104.57
       f16wmma(...+stage3+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:2.572536ms, swizzle: 1024, TFLOPS: 106.85
       f16wmma(...+stage2+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:2.531862ms, swizzle: 1024, TFLOPS: 108.57(+0.07%)
                             f16(cublas): ['-48.6875  ', '-19.71875 '], time:2.625203ms, swizzle: NOOP, TFLOPS: 104.71
                                  f16_th: ['-48.75    ', '-19.765625'], time:2.406597ms, swizzle: NOOP, TFLOPS: 114.22(+5.21%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=4096
                     f16x8pack(t8x8+bcf): ['-7.390625 ', '-9.75     '], time:11.23223ms, swizzle: NOOP, TFLOPS: 48.94 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-7.390625 ', '-9.75     '], time:11.18514ms, swizzle: NOOP, TFLOPS: 49.15 (+0.42%)
                f16x8pack(t8x8+k16+dbuf): ['-7.390625 ', '-9.75     '], time:10.72626ms, swizzle: NOOP, TFLOPS: 51.25 (+4.28%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-7.765625 ', '-9.59375  '], time:11.72261ms, swizzle: NOOP, TFLOPS: 46.90
                 f16wmma(mma4x2+warp2x4): ['-7.765625 ', '-9.59375  '], time:7.008528ms, swizzle: NOOP, TFLOPS: 78.44 (+53.05%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-7.765625 ', '-9.59375  '], time:5.050563ms, swizzle: NOOP, TFLOPS: 108.85(+38.77%)
          f16wmma(mma2x4+warp2x4+stage4): ['-7.765625 ', '-9.59375  '], time:5.105209ms, swizzle: NOOP, TFLOPS: 107.69
          f16wmma(mma2x4+warp2x4+stage3): ['-7.765625 ', '-9.59375  '], time:5.026364ms, swizzle: NOOP, TFLOPS: 109.37(+0.48%)
          f16wmma(mma2x4+warp2x4+stage2): ['-7.765625 ', '-9.59375  '], time:5.057692ms, swizzle: NOOP, TFLOPS: 108.70
        f16wmma(mma2x4+...+stage4+dsmem): ['-7.765625 ', '-9.59375  '], time:5.067157ms, swizzle: NOOP, TFLOPS: 108.49
        f16wmma(mma2x4+...+stage3+dsmem): ['-7.765625 ', '-9.59375  '], time:5.055761ms, swizzle: NOOP, TFLOPS: 108.74
        f16wmma(mma2x4+...+stage2+dsmem): ['-7.765625 ', '-9.59375  '], time:5.043077ms, swizzle: NOOP, TFLOPS: 109.01
      f16wmma(mma2x4+...+stage4+swizzle): ['-7.765625 ', '-9.59375  '], time:5.160617ms, swizzle: 1024, TFLOPS: 106.53
      f16wmma(mma2x4+...+stage3+swizzle): ['-7.765625 ', '-9.59375  '], time:5.114912ms, swizzle: 1024, TFLOPS: 107.48
      f16wmma(mma2x4+...+stage2+swizzle): ['-7.765625 ', '-9.59375  '], time:4.990029ms, swizzle: 1024, TFLOPS: 110.17(+0.73%)
       f16wmma(...+stage4+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:5.138397ms, swizzle: 1024, TFLOPS: 106.99
       f16wmma(...+stage3+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:5.164527ms, swizzle: 1024, TFLOPS: 106.45
       f16wmma(...+stage2+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:5.023026ms, swizzle: 1024, TFLOPS: 109.45
                             f16(cublas): ['-7.765625 ', '-9.59375  '], time:4.999256ms, swizzle: NOOP, TFLOPS: 109.97
                                  f16_th: ['-7.9101562', '-9.703125 '], time:4.903054ms, swizzle: NOOP, TFLOPS: 112.13(+1.77%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=8192
                     f16x8pack(t8x8+bcf): ['-34.6875  ', '109.75    '], time:22.49085ms, swizzle: NOOP, TFLOPS: 48.89 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-34.6875  ', '109.75    '], time:22.22628ms, swizzle: NOOP, TFLOPS: 49.47 (+1.19%)
                f16x8pack(t8x8+k16+dbuf): ['-34.6875  ', '109.75    '], time:21.41447ms, swizzle: NOOP, TFLOPS: 51.34 (+3.79%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-34.6875  ', '108.625   '], time:23.33004ms, swizzle: NOOP, TFLOPS: 47.13
                 f16wmma(mma4x2+warp2x4): ['-34.6875  ', '108.625   '], time:13.85409ms, swizzle: NOOP, TFLOPS: 79.36 (+54.57%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-34.6875  ', '108.625   '], time:10.07294ms, swizzle: NOOP, TFLOPS: 109.15(+37.54%)
          f16wmma(mma2x4+warp2x4+stage4): ['-34.6875  ', '108.625   '], time:10.17265ms, swizzle: NOOP, TFLOPS: 108.09
          f16wmma(mma2x4+warp2x4+stage3): ['-34.6875  ', '108.625   '], time:10.06150ms, swizzle: NOOP, TFLOPS: 109.28(+0.11%)
          f16wmma(mma2x4+warp2x4+stage2): ['-34.6875  ', '108.625   '], time:10.08632ms, swizzle: NOOP, TFLOPS: 109.01
        f16wmma(mma2x4+...+stage4+dsmem): ['-34.6875  ', '108.625   '], time:10.11593ms, swizzle: NOOP, TFLOPS: 108.69
        f16wmma(mma2x4+...+stage3+dsmem): ['-34.6875  ', '108.625   '], time:10.08446ms, swizzle: NOOP, TFLOPS: 109.03
        f16wmma(mma2x4+...+stage2+dsmem): ['-34.6875  ', '108.625   '], time:9.997367ms, swizzle: NOOP, TFLOPS: 109.98(+0.64%)
      f16wmma(mma2x4+...+stage4+swizzle): ['-34.6875  ', '108.625   '], time:10.45885ms, swizzle: 1024, TFLOPS: 105.13
      f16wmma(mma2x4+...+stage3+swizzle): ['-34.6875  ', '108.625   '], time:10.32109ms, swizzle: 1024, TFLOPS: 106.53
      f16wmma(mma2x4+...+stage2+swizzle): ['-34.6875  ', '108.625   '], time:10.17162ms, swizzle: 1024, TFLOPS: 108.10
       f16wmma(...+stage4+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:10.50686ms, swizzle: 1024, TFLOPS: 104.65
       f16wmma(...+stage3+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:10.45265ms, swizzle: 1024, TFLOPS: 105.19
       f16wmma(...+stage2+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:10.19203ms, swizzle: 1024, TFLOPS: 107.88
                             f16(cublas): ['-34.6875  ', '108.625   '], time:9.719204ms, swizzle: NOOP, TFLOPS: 113.13(+2.86%)
                                  f16_th: ['-34.90625 ', '108.5625  '], time:9.800958ms, swizzle: NOOP, TFLOPS: 112.18
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=2048
                     f16x8pack(t8x8+bcf): ['-48.625   ', '-19.59375 '], time:11.08081ms, swizzle: NOOP, TFLOPS: 49.61 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-48.625   ', '-19.59375 '], time:11.19740ms, swizzle: NOOP, TFLOPS: 49.10
                f16x8pack(t8x8+k16+dbuf): ['-48.625   ', '-19.59375 '], time:10.52210ms, swizzle: NOOP, TFLOPS: 52.25 (+5.31%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-48.6875  ', '-19.71875 '], time:11.76390ms, swizzle: NOOP, TFLOPS: 46.73
                 f16wmma(mma4x2+warp2x4): ['-48.6875  ', '-19.71875 '], time:6.968641ms, swizzle: NOOP, TFLOPS: 78.89 (+50.99%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-48.6875  ', '-19.71875 '], time:5.157113ms, swizzle: NOOP, TFLOPS: 106.60(+35.13%)
          f16wmma(mma2x4+warp2x4+stage4): ['-48.6875  ', '-19.71875 '], time:5.204582ms, swizzle: NOOP, TFLOPS: 105.63
          f16wmma(mma2x4+warp2x4+stage3): ['-48.6875  ', '-19.71875 '], time:5.057024ms, swizzle: NOOP, TFLOPS: 108.71(+1.98%)
          f16wmma(mma2x4+warp2x4+stage2): ['-48.6875  ', '-19.71875 '], time:5.080699ms, swizzle: NOOP, TFLOPS: 108.20
        f16wmma(mma2x4+...+stage4+dsmem): ['-48.6875  ', '-19.71875 '], time:5.175352ms, swizzle: NOOP, TFLOPS: 106.23
        f16wmma(mma2x4+...+stage3+dsmem): ['-48.6875  ', '-19.71875 '], time:5.136537ms, swizzle: NOOP, TFLOPS: 107.03
        f16wmma(mma2x4+...+stage2+dsmem): ['-48.6875  ', '-19.71875 '], time:5.098414ms, swizzle: NOOP, TFLOPS: 107.83
      f16wmma(mma2x4+...+stage4+swizzle): ['-48.6875  ', '-19.71875 '], time:5.205440ms, swizzle: 2048, TFLOPS: 105.61
      f16wmma(mma2x4+...+stage3+swizzle): ['-48.6875  ', '-19.71875 '], time:5.152535ms, swizzle: 2048, TFLOPS: 106.70
      f16wmma(mma2x4+...+stage2+swizzle): ['-48.6875  ', '-19.71875 '], time:5.108690ms, swizzle: 2048, TFLOPS: 107.61
       f16wmma(...+stage4+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:5.251216ms, swizzle: 2048, TFLOPS: 104.69
       f16wmma(...+stage3+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:5.198860ms, swizzle: 2048, TFLOPS: 105.75
       f16wmma(...+stage2+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:5.095887ms, swizzle: 2048, TFLOPS: 107.88
                             f16(cublas): ['-48.6875  ', '-19.71875 '], time:4.987215ms, swizzle: NOOP, TFLOPS: 110.23(+1.40%)
                                  f16_th: ['-48.75    ', '-19.765625'], time:4.788064ms, swizzle: NOOP, TFLOPS: 114.82(+4.16%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=4096
                     f16x8pack(t8x8+bcf): ['-7.390625 ', '-9.75     '], time:22.30019ms, swizzle: NOOP, TFLOPS: 49.31 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-7.390625 ', '-9.75     '], time:22.29042ms, swizzle: NOOP, TFLOPS: 49.33 (+0.04%)
                f16x8pack(t8x8+k16+dbuf): ['-7.390625 ', '-9.75     '], time:21.34160ms, swizzle: NOOP, TFLOPS: 51.52 (+4.45%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-7.765625 ', '-9.59375  '], time:23.37560ms, swizzle: NOOP, TFLOPS: 47.04
                 f16wmma(mma4x2+warp2x4): ['-7.765625 ', '-9.59375  '], time:13.63310ms, swizzle: NOOP, TFLOPS: 80.65 (+56.54%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-7.765625 ', '-9.59375  '], time:10.17763ms, swizzle: NOOP, TFLOPS: 108.03(+33.95%)
          f16wmma(mma2x4+warp2x4+stage4): ['-7.765625 ', '-9.59375  '], time:10.21945ms, swizzle: NOOP, TFLOPS: 107.59
          f16wmma(mma2x4+warp2x4+stage3): ['-7.765625 ', '-9.59375  '], time:10.08067ms, swizzle: NOOP, TFLOPS: 109.07(+0.96%)
          f16wmma(mma2x4+warp2x4+stage2): ['-7.765625 ', '-9.59375  '], time:10.05825ms, swizzle: NOOP, TFLOPS: 109.31(+0.22%)
        f16wmma(mma2x4+...+stage4+dsmem): ['-7.765625 ', '-9.59375  '], time:10.17577ms, swizzle: NOOP, TFLOPS: 108.05
        f16wmma(mma2x4+...+stage3+dsmem): ['-7.765625 ', '-9.59375  '], time:10.16266ms, swizzle: NOOP, TFLOPS: 108.19
        f16wmma(mma2x4+...+stage2+dsmem): ['-7.765625 ', '-9.59375  '], time:10.10549ms, swizzle: NOOP, TFLOPS: 108.80
      f16wmma(mma2x4+...+stage4+swizzle): ['-7.765625 ', '-9.59375  '], time:10.47282ms, swizzle: 2048, TFLOPS: 104.99
      f16wmma(mma2x4+...+stage3+swizzle): ['-7.765625 ', '-9.59375  '], time:10.33661ms, swizzle: 2048, TFLOPS: 106.37
      f16wmma(mma2x4+...+stage2+swizzle): ['-7.765625 ', '-9.59375  '], time:10.15996ms, swizzle: 2048, TFLOPS: 108.22
       f16wmma(...+stage4+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:10.34319ms, swizzle: 2048, TFLOPS: 106.30
       f16wmma(...+stage3+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:10.33718ms, swizzle: 2048, TFLOPS: 106.36
       f16wmma(...+stage2+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:10.03918ms, swizzle: 2048, TFLOPS: 109.52(+0.19%)
                             f16(cublas): ['-7.765625 ', '-9.59375  '], time:9.697818ms, swizzle: NOOP, TFLOPS: 113.38(+3.52%)
                                  f16_th: ['-7.9101562', '-9.703125 '], time:9.570813ms, swizzle: NOOP, TFLOPS: 114.88(+1.33%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=8192
                     f16x8pack(t8x8+bcf): ['-34.6875  ', '109.75    '], time:48.86844ms, swizzle: NOOP, TFLOPS: 45.00 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-34.6875  ', '109.75    '], time:49.11093ms, swizzle: NOOP, TFLOPS: 44.78
                f16x8pack(t8x8+k16+dbuf): ['-34.6875  ', '109.75    '], time:47.70295ms, swizzle: NOOP, TFLOPS: 46.10 (+2.44%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-34.6875  ', '108.625   '], time:50.71811ms, swizzle: NOOP, TFLOPS: 43.36
                 f16wmma(mma4x2+warp2x4): ['-34.6875  ', '108.625   '], time:28.04534ms, swizzle: NOOP, TFLOPS: 78.41 (+70.09%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-34.6875  ', '108.625   '], time:22.36926ms, swizzle: NOOP, TFLOPS: 98.31 (+25.37%)
          f16wmma(mma2x4+warp2x4+stage4): ['-34.6875  ', '108.625   '], time:22.50823ms, swizzle: NOOP, TFLOPS: 97.70
          f16wmma(mma2x4+warp2x4+stage3): ['-34.6875  ', '108.625   '], time:22.26257ms, swizzle: NOOP, TFLOPS: 98.78 (+0.48%)
          f16wmma(mma2x4+warp2x4+stage2): ['-34.6875  ', '108.625   '], time:22.15859ms, swizzle: NOOP, TFLOPS: 99.24 (+0.47%)
        f16wmma(mma2x4+...+stage4+dsmem): ['-34.6875  ', '108.625   '], time:22.39511ms, swizzle: NOOP, TFLOPS: 98.19
        f16wmma(mma2x4+...+stage3+dsmem): ['-34.6875  ', '108.625   '], time:22.29902ms, swizzle: NOOP, TFLOPS: 98.62
        f16wmma(mma2x4+...+stage2+dsmem): ['-34.6875  ', '108.625   '], time:22.10431ms, swizzle: NOOP, TFLOPS: 99.48 (+0.25%)
      f16wmma(mma2x4+...+stage4+swizzle): ['-34.6875  ', '108.625   '], time:20.68877ms, swizzle: 2048, TFLOPS: 106.29(+6.84%)
      f16wmma(mma2x4+...+stage3+swizzle): ['-34.6875  ', '108.625   '], time:20.35188ms, swizzle: 2048, TFLOPS: 108.05(+1.66%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-34.6875  ', '108.625   '], time:20.17860ms, swizzle: 2048, TFLOPS: 108.98(+0.86%)
       f16wmma(...+stage4+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:20.66051ms, swizzle: 2048, TFLOPS: 106.44
       f16wmma(...+stage3+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:20.54803ms, swizzle: 2048, TFLOPS: 107.02
       f16wmma(...+stage2+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:20.05717ms, swizzle: 2048, TFLOPS: 109.64(+0.61%)
                             f16(cublas): ['-34.6875  ', '108.625   '], time:19.39449ms, swizzle: NOOP, TFLOPS: 113.38(+3.42%)
                                  f16_th: ['-34.9375  ', '108.625   '], time:19.38049ms, swizzle: NOOP, TFLOPS: 113.47(+0.07%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=2048
                     f16x8pack(t8x8+bcf): ['-48.625   ', '-19.59375 '], time:22.51811ms, swizzle: NOOP, TFLOPS: 48.83 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-48.625   ', '-19.59375 '], time:22.31686ms, swizzle: NOOP, TFLOPS: 49.27 (+0.90%)
                f16x8pack(t8x8+k16+dbuf): ['-48.625   ', '-19.59375 '], time:21.36774ms, swizzle: NOOP, TFLOPS: 51.46 (+4.44%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-48.6875  ', '-19.71875 '], time:23.49147ms, swizzle: NOOP, TFLOPS: 46.80
                 f16wmma(mma4x2+warp2x4): ['-48.6875  ', '-19.71875 '], time:13.65840ms, swizzle: NOOP, TFLOPS: 80.50 (+56.44%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-48.6875  ', '-19.71875 '], time:10.41624ms, swizzle: NOOP, TFLOPS: 105.56(+31.13%)
          f16wmma(mma2x4+warp2x4+stage4): ['-48.6875  ', '-19.71875 '], time:10.48259ms, swizzle: NOOP, TFLOPS: 104.89
          f16wmma(mma2x4+warp2x4+stage3): ['-48.6875  ', '-19.71875 '], time:10.30383ms, swizzle: NOOP, TFLOPS: 106.71(+1.09%)
          f16wmma(mma2x4+warp2x4+stage2): ['-48.6875  ', '-19.71875 '], time:10.32066ms, swizzle: NOOP, TFLOPS: 106.53
        f16wmma(mma2x4+...+stage4+dsmem): ['-48.6875  ', '-19.71875 '], time:10.49389ms, swizzle: NOOP, TFLOPS: 104.78
        f16wmma(mma2x4+...+stage3+dsmem): ['-48.6875  ', '-19.71875 '], time:10.38987ms, swizzle: NOOP, TFLOPS: 105.83
        f16wmma(mma2x4+...+stage2+dsmem): ['-48.6875  ', '-19.71875 '], time:10.30273ms, swizzle: NOOP, TFLOPS: 106.72(+0.01%)
      f16wmma(mma2x4+...+stage4+swizzle): ['-48.6875  ', '-19.71875 '], time:10.61742ms, swizzle: 4096, TFLOPS: 103.56
      f16wmma(mma2x4+...+stage3+swizzle): ['-48.6875  ', '-19.71875 '], time:10.35964ms, swizzle: 4096, TFLOPS: 106.13
      f16wmma(mma2x4+...+stage2+swizzle): ['-48.6875  ', '-19.71875 '], time:10.20922ms, swizzle: 4096, TFLOPS: 107.70(+0.92%)
       f16wmma(...+stage4+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:10.56325ms, swizzle: 4096, TFLOPS: 104.09
       f16wmma(...+stage3+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:10.32545ms, swizzle: 4096, TFLOPS: 106.49
       f16wmma(...+stage2+dsmem+swizzle): ['-48.6875  ', '-19.71875 '], time:10.25662ms, swizzle: 4096, TFLOPS: 107.20
                             f16(cublas): ['-48.6875  ', '-19.71875 '], time:9.752964ms, swizzle: NOOP, TFLOPS: 112.74(+4.68%)
                                  f16_th: ['-48.75    ', '-19.765625'], time:9.698748ms, swizzle: NOOP, TFLOPS: 113.37(+0.56%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=4096
                     f16x8pack(t8x8+bcf): ['-7.390625 ', '-9.75     '], time:48.67928ms, swizzle: NOOP, TFLOPS: 45.17 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-7.390625 ', '-9.75     '], time:49.57225ms, swizzle: NOOP, TFLOPS: 44.36
                f16x8pack(t8x8+k16+dbuf): ['-7.390625 ', '-9.75     '], time:48.31402ms, swizzle: NOOP, TFLOPS: 45.52 (+0.76%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-7.765625 ', '-9.59375  '], time:63.08031ms, swizzle: NOOP, TFLOPS: 34.86
                 f16wmma(mma4x2+warp2x4): ['-7.765625 ', '-9.59375  '], time:28.37953ms, swizzle: NOOP, TFLOPS: 77.49 (+70.24%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-7.765625 ', '-9.59375  '], time:22.57206ms, swizzle: NOOP, TFLOPS: 97.42 (+25.73%)
          f16wmma(mma2x4+warp2x4+stage4): ['-7.765625 ', '-9.59375  '], time:23.40888ms, swizzle: NOOP, TFLOPS: 93.94
          f16wmma(mma2x4+warp2x4+stage3): ['-7.765625 ', '-9.59375  '], time:23.52602ms, swizzle: NOOP, TFLOPS: 93.47
          f16wmma(mma2x4+warp2x4+stage2): ['-7.765625 ', '-9.59375  '], time:22.71277ms, swizzle: NOOP, TFLOPS: 96.82
        f16wmma(mma2x4+...+stage4+dsmem): ['-7.765625 ', '-9.59375  '], time:23.63452ms, swizzle: NOOP, TFLOPS: 93.04
        f16wmma(mma2x4+...+stage3+dsmem): ['-7.765625 ', '-9.59375  '], time:23.45573ms, swizzle: NOOP, TFLOPS: 93.75
        f16wmma(mma2x4+...+stage2+dsmem): ['-7.765625 ', '-9.59375  '], time:22.61314ms, swizzle: NOOP, TFLOPS: 97.25
      f16wmma(mma2x4+...+stage4+swizzle): ['-7.765625 ', '-9.59375  '], time:20.72081ms, swizzle: 4096, TFLOPS: 106.13(+8.93%)
      f16wmma(mma2x4+...+stage3+swizzle): ['-7.765625 ', '-9.59375  '], time:20.47779ms, swizzle: 4096, TFLOPS: 107.39(+1.19%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-7.765625 ', '-9.59375  '], time:20.16084ms, swizzle: 4096, TFLOPS: 109.07(+1.57%)
       f16wmma(...+stage4+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:20.65706ms, swizzle: 4096, TFLOPS: 106.45
       f16wmma(...+stage3+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:20.57960ms, swizzle: 4096, TFLOPS: 106.85
       f16wmma(...+stage2+dsmem+swizzle): ['-7.765625 ', '-9.59375  '], time:20.07975ms, swizzle: 4096, TFLOPS: 109.51(+0.40%)
                             f16(cublas): ['-7.765625 ', '-9.59375  '], time:19.41347ms, swizzle: NOOP, TFLOPS: 113.27(+3.43%)
                                  f16_th: ['-7.9101562', '-9.703125 '], time:19.11869ms, swizzle: NOOP, TFLOPS: 115.02(+1.54%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=8192
                     f16x8pack(t8x8+bcf): ['-34.6875  ', '109.75    '], time:105.4982ms, swizzle: NOOP, TFLOPS: 41.69 (+0.00%)
                f16x8pack(t8x8+bcf+dbuf): ['-34.6875  ', '109.75    '], time:105.9915ms, swizzle: NOOP, TFLOPS: 41.49
                f16x8pack(t8x8+k16+dbuf): ['-34.6875  ', '109.75    '], time:104.0459ms, swizzle: NOOP, TFLOPS: 42.27 (+1.40%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                         f16wmma(mma4x2): ['-34.6875  ', '108.625   '], time:124.6572ms, swizzle: NOOP, TFLOPS: 35.28
                 f16wmma(mma4x2+warp2x4): ['-34.6875  ', '108.625   '], time:55.98595ms, swizzle: NOOP, TFLOPS: 78.56 (+85.84%)
            f16wmma(mma2x4+warp2x4+dbuf): ['-34.6875  ', '108.625   '], time:50.58474ms, swizzle: NOOP, TFLOPS: 86.94 (+10.68%)
          f16wmma(mma2x4+warp2x4+stage4): ['-34.6875  ', '108.625   '], time:49.78964ms, swizzle: NOOP, TFLOPS: 88.33 (+1.60%)
          f16wmma(mma2x4+warp2x4+stage3): ['-34.6875  ', '108.625   '], time:49.91416ms, swizzle: NOOP, TFLOPS: 88.11
          f16wmma(mma2x4+warp2x4+stage2): ['-34.6875  ', '108.625   '], time:50.05099ms, swizzle: NOOP, TFLOPS: 87.87
        f16wmma(mma2x4+...+stage4+dsmem): ['-34.6875  ', '108.625   '], time:50.12478ms, swizzle: NOOP, TFLOPS: 87.74
        f16wmma(mma2x4+...+stage3+dsmem): ['-34.6875  ', '108.625   '], time:49.89173ms, swizzle: NOOP, TFLOPS: 88.15
        f16wmma(mma2x4+...+stage2+dsmem): ['-34.6875  ', '108.625   '], time:50.12190ms, swizzle: NOOP, TFLOPS: 87.75
      f16wmma(mma2x4+...+stage4+swizzle): ['-34.6875  ', '108.625   '], time:41.31774ms, swizzle: 4096, TFLOPS: 106.44(+20.50%)
      f16wmma(mma2x4+...+stage3+swizzle): ['-34.6875  ', '108.625   '], time:40.67180ms, swizzle: 4096, TFLOPS: 108.14(+1.59%)
      f16wmma(mma2x4+...+stage2+swizzle): ['-34.6875  ', '108.625   '], time:40.11447ms, swizzle: 4096, TFLOPS: 109.64(+1.39%)
       f16wmma(...+stage4+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:41.21651ms, swizzle: 4096, TFLOPS: 106.71
       f16wmma(...+stage3+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:41.71590ms, swizzle: 4096, TFLOPS: 105.43
       f16wmma(...+stage2+dsmem+swizzle): ['-34.6875  ', '108.625   '], time:40.29407ms, swizzle: 4096, TFLOPS: 109.15
                             f16(cublas): ['-34.6875  ', '108.625   '], time:38.75672ms, swizzle: NOOP, TFLOPS: 113.48(+3.50%)
                                  f16_th: ['-34.9375  ', '108.625   '], time:38.51907ms, swizzle: NOOP, TFLOPS: 114.18(+0.62%)
----------------------------------------------------------------------------------------------------------------------------------
```
