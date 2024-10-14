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
- [X] hgemm_wmma_m16n16k16_naive(WMMA API, Tensor Cores) 
- [X] hgemm_wmma_m16n16k16_mma4x2(Tensor Cores, Tile MMA) 
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4(Tensor Cores, Tile MMA/Warp, pack) 
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_async(Tensor Cores, Tile MMA/Warp, Copy Async) 
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_async_offset(Tensor Cores, Tile MMA/Warp, Copy Async, Pad)
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(Tensor Cores, Tile MMA/Warp, Copy Async, Double Buffers, Pad)  
- [X] hgemm_wmma_m16n16k16_mma4x4_warp2x2x2_dbuf_async(Tensor Cores, Tile MMA/Warp, Copy Async, Double Buffers, Pad)  
- [X] hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async(Tensor Cores, Tile MMA/Warp, Copy Async, Double Buffers, Pad)
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_rbuf_async(Tensor Cores, Tile MMA/Warp, Copy Async, Double/Reg Buffers, Pad)
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_stage2/3/4(Tensor Cores, Tile MMA/Warp, Copy Async, Stage, Pad) 
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

- RTX 3080

```bash  
------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=1024, K=512
                                           out_f16: ['13.25       ', '24.9375     ', '-35.34375   '], time:1.671815ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['13.2421875  ', '25.0        ', '-35.34375   '], time:0.116563ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['13.2421875  ', '25.0        ', '-35.34375   '], time:0.111651ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['13.2421875  ', '25.0        ', '-35.34375   '], time:0.107026ms
                                        out_f16_th: ['13.25       ', '25.0        ', '-35.34375   '], time:0.095415ms
                                   out_f16(cublas): ['13.234375   ', '25.0        ', '-35.375     '], time:0.407219ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=1024, K=1024
                                           out_f16: ['-23.796875  ', '-11.3046875 ', '-3.28710938 '], time:3.394365ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['-23.6875    ', '-11.2734375 ', '-3.32226562 '], time:0.107622ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['-23.6875    ', '-11.2734375 ', '-3.32226562 '], time:0.111222ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['-23.6875    ', '-11.2734375 ', '-3.32226562 '], time:0.133324ms
                                        out_f16_th: ['-23.703125  ', '-11.2578125 ', '-3.30859375 '], time:0.188041ms
                                   out_f16(cublas): ['-23.703125  ', '-11.2734375 ', '-3.31640625 '], time:0.387073ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=2048, K=256
                                           out_f16: ['0.73632812  ', '8.8359375   ', '9.953125    '], time:1.643395ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['0.74414062  ', '8.84375     ', '9.9765625   '], time:0.130510ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['0.74414062  ', '8.84375     ', '9.9765625   '], time:0.106311ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['0.74414062  ', '8.84375     ', '9.9765625   '], time:0.122428ms
                                        out_f16_th: ['0.74902344  ', '8.8359375   ', '9.9609375   '], time:0.174332ms
                                   out_f16(cublas): ['0.74414062  ', '8.84375     ', '9.9765625   '], time:0.437212ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=2048, K=512
                                           out_f16: ['-5.70703125 ', '0.25488281  ', '32.71875    '], time:4.639077ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['-5.68359375 ', '0.31274414  ', '32.96875    '], time:0.188875ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['-5.68359375 ', '0.31274414  ', '32.96875    '], time:0.191307ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['-5.68359375 ', '0.31274414  ', '32.96875    '], time:0.220180ms
                                        out_f16_th: ['-5.70703125 ', '0.30688477  ', '32.9375     '], time:0.237679ms
                                   out_f16(cublas): ['-5.68359375 ', '0.31274414  ', '32.96875    '], time:0.469875ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=2048, K=1024
                                           out_f16: ['-43.4375    ', '1.90332031  ', '-47.5       '], time:11.322880ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['-43.375     ', '1.94726562  ', '-47.71875   '], time:0.226259ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['-43.375     ', '1.94726562  ', '-47.71875   '], time:0.252080ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['-43.375     ', '1.94726562  ', '-47.71875   '], time:0.251913ms
                                        out_f16_th: ['-43.5       ', '1.95410156  ', '-47.625     '], time:0.263572ms
                                   out_f16(cublas): ['-43.5       ', '1.9765625   ', '-47.6875    '], time:0.430346ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=4096, K=256
                                           out_f16: ['18.328125   ', '-7.1015625  ', '-24.953125  '], time:3.478360ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['18.359375   ', '-7.08203125 ', '-25.0       '], time:0.159526ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['18.359375   ', '-7.08203125 ', '-25.0       '], time:0.179887ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['18.359375   ', '-7.08203125 ', '-25.0       '], time:0.193286ms
                                        out_f16_th: ['18.375      ', '-7.0859375  ', '-25.0       '], time:0.205994ms
                                   out_f16(cublas): ['18.359375   ', '-7.08203125 ', '-25.0       '], time:0.430655ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=4096, K=512
                                           out_f16: ['24.734375   ', '23.71875    ', '30.859375   '], time:7.823086ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['24.71875    ', '23.6875     ', '30.953125   '], time:0.261283ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['24.71875    ', '23.6875     ', '30.953125   '], time:0.212145ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['24.71875    ', '23.6875     ', '30.953125   '], time:0.197577ms
                                        out_f16_th: ['24.71875    ', '23.703125   ', '30.84375    '], time:0.287414ms
                                   out_f16(cublas): ['24.71875    ', '23.6875     ', '30.953125   '], time:0.432539ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=4096, K=1024
                                           out_f16: ['-14.1328125 ', '3.12109375  ', '50.34375    '], time:17.128420ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['-14.046875  ', '3.00976562  ', '50.5625     '], time:0.349951ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['-14.046875  ', '3.00976562  ', '50.5625     '], time:0.401115ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['-14.046875  ', '3.00976562  ', '50.5625     '], time:0.331616ms
                                        out_f16_th: ['-14.078125  ', '3.00976562  ', '50.59375    '], time:0.438166ms
                                   out_f16(cublas): ['-14.046875  ', '2.99609375  ', '50.5625     '], time:0.429440ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=1024, K=256
                                           out_f16: ['8.625       ', '0.84082031  ', '1.43652344  '], time:1.679683ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['8.65625     ', '0.82226562  ', '1.47070312  '], time:0.130677ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['8.65625     ', '0.82226562  ', '1.47070312  '], time:0.111318ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['8.65625     ', '0.82226562  ', '1.47070312  '], time:0.120139ms
                                        out_f16_th: ['8.6484375   ', '0.81787109  ', '1.46289062  '], time:0.127411ms
                                   out_f16(cublas): ['8.65625     ', '0.82226562  ', '1.47070312  '], time:0.390100ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=1024, K=512
                                           out_f16: ['-24.375     ', '10.1640625  ', '12.8359375  '], time:3.299665ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['-24.640625  ', '10.1484375  ', '12.7578125  '], time:0.117540ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['-24.640625  ', '10.1484375  ', '12.7578125  '], time:0.173593ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['-24.640625  ', '10.1484375  ', '12.7578125  '], time:0.145531ms
                                        out_f16_th: ['-24.546875  ', '10.15625    ', '12.7578125  '], time:0.209188ms
                                   out_f16(cublas): ['-24.640625  ', '10.1484375  ', '12.7578125  '], time:0.483894ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=1024, K=1024
                                           out_f16: ['56.71875    ', '17.703125   ', '55.8125     '], time:9.488654ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['56.71875    ', '17.6875     ', '55.6875     '], time:0.212836ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['56.71875    ', '17.6875     ', '55.6875     '], time:0.215673ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['56.71875    ', '17.6875     ', '55.6875     '], time:0.187373ms
                                        out_f16_th: ['56.6875     ', '17.734375   ', '55.6875     '], time:0.288391ms
                                   out_f16(cublas): ['56.71875    ', '17.6875     ', '55.6875     '], time:0.469208ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=2048, K=256
                                           out_f16: ['-5.38671875 ', '-23.28125   ', '19.40625    '], time:3.370953ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['-5.37109375 ', '-23.328125  ', '19.40625    '], time:0.186849ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['-5.37109375 ', '-23.328125  ', '19.40625    '], time:0.185466ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['-5.37109375 ', '-23.328125  ', '19.40625    '], time:0.126505ms
                                        out_f16_th: ['-5.390625   ', '-23.328125  ', '19.40625    '], time:0.190234ms
                                   out_f16(cublas): ['-5.37109375 ', '-23.328125  ', '19.40625    '], time:0.443292ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=2048, K=512
                                           out_f16: ['-21.8125    ', '24.96875    ', '48.8125     '], time:6.924272ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['-21.828125  ', '24.796875   ', '48.71875    '], time:0.202227ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['-21.828125  ', '24.796875   ', '48.71875    '], time:0.218844ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['-21.828125  ', '24.796875   ', '48.71875    '], time:0.253916ms
                                        out_f16_th: ['-21.8125    ', '24.84375    ', '48.8125     '], time:0.268745ms
                                   out_f16(cublas): ['-21.828125  ', '24.796875   ', '48.71875    '], time:0.432682ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=2048, K=1024
                                           out_f16: ['-26.5625    ', '17.234375   ', '-47.65625   '], time:14.918900ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['-26.59375   ', '17.203125   ', '-47.75      '], time:0.347900ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['-26.59375   ', '17.203125   ', '-47.75      '], time:0.357652ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['-26.59375   ', '17.203125   ', '-47.75      '], time:0.333190ms
                                        out_f16_th: ['-26.625     ', '17.21875    ', '-47.71875   '], time:0.530505ms
                                   out_f16(cublas): ['-26.65625   ', '17.171875   ', '-47.8125    '], time:0.461888ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=256
                                           out_f16: ['37.3125     ', '6.1015625   ', '-11.6484375 '], time:6.849790ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['37.25       ', '6.078125    ', '-11.640625  '], time:0.268483ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['37.25       ', '6.078125    ', '-11.640625  '], time:0.245142ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['37.25       ', '6.078125    ', '-11.640625  '], time:0.239062ms
                                        out_f16_th: ['37.25       ', '6.0703125   ', '-11.640625  '], time:0.302052ms
                                   out_f16(cublas): ['37.25       ', '6.078125    ', '-11.640625  '], time:0.394297ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=512
                                           out_f16: ['2.56640625  ', '-3.1953125  ', '56.59375    '], time:16.299248ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['2.5625      ', '-3.16601562 ', '56.625      '], time:0.394201ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['2.5625      ', '-3.16601562 ', '56.625      '], time:0.384831ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['2.5625      ', '-3.16601562 ', '56.625      '], time:0.367522ms
                                        out_f16_th: ['2.56054688  ', '-3.171875   ', '56.53125    '], time:0.519037ms
                                   out_f16(cublas): ['2.5625      ', '-3.16601562 ', '56.625      '], time:0.403666ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=1024
                                           out_f16: ['39.0625     ', '2.04101562  ', '-8.3046875  '], time:34.225392ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['39.21875    ', '2.06640625  ', '-8.3203125  '], time:0.716949ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['39.21875    ', '2.06640625  ', '-8.3203125  '], time:0.722528ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['39.21875    ', '2.06640625  ', '-8.3203125  '], time:0.645280ms
                                        out_f16_th: ['39.21875    ', '2.08398438  ', '-8.328125   '], time:0.852585ms
                                   out_f16(cublas): ['39.21875    ', '2.06640625  ', '-8.3203125  '], time:0.554204ms
------------------------------------------------------------------------------------------------------------------------
```

- NVIDIA L20  
```bash
------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=2048, K=512
                                           out_f16: ['38.28125    ', '-25.0625    ', '35.96875    '], time:1.199555ms
                                       out_f16(sk): ['38.28125    ', '-25.0625    ', '35.96875    '], time:0.896072ms
                            out_f16x4pack(t4x4bcf): ['38.28125    ', '-25.0625    ', '35.96875    '], time:0.140357ms
                         out_f16x4pack(t4x4offset): ['38.28125    ', '-25.0625    ', '35.96875    '], time:0.137520ms
                                 out_f16x4(t8x8sk): ['38.28125    ', '-25.0625    ', '35.96875    '], time:0.123143ms
                                out_f16x4(t8x8bcf): ['38.28125    ', '-25.0625    ', '35.96875    '], time:0.112605ms
                             out_f16x4pack(t8x8sk): ['38.28125    ', '-25.0625    ', '35.96875    '], time:0.112176ms
                                out_f16x4pack(bcf): ['38.28125    ', '-25.0625    ', '35.96875    '], time:0.104260ms
                         out_f16x4pack(bcf+offset): ['38.28125    ', '-25.0625    ', '35.96875    '], time:0.104189ms
                                out_f16x8pack(bcf): ['38.28125    ', '-25.0625    ', '35.96875    '], time:0.101280ms
                         out_f16x8pack(bcf+offset): ['38.28125    ', '-25.0625    ', '35.96875    '], time:0.100660ms
                           out_f16x8pack(bcf+dbuf): ['38.28125    ', '-25.0625    ', '35.96875    '], time:0.095367ms
----------------------------------------------------------Async---------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['38.28125    ', '-25.0625    ', '35.96875    '], time:0.088716ms
                    out_f16x8pack(k16+dbuf+offset): ['38.28125    ', '-25.0625    ', '35.96875    '], time:0.088477ms
                     out_f16x8pack(k16+dbuf+async): ['38.28125    ', '-25.0625    ', '35.96875    '], time:0.090909ms
                           out_f16x8pack(k32+dbuf): ['38.28125    ', '-25.0625    ', '35.96875    '], time:0.091910ms
                     out_f16x8pack(k32+dbuf+async): ['38.28125    ', '-25.0625    ', '35.96875    '], time:0.089765ms
                     out_f16x8pack(k32+dbuf+t16x8): ['38.28125    ', '-25.0625    ', '35.96875    '], time:0.095749ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['38.28125    ', '-25.0625    ', '35.96875    '], time:0.091362ms
----------------------------------------------------------WMMA----------------------------------------------------------
                                out_f16wmma(naive): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.190115ms
                               out_f16wmma(mma4x2): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.100970ms
                       out_f16wmma(mma4x2+warp2x4): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.075197ms
                 out_f16wmma(mma4x2+warp2x4+async): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.065279ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.093460ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.058675ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.068760ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.085402ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.068665ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.068760ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.065923ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.060368ms
                out_f16wmma(mma2x4+warp2x4+stage2): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.066328ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.062728ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.066328ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.068784ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.056982ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.057745ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.056672ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.055695ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.055957ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.055957ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.055432ms
                                   out_f16(cublas): ['38.3125     ', '-25.15625   ', '35.9375     '], time:0.151777ms
                                        out_f16_th: ['38.3125     ', '-25.15625   ', '35.875      '], time:0.046754ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=2048, K=1024
                                           out_f16: ['-26.28125   ', '72.75       ', '-15.3125    '], time:2.384520ms
                                       out_f16(sk): ['-26.28125   ', '72.75       ', '-15.3125    '], time:1.775098ms
                            out_f16x4pack(t4x4bcf): ['-26.28125   ', '72.75       ', '-15.3125    '], time:0.277448ms
                         out_f16x4pack(t4x4offset): ['-26.28125   ', '72.75       ', '-15.3125    '], time:0.273013ms
                                 out_f16x4(t8x8sk): ['-26.28125   ', '72.75       ', '-15.3125    '], time:0.233197ms
                                out_f16x4(t8x8bcf): ['-26.28125   ', '72.75       ', '-15.3125    '], time:0.220323ms
                             out_f16x4pack(t8x8sk): ['-26.28125   ', '72.75       ', '-15.3125    '], time:0.217295ms
                                out_f16x4pack(bcf): ['-26.28125   ', '72.75       ', '-15.3125    '], time:0.203347ms
                         out_f16x4pack(bcf+offset): ['-26.28125   ', '72.75       ', '-15.3125    '], time:0.203109ms
                                out_f16x8pack(bcf): ['-26.28125   ', '72.75       ', '-15.3125    '], time:0.197411ms
                         out_f16x8pack(bcf+offset): ['-26.28125   ', '72.75       ', '-15.3125    '], time:0.196171ms
                           out_f16x8pack(bcf+dbuf): ['-26.28125   ', '72.75       ', '-15.3125    '], time:0.185394ms
----------------------------------------------------------Async---------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['-26.28125   ', '72.75       ', '-15.3125    '], time:0.172138ms
                    out_f16x8pack(k16+dbuf+offset): ['-26.28125   ', '72.75       ', '-15.3125    '], time:0.171804ms
                     out_f16x8pack(k16+dbuf+async): ['-26.28125   ', '72.75       ', '-15.3125    '], time:0.176287ms
                           out_f16x8pack(k32+dbuf): ['-26.28125   ', '72.75       ', '-15.3125    '], time:0.177956ms
                     out_f16x8pack(k32+dbuf+async): ['-26.28125   ', '72.75       ', '-15.3125    '], time:0.174069ms
                     out_f16x8pack(k32+dbuf+t16x8): ['-26.28125   ', '72.75       ', '-15.3125    '], time:0.186706ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['-26.28125   ', '72.75       ', '-15.3125    '], time:0.177073ms
----------------------------------------------------------WMMA----------------------------------------------------------
                                out_f16wmma(naive): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.376987ms
                               out_f16wmma(mma4x2): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.194907ms
                       out_f16wmma(mma4x2+warp2x4): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.157976ms
                 out_f16wmma(mma4x2+warp2x4+async): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.117326ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.176716ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.104260ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.124407ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.157404ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.123453ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.123668ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.118518ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.107741ms
                out_f16wmma(mma2x4+warp2x4+stage2): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.118423ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.112605ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.120020ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.123906ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.099802ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.101995ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.099540ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.097704ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.097632ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.097728ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.097632ms
                                   out_f16(cublas): ['-26.21875   ', '72.8125     ', '-15.5078125 '], time:0.165248ms
                                        out_f16_th: ['-26.140625  ', '72.75       ', '-15.484375  '], time:0.085592ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=4096, K=512
                                           out_f16: ['8.453125    ', '-18.75      ', '-10.5390625 '], time:2.385879ms
                                       out_f16(sk): ['8.453125    ', '-18.75      ', '-10.5390625 '], time:1.777363ms
                            out_f16x4pack(t4x4bcf): ['8.453125    ', '-18.75      ', '-10.5390625 '], time:0.260258ms
                         out_f16x4pack(t4x4offset): ['8.453125    ', '-18.75      ', '-10.5390625 '], time:0.255489ms
                                 out_f16x4(t8x8sk): ['8.453125    ', '-18.75      ', '-10.5390625 '], time:0.231218ms
                                out_f16x4(t8x8bcf): ['8.453125    ', '-18.75      ', '-10.5390625 '], time:0.212526ms
                             out_f16x4pack(t8x8sk): ['8.453125    ', '-18.75      ', '-10.5390625 '], time:0.208950ms
                                out_f16x4pack(bcf): ['8.453125    ', '-18.75      ', '-10.5390625 '], time:0.195885ms
                         out_f16x4pack(bcf+offset): ['8.453125    ', '-18.75      ', '-10.5390625 '], time:0.195003ms
                                out_f16x8pack(bcf): ['8.453125    ', '-18.75      ', '-10.5390625 '], time:0.189447ms
                         out_f16x8pack(bcf+offset): ['8.453125    ', '-18.75      ', '-10.5390625 '], time:0.187802ms
                           out_f16x8pack(bcf+dbuf): ['8.453125    ', '-18.75      ', '-10.5390625 '], time:0.182128ms
----------------------------------------------------------Async---------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['8.453125    ', '-18.75      ', '-10.5390625 '], time:0.171471ms
                    out_f16x8pack(k16+dbuf+offset): ['8.453125    ', '-18.75      ', '-10.5390625 '], time:0.171208ms
                     out_f16x8pack(k16+dbuf+async): ['8.453125    ', '-18.75      ', '-10.5390625 '], time:0.175214ms
                           out_f16x8pack(k32+dbuf): ['8.453125    ', '-18.75      ', '-10.5390625 '], time:0.175452ms
                     out_f16x8pack(k32+dbuf+async): ['8.453125    ', '-18.75      ', '-10.5390625 '], time:0.172830ms
                     out_f16x8pack(k32+dbuf+t16x8): ['8.453125    ', '-18.75      ', '-10.5390625 '], time:0.182533ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['8.453125    ', '-18.75      ', '-10.5390625 '], time:0.174642ms
----------------------------------------------------------WMMA----------------------------------------------------------
                                out_f16wmma(naive): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.374079ms
                               out_f16wmma(mma4x2): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.193405ms
                       out_f16wmma(mma4x2+warp2x4): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.136924ms
                 out_f16wmma(mma4x2+warp2x4+async): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.125122ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.145483ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.109267ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.124145ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.166845ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.128865ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.128508ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.124669ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.113535ms
                out_f16wmma(mma2x4+warp2x4+stage2): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.124979ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.119209ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.125074ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.130367ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.106835ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.110674ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.106335ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.105405ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.106549ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.105977ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.104499ms
                                   out_f16(cublas): ['8.4375      ', '-18.78125   ', '-10.5390625 '], time:0.169039ms
                                        out_f16_th: ['8.421875    ', '-18.75      ', '-10.546875  '], time:0.091600ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=4096, K=1024
                                           out_f16: ['14.7890625  ', '1.8515625   ', '-25.015625  '], time:4.745483ms
                                       out_f16(sk): ['14.7890625  ', '1.8515625   ', '-25.015625  '], time:3.525496ms
                            out_f16x4pack(t4x4bcf): ['14.7890625  ', '1.8515625   ', '-25.015625  '], time:0.516915ms
                         out_f16x4pack(t4x4offset): ['14.7890625  ', '1.8515625   ', '-25.015625  '], time:0.507474ms
                                 out_f16x4(t8x8sk): ['14.7890625  ', '1.8515625   ', '-25.015625  '], time:0.442910ms
                                out_f16x4(t8x8bcf): ['14.7890625  ', '1.8515625   ', '-25.015625  '], time:0.418544ms
                             out_f16x4pack(t8x8sk): ['14.7890625  ', '1.8515625   ', '-25.015625  '], time:0.408125ms
                                out_f16x4pack(bcf): ['14.7890625  ', '1.8515625   ', '-25.015625  '], time:0.385118ms
                         out_f16x4pack(bcf+offset): ['14.7890625  ', '1.8515625   ', '-25.015625  '], time:0.383592ms
                                out_f16x8pack(bcf): ['14.7890625  ', '1.8515625   ', '-25.015625  '], time:0.372052ms
                         out_f16x8pack(bcf+offset): ['14.7890625  ', '1.8515625   ', '-25.015625  '], time:0.368571ms
                           out_f16x8pack(bcf+dbuf): ['14.7890625  ', '1.8515625   ', '-25.015625  '], time:0.358248ms
----------------------------------------------------------Async---------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['14.7890625  ', '1.8515625   ', '-25.015625  '], time:0.336528ms
                    out_f16x8pack(k16+dbuf+offset): ['14.7890625  ', '1.8515625   ', '-25.015625  '], time:0.336361ms
                     out_f16x8pack(k16+dbuf+async): ['14.7890625  ', '1.8515625   ', '-25.015625  '], time:0.344133ms
                           out_f16x8pack(k32+dbuf): ['14.7890625  ', '1.8515625   ', '-25.015625  '], time:0.344944ms
                     out_f16x8pack(k32+dbuf+async): ['14.7890625  ', '1.8515625   ', '-25.015625  '], time:0.339460ms
                     out_f16x8pack(k32+dbuf+t16x8): ['14.7890625  ', '1.8515625   ', '-25.015625  '], time:0.359678ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['14.7890625  ', '1.8515625   ', '-25.015625  '], time:0.342989ms
----------------------------------------------------------WMMA----------------------------------------------------------
                                out_f16wmma(naive): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.739789ms
                               out_f16wmma(mma4x2): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.376129ms
                       out_f16wmma(mma4x2+warp2x4): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.247812ms
                 out_f16wmma(mma4x2+warp2x4+async): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.228405ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.273418ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.196147ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.229216ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.310278ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.236797ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.236654ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.227308ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.204539ms
                out_f16wmma(mma2x4+warp2x4+stage2): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.227642ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.217748ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.229073ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.236750ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.191212ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.199699ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.190020ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.187635ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.188828ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.187969ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.186920ms
                                   out_f16(cublas): ['14.8125     ', '1.87207031  ', '-25.09375   '], time:0.182438ms
                                        out_f16_th: ['14.84375    ', '1.88085938  ', '-25.09375   '], time:0.169349ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=2048, K=512
                                           out_f16: ['44.375      ', '6.51171875  ', '-4.6796875  '], time:2.385640ms
                                       out_f16(sk): ['44.375      ', '6.51171875  ', '-4.6796875  '], time:1.779795ms
                            out_f16x4pack(t4x4bcf): ['44.375      ', '6.51171875  ', '-4.6796875  '], time:0.260282ms
                         out_f16x4pack(t4x4offset): ['44.375      ', '6.51171875  ', '-4.6796875  '], time:0.255656ms
                                 out_f16x4(t8x8sk): ['44.375      ', '6.51171875  ', '-4.6796875  '], time:0.235510ms
                                out_f16x4(t8x8bcf): ['44.375      ', '6.51171875  ', '-4.6796875  '], time:0.212479ms
                             out_f16x4pack(t8x8sk): ['44.375      ', '6.51171875  ', '-4.6796875  '], time:0.210452ms
                                out_f16x4pack(bcf): ['44.375      ', '6.51171875  ', '-4.6796875  '], time:0.195527ms
                         out_f16x4pack(bcf+offset): ['44.375      ', '6.51171875  ', '-4.6796875  '], time:0.195026ms
                                out_f16x8pack(bcf): ['44.375      ', '6.51171875  ', '-4.6796875  '], time:0.189376ms
                         out_f16x8pack(bcf+offset): ['44.375      ', '6.51171875  ', '-4.6796875  '], time:0.188065ms
                           out_f16x8pack(bcf+dbuf): ['44.375      ', '6.51171875  ', '-4.6796875  '], time:0.182390ms
----------------------------------------------------------Async---------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['44.375      ', '6.51171875  ', '-4.6796875  '], time:0.171757ms
                    out_f16x8pack(k16+dbuf+offset): ['44.375      ', '6.51171875  ', '-4.6796875  '], time:0.171447ms
                     out_f16x8pack(k16+dbuf+async): ['44.375      ', '6.51171875  ', '-4.6796875  '], time:0.175524ms
                           out_f16x8pack(k32+dbuf): ['44.375      ', '6.51171875  ', '-4.6796875  '], time:0.176120ms
                     out_f16x8pack(k32+dbuf+async): ['44.375      ', '6.51171875  ', '-4.6796875  '], time:0.172973ms
                     out_f16x8pack(k32+dbuf+t16x8): ['44.375      ', '6.51171875  ', '-4.6796875  '], time:0.182414ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['44.375      ', '6.51171875  ', '-4.6796875  '], time:0.174809ms
----------------------------------------------------------WMMA----------------------------------------------------------
                                out_f16wmma(naive): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.371265ms
                               out_f16wmma(mma4x2): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.193739ms
                       out_f16wmma(mma4x2+warp2x4): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.136924ms
                 out_f16wmma(mma4x2+warp2x4+async): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.124979ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.145340ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.109649ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.124836ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.166821ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.129461ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.129199ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.124764ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.113034ms
                out_f16wmma(mma2x4+warp2x4+stage2): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.125217ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.118804ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.125217ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.129437ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.106931ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.110960ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.106716ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.105691ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.106812ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.105906ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.104046ms
                                   out_f16(cublas): ['44.28125    ', '6.50390625  ', '-4.6640625  '], time:0.166273ms
                                        out_f16_th: ['44.28125    ', '6.5078125   ', '-4.66796875 '], time:0.087380ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=2048, K=1024
                                           out_f16: ['5.36328125  ', '-2.14257812 ', '-16.6875    '], time:4.745483ms
                                       out_f16(sk): ['5.36328125  ', '-2.14257812 ', '-16.6875    '], time:3.529882ms
                            out_f16x4pack(t4x4bcf): ['5.36328125  ', '-2.14257812 ', '-16.6875    '], time:0.517035ms
                         out_f16x4pack(t4x4offset): ['5.36328125  ', '-2.14257812 ', '-16.6875    '], time:0.507474ms
                                 out_f16x4(t8x8sk): ['5.36328125  ', '-2.14257812 ', '-16.6875    '], time:0.446773ms
                                out_f16x4(t8x8bcf): ['5.36328125  ', '-2.14257812 ', '-16.6875    '], time:0.418544ms
                             out_f16x4pack(t8x8sk): ['5.36328125  ', '-2.14257812 ', '-16.6875    '], time:0.408745ms
                                out_f16x4pack(bcf): ['5.36328125  ', '-2.14257812 ', '-16.6875    '], time:0.384212ms
                         out_f16x4pack(bcf+offset): ['5.36328125  ', '-2.14257812 ', '-16.6875    '], time:0.383067ms
                                out_f16x8pack(bcf): ['5.36328125  ', '-2.14257812 ', '-16.6875    '], time:0.371885ms
                         out_f16x8pack(bcf+offset): ['5.36328125  ', '-2.14257812 ', '-16.6875    '], time:0.368905ms
                           out_f16x8pack(bcf+dbuf): ['5.36328125  ', '-2.14257812 ', '-16.6875    '], time:0.358176ms
----------------------------------------------------------Async---------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['5.36328125  ', '-2.14257812 ', '-16.6875    '], time:0.336695ms
                    out_f16x8pack(k16+dbuf+offset): ['5.36328125  ', '-2.14257812 ', '-16.6875    '], time:0.336432ms
                     out_f16x8pack(k16+dbuf+async): ['5.36328125  ', '-2.14257812 ', '-16.6875    '], time:0.344229ms
                           out_f16x8pack(k32+dbuf): ['5.36328125  ', '-2.14257812 ', '-16.6875    '], time:0.345898ms
                     out_f16x8pack(k32+dbuf+async): ['5.36328125  ', '-2.14257812 ', '-16.6875    '], time:0.339460ms
                     out_f16x8pack(k32+dbuf+t16x8): ['5.36328125  ', '-2.14257812 ', '-16.6875    '], time:0.359011ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['5.36328125  ', '-2.14257812 ', '-16.6875    '], time:0.342894ms
----------------------------------------------------------WMMA----------------------------------------------------------
                                out_f16wmma(naive): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.735736ms
                               out_f16wmma(mma4x2): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.376272ms
                       out_f16wmma(mma4x2+warp2x4): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.247812ms
                 out_f16wmma(mma4x2+warp2x4+async): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.228572ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.273108ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.197101ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.229859ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.310135ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.237298ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.236726ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.227308ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.204492ms
                out_f16wmma(mma2x4+warp2x4+stage2): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.227427ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.217581ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.229049ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.236988ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.191283ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.199676ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.190210ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.187492ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.188684ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.188041ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.186801ms
                                   out_f16(cublas): ['5.3203125   ', '-2.1484375  ', '-16.5       '], time:0.181031ms
                                        out_f16_th: ['5.328125    ', '-2.1484375  ', '-16.5       '], time:0.169516ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=512
                                           out_f16: ['-21.40625   ', '8.09375     ', '-11.5625    '], time:4.755259ms
                                       out_f16(sk): ['-21.40625   ', '8.09375     ', '-11.5625    '], time:3.542209ms
                            out_f16x4pack(t4x4bcf): ['-21.40625   ', '8.09375     ', '-11.5625    '], time:0.501251ms
                         out_f16x4pack(t4x4offset): ['-21.40625   ', '8.09375     ', '-11.5625    '], time:0.491834ms
                                 out_f16x4(t8x8sk): ['-21.40625   ', '8.09375     ', '-11.5625    '], time:0.441360ms
                                out_f16x4(t8x8bcf): ['-21.40625   ', '8.09375     ', '-11.5625    '], time:0.410533ms
                             out_f16x4pack(t8x8sk): ['-21.40625   ', '8.09375     ', '-11.5625    '], time:0.401521ms
                                out_f16x4pack(bcf): ['-21.40625   ', '8.09375     ', '-11.5625    '], time:0.375843ms
                         out_f16x4pack(bcf+offset): ['-21.40625   ', '8.09375     ', '-11.5625    '], time:0.374985ms
                                out_f16x8pack(bcf): ['-21.40625   ', '8.09375     ', '-11.5625    '], time:0.362182ms
                         out_f16x8pack(bcf+offset): ['-21.40625   ', '8.09375     ', '-11.5625    '], time:0.359154ms
                           out_f16x8pack(bcf+dbuf): ['-21.40625   ', '8.09375     ', '-11.5625    '], time:0.356054ms
----------------------------------------------------------Async---------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['-21.40625   ', '8.09375     ', '-11.5625    '], time:0.337267ms
                    out_f16x8pack(k16+dbuf+offset): ['-21.40625   ', '8.09375     ', '-11.5625    '], time:0.337052ms
                     out_f16x8pack(k16+dbuf+async): ['-21.40625   ', '8.09375     ', '-11.5625    '], time:0.344729ms
                           out_f16x8pack(k32+dbuf): ['-21.40625   ', '8.09375     ', '-11.5625    '], time:0.343919ms
                     out_f16x8pack(k32+dbuf+async): ['-21.40625   ', '8.09375     ', '-11.5625    '], time:0.339770ms
                     out_f16x8pack(k32+dbuf+t16x8): ['-21.40625   ', '8.09375     ', '-11.5625    '], time:0.356388ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['-21.40625   ', '8.09375     ', '-11.5625    '], time:0.341940ms
----------------------------------------------------------WMMA----------------------------------------------------------
                                out_f16wmma(naive): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.742984ms
                               out_f16wmma(mma4x2): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.380969ms
                       out_f16wmma(mma4x2+warp2x4): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.266933ms
                 out_f16wmma(mma4x2+warp2x4+async): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.237465ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.280786ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.203037ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.231814ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.321651ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.244045ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.244021ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.236011ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.213337ms
                out_f16wmma(mma2x4+warp2x4+stage2): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.237584ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.226831ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.235629ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.244808ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.200748ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.210571ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.200582ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.197792ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.199533ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.198674ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.192189ms
                                   out_f16(cublas): ['-21.40625   ', '8.09375     ', '-11.59375   '], time:0.199127ms
                                        out_f16_th: ['-21.34375   ', '8.1015625   ', '-11.5859375 '], time:0.169325ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=1024
                                           out_f16: ['34.4375     ', '47.90625    ', '51.40625    '], time:9.459186ms
                                       out_f16(sk): ['34.4375     ', '47.90625    ', '51.40625    '], time:7.025671ms
                            out_f16x4pack(t4x4bcf): ['34.4375     ', '47.90625    ', '51.40625    '], time:0.996733ms
                         out_f16x4pack(t4x4offset): ['34.4375     ', '47.90625    ', '51.40625    '], time:0.978136ms
                                 out_f16x4(t8x8sk): ['34.4375     ', '47.90625    ', '51.40625    '], time:0.855565ms
                                out_f16x4(t8x8bcf): ['34.4375     ', '47.90625    ', '51.40625    '], time:0.811291ms
                             out_f16x4pack(t8x8sk): ['34.4375     ', '47.90625    ', '51.40625    '], time:0.788450ms
                                out_f16x4pack(bcf): ['34.4375     ', '47.90625    ', '51.40625    '], time:0.742722ms
                         out_f16x4pack(bcf+offset): ['34.4375     ', '47.90625    ', '51.40625    '], time:0.740600ms
                                out_f16x8pack(bcf): ['34.4375     ', '47.90625    ', '51.40625    '], time:0.713921ms
                         out_f16x8pack(bcf+offset): ['34.4375     ', '47.90625    ', '51.40625    '], time:0.708270ms
                           out_f16x8pack(bcf+dbuf): ['34.4375     ', '47.90625    ', '51.40625    '], time:0.703859ms
----------------------------------------------------------Async---------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['34.4375     ', '47.90625    ', '51.40625    '], time:0.666332ms
                    out_f16x8pack(k16+dbuf+offset): ['34.4375     ', '47.90625    ', '51.40625    '], time:0.665641ms
                     out_f16x8pack(k16+dbuf+async): ['34.4375     ', '47.90625    ', '51.40625    '], time:0.680637ms
                           out_f16x8pack(k32+dbuf): ['34.4375     ', '47.90625    ', '51.40625    '], time:0.680113ms
                     out_f16x8pack(k32+dbuf+async): ['34.4375     ', '47.90625    ', '51.40625    '], time:0.671458ms
                     out_f16x8pack(k32+dbuf+t16x8): ['34.4375     ', '47.90625    ', '51.40625    '], time:0.704813ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['34.4375     ', '47.90625    ', '51.40625    '], time:0.676036ms
----------------------------------------------------------WMMA----------------------------------------------------------
                                out_f16wmma(naive): ['34.5625     ', '47.84375    ', '51.40625    '], time:1.470900ms
                               out_f16wmma(mma4x2): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.744796ms
                       out_f16wmma(mma4x2+warp2x4): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.488949ms
                 out_f16wmma(mma4x2+warp2x4+async): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.443792ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.532842ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.373840ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.439501ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.607538ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.467324ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.466919ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.440288ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.393772ms
                out_f16wmma(mma2x4+warp2x4+stage2): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.441861ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.421214ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.442219ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.467348ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.370359ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.389647ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.366020ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.362802ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.365353ms
         out_f16wmma(mma4x2+warp2x4+stage2+offset): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.363278ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.356960ms
                                   out_f16(cublas): ['34.5625     ', '47.84375    ', '51.40625    '], time:0.357294ms
                                        out_f16_th: ['34.5625     ', '47.875      ', '51.375      '], time:0.334740ms
------------------------------------------------------------------------------------------------------------------------
```
