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
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['-15.9140625 ', '36.125      ', '17.59375    '], time:0.096464ms
                               out_f16wmma(mma4x2): ['-15.9140625 ', '36.125      ', '17.59375    '], time:0.054026ms
                       out_f16wmma(mma4x2+warp2x4): ['-15.9140625 ', '36.125      ', '17.59375    '], time:0.044680ms
                 out_f16wmma(mma4x2+warp2x4+async): ['-15.9140625 ', '36.125      ', '17.59375    '], time:0.038791ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['-15.9140625 ', '36.125      ', '17.59375    '], time:0.050926ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['-15.9140625 ', '36.125      ', '17.59375    '], time:0.035858ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['-15.9140625 ', '36.125      ', '17.59375    '], time:0.040889ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['-15.9140625 ', '36.125      ', '17.59375    '], time:0.049448ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['-15.9140625 ', '36.125      ', '17.59375    '], time:0.043225ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['-15.9140625 ', '36.125      ', '17.59375    '], time:0.041819ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['-15.9140625 ', '36.125      ', '17.59375    '], time:0.039697ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-15.9140625 ', '36.125      ', '17.59375    '], time:0.037003ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['-15.9140625 ', '36.125      ', '17.59375    '], time:0.037551ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['-15.9140625 ', '36.125      ', '17.59375    '], time:0.039935ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['-15.9140625 ', '36.125      ', '17.59375    '], time:0.042176ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['-15.9140625 ', '36.125      ', '17.59375    '], time:0.035834ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['-15.9140625 ', '36.125      ', '17.59375    '], time:0.035429ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['-15.9140625 ', '36.125      ', '17.59375    '], time:0.035501ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['-15.9140625 ', '36.125      ', '17.59375    '], time:0.035262ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['-15.9140625 ', '36.125      ', '17.59375    '], time:0.035167ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['-15.9140625 ', '36.125      ', '17.59375    '], time:0.034857ms
                                        out_f16_th: ['-15.9140625 ', '36.1875     ', '17.59375    '], time:0.026488ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=1024, K=512
                                           out_f16: ['3.04101562  ', '-24.71875   ', '-8.140625   '], time:1.199675ms
                                       out_f16(sk): ['3.04101562  ', '-24.71875   ', '-8.140625   '], time:0.895953ms
                            out_f16x4pack(t4x4bcf): ['3.04101562  ', '-24.71875   ', '-8.140625   '], time:0.140190ms
                         out_f16x4pack(t4x4offset): ['3.04101562  ', '-24.71875   ', '-8.140625   '], time:0.137615ms
                                 out_f16x4(t8x8sk): ['3.04101562  ', '-24.71875   ', '-8.140625   '], time:0.121617ms
                                out_f16x4(t8x8bcf): ['3.04101562  ', '-24.71875   ', '-8.140625   '], time:0.112534ms
                             out_f16x4pack(t8x8sk): ['3.04101562  ', '-24.71875   ', '-8.140625   '], time:0.111461ms
                                out_f16x4pack(bcf): ['3.04101562  ', '-24.71875   ', '-8.140625   '], time:0.104403ms
                         out_f16x4pack(bcf+offset): ['3.04101562  ', '-24.71875   ', '-8.140625   '], time:0.104260ms
                                out_f16x8pack(bcf): ['3.04101562  ', '-24.71875   ', '-8.140625   '], time:0.101185ms
                         out_f16x8pack(bcf+offset): ['3.04101562  ', '-24.71875   ', '-8.140625   '], time:0.100732ms
                           out_f16x8pack(bcf+dbuf): ['3.04101562  ', '-24.71875   ', '-8.140625   '], time:0.095201ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['3.04101562  ', '-24.71875   ', '-8.140625   '], time:0.088716ms
                    out_f16x8pack(k16+dbuf+offset): ['3.04101562  ', '-24.71875   ', '-8.140625   '], time:0.088596ms
                     out_f16x8pack(k16+dbuf+async): ['3.04101562  ', '-24.71875   ', '-8.140625   '], time:0.090790ms
                           out_f16x8pack(k32+dbuf): ['3.04101562  ', '-24.71875   ', '-8.140625   '], time:0.091410ms
                     out_f16x8pack(k32+dbuf+async): ['3.04101562  ', '-24.71875   ', '-8.140625   '], time:0.089741ms
                     out_f16x8pack(k32+dbuf+t16x8): ['3.04101562  ', '-24.71875   ', '-8.140625   '], time:0.095892ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['3.04101562  ', '-24.71875   ', '-8.140625   '], time:0.091338ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['3.05078125  ', '-24.703125  ', '-8.078125   '], time:0.188494ms
                               out_f16wmma(mma4x2): ['3.05078125  ', '-24.703125  ', '-8.078125   '], time:0.100827ms
                       out_f16wmma(mma4x2+warp2x4): ['3.05078125  ', '-24.703125  ', '-8.078125   '], time:0.075126ms
                 out_f16wmma(mma4x2+warp2x4+async): ['3.05078125  ', '-24.703125  ', '-8.078125   '], time:0.065660ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['3.05078125  ', '-24.703125  ', '-8.078125   '], time:0.094128ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['3.05078125  ', '-24.703125  ', '-8.078125   '], time:0.058484ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['3.05078125  ', '-24.703125  ', '-8.078125   '], time:0.068688ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['3.05078125  ', '-24.703125  ', '-8.078125   '], time:0.085449ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['3.05078125  ', '-24.703125  ', '-8.078125   '], time:0.068688ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['3.05078125  ', '-24.703125  ', '-8.078125   '], time:0.068617ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['3.05078125  ', '-24.703125  ', '-8.078125   '], time:0.066042ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['3.05078125  ', '-24.703125  ', '-8.078125   '], time:0.060344ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['3.05078125  ', '-24.703125  ', '-8.078125   '], time:0.063562ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['3.05078125  ', '-24.703125  ', '-8.078125   '], time:0.066733ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['3.05078125  ', '-24.703125  ', '-8.078125   '], time:0.068593ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['3.05078125  ', '-24.703125  ', '-8.078125   '], time:0.057268ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['3.05078125  ', '-24.703125  ', '-8.078125   '], time:0.057721ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['3.05078125  ', '-24.703125  ', '-8.078125   '], time:0.056958ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['3.05078125  ', '-24.703125  ', '-8.078125   '], time:0.056529ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['3.05078125  ', '-24.703125  ', '-8.078125   '], time:0.055814ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['3.05078125  ', '-24.703125  ', '-8.078125   '], time:0.055909ms
                                        out_f16_th: ['3.05664062  ', '-24.75      ', '-8.078125   '], time:0.046062ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=1024, K=1024
                                           out_f16: ['-33.0625    ', '-16.4375    ', '14.3203125  '], time:2.384472ms
                                       out_f16(sk): ['-33.0625    ', '-16.4375    ', '14.3203125  '], time:1.775122ms
                            out_f16x4pack(t4x4bcf): ['-33.0625    ', '-16.4375    ', '14.3203125  '], time:0.277495ms
                         out_f16x4pack(t4x4offset): ['-33.0625    ', '-16.4375    ', '14.3203125  '], time:0.272536ms
                                 out_f16x4(t8x8sk): ['-33.0625    ', '-16.4375    ', '14.3203125  '], time:0.231552ms
                                out_f16x4(t8x8bcf): ['-33.0625    ', '-16.4375    ', '14.3203125  '], time:0.220180ms
                             out_f16x4pack(t8x8sk): ['-33.0625    ', '-16.4375    ', '14.3203125  '], time:0.216675ms
                                out_f16x4pack(bcf): ['-33.0625    ', '-16.4375    ', '14.3203125  '], time:0.203872ms
                         out_f16x4pack(bcf+offset): ['-33.0625    ', '-16.4375    ', '14.3203125  '], time:0.203300ms
                                out_f16x8pack(bcf): ['-33.0625    ', '-16.4375    ', '14.3203125  '], time:0.197339ms
                         out_f16x8pack(bcf+offset): ['-33.0625    ', '-16.4375    ', '14.3203125  '], time:0.196552ms
                           out_f16x8pack(bcf+dbuf): ['-33.0625    ', '-16.4375    ', '14.3203125  '], time:0.185442ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['-33.0625    ', '-16.4375    ', '14.3203125  '], time:0.172043ms
                    out_f16x8pack(k16+dbuf+offset): ['-33.0625    ', '-16.4375    ', '14.3203125  '], time:0.171661ms
                     out_f16x8pack(k16+dbuf+async): ['-33.0625    ', '-16.4375    ', '14.3203125  '], time:0.176215ms
                           out_f16x8pack(k32+dbuf): ['-33.0625    ', '-16.4375    ', '14.3203125  '], time:0.177336ms
                     out_f16x8pack(k32+dbuf+async): ['-33.0625    ', '-16.4375    ', '14.3203125  '], time:0.173926ms
                     out_f16x8pack(k32+dbuf+t16x8): ['-33.0625    ', '-16.4375    ', '14.3203125  '], time:0.187135ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['-33.0625    ', '-16.4375    ', '14.3203125  '], time:0.177026ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['-33.1875    ', '-16.546875  ', '14.3671875  '], time:0.372267ms
                               out_f16wmma(mma4x2): ['-33.1875    ', '-16.546875  ', '14.3671875  '], time:0.194645ms
                       out_f16wmma(mma4x2+warp2x4): ['-33.1875    ', '-16.546875  ', '14.3671875  '], time:0.135541ms
                 out_f16wmma(mma4x2+warp2x4+async): ['-33.1875    ', '-16.546875  ', '14.3671875  '], time:0.117469ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['-33.1875    ', '-16.546875  ', '14.3671875  '], time:0.177193ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['-33.1875    ', '-16.546875  ', '14.3671875  '], time:0.103998ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['-33.1875    ', '-16.546875  ', '14.3671875  '], time:0.123930ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['-33.1875    ', '-16.546875  ', '14.3671875  '], time:0.157738ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['-33.1875    ', '-16.546875  ', '14.3671875  '], time:0.123024ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['-33.1875    ', '-16.546875  ', '14.3671875  '], time:0.123167ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['-33.1875    ', '-16.546875  ', '14.3671875  '], time:0.118709ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-33.1875    ', '-16.546875  ', '14.3671875  '], time:0.107145ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['-33.1875    ', '-16.546875  ', '14.3671875  '], time:0.115371ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['-33.1875    ', '-16.546875  ', '14.3671875  '], time:0.120950ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['-33.1875    ', '-16.546875  ', '14.3671875  '], time:0.123334ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['-33.1875    ', '-16.546875  ', '14.3671875  '], time:0.099754ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['-33.1875    ', '-16.546875  ', '14.3671875  '], time:0.102019ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['-33.1875    ', '-16.546875  ', '14.3671875  '], time:0.099945ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['-33.1875    ', '-16.546875  ', '14.3671875  '], time:0.097823ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['-33.1875    ', '-16.546875  ', '14.3671875  '], time:0.097537ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['-33.1875    ', '-16.546875  ', '14.3671875  '], time:0.098538ms
                                        out_f16_th: ['-33.125     ', '-16.515625  ', '14.3359375  '], time:0.085664ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=2048, K=256
                                           out_f16: ['-10.8359375 ', '24.734375   ', '12.40625    '], time:1.206017ms
                                       out_f16(sk): ['-10.8359375 ', '24.734375   ', '12.40625    '], time:0.905061ms
                            out_f16x4pack(t4x4bcf): ['-10.8359375 ', '24.734375   ', '12.40625    '], time:0.132108ms
                         out_f16x4pack(t4x4offset): ['-10.8359375 ', '24.734375   ', '12.40625    '], time:0.129628ms
                                 out_f16x4(t8x8sk): ['-10.8359375 ', '24.734375   ', '12.40625    '], time:0.129080ms
                                out_f16x4(t8x8bcf): ['-10.8359375 ', '24.734375   ', '12.40625    '], time:0.109315ms
                             out_f16x4pack(t8x8sk): ['-10.8359375 ', '24.734375   ', '12.40625    '], time:0.110602ms
                                out_f16x4pack(bcf): ['-10.8359375 ', '24.734375   ', '12.40625    '], time:0.101447ms
                         out_f16x4pack(bcf+offset): ['-10.8359375 ', '24.734375   ', '12.40625    '], time:0.100470ms
                                out_f16x8pack(bcf): ['-10.8359375 ', '24.734375   ', '12.40625    '], time:0.098014ms
                         out_f16x8pack(bcf+offset): ['-10.8359375 ', '24.734375   ', '12.40625    '], time:0.097108ms
                           out_f16x8pack(bcf+dbuf): ['-10.8359375 ', '24.734375   ', '12.40625    '], time:0.094128ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['-10.8359375 ', '24.734375   ', '12.40625    '], time:0.089407ms
                    out_f16x8pack(k16+dbuf+offset): ['-10.8359375 ', '24.734375   ', '12.40625    '], time:0.089335ms
                     out_f16x8pack(k16+dbuf+async): ['-10.8359375 ', '24.734375   ', '12.40625    '], time:0.091100ms
                           out_f16x8pack(k32+dbuf): ['-10.8359375 ', '24.734375   ', '12.40625    '], time:0.090957ms
                     out_f16x8pack(k32+dbuf+async): ['-10.8359375 ', '24.734375   ', '12.40625    '], time:0.089574ms
                     out_f16x8pack(k32+dbuf+t16x8): ['-10.8359375 ', '24.734375   ', '12.40625    '], time:0.094652ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['-10.8359375 ', '24.734375   ', '12.40625    '], time:0.090528ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['-10.8359375 ', '24.78125    ', '12.390625   '], time:0.189352ms
                               out_f16wmma(mma4x2): ['-10.8359375 ', '24.78125    ', '12.390625   '], time:0.102687ms
                       out_f16wmma(mma4x2+warp2x4): ['-10.8359375 ', '24.78125    ', '12.390625   '], time:0.081158ms
                 out_f16wmma(mma4x2+warp2x4+async): ['-10.8359375 ', '24.78125    ', '12.390625   '], time:0.073290ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['-10.8359375 ', '24.78125    ', '12.390625   '], time:0.083494ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['-10.8359375 ', '24.78125    ', '12.390625   '], time:0.066614ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['-10.8359375 ', '24.78125    ', '12.390625   '], time:0.072050ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['-10.8359375 ', '24.78125    ', '12.390625   '], time:0.095201ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['-10.8359375 ', '24.78125    ', '12.390625   '], time:0.076270ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['-10.8359375 ', '24.78125    ', '12.390625   '], time:0.076079ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['-10.8359375 ', '24.78125    ', '12.390625   '], time:0.073814ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-10.8359375 ', '24.78125    ', '12.390625   '], time:0.067353ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['-10.8359375 ', '24.78125    ', '12.390625   '], time:0.070477ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['-10.8359375 ', '24.78125    ', '12.390625   '], time:0.072885ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['-10.8359375 ', '24.78125    ', '12.390625   '], time:0.076437ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['-10.8359375 ', '24.78125    ', '12.390625   '], time:0.064874ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['-10.8359375 ', '24.78125    ', '12.390625   '], time:0.066471ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['-10.8359375 ', '24.78125    ', '12.390625   '], time:0.064921ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['-10.8359375 ', '24.78125    ', '12.390625   '], time:0.065279ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['-10.8359375 ', '24.78125    ', '12.390625   '], time:0.064421ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['-10.8359375 ', '24.78125    ', '12.390625   '], time:0.063634ms
                                        out_f16_th: ['-10.8203125 ', '24.78125    ', '12.390625   '], time:0.047994ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=2048, K=512
                                           out_f16: ['-4.97265625 ', '46.9375     ', '-20.640625  '], time:2.385473ms
                                       out_f16(sk): ['-4.97265625 ', '46.9375     ', '-20.640625  '], time:1.779890ms
                            out_f16x4pack(t4x4bcf): ['-4.97265625 ', '46.9375     ', '-20.640625  '], time:0.260282ms
                         out_f16x4pack(t4x4offset): ['-4.97265625 ', '46.9375     ', '-20.640625  '], time:0.255466ms
                                 out_f16x4(t8x8sk): ['-4.97265625 ', '46.9375     ', '-20.640625  '], time:0.235271ms
                                out_f16x4(t8x8bcf): ['-4.97265625 ', '46.9375     ', '-20.640625  '], time:0.212359ms
                             out_f16x4pack(t8x8sk): ['-4.97265625 ', '46.9375     ', '-20.640625  '], time:0.210118ms
                                out_f16x4pack(bcf): ['-4.97265625 ', '46.9375     ', '-20.640625  '], time:0.195575ms
                         out_f16x4pack(bcf+offset): ['-4.97265625 ', '46.9375     ', '-20.640625  '], time:0.195003ms
                                out_f16x8pack(bcf): ['-4.97265625 ', '46.9375     ', '-20.640625  '], time:0.189281ms
                         out_f16x8pack(bcf+offset): ['-4.97265625 ', '46.9375     ', '-20.640625  '], time:0.187993ms
                           out_f16x8pack(bcf+dbuf): ['-4.97265625 ', '46.9375     ', '-20.640625  '], time:0.182176ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['-4.97265625 ', '46.9375     ', '-20.640625  '], time:0.171661ms
                    out_f16x8pack(k16+dbuf+offset): ['-4.97265625 ', '46.9375     ', '-20.640625  '], time:0.171161ms
                     out_f16x8pack(k16+dbuf+async): ['-4.97265625 ', '46.9375     ', '-20.640625  '], time:0.175500ms
                           out_f16x8pack(k32+dbuf): ['-4.97265625 ', '46.9375     ', '-20.640625  '], time:0.175881ms
                     out_f16x8pack(k32+dbuf+async): ['-4.97265625 ', '46.9375     ', '-20.640625  '], time:0.172901ms
                     out_f16x8pack(k32+dbuf+t16x8): ['-4.97265625 ', '46.9375     ', '-20.640625  '], time:0.182557ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['-4.97265625 ', '46.9375     ', '-20.640625  '], time:0.174761ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['-4.96484375 ', '46.75       ', '-20.65625   '], time:0.371170ms
                               out_f16wmma(mma4x2): ['-4.96484375 ', '46.75       ', '-20.65625   '], time:0.193906ms
                       out_f16wmma(mma4x2+warp2x4): ['-4.96484375 ', '46.75       ', '-20.65625   '], time:0.136995ms
                 out_f16wmma(mma4x2+warp2x4+async): ['-4.96484375 ', '46.75       ', '-20.65625   '], time:0.124860ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['-4.96484375 ', '46.75       ', '-20.65625   '], time:0.145268ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['-4.96484375 ', '46.75       ', '-20.65625   '], time:0.109553ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['-4.96484375 ', '46.75       ', '-20.65625   '], time:0.124979ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['-4.96484375 ', '46.75       ', '-20.65625   '], time:0.166750ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['-4.96484375 ', '46.75       ', '-20.65625   '], time:0.129437ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['-4.96484375 ', '46.75       ', '-20.65625   '], time:0.129390ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['-4.96484375 ', '46.75       ', '-20.65625   '], time:0.124693ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-4.96484375 ', '46.75       ', '-20.65625   '], time:0.113153ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['-4.96484375 ', '46.75       ', '-20.65625   '], time:0.120449ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['-4.96484375 ', '46.75       ', '-20.65625   '], time:0.125408ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['-4.96484375 ', '46.75       ', '-20.65625   '], time:0.129342ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['-4.96484375 ', '46.75       ', '-20.65625   '], time:0.106788ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['-4.96484375 ', '46.75       ', '-20.65625   '], time:0.110841ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['-4.96484375 ', '46.75       ', '-20.65625   '], time:0.106955ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['-4.96484375 ', '46.75       ', '-20.65625   '], time:0.106549ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['-4.96484375 ', '46.75       ', '-20.65625   '], time:0.105691ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['-4.96484375 ', '46.75       ', '-20.65625   '], time:0.104690ms
                                        out_f16_th: ['-4.9453125  ', '46.71875    ', '-20.59375   '], time:0.087333ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=2048, K=1024
                                           out_f16: ['45.25       ', '24.140625   ', '3.83007812  '], time:4.744935ms
                                       out_f16(sk): ['45.25       ', '24.140625   ', '3.83007812  '], time:3.529906ms
                            out_f16x4pack(t4x4bcf): ['45.25       ', '24.140625   ', '3.83007812  '], time:0.516844ms
                         out_f16x4pack(t4x4offset): ['45.25       ', '24.140625   ', '3.83007812  '], time:0.507426ms
                                 out_f16x4(t8x8sk): ['45.25       ', '24.140625   ', '3.83007812  '], time:0.446701ms
                                out_f16x4(t8x8bcf): ['45.25       ', '24.140625   ', '3.83007812  '], time:0.418591ms
                             out_f16x4pack(t8x8sk): ['45.25       ', '24.140625   ', '3.83007812  '], time:0.408840ms
                                out_f16x4pack(bcf): ['45.25       ', '24.140625   ', '3.83007812  '], time:0.384307ms
                         out_f16x4pack(bcf+offset): ['45.25       ', '24.140625   ', '3.83007812  '], time:0.383377ms
                                out_f16x8pack(bcf): ['45.25       ', '24.140625   ', '3.83007812  '], time:0.371933ms
                         out_f16x8pack(bcf+offset): ['45.25       ', '24.140625   ', '3.83007812  '], time:0.368762ms
                           out_f16x8pack(bcf+dbuf): ['45.25       ', '24.140625   ', '3.83007812  '], time:0.358057ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['45.25       ', '24.140625   ', '3.83007812  '], time:0.336719ms
                    out_f16x8pack(k16+dbuf+offset): ['45.25       ', '24.140625   ', '3.83007812  '], time:0.336242ms
                     out_f16x8pack(k16+dbuf+async): ['45.25       ', '24.140625   ', '3.83007812  '], time:0.344372ms
                           out_f16x8pack(k32+dbuf): ['45.25       ', '24.140625   ', '3.83007812  '], time:0.345874ms
                     out_f16x8pack(k32+dbuf+async): ['45.25       ', '24.140625   ', '3.83007812  '], time:0.339556ms
                     out_f16x8pack(k32+dbuf+t16x8): ['45.25       ', '24.140625   ', '3.83007812  '], time:0.359797ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['45.25       ', '24.140625   ', '3.83007812  '], time:0.342870ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['45.28125    ', '24.25       ', '3.90234375  '], time:0.735450ms
                               out_f16wmma(mma4x2): ['45.28125    ', '24.25       ', '3.90234375  '], time:0.376248ms
                       out_f16wmma(mma4x2+warp2x4): ['45.28125    ', '24.25       ', '3.90234375  '], time:0.247836ms
                 out_f16wmma(mma4x2+warp2x4+async): ['45.28125    ', '24.25       ', '3.90234375  '], time:0.228620ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['45.28125    ', '24.25       ', '3.90234375  '], time:0.272059ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['45.28125    ', '24.25       ', '3.90234375  '], time:0.197005ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['45.28125    ', '24.25       ', '3.90234375  '], time:0.229740ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['45.28125    ', '24.25       ', '3.90234375  '], time:0.310111ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['45.28125    ', '24.25       ', '3.90234375  '], time:0.237203ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['45.28125    ', '24.25       ', '3.90234375  '], time:0.236964ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['45.28125    ', '24.25       ', '3.90234375  '], time:0.227427ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['45.28125    ', '24.25       ', '3.90234375  '], time:0.204277ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['45.28125    ', '24.25       ', '3.90234375  '], time:0.222158ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['45.28125    ', '24.25       ', '3.90234375  '], time:0.230742ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['45.28125    ', '24.25       ', '3.90234375  '], time:0.237060ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['45.28125    ', '24.25       ', '3.90234375  '], time:0.191450ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['45.28125    ', '24.25       ', '3.90234375  '], time:0.199676ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['45.28125    ', '24.25       ', '3.90234375  '], time:0.189853ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['45.28125    ', '24.25       ', '3.90234375  '], time:0.188637ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['45.28125    ', '24.25       ', '3.90234375  '], time:0.187707ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['45.28125    ', '24.25       ', '3.90234375  '], time:0.187111ms
                                        out_f16_th: ['45.28125    ', '24.171875   ', '3.94140625  '], time:0.169444ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=256
                                           out_f16: ['-0.58691406 ', '-17.953125  ', '7.4140625   '], time:2.402997ms
                                       out_f16(sk): ['-0.58691406 ', '-17.953125  ', '7.4140625   '], time:1.800108ms
                            out_f16x4pack(t4x4bcf): ['-0.58691406 ', '-17.953125  ', '7.4140625   '], time:0.253296ms
                         out_f16x4pack(t4x4offset): ['-0.58691406 ', '-17.953125  ', '7.4140625   '], time:0.248623ms
                                 out_f16x4(t8x8sk): ['-0.58691406 ', '-17.953125  ', '7.4140625   '], time:0.230861ms
                                out_f16x4(t8x8bcf): ['-0.58691406 ', '-17.953125  ', '7.4140625   '], time:0.208879ms
                             out_f16x4pack(t8x8sk): ['-0.58691406 ', '-17.953125  ', '7.4140625   '], time:0.207138ms
                                out_f16x4pack(bcf): ['-0.58691406 ', '-17.953125  ', '7.4140625   '], time:0.192046ms
                         out_f16x4pack(bcf+offset): ['-0.58691406 ', '-17.953125  ', '7.4140625   '], time:0.191498ms
                                out_f16x8pack(bcf): ['-0.58691406 ', '-17.953125  ', '7.4140625   '], time:0.185490ms
                         out_f16x8pack(bcf+offset): ['-0.58691406 ', '-17.953125  ', '7.4140625   '], time:0.183773ms
                           out_f16x8pack(bcf+dbuf): ['-0.58691406 ', '-17.953125  ', '7.4140625   '], time:0.181341ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['-0.58691406 ', '-17.953125  ', '7.4140625   '], time:0.172448ms
                    out_f16x8pack(k16+dbuf+offset): ['-0.58691406 ', '-17.953125  ', '7.4140625   '], time:0.172043ms
                     out_f16x8pack(k16+dbuf+async): ['-0.58691406 ', '-17.953125  ', '7.4140625   '], time:0.175595ms
                           out_f16x8pack(k32+dbuf): ['-0.58691406 ', '-17.953125  ', '7.4140625   '], time:0.175357ms
                     out_f16x8pack(k32+dbuf+async): ['-0.58691406 ', '-17.953125  ', '7.4140625   '], time:0.172949ms
                     out_f16x8pack(k32+dbuf+t16x8): ['-0.58691406 ', '-17.953125  ', '7.4140625   '], time:0.181293ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['-0.58691406 ', '-17.953125  ', '7.4140625   '], time:0.174093ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['-0.60009766 ', '-17.953125  ', '7.390625    '], time:0.377798ms
                               out_f16wmma(mma4x2): ['-0.60009766 ', '-17.953125  ', '7.390625    '], time:0.200987ms
                       out_f16wmma(mma4x2+warp2x4): ['-0.60009766 ', '-17.953125  ', '7.390625    '], time:0.155807ms
                 out_f16wmma(mma4x2+warp2x4+async): ['-0.60009766 ', '-17.953125  ', '7.390625    '], time:0.134206ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['-0.60009766 ', '-17.953125  ', '7.390625    '], time:0.157523ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['-0.60009766 ', '-17.953125  ', '7.390625    '], time:0.118470ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['-0.60009766 ', '-17.953125  ', '7.390625    '], time:0.129557ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['-0.60009766 ', '-17.953125  ', '7.390625    '], time:0.180840ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['-0.60009766 ', '-17.953125  ', '7.390625    '], time:0.137234ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['-0.60009766 ', '-17.953125  ', '7.390625    '], time:0.137496ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['-0.60009766 ', '-17.953125  ', '7.390625    '], time:0.134540ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-0.60009766 ', '-17.953125  ', '7.390625    '], time:0.120592ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['-0.60009766 ', '-17.953125  ', '7.390625    '], time:0.128198ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['-0.60009766 ', '-17.953125  ', '7.390625    '], time:0.132465ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['-0.60009766 ', '-17.953125  ', '7.390625    '], time:0.137448ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['-0.60009766 ', '-17.953125  ', '7.390625    '], time:0.116706ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['-0.60009766 ', '-17.953125  ', '7.390625    '], time:0.122643ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['-0.60009766 ', '-17.953125  ', '7.390625    '], time:0.117302ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['-0.60009766 ', '-17.953125  ', '7.390625    '], time:0.116229ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['-0.60009766 ', '-17.953125  ', '7.390625    '], time:0.114942ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['-0.60009766 ', '-17.953125  ', '7.390625    '], time:0.111628ms
                                        out_f16_th: ['-0.59521484 ', '-17.921875  ', '7.3984375   '], time:0.090408ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=512
                                           out_f16: ['-23.515625  ', '12.875      ', '7.078125    '], time:4.755092ms
                                       out_f16(sk): ['-23.515625  ', '12.875      ', '7.078125    '], time:3.542042ms
                            out_f16x4pack(t4x4bcf): ['-23.515625  ', '12.875      ', '7.078125    '], time:0.501323ms
                         out_f16x4pack(t4x4offset): ['-23.515625  ', '12.875      ', '7.078125    '], time:0.491619ms
                                 out_f16x4(t8x8sk): ['-23.515625  ', '12.875      ', '7.078125    '], time:0.441432ms
                                out_f16x4(t8x8bcf): ['-23.515625  ', '12.875      ', '7.078125    '], time:0.410342ms
                             out_f16x4pack(t8x8sk): ['-23.515625  ', '12.875      ', '7.078125    '], time:0.401425ms
                                out_f16x4pack(bcf): ['-23.515625  ', '12.875      ', '7.078125    '], time:0.376105ms
                         out_f16x4pack(bcf+offset): ['-23.515625  ', '12.875      ', '7.078125    '], time:0.374722ms
                                out_f16x8pack(bcf): ['-23.515625  ', '12.875      ', '7.078125    '], time:0.362206ms
                         out_f16x8pack(bcf+offset): ['-23.515625  ', '12.875      ', '7.078125    '], time:0.359178ms
                           out_f16x8pack(bcf+dbuf): ['-23.515625  ', '12.875      ', '7.078125    '], time:0.355887ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['-23.515625  ', '12.875      ', '7.078125    '], time:0.337195ms
                    out_f16x8pack(k16+dbuf+offset): ['-23.515625  ', '12.875      ', '7.078125    '], time:0.337029ms
                     out_f16x8pack(k16+dbuf+async): ['-23.515625  ', '12.875      ', '7.078125    '], time:0.344467ms
                           out_f16x8pack(k32+dbuf): ['-23.515625  ', '12.875      ', '7.078125    '], time:0.344253ms
                     out_f16x8pack(k32+dbuf+async): ['-23.515625  ', '12.875      ', '7.078125    '], time:0.339985ms
                     out_f16x8pack(k32+dbuf+t16x8): ['-23.515625  ', '12.875      ', '7.078125    '], time:0.356221ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['-23.515625  ', '12.875      ', '7.078125    '], time:0.342131ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['-23.5       ', '12.8984375  ', '7.1015625   '], time:0.742722ms
                               out_f16wmma(mma4x2): ['-23.5       ', '12.8984375  ', '7.1015625   '], time:0.380898ms
                       out_f16wmma(mma4x2+warp2x4): ['-23.5       ', '12.8984375  ', '7.1015625   '], time:0.267577ms
                 out_f16wmma(mma4x2+warp2x4+async): ['-23.5       ', '12.8984375  ', '7.1015625   '], time:0.237274ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['-23.5       ', '12.8984375  ', '7.1015625   '], time:0.280190ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['-23.5       ', '12.8984375  ', '7.1015625   '], time:0.203109ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['-23.5       ', '12.8984375  ', '7.1015625   '], time:0.232816ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['-23.5       ', '12.8984375  ', '7.1015625   '], time:0.320888ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['-23.5       ', '12.8984375  ', '7.1015625   '], time:0.243497ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['-23.5       ', '12.8984375  ', '7.1015625   '], time:0.243711ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['-23.5       ', '12.8984375  ', '7.1015625   '], time:0.236869ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['-23.5       ', '12.8984375  ', '7.1015625   '], time:0.212741ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['-23.5       ', '12.8984375  ', '7.1015625   '], time:0.227141ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['-23.5       ', '12.8984375  ', '7.1015625   '], time:0.235581ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['-23.5       ', '12.8984375  ', '7.1015625   '], time:0.243592ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['-23.5       ', '12.8984375  ', '7.1015625   '], time:0.200629ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['-23.5       ', '12.8984375  ', '7.1015625   '], time:0.210690ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['-23.5       ', '12.8984375  ', '7.1015625   '], time:0.199366ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['-23.5       ', '12.8984375  ', '7.1015625   '], time:0.197983ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['-23.5       ', '12.8984375  ', '7.1015625   '], time:0.197291ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['-23.5       ', '12.8984375  ', '7.1015625   '], time:0.191116ms
                                        out_f16_th: ['-23.484375  ', '12.9609375  ', '7.11328125  '], time:0.169373ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=1024
                                           out_f16: ['41.59375    ', '-34.875     ', '-8.9921875  '], time:9.460735ms
                                       out_f16(sk): ['41.59375    ', '-34.875     ', '-8.9921875  '], time:7.025695ms
                            out_f16x4pack(t4x4bcf): ['41.59375    ', '-34.875     ', '-8.9921875  '], time:0.996423ms
                         out_f16x4pack(t4x4offset): ['41.59375    ', '-34.875     ', '-8.9921875  '], time:0.978088ms
                                 out_f16x4(t8x8sk): ['41.59375    ', '-34.875     ', '-8.9921875  '], time:0.855255ms
                                out_f16x4(t8x8bcf): ['41.59375    ', '-34.875     ', '-8.9921875  '], time:0.811362ms
                             out_f16x4pack(t8x8sk): ['41.59375    ', '-34.875     ', '-8.9921875  '], time:0.788498ms
                                out_f16x4pack(bcf): ['41.59375    ', '-34.875     ', '-8.9921875  '], time:0.742698ms
                         out_f16x4pack(bcf+offset): ['41.59375    ', '-34.875     ', '-8.9921875  '], time:0.740671ms
                                out_f16x8pack(bcf): ['41.59375    ', '-34.875     ', '-8.9921875  '], time:0.713873ms
                         out_f16x8pack(bcf+offset): ['41.59375    ', '-34.875     ', '-8.9921875  '], time:0.708413ms
                           out_f16x8pack(bcf+dbuf): ['41.59375    ', '-34.875     ', '-8.9921875  '], time:0.703883ms
---------------------------------------------------------Async----------------------------------------------------------
                           out_f16x8pack(k16+dbuf): ['41.59375    ', '-34.875     ', '-8.9921875  '], time:0.666285ms
                    out_f16x8pack(k16+dbuf+offset): ['41.59375    ', '-34.875     ', '-8.9921875  '], time:0.665522ms
                     out_f16x8pack(k16+dbuf+async): ['41.59375    ', '-34.875     ', '-8.9921875  '], time:0.680661ms
                           out_f16x8pack(k32+dbuf): ['41.59375    ', '-34.875     ', '-8.9921875  '], time:0.680065ms
                     out_f16x8pack(k32+dbuf+async): ['41.59375    ', '-34.875     ', '-8.9921875  '], time:0.671506ms
                     out_f16x8pack(k32+dbuf+t16x8): ['41.59375    ', '-34.875     ', '-8.9921875  '], time:0.705028ms
               out_f16x8pack(k32+dbuf+t16x8+async): ['41.59375    ', '-34.875     ', '-8.9921875  '], time:0.676179ms
----------------------------------------------------------WMMA----------------------------------------------------------
                               out_f16wmma(+naive): ['41.625      ', '-35.0       ', '-9.03125    '], time:1.470327ms
                               out_f16wmma(mma4x2): ['41.625      ', '-35.0       ', '-9.03125    '], time:0.744915ms
                       out_f16wmma(mma4x2+warp2x4): ['41.625      ', '-35.0       ', '-9.03125    '], time:0.488520ms
                 out_f16wmma(mma4x2+warp2x4+async): ['41.625      ', '-35.0       ', '-9.03125    '], time:0.443053ms
               out_f16wmma(mma4x2+warp2x4x2+async): ['41.625      ', '-35.0       ', '-9.03125    '], time:0.533032ms
          out_f16wmma(mma4x2+warp2x4+async+offset): ['41.625      ', '-35.0       ', '-9.03125    '], time:0.374126ms
        out_f16wmma(mma4x2+warp2x4x2+async+offset): ['41.625      ', '-35.0       ', '-9.03125    '], time:0.439787ms
                out_f16wmma(mma4x4+warp2x2x2+dbuf): ['41.625      ', '-35.0       ', '-9.03125    '], time:0.607800ms
                out_f16wmma(mma4x2+warp2x4x2+dbuf): ['41.625      ', '-35.0       ', '-9.03125    '], time:0.459790ms
                out_f16wmma(mma4x2+warp2x4x2+rbuf): ['41.625      ', '-35.0       ', '-9.03125    '], time:0.459981ms
                  out_f16wmma(mma4x2+warp2x4+dbuf): ['41.625      ', '-35.0       ', '-9.03125    '], time:0.440788ms
         out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['41.625      ', '-35.0       ', '-9.03125    '], time:0.394583ms
                out_f16wmma(mma2x4+warp2x4+stage3): ['41.625      ', '-35.0       ', '-9.03125    '], time:0.428200ms
                out_f16wmma(mma2x4+warp2x4+stage4): ['41.625      ', '-35.0       ', '-9.03125    '], time:0.445366ms
         out_f16wmma(mma4x2+warp2x4x2+rbuf+offset): ['41.625      ', '-35.0       ', '-9.03125    '], time:0.459909ms
         out_f16wmma(mma4x2+warp2x4x2+dbuf+offset): ['41.625      ', '-35.0       ', '-9.03125    '], time:0.370097ms
         out_f16wmma(mma4x4+warp2x2x2+dbuf+offset): ['41.625      ', '-35.0       ', '-9.03125    '], time:0.388718ms
  out_f16wmma(m32n8k16+mma2x4+warp2x4+dbuf+offset): ['41.625      ', '-35.0       ', '-9.03125    '], time:0.366640ms
         out_f16wmma(mma4x2+warp2x4+stage4+offset): ['41.625      ', '-35.0       ', '-9.03125    '], time:0.362015ms
           out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['41.625      ', '-35.0       ', '-9.03125    '], time:0.362086ms
         out_f16wmma(mma4x2+warp2x4+stage3+offset): ['41.625      ', '-35.0       ', '-9.03125    '], time:0.355792ms
                                        out_f16_th: ['41.625      ', '-35.0       ', '-8.984375   '], time:0.334668ms
------------------------------------------------------------------------------------------------------------------------
```
