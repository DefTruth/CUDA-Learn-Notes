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
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_stage2/3/4(Tensor Cores, Tile MMA/Warp, Copy Async, Stage, Pad, Thread block swizzle) 
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
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=1024
                                   f16(naive): ['21.78125    ', '17.46875    ', '-22.109375  '], time:18.93934ms, TFLOPS: 3.63
                          f16x8pack(t8x8+bcf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:1.344370ms, TFLOPS: 51.12
                     f16x8pack(t8x8+bcf+dbuf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:1.333773ms, TFLOPS: 51.52
                     f16x8pack(t8x8+k16+dbuf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:1.264774ms, TFLOPS: 54.33
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['21.859375   ', '17.53125    ', '-22.234375  '], time:3.005743ms, TFLOPS: 22.86
                              f16wmma(mma4x2): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.493060ms, TFLOPS: 46.03
                      f16wmma(mma4x2+warp2x4): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.958561ms, TFLOPS: 71.69
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.691437ms, TFLOPS: 99.39
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.697398ms, TFLOPS: 98.54
               f16wmma(mma2x4+warp2x4+stage3): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.675678ms, TFLOPS: 101.70
               f16wmma(mma2x4+warp2x4+stage2): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.680208ms, TFLOPS: 101.03
            f16wmma(warp2x4+...+stage3+dsmem): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.678741ms, TFLOPS: 101.25
            f16wmma(warp2x4+...+stage2+dsmem): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.682616ms, TFLOPS: 100.67
          f16wmma(warp2x4+...+stage3+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.675940ms, TFLOPS: 101.66
          f16wmma(warp2x4+...+stage2+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.667917ms, TFLOPS: 102.89
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.675320ms, TFLOPS: 101.76
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.668203ms, TFLOPS: 102.84
                                  f16(cublas): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.633251ms, TFLOPS: 108.52
                                       f16_th: ['21.875      ', '17.484375   ', '-22.25      '], time:0.620317ms, TFLOPS: 110.78
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=2048
                                   f16(naive): ['49.875      ', '43.28125    ', '39.90625    '], time:37.66349ms, TFLOPS: 3.65
                          f16x8pack(t8x8+bcf): ['49.875      ', '43.28125    ', '39.90625    '], time:2.740716ms, TFLOPS: 50.15
                     f16x8pack(t8x8+bcf+dbuf): ['49.875      ', '43.28125    ', '39.90625    '], time:2.753663ms, TFLOPS: 49.91
                     f16x8pack(t8x8+k16+dbuf): ['49.875      ', '43.28125    ', '39.90625    '], time:2.644252ms, TFLOPS: 51.98
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['50.15625    ', '43.0        ', '39.8125     '], time:5.973792ms, TFLOPS: 23.01
                              f16wmma(mma4x2): ['50.15625    ', '43.0        ', '39.8125     '], time:2.955889ms, TFLOPS: 46.50
                      f16wmma(mma4x2+warp2x4): ['50.15625    ', '43.0        ', '39.8125     '], time:1.849830ms, TFLOPS: 74.30
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['50.15625    ', '43.0        ', '39.8125     '], time:1.328873ms, TFLOPS: 103.43
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['50.15625    ', '43.0        ', '39.8125     '], time:1.340615ms, TFLOPS: 102.52
               f16wmma(mma2x4+warp2x4+stage3): ['50.15625    ', '43.0        ', '39.8125     '], time:1.307237ms, TFLOPS: 105.14
               f16wmma(mma2x4+warp2x4+stage2): ['50.15625    ', '43.0        ', '39.8125     '], time:1.312625ms, TFLOPS: 104.71
            f16wmma(warp2x4+...+stage3+dsmem): ['50.15625    ', '43.0        ', '39.8125     '], time:1.308298ms, TFLOPS: 105.05
            f16wmma(warp2x4+...+stage2+dsmem): ['50.15625    ', '43.0        ', '39.8125     '], time:1.304531ms, TFLOPS: 105.36
          f16wmma(warp2x4+...+stage3+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:1.297235ms, TFLOPS: 105.95
          f16wmma(warp2x4+...+stage2+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:1.285755ms, TFLOPS: 106.89
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:1.298272ms, TFLOPS: 105.86
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:1.285552ms, TFLOPS: 106.91
                                  f16(cublas): ['50.15625    ', '43.0        ', '39.8125     '], time:1.231002ms, TFLOPS: 111.65
                                       f16_th: ['50.1875     ', '43.09375    ', '39.78125    '], time:1.288235ms, TFLOPS: 106.69
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=512
                                   f16(naive): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:18.95830ms, TFLOPS: 3.62
                          f16x8pack(t8x8+bcf): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:1.326978ms, TFLOPS: 51.79
                     f16x8pack(t8x8+bcf+dbuf): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:1.306009ms, TFLOPS: 52.62
                     f16x8pack(t8x8+k16+dbuf): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:1.241207ms, TFLOPS: 55.37
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:3.100228ms, TFLOPS: 22.17
                              f16wmma(mma4x2): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.510787ms, TFLOPS: 45.49
                      f16wmma(mma4x2+warp2x4): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.982809ms, TFLOPS: 69.92
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.723981ms, TFLOPS: 94.92
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.732052ms, TFLOPS: 93.87
               f16wmma(mma2x4+warp2x4+stage3): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.698947ms, TFLOPS: 98.32
               f16wmma(mma2x4+warp2x4+stage2): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.715672ms, TFLOPS: 96.02
            f16wmma(warp2x4+...+stage3+dsmem): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.719213ms, TFLOPS: 95.55
            f16wmma(warp2x4+...+stage2+dsmem): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.720822ms, TFLOPS: 95.33
          f16wmma(warp2x4+...+stage3+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.718688ms, TFLOPS: 95.62
          f16wmma(warp2x4+...+stage2+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.708985ms, TFLOPS: 96.93
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.718843ms, TFLOPS: 95.60
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.711393ms, TFLOPS: 96.60
                                  f16(cublas): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.652146ms, TFLOPS: 105.37
                                       f16_th: ['5.44921875  ', '11.53125    ', '-11.5625    '], time:0.621509ms, TFLOPS: 110.57
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=1024
                                   f16(naive): ['21.78125    ', '17.46875    ', '-22.109375  '], time:37.76177ms, TFLOPS: 3.64
                          f16x8pack(t8x8+bcf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:2.628862ms, TFLOPS: 52.28
                     f16x8pack(t8x8+bcf+dbuf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:2.659785ms, TFLOPS: 51.67
                     f16x8pack(t8x8+k16+dbuf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:2.635407ms, TFLOPS: 52.15
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['21.859375   ', '17.53125    ', '-22.234375  '], time:5.941581ms, TFLOPS: 23.13
                              f16wmma(mma4x2): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.955853ms, TFLOPS: 46.50
                      f16wmma(mma4x2+warp2x4): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.832103ms, TFLOPS: 75.02
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.345503ms, TFLOPS: 102.15
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.358783ms, TFLOPS: 101.15
               f16wmma(mma2x4+warp2x4+stage3): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.310122ms, TFLOPS: 104.91
               f16wmma(mma2x4+warp2x4+stage2): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.336181ms, TFLOPS: 102.86
            f16wmma(warp2x4+...+stage3+dsmem): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.333665ms, TFLOPS: 103.05
            f16wmma(warp2x4+...+stage2+dsmem): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.334536ms, TFLOPS: 102.99
          f16wmma(warp2x4+...+stage3+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.326632ms, TFLOPS: 103.60
          f16wmma(warp2x4+...+stage2+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.321804ms, TFLOPS: 103.98
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.333987ms, TFLOPS: 103.03
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.320850ms, TFLOPS: 104.05
                                  f16(cublas): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.239788ms, TFLOPS: 110.86
                                       f16_th: ['21.875      ', '17.484375   ', '-22.25      '], time:1.211833ms, TFLOPS: 113.41
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=2048
                                   f16(naive): ['49.875      ', '43.28125    ', '39.90625    '], time:75.27230ms, TFLOPS: 3.65
                          f16x8pack(t8x8+bcf): ['49.875      ', '43.28125    ', '39.90625    '], time:5.465996ms, TFLOPS: 50.29
                     f16x8pack(t8x8+bcf+dbuf): ['49.875      ', '43.28125    ', '39.90625    '], time:5.459189ms, TFLOPS: 50.35
                     f16x8pack(t8x8+k16+dbuf): ['49.875      ', '43.28125    ', '39.90625    '], time:5.234515ms, TFLOPS: 52.51
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['50.15625    ', '43.0        ', '39.8125     '], time:11.67826ms, TFLOPS: 23.54
                              f16wmma(mma4x2): ['50.15625    ', '43.0        ', '39.8125     '], time:5.873489ms, TFLOPS: 46.80
                      f16wmma(mma4x2+warp2x4): ['50.15625    ', '43.0        ', '39.8125     '], time:3.521764ms, TFLOPS: 78.05
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['50.15625    ', '43.0        ', '39.8125     '], time:2.574265ms, TFLOPS: 106.78
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['50.15625    ', '43.0        ', '39.8125     '], time:2.598905ms, TFLOPS: 105.77
               f16wmma(mma2x4+warp2x4+stage3): ['50.15625    ', '43.0        ', '39.8125     '], time:2.539229ms, TFLOPS: 108.25
               f16wmma(mma2x4+warp2x4+stage2): ['50.15625    ', '43.0        ', '39.8125     '], time:2.555441ms, TFLOPS: 107.57
            f16wmma(warp2x4+...+stage3+dsmem): ['50.15625    ', '43.0        ', '39.8125     '], time:2.555561ms, TFLOPS: 107.56
            f16wmma(warp2x4+...+stage2+dsmem): ['50.15625    ', '43.0        ', '39.8125     '], time:2.549624ms, TFLOPS: 107.81
          f16wmma(warp2x4+...+stage3+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:2.555215ms, TFLOPS: 107.58
          f16wmma(warp2x4+...+stage2+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:2.549922ms, TFLOPS: 107.80
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:2.569234ms, TFLOPS: 106.99
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:2.549672ms, TFLOPS: 107.81
                                  f16(cublas): ['50.15625    ', '43.0        ', '39.8125     '], time:2.422904ms, TFLOPS: 113.45
                                       f16_th: ['50.1875     ', '43.09375    ', '39.78125    '], time:2.402675ms, TFLOPS: 114.40
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=512
                                   f16(naive): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:9.542524ms, TFLOPS: 3.60
                          f16x8pack(t8x8+bcf): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:0.681710ms, TFLOPS: 50.40
                     f16x8pack(t8x8+bcf+dbuf): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:0.684034ms, TFLOPS: 50.23
                     f16x8pack(t8x8+k16+dbuf): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:0.649583ms, TFLOPS: 52.90
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.560008ms, TFLOPS: 22.03
                              f16wmma(mma4x2): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.780928ms, TFLOPS: 44.00
                      f16wmma(mma4x2+warp2x4): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.536060ms, TFLOPS: 64.10
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.384855ms, TFLOPS: 89.28
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.387156ms, TFLOPS: 88.75
               f16wmma(mma2x4+warp2x4+stage3): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.364565ms, TFLOPS: 94.25
               f16wmma(mma2x4+warp2x4+stage2): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.363349ms, TFLOPS: 94.56
            f16wmma(warp2x4+...+stage3+dsmem): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.363707ms, TFLOPS: 94.47
            f16wmma(warp2x4+...+stage2+dsmem): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.364458ms, TFLOPS: 94.28
          f16wmma(warp2x4+...+stage3+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.364172ms, TFLOPS: 94.35
          f16wmma(warp2x4+...+stage2+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.357496ms, TFLOPS: 96.11
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.366485ms, TFLOPS: 93.75
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.360190ms, TFLOPS: 95.39
                                  f16(cublas): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.336992ms, TFLOPS: 101.96
                                       f16_th: ['5.44921875  ', '11.53125    ', '-11.5625    '], time:0.324618ms, TFLOPS: 105.85
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=1024
                                   f16(naive): ['21.78125    ', '17.46875    ', '-22.109375  '], time:18.94689ms, TFLOPS: 3.63
                          f16x8pack(t8x8+bcf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:1.343750ms, TFLOPS: 51.14
                     f16x8pack(t8x8+bcf+dbuf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:1.357579ms, TFLOPS: 50.62
                     f16x8pack(t8x8+k16+dbuf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:1.335418ms, TFLOPS: 51.46
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['21.859375   ', '17.53125    ', '-22.234375  '], time:3.016197ms, TFLOPS: 22.78
                              f16wmma(mma4x2): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.495897ms, TFLOPS: 45.94
                      f16wmma(mma4x2+warp2x4): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.961041ms, TFLOPS: 71.51
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.690722ms, TFLOPS: 99.49
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.696194ms, TFLOPS: 98.71
               f16wmma(mma2x4+warp2x4+stage3): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.674891ms, TFLOPS: 101.82
               f16wmma(mma2x4+warp2x4+stage2): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.678718ms, TFLOPS: 101.25
            f16wmma(warp2x4+...+stage3+dsmem): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.677633ms, TFLOPS: 101.41
            f16wmma(warp2x4+...+stage2+dsmem): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.678062ms, TFLOPS: 101.35
          f16wmma(warp2x4+...+stage3+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.674557ms, TFLOPS: 101.87
          f16wmma(warp2x4+...+stage2+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.666856ms, TFLOPS: 103.05
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.675451ms, TFLOPS: 101.74
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.667393ms, TFLOPS: 102.97
                                  f16(cublas): ['21.859375   ', '17.53125    ', '-22.234375  '], time:0.632560ms, TFLOPS: 108.64
                                       f16_th: ['21.875      ', '17.484375   ', '-22.25      '], time:0.619578ms, TFLOPS: 110.91
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=2048
                                   f16(naive): ['49.875      ', '43.28125    ', '39.90625    '], time:37.68422ms, TFLOPS: 3.65
                          f16x8pack(t8x8+bcf): ['49.875      ', '43.28125    ', '39.90625    '], time:2.768194ms, TFLOPS: 49.65
                     f16x8pack(t8x8+bcf+dbuf): ['49.875      ', '43.28125    ', '39.90625    '], time:2.800679ms, TFLOPS: 49.07
                     f16x8pack(t8x8+k16+dbuf): ['49.875      ', '43.28125    ', '39.90625    '], time:2.673280ms, TFLOPS: 51.41
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['50.15625    ', '43.0        ', '39.8125     '], time:5.933570ms, TFLOPS: 23.16
                              f16wmma(mma4x2): ['50.15625    ', '43.0        ', '39.8125     '], time:2.955126ms, TFLOPS: 46.51
                      f16wmma(mma4x2+warp2x4): ['50.15625    ', '43.0        ', '39.8125     '], time:1.821017ms, TFLOPS: 75.47
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['50.15625    ', '43.0        ', '39.8125     '], time:1.328396ms, TFLOPS: 103.46
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['50.15625    ', '43.0        ', '39.8125     '], time:1.339840ms, TFLOPS: 102.58
               f16wmma(mma2x4+warp2x4+stage3): ['50.15625    ', '43.0        ', '39.8125     '], time:1.304149ms, TFLOPS: 105.39
               f16wmma(mma2x4+warp2x4+stage2): ['50.15625    ', '43.0        ', '39.8125     '], time:1.305365ms, TFLOPS: 105.29
            f16wmma(warp2x4+...+stage3+dsmem): ['50.15625    ', '43.0        ', '39.8125     '], time:1.306915ms, TFLOPS: 105.16
            f16wmma(warp2x4+...+stage2+dsmem): ['50.15625    ', '43.0        ', '39.8125     '], time:1.304483ms, TFLOPS: 105.36
          f16wmma(warp2x4+...+stage3+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:1.297247ms, TFLOPS: 105.95
          f16wmma(warp2x4+...+stage2+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:1.285398ms, TFLOPS: 106.92
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:1.297080ms, TFLOPS: 105.96
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:1.285696ms, TFLOPS: 106.90
                                  f16(cublas): ['50.15625    ', '43.0        ', '39.8125     '], time:1.232063ms, TFLOPS: 111.55
                                       f16_th: ['50.1875     ', '43.09375    ', '39.78125    '], time:1.288735ms, TFLOPS: 106.65
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=512
                                   f16(naive): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:18.95883ms, TFLOPS: 3.62
                          f16x8pack(t8x8+bcf): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:1.332581ms, TFLOPS: 51.57
                     f16x8pack(t8x8+bcf+dbuf): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:1.307284ms, TFLOPS: 52.57
                     f16x8pack(t8x8+k16+dbuf): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:1.344895ms, TFLOPS: 51.10
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:3.099954ms, TFLOPS: 22.17
                              f16wmma(mma4x2): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.535665ms, TFLOPS: 44.75
                      f16wmma(mma4x2+warp2x4): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.982987ms, TFLOPS: 69.91
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.723874ms, TFLOPS: 94.93
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.731360ms, TFLOPS: 93.96
               f16wmma(mma2x4+warp2x4+stage3): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.702631ms, TFLOPS: 97.80
               f16wmma(mma2x4+warp2x4+stage2): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.724279ms, TFLOPS: 94.88
            f16wmma(warp2x4+...+stage3+dsmem): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.724601ms, TFLOPS: 94.84
            f16wmma(warp2x4+...+stage2+dsmem): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.726795ms, TFLOPS: 94.55
          f16wmma(warp2x4+...+stage3+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.723958ms, TFLOPS: 94.92
          f16wmma(warp2x4+...+stage2+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.712919ms, TFLOPS: 96.39
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.725376ms, TFLOPS: 94.74
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.714111ms, TFLOPS: 96.23
                                  f16(cublas): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.657725ms, TFLOPS: 104.48
                                       f16_th: ['5.44921875  ', '11.53125    ', '-11.5625    '], time:0.636243ms, TFLOPS: 108.01
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=1024
                                   f16(naive): ['21.78125    ', '17.46875    ', '-22.109375  '], time:37.76819ms, TFLOPS: 3.64
                          f16x8pack(t8x8+bcf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:2.648973ms, TFLOPS: 51.88
                     f16x8pack(t8x8+bcf+dbuf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:2.781200ms, TFLOPS: 49.42
                     f16x8pack(t8x8+k16+dbuf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:2.651882ms, TFLOPS: 51.83
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['21.859375   ', '17.53125    ', '-22.234375  '], time:5.951118ms, TFLOPS: 23.09
                              f16wmma(mma4x2): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.956569ms, TFLOPS: 46.49
                      f16wmma(mma4x2+warp2x4): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.840198ms, TFLOPS: 74.69
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.345336ms, TFLOPS: 102.16
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.359689ms, TFLOPS: 101.08
               f16wmma(mma2x4+warp2x4+stage3): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.323914ms, TFLOPS: 103.81
               f16wmma(mma2x4+warp2x4+stage2): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.335108ms, TFLOPS: 102.94
            f16wmma(warp2x4+...+stage3+dsmem): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.333093ms, TFLOPS: 103.10
            f16wmma(warp2x4+...+stage2+dsmem): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.340603ms, TFLOPS: 102.52
          f16wmma(warp2x4+...+stage3+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.334607ms, TFLOPS: 102.98
          f16wmma(warp2x4+...+stage2+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.320970ms, TFLOPS: 104.04
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.339578ms, TFLOPS: 102.60
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.327419ms, TFLOPS: 103.54
                                  f16(cublas): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.238250ms, TFLOPS: 110.99
                                       f16_th: ['21.875      ', '17.484375   ', '-22.25      '], time:1.220488ms, TFLOPS: 112.61
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=2048
                                   f16(naive): ['49.875      ', '43.28125    ', '39.90625    '], time:75.26080ms, TFLOPS: 3.65
                          f16x8pack(t8x8+bcf): ['49.875      ', '43.28125    ', '39.90625    '], time:5.473327ms, TFLOPS: 50.22
                     f16x8pack(t8x8+bcf+dbuf): ['49.875      ', '43.28125    ', '39.90625    '], time:5.490589ms, TFLOPS: 50.06
                     f16x8pack(t8x8+k16+dbuf): ['49.875      ', '43.28125    ', '39.90625    '], time:5.267274ms, TFLOPS: 52.19
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['50.15625    ', '43.0        ', '39.8125     '], time:11.67755ms, TFLOPS: 23.54
                              f16wmma(mma4x2): ['50.15625    ', '43.0        ', '39.8125     '], time:5.863451ms, TFLOPS: 46.88
                      f16wmma(mma4x2+warp2x4): ['50.15625    ', '43.0        ', '39.8125     '], time:3.542840ms, TFLOPS: 77.59
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['50.15625    ', '43.0        ', '39.8125     '], time:2.574849ms, TFLOPS: 106.75
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['50.15625    ', '43.0        ', '39.8125     '], time:2.598297ms, TFLOPS: 105.79
               f16wmma(mma2x4+warp2x4+stage3): ['50.15625    ', '43.0        ', '39.8125     '], time:2.545177ms, TFLOPS: 108.00
               f16wmma(mma2x4+warp2x4+stage2): ['50.15625    ', '43.0        ', '39.8125     '], time:2.558159ms, TFLOPS: 107.45
            f16wmma(warp2x4+...+stage3+dsmem): ['50.15625    ', '43.0        ', '39.8125     '], time:2.561187ms, TFLOPS: 107.32
            f16wmma(warp2x4+...+stage2+dsmem): ['50.15625    ', '43.0        ', '39.8125     '], time:2.560973ms, TFLOPS: 107.33
          f16wmma(warp2x4+...+stage3+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:2.562332ms, TFLOPS: 107.28
          f16wmma(warp2x4+...+stage2+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:2.563309ms, TFLOPS: 107.24
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:2.584850ms, TFLOPS: 106.34
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:2.564442ms, TFLOPS: 107.19
                                  f16(cublas): ['50.15625    ', '43.0        ', '39.8125     '], time:2.413868ms, TFLOPS: 113.87
                                       f16_th: ['50.1875     ', '43.09375    ', '39.78125    '], time:2.392888ms, TFLOPS: 114.87
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=512
                                   f16(naive): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:37.93582ms, TFLOPS: 3.62
                          f16x8pack(t8x8+bcf): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:2.760803ms, TFLOPS: 49.78
                     f16x8pack(t8x8+bcf+dbuf): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:2.843928ms, TFLOPS: 48.33
                     f16x8pack(t8x8+k16+dbuf): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:2.719151ms, TFLOPS: 50.54
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:6.035351ms, TFLOPS: 22.77
                              f16wmma(mma4x2): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:2.997374ms, TFLOPS: 45.85
                      f16wmma(mma4x2+warp2x4): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.910567ms, TFLOPS: 71.94
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.416456ms, TFLOPS: 97.03
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.464307ms, TFLOPS: 93.86
               f16wmma(mma2x4+warp2x4+stage3): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.407122ms, TFLOPS: 97.67
               f16wmma(mma2x4+warp2x4+stage2): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.434707ms, TFLOPS: 95.80
            f16wmma(warp2x4+...+stage3+dsmem): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.426398ms, TFLOPS: 96.35
            f16wmma(warp2x4+...+stage2+dsmem): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.432061ms, TFLOPS: 95.97
          f16wmma(warp2x4+...+stage3+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.428449ms, TFLOPS: 96.22
          f16wmma(warp2x4+...+stage2+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.407456ms, TFLOPS: 97.65
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.428699ms, TFLOPS: 96.20
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.407432ms, TFLOPS: 97.65
                                  f16(cublas): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.305568ms, TFLOPS: 105.27
                                       f16_th: ['5.44921875  ', '11.53125    ', '-11.5625    '], time:1.266074ms, TFLOPS: 108.56
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=1024
                                   f16(naive): ['21.78125    ', '17.46875    ', '-22.109375  '], time:75.43574ms, TFLOPS: 3.64
                          f16x8pack(t8x8+bcf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:5.563449ms, TFLOPS: 49.41
                     f16x8pack(t8x8+bcf+dbuf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:5.567681ms, TFLOPS: 49.37
                     f16x8pack(t8x8+k16+dbuf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:5.325198ms, TFLOPS: 51.62
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['21.859375   ', '17.53125    ', '-22.234375  '], time:11.74896ms, TFLOPS: 23.40
                              f16wmma(mma4x2): ['21.859375   ', '17.53125    ', '-22.234375  '], time:5.875051ms, TFLOPS: 46.79
                      f16wmma(mma4x2+warp2x4): ['21.859375   ', '17.53125    ', '-22.234375  '], time:3.553664ms, TFLOPS: 77.35
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.657103ms, TFLOPS: 103.45
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.719938ms, TFLOPS: 101.06
               f16wmma(mma2x4+warp2x4+stage3): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.645480ms, TFLOPS: 103.90
               f16wmma(mma2x4+warp2x4+stage2): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.654719ms, TFLOPS: 103.54
            f16wmma(warp2x4+...+stage3+dsmem): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.659046ms, TFLOPS: 103.37
            f16wmma(warp2x4+...+stage2+dsmem): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.660059ms, TFLOPS: 103.34
          f16wmma(warp2x4+...+stage3+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.660369ms, TFLOPS: 103.32
          f16wmma(warp2x4+...+stage2+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.640891ms, TFLOPS: 104.09
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.666389ms, TFLOPS: 103.09
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.640783ms, TFLOPS: 104.09
                                  f16(cublas): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.477729ms, TFLOPS: 110.94
                                       f16_th: ['21.875      ', '17.484375   ', '-22.25      '], time:2.460813ms, TFLOPS: 111.70
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=2048
                                   f16(naive): ['49.875      ', '43.28125    ', '39.90625    '], time:150.4811ms, TFLOPS: 3.65
                          f16x8pack(t8x8+bcf): ['49.875      ', '43.28125    ', '39.90625    '], time:11.02992ms, TFLOPS: 49.84
                     f16x8pack(t8x8+bcf+dbuf): ['49.875      ', '43.28125    ', '39.90625    '], time:11.02312ms, TFLOPS: 49.87
                     f16x8pack(t8x8+k16+dbuf): ['49.875      ', '43.28125    ', '39.90625    '], time:10.62492ms, TFLOPS: 51.74
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['50.15625    ', '43.0        ', '39.8125     '], time:23.20250ms, TFLOPS: 23.69
                              f16wmma(mma4x2): ['50.15625    ', '43.0        ', '39.8125     '], time:11.68549ms, TFLOPS: 47.05
                      f16wmma(mma4x2+warp2x4): ['50.15625    ', '43.0        ', '39.8125     '], time:6.879770ms, TFLOPS: 79.91
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['50.15625    ', '43.0        ', '39.8125     '], time:5.128955ms, TFLOPS: 107.19
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['50.15625    ', '43.0        ', '39.8125     '], time:5.182540ms, TFLOPS: 106.08
               f16wmma(mma2x4+warp2x4+stage3): ['50.15625    ', '43.0        ', '39.8125     '], time:5.105221ms, TFLOPS: 107.69
               f16wmma(mma2x4+warp2x4+stage2): ['50.15625    ', '43.0        ', '39.8125     '], time:5.134797ms, TFLOPS: 107.06
            f16wmma(warp2x4+...+stage3+dsmem): ['50.15625    ', '43.0        ', '39.8125     '], time:5.138874ms, TFLOPS: 106.98
            f16wmma(warp2x4+...+stage2+dsmem): ['50.15625    ', '43.0        ', '39.8125     '], time:5.151510ms, TFLOPS: 106.72
          f16wmma(warp2x4+...+stage3+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:5.283868ms, TFLOPS: 104.04
          f16wmma(warp2x4+...+stage2+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:5.210363ms, TFLOPS: 105.51
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:5.253946ms, TFLOPS: 104.64
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:5.099475ms, TFLOPS: 107.81
                                  f16(cublas): ['50.15625    ', '43.0        ', '39.8125     '], time:4.811024ms, TFLOPS: 114.27
                                       f16_th: ['50.1875     ', '43.09375    ', '39.78125    '], time:4.843711ms, TFLOPS: 113.50
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=512
                                   f16(naive): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:19.02226ms, TFLOPS: 3.61
                          f16x8pack(t8x8+bcf): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:1.360547ms, TFLOPS: 50.51
                     f16x8pack(t8x8+bcf+dbuf): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:1.403248ms, TFLOPS: 48.97
                     f16x8pack(t8x8+k16+dbuf): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:1.335883ms, TFLOPS: 51.44
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:3.119313ms, TFLOPS: 22.03
                              f16wmma(mma4x2): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.523625ms, TFLOPS: 45.10
                      f16wmma(mma4x2+warp2x4): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.986111ms, TFLOPS: 69.69
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.723576ms, TFLOPS: 94.97
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.730907ms, TFLOPS: 94.02
               f16wmma(mma2x4+warp2x4+stage3): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.705850ms, TFLOPS: 97.36
               f16wmma(mma2x4+warp2x4+stage2): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.727427ms, TFLOPS: 94.47
            f16wmma(warp2x4+...+stage3+dsmem): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.723218ms, TFLOPS: 95.02
            f16wmma(warp2x4+...+stage2+dsmem): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.729596ms, TFLOPS: 94.19
          f16wmma(warp2x4+...+stage3+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.727450ms, TFLOPS: 94.47
          f16wmma(warp2x4+...+stage2+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.716471ms, TFLOPS: 95.91
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.726950ms, TFLOPS: 94.53
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.716567ms, TFLOPS: 95.90
                                  f16(cublas): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:0.660777ms, TFLOPS: 104.00
                                       f16_th: ['5.44921875  ', '11.53125    ', '-11.5625    '], time:0.639891ms, TFLOPS: 107.39
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=1024
                                   f16(naive): ['21.78125    ', '17.46875    ', '-22.109375  '], time:37.77809ms, TFLOPS: 3.64
                          f16x8pack(t8x8+bcf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:2.740108ms, TFLOPS: 50.16
                     f16x8pack(t8x8+bcf+dbuf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:2.793872ms, TFLOPS: 49.19
                     f16x8pack(t8x8+k16+dbuf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:2.697050ms, TFLOPS: 50.96
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['21.859375   ', '17.53125    ', '-22.234375  '], time:5.947458ms, TFLOPS: 23.11
                              f16wmma(mma4x2): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.957451ms, TFLOPS: 46.47
                      f16wmma(mma4x2+warp2x4): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.838850ms, TFLOPS: 74.74
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.344573ms, TFLOPS: 102.22
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.370573ms, TFLOPS: 100.28
               f16wmma(mma2x4+warp2x4+stage3): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.329469ms, TFLOPS: 103.38
               f16wmma(mma2x4+warp2x4+stage2): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.349401ms, TFLOPS: 101.85
            f16wmma(warp2x4+...+stage3+dsmem): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.347506ms, TFLOPS: 102.00
            f16wmma(warp2x4+...+stage2+dsmem): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.349318ms, TFLOPS: 101.86
          f16wmma(warp2x4+...+stage3+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.341378ms, TFLOPS: 102.46
          f16wmma(warp2x4+...+stage2+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.327323ms, TFLOPS: 103.55
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.341474ms, TFLOPS: 102.45
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.327645ms, TFLOPS: 103.52
                                  f16(cublas): ['21.859375   ', '17.53125    ', '-22.234375  '], time:1.242041ms, TFLOPS: 110.66
                                       f16_th: ['21.875      ', '17.484375   ', '-22.25      '], time:1.221001ms, TFLOPS: 112.56
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=2048
                                   f16(naive): ['49.875      ', '43.28125    ', '39.90625    '], time:75.25591ms, TFLOPS: 3.65
                          f16x8pack(t8x8+bcf): ['49.875      ', '43.28125    ', '39.90625    '], time:5.538845ms, TFLOPS: 49.63
                     f16x8pack(t8x8+bcf+dbuf): ['49.875      ', '43.28125    ', '39.90625    '], time:5.563235ms, TFLOPS: 49.41
                     f16x8pack(t8x8+k16+dbuf): ['49.875      ', '43.28125    ', '39.90625    '], time:5.324435ms, TFLOPS: 51.63
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['50.15625    ', '43.0        ', '39.8125     '], time:11.68662ms, TFLOPS: 23.52
                              f16wmma(mma4x2): ['50.15625    ', '43.0        ', '39.8125     '], time:5.871593ms, TFLOPS: 46.81
                      f16wmma(mma4x2+warp2x4): ['50.15625    ', '43.0        ', '39.8125     '], time:3.539860ms, TFLOPS: 77.65
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['50.15625    ', '43.0        ', '39.8125     '], time:2.574515ms, TFLOPS: 106.77
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['50.15625    ', '43.0        ', '39.8125     '], time:2.612614ms, TFLOPS: 105.21
               f16wmma(mma2x4+warp2x4+stage3): ['50.15625    ', '43.0        ', '39.8125     '], time:2.564680ms, TFLOPS: 107.18
               f16wmma(mma2x4+warp2x4+stage2): ['50.15625    ', '43.0        ', '39.8125     '], time:2.584993ms, TFLOPS: 106.34
            f16wmma(warp2x4+...+stage3+dsmem): ['50.15625    ', '43.0        ', '39.8125     '], time:2.586770ms, TFLOPS: 106.26
            f16wmma(warp2x4+...+stage2+dsmem): ['50.15625    ', '43.0        ', '39.8125     '], time:2.575898ms, TFLOPS: 106.71
          f16wmma(warp2x4+...+stage3+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:2.568340ms, TFLOPS: 107.03
          f16wmma(warp2x4+...+stage2+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:2.560615ms, TFLOPS: 107.35
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:2.582669ms, TFLOPS: 106.43
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:2.560365ms, TFLOPS: 107.36
                                  f16(cublas): ['50.15625    ', '43.0        ', '39.8125     '], time:2.415192ms, TFLOPS: 113.81
                                       f16_th: ['50.1875     ', '43.09375    ', '39.78125    '], time:2.405774ms, TFLOPS: 114.26
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=512
                                   f16(naive): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:37.93667ms, TFLOPS: 3.62
                          f16x8pack(t8x8+bcf): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:2.781772ms, TFLOPS: 49.41
                     f16x8pack(t8x8+bcf+dbuf): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:2.859652ms, TFLOPS: 48.06
                     f16x8pack(t8x8+k16+dbuf): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:2.735054ms, TFLOPS: 50.25
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:6.113445ms, TFLOPS: 22.48
                              f16wmma(mma4x2): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:3.003191ms, TFLOPS: 45.76
                      f16wmma(mma4x2+warp2x4): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.910603ms, TFLOPS: 71.93
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.413738ms, TFLOPS: 97.22
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.477360ms, TFLOPS: 93.03
               f16wmma(mma2x4+warp2x4+stage3): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.422452ms, TFLOPS: 96.62
               f16wmma(mma2x4+warp2x4+stage2): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.444220ms, TFLOPS: 95.16
            f16wmma(warp2x4+...+stage3+dsmem): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.431345ms, TFLOPS: 96.02
            f16wmma(warp2x4+...+stage2+dsmem): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.441955ms, TFLOPS: 95.31
          f16wmma(warp2x4+...+stage3+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.438057ms, TFLOPS: 95.57
          f16wmma(warp2x4+...+stage2+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.430964ms, TFLOPS: 96.05
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.439440ms, TFLOPS: 95.48
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.416802ms, TFLOPS: 97.01
                                  f16(cublas): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:1.303493ms, TFLOPS: 105.44
                                       f16_th: ['5.44921875  ', '11.53125    ', '-11.5625    '], time:1.276326ms, TFLOPS: 107.68
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=1024
                                   f16(naive): ['21.78125    ', '17.46875    ', '-22.109375  '], time:75.42434ms, TFLOPS: 3.64
                          f16x8pack(t8x8+bcf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:5.572676ms, TFLOPS: 49.33
                     f16x8pack(t8x8+bcf+dbuf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:5.576443ms, TFLOPS: 49.29
                     f16x8pack(t8x8+k16+dbuf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:5.352568ms, TFLOPS: 51.35
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['21.859375   ', '17.53125    ', '-22.234375  '], time:11.80815ms, TFLOPS: 23.28
                              f16wmma(mma4x2): ['21.859375   ', '17.53125    ', '-22.234375  '], time:5.884182ms, TFLOPS: 46.71
                      f16wmma(mma4x2+warp2x4): ['21.859375   ', '17.53125    ', '-22.234375  '], time:3.569900ms, TFLOPS: 77.00
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.661347ms, TFLOPS: 103.29
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.733218ms, TFLOPS: 100.57
               f16wmma(mma2x4+warp2x4+stage3): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.664411ms, TFLOPS: 103.17
               f16wmma(mma2x4+warp2x4+stage2): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.673447ms, TFLOPS: 102.82
            f16wmma(warp2x4+...+stage3+dsmem): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.674305ms, TFLOPS: 102.78
            f16wmma(warp2x4+...+stage2+dsmem): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.666950ms, TFLOPS: 103.07
          f16wmma(warp2x4+...+stage3+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.675127ms, TFLOPS: 102.75
          f16wmma(warp2x4+...+stage2+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.658760ms, TFLOPS: 103.39
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.688860ms, TFLOPS: 102.23
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.674138ms, TFLOPS: 102.79
                                  f16(cublas): ['21.859375   ', '17.53125    ', '-22.234375  '], time:2.494108ms, TFLOPS: 110.21
                                       f16_th: ['21.875      ', '17.484375   ', '-22.25      '], time:2.472102ms, TFLOPS: 111.19
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=2048
                                   f16(naive): ['49.875      ', '43.28125    ', '39.90625    '], time:150.4682ms, TFLOPS: 3.65
                          f16x8pack(t8x8+bcf): ['49.875      ', '43.28125    ', '39.90625    '], time:11.04806ms, TFLOPS: 49.76
                     f16x8pack(t8x8+bcf+dbuf): ['49.875      ', '43.28125    ', '39.90625    '], time:11.02757ms, TFLOPS: 49.85
                     f16x8pack(t8x8+k16+dbuf): ['49.875      ', '43.28125    ', '39.90625    '], time:10.73905ms, TFLOPS: 51.19
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['50.15625    ', '43.0        ', '39.8125     '], time:23.23439ms, TFLOPS: 23.66
                              f16wmma(mma4x2): ['50.15625    ', '43.0        ', '39.8125     '], time:11.69081ms, TFLOPS: 47.02
                      f16wmma(mma4x2+warp2x4): ['50.15625    ', '43.0        ', '39.8125     '], time:6.921041ms, TFLOPS: 79.43
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['50.15625    ', '43.0        ', '39.8125     '], time:5.148756ms, TFLOPS: 106.77
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['50.15625    ', '43.0        ', '39.8125     '], time:5.213224ms, TFLOPS: 105.45
               f16wmma(mma2x4+warp2x4+stage3): ['50.15625    ', '43.0        ', '39.8125     '], time:5.113995ms, TFLOPS: 107.50
               f16wmma(mma2x4+warp2x4+stage2): ['50.15625    ', '43.0        ', '39.8125     '], time:5.153918ms, TFLOPS: 106.67
            f16wmma(warp2x4+...+stage3+dsmem): ['50.15625    ', '43.0        ', '39.8125     '], time:5.159771ms, TFLOPS: 106.55
            f16wmma(warp2x4+...+stage2+dsmem): ['50.15625    ', '43.0        ', '39.8125     '], time:5.153131ms, TFLOPS: 106.68
          f16wmma(warp2x4+...+stage3+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:5.258488ms, TFLOPS: 104.55
          f16wmma(warp2x4+...+stage2+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:5.244195ms, TFLOPS: 104.83
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:5.256378ms, TFLOPS: 104.59
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:5.119884ms, TFLOPS: 107.38
                                  f16(cublas): ['50.15625    ', '43.0        ', '39.8125     '], time:4.792737ms, TFLOPS: 114.71
                                       f16_th: ['50.1875     ', '43.09375    ', '39.78125    '], time:4.832541ms, TFLOPS: 113.76
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=512
                                   f16(naive): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:75.78649ms, TFLOPS: 3.63
                          f16x8pack(t8x8+bcf): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:5.487596ms, TFLOPS: 50.09
                     f16x8pack(t8x8+bcf+dbuf): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:5.707871ms, TFLOPS: 48.16
                     f16x8pack(t8x8+k16+dbuf): ['5.453125    ', '11.5625     ', '-11.5546875 '], time:5.461335ms, TFLOPS: 50.33
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:11.96372ms, TFLOPS: 22.98
                              f16wmma(mma4x2): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:5.971813ms, TFLOPS: 46.03
                      f16wmma(mma4x2+warp2x4): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:3.735268ms, TFLOPS: 73.59
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:2.849113ms, TFLOPS: 96.48
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:2.955138ms, TFLOPS: 93.02
               f16wmma(mma2x4+warp2x4+stage3): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:2.829158ms, TFLOPS: 97.16
               f16wmma(mma2x4+warp2x4+stage2): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:2.854382ms, TFLOPS: 96.30
            f16wmma(warp2x4+...+stage3+dsmem): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:2.846121ms, TFLOPS: 96.58
            f16wmma(warp2x4+...+stage2+dsmem): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:2.852582ms, TFLOPS: 96.36
          f16wmma(warp2x4+...+stage3+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:2.872502ms, TFLOPS: 95.69
          f16wmma(warp2x4+...+stage2+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:2.861547ms, TFLOPS: 96.06
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:2.983462ms, TFLOPS: 92.13
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:2.925503ms, TFLOPS: 93.96
                                  f16(cublas): ['5.43359375  ', '11.546875   ', '-11.5625    '], time:2.628958ms, TFLOPS: 104.56
                                       f16_th: ['5.44921875  ', '11.53125    ', '-11.5625    '], time:2.611851ms, TFLOPS: 105.24
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=1024
                                   f16(naive): ['21.78125    ', '17.46875    ', '-22.109375  '], time:150.8428ms, TFLOPS: 3.64
                          f16x8pack(t8x8+bcf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:11.12993ms, TFLOPS: 49.39
                     f16x8pack(t8x8+bcf+dbuf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:11.12456ms, TFLOPS: 49.42
                     f16x8pack(t8x8+k16+dbuf): ['21.78125    ', '17.46875    ', '-22.109375  '], time:10.87856ms, TFLOPS: 50.54
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['21.859375   ', '17.53125    ', '-22.234375  '], time:23.37963ms, TFLOPS: 23.51
                              f16wmma(mma4x2): ['21.859375   ', '17.53125    ', '-22.234375  '], time:11.72295ms, TFLOPS: 46.90
                      f16wmma(mma4x2+warp2x4): ['21.859375   ', '17.53125    ', '-22.234375  '], time:7.029008ms, TFLOPS: 78.21
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['21.859375   ', '17.53125    ', '-22.234375  '], time:5.364477ms, TFLOPS: 102.48
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['21.859375   ', '17.53125    ', '-22.234375  '], time:5.427086ms, TFLOPS: 101.30
               f16wmma(mma2x4+warp2x4+stage3): ['21.859375   ', '17.53125    ', '-22.234375  '], time:5.313682ms, TFLOPS: 103.46
               f16wmma(mma2x4+warp2x4+stage2): ['21.859375   ', '17.53125    ', '-22.234375  '], time:5.323040ms, TFLOPS: 103.28
            f16wmma(warp2x4+...+stage3+dsmem): ['21.859375   ', '17.53125    ', '-22.234375  '], time:5.318355ms, TFLOPS: 103.37
            f16wmma(warp2x4+...+stage2+dsmem): ['21.859375   ', '17.53125    ', '-22.234375  '], time:5.380594ms, TFLOPS: 102.17
          f16wmma(warp2x4+...+stage3+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:5.495917ms, TFLOPS: 100.03
          f16wmma(warp2x4+...+stage2+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:5.439901ms, TFLOPS: 101.06
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:5.464541ms, TFLOPS: 100.60
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['21.859375   ', '17.53125    ', '-22.234375  '], time:5.304145ms, TFLOPS: 103.65
                                  f16(cublas): ['21.859375   ', '17.53125    ', '-22.234375  '], time:4.905176ms, TFLOPS: 112.08
                                       f16_th: ['21.875      ', '17.484375   ', '-22.25      '], time:4.924702ms, TFLOPS: 111.63
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=2048
                                   f16(naive): ['49.875      ', '43.28125    ', '39.90625    '], time:300.8541ms, TFLOPS: 3.65
                          f16x8pack(t8x8+bcf): ['49.875      ', '43.28125    ', '39.90625    '], time:22.02371ms, TFLOPS: 49.92
                     f16x8pack(t8x8+bcf+dbuf): ['49.875      ', '43.28125    ', '39.90625    '], time:22.37513ms, TFLOPS: 49.14
                     f16x8pack(t8x8+k16+dbuf): ['49.875      ', '43.28125    ', '39.90625    '], time:21.08974ms, TFLOPS: 52.13
--------------------------------------------------------------------WMMA----------------------------------------------------------
                               f16wmma(naive): ['50.15625    ', '43.0        ', '39.8125     '], time:46.26761ms, TFLOPS: 23.76
                              f16wmma(mma4x2): ['50.15625    ', '43.0        ', '39.8125     '], time:23.35487ms, TFLOPS: 47.08
                      f16wmma(mma4x2+warp2x4): ['50.15625    ', '43.0        ', '39.8125     '], time:13.64753ms, TFLOPS: 80.56
       f16wmma(m16n16k16+mma2x4+warp2x4+dbuf): ['50.15625    ', '43.0        ', '39.8125     '], time:10.33419ms, TFLOPS: 106.40
        f16wmma(m32n8k16+mma2x4+warp2x4+dbuf): ['50.15625    ', '43.0        ', '39.8125     '], time:10.42598ms, TFLOPS: 105.46
               f16wmma(mma2x4+warp2x4+stage3): ['50.15625    ', '43.0        ', '39.8125     '], time:10.25907ms, TFLOPS: 107.17
               f16wmma(mma2x4+warp2x4+stage2): ['50.15625    ', '43.0        ', '39.8125     '], time:10.42851ms, TFLOPS: 105.43
            f16wmma(warp2x4+...+stage3+dsmem): ['50.15625    ', '43.0        ', '39.8125     '], time:10.39245ms, TFLOPS: 105.80
            f16wmma(warp2x4+...+stage2+dsmem): ['50.15625    ', '43.0        ', '39.8125     '], time:10.23268ms, TFLOPS: 107.45
          f16wmma(warp2x4+...+stage3+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:10.41290ms, TFLOPS: 105.59
          f16wmma(warp2x4+...+stage2+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:10.36430ms, TFLOPS: 106.09
    f16wmma(warp2x4+...+stage3+dsmem+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:10.36210ms, TFLOPS: 106.11
    f16wmma(warp2x4+...+stage2+dsmem+swizzle): ['50.15625    ', '43.0        ', '39.8125     '], time:10.15591ms, TFLOPS: 108.26
                                  f16(cublas): ['50.15625    ', '43.0        ', '39.8125     '], time:9.630787ms, TFLOPS: 114.17
                                       f16_th: ['50.1875     ', '43.09375    ', '39.78125    '], time:9.693896ms, TFLOPS: 113.42
----------------------------------------------------------------------------------------------------------------------------------
```
