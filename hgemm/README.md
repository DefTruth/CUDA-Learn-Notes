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

目前最优的实现，在L20上，能达到cuBLAS大概93%~95%左右的性能(TFLOPS)，已知问题为bank conflicts没有完全消除，目前通过padding的方式缓解bank conflicts会导致shared memory浪费，也会影响SM occupancy。并且尚未手工实现Warp swizzle(受限于WMMA API的灵活性以及本人的能力)，后续将会尝试通过MMA PTX实现warp swizzle。

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
                                                       M=4096, N=4096, K=2048
                         f16(naive): ['30.0625   ', '-9.34375  '], time:18.87199ms, swizzle: NOOP, TFLOPS: 3.64  (+0.00%)
                f16x8pack(t8x8+bcf): ['30.0625   ', '-9.34375  '], time:1.412582ms, swizzle: NOOP, TFLOPS: 48.65 (+1235.99%)
           f16x8pack(t8x8+bcf+dbuf): ['30.0625   ', '-9.34375  '], time:1.398515ms, swizzle: NOOP, TFLOPS: 49.14 (+1.01%)
           f16x8pack(t8x8+k16+dbuf): ['30.0625   ', '-9.34375  '], time:1.342988ms, swizzle: NOOP, TFLOPS: 51.17 (+4.13%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['29.921875 ', '-9.5078125'], time:2.957487ms, swizzle: NOOP, TFLOPS: 23.24
                    f16wmma(mma4x2): ['29.921875 ', '-9.5078125'], time:1.494514ms, swizzle: NOOP, TFLOPS: 45.98
            f16wmma(mma4x2+warp2x4): ['29.921875 ', '-9.5078125'], time:0.936377ms, swizzle: NOOP, TFLOPS: 73.39 (+43.42%)
       f16wmma(mma2x4+warp2x4+dbuf): ['29.921875 ', '-9.5078125'], time:0.695562ms, swizzle: NOOP, TFLOPS: 98.80 (+34.62%)
     f16wmma(mma2x4+warp2x4+stage3): ['29.921875 ', '-9.5078125'], time:0.687670ms, swizzle: NOOP, TFLOPS: 99.93 (+1.15%)
     f16wmma(mma2x4+warp2x4+stage2): ['29.921875 ', '-9.5078125'], time:0.690507ms, swizzle: NOOP, TFLOPS: 99.52
   f16wmma(mma2x4+...+stage3+dsmem): ['29.921875 ', '-9.5078125'], time:0.692045ms, swizzle: NOOP, TFLOPS: 99.30
   f16wmma(mma2x4+...+stage2+dsmem): ['29.921875 ', '-9.5078125'], time:0.689327ms, swizzle: NOOP, TFLOPS: 99.69
 f16wmma(mma2x4+...+stage3+swizzle): ['29.921875 ', '-9.5078125'], time:0.686764ms, swizzle: 1024, TFLOPS: 100.06(+0.13%)
 f16wmma(mma2x4+...+stage2+swizzle): ['29.921875 ', '-9.5078125'], time:0.679647ms, swizzle: 1024, TFLOPS: 101.11(+1.05%)
  f16wmma(...+stage3+dsmem+swizzle): ['29.921875 ', '-9.5078125'], time:0.686705ms, swizzle: 1024, TFLOPS: 100.07
  f16wmma(...+stage2+dsmem+swizzle): ['29.921875 ', '-9.5078125'], time:0.678586ms, swizzle: 1024, TFLOPS: 101.27(+0.16%)
                        f16(cublas): ['29.921875 ', '-9.5078125'], time:0.660037ms, swizzle: NOOP, TFLOPS: 104.11(+2.81%)
                             f16_th: ['29.9375   ', '-9.5703125'], time:0.645995ms, swizzle: NOOP, TFLOPS: 106.38(+2.17%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=4096
                         f16(naive): ['18.078125 ', '32.90625  '], time:37.67147ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                f16x8pack(t8x8+bcf): ['18.078125 ', '32.90625  '], time:2.861094ms, swizzle: NOOP, TFLOPS: 48.04 (+1216.68%)
           f16x8pack(t8x8+bcf+dbuf): ['18.078125 ', '32.90625  '], time:2.854299ms, swizzle: NOOP, TFLOPS: 48.15 (+0.24%)
           f16x8pack(t8x8+k16+dbuf): ['18.078125 ', '32.90625  '], time:2.722513ms, swizzle: NOOP, TFLOPS: 50.48 (+4.84%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['17.953125 ', '32.8125   '], time:5.859076ms, swizzle: NOOP, TFLOPS: 23.46
                    f16wmma(mma4x2): ['17.953125 ', '32.8125   '], time:2.957177ms, swizzle: NOOP, TFLOPS: 46.48
            f16wmma(mma4x2+warp2x4): ['17.953125 ', '32.8125   '], time:1.826405ms, swizzle: NOOP, TFLOPS: 75.25 (+49.06%)
       f16wmma(mma2x4+warp2x4+dbuf): ['17.953125 ', '32.8125   '], time:1.359379ms, swizzle: NOOP, TFLOPS: 101.10(+34.36%)
     f16wmma(mma2x4+warp2x4+stage3): ['17.953125 ', '32.8125   '], time:1.346862ms, swizzle: NOOP, TFLOPS: 102.04(+0.93%)
     f16wmma(mma2x4+warp2x4+stage2): ['17.953125 ', '32.8125   '], time:1.348721ms, swizzle: NOOP, TFLOPS: 101.90
   f16wmma(mma2x4+...+stage3+dsmem): ['17.953125 ', '32.8125   '], time:1.348102ms, swizzle: NOOP, TFLOPS: 101.95
   f16wmma(mma2x4+...+stage2+dsmem): ['17.953125 ', '32.8125   '], time:1.341021ms, swizzle: NOOP, TFLOPS: 102.49(+0.44%)
 f16wmma(mma2x4+...+stage3+swizzle): ['17.953125 ', '32.8125   '], time:1.335406ms, swizzle: 1024, TFLOPS: 102.92(+0.42%)
 f16wmma(mma2x4+...+stage2+swizzle): ['17.953125 ', '32.8125   '], time:1.322257ms, swizzle: 1024, TFLOPS: 103.94(+0.99%)
  f16wmma(...+stage3+dsmem+swizzle): ['17.953125 ', '32.8125   '], time:1.335537ms, swizzle: 1024, TFLOPS: 102.91
  f16wmma(...+stage2+dsmem+swizzle): ['17.953125 ', '32.8125   '], time:1.322436ms, swizzle: 1024, TFLOPS: 103.93
                        f16(cublas): ['17.953125 ', '32.8125   '], time:1.282560ms, swizzle: NOOP, TFLOPS: 107.16(+3.10%)
                             f16_th: ['17.96875  ', '32.75     '], time:1.272284ms, swizzle: NOOP, TFLOPS: 108.03(+0.81%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=8192
                         f16(naive): ['69.5      ', '17.9375   '], time:75.29305ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                f16x8pack(t8x8+bcf): ['69.5      ', '17.9375   '], time:5.882036ms, swizzle: NOOP, TFLOPS: 46.73 (+1180.05%)
           f16x8pack(t8x8+bcf+dbuf): ['69.5      ', '17.9375   '], time:5.787312ms, swizzle: NOOP, TFLOPS: 47.50 (+1.64%)
           f16x8pack(t8x8+k16+dbuf): ['69.5      ', '17.9375   '], time:5.527842ms, swizzle: NOOP, TFLOPS: 49.73 (+4.69%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['69.25     ', '18.078125 '], time:11.61112ms, swizzle: NOOP, TFLOPS: 23.67
                    f16wmma(mma4x2): ['69.25     ', '18.078125 '], time:5.882251ms, swizzle: NOOP, TFLOPS: 46.73
            f16wmma(mma4x2+warp2x4): ['69.25     ', '18.078125 '], time:3.625464ms, swizzle: NOOP, TFLOPS: 75.82 (+52.47%)
       f16wmma(mma2x4+warp2x4+dbuf): ['69.25     ', '18.078125 '], time:2.672624ms, swizzle: NOOP, TFLOPS: 102.85(+35.65%)
     f16wmma(mma2x4+warp2x4+stage3): ['69.25     ', '18.078125 '], time:2.649033ms, swizzle: NOOP, TFLOPS: 103.77(+0.89%)
     f16wmma(mma2x4+warp2x4+stage2): ['69.25     ', '18.078125 '], time:2.655041ms, swizzle: NOOP, TFLOPS: 103.53
   f16wmma(mma2x4+...+stage3+dsmem): ['69.25     ', '18.078125 '], time:2.666223ms, swizzle: NOOP, TFLOPS: 103.10
   f16wmma(mma2x4+...+stage2+dsmem): ['69.25     ', '18.078125 '], time:2.652776ms, swizzle: NOOP, TFLOPS: 103.62
 f16wmma(mma2x4+...+stage3+swizzle): ['69.25     ', '18.078125 '], time:2.658081ms, swizzle: 1024, TFLOPS: 103.41
 f16wmma(mma2x4+...+stage2+swizzle): ['69.25     ', '18.078125 '], time:2.636897ms, swizzle: 1024, TFLOPS: 104.24(+0.46%)
  f16wmma(...+stage3+dsmem+swizzle): ['69.25     ', '18.078125 '], time:2.662909ms, swizzle: 1024, TFLOPS: 103.22
  f16wmma(...+stage2+dsmem+swizzle): ['69.25     ', '18.078125 '], time:2.649581ms, swizzle: 1024, TFLOPS: 103.74
                        f16(cublas): ['69.3125   ', '18.0625   '], time:2.444374ms, swizzle: NOOP, TFLOPS: 112.45(+7.88%)
                             f16_th: ['69.3125   ', '18.109375 '], time:2.403986ms, swizzle: NOOP, TFLOPS: 114.34(+1.68%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=2048
                         f16(naive): ['30.0625   ', '-9.34375  '], time:37.68898ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                f16x8pack(t8x8+bcf): ['30.0625   ', '-9.34375  '], time:2.739918ms, swizzle: NOOP, TFLOPS: 50.16 (+1275.55%)
           f16x8pack(t8x8+bcf+dbuf): ['30.0625   ', '-9.34375  '], time:2.786326ms, swizzle: NOOP, TFLOPS: 49.33
           f16x8pack(t8x8+k16+dbuf): ['30.0625   ', '-9.34375  '], time:2.674496ms, swizzle: NOOP, TFLOPS: 51.39 (+2.45%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['29.921875 ', '-9.5078125'], time:5.887448ms, swizzle: NOOP, TFLOPS: 23.34
                    f16wmma(mma4x2): ['29.921875 ', '-9.5078125'], time:2.955126ms, swizzle: NOOP, TFLOPS: 46.51
            f16wmma(mma4x2+warp2x4): ['29.921875 ', '-9.5078125'], time:1.840353ms, swizzle: NOOP, TFLOPS: 74.68 (+45.33%)
       f16wmma(mma2x4+warp2x4+dbuf): ['29.921875 ', '-9.5078125'], time:1.328313ms, swizzle: NOOP, TFLOPS: 103.47(+38.55%)
     f16wmma(mma2x4+warp2x4+stage3): ['29.921875 ', '-9.5078125'], time:1.306402ms, swizzle: NOOP, TFLOPS: 105.20(+1.68%)
     f16wmma(mma2x4+warp2x4+stage2): ['29.921875 ', '-9.5078125'], time:1.311874ms, swizzle: NOOP, TFLOPS: 104.77
   f16wmma(mma2x4+...+stage3+dsmem): ['29.921875 ', '-9.5078125'], time:1.308524ms, swizzle: NOOP, TFLOPS: 105.03
   f16wmma(mma2x4+...+stage2+dsmem): ['29.921875 ', '-9.5078125'], time:1.304471ms, swizzle: NOOP, TFLOPS: 105.36(+0.15%)
 f16wmma(mma2x4+...+stage3+swizzle): ['29.921875 ', '-9.5078125'], time:1.298165ms, swizzle: 2048, TFLOPS: 105.87(+0.49%)
 f16wmma(mma2x4+...+stage2+swizzle): ['29.921875 ', '-9.5078125'], time:1.286089ms, swizzle: 2048, TFLOPS: 106.87(+0.94%)
  f16wmma(...+stage3+dsmem+swizzle): ['29.921875 ', '-9.5078125'], time:1.297760ms, swizzle: 2048, TFLOPS: 105.90
  f16wmma(...+stage2+dsmem+swizzle): ['29.921875 ', '-9.5078125'], time:1.293647ms, swizzle: 2048, TFLOPS: 106.24
                        f16(cublas): ['29.921875 ', '-9.5078125'], time:1.235735ms, swizzle: NOOP, TFLOPS: 111.22(+4.07%)
                             f16_th: ['29.9375   ', '-9.5703125'], time:1.295924ms, swizzle: NOOP, TFLOPS: 106.05
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=4096
                         f16(naive): ['18.078125 ', '32.90625  '], time:75.22808ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                f16x8pack(t8x8+bcf): ['18.078125 ', '32.90625  '], time:5.625784ms, swizzle: NOOP, TFLOPS: 48.86 (+1237.20%)
           f16x8pack(t8x8+bcf+dbuf): ['18.078125 ', '32.90625  '], time:5.564570ms, swizzle: NOOP, TFLOPS: 49.40 (+1.10%)
           f16x8pack(t8x8+k16+dbuf): ['18.078125 ', '32.90625  '], time:5.350506ms, swizzle: NOOP, TFLOPS: 51.37 (+4.00%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['17.953125 ', '32.8125   '], time:11.64093ms, swizzle: NOOP, TFLOPS: 23.61
                    f16wmma(mma4x2): ['17.953125 ', '32.8125   '], time:5.853724ms, swizzle: NOOP, TFLOPS: 46.96
            f16wmma(mma4x2+warp2x4): ['17.953125 ', '32.8125   '], time:3.614306ms, swizzle: NOOP, TFLOPS: 76.05 (+48.04%)
       f16wmma(mma2x4+warp2x4+dbuf): ['17.953125 ', '32.8125   '], time:2.582406ms, swizzle: NOOP, TFLOPS: 106.44(+39.96%)
     f16wmma(mma2x4+warp2x4+stage3): ['17.953125 ', '32.8125   '], time:2.549886ms, swizzle: NOOP, TFLOPS: 107.80(+1.28%)
     f16wmma(mma2x4+warp2x4+stage2): ['17.953125 ', '32.8125   '], time:2.571272ms, swizzle: NOOP, TFLOPS: 106.90
   f16wmma(mma2x4+...+stage3+dsmem): ['17.953125 ', '32.8125   '], time:2.576792ms, swizzle: NOOP, TFLOPS: 106.67
   f16wmma(mma2x4+...+stage2+dsmem): ['17.953125 ', '32.8125   '], time:2.569878ms, swizzle: NOOP, TFLOPS: 106.96
 f16wmma(mma2x4+...+stage3+swizzle): ['17.953125 ', '32.8125   '], time:2.562510ms, swizzle: 2048, TFLOPS: 107.27
 f16wmma(mma2x4+...+stage2+swizzle): ['17.953125 ', '32.8125   '], time:2.551829ms, swizzle: 2048, TFLOPS: 107.72
  f16wmma(...+stage3+dsmem+swizzle): ['17.953125 ', '32.8125   '], time:2.570784ms, swizzle: 2048, TFLOPS: 106.92
  f16wmma(...+stage2+dsmem+swizzle): ['17.953125 ', '32.8125   '], time:2.554595ms, swizzle: 2048, TFLOPS: 107.60
                        f16(cublas): ['17.953125 ', '32.8125   '], time:2.444684ms, swizzle: NOOP, TFLOPS: 112.44(+4.30%)
                             f16_th: ['17.96875  ', '32.75     '], time:2.553021ms, swizzle: NOOP, TFLOPS: 107.67
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=8192
                         f16(naive): ['69.5      ', '17.9375   '], time:221.6357ms, swizzle: NOOP, TFLOPS: 2.48  (+0.00%)
                f16x8pack(t8x8+bcf): ['69.5      ', '17.9375   '], time:12.29833ms, swizzle: NOOP, TFLOPS: 44.70 (+1702.16%)
           f16x8pack(t8x8+bcf+dbuf): ['69.5      ', '17.9375   '], time:12.03058ms, swizzle: NOOP, TFLOPS: 45.70 (+2.23%)
           f16x8pack(t8x8+k16+dbuf): ['69.5      ', '17.9375   '], time:12.48524ms, swizzle: NOOP, TFLOPS: 44.03
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['69.25     ', '18.078125 '], time:24.38898ms, swizzle: NOOP, TFLOPS: 22.54
                    f16wmma(mma4x2): ['69.25     ', '18.078125 '], time:12.68216ms, swizzle: NOOP, TFLOPS: 43.35
            f16wmma(mma4x2+warp2x4): ['69.25     ', '18.078125 '], time:7.341659ms, swizzle: NOOP, TFLOPS: 74.88 (+63.87%)
       f16wmma(mma2x4+warp2x4+dbuf): ['69.25     ', '18.078125 '], time:5.549454ms, swizzle: NOOP, TFLOPS: 99.06 (+32.30%)
     f16wmma(mma2x4+warp2x4+stage3): ['69.25     ', '18.078125 '], time:5.570328ms, swizzle: NOOP, TFLOPS: 98.69
     f16wmma(mma2x4+warp2x4+stage2): ['69.25     ', '18.078125 '], time:5.587530ms, swizzle: NOOP, TFLOPS: 98.39
   f16wmma(mma2x4+...+stage3+dsmem): ['69.25     ', '18.078125 '], time:5.602407ms, swizzle: NOOP, TFLOPS: 98.13
   f16wmma(mma2x4+...+stage2+dsmem): ['69.25     ', '18.078125 '], time:5.590212ms, swizzle: NOOP, TFLOPS: 98.34
 f16wmma(mma2x4+...+stage3+swizzle): ['69.25     ', '18.078125 '], time:5.302071ms, swizzle: 2048, TFLOPS: 103.69(+4.67%)
 f16wmma(mma2x4+...+stage2+swizzle): ['69.25     ', '18.078125 '], time:5.218410ms, swizzle: 2048, TFLOPS: 105.35(+1.60%)
  f16wmma(...+stage3+dsmem+swizzle): ['69.25     ', '18.078125 '], time:5.235731ms, swizzle: 2048, TFLOPS: 105.00
  f16wmma(...+stage2+dsmem+swizzle): ['69.25     ', '18.078125 '], time:5.117344ms, swizzle: 2048, TFLOPS: 107.43(+1.97%)
                        f16(cublas): ['69.25     ', '18.078125 '], time:4.885363ms, swizzle: NOOP, TFLOPS: 112.53(+4.75%)
                             f16_th: ['69.3125   ', '18.09375  '], time:4.924082ms, swizzle: NOOP, TFLOPS: 111.65
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=2048
                         f16(naive): ['30.0625   ', '-9.34375  '], time:75.27105ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                f16x8pack(t8x8+bcf): ['30.0625   ', '-9.34375  '], time:5.562174ms, swizzle: NOOP, TFLOPS: 49.42 (+1253.27%)
           f16x8pack(t8x8+bcf+dbuf): ['30.0625   ', '-9.34375  '], time:5.530333ms, swizzle: NOOP, TFLOPS: 49.70 (+0.58%)
           f16x8pack(t8x8+k16+dbuf): ['30.0625   ', '-9.34375  '], time:5.303525ms, swizzle: NOOP, TFLOPS: 51.83 (+4.28%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['29.921875 ', '-9.5078125'], time:11.68239ms, swizzle: NOOP, TFLOPS: 23.53
                    f16wmma(mma4x2): ['29.921875 ', '-9.5078125'], time:5.863881ms, swizzle: NOOP, TFLOPS: 46.88
            f16wmma(mma4x2+warp2x4): ['29.921875 ', '-9.5078125'], time:3.521573ms, swizzle: NOOP, TFLOPS: 78.06 (+50.60%)
       f16wmma(mma2x4+warp2x4+dbuf): ['29.921875 ', '-9.5078125'], time:2.580308ms, swizzle: NOOP, TFLOPS: 106.53(+36.48%)
     f16wmma(mma2x4+warp2x4+stage3): ['29.921875 ', '-9.5078125'], time:2.565371ms, swizzle: NOOP, TFLOPS: 107.15(+0.58%)
     f16wmma(mma2x4+warp2x4+stage2): ['29.921875 ', '-9.5078125'], time:2.597820ms, swizzle: NOOP, TFLOPS: 105.81
   f16wmma(mma2x4+...+stage3+dsmem): ['29.921875 ', '-9.5078125'], time:2.602386ms, swizzle: NOOP, TFLOPS: 105.63
   f16wmma(mma2x4+...+stage2+dsmem): ['29.921875 ', '-9.5078125'], time:2.596187ms, swizzle: NOOP, TFLOPS: 105.88
 f16wmma(mma2x4+...+stage3+swizzle): ['29.921875 ', '-9.5078125'], time:2.584767ms, swizzle: 4096, TFLOPS: 106.35
 f16wmma(mma2x4+...+stage2+swizzle): ['29.921875 ', '-9.5078125'], time:2.563869ms, swizzle: 4096, TFLOPS: 107.21(+0.06%)
  f16wmma(...+stage3+dsmem+swizzle): ['29.921875 ', '-9.5078125'], time:2.592587ms, swizzle: 4096, TFLOPS: 106.02
  f16wmma(...+stage2+dsmem+swizzle): ['29.921875 ', '-9.5078125'], time:2.574265ms, swizzle: 4096, TFLOPS: 106.78
                        f16(cublas): ['29.921875 ', '-9.5078125'], time:2.438402ms, swizzle: NOOP, TFLOPS: 112.73(+5.15%)
                             f16_th: ['29.9375   ', '-9.5703125'], time:2.436280ms, swizzle: NOOP, TFLOPS: 112.83(+0.09%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=4096
                         f16(naive): ['18.078125 ', '32.90625  '], time:229.3893ms, swizzle: NOOP, TFLOPS: 2.40  (+0.00%)
                f16x8pack(t8x8+bcf): ['18.078125 ', '32.90625  '], time:12.10900ms, swizzle: NOOP, TFLOPS: 45.40 (+1794.37%)
           f16x8pack(t8x8+bcf+dbuf): ['18.078125 ', '32.90625  '], time:12.12658ms, swizzle: NOOP, TFLOPS: 45.33
           f16x8pack(t8x8+k16+dbuf): ['18.078125 ', '32.90625  '], time:12.65550ms, swizzle: NOOP, TFLOPS: 43.44
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['17.953125 ', '32.8125   '], time:37.73040ms, swizzle: NOOP, TFLOPS: 14.57
                    f16wmma(mma4x2): ['17.953125 ', '32.8125   '], time:15.56271ms, swizzle: NOOP, TFLOPS: 35.33
            f16wmma(mma4x2+warp2x4): ['17.953125 ', '32.8125   '], time:7.286703ms, swizzle: NOOP, TFLOPS: 75.45 (+66.18%)
       f16wmma(mma2x4+warp2x4+dbuf): ['17.953125 ', '32.8125   '], time:5.685663ms, swizzle: NOOP, TFLOPS: 96.69 (+28.16%)
     f16wmma(mma2x4+warp2x4+stage3): ['17.953125 ', '32.8125   '], time:6.107950ms, swizzle: NOOP, TFLOPS: 90.01
     f16wmma(mma2x4+warp2x4+stage2): ['17.953125 ', '32.8125   '], time:5.713760ms, swizzle: NOOP, TFLOPS: 96.22
   f16wmma(mma2x4+...+stage3+dsmem): ['17.953125 ', '32.8125   '], time:5.680024ms, swizzle: NOOP, TFLOPS: 96.79 (+0.10%)
   f16wmma(mma2x4+...+stage2+dsmem): ['17.953125 ', '32.8125   '], time:5.583977ms, swizzle: NOOP, TFLOPS: 98.45 (+1.72%)
 f16wmma(mma2x4+...+stage3+swizzle): ['17.953125 ', '32.8125   '], time:5.171418ms, swizzle: 4096, TFLOPS: 106.31(+7.98%)
 f16wmma(mma2x4+...+stage2+swizzle): ['17.953125 ', '32.8125   '], time:5.069971ms, swizzle: 4096, TFLOPS: 108.43(+2.00%)
  f16wmma(...+stage3+dsmem+swizzle): ['17.953125 ', '32.8125   '], time:5.108344ms, swizzle: 4096, TFLOPS: 107.62
  f16wmma(...+stage2+dsmem+swizzle): ['17.953125 ', '32.8125   '], time:5.106234ms, swizzle: 4096, TFLOPS: 107.66
                        f16(cublas): ['17.953125 ', '32.8125   '], time:4.881918ms, swizzle: NOOP, TFLOPS: 112.61(+3.85%)
                             f16_th: ['17.96875  ', '32.75     '], time:4.905498ms, swizzle: NOOP, TFLOPS: 112.07
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=8192
                         f16(naive): ['69.5      ', '17.9375   '], time:458.1406ms, swizzle: NOOP, TFLOPS: 2.40  (+0.00%)
                f16x8pack(t8x8+bcf): ['69.5      ', '17.9375   '], time:25.22308ms, swizzle: NOOP, TFLOPS: 43.59 (+1716.35%)
           f16x8pack(t8x8+bcf+dbuf): ['69.5      ', '17.9375   '], time:27.05860ms, swizzle: NOOP, TFLOPS: 40.63
           f16x8pack(t8x8+k16+dbuf): ['69.5      ', '17.9375   '], time:25.67851ms, swizzle: NOOP, TFLOPS: 42.82
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['69.25     ', '18.078125 '], time:198.8110ms, swizzle: NOOP, TFLOPS: 5.53
                    f16wmma(mma4x2): ['69.25     ', '18.078125 '], time:30.96374ms, swizzle: NOOP, TFLOPS: 35.51
            f16wmma(mma4x2+warp2x4): ['69.25     ', '18.078125 '], time:14.41694ms, swizzle: NOOP, TFLOPS: 76.27 (+74.95%)
       f16wmma(mma2x4+warp2x4+dbuf): ['69.25     ', '18.078125 '], time:12.65356ms, swizzle: NOOP, TFLOPS: 86.89 (+13.94%)
     f16wmma(mma2x4+warp2x4+stage3): ['69.25     ', '18.078125 '], time:12.48985ms, swizzle: NOOP, TFLOPS: 88.03 (+1.31%)
     f16wmma(mma2x4+warp2x4+stage2): ['69.25     ', '18.078125 '], time:12.57493ms, swizzle: NOOP, TFLOPS: 87.44
   f16wmma(mma2x4+...+stage3+dsmem): ['69.25     ', '18.078125 '], time:12.49730ms, swizzle: NOOP, TFLOPS: 87.98
   f16wmma(mma2x4+...+stage2+dsmem): ['69.25     ', '18.078125 '], time:12.41034ms, swizzle: NOOP, TFLOPS: 88.60 (+0.64%)
 f16wmma(mma2x4+...+stage3+swizzle): ['69.25     ', '18.078125 '], time:10.31826ms, swizzle: 4096, TFLOPS: 106.56(+20.28%)
 f16wmma(mma2x4+...+stage2+swizzle): ['69.25     ', '18.078125 '], time:10.17751ms, swizzle: 4096, TFLOPS: 108.03(+1.38%)
  f16wmma(...+stage3+dsmem+swizzle): ['69.25     ', '18.078125 '], time:10.34779ms, swizzle: 4096, TFLOPS: 106.26
  f16wmma(...+stage2+dsmem+swizzle): ['69.25     ', '18.078125 '], time:10.15803ms, swizzle: 4096, TFLOPS: 108.24(+0.19%)
                        f16(cublas): ['69.25     ', '18.078125 '], time:9.691882ms, swizzle: NOOP, TFLOPS: 113.45(+4.81%)
                             f16_th: ['69.3125   ', '18.09375  '], time:9.703350ms, swizzle: NOOP, TFLOPS: 113.31
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=2048
                         f16(naive): ['30.0625   ', '-9.34375  '], time:37.67192ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                f16x8pack(t8x8+bcf): ['30.0625   ', '-9.34375  '], time:2.825665ms, swizzle: NOOP, TFLOPS: 48.64 (+1233.21%)
           f16x8pack(t8x8+bcf+dbuf): ['30.0625   ', '-9.34375  '], time:2.841365ms, swizzle: NOOP, TFLOPS: 48.37
           f16x8pack(t8x8+k16+dbuf): ['30.0625   ', '-9.34375  '], time:2.718400ms, swizzle: NOOP, TFLOPS: 50.56 (+3.95%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['29.921875 ', '-9.5078125'], time:5.906438ms, swizzle: NOOP, TFLOPS: 23.27
                    f16wmma(mma4x2): ['29.921875 ', '-9.5078125'], time:2.955245ms, swizzle: NOOP, TFLOPS: 46.51
            f16wmma(mma4x2+warp2x4): ['29.921875 ', '-9.5078125'], time:1.827824ms, swizzle: NOOP, TFLOPS: 75.19 (+48.72%)
       f16wmma(mma2x4+warp2x4+dbuf): ['29.921875 ', '-9.5078125'], time:1.327943ms, swizzle: NOOP, TFLOPS: 103.50(+37.64%)
     f16wmma(mma2x4+warp2x4+stage3): ['29.921875 ', '-9.5078125'], time:1.307272ms, swizzle: NOOP, TFLOPS: 105.13(+1.58%)
     f16wmma(mma2x4+warp2x4+stage2): ['29.921875 ', '-9.5078125'], time:1.329267ms, swizzle: NOOP, TFLOPS: 103.39
   f16wmma(mma2x4+...+stage3+dsmem): ['29.921875 ', '-9.5078125'], time:1.331329ms, swizzle: NOOP, TFLOPS: 103.23
   f16wmma(mma2x4+...+stage2+dsmem): ['29.921875 ', '-9.5078125'], time:1.328444ms, swizzle: NOOP, TFLOPS: 103.46
 f16wmma(mma2x4+...+stage3+swizzle): ['29.921875 ', '-9.5078125'], time:1.320767ms, swizzle: 1024, TFLOPS: 104.06
 f16wmma(mma2x4+...+stage2+swizzle): ['29.921875 ', '-9.5078125'], time:1.308536ms, swizzle: 1024, TFLOPS: 105.03
  f16wmma(...+stage3+dsmem+swizzle): ['29.921875 ', '-9.5078125'], time:1.320600ms, swizzle: 1024, TFLOPS: 104.07
  f16wmma(...+stage2+dsmem+swizzle): ['29.921875 ', '-9.5078125'], time:1.309466ms, swizzle: 1024, TFLOPS: 104.96
                        f16(cublas): ['29.921875 ', '-9.5078125'], time:1.241862ms, swizzle: NOOP, TFLOPS: 110.67(+5.27%)
                             f16_th: ['29.9375   ', '-9.5703125'], time:1.296436ms, swizzle: NOOP, TFLOPS: 106.01
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=4096
                         f16(naive): ['18.078125 ', '32.90625  '], time:75.24089ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                f16x8pack(t8x8+bcf): ['18.078125 ', '32.90625  '], time:5.732536ms, swizzle: NOOP, TFLOPS: 47.95 (+1212.52%)
           f16x8pack(t8x8+bcf+dbuf): ['18.078125 ', '32.90625  '], time:5.681359ms, swizzle: NOOP, TFLOPS: 48.38 (+0.90%)
           f16x8pack(t8x8+k16+dbuf): ['18.078125 ', '32.90625  '], time:5.430412ms, swizzle: NOOP, TFLOPS: 50.62 (+4.62%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['17.953125 ', '32.8125   '], time:11.64469ms, swizzle: NOOP, TFLOPS: 23.61
                    f16wmma(mma4x2): ['17.953125 ', '32.8125   '], time:5.856013ms, swizzle: NOOP, TFLOPS: 46.94
            f16wmma(mma4x2+warp2x4): ['17.953125 ', '32.8125   '], time:3.582108ms, swizzle: NOOP, TFLOPS: 76.74 (+51.60%)
       f16wmma(mma2x4+warp2x4+dbuf): ['17.953125 ', '32.8125   '], time:2.590525ms, swizzle: NOOP, TFLOPS: 106.11(+38.28%)
     f16wmma(mma2x4+warp2x4+stage3): ['17.953125 ', '32.8125   '], time:2.588713ms, swizzle: NOOP, TFLOPS: 106.18(+0.07%)
     f16wmma(mma2x4+warp2x4+stage2): ['17.953125 ', '32.8125   '], time:2.610671ms, swizzle: NOOP, TFLOPS: 105.29
   f16wmma(mma2x4+...+stage3+dsmem): ['17.953125 ', '32.8125   '], time:2.617657ms, swizzle: NOOP, TFLOPS: 105.01
   f16wmma(mma2x4+...+stage2+dsmem): ['17.953125 ', '32.8125   '], time:2.610099ms, swizzle: NOOP, TFLOPS: 105.31
 f16wmma(mma2x4+...+stage3+swizzle): ['17.953125 ', '32.8125   '], time:2.601087ms, swizzle: 1024, TFLOPS: 105.68
 f16wmma(mma2x4+...+stage2+swizzle): ['17.953125 ', '32.8125   '], time:2.591907ms, swizzle: 1024, TFLOPS: 106.05
  f16wmma(...+stage3+dsmem+swizzle): ['17.953125 ', '32.8125   '], time:2.608931ms, swizzle: 1024, TFLOPS: 105.36
  f16wmma(...+stage2+dsmem+swizzle): ['17.953125 ', '32.8125   '], time:2.586269ms, swizzle: 1024, TFLOPS: 106.28(+0.09%)
                        f16(cublas): ['17.953125 ', '32.8125   '], time:2.462971ms, swizzle: NOOP, TFLOPS: 111.60(+5.01%)
                             f16_th: ['17.96875  ', '32.75     '], time:2.555072ms, swizzle: NOOP, TFLOPS: 107.58
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=8192
                         f16(naive): ['69.5      ', '17.9375   '], time:150.3833ms, swizzle: NOOP, TFLOPS: 3.66  (+0.00%)
                f16x8pack(t8x8+bcf): ['69.5      ', '17.9375   '], time:11.42700ms, swizzle: NOOP, TFLOPS: 48.11 (+1216.03%)
           f16x8pack(t8x8+bcf+dbuf): ['69.5      ', '17.9375   '], time:11.32282ms, swizzle: NOOP, TFLOPS: 48.55 (+0.92%)
           f16x8pack(t8x8+k16+dbuf): ['69.5      ', '17.9375   '], time:11.10188ms, swizzle: NOOP, TFLOPS: 49.52 (+1.99%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['69.25     ', '18.078125 '], time:23.05684ms, swizzle: NOOP, TFLOPS: 23.84
                    f16wmma(mma4x2): ['69.25     ', '18.078125 '], time:11.64009ms, swizzle: NOOP, TFLOPS: 47.23
            f16wmma(mma4x2+warp2x4): ['69.25     ', '18.078125 '], time:7.079064ms, swizzle: NOOP, TFLOPS: 77.66 (+56.83%)
       f16wmma(mma2x4+warp2x4+dbuf): ['69.25     ', '18.078125 '], time:5.142569ms, swizzle: NOOP, TFLOPS: 106.90(+37.66%)
     f16wmma(mma2x4+warp2x4+stage3): ['69.25     ', '18.078125 '], time:5.137395ms, swizzle: NOOP, TFLOPS: 107.01(+0.10%)
     f16wmma(mma2x4+warp2x4+stage2): ['69.25     ', '18.078125 '], time:5.153763ms, swizzle: NOOP, TFLOPS: 106.67
   f16wmma(mma2x4+...+stage3+dsmem): ['69.25     ', '18.078125 '], time:5.166745ms, swizzle: NOOP, TFLOPS: 106.40
   f16wmma(mma2x4+...+stage2+dsmem): ['69.25     ', '18.078125 '], time:5.151641ms, swizzle: NOOP, TFLOPS: 106.71
 f16wmma(mma2x4+...+stage3+swizzle): ['69.25     ', '18.078125 '], time:5.244612ms, swizzle: 1024, TFLOPS: 104.82
 f16wmma(mma2x4+...+stage2+swizzle): ['69.25     ', '18.078125 '], time:5.279600ms, swizzle: 1024, TFLOPS: 104.13
  f16wmma(...+stage3+dsmem+swizzle): ['69.25     ', '18.078125 '], time:5.390441ms, swizzle: 1024, TFLOPS: 101.99
  f16wmma(...+stage2+dsmem+swizzle): ['69.25     ', '18.078125 '], time:5.245709ms, swizzle: 1024, TFLOPS: 104.80
                        f16(cublas): ['69.25     ', '18.078125 '], time:4.908752ms, swizzle: NOOP, TFLOPS: 112.00(+4.66%)
                             f16_th: ['69.3125   ', '18.109375 '], time:5.001747ms, swizzle: NOOP, TFLOPS: 109.91
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=2048
                         f16(naive): ['30.0625   ', '-9.34375  '], time:75.26348ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                f16x8pack(t8x8+bcf): ['30.0625   ', '-9.34375  '], time:5.520558ms, swizzle: NOOP, TFLOPS: 49.79 (+1263.33%)
           f16x8pack(t8x8+bcf+dbuf): ['30.0625   ', '-9.34375  '], time:5.567967ms, swizzle: NOOP, TFLOPS: 49.37
           f16x8pack(t8x8+k16+dbuf): ['30.0625   ', '-9.34375  '], time:5.330693ms, swizzle: NOOP, TFLOPS: 51.57 (+3.56%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['29.921875 ', '-9.5078125'], time:11.68832ms, swizzle: NOOP, TFLOPS: 23.52
                    f16wmma(mma4x2): ['29.921875 ', '-9.5078125'], time:5.869758ms, swizzle: NOOP, TFLOPS: 46.83
            f16wmma(mma4x2+warp2x4): ['29.921875 ', '-9.5078125'], time:3.541529ms, swizzle: NOOP, TFLOPS: 77.62 (+50.52%)
       f16wmma(mma2x4+warp2x4+dbuf): ['29.921875 ', '-9.5078125'], time:2.588903ms, swizzle: NOOP, TFLOPS: 106.18(+36.80%)
     f16wmma(mma2x4+warp2x4+stage3): ['29.921875 ', '-9.5078125'], time:2.589297ms, swizzle: NOOP, TFLOPS: 106.16
     f16wmma(mma2x4+warp2x4+stage2): ['29.921875 ', '-9.5078125'], time:2.600646ms, swizzle: NOOP, TFLOPS: 105.70
   f16wmma(mma2x4+...+stage3+dsmem): ['29.921875 ', '-9.5078125'], time:2.604603ms, swizzle: NOOP, TFLOPS: 105.54
   f16wmma(mma2x4+...+stage2+dsmem): ['29.921875 ', '-9.5078125'], time:2.599084ms, swizzle: NOOP, TFLOPS: 105.76
 f16wmma(mma2x4+...+stage3+swizzle): ['29.921875 ', '-9.5078125'], time:2.595067ms, swizzle: 2048, TFLOPS: 105.92
 f16wmma(mma2x4+...+stage2+swizzle): ['29.921875 ', '-9.5078125'], time:2.574205ms, swizzle: 2048, TFLOPS: 106.78(+0.57%)
  f16wmma(...+stage3+dsmem+swizzle): ['29.921875 ', '-9.5078125'], time:2.599978ms, swizzle: 2048, TFLOPS: 105.72
  f16wmma(...+stage2+dsmem+swizzle): ['29.921875 ', '-9.5078125'], time:2.580559ms, swizzle: 2048, TFLOPS: 106.52
                        f16(cublas): ['29.921875 ', '-9.5078125'], time:2.448010ms, swizzle: NOOP, TFLOPS: 112.29(+5.16%)
                             f16_th: ['29.9375   ', '-9.5703125'], time:2.432334ms, swizzle: NOOP, TFLOPS: 113.01(+0.64%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=4096
                         f16(naive): ['18.078125 ', '32.90625  '], time:150.3454ms, swizzle: NOOP, TFLOPS: 3.66  (+0.00%)
                f16x8pack(t8x8+bcf): ['18.078125 ', '32.90625  '], time:11.15337ms, swizzle: NOOP, TFLOPS: 49.29 (+1247.98%)
           f16x8pack(t8x8+bcf+dbuf): ['18.078125 ', '32.90625  '], time:11.13945ms, swizzle: NOOP, TFLOPS: 49.35 (+0.12%)
           f16x8pack(t8x8+k16+dbuf): ['18.078125 ', '32.90625  '], time:10.81538ms, swizzle: NOOP, TFLOPS: 50.83 (+3.00%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['17.953125 ', '32.8125   '], time:23.15959ms, swizzle: NOOP, TFLOPS: 23.74
                    f16wmma(mma4x2): ['17.953125 ', '32.8125   '], time:11.63694ms, swizzle: NOOP, TFLOPS: 47.24
            f16wmma(mma4x2+warp2x4): ['17.953125 ', '32.8125   '], time:6.961476ms, swizzle: NOOP, TFLOPS: 78.97 (+55.36%)
       f16wmma(mma2x4+warp2x4+dbuf): ['17.953125 ', '32.8125   '], time:5.086100ms, swizzle: NOOP, TFLOPS: 108.09(+36.87%)
     f16wmma(mma2x4+warp2x4+stage3): ['17.953125 ', '32.8125   '], time:5.071794ms, swizzle: NOOP, TFLOPS: 108.39(+0.28%)
     f16wmma(mma2x4+warp2x4+stage2): ['17.953125 ', '32.8125   '], time:5.084371ms, swizzle: NOOP, TFLOPS: 108.13
   f16wmma(mma2x4+...+stage3+dsmem): ['17.953125 ', '32.8125   '], time:5.083274ms, swizzle: NOOP, TFLOPS: 108.15
   f16wmma(mma2x4+...+stage2+dsmem): ['17.953125 ', '32.8125   '], time:5.072879ms, swizzle: NOOP, TFLOPS: 108.37
 f16wmma(mma2x4+...+stage3+swizzle): ['17.953125 ', '32.8125   '], time:5.158448ms, swizzle: 2048, TFLOPS: 106.57
 f16wmma(mma2x4+...+stage2+swizzle): ['17.953125 ', '32.8125   '], time:5.199003ms, swizzle: 2048, TFLOPS: 105.74
  f16wmma(...+stage3+dsmem+swizzle): ['17.953125 ', '32.8125   '], time:5.281746ms, swizzle: 2048, TFLOPS: 104.09
  f16wmma(...+stage2+dsmem+swizzle): ['17.953125 ', '32.8125   '], time:5.193662ms, swizzle: 2048, TFLOPS: 105.85
                        f16(cublas): ['17.953125 ', '32.8125   '], time:4.811286ms, swizzle: NOOP, TFLOPS: 114.26(+5.41%)
                             f16_th: ['17.96875  ', '32.75     '], time:4.888451ms, swizzle: NOOP, TFLOPS: 112.46
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=8192
                         f16(naive): ['69.5      ', '17.9375   '], time:443.4183ms, swizzle: NOOP, TFLOPS: 2.48  (+0.00%)
                f16x8pack(t8x8+bcf): ['69.5      ', '17.9375   '], time:24.40258ms, swizzle: NOOP, TFLOPS: 45.06 (+1717.10%)
           f16x8pack(t8x8+bcf+dbuf): ['69.5      ', '17.9375   '], time:26.13687ms, swizzle: NOOP, TFLOPS: 42.07
           f16x8pack(t8x8+k16+dbuf): ['69.5      ', '17.9375   '], time:24.38509ms, swizzle: NOOP, TFLOPS: 45.09 (+0.07%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['69.25     ', '18.078125 '], time:49.62843ms, swizzle: NOOP, TFLOPS: 22.15
                    f16wmma(mma4x2): ['69.25     ', '18.078125 '], time:25.53668ms, swizzle: NOOP, TFLOPS: 43.06
            f16wmma(mma4x2+warp2x4): ['69.25     ', '18.078125 '], time:14.22798ms, swizzle: NOOP, TFLOPS: 77.28 (+71.39%)
       f16wmma(mma2x4+warp2x4+dbuf): ['69.25     ', '18.078125 '], time:11.14211ms, swizzle: NOOP, TFLOPS: 98.68 (+27.70%)
     f16wmma(mma2x4+warp2x4+stage3): ['69.25     ', '18.078125 '], time:11.10615ms, swizzle: NOOP, TFLOPS: 99.00 (+0.32%)
     f16wmma(mma2x4+warp2x4+stage2): ['69.25     ', '18.078125 '], time:11.75935ms, swizzle: NOOP, TFLOPS: 93.50
   f16wmma(mma2x4+...+stage3+dsmem): ['69.25     ', '18.078125 '], time:11.71027ms, swizzle: NOOP, TFLOPS: 93.89
   f16wmma(mma2x4+...+stage2+dsmem): ['69.25     ', '18.078125 '], time:11.13451ms, swizzle: NOOP, TFLOPS: 98.75
 f16wmma(mma2x4+...+stage3+swizzle): ['69.25     ', '18.078125 '], time:10.18273ms, swizzle: 2048, TFLOPS: 107.98(+9.07%)
 f16wmma(mma2x4+...+stage2+swizzle): ['69.25     ', '18.078125 '], time:10.16882ms, swizzle: 2048, TFLOPS: 108.13(+0.14%)
  f16wmma(...+stage3+dsmem+swizzle): ['69.25     ', '18.078125 '], time:10.23535ms, swizzle: 2048, TFLOPS: 107.42
  f16wmma(...+stage2+dsmem+swizzle): ['69.25     ', '18.078125 '], time:10.11298ms, swizzle: 2048, TFLOPS: 108.72(+0.55%)
                        f16(cublas): ['69.25     ', '18.078125 '], time:9.575200ms, swizzle: NOOP, TFLOPS: 114.83(+5.62%)
                             f16_th: ['69.3125   ', '18.09375  '], time:9.703755ms, swizzle: NOOP, TFLOPS: 113.31
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=2048
                         f16(naive): ['30.0625   ', '-9.34375  '], time:150.4965ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                f16x8pack(t8x8+bcf): ['30.0625   ', '-9.34375  '], time:11.25679ms, swizzle: NOOP, TFLOPS: 48.84 (+1236.94%)
           f16x8pack(t8x8+bcf+dbuf): ['30.0625   ', '-9.34375  '], time:11.15907ms, swizzle: NOOP, TFLOPS: 49.27 (+0.88%)
           f16x8pack(t8x8+k16+dbuf): ['30.0625   ', '-9.34375  '], time:10.77163ms, swizzle: NOOP, TFLOPS: 51.04 (+3.60%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['29.921875 ', '-9.5078125'], time:23.21388ms, swizzle: NOOP, TFLOPS: 23.68
                    f16wmma(mma4x2): ['29.921875 ', '-9.5078125'], time:11.69066ms, swizzle: NOOP, TFLOPS: 47.03
            f16wmma(mma4x2+warp2x4): ['29.921875 ', '-9.5078125'], time:6.881964ms, swizzle: NOOP, TFLOPS: 79.88 (+56.52%)
       f16wmma(mma2x4+warp2x4+dbuf): ['29.921875 ', '-9.5078125'], time:5.192875ms, swizzle: NOOP, TFLOPS: 105.87(+32.53%)
     f16wmma(mma2x4+warp2x4+stage3): ['29.921875 ', '-9.5078125'], time:5.165994ms, swizzle: NOOP, TFLOPS: 106.42(+0.52%)
     f16wmma(mma2x4+warp2x4+stage2): ['29.921875 ', '-9.5078125'], time:5.184519ms, swizzle: NOOP, TFLOPS: 106.04
   f16wmma(mma2x4+...+stage3+dsmem): ['29.921875 ', '-9.5078125'], time:5.191397ms, swizzle: NOOP, TFLOPS: 105.90
   f16wmma(mma2x4+...+stage2+dsmem): ['29.921875 ', '-9.5078125'], time:5.183148ms, swizzle: NOOP, TFLOPS: 106.07
 f16wmma(mma2x4+...+stage3+swizzle): ['29.921875 ', '-9.5078125'], time:5.240654ms, swizzle: 4096, TFLOPS: 104.90
 f16wmma(mma2x4+...+stage2+swizzle): ['29.921875 ', '-9.5078125'], time:5.309236ms, swizzle: 4096, TFLOPS: 103.55
  f16wmma(...+stage3+dsmem+swizzle): ['29.921875 ', '-9.5078125'], time:5.356001ms, swizzle: 4096, TFLOPS: 102.64
  f16wmma(...+stage2+dsmem+swizzle): ['29.921875 ', '-9.5078125'], time:5.244457ms, swizzle: 4096, TFLOPS: 104.83
                        f16(cublas): ['29.921875 ', '-9.5078125'], time:4.865396ms, swizzle: NOOP, TFLOPS: 112.99(+6.18%)
                             f16_th: ['29.9375   ', '-9.5703125'], time:4.924583ms, swizzle: NOOP, TFLOPS: 111.63
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=4096
                         f16(naive): ['18.078125 ', '32.90625  '], time:458.6103ms, swizzle: NOOP, TFLOPS: 2.40  (+0.00%)
                f16x8pack(t8x8+bcf): ['18.078125 ', '32.90625  '], time:24.35351ms, swizzle: NOOP, TFLOPS: 45.15 (+1783.14%)
           f16x8pack(t8x8+bcf+dbuf): ['18.078125 ', '32.90625  '], time:26.28115ms, swizzle: NOOP, TFLOPS: 41.84
           f16x8pack(t8x8+k16+dbuf): ['18.078125 ', '32.90625  '], time:24.56663ms, swizzle: NOOP, TFLOPS: 44.76
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['17.953125 ', '32.8125   '], time:75.99920ms, swizzle: NOOP, TFLOPS: 14.47
                    f16wmma(mma4x2): ['17.953125 ', '32.8125   '], time:31.42518ms, swizzle: NOOP, TFLOPS: 34.99
            f16wmma(mma4x2+warp2x4): ['17.953125 ', '32.8125   '], time:14.28223ms, swizzle: NOOP, TFLOPS: 76.98 (+70.52%)
       f16wmma(mma2x4+warp2x4+dbuf): ['17.953125 ', '32.8125   '], time:11.23386ms, swizzle: NOOP, TFLOPS: 97.87 (+27.14%)
     f16wmma(mma2x4+warp2x4+stage3): ['17.953125 ', '32.8125   '], time:12.16307ms, swizzle: NOOP, TFLOPS: 90.40
     f16wmma(mma2x4+warp2x4+stage2): ['17.953125 ', '32.8125   '], time:11.72069ms, swizzle: NOOP, TFLOPS: 93.81
   f16wmma(mma2x4+...+stage3+dsmem): ['17.953125 ', '32.8125   '], time:11.45683ms, swizzle: NOOP, TFLOPS: 95.97
   f16wmma(mma2x4+...+stage2+dsmem): ['17.953125 ', '32.8125   '], time:11.26756ms, swizzle: NOOP, TFLOPS: 97.58
 f16wmma(mma2x4+...+stage3+swizzle): ['17.953125 ', '32.8125   '], time:10.29720ms, swizzle: 4096, TFLOPS: 106.78(+9.10%)
 f16wmma(mma2x4+...+stage2+swizzle): ['17.953125 ', '32.8125   '], time:10.10987ms, swizzle: 4096, TFLOPS: 108.76(+1.85%)
  f16wmma(...+stage3+dsmem+swizzle): ['17.953125 ', '32.8125   '], time:10.21662ms, swizzle: 4096, TFLOPS: 107.62
  f16wmma(...+stage2+dsmem+swizzle): ['17.953125 ', '32.8125   '], time:10.23129ms, swizzle: 4096, TFLOPS: 107.47
                        f16(cublas): ['17.953125 ', '32.8125   '], time:9.730219ms, swizzle: NOOP, TFLOPS: 113.00(+3.90%)
                             f16_th: ['17.96875  ', '32.75     '], time:9.567761ms, swizzle: NOOP, TFLOPS: 114.92(+1.70%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=8192
                         f16(naive): ['69.5      ', '17.9375   '], time:915.9307ms, swizzle: NOOP, TFLOPS: 2.40  (+0.00%)
                f16x8pack(t8x8+bcf): ['69.5      ', '17.9375   '], time:52.36693ms, swizzle: NOOP, TFLOPS: 41.99 (+1649.06%)
           f16x8pack(t8x8+bcf+dbuf): ['69.5      ', '17.9375   '], time:52.84373ms, swizzle: NOOP, TFLOPS: 41.61
           f16x8pack(t8x8+k16+dbuf): ['69.5      ', '17.9375   '], time:51.87792ms, swizzle: NOOP, TFLOPS: 42.39 (+0.94%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['69.25     ', '18.078125 '], time:407.3698ms, swizzle: NOOP, TFLOPS: 5.40
                    f16wmma(mma4x2): ['69.25     ', '18.078125 '], time:62.13268ms, swizzle: NOOP, TFLOPS: 35.39
            f16wmma(mma4x2+warp2x4): ['69.25     ', '18.078125 '], time:28.18557ms, swizzle: NOOP, TFLOPS: 78.02 (+84.06%)
       f16wmma(mma2x4+warp2x4+dbuf): ['69.25     ', '18.078125 '], time:25.32980ms, swizzle: NOOP, TFLOPS: 86.82 (+11.27%)
     f16wmma(mma2x4+warp2x4+stage3): ['69.25     ', '18.078125 '], time:25.23270ms, swizzle: NOOP, TFLOPS: 87.15 (+0.38%)
     f16wmma(mma2x4+warp2x4+stage2): ['69.25     ', '18.078125 '], time:25.03691ms, swizzle: NOOP, TFLOPS: 87.83 (+0.78%)
   f16wmma(mma2x4+...+stage3+dsmem): ['69.25     ', '18.078125 '], time:25.10446ms, swizzle: NOOP, TFLOPS: 87.59
   f16wmma(mma2x4+...+stage2+dsmem): ['69.25     ', '18.078125 '], time:25.04101ms, swizzle: NOOP, TFLOPS: 87.82
 f16wmma(mma2x4+...+stage3+swizzle): ['69.25     ', '18.078125 '], time:20.55739ms, swizzle: 4096, TFLOPS: 106.97(+21.79%)
 f16wmma(mma2x4+...+stage2+swizzle): ['69.25     ', '18.078125 '], time:20.30926ms, swizzle: 4096, TFLOPS: 108.28(+1.22%)
  f16wmma(...+stage3+dsmem+swizzle): ['69.25     ', '18.078125 '], time:20.56778ms, swizzle: 4096, TFLOPS: 106.92
  f16wmma(...+stage2+dsmem+swizzle): ['69.25     ', '18.078125 '], time:20.31245ms, swizzle: 4096, TFLOPS: 108.26
                        f16(cublas): ['69.25     ', '18.078125 '], time:19.35142ms, swizzle: NOOP, TFLOPS: 113.64(+4.95%)
                             f16_th: ['69.3125   ', '18.09375  '], time:19.31538ms, swizzle: NOOP, TFLOPS: 113.85(+0.19%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=2048
                         f16(naive): ['30.0625   ', '-9.34375  '], time:75.27084ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                f16x8pack(t8x8+bcf): ['30.0625   ', '-9.34375  '], time:5.641674ms, swizzle: NOOP, TFLOPS: 48.72 (+1234.19%)
           f16x8pack(t8x8+bcf+dbuf): ['30.0625   ', '-9.34375  '], time:5.619812ms, swizzle: NOOP, TFLOPS: 48.91 (+0.39%)
           f16x8pack(t8x8+k16+dbuf): ['30.0625   ', '-9.34375  '], time:5.412900ms, swizzle: NOOP, TFLOPS: 50.78 (+3.82%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['29.921875 ', '-9.5078125'], time:11.67960ms, swizzle: NOOP, TFLOPS: 23.53
                    f16wmma(mma4x2): ['29.921875 ', '-9.5078125'], time:5.873727ms, swizzle: NOOP, TFLOPS: 46.80
            f16wmma(mma4x2+warp2x4): ['29.921875 ', '-9.5078125'], time:3.539693ms, swizzle: NOOP, TFLOPS: 77.66 (+52.92%)
       f16wmma(mma2x4+warp2x4+dbuf): ['29.921875 ', '-9.5078125'], time:2.593779ms, swizzle: NOOP, TFLOPS: 105.98(+36.47%)
     f16wmma(mma2x4+warp2x4+stage3): ['29.921875 ', '-9.5078125'], time:2.594280ms, swizzle: NOOP, TFLOPS: 105.96
     f16wmma(mma2x4+warp2x4+stage2): ['29.921875 ', '-9.5078125'], time:2.603912ms, swizzle: NOOP, TFLOPS: 105.56
   f16wmma(mma2x4+...+stage3+dsmem): ['29.921875 ', '-9.5078125'], time:2.603149ms, swizzle: NOOP, TFLOPS: 105.59
   f16wmma(mma2x4+...+stage2+dsmem): ['29.921875 ', '-9.5078125'], time:2.597057ms, swizzle: NOOP, TFLOPS: 105.84
 f16wmma(mma2x4+...+stage3+swizzle): ['29.921875 ', '-9.5078125'], time:2.637946ms, swizzle: 1024, TFLOPS: 104.20
 f16wmma(mma2x4+...+stage2+swizzle): ['29.921875 ', '-9.5078125'], time:2.631723ms, swizzle: 1024, TFLOPS: 104.45
  f16wmma(...+stage3+dsmem+swizzle): ['29.921875 ', '-9.5078125'], time:2.651989ms, swizzle: 1024, TFLOPS: 103.65
  f16wmma(...+stage2+dsmem+swizzle): ['29.921875 ', '-9.5078125'], time:2.653670ms, swizzle: 1024, TFLOPS: 103.58
                        f16(cublas): ['29.921875 ', '-9.5078125'], time:2.528202ms, swizzle: NOOP, TFLOPS: 108.72(+2.59%)
                             f16_th: ['29.9375   ', '-9.5703125'], time:2.461636ms, swizzle: NOOP, TFLOPS: 111.66(+2.70%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=4096
                         f16(naive): ['18.078125 ', '32.90625  '], time:150.3604ms, swizzle: NOOP, TFLOPS: 3.66  (+0.00%)
                f16x8pack(t8x8+bcf): ['18.078125 ', '32.90625  '], time:11.28156ms, swizzle: NOOP, TFLOPS: 48.73 (+1232.80%)
           f16x8pack(t8x8+bcf+dbuf): ['18.078125 ', '32.90625  '], time:11.18837ms, swizzle: NOOP, TFLOPS: 49.14 (+0.83%)
           f16x8pack(t8x8+k16+dbuf): ['18.078125 ', '32.90625  '], time:10.94454ms, swizzle: NOOP, TFLOPS: 50.23 (+2.23%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['17.953125 ', '32.8125   '], time:23.15819ms, swizzle: NOOP, TFLOPS: 23.74
                    f16wmma(mma4x2): ['17.953125 ', '32.8125   '], time:11.64075ms, swizzle: NOOP, TFLOPS: 47.23
            f16wmma(mma4x2+warp2x4): ['17.953125 ', '32.8125   '], time:6.959927ms, swizzle: NOOP, TFLOPS: 78.99 (+57.25%)
       f16wmma(mma2x4+warp2x4+dbuf): ['17.953125 ', '32.8125   '], time:5.114102ms, swizzle: NOOP, TFLOPS: 107.50(+36.09%)
     f16wmma(mma2x4+warp2x4+stage3): ['17.953125 ', '32.8125   '], time:5.091762ms, swizzle: NOOP, TFLOPS: 107.97(+0.44%)
     f16wmma(mma2x4+warp2x4+stage2): ['17.953125 ', '32.8125   '], time:5.101358ms, swizzle: NOOP, TFLOPS: 107.77
   f16wmma(mma2x4+...+stage3+dsmem): ['17.953125 ', '32.8125   '], time:5.129981ms, swizzle: NOOP, TFLOPS: 107.17
   f16wmma(mma2x4+...+stage2+dsmem): ['17.953125 ', '32.8125   '], time:5.100822ms, swizzle: NOOP, TFLOPS: 107.78
 f16wmma(mma2x4+...+stage3+swizzle): ['17.953125 ', '32.8125   '], time:5.289304ms, swizzle: 1024, TFLOPS: 103.94
 f16wmma(mma2x4+...+stage2+swizzle): ['17.953125 ', '32.8125   '], time:5.343866ms, swizzle: 1024, TFLOPS: 102.88
  f16wmma(...+stage3+dsmem+swizzle): ['17.953125 ', '32.8125   '], time:5.399477ms, swizzle: 1024, TFLOPS: 101.82
  f16wmma(...+stage2+dsmem+swizzle): ['17.953125 ', '32.8125   '], time:5.260670ms, swizzle: 1024, TFLOPS: 104.50
                        f16(cublas): ['17.953125 ', '32.8125   '], time:4.835724ms, swizzle: NOOP, TFLOPS: 113.69(+5.29%)
                             f16_th: ['17.96875  ', '32.75     '], time:4.891443ms, swizzle: NOOP, TFLOPS: 112.39
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=8192
                         f16(naive): ['69.5      ', '17.9375   '], time:300.5170ms, swizzle: NOOP, TFLOPS: 3.66  (+0.00%)
                f16x8pack(t8x8+bcf): ['69.5      ', '17.9375   '], time:22.59441ms, swizzle: NOOP, TFLOPS: 48.66 (+1230.05%)
           f16x8pack(t8x8+bcf+dbuf): ['69.5      ', '17.9375   '], time:22.84268ms, swizzle: NOOP, TFLOPS: 48.13
           f16x8pack(t8x8+k16+dbuf): ['69.5      ', '17.9375   '], time:21.51962ms, swizzle: NOOP, TFLOPS: 51.09 (+4.99%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['69.25     ', '18.078125 '], time:45.93516ms, swizzle: NOOP, TFLOPS: 23.94
                    f16wmma(mma4x2): ['69.25     ', '18.078125 '], time:23.20855ms, swizzle: NOOP, TFLOPS: 47.38
            f16wmma(mma4x2+warp2x4): ['69.25     ', '18.078125 '], time:13.75201ms, swizzle: NOOP, TFLOPS: 79.95 (+56.48%)
       f16wmma(mma2x4+warp2x4+dbuf): ['69.25     ', '18.078125 '], time:10.15893ms, swizzle: NOOP, TFLOPS: 108.23(+35.37%)
     f16wmma(mma2x4+warp2x4+stage3): ['69.25     ', '18.078125 '], time:10.07235ms, swizzle: NOOP, TFLOPS: 109.16(+0.86%)
     f16wmma(mma2x4+warp2x4+stage2): ['69.25     ', '18.078125 '], time:10.10667ms, swizzle: NOOP, TFLOPS: 108.79
   f16wmma(mma2x4+...+stage3+dsmem): ['69.25     ', '18.078125 '], time:10.37973ms, swizzle: NOOP, TFLOPS: 105.93
   f16wmma(mma2x4+...+stage2+dsmem): ['69.25     ', '18.078125 '], time:10.14732ms, swizzle: NOOP, TFLOPS: 108.35
 f16wmma(mma2x4+...+stage3+swizzle): ['69.25     ', '18.078125 '], time:10.42317ms, swizzle: 1024, TFLOPS: 105.49
 f16wmma(mma2x4+...+stage2+swizzle): ['69.25     ', '18.078125 '], time:10.29515ms, swizzle: 1024, TFLOPS: 106.80
  f16wmma(...+stage3+dsmem+swizzle): ['69.25     ', '18.078125 '], time:10.66128ms, swizzle: 1024, TFLOPS: 103.13
  f16wmma(...+stage2+dsmem+swizzle): ['69.25     ', '18.078125 '], time:10.28729ms, swizzle: 1024, TFLOPS: 106.88
                        f16(cublas): ['69.25     ', '18.078125 '], time:9.605360ms, swizzle: NOOP, TFLOPS: 114.47(+4.86%)
                             f16_th: ['69.3125   ', '18.109375 '], time:9.978485ms, swizzle: NOOP, TFLOPS: 110.19
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=2048
                         f16(naive): ['30.0625   ', '-9.34375  '], time:150.4853ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                f16x8pack(t8x8+bcf): ['30.0625   ', '-9.34375  '], time:11.20852ms, swizzle: NOOP, TFLOPS: 49.05 (+1242.60%)
           f16x8pack(t8x8+bcf+dbuf): ['30.0625   ', '-9.34375  '], time:11.20126ms, swizzle: NOOP, TFLOPS: 49.08 (+0.06%)
           f16x8pack(t8x8+k16+dbuf): ['30.0625   ', '-9.34375  '], time:11.05253ms, swizzle: NOOP, TFLOPS: 49.74 (+1.35%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['29.921875 ', '-9.5078125'], time:23.23106ms, swizzle: NOOP, TFLOPS: 23.66
                    f16wmma(mma4x2): ['29.921875 ', '-9.5078125'], time:11.69289ms, swizzle: NOOP, TFLOPS: 47.02
            f16wmma(mma4x2+warp2x4): ['29.921875 ', '-9.5078125'], time:6.919693ms, swizzle: NOOP, TFLOPS: 79.45 (+59.73%)
       f16wmma(mma2x4+warp2x4+dbuf): ['29.921875 ', '-9.5078125'], time:5.214202ms, swizzle: NOOP, TFLOPS: 105.43(+32.71%)
     f16wmma(mma2x4+warp2x4+stage3): ['29.921875 ', '-9.5078125'], time:5.194151ms, swizzle: NOOP, TFLOPS: 105.84(+0.39%)
     f16wmma(mma2x4+warp2x4+stage2): ['29.921875 ', '-9.5078125'], time:5.186319ms, swizzle: NOOP, TFLOPS: 106.00(+0.15%)
   f16wmma(mma2x4+...+stage3+dsmem): ['29.921875 ', '-9.5078125'], time:5.222153ms, swizzle: NOOP, TFLOPS: 105.27
   f16wmma(mma2x4+...+stage2+dsmem): ['29.921875 ', '-9.5078125'], time:5.189323ms, swizzle: NOOP, TFLOPS: 105.94
 f16wmma(mma2x4+...+stage3+swizzle): ['29.921875 ', '-9.5078125'], time:5.349266ms, swizzle: 2048, TFLOPS: 102.77
 f16wmma(mma2x4+...+stage2+swizzle): ['29.921875 ', '-9.5078125'], time:5.379891ms, swizzle: 2048, TFLOPS: 102.19
  f16wmma(...+stage3+dsmem+swizzle): ['29.921875 ', '-9.5078125'], time:5.450582ms, swizzle: 2048, TFLOPS: 100.86
  f16wmma(...+stage2+dsmem+swizzle): ['29.921875 ', '-9.5078125'], time:5.306792ms, swizzle: 2048, TFLOPS: 103.59
                        f16(cublas): ['29.921875 ', '-9.5078125'], time:4.886972ms, swizzle: NOOP, TFLOPS: 112.49(+6.13%)
                             f16_th: ['29.9375   ', '-9.5703125'], time:4.879462ms, swizzle: NOOP, TFLOPS: 112.67(+0.15%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=4096
                         f16(naive): ['18.078125 ', '32.90625  '], time:300.5843ms, swizzle: NOOP, TFLOPS: 3.66  (+0.00%)
                f16x8pack(t8x8+bcf): ['18.078125 ', '32.90625  '], time:22.31094ms, swizzle: NOOP, TFLOPS: 49.28 (+1247.25%)
           f16x8pack(t8x8+bcf+dbuf): ['18.078125 ', '32.90625  '], time:22.74484ms, swizzle: NOOP, TFLOPS: 48.34
           f16x8pack(t8x8+k16+dbuf): ['18.078125 ', '32.90625  '], time:21.43429ms, swizzle: NOOP, TFLOPS: 51.30 (+4.09%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['17.953125 ', '32.8125   '], time:46.16487ms, swizzle: NOOP, TFLOPS: 23.82
                    f16wmma(mma4x2): ['17.953125 ', '32.8125   '], time:23.24142ms, swizzle: NOOP, TFLOPS: 47.31
            f16wmma(mma4x2+warp2x4): ['17.953125 ', '32.8125   '], time:13.62026ms, swizzle: NOOP, TFLOPS: 80.73 (+57.37%)
       f16wmma(mma2x4+warp2x4+dbuf): ['17.953125 ', '32.8125   '], time:10.22562ms, swizzle: NOOP, TFLOPS: 107.53(+33.20%)
     f16wmma(mma2x4+warp2x4+stage3): ['17.953125 ', '32.8125   '], time:10.17456ms, swizzle: NOOP, TFLOPS: 108.06(+0.50%)
     f16wmma(mma2x4+warp2x4+stage2): ['17.953125 ', '32.8125   '], time:10.15427ms, swizzle: NOOP, TFLOPS: 108.28(+0.20%)
   f16wmma(mma2x4+...+stage3+dsmem): ['17.953125 ', '32.8125   '], time:10.42321ms, swizzle: NOOP, TFLOPS: 105.49
   f16wmma(mma2x4+...+stage2+dsmem): ['17.953125 ', '32.8125   '], time:10.21900ms, swizzle: NOOP, TFLOPS: 107.59
 f16wmma(mma2x4+...+stage3+swizzle): ['17.953125 ', '32.8125   '], time:10.34308ms, swizzle: 2048, TFLOPS: 106.30
 f16wmma(mma2x4+...+stage2+swizzle): ['17.953125 ', '32.8125   '], time:10.24296ms, swizzle: 2048, TFLOPS: 107.34
  f16wmma(...+stage3+dsmem+swizzle): ['17.953125 ', '32.8125   '], time:10.60774ms, swizzle: 2048, TFLOPS: 103.65
  f16wmma(...+stage2+dsmem+swizzle): ['17.953125 ', '32.8125   '], time:10.28887ms, swizzle: 2048, TFLOPS: 106.86
                        f16(cublas): ['17.953125 ', '32.8125   '], time:9.570908ms, swizzle: NOOP, TFLOPS: 114.88(+6.10%)
                             f16_th: ['17.96875  ', '32.75     '], time:9.563028ms, swizzle: NOOP, TFLOPS: 114.98(+0.08%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=8192
                         f16(naive): ['69.5      ', '17.9375   '], time:887.2872ms, swizzle: NOOP, TFLOPS: 2.48  (+0.00%)
                f16x8pack(t8x8+bcf): ['69.5      ', '17.9375   '], time:50.07082ms, swizzle: NOOP, TFLOPS: 43.92 (+1672.06%)
           f16x8pack(t8x8+bcf+dbuf): ['69.5      ', '17.9375   '], time:51.13327ms, swizzle: NOOP, TFLOPS: 43.01
           f16x8pack(t8x8+k16+dbuf): ['69.5      ', '17.9375   '], time:50.14414ms, swizzle: NOOP, TFLOPS: 43.85
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['69.25     ', '18.078125 '], time:99.65944ms, swizzle: NOOP, TFLOPS: 22.07
                    f16wmma(mma4x2): ['69.25     ', '18.078125 '], time:51.41550ms, swizzle: NOOP, TFLOPS: 42.77
            f16wmma(mma4x2+warp2x4): ['69.25     ', '18.078125 '], time:28.02135ms, swizzle: NOOP, TFLOPS: 78.48 (+78.69%)
       f16wmma(mma2x4+warp2x4+dbuf): ['69.25     ', '18.078125 '], time:22.39497ms, swizzle: NOOP, TFLOPS: 98.19 (+25.12%)
     f16wmma(mma2x4+warp2x4+stage3): ['69.25     ', '18.078125 '], time:23.69903ms, swizzle: NOOP, TFLOPS: 92.79
     f16wmma(mma2x4+warp2x4+stage2): ['69.25     ', '18.078125 '], time:22.66047ms, swizzle: NOOP, TFLOPS: 97.04
   f16wmma(mma2x4+...+stage3+dsmem): ['69.25     ', '18.078125 '], time:23.20532ms, swizzle: NOOP, TFLOPS: 94.76
   f16wmma(mma2x4+...+stage2+dsmem): ['69.25     ', '18.078125 '], time:23.02054ms, swizzle: NOOP, TFLOPS: 95.52
 f16wmma(mma2x4+...+stage3+swizzle): ['69.25     ', '18.078125 '], time:20.40696ms, swizzle: 2048, TFLOPS: 107.76(+9.74%)
 f16wmma(mma2x4+...+stage2+swizzle): ['69.25     ', '18.078125 '], time:20.46033ms, swizzle: 2048, TFLOPS: 107.48
  f16wmma(...+stage3+dsmem+swizzle): ['69.25     ', '18.078125 '], time:20.46492ms, swizzle: 2048, TFLOPS: 107.45
  f16wmma(...+stage2+dsmem+swizzle): ['69.25     ', '18.078125 '], time:20.42322ms, swizzle: 2048, TFLOPS: 107.67
                        f16(cublas): ['69.25     ', '18.078125 '], time:19.28085ms, swizzle: NOOP, TFLOPS: 114.05(+5.84%)
                             f16_th: ['69.3125   ', '18.09375  '], time:19.37738ms, swizzle: NOOP, TFLOPS: 113.48
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=2048
                         f16(naive): ['30.0625   ', '-9.34375  '], time:300.9566ms, swizzle: NOOP, TFLOPS: 3.65  (+0.00%)
                f16x8pack(t8x8+bcf): ['30.0625   ', '-9.34375  '], time:22.56891ms, swizzle: NOOP, TFLOPS: 48.72 (+1233.50%)
           f16x8pack(t8x8+bcf+dbuf): ['30.0625   ', '-9.34375  '], time:22.75469ms, swizzle: NOOP, TFLOPS: 48.32
           f16x8pack(t8x8+k16+dbuf): ['30.0625   ', '-9.34375  '], time:21.56084ms, swizzle: NOOP, TFLOPS: 51.00 (+4.68%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['29.921875 ', '-9.5078125'], time:46.28647ms, swizzle: NOOP, TFLOPS: 23.75
                    f16wmma(mma4x2): ['29.921875 ', '-9.5078125'], time:23.49380ms, swizzle: NOOP, TFLOPS: 46.80
            f16wmma(mma4x2+warp2x4): ['29.921875 ', '-9.5078125'], time:13.70358ms, swizzle: NOOP, TFLOPS: 80.24 (+57.34%)
       f16wmma(mma2x4+warp2x4+dbuf): ['29.921875 ', '-9.5078125'], time:10.52159ms, swizzle: NOOP, TFLOPS: 104.50(+30.24%)
     f16wmma(mma2x4+warp2x4+stage3): ['29.921875 ', '-9.5078125'], time:10.39229ms, swizzle: NOOP, TFLOPS: 105.80(+1.24%)
     f16wmma(mma2x4+warp2x4+stage2): ['29.921875 ', '-9.5078125'], time:10.39220ms, swizzle: NOOP, TFLOPS: 105.80(+0.00%)
   f16wmma(mma2x4+...+stage3+dsmem): ['29.921875 ', '-9.5078125'], time:10.70573ms, swizzle: NOOP, TFLOPS: 102.70
   f16wmma(mma2x4+...+stage2+dsmem): ['29.921875 ', '-9.5078125'], time:10.50692ms, swizzle: NOOP, TFLOPS: 104.65
 f16wmma(mma2x4+...+stage3+swizzle): ['29.921875 ', '-9.5078125'], time:10.52926ms, swizzle: 4096, TFLOPS: 104.42
 f16wmma(mma2x4+...+stage2+swizzle): ['29.921875 ', '-9.5078125'], time:10.44615ms, swizzle: 4096, TFLOPS: 105.26
  f16wmma(...+stage3+dsmem+swizzle): ['29.921875 ', '-9.5078125'], time:10.70200ms, swizzle: 4096, TFLOPS: 102.74
  f16wmma(...+stage2+dsmem+swizzle): ['29.921875 ', '-9.5078125'], time:10.43186ms, swizzle: 4096, TFLOPS: 105.40
                        f16(cublas): ['29.921875 ', '-9.5078125'], time:9.740543ms, swizzle: NOOP, TFLOPS: 112.88(+6.69%)
                             f16_th: ['29.9375   ', '-9.5703125'], time:9.693014ms, swizzle: NOOP, TFLOPS: 113.43(+0.49%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=4096
                         f16(naive): ['18.078125 ', '32.90625  '], time:917.0645ms, swizzle: NOOP, TFLOPS: 2.40  (+0.00%)
                f16x8pack(t8x8+bcf): ['18.078125 ', '32.90625  '], time:50.57306ms, swizzle: NOOP, TFLOPS: 43.48 (+1713.35%)
           f16x8pack(t8x8+bcf+dbuf): ['18.078125 ', '32.90625  '], time:51.68246ms, swizzle: NOOP, TFLOPS: 42.55
           f16x8pack(t8x8+k16+dbuf): ['18.078125 ', '32.90625  '], time:49.95229ms, swizzle: NOOP, TFLOPS: 44.02 (+1.24%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['17.953125 ', '32.8125   '], time:151.9513ms, swizzle: NOOP, TFLOPS: 14.47
                    f16wmma(mma4x2): ['17.953125 ', '32.8125   '], time:63.35564ms, swizzle: NOOP, TFLOPS: 34.71
            f16wmma(mma4x2+warp2x4): ['17.953125 ', '32.8125   '], time:28.43757ms, swizzle: NOOP, TFLOPS: 77.33 (+75.66%)
       f16wmma(mma2x4+warp2x4+dbuf): ['17.953125 ', '32.8125   '], time:24.45222ms, swizzle: NOOP, TFLOPS: 89.93 (+16.30%)
     f16wmma(mma2x4+warp2x4+stage3): ['17.953125 ', '32.8125   '], time:24.54504ms, swizzle: NOOP, TFLOPS: 89.59
     f16wmma(mma2x4+warp2x4+stage2): ['17.953125 ', '32.8125   '], time:23.58353ms, swizzle: NOOP, TFLOPS: 93.24 (+3.68%)
   f16wmma(mma2x4+...+stage3+dsmem): ['17.953125 ', '32.8125   '], time:25.04811ms, swizzle: NOOP, TFLOPS: 87.79
   f16wmma(mma2x4+...+stage2+dsmem): ['17.953125 ', '32.8125   '], time:23.30942ms, swizzle: NOOP, TFLOPS: 94.34 (+1.18%)
 f16wmma(mma2x4+...+stage3+swizzle): ['17.953125 ', '32.8125   '], time:20.55881ms, swizzle: 4096, TFLOPS: 106.96(+13.38%)
 f16wmma(mma2x4+...+stage2+swizzle): ['17.953125 ', '32.8125   '], time:20.38375ms, swizzle: 4096, TFLOPS: 107.88(+0.86%)
  f16wmma(...+stage3+dsmem+swizzle): ['17.953125 ', '32.8125   '], time:20.68349ms, swizzle: 4096, TFLOPS: 106.32
  f16wmma(...+stage2+dsmem+swizzle): ['17.953125 ', '32.8125   '], time:20.32511ms, swizzle: 4096, TFLOPS: 108.19(+0.29%)
                        f16(cublas): ['17.953125 ', '32.8125   '], time:19.43935ms, swizzle: NOOP, TFLOPS: 113.12(+4.56%)
                             f16_th: ['17.96875  ', '32.75     '], time:19.07173ms, swizzle: NOOP, TFLOPS: 115.30(+1.93%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=8192
                         f16(naive): ['69.5      ', '17.9375   '], time:1831.563ms, swizzle: NOOP, TFLOPS: 2.40  (+0.00%)
                f16x8pack(t8x8+bcf): ['69.5      ', '17.9375   '], time:106.1777ms, swizzle: NOOP, TFLOPS: 41.42 (+1625.00%)
           f16x8pack(t8x8+bcf+dbuf): ['69.5      ', '17.9375   '], time:107.1921ms, swizzle: NOOP, TFLOPS: 41.03
           f16x8pack(t8x8+k16+dbuf): ['69.5      ', '17.9375   '], time:105.5787ms, swizzle: NOOP, TFLOPS: 41.66 (+0.57%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                     f16wmma(naive): ['69.25     ', '18.078125 '], time:824.0843ms, swizzle: NOOP, TFLOPS: 5.34
                    f16wmma(mma4x2): ['69.25     ', '18.078125 '], time:125.0671ms, swizzle: NOOP, TFLOPS: 35.17
            f16wmma(mma4x2+warp2x4): ['69.25     ', '18.078125 '], time:56.59587ms, swizzle: NOOP, TFLOPS: 77.71 (+86.55%)
       f16wmma(mma2x4+warp2x4+dbuf): ['69.25     ', '18.078125 '], time:50.86129ms, swizzle: NOOP, TFLOPS: 86.47 (+11.27%)
     f16wmma(mma2x4+warp2x4+stage3): ['69.25     ', '18.078125 '], time:50.88164ms, swizzle: NOOP, TFLOPS: 86.44
     f16wmma(mma2x4+warp2x4+stage2): ['69.25     ', '18.078125 '], time:50.45264ms, swizzle: NOOP, TFLOPS: 87.17 (+0.81%)
   f16wmma(mma2x4+...+stage3+dsmem): ['69.25     ', '18.078125 '], time:50.89710ms, swizzle: NOOP, TFLOPS: 86.41
   f16wmma(mma2x4+...+stage2+dsmem): ['69.25     ', '18.078125 '], time:50.32118ms, swizzle: NOOP, TFLOPS: 87.40 (+0.26%)
 f16wmma(mma2x4+...+stage3+swizzle): ['69.25     ', '18.078125 '], time:41.41900ms, swizzle: 4096, TFLOPS: 106.18(+21.49%)
 f16wmma(mma2x4+...+stage2+swizzle): ['69.25     ', '18.078125 '], time:40.82697ms, swizzle: 4096, TFLOPS: 107.72(+1.45%)
  f16wmma(...+stage3+dsmem+swizzle): ['69.25     ', '18.078125 '], time:41.47491ms, swizzle: 4096, TFLOPS: 106.04
  f16wmma(...+stage2+dsmem+swizzle): ['69.25     ', '18.078125 '], time:40.86352ms, swizzle: 4096, TFLOPS: 107.63
                        f16(cublas): ['69.25     ', '18.078125 '], time:38.96132ms, swizzle: NOOP, TFLOPS: 112.88(+4.79%)
                             f16_th: ['69.3125   ', '18.09375  '], time:38.87616ms, swizzle: NOOP, TFLOPS: 113.13(+0.22%)
----------------------------------------------------------------------------------------------------------------------------------
```
