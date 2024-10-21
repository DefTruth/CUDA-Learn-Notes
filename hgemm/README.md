# HGEMM 

## 0x00 说明

包含以下内容：

- [X] hgemm_sliced_k_f16_kernel 
- [X] hgemm_t_8x8_sliced_k_f16x4_kernel(unpack)
- [X] hgemm_t_8x8_sliced_k_f16x4_pack_kernel(pack 16x4)
- [X] hgemm_t_8x8_sliced_k_f16x4_bcf_kernel(bank conflicts reduce)
- [X] hgemm_t_8x8_sliced_k_f16x4_pack_bcf_kernel(bank conflicts reduce, pack)
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
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages(WMMA, Tile MMA/Warp, Copy Async, Stages, Pad, Block swizzle) 
- [X] hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages(WMMA, Tile MMA/Warp, Copy Async, Stages, Pad, Block swizzle)
- [X] hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages(WMMA, Tile MMA/Warp, Copy Async, Stages, Pad, Block swizzle) 
- [X] hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async(WMMA, Tile MMA/Warp, Copy Async, Double Buffers, Pad)
- [X] PyTorch bindings

## 目前性能  

- NVIDIA L20  

目前最优的实现，在L20上（理论Tensor Cores FP16算力为 119.5 TFLOPS），能达到cuBLAS大概95%~98%左右的性能(105-113 TFLOPS vs 105-115 TFLOPS)，部分case会超越cuBLAS。已知问题为bank conflicts没有完全消除，目前通过padding的方式缓解bank conflicts会导致shared memory浪费，也会影响SM occupancy。并且尚未手工实现Warp swizzle(受限于WMMA API的灵活性以及本人的能力)，后续将会尝试通过MMA PTX实现warp swizzle，[点击查看性能数据](#NV-L20)。

- NVIDIA GeForce RTX 3080 Laptop   

在NVIDIA GeForce RTX 3080 Laptop上测试，使用mma4x4_warp4x4（16 MMA m16n16k16 ops, warp tile 64x64）以及Thread block swizzle，大部分case能持平甚至超过cuBLAS，[点击查看性能数据](#NV-RTX-3080)。

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

## 测试命令

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长: Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada 
python3 hgemm.py # default, test some wmma kernels for all MNK
python3 hgemm.py --wmma # test all wmma kernels for all MNK
python3 hgemm.py --M 16384 --N 16384 --K 8192 --i 10 --wmma # test all wmma kernels for specific MNK
python3 hgemm.py --wmma --no-default # test all wmma kernels, but exclude the default part.
```


## NVIDIA L20 
<div id="NV-L20"></div>

Up to 113.76 TFLOPS, 113.76/119.5=95.19% TFLOPS utilization.

```bash
python3 hgemm.py
```
输出：
```bash
----------------------------------------------------------------------------------------------------------------------------------
                                        M=4096, N=4096, K=2048, Warmup=5, Iters=20, 1/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['14.765625 ', '-18.640625'], time:1.425385ms, swizzle: NOOP, TFLOPS: 48.21 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['14.765625 ', '-18.640625'], time:1.331329ms, swizzle: NOOP, TFLOPS: 51.62 (+7.06%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['14.28125  ', '-18.6875  '], time:1.493430ms, swizzle: NOOP, TFLOPS: 46.01
                        (mma4x2+warp2x4): ['14.28125  ', '-18.6875  '], time:0.943207ms, swizzle: NOOP, TFLOPS: 72.86 (+41.15%)
                 (mma4x2+warp2x4+stage3): ['14.28125  ', '-18.6875  '], time:0.700759ms, swizzle: NOOP, TFLOPS: 98.06 (+34.60%)
                 (mma4x2+warp2x4+stage2): ['14.28125  ', '-18.6875  '], time:0.694894ms, swizzle: NOOP, TFLOPS: 98.89 (+0.84%)
           (mma4x2+warp2x4+stage3+dsmem): ['14.28125  ', '-18.6875  '], time:0.694680ms, swizzle: NOOP, TFLOPS: 98.92 (+0.03%)
           (mma4x2+warp2x4+stage2+dsmem): ['14.28125  ', '-18.6875  '], time:0.699853ms, swizzle: NOOP, TFLOPS: 98.19
         (mma4x2+warp2x4+stage3+swizzle): ['14.28125  ', '-18.6875  '], time:0.696039ms, swizzle: 1024, TFLOPS: 98.73
         (mma4x2+warp2x4+stage2+swizzle): ['14.28125  ', '-18.6875  '], time:0.684642ms, swizzle: 1024, TFLOPS: 100.37(+1.47%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['14.28125  ', '-18.6875  '], time:0.698900ms, swizzle: 1024, TFLOPS: 98.33
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['14.28125  ', '-18.6875  '], time:0.685286ms, swizzle: 1024, TFLOPS: 100.28
                                (cublas): ['14.28125  ', '-18.6875  '], time:0.840950ms, swizzle: NOOP, TFLOPS: 81.72
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=4096, N=4096, K=4096, Warmup=5, Iters=20, 2/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['10.296875 ', '-46.6875  '], time:2.834367ms, swizzle: NOOP, TFLOPS: 48.49 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['10.296875 ', '-46.6875  '], time:2.637004ms, swizzle: NOOP, TFLOPS: 52.12 (+7.48%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['9.84375   ', '-46.71875 '], time:2.951836ms, swizzle: NOOP, TFLOPS: 46.56
                        (mma4x2+warp2x4): ['9.84375   ', '-46.71875 '], time:1.830053ms, swizzle: NOOP, TFLOPS: 75.10 (+44.09%)
                 (mma4x2+warp2x4+stage3): ['9.84375   ', '-46.71875 '], time:1.363086ms, swizzle: NOOP, TFLOPS: 100.83(+34.26%)
                 (mma4x2+warp2x4+stage2): ['9.84375   ', '-46.71875 '], time:1.352930ms, swizzle: NOOP, TFLOPS: 101.59(+0.75%)
           (mma4x2+warp2x4+stage3+dsmem): ['9.84375   ', '-46.71875 '], time:1.352334ms, swizzle: NOOP, TFLOPS: 101.63(+0.04%)
           (mma4x2+warp2x4+stage2+dsmem): ['9.84375   ', '-46.71875 '], time:1.352477ms, swizzle: NOOP, TFLOPS: 101.62
         (mma4x2+warp2x4+stage3+swizzle): ['9.84375   ', '-46.71875 '], time:1.355242ms, swizzle: 1024, TFLOPS: 101.41
         (mma4x2+warp2x4+stage2+swizzle): ['9.84375   ', '-46.71875 '], time:1.449298ms, swizzle: 1024, TFLOPS: 94.83
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['9.84375   ', '-46.71875 '], time:1.359033ms, swizzle: 1024, TFLOPS: 101.13
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['9.84375   ', '-46.71875 '], time:1.333761ms, swizzle: 1024, TFLOPS: 103.05(+1.39%)
                                (cublas): ['9.84375   ', '-46.71875 '], time:1.489806ms, swizzle: NOOP, TFLOPS: 92.25
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=4096, N=4096, K=8192, Warmup=5, Iters=20, 3/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['47.53125  ', '-51.5     '], time:5.691790ms, swizzle: NOOP, TFLOPS: 48.29 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['47.53125  ', '-51.5     '], time:5.279827ms, swizzle: NOOP, TFLOPS: 52.06 (+7.80%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['47.0      ', '-52.25    '], time:5.903649ms, swizzle: NOOP, TFLOPS: 46.56
                        (mma4x2+warp2x4): ['47.0      ', '-52.25    '], time:3.659152ms, swizzle: NOOP, TFLOPS: 75.12 (+44.29%)
                 (mma4x2+warp2x4+stage3): ['47.0      ', '-52.25    '], time:2.691316ms, swizzle: NOOP, TFLOPS: 102.14(+35.96%)
                 (mma4x2+warp2x4+stage2): ['47.0      ', '-52.25    '], time:2.671480ms, swizzle: NOOP, TFLOPS: 102.89(+0.74%)
           (mma4x2+warp2x4+stage3+dsmem): ['47.0      ', '-52.25    '], time:2.669262ms, swizzle: NOOP, TFLOPS: 102.98(+0.08%)
           (mma4x2+warp2x4+stage2+dsmem): ['47.0      ', '-52.25    '], time:2.671861ms, swizzle: NOOP, TFLOPS: 102.88
         (mma4x2+warp2x4+stage3+swizzle): ['47.0      ', '-52.25    '], time:2.674126ms, swizzle: 1024, TFLOPS: 102.79
         (mma4x2+warp2x4+stage2+swizzle): ['47.0      ', '-52.25    '], time:2.632570ms, swizzle: 1024, TFLOPS: 104.41(+1.39%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['47.0      ', '-52.25    '], time:2.682542ms, swizzle: 1024, TFLOPS: 102.47
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['47.0      ', '-52.25    '], time:2.632832ms, swizzle: 1024, TFLOPS: 104.40
                                (cublas): ['47.09375  ', '-51.65625 '], time:2.653670ms, swizzle: NOOP, TFLOPS: 103.58
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=4096, N=8192, K=2048, Warmup=5, Iters=20, 4/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['14.765625 ', '-18.640625'], time:2.700662ms, swizzle: NOOP, TFLOPS: 50.89 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['14.765625 ', '-18.640625'], time:2.537584ms, swizzle: NOOP, TFLOPS: 54.16 (+6.43%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['14.28125  ', '-18.6875  '], time:2.964186ms, swizzle: NOOP, TFLOPS: 46.37
                        (mma4x2+warp2x4): ['14.28125  ', '-18.6875  '], time:1.854801ms, swizzle: NOOP, TFLOPS: 74.10 (+36.81%)
                 (mma4x2+warp2x4+stage3): ['14.28125  ', '-18.6875  '], time:1.317334ms, swizzle: NOOP, TFLOPS: 104.33(+40.80%)
                 (mma4x2+warp2x4+stage2): ['14.28125  ', '-18.6875  '], time:1.308989ms, swizzle: NOOP, TFLOPS: 105.00(+0.64%)
           (mma4x2+warp2x4+stage3+dsmem): ['14.28125  ', '-18.6875  '], time:1.308083ms, swizzle: NOOP, TFLOPS: 105.07(+0.07%)
           (mma4x2+warp2x4+stage2+dsmem): ['14.28125  ', '-18.6875  '], time:1.309251ms, swizzle: NOOP, TFLOPS: 104.98
         (mma4x2+warp2x4+stage3+swizzle): ['14.28125  ', '-18.6875  '], time:1.309061ms, swizzle: 2048, TFLOPS: 104.99
         (mma4x2+warp2x4+stage2+swizzle): ['14.28125  ', '-18.6875  '], time:1.293468ms, swizzle: 2048, TFLOPS: 106.26(+1.13%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['14.28125  ', '-18.6875  '], time:1.314473ms, swizzle: 2048, TFLOPS: 104.56
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['14.28125  ', '-18.6875  '], time:1.358604ms, swizzle: 2048, TFLOPS: 101.16
                                (cublas): ['14.28125  ', '-18.6875  '], time:1.459145ms, swizzle: NOOP, TFLOPS: 94.19
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=4096, N=8192, K=4096, Warmup=5, Iters=20, 5/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['10.296875 ', '-46.6875  '], time:5.430340ms, swizzle: NOOP, TFLOPS: 50.62 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['10.296875 ', '-46.6875  '], time:5.125904ms, swizzle: NOOP, TFLOPS: 53.63 (+5.94%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['9.84375   ', '-46.71875 '], time:5.883288ms, swizzle: NOOP, TFLOPS: 46.72
                        (mma4x2+warp2x4): ['9.84375   ', '-46.71875 '], time:3.645515ms, swizzle: NOOP, TFLOPS: 75.40 (+40.61%)
                 (mma4x2+warp2x4+stage3): ['9.84375   ', '-46.71875 '], time:2.588868ms, swizzle: NOOP, TFLOPS: 106.18(+40.82%)
                 (mma4x2+warp2x4+stage2): ['9.84375   ', '-46.71875 '], time:2.570867ms, swizzle: NOOP, TFLOPS: 106.92(+0.70%)
           (mma4x2+warp2x4+stage3+dsmem): ['9.84375   ', '-46.71875 '], time:2.572536ms, swizzle: NOOP, TFLOPS: 106.85
           (mma4x2+warp2x4+stage2+dsmem): ['9.84375   ', '-46.71875 '], time:2.571439ms, swizzle: NOOP, TFLOPS: 106.90
         (mma4x2+warp2x4+stage3+swizzle): ['9.84375   ', '-46.71875 '], time:2.570629ms, swizzle: 2048, TFLOPS: 106.93(+0.01%)
         (mma4x2+warp2x4+stage2+swizzle): ['9.84375   ', '-46.71875 '], time:2.531552ms, swizzle: 2048, TFLOPS: 108.58(+1.54%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['9.84375   ', '-46.71875 '], time:2.573418ms, swizzle: 2048, TFLOPS: 106.81
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['9.84375   ', '-46.71875 '], time:2.533483ms, swizzle: 2048, TFLOPS: 108.50
                                (cublas): ['9.84375   ', '-46.71875 '], time:2.661132ms, swizzle: NOOP, TFLOPS: 103.29
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=4096, N=8192, K=8192, Warmup=5, Iters=20, 6/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['47.53125  ', '-51.5     '], time:11.79177ms, swizzle: NOOP, TFLOPS: 46.62 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['47.53125  ', '-51.5     '], time:11.25807ms, swizzle: NOOP, TFLOPS: 48.83 (+4.74%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['47.0      ', '-52.25    '], time:12.28225ms, swizzle: NOOP, TFLOPS: 44.76
                        (mma4x2+warp2x4): ['47.0      ', '-52.25    '], time:7.306694ms, swizzle: NOOP, TFLOPS: 75.24 (+54.08%)
                 (mma4x2+warp2x4+stage3): ['47.0      ', '-52.25    '], time:5.185413ms, swizzle: NOOP, TFLOPS: 106.02(+40.91%)
                 (mma4x2+warp2x4+stage2): ['47.0      ', '-52.25    '], time:5.128622ms, swizzle: NOOP, TFLOPS: 107.19(+1.11%)
           (mma4x2+warp2x4+stage3+dsmem): ['47.0      ', '-52.25    '], time:5.165719ms, swizzle: NOOP, TFLOPS: 106.42
           (mma4x2+warp2x4+stage2+dsmem): ['47.0      ', '-52.25    '], time:5.137014ms, swizzle: NOOP, TFLOPS: 107.02
         (mma4x2+warp2x4+stage3+swizzle): ['47.0      ', '-52.25    '], time:5.096411ms, swizzle: 2048, TFLOPS: 107.87(+0.63%)
         (mma4x2+warp2x4+stage2+swizzle): ['47.0      ', '-52.25    '], time:5.036878ms, swizzle: 2048, TFLOPS: 109.15(+1.18%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['47.0      ', '-52.25    '], time:5.087852ms, swizzle: 2048, TFLOPS: 108.05
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['47.0      ', '-52.25    '], time:5.011391ms, swizzle: 2048, TFLOPS: 109.70(+0.51%)
                                (cublas): ['47.0      ', '-52.25    '], time:5.063843ms, swizzle: NOOP, TFLOPS: 108.56
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=4096, N=16384, K=2048, Warmup=5, Iters=20, 7/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['14.765625 ', '-18.640625'], time:5.306124ms, swizzle: NOOP, TFLOPS: 51.80 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['14.765625 ', '-18.640625'], time:5.044364ms, swizzle: NOOP, TFLOPS: 54.49 (+5.19%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['14.28125  ', '-18.6875  '], time:5.916452ms, swizzle: NOOP, TFLOPS: 46.46
                        (mma4x2+warp2x4): ['14.28125  ', '-18.6875  '], time:3.550720ms, swizzle: NOOP, TFLOPS: 77.41 (+42.07%)
                 (mma4x2+warp2x4+stage3): ['14.28125  ', '-18.6875  '], time:2.552175ms, swizzle: NOOP, TFLOPS: 107.70(+39.13%)
                 (mma4x2+warp2x4+stage2): ['14.28125  ', '-18.6875  '], time:2.537274ms, swizzle: NOOP, TFLOPS: 108.34(+0.59%)
           (mma4x2+warp2x4+stage3+dsmem): ['14.28125  ', '-18.6875  '], time:2.545833ms, swizzle: NOOP, TFLOPS: 107.97
           (mma4x2+warp2x4+stage2+dsmem): ['14.28125  ', '-18.6875  '], time:2.546501ms, swizzle: NOOP, TFLOPS: 107.94
         (mma4x2+warp2x4+stage3+swizzle): ['14.28125  ', '-18.6875  '], time:2.544927ms, swizzle: 4096, TFLOPS: 108.01
         (mma4x2+warp2x4+stage2+swizzle): ['14.28125  ', '-18.6875  '], time:2.518939ms, swizzle: 4096, TFLOPS: 109.12(+0.73%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['14.28125  ', '-18.6875  '], time:2.547931ms, swizzle: 4096, TFLOPS: 107.88
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['14.28125  ', '-18.6875  '], time:2.512478ms, swizzle: 4096, TFLOPS: 109.41(+0.26%)
                                (cublas): ['14.28125  ', '-18.6875  '], time:2.635645ms, swizzle: NOOP, TFLOPS: 104.29
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=4096, N=16384, K=4096, Warmup=5, Iters=20, 8/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['10.296875 ', '-46.6875  '], time:11.61146ms, swizzle: NOOP, TFLOPS: 47.35 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['10.296875 ', '-46.6875  '], time:11.02995ms, swizzle: NOOP, TFLOPS: 49.84 (+5.27%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['9.84375   ', '-46.71875 '], time:15.55149ms, swizzle: NOOP, TFLOPS: 35.35
                        (mma4x2+warp2x4): ['9.84375   ', '-46.71875 '], time:7.264566ms, swizzle: NOOP, TFLOPS: 75.68 (+51.83%)
                 (mma4x2+warp2x4+stage3): ['9.84375   ', '-46.71875 '], time:5.160856ms, swizzle: NOOP, TFLOPS: 106.52(+40.76%)
                 (mma4x2+warp2x4+stage2): ['9.84375   ', '-46.71875 '], time:5.038166ms, swizzle: NOOP, TFLOPS: 109.12(+2.44%)
           (mma4x2+warp2x4+stage3+dsmem): ['9.84375   ', '-46.71875 '], time:5.177164ms, swizzle: NOOP, TFLOPS: 106.19
           (mma4x2+warp2x4+stage2+dsmem): ['9.84375   ', '-46.71875 '], time:5.098938ms, swizzle: NOOP, TFLOPS: 107.82
         (mma4x2+warp2x4+stage3+swizzle): ['9.84375   ', '-46.71875 '], time:5.004787ms, swizzle: 4096, TFLOPS: 109.85(+0.67%)
         (mma4x2+warp2x4+stage2+swizzle): ['9.84375   ', '-46.71875 '], time:4.954004ms, swizzle: 4096, TFLOPS: 110.97(+1.03%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['9.84375   ', '-46.71875 '], time:5.003094ms, swizzle: 4096, TFLOPS: 109.88
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['9.84375   ', '-46.71875 '], time:4.933691ms, swizzle: 4096, TFLOPS: 111.43(+0.41%)
                                (cublas): ['9.84375   ', '-46.71875 '], time:4.990887ms, swizzle: NOOP, TFLOPS: 110.15
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=4096, N=16384, K=8192, Warmup=5, Iters=20, 9/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['47.53125  ', '-51.5     '], time:24.35543ms, swizzle: NOOP, TFLOPS: 45.14 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['47.53125  ', '-51.5     '], time:23.57738ms, swizzle: NOOP, TFLOPS: 46.63 (+3.30%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['47.0      ', '-52.25    '], time:31.01222ms, swizzle: NOOP, TFLOPS: 35.45
                        (mma4x2+warp2x4): ['47.0      ', '-52.25    '], time:14.37473ms, swizzle: NOOP, TFLOPS: 76.49 (+64.02%)
                 (mma4x2+warp2x4+stage3): ['47.0      ', '-52.25    '], time:12.40768ms, swizzle: NOOP, TFLOPS: 88.62 (+15.85%)
                 (mma4x2+warp2x4+stage2): ['47.0      ', '-52.25    '], time:12.25883ms, swizzle: NOOP, TFLOPS: 89.69 (+1.21%)
           (mma4x2+warp2x4+stage3+dsmem): ['47.0      ', '-52.25    '], time:12.40663ms, swizzle: NOOP, TFLOPS: 88.62
           (mma4x2+warp2x4+stage2+dsmem): ['47.0      ', '-52.25    '], time:12.26737ms, swizzle: NOOP, TFLOPS: 89.63
         (mma4x2+warp2x4+stage3+swizzle): ['47.0      ', '-52.25    '], time:9.920740ms, swizzle: 4096, TFLOPS: 110.83(+23.57%)
         (mma4x2+warp2x4+stage2+swizzle): ['47.0      ', '-52.25    '], time:9.804654ms, swizzle: 4096, TFLOPS: 112.14(+1.18%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['47.0      ', '-52.25    '], time:9.917545ms, swizzle: 4096, TFLOPS: 110.87
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['47.0      ', '-52.25    '], time:9.778022ms, swizzle: 4096, TFLOPS: 112.45(+0.27%)
                                (cublas): ['47.0      ', '-52.25    '], time:9.679126ms, swizzle: NOOP, TFLOPS: 113.60(+1.02%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=8192, N=4096, K=2048, Warmup=5, Iters=20, 10/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['14.765625 ', '-18.640625'], time:2.697157ms, swizzle: NOOP, TFLOPS: 50.96 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['14.765625 ', '-18.640625'], time:2.536106ms, swizzle: NOOP, TFLOPS: 54.19 (+6.35%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['14.28125  ', '-18.6875  '], time:2.964568ms, swizzle: NOOP, TFLOPS: 46.36
                        (mma4x2+warp2x4): ['14.28125  ', '-18.6875  '], time:1.834273ms, swizzle: NOOP, TFLOPS: 74.93 (+38.26%)
                 (mma4x2+warp2x4+stage3): ['14.28125  ', '-18.6875  '], time:1.318478ms, swizzle: NOOP, TFLOPS: 104.24(+39.12%)
                 (mma4x2+warp2x4+stage2): ['14.28125  ', '-18.6875  '], time:1.309275ms, swizzle: NOOP, TFLOPS: 104.97(+0.70%)
           (mma4x2+warp2x4+stage3+dsmem): ['14.28125  ', '-18.6875  '], time:1.308512ms, swizzle: NOOP, TFLOPS: 105.03(+0.06%)
           (mma4x2+warp2x4+stage2+dsmem): ['14.28125  ', '-18.6875  '], time:1.310014ms, swizzle: NOOP, TFLOPS: 104.91
         (mma4x2+warp2x4+stage3+swizzle): ['14.28125  ', '-18.6875  '], time:1.308369ms, swizzle: 1024, TFLOPS: 105.05(+0.01%)
         (mma4x2+warp2x4+stage2+swizzle): ['14.28125  ', '-18.6875  '], time:1.292586ms, swizzle: 1024, TFLOPS: 106.33(+1.22%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['14.28125  ', '-18.6875  '], time:1.313900ms, swizzle: 1024, TFLOPS: 104.60
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['14.28125  ', '-18.6875  '], time:1.293778ms, swizzle: 1024, TFLOPS: 106.23
                                (cublas): ['14.28125  ', '-18.6875  '], time:1.471805ms, swizzle: NOOP, TFLOPS: 93.38
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=8192, N=4096, K=4096, Warmup=5, Iters=20, 11/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['10.296875 ', '-46.6875  '], time:5.442857ms, swizzle: NOOP, TFLOPS: 50.50 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['10.296875 ', '-46.6875  '], time:5.149674ms, swizzle: NOOP, TFLOPS: 53.38 (+5.69%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['9.84375   ', '-46.71875 '], time:5.882453ms, swizzle: NOOP, TFLOPS: 46.73
                        (mma4x2+warp2x4): ['9.84375   ', '-46.71875 '], time:3.618168ms, swizzle: NOOP, TFLOPS: 75.97 (+42.33%)
                 (mma4x2+warp2x4+stage3): ['9.84375   ', '-46.71875 '], time:2.574682ms, swizzle: NOOP, TFLOPS: 106.76(+40.53%)
                 (mma4x2+warp2x4+stage2): ['9.84375   ', '-46.71875 '], time:2.565002ms, swizzle: NOOP, TFLOPS: 107.16(+0.38%)
           (mma4x2+warp2x4+stage3+dsmem): ['9.84375   ', '-46.71875 '], time:2.564716ms, swizzle: NOOP, TFLOPS: 107.18(+0.01%)
           (mma4x2+warp2x4+stage2+dsmem): ['9.84375   ', '-46.71875 '], time:2.564477ms, swizzle: NOOP, TFLOPS: 107.19(+0.01%)
         (mma4x2+warp2x4+stage3+swizzle): ['9.84375   ', '-46.71875 '], time:2.564001ms, swizzle: 1024, TFLOPS: 107.21(+0.02%)
         (mma4x2+warp2x4+stage2+swizzle): ['9.84375   ', '-46.71875 '], time:2.531504ms, swizzle: 1024, TFLOPS: 108.58(+1.28%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['9.84375   ', '-46.71875 '], time:2.574038ms, swizzle: 1024, TFLOPS: 106.79
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['9.84375   ', '-46.71875 '], time:2.533769ms, swizzle: 1024, TFLOPS: 108.49
                                (cublas): ['9.84375   ', '-46.71875 '], time:2.670454ms, swizzle: NOOP, TFLOPS: 102.93
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=8192, N=4096, K=8192, Warmup=5, Iters=20, 12/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['47.53125  ', '-51.5     '], time:11.15067ms, swizzle: NOOP, TFLOPS: 49.30 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['47.53125  ', '-51.5     '], time:10.48223ms, swizzle: NOOP, TFLOPS: 52.45 (+6.38%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['47.0      ', '-52.25    '], time:11.71836ms, swizzle: NOOP, TFLOPS: 46.91
                        (mma4x2+warp2x4): ['47.0      ', '-52.25    '], time:7.112240ms, swizzle: NOOP, TFLOPS: 77.30 (+47.38%)
                 (mma4x2+warp2x4+stage3): ['47.0      ', '-52.25    '], time:5.119061ms, swizzle: NOOP, TFLOPS: 107.39(+38.94%)
                 (mma4x2+warp2x4+stage2): ['47.0      ', '-52.25    '], time:5.075407ms, swizzle: NOOP, TFLOPS: 108.32(+0.86%)
           (mma4x2+warp2x4+stage3+dsmem): ['47.0      ', '-52.25    '], time:5.083894ms, swizzle: NOOP, TFLOPS: 108.14
           (mma4x2+warp2x4+stage2+dsmem): ['47.0      ', '-52.25    '], time:5.075025ms, swizzle: NOOP, TFLOPS: 108.33(+0.01%)
         (mma4x2+warp2x4+stage3+swizzle): ['47.0      ', '-52.25    '], time:5.082964ms, swizzle: 1024, TFLOPS: 108.16
         (mma4x2+warp2x4+stage2+swizzle): ['47.0      ', '-52.25    '], time:5.004644ms, swizzle: 1024, TFLOPS: 109.85(+1.41%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['47.0      ', '-52.25    '], time:5.098199ms, swizzle: 1024, TFLOPS: 107.83
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['47.0      ', '-52.25    '], time:5.003476ms, swizzle: 1024, TFLOPS: 109.87(+0.02%)
                                (cublas): ['47.0      ', '-52.25    '], time:5.096864ms, swizzle: NOOP, TFLOPS: 107.86
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=8192, N=8192, K=2048, Warmup=5, Iters=20, 13/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['14.765625 ', '-18.640625'], time:5.346417ms, swizzle: NOOP, TFLOPS: 51.41 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['14.765625 ', '-18.640625'], time:4.942703ms, swizzle: NOOP, TFLOPS: 55.61 (+8.17%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['14.28125  ', '-18.6875  '], time:5.900359ms, swizzle: NOOP, TFLOPS: 46.59
                        (mma4x2+warp2x4): ['14.28125  ', '-18.6875  '], time:3.572225ms, swizzle: NOOP, TFLOPS: 76.95 (+38.36%)
                 (mma4x2+warp2x4+stage3): ['14.28125  ', '-18.6875  '], time:2.547502ms, swizzle: NOOP, TFLOPS: 107.90(+40.22%)
                 (mma4x2+warp2x4+stage2): ['14.28125  ', '-18.6875  '], time:2.539443ms, swizzle: NOOP, TFLOPS: 108.24(+0.32%)
           (mma4x2+warp2x4+stage3+dsmem): ['14.28125  ', '-18.6875  '], time:2.537584ms, swizzle: NOOP, TFLOPS: 108.32(+0.07%)
           (mma4x2+warp2x4+stage2+dsmem): ['14.28125  ', '-18.6875  '], time:2.540159ms, swizzle: NOOP, TFLOPS: 108.21
         (mma4x2+warp2x4+stage3+swizzle): ['14.28125  ', '-18.6875  '], time:2.535915ms, swizzle: 2048, TFLOPS: 108.39(+0.07%)
         (mma4x2+warp2x4+stage2+swizzle): ['14.28125  ', '-18.6875  '], time:2.510333ms, swizzle: 2048, TFLOPS: 109.50(+1.02%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['14.28125  ', '-18.6875  '], time:2.547550ms, swizzle: 2048, TFLOPS: 107.90
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['14.28125  ', '-18.6875  '], time:2.511882ms, swizzle: 2048, TFLOPS: 109.43
                                (cublas): ['14.28125  ', '-18.6875  '], time:2.635979ms, swizzle: NOOP, TFLOPS: 104.28
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=8192, N=8192, K=4096, Warmup=5, Iters=20, 14/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['10.296875 ', '-46.6875  '], time:10.91315ms, swizzle: NOOP, TFLOPS: 50.38 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['10.296875 ', '-46.6875  '], time:10.32221ms, swizzle: NOOP, TFLOPS: 53.26 (+5.72%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['9.84375   ', '-46.71875 '], time:11.72120ms, swizzle: NOOP, TFLOPS: 46.90
                        (mma4x2+warp2x4): ['9.84375   ', '-46.71875 '], time:6.984162ms, swizzle: NOOP, TFLOPS: 78.71 (+47.79%)
                 (mma4x2+warp2x4+stage3): ['9.84375   ', '-46.71875 '], time:5.013513ms, swizzle: NOOP, TFLOPS: 109.65(+39.31%)
                 (mma4x2+warp2x4+stage2): ['9.84375   ', '-46.71875 '], time:4.993677ms, swizzle: NOOP, TFLOPS: 110.09(+0.40%)
           (mma4x2+warp2x4+stage3+dsmem): ['9.84375   ', '-46.71875 '], time:4.979777ms, swizzle: NOOP, TFLOPS: 110.40(+0.28%)
           (mma4x2+warp2x4+stage2+dsmem): ['9.84375   ', '-46.71875 '], time:5.007362ms, swizzle: NOOP, TFLOPS: 109.79
         (mma4x2+warp2x4+stage3+swizzle): ['9.84375   ', '-46.71875 '], time:4.975485ms, swizzle: 2048, TFLOPS: 110.49(+0.09%)
         (mma4x2+warp2x4+stage2+swizzle): ['9.84375   ', '-46.71875 '], time:4.935383ms, swizzle: 2048, TFLOPS: 111.39(+0.81%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['9.84375   ', '-46.71875 '], time:4.990983ms, swizzle: 2048, TFLOPS: 110.15
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['9.84375   ', '-46.71875 '], time:4.953265ms, swizzle: 2048, TFLOPS: 110.99
                                (cublas): ['9.84375   ', '-46.71875 '], time:4.983496ms, swizzle: NOOP, TFLOPS: 110.32
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=8192, N=8192, K=8192, Warmup=5, Iters=20, 15/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['47.53125  ', '-51.5     '], time:23.78494ms, swizzle: NOOP, TFLOPS: 46.23 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['47.53125  ', '-51.5     '], time:22.85294ms, swizzle: NOOP, TFLOPS: 48.11 (+4.08%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['47.0      ', '-52.25    '], time:24.70605ms, swizzle: NOOP, TFLOPS: 44.50
                        (mma4x2+warp2x4): ['47.0      ', '-52.25    '], time:14.21954ms, swizzle: NOOP, TFLOPS: 77.32 (+60.72%)
                 (mma4x2+warp2x4+stage3): ['47.0      ', '-52.25    '], time:10.34536ms, swizzle: NOOP, TFLOPS: 106.28(+37.45%)
                 (mma4x2+warp2x4+stage2): ['47.0      ', '-52.25    '], time:10.25786ms, swizzle: NOOP, TFLOPS: 107.19(+0.85%)
           (mma4x2+warp2x4+stage3+dsmem): ['47.0      ', '-52.25    '], time:10.49890ms, swizzle: NOOP, TFLOPS: 104.73
           (mma4x2+warp2x4+stage2+dsmem): ['47.0      ', '-52.25    '], time:10.29896ms, swizzle: NOOP, TFLOPS: 106.76
         (mma4x2+warp2x4+stage3+swizzle): ['47.0      ', '-52.25    '], time:9.953498ms, swizzle: 2048, TFLOPS: 110.46(+3.06%)
         (mma4x2+warp2x4+stage2+swizzle): ['47.0      ', '-52.25    '], time:9.775471ms, swizzle: 2048, TFLOPS: 112.48(+1.82%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['47.0      ', '-52.25    '], time:9.905838ms, swizzle: 2048, TFLOPS: 111.00
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['47.0      ', '-52.25    '], time:9.768342ms, swizzle: 2048, TFLOPS: 112.56(+0.07%)
                                (cublas): ['47.0      ', '-52.25    '], time:9.739327ms, swizzle: NOOP, TFLOPS: 112.89(+0.30%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=8192, N=16384, K=2048, Warmup=5, Iters=20, 16/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['14.765625 ', '-18.640625'], time:10.92975ms, swizzle: NOOP, TFLOPS: 50.30 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['14.765625 ', '-18.640625'], time:10.32440ms, swizzle: NOOP, TFLOPS: 53.25 (+5.86%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['14.28125  ', '-18.6875  '], time:11.78483ms, swizzle: NOOP, TFLOPS: 46.65
                        (mma4x2+warp2x4): ['14.28125  ', '-18.6875  '], time:6.915855ms, swizzle: NOOP, TFLOPS: 79.49 (+49.29%)
                 (mma4x2+warp2x4+stage3): ['14.28125  ', '-18.6875  '], time:5.065703ms, swizzle: NOOP, TFLOPS: 108.53(+36.52%)
                 (mma4x2+warp2x4+stage2): ['14.28125  ', '-18.6875  '], time:5.030226ms, swizzle: NOOP, TFLOPS: 109.29(+0.71%)
           (mma4x2+warp2x4+stage3+dsmem): ['14.28125  ', '-18.6875  '], time:5.033874ms, swizzle: NOOP, TFLOPS: 109.21
           (mma4x2+warp2x4+stage2+dsmem): ['14.28125  ', '-18.6875  '], time:5.028772ms, swizzle: NOOP, TFLOPS: 109.32(+0.03%)
         (mma4x2+warp2x4+stage3+swizzle): ['14.28125  ', '-18.6875  '], time:5.024838ms, swizzle: 4096, TFLOPS: 109.41(+0.08%)
         (mma4x2+warp2x4+stage2+swizzle): ['14.28125  ', '-18.6875  '], time:4.977488ms, swizzle: 4096, TFLOPS: 110.45(+0.95%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['14.28125  ', '-18.6875  '], time:5.048298ms, swizzle: 4096, TFLOPS: 108.90
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['14.28125  ', '-18.6875  '], time:4.978847ms, swizzle: 4096, TFLOPS: 110.42
                                (cublas): ['14.28125  ', '-18.6875  '], time:5.005145ms, swizzle: NOOP, TFLOPS: 109.84
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=8192, N=16384, K=4096, Warmup=5, Iters=20, 17/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['10.296875 ', '-46.6875  '], time:23.64189ms, swizzle: NOOP, TFLOPS: 46.51 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['10.296875 ', '-46.6875  '], time:22.93310ms, swizzle: NOOP, TFLOPS: 47.94 (+3.09%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['9.84375   ', '-46.71875 '], time:31.19268ms, swizzle: NOOP, TFLOPS: 35.25
                        (mma4x2+warp2x4): ['9.84375   ', '-46.71875 '], time:14.21802ms, swizzle: NOOP, TFLOPS: 77.33 (+61.30%)
                 (mma4x2+warp2x4+stage3): ['9.84375   ', '-46.71875 '], time:10.72919ms, swizzle: NOOP, TFLOPS: 102.48(+32.52%)
                 (mma4x2+warp2x4+stage2): ['9.84375   ', '-46.71875 '], time:10.53795ms, swizzle: NOOP, TFLOPS: 104.34(+1.81%)
           (mma4x2+warp2x4+stage3+dsmem): ['9.84375   ', '-46.71875 '], time:10.60345ms, swizzle: NOOP, TFLOPS: 103.69
           (mma4x2+warp2x4+stage2+dsmem): ['9.84375   ', '-46.71875 '], time:10.46254ms, swizzle: NOOP, TFLOPS: 105.09(+0.72%)
         (mma4x2+warp2x4+stage3+swizzle): ['9.84375   ', '-46.71875 '], time:9.963369ms, swizzle: 4096, TFLOPS: 110.36(+5.01%)
         (mma4x2+warp2x4+stage2+swizzle): ['9.84375   ', '-46.71875 '], time:9.808254ms, swizzle: 4096, TFLOPS: 112.10(+1.58%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['9.84375   ', '-46.71875 '], time:9.931588ms, swizzle: 4096, TFLOPS: 110.71
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['9.84375   ', '-46.71875 '], time:9.800553ms, swizzle: 4096, TFLOPS: 112.19(+0.08%)
                                (cublas): ['9.84375   ', '-46.71875 '], time:9.695315ms, swizzle: NOOP, TFLOPS: 113.41(+1.09%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=8192, N=16384, K=8192, Warmup=5, Iters=20, 18/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['47.53125  ', '-51.5     '], time:49.22699ms, swizzle: NOOP, TFLOPS: 44.67 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['47.53125  ', '-51.5     '], time:48.20067ms, swizzle: NOOP, TFLOPS: 45.62 (+2.13%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['47.0      ', '-52.25    '], time:61.79366ms, swizzle: NOOP, TFLOPS: 35.59
                        (mma4x2+warp2x4): ['47.0      ', '-52.25    '], time:28.10072ms, swizzle: NOOP, TFLOPS: 78.26 (+71.53%)
                 (mma4x2+warp2x4+stage3): ['47.0      ', '-52.25    '], time:24.79410ms, swizzle: NOOP, TFLOPS: 88.69 (+13.34%)
                 (mma4x2+warp2x4+stage2): ['47.0      ', '-52.25    '], time:24.75156ms, swizzle: NOOP, TFLOPS: 88.84 (+0.17%)
           (mma4x2+warp2x4+stage3+dsmem): ['47.0      ', '-52.25    '], time:24.81336ms, swizzle: NOOP, TFLOPS: 88.62
           (mma4x2+warp2x4+stage2+dsmem): ['47.0      ', '-52.25    '], time:24.72374ms, swizzle: NOOP, TFLOPS: 88.94 (+0.11%)
         (mma4x2+warp2x4+stage3+swizzle): ['47.0      ', '-52.25    '], time:19.72281ms, swizzle: 4096, TFLOPS: 111.50(+25.36%)
         (mma4x2+warp2x4+stage2+swizzle): ['47.0      ', '-52.25    '], time:19.45309ms, swizzle: 4096, TFLOPS: 113.04(+1.39%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['47.0      ', '-52.25    '], time:19.70534ms, swizzle: 4096, TFLOPS: 111.60
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['47.0      ', '-52.25    '], time:19.46532ms, swizzle: 4096, TFLOPS: 112.97
                                (cublas): ['47.0      ', '-52.25    '], time:19.07067ms, swizzle: NOOP, TFLOPS: 115.31(+2.01%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=16384, N=4096, K=2048, Warmup=5, Iters=20, 19/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['14.765625 ', '-18.640625'], time:5.374526ms, swizzle: NOOP, TFLOPS: 51.14 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['14.765625 ', '-18.640625'], time:4.960513ms, swizzle: NOOP, TFLOPS: 55.41 (+8.35%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['14.28125  ', '-18.6875  '], time:5.899548ms, swizzle: NOOP, TFLOPS: 46.59
                        (mma4x2+warp2x4): ['14.28125  ', '-18.6875  '], time:3.571367ms, swizzle: NOOP, TFLOPS: 76.97 (+38.90%)
                 (mma4x2+warp2x4+stage3): ['14.28125  ', '-18.6875  '], time:2.558302ms, swizzle: NOOP, TFLOPS: 107.45(+39.60%)
                 (mma4x2+warp2x4+stage2): ['14.28125  ', '-18.6875  '], time:2.541303ms, swizzle: NOOP, TFLOPS: 108.16(+0.67%)
           (mma4x2+warp2x4+stage3+dsmem): ['14.28125  ', '-18.6875  '], time:2.538442ms, swizzle: NOOP, TFLOPS: 108.29(+0.11%)
           (mma4x2+warp2x4+stage2+dsmem): ['14.28125  ', '-18.6875  '], time:2.542233ms, swizzle: NOOP, TFLOPS: 108.12
         (mma4x2+warp2x4+stage3+swizzle): ['14.28125  ', '-18.6875  '], time:2.538371ms, swizzle: 1024, TFLOPS: 108.29(+0.00%)
         (mma4x2+warp2x4+stage2+swizzle): ['14.28125  ', '-18.6875  '], time:2.512073ms, swizzle: 1024, TFLOPS: 109.42(+1.05%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['14.28125  ', '-18.6875  '], time:2.551054ms, swizzle: 1024, TFLOPS: 107.75
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['14.28125  ', '-18.6875  '], time:2.513241ms, swizzle: 1024, TFLOPS: 109.37
                                (cublas): ['14.28125  ', '-18.6875  '], time:2.619862ms, swizzle: NOOP, TFLOPS: 104.92
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=16384, N=4096, K=4096, Warmup=5, Iters=20, 20/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['10.296875 ', '-46.6875  '], time:11.02216ms, swizzle: NOOP, TFLOPS: 49.88 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['10.296875 ', '-46.6875  '], time:10.51564ms, swizzle: NOOP, TFLOPS: 52.28 (+4.82%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['9.84375   ', '-46.71875 '], time:11.72006ms, swizzle: NOOP, TFLOPS: 46.91
                        (mma4x2+warp2x4): ['9.84375   ', '-46.71875 '], time:6.976056ms, swizzle: NOOP, TFLOPS: 78.81 (+50.74%)
                 (mma4x2+warp2x4+stage3): ['9.84375   ', '-46.71875 '], time:5.025959ms, swizzle: NOOP, TFLOPS: 109.38(+38.80%)
                 (mma4x2+warp2x4+stage2): ['9.84375   ', '-46.71875 '], time:4.991459ms, swizzle: NOOP, TFLOPS: 110.14(+0.69%)
           (mma4x2+warp2x4+stage3+dsmem): ['9.84375   ', '-46.71875 '], time:4.996275ms, swizzle: NOOP, TFLOPS: 110.03
           (mma4x2+warp2x4+stage2+dsmem): ['9.84375   ', '-46.71875 '], time:4.992103ms, swizzle: NOOP, TFLOPS: 110.13
         (mma4x2+warp2x4+stage3+swizzle): ['9.84375   ', '-46.71875 '], time:4.988074ms, swizzle: 1024, TFLOPS: 110.21(+0.07%)
         (mma4x2+warp2x4+stage2+swizzle): ['9.84375   ', '-46.71875 '], time:4.960775ms, swizzle: 1024, TFLOPS: 110.82(+0.55%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['9.84375   ', '-46.71875 '], time:5.020546ms, swizzle: 1024, TFLOPS: 109.50
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['9.84375   ', '-46.71875 '], time:4.957389ms, swizzle: 1024, TFLOPS: 110.90(+0.07%)
                                (cublas): ['9.84375   ', '-46.71875 '], time:4.962539ms, swizzle: NOOP, TFLOPS: 110.78
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=16384, N=4096, K=8192, Warmup=5, Iters=20, 21/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['47.53125  ', '-51.5     '], time:22.25213ms, swizzle: NOOP, TFLOPS: 49.41 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['47.53125  ', '-51.5     '], time:21.25067ms, swizzle: NOOP, TFLOPS: 51.74 (+4.71%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['47.0      ', '-52.25    '], time:23.32034ms, swizzle: NOOP, TFLOPS: 47.15
                        (mma4x2+warp2x4): ['47.0      ', '-52.25    '], time:13.78231ms, swizzle: NOOP, TFLOPS: 79.78 (+54.19%)
                 (mma4x2+warp2x4+stage3): ['47.0      ', '-52.25    '], time:9.944629ms, swizzle: NOOP, TFLOPS: 110.56(+38.59%)
                 (mma4x2+warp2x4+stage2): ['47.0      ', '-52.25    '], time:9.877133ms, swizzle: NOOP, TFLOPS: 111.32(+0.68%)
           (mma4x2+warp2x4+stage3+dsmem): ['47.0      ', '-52.25    '], time:9.891724ms, swizzle: NOOP, TFLOPS: 111.15
           (mma4x2+warp2x4+stage2+dsmem): ['47.0      ', '-52.25    '], time:9.875774ms, swizzle: NOOP, TFLOPS: 111.33(+0.01%)
         (mma4x2+warp2x4+stage3+swizzle): ['47.0      ', '-52.25    '], time:9.909319ms, swizzle: 1024, TFLOPS: 110.96
         (mma4x2+warp2x4+stage2+swizzle): ['47.0      ', '-52.25    '], time:9.821128ms, swizzle: 1024, TFLOPS: 111.95(+0.56%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['47.0      ', '-52.25    '], time:10.00571ms, swizzle: 1024, TFLOPS: 109.89
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['47.0      ', '-52.25    '], time:9.818959ms, swizzle: 1024, TFLOPS: 111.98(+0.02%)
                                (cublas): ['47.0      ', '-52.25    '], time:9.649991ms, swizzle: NOOP, TFLOPS: 113.94(+1.75%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=16384, N=8192, K=2048, Warmup=5, Iters=20, 22/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['14.765625 ', '-18.640625'], time:10.99567ms, swizzle: NOOP, TFLOPS: 50.00 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['14.765625 ', '-18.640625'], time:10.49816ms, swizzle: NOOP, TFLOPS: 52.37 (+4.74%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['14.28125  ', '-18.6875  '], time:11.76815ms, swizzle: NOOP, TFLOPS: 46.72
                        (mma4x2+warp2x4): ['14.28125  ', '-18.6875  '], time:6.931424ms, swizzle: NOOP, TFLOPS: 79.31 (+51.46%)
                 (mma4x2+warp2x4+stage3): ['14.28125  ', '-18.6875  '], time:5.055880ms, swizzle: NOOP, TFLOPS: 108.74(+37.10%)
                 (mma4x2+warp2x4+stage2): ['14.28125  ', '-18.6875  '], time:5.022001ms, swizzle: NOOP, TFLOPS: 109.47(+0.67%)
           (mma4x2+warp2x4+stage3+dsmem): ['14.28125  ', '-18.6875  '], time:5.026936ms, swizzle: NOOP, TFLOPS: 109.36
           (mma4x2+warp2x4+stage2+dsmem): ['14.28125  ', '-18.6875  '], time:5.020689ms, swizzle: NOOP, TFLOPS: 109.50(+0.03%)
         (mma4x2+warp2x4+stage3+swizzle): ['14.28125  ', '-18.6875  '], time:5.018496ms, swizzle: 2048, TFLOPS: 109.55(+0.04%)
         (mma4x2+warp2x4+stage2+swizzle): ['14.28125  ', '-18.6875  '], time:4.968738ms, swizzle: 2048, TFLOPS: 110.64(+1.00%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['14.28125  ', '-18.6875  '], time:5.040884ms, swizzle: 2048, TFLOPS: 109.06
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['14.28125  ', '-18.6875  '], time:4.972743ms, swizzle: 2048, TFLOPS: 110.55
                                (cublas): ['14.28125  ', '-18.6875  '], time:4.969763ms, swizzle: NOOP, TFLOPS: 110.62
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=16384, N=8192, K=4096, Warmup=5, Iters=20, 23/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['10.296875 ', '-46.6875  '], time:22.06621ms, swizzle: NOOP, TFLOPS: 49.83 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['10.296875 ', '-46.6875  '], time:21.04604ms, swizzle: NOOP, TFLOPS: 52.24 (+4.85%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['9.84375   ', '-46.71875 '], time:23.35319ms, swizzle: NOOP, TFLOPS: 47.08
                        (mma4x2+warp2x4): ['9.84375   ', '-46.71875 '], time:13.64452ms, swizzle: NOOP, TFLOPS: 80.58 (+54.25%)
                 (mma4x2+warp2x4+stage3): ['9.84375   ', '-46.71875 '], time:9.970688ms, swizzle: NOOP, TFLOPS: 110.27(+36.85%)
                 (mma4x2+warp2x4+stage2): ['9.84375   ', '-46.71875 '], time:9.907126ms, swizzle: NOOP, TFLOPS: 110.98(+0.64%)
           (mma4x2+warp2x4+stage3+dsmem): ['9.84375   ', '-46.71875 '], time:9.919929ms, swizzle: NOOP, TFLOPS: 110.84
           (mma4x2+warp2x4+stage2+dsmem): ['9.84375   ', '-46.71875 '], time:9.905028ms, swizzle: NOOP, TFLOPS: 111.01(+0.02%)
         (mma4x2+warp2x4+stage3+swizzle): ['9.84375   ', '-46.71875 '], time:9.890699ms, swizzle: 2048, TFLOPS: 111.17(+0.14%)
         (mma4x2+warp2x4+stage2+swizzle): ['9.84375   ', '-46.71875 '], time:9.796810ms, swizzle: 2048, TFLOPS: 112.23(+0.96%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['9.84375   ', '-46.71875 '], time:9.931468ms, swizzle: 2048, TFLOPS: 110.71
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['9.84375   ', '-46.71875 '], time:9.799408ms, swizzle: 2048, TFLOPS: 112.20
                                (cublas): ['9.84375   ', '-46.71875 '], time:9.670424ms, swizzle: NOOP, TFLOPS: 113.70(+1.31%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=16384, N=8192, K=8192, Warmup=5, Iters=20, 24/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['47.53125  ', '-51.5     '], time:47.87569ms, swizzle: NOOP, TFLOPS: 45.93 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['47.53125  ', '-51.5     '], time:46.51024ms, swizzle: NOOP, TFLOPS: 47.28 (+2.94%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['47.0      ', '-52.25    '], time:49.74699ms, swizzle: NOOP, TFLOPS: 44.20
                        (mma4x2+warp2x4): ['47.0      ', '-52.25    '], time:28.01880ms, swizzle: NOOP, TFLOPS: 78.48 (+66.00%)
                 (mma4x2+warp2x4+stage3): ['47.0      ', '-52.25    '], time:21.40111ms, swizzle: NOOP, TFLOPS: 102.75(+30.92%)
                 (mma4x2+warp2x4+stage2): ['47.0      ', '-52.25    '], time:20.99103ms, swizzle: NOOP, TFLOPS: 104.76(+1.95%)
           (mma4x2+warp2x4+stage3+dsmem): ['47.0      ', '-52.25    '], time:21.22135ms, swizzle: NOOP, TFLOPS: 103.62
           (mma4x2+warp2x4+stage2+dsmem): ['47.0      ', '-52.25    '], time:21.01814ms, swizzle: NOOP, TFLOPS: 104.62
         (mma4x2+warp2x4+stage3+swizzle): ['47.0      ', '-52.25    '], time:19.75324ms, swizzle: 2048, TFLOPS: 111.32(+6.27%)
         (mma4x2+warp2x4+stage2+swizzle): ['47.0      ', '-52.25    '], time:19.45850ms, swizzle: 2048, TFLOPS: 113.01(+1.51%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['47.0      ', '-52.25    '], time:19.70596ms, swizzle: 2048, TFLOPS: 111.59
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['47.0      ', '-52.25    '], time:19.45621ms, swizzle: 2048, TFLOPS: 113.02(+0.01%)
                                (cublas): ['47.0      ', '-52.25    '], time:19.04292ms, swizzle: NOOP, TFLOPS: 115.48(+2.17%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=16384, N=16384, K=2048, Warmup=5, Iters=20, 25/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['14.765625 ', '-18.640625'], time:22.17438ms, swizzle: NOOP, TFLOPS: 49.58 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['14.765625 ', '-18.640625'], time:21.11024ms, swizzle: NOOP, TFLOPS: 52.08 (+5.04%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['14.28125  ', '-18.6875  '], time:23.47218ms, swizzle: NOOP, TFLOPS: 46.84
                        (mma4x2+warp2x4): ['14.28125  ', '-18.6875  '], time:13.66291ms, swizzle: NOOP, TFLOPS: 80.47 (+54.51%)
                 (mma4x2+warp2x4+stage3): ['14.28125  ', '-18.6875  '], time:10.03656ms, swizzle: NOOP, TFLOPS: 109.55(+36.13%)
                 (mma4x2+warp2x4+stage2): ['14.28125  ', '-18.6875  '], time:9.965801ms, swizzle: NOOP, TFLOPS: 110.33(+0.71%)
           (mma4x2+warp2x4+stage3+dsmem): ['14.28125  ', '-18.6875  '], time:9.974455ms, swizzle: NOOP, TFLOPS: 110.23
           (mma4x2+warp2x4+stage2+dsmem): ['14.28125  ', '-18.6875  '], time:9.966921ms, swizzle: NOOP, TFLOPS: 110.32
         (mma4x2+warp2x4+stage3+swizzle): ['14.28125  ', '-18.6875  '], time:9.954690ms, swizzle: 4096, TFLOPS: 110.45(+0.11%)
         (mma4x2+warp2x4+stage2+swizzle): ['14.28125  ', '-18.6875  '], time:9.859776ms, swizzle: 4096, TFLOPS: 111.51(+0.96%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['14.28125  ', '-18.6875  '], time:9.996986ms, swizzle: 4096, TFLOPS: 109.98
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['14.28125  ', '-18.6875  '], time:9.865450ms, swizzle: 4096, TFLOPS: 111.45
                                (cublas): ['14.28125  ', '-18.6875  '], time:9.698772ms, swizzle: NOOP, TFLOPS: 113.37(+1.66%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=16384, N=16384, K=4096, Warmup=5, Iters=20, 26/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['10.296875 ', '-46.6875  '], time:47.91038ms, swizzle: NOOP, TFLOPS: 45.90 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['10.296875 ', '-46.6875  '], time:46.62165ms, swizzle: NOOP, TFLOPS: 47.17 (+2.76%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['9.84375   ', '-46.71875 '], time:62.72253ms, swizzle: NOOP, TFLOPS: 35.06
                        (mma4x2+warp2x4): ['9.84375   ', '-46.71875 '], time:28.36484ms, swizzle: NOOP, TFLOPS: 77.53 (+64.36%)
                 (mma4x2+warp2x4+stage3): ['9.84375   ', '-46.71875 '], time:21.75440ms, swizzle: NOOP, TFLOPS: 101.08(+30.39%)
                 (mma4x2+warp2x4+stage2): ['9.84375   ', '-46.71875 '], time:21.36998ms, swizzle: NOOP, TFLOPS: 102.90(+1.80%)
           (mma4x2+warp2x4+stage3+dsmem): ['9.84375   ', '-46.71875 '], time:21.83983ms, swizzle: NOOP, TFLOPS: 100.69
           (mma4x2+warp2x4+stage2+dsmem): ['9.84375   ', '-46.71875 '], time:21.24958ms, swizzle: NOOP, TFLOPS: 103.49(+0.57%)
         (mma4x2+warp2x4+stage3+swizzle): ['9.84375   ', '-46.71875 '], time:19.73850ms, swizzle: 4096, TFLOPS: 111.41(+7.66%)
         (mma4x2+warp2x4+stage2+swizzle): ['9.84375   ', '-46.71875 '], time:19.45424ms, swizzle: 4096, TFLOPS: 113.04(+1.46%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['9.84375   ', '-46.71875 '], time:19.71774ms, swizzle: 4096, TFLOPS: 111.53
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['9.84375   ', '-46.71875 '], time:19.46449ms, swizzle: 4096, TFLOPS: 112.98
                                (cublas): ['9.84375   ', '-46.71875 '], time:19.74663ms, swizzle: NOOP, TFLOPS: 111.36
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                        M=16384, N=16384, K=8192, Warmup=5, Iters=20, 27/27
----------------------------------------------------------------------------------------------------------------------------------
                   (f16x8pack+t8x8+dbuf): ['47.53125  ', '-51.5     '], time:104.1649ms, swizzle: NOOP, TFLOPS: 42.22 (+0.00%)
               (f16x8pack+t8x8+k16+dbuf): ['47.53125  ', '-51.5     '], time:101.1433ms, swizzle: NOOP, TFLOPS: 43.48 (+2.99%)
--------------------------------------------------------------------WMMA----------------------------------------------------------
                                (mma4x2): ['47.0      ', '-52.25    '], time:124.0453ms, swizzle: NOOP, TFLOPS: 35.46
                        (mma4x2+warp2x4): ['47.0      ', '-52.25    '], time:55.97209ms, swizzle: NOOP, TFLOPS: 78.58 (+80.70%)
                 (mma4x2+warp2x4+stage3): ['47.0      ', '-52.25    '], time:49.65360ms, swizzle: NOOP, TFLOPS: 88.57 (+12.73%)
                 (mma4x2+warp2x4+stage2): ['47.0      ', '-52.25    '], time:49.79972ms, swizzle: NOOP, TFLOPS: 88.31
           (mma4x2+warp2x4+stage3+dsmem): ['47.0      ', '-52.25    '], time:49.65186ms, swizzle: NOOP, TFLOPS: 88.58 (+0.00%)
           (mma4x2+warp2x4+stage2+dsmem): ['47.0      ', '-52.25    '], time:49.75903ms, swizzle: NOOP, TFLOPS: 88.39
         (mma4x2+warp2x4+stage3+swizzle): ['47.0      ', '-52.25    '], time:39.23461ms, swizzle: 4096, TFLOPS: 112.10(+26.55%)
         (mma4x2+warp2x4+stage2+swizzle): ['47.0      ', '-52.25    '], time:38.65928ms, swizzle: 4096, TFLOPS: 113.76(+1.49%)
   (mma4x2+warp2x4+stage3+dsmem+swizzle): ['47.0      ', '-52.25    '], time:39.22693ms, swizzle: 4096, TFLOPS: 112.12
   (mma4x2+warp2x4+stage2+dsmem+swizzle): ['47.0      ', '-52.25    '], time:38.66374ms, swizzle: 4096, TFLOPS: 113.75
                                (cublas): ['47.0      ', '-52.25    '], time:38.13705ms, swizzle: NOOP, TFLOPS: 115.32(+1.37%)
----------------------------------------------------------------------------------------------------------------------------------
```

## NVIDIA GeForce RTX 3080 Laptop 
<div id="NV-RTX-3080"></div>

```bash
python3 hgemm.py --wmma --no-default
```
输出：
```bash
----------------------------------------------------------------------------------------------------------------------------------
                              M=4096, N=4096, K=2048, Warmup=5, Iters=20, 1/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['-34.9375  ', '2.25585938'], time:1.397085ms, swizzle: NOOP, TFLOPS: 49.19 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['-34.9375  ', '2.25585938'], time:1.632452ms, swizzle: NOOP, TFLOPS: 42.10
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['-34.9375  ', '2.25585938'], time:1.392316ms, swizzle: 1024, TFLOPS: 49.36 (+0.34%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['-34.9375  ', '2.25585938'], time:1.537656ms, swizzle: 1024, TFLOPS: 44.69
                                (cublas): ['-34.90625 ', '2.21875   '], time:1.072788ms, swizzle: NOOP, TFLOPS: 64.06 (+29.78%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=4096, N=4096, K=4096, Warmup=5, Iters=20, 2/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['10.8515625', '9.4140625 '], time:3.154301ms, swizzle: NOOP, TFLOPS: 43.57 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['10.8515625', '9.4140625 '], time:3.152799ms, swizzle: NOOP, TFLOPS: 43.59 (+0.05%)
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['10.8515625', '9.4140625 '], time:2.640366ms, swizzle: 1024, TFLOPS: 52.05 (+19.41%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['10.8515625', '9.4140625 '], time:3.021883ms, swizzle: 1024, TFLOPS: 45.48
                                (cublas): ['10.8515625', '9.4140625 '], time:2.330613ms, swizzle: NOOP, TFLOPS: 58.97 (+13.29%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=4096, N=4096, K=8192, Warmup=5, Iters=20, 3/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['68.375    ', '-2.234375 '], time:5.776286ms, swizzle: NOOP, TFLOPS: 47.59 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['68.375    ', '-2.234375 '], time:6.212115ms, swizzle: NOOP, TFLOPS: 44.25
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['68.375    ', '-2.234375 '], time:5.236458ms, swizzle: 1024, TFLOPS: 52.49 (+10.31%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['68.375    ', '-2.234375 '], time:5.674219ms, swizzle: 1024, TFLOPS: 48.44
                                (cublas): ['68.375    ', '-2.234375 '], time:5.311441ms, swizzle: NOOP, TFLOPS: 51.75
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=4096, N=8192, K=2048, Warmup=5, Iters=20, 4/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['-34.9375  ', '2.25585938'], time:3.303718ms, swizzle: NOOP, TFLOPS: 41.60 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['-34.9375  ', '2.25585938'], time:3.193497ms, swizzle: NOOP, TFLOPS: 43.04 (+3.45%)
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['-34.9375  ', '2.25585938'], time:2.624654ms, swizzle: 2048, TFLOPS: 52.36 (+21.67%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['-34.9375  ', '2.25585938'], time:2.863550ms, swizzle: 2048, TFLOPS: 48.00
                                (cublas): ['-34.90625 ', '2.21875   '], time:2.649235ms, swizzle: NOOP, TFLOPS: 51.88
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=4096, N=8192, K=4096, Warmup=5, Iters=20, 5/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['10.8515625', '9.4140625 '], time:5.747509ms, swizzle: NOOP, TFLOPS: 47.83 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['10.8515625', '9.4140625 '], time:6.356692ms, swizzle: NOOP, TFLOPS: 43.24
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['10.8515625', '9.4140625 '], time:5.048251ms, swizzle: 2048, TFLOPS: 54.45 (+13.85%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['10.8515625', '9.4140625 '], time:5.489063ms, swizzle: 2048, TFLOPS: 50.08
                                (cublas): ['10.8515625', '9.4140625 '], time:6.013441ms, swizzle: NOOP, TFLOPS: 45.71
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=4096, N=8192, K=8192, Warmup=5, Iters=20, 6/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['68.375    ', '-2.234375 '], time:11.15694ms, swizzle: NOOP, TFLOPS: 49.27 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['68.375    ', '-2.234375 '], time:12.09821ms, swizzle: NOOP, TFLOPS: 45.44
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['68.375    ', '-2.234375 '], time:9.958195ms, swizzle: 2048, TFLOPS: 55.21 (+12.04%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['68.375    ', '-2.234375 '], time:10.67364ms, swizzle: 2048, TFLOPS: 51.51
                                (cublas): ['68.375    ', '-2.234375 '], time:12.02430ms, swizzle: NOOP, TFLOPS: 45.72
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=4096, N=16384, K=2048, Warmup=5, Iters=20, 7/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['-34.9375  ', '2.25585938'], time:6.608533ms, swizzle: NOOP, TFLOPS: 41.59 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['-34.9375  ', '2.25585938'], time:6.812095ms, swizzle: NOOP, TFLOPS: 40.35
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['-34.9375  ', '2.25585938'], time:5.446910ms, swizzle: 4096, TFLOPS: 50.46 (+21.33%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['-34.9375  ', '2.25585938'], time:5.769944ms, swizzle: 4096, TFLOPS: 47.64
                                (cublas): ['-34.90625 ', '2.21875   '], time:6.295609ms, swizzle: NOOP, TFLOPS: 43.66
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=4096, N=16384, K=4096, Warmup=5, Iters=20, 8/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['10.8515625', '9.4140625 '], time:11.90752ms, swizzle: NOOP, TFLOPS: 46.17 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['10.8515625', '9.4140625 '], time:12.66958ms, swizzle: NOOP, TFLOPS: 43.39
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['10.8515625', '9.4140625 '], time:10.72070ms, swizzle: 4096, TFLOPS: 51.28 (+11.07%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['10.8515625', '9.4140625 '], time:11.09249ms, swizzle: 4096, TFLOPS: 49.56
                                (cublas): ['10.8515625', '9.4140625 '], time:9.910416ms, swizzle: NOOP, TFLOPS: 55.47 (+8.18%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=4096, N=16384, K=8192, Warmup=5, Iters=20, 9/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['68.375    ', '-2.234375 '], time:23.75357ms, swizzle: NOOP, TFLOPS: 46.29 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['68.375    ', '-2.234375 '], time:25.33891ms, swizzle: NOOP, TFLOPS: 43.39
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['68.375    ', '-2.234375 '], time:20.78440ms, swizzle: 4096, TFLOPS: 52.90 (+14.29%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['68.375    ', '-2.234375 '], time:22.58212ms, swizzle: 4096, TFLOPS: 48.69
                                (cublas): ['68.375    ', '-2.234375 '], time:23.13928ms, swizzle: NOOP, TFLOPS: 47.52
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=8192, N=4096, K=2048, Warmup=5, Iters=20, 10/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['-34.9375  ', '2.25585938'], time:3.206682ms, swizzle: NOOP, TFLOPS: 42.86 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['-34.9375  ', '2.25585938'], time:3.255009ms, swizzle: NOOP, TFLOPS: 42.22
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['-34.9375  ', '2.25585938'], time:2.551007ms, swizzle: 1024, TFLOPS: 53.88 (+25.70%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['-34.9375  ', '2.25585938'], time:2.943944ms, swizzle: 1024, TFLOPS: 46.69
                                (cublas): ['-34.90625 ', '2.21875   '], time:2.616691ms, swizzle: NOOP, TFLOPS: 52.52
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=8192, N=4096, K=4096, Warmup=5, Iters=20, 11/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['10.8515625', '9.4140625 '], time:5.581545ms, swizzle: NOOP, TFLOPS: 49.25 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['10.8515625', '9.4140625 '], time:5.918717ms, swizzle: NOOP, TFLOPS: 46.44
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['10.8515625', '9.4140625 '], time:5.013823ms, swizzle: 1024, TFLOPS: 54.82 (+11.32%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['10.8515625', '9.4140625 '], time:5.475091ms, swizzle: 1024, TFLOPS: 50.21
                                (cublas): ['10.8515625', '9.4140625 '], time:5.620026ms, swizzle: NOOP, TFLOPS: 48.91
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=8192, N=4096, K=8192, Warmup=5, Iters=20, 12/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['68.375    ', '-2.234375 '], time:10.63799ms, swizzle: NOOP, TFLOPS: 51.68 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['68.375    ', '-2.234375 '], time:11.95423ms, swizzle: NOOP, TFLOPS: 45.99
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['68.375    ', '-2.234375 '], time:10.08455ms, swizzle: 1024, TFLOPS: 54.51 (+5.49%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['68.375    ', '-2.234375 '], time:10.80915ms, swizzle: 1024, TFLOPS: 50.86
                                (cublas): ['68.375    ', '-2.234375 '], time:12.14854ms, swizzle: NOOP, TFLOPS: 45.25
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=8192, N=8192, K=2048, Warmup=5, Iters=20, 13/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['-34.9375  ', '2.25585938'], time:6.046414ms, swizzle: NOOP, TFLOPS: 45.46 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['-34.9375  ', '2.25585938'], time:6.623530ms, swizzle: NOOP, TFLOPS: 41.50
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['-34.9375  ', '2.25585938'], time:5.341410ms, swizzle: 2048, TFLOPS: 51.46 (+13.20%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['-34.9375  ', '2.25585938'], time:5.689215ms, swizzle: 2048, TFLOPS: 48.32
                                (cublas): ['-34.90625 ', '2.21875   '], time:6.602764ms, swizzle: NOOP, TFLOPS: 41.63
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=8192, N=8192, K=4096, Warmup=5, Iters=20, 14/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['10.8515625', '9.4140625 '], time:11.54751ms, swizzle: NOOP, TFLOPS: 47.61 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['10.8515625', '9.4140625 '], time:12.49833ms, swizzle: NOOP, TFLOPS: 43.99
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['10.8515625', '9.4140625 '], time:10.34743ms, swizzle: 2048, TFLOPS: 53.13 (+11.60%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['10.8515625', '9.4140625 '], time:10.89727ms, swizzle: 2048, TFLOPS: 50.45
                                (cublas): ['10.8515625', '9.4140625 '], time:11.89055ms, swizzle: NOOP, TFLOPS: 46.23
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=8192, N=8192, K=8192, Warmup=5, Iters=20, 15/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['68.375    ', '-2.234375 '], time:23.22742ms, swizzle: NOOP, TFLOPS: 47.34 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['68.375    ', '-2.234375 '], time:25.00588ms, swizzle: NOOP, TFLOPS: 43.97
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['68.375    ', '-2.234375 '], time:20.04830ms, swizzle: 2048, TFLOPS: 54.84 (+15.86%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['68.375    ', '-2.234375 '], time:21.89767ms, swizzle: 2048, TFLOPS: 50.21
                                (cublas): ['68.375    ', '-2.234375 '], time:23.18794ms, swizzle: NOOP, TFLOPS: 47.42
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=8192, N=16384, K=2048, Warmup=5, Iters=20, 16/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['-34.9375  ', '2.25585938'], time:12.24069ms, swizzle: NOOP, TFLOPS: 44.91 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['-34.9375  ', '2.25585938'], time:13.07930ms, swizzle: NOOP, TFLOPS: 42.03
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['-34.9375  ', '2.25585938'], time:10.82205ms, swizzle: 4096, TFLOPS: 50.80 (+13.11%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['-34.9375  ', '2.25585938'], time:11.43186ms, swizzle: 4096, TFLOPS: 48.09
                                (cublas): ['-34.90625 ', '2.21875   '], time:13.87636ms, swizzle: NOOP, TFLOPS: 39.62
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=8192, N=16384, K=4096, Warmup=5, Iters=20, 17/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['10.8515625', '9.4140625 '], time:23.84941ms, swizzle: NOOP, TFLOPS: 46.10 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['10.8515625', '9.4140625 '], time:31.07695ms, swizzle: NOOP, TFLOPS: 35.38
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['10.8515625', '9.4140625 '], time:23.16045ms, swizzle: 4096, TFLOPS: 47.47 (+2.97%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['10.8515625', '9.4140625 '], time:25.17983ms, swizzle: 4096, TFLOPS: 43.67
                                (cublas): ['10.8515625', '9.4140625 '], time:20.92361ms, swizzle: NOOP, TFLOPS: 52.55 (+10.69%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=8192, N=16384, K=8192, Warmup=5, Iters=20, 18/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['68.375    ', '-2.234375 '], time:48.17764ms, swizzle: NOOP, TFLOPS: 45.64 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['68.375    ', '-2.234375 '], time:51.66683ms, swizzle: NOOP, TFLOPS: 42.56
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['68.375    ', '-2.234375 '], time:42.50290ms, swizzle: 4096, TFLOPS: 51.74 (+13.35%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['68.375    ', '-2.234375 '], time:46.67718ms, swizzle: 4096, TFLOPS: 47.11
                                (cublas): ['68.375    ', '-2.234375 '], time:45.62001ms, swizzle: NOOP, TFLOPS: 48.20
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=16384, N=4096, K=2048, Warmup=5, Iters=20, 19/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['-34.9375  ', '2.25585938'], time:5.999112ms, swizzle: NOOP, TFLOPS: 45.82 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['-34.9375  ', '2.25585938'], time:6.952166ms, swizzle: NOOP, TFLOPS: 39.54
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['-34.9375  ', '2.25585938'], time:5.714607ms, swizzle: 1024, TFLOPS: 48.10 (+4.98%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['-34.9375  ', '2.25585938'], time:5.846762ms, swizzle: 1024, TFLOPS: 47.01
                                (cublas): ['-34.9375  ', '2.25585938'], time:5.578041ms, swizzle: NOOP, TFLOPS: 49.28 (+2.45%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=16384, N=4096, K=4096, Warmup=5, Iters=20, 20/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['10.8515625', '9.4140625 '], time:11.36004ms, swizzle: NOOP, TFLOPS: 48.39 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['10.8515625', '9.4140625 '], time:12.24460ms, swizzle: NOOP, TFLOPS: 44.90
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['10.8515625', '9.4140625 '], time:10.57424ms, swizzle: 1024, TFLOPS: 51.99 (+7.43%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['10.8515625', '9.4140625 '], time:11.31019ms, swizzle: 1024, TFLOPS: 48.61
                                (cublas): ['10.8515625', '9.4140625 '], time:10.14137ms, swizzle: NOOP, TFLOPS: 54.21 (+4.27%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=16384, N=4096, K=8192, Warmup=5, Iters=20, 21/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['68.375    ', '-2.234375 '], time:21.54934ms, swizzle: NOOP, TFLOPS: 51.02 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['68.375    ', '-2.234375 '], time:25.34153ms, swizzle: NOOP, TFLOPS: 43.39
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['68.375    ', '-2.234375 '], time:21.18096ms, swizzle: 1024, TFLOPS: 51.91 (+1.74%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['68.375    ', '-2.234375 '], time:22.19107ms, swizzle: 1024, TFLOPS: 49.55
                                (cublas): ['68.375    ', '-2.234375 '], time:23.78721ms, swizzle: NOOP, TFLOPS: 46.22
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=16384, N=8192, K=2048, Warmup=5, Iters=20, 22/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['-34.9375  ', '2.25585938'], time:12.14342ms, swizzle: NOOP, TFLOPS: 45.27 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['-34.9375  ', '2.25585938'], time:13.07780ms, swizzle: NOOP, TFLOPS: 42.04
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['-34.9375  ', '2.25585938'], time:10.68298ms, swizzle: 2048, TFLOPS: 51.46 (+13.67%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['-34.9375  ', '2.25585938'], time:11.51511ms, swizzle: 2048, TFLOPS: 47.74
                                (cublas): ['-34.9375  ', '2.25585938'], time:12.36820ms, swizzle: NOOP, TFLOPS: 44.45
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=16384, N=8192, K=4096, Warmup=5, Iters=20, 23/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['10.8515625', '9.4140625 '], time:23.26002ms, swizzle: NOOP, TFLOPS: 47.27 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['10.8515625', '9.4140625 '], time:25.28347ms, swizzle: NOOP, TFLOPS: 43.49
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['10.8515625', '9.4140625 '], time:20.98624ms, swizzle: 2048, TFLOPS: 52.39 (+10.83%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['10.8515625', '9.4140625 '], time:22.29118ms, swizzle: 2048, TFLOPS: 49.32
                                (cublas): ['10.8515625', '9.4140625 '], time:23.58868ms, swizzle: NOOP, TFLOPS: 46.61
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=16384, N=8192, K=8192, Warmup=5, Iters=20, 24/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['68.375    ', '-2.234375 '], time:46.57695ms, swizzle: NOOP, TFLOPS: 47.21 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['68.375    ', '-2.234375 '], time:50.11103ms, swizzle: NOOP, TFLOPS: 43.88
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['68.375    ', '-2.234375 '], time:39.97759ms, swizzle: 2048, TFLOPS: 55.01 (+16.51%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['68.375    ', '-2.234375 '], time:45.07379ms, swizzle: 2048, TFLOPS: 48.79
                                (cublas): ['68.375    ', '-2.234375 '], time:46.13645ms, swizzle: NOOP, TFLOPS: 47.66
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=16384, N=16384, K=2048, Warmup=5, Iters=20, 25/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['-34.9375  ', '2.25585938'], time:24.82917ms, swizzle: NOOP, TFLOPS: 44.28 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['-34.9375  ', '2.25585938'], time:26.81517ms, swizzle: NOOP, TFLOPS: 41.00
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['-34.9375  ', '2.25585938'], time:22.22962ms, swizzle: 4096, TFLOPS: 49.46 (+11.69%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['-34.9375  ', '2.25585938'], time:23.27709ms, swizzle: 4096, TFLOPS: 47.24
                                (cublas): ['-34.90625 ', '2.21875   '], time:25.84185ms, swizzle: NOOP, TFLOPS: 42.55
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=16384, N=16384, K=4096, Warmup=5, Iters=20, 26/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['10.8515625', '9.4140625 '], time:48.43459ms, swizzle: NOOP, TFLOPS: 45.40 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['10.8515625', '9.4140625 '], time:52.00080ms, swizzle: NOOP, TFLOPS: 42.29
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['10.8515625', '9.4140625 '], time:43.28680ms, swizzle: 4096, TFLOPS: 50.80 (+11.89%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['10.8515625', '9.4140625 '], time:47.73476ms, swizzle: 4096, TFLOPS: 46.07
                                (cublas): ['10.8515625', '9.4140625 '], time:40.64793ms, swizzle: NOOP, TFLOPS: 54.10 (+6.49%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                              M=16384, N=16384, K=8192, Warmup=5, Iters=20, 27/27
----------------------------------------------------------------------------------------------------------------------------------
           (mma4x4+warp4x4+stage3+dsmem): ['68.375    ', '-2.234375 '], time:96.91984ms, swizzle: NOOP, TFLOPS: 45.38 (+0.00%)
           (mma4x4+warp4x4+stage2+dsmem): ['68.375    ', '-2.234375 '], time:102.8722ms, swizzle: NOOP, TFLOPS: 42.75
   (mma4x4+warp4x4+stage3+dsmem+swizzle): ['68.375    ', '-2.234375 '], time:85.65800ms, swizzle: 4096, TFLOPS: 51.34 (+13.15%)
   (mma4x4+warp4x4+stage2+dsmem+swizzle): ['68.375    ', '-2.234375 '], time:95.70884ms, swizzle: 4096, TFLOPS: 45.95
                                (cublas): ['68.375    ', '-2.234375 '], time:104.2092ms, swizzle: NOOP, TFLOPS: 42.20
----------------------------------------------------------------------------------------------------------------------------------
```
