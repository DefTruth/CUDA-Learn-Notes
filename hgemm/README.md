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
- [X] hgemm_wmma_m16n16k16_mma4x2(WMMA API, Tensor Cores, Tile MMA) 
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4(WMMA API, Tensor Cores, Tile MMA, Tile Warp, pack) 
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_async(WMMA API, Tensor Cores, Tile MMA, Tile Warp, Copy Async) 
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_async_offset(WMMA API, Tensor Cores, Tile MMA, Tile Warp, Copy Async, Pad(bank conflicts reduce)) 
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

- SASS

```C
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

```bash
--------------------------------------------------------------------------------------------------------------
                                             M=1024, N=1024, K=512
                                 out_f16: ['-26.75      ', '-56.21875   ', '5.89453125  '], time:0.320697ms
                             out_f16(sk): ['-26.75      ', '-56.21875   ', '5.89453125  '], time:0.239539ms
                  out_f16x4pack(t4x4bcf): ['-26.75      ', '-56.21875   ', '5.89453125  '], time:0.041521ms
               out_f16x4pack(t4x4offset): ['-26.75      ', '-56.21875   ', '5.89453125  '], time:0.040841ms
                       out_f16x4(t8x8sk): ['-26.75      ', '-56.21875   ', '5.89453125  '], time:0.052857ms
                      out_f16x4(t8x8bcf): ['-26.75      ', '-56.21875   ', '5.89453125  '], time:0.048614ms
                   out_f16x4pack(t8x8sk): ['-26.75      ', '-56.21875   ', '5.89453125  '], time:0.049174ms
                      out_f16x4pack(bcf): ['-26.75      ', '-56.21875   ', '5.89453125  '], time:0.045705ms
               out_f16x4pack(bcf+offset): ['-26.75      ', '-56.21875   ', '5.89453125  '], time:0.045598ms
                      out_f16x8pack(bcf): ['-26.75      ', '-56.21875   ', '5.89453125  '], time:0.045764ms
               out_f16x8pack(bcf+offset): ['-26.75      ', '-56.21875   ', '5.89453125  '], time:0.045478ms
                 out_f16x8pack(bcf+dbuf): ['-26.75      ', '-56.21875   ', '5.89453125  '], time:0.036132ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['-26.75      ', '-56.21875   ', '5.89453125  '], time:0.032198ms
          out_f16x8pack(k16+dbuf+offset): ['-26.75      ', '-56.21875   ', '5.89453125  '], time:0.032032ms
           out_f16x8pack(k16+dbuf+async): ['-26.75      ', '-56.21875   ', '5.89453125  '], time:0.032830ms
                 out_f16x8pack(k32+dbuf): ['-26.75      ', '-56.21875   ', '5.89453125  '], time:0.033486ms
           out_f16x8pack(k32+dbuf+async): ['-26.75      ', '-56.21875   ', '5.89453125  '], time:0.032258ms
           out_f16x8pack(k32+dbuf+t16x8): ['-26.75      ', '-56.21875   ', '5.89453125  '], time:0.037527ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['-26.75      ', '-56.21875   ', '5.89453125  '], time:0.034297ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['-26.765625  ', '-56.21875   ', '5.89453125  '], time:0.051379ms
                     out_f16wmma(mma4x2): ['-26.765625  ', '-56.21875   ', '5.89453125  '], time:0.028968ms
             out_f16wmma(mma4x2+warp2x4): ['-26.765625  ', '-56.21875   ', '5.89453125  '], time:0.029492ms
       out_f16wmma(mma4x2+warp2x4+async): ['-26.765625  ', '-56.21875   ', '5.89453125  '], time:0.030243ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['-26.765625  ', '-56.21875   ', '5.89453125  '], time:0.028145ms
                              out_f16_th: ['-26.765625  ', '-56.21875   ', '5.921875    '], time:0.017655ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=1024, N=1024, K=1024
                                 out_f16: ['-13.1875    ', '-56.03125   ', '-50.0       '], time:0.635300ms
                             out_f16(sk): ['-13.1875    ', '-56.03125   ', '-50.0       '], time:0.472727ms
                  out_f16x4pack(t4x4bcf): ['-13.1875    ', '-56.03125   ', '-50.0       '], time:0.080142ms
               out_f16x4pack(t4x4offset): ['-13.1875    ', '-56.03125   ', '-50.0       '], time:0.078721ms
                       out_f16x4(t8x8sk): ['-13.1875    ', '-56.03125   ', '-50.0       '], time:0.100036ms
                      out_f16x4(t8x8bcf): ['-13.1875    ', '-56.03125   ', '-50.0       '], time:0.093884ms
                   out_f16x4pack(t8x8sk): ['-13.1875    ', '-56.03125   ', '-50.0       '], time:0.094185ms
                      out_f16x4pack(bcf): ['-13.1875    ', '-56.03125   ', '-50.0       '], time:0.087967ms
               out_f16x4pack(bcf+offset): ['-13.1875    ', '-56.03125   ', '-50.0       '], time:0.087862ms
                      out_f16x8pack(bcf): ['-13.1875    ', '-56.03125   ', '-50.0       '], time:0.088115ms
               out_f16x8pack(bcf+offset): ['-13.1875    ', '-56.03125   ', '-50.0       '], time:0.087733ms
                 out_f16x8pack(bcf+dbuf): ['-13.1875    ', '-56.03125   ', '-50.0       '], time:0.068917ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['-13.1875    ', '-56.03125   ', '-50.0       '], time:0.060930ms
          out_f16x8pack(k16+dbuf+offset): ['-13.1875    ', '-56.03125   ', '-50.0       '], time:0.060573ms
           out_f16x8pack(k16+dbuf+async): ['-13.1875    ', '-56.03125   ', '-50.0       '], time:0.062151ms
                 out_f16x8pack(k32+dbuf): ['-13.1875    ', '-56.03125   ', '-50.0       '], time:0.063453ms
           out_f16x8pack(k32+dbuf+async): ['-13.1875    ', '-56.03125   ', '-50.0       '], time:0.061073ms
           out_f16x8pack(k32+dbuf+t16x8): ['-13.1875    ', '-56.03125   ', '-50.0       '], time:0.071521ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['-13.1875    ', '-56.03125   ', '-50.0       '], time:0.064721ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['-13.1640625 ', '-56.09375   ', '-50.09375   '], time:0.099897ms
                     out_f16wmma(mma4x2): ['-13.1640625 ', '-56.09375   ', '-50.09375   '], time:0.054007ms
             out_f16wmma(mma4x2+warp2x4): ['-13.1640625 ', '-56.09375   ', '-50.09375   '], time:0.052590ms
       out_f16wmma(mma4x2+warp2x4+async): ['-13.1640625 ', '-56.09375   ', '-50.09375   '], time:0.054293ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['-13.1640625 ', '-56.09375   ', '-50.09375   '], time:0.049825ms
                              out_f16_th: ['-13.1640625 ', '-56.09375   ', '-50.0       '], time:0.030451ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=1024, N=2048, K=256
                                 out_f16: ['-12.9921875 ', '61.03125    ', '-27.015625  '], time:0.310507ms
                             out_f16(sk): ['-12.9921875 ', '61.03125    ', '-27.015625  '], time:0.233178ms
                  out_f16x4pack(t4x4bcf): ['-12.9921875 ', '61.03125    ', '-27.015625  '], time:0.038066ms
               out_f16x4pack(t4x4offset): ['-12.9921875 ', '61.03125    ', '-27.015625  '], time:0.037398ms
                       out_f16x4(t8x8sk): ['-12.9921875 ', '61.03125    ', '-27.015625  '], time:0.045404ms
                      out_f16x4(t8x8bcf): ['-12.9921875 ', '61.03125    ', '-27.015625  '], time:0.041518ms
                   out_f16x4pack(t8x8sk): ['-12.9921875 ', '61.03125    ', '-27.015625  '], time:0.040779ms
                      out_f16x4pack(bcf): ['-12.9921875 ', '61.03125    ', '-27.015625  '], time:0.038395ms
               out_f16x4pack(bcf+offset): ['-12.9921875 ', '61.03125    ', '-27.015625  '], time:0.038462ms
                      out_f16x8pack(bcf): ['-12.9921875 ', '61.03125    ', '-27.015625  '], time:0.036817ms
               out_f16x8pack(bcf+offset): ['-12.9921875 ', '61.03125    ', '-27.015625  '], time:0.036678ms
                 out_f16x8pack(bcf+dbuf): ['-12.9921875 ', '61.03125    ', '-27.015625  '], time:0.034232ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['-12.9921875 ', '61.03125    ', '-27.015625  '], time:0.031495ms
          out_f16x8pack(k16+dbuf+offset): ['-12.9921875 ', '61.03125    ', '-27.015625  '], time:0.031400ms
           out_f16x8pack(k16+dbuf+async): ['-12.9921875 ', '61.03125    ', '-27.015625  '], time:0.032153ms
                 out_f16x8pack(k32+dbuf): ['-12.9921875 ', '61.03125    ', '-27.015625  '], time:0.032735ms
           out_f16x8pack(k32+dbuf+async): ['-12.9921875 ', '61.03125    ', '-27.015625  '], time:0.031948ms
           out_f16x8pack(k32+dbuf+t16x8): ['-12.9921875 ', '61.03125    ', '-27.015625  '], time:0.034952ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['-12.9921875 ', '61.03125    ', '-27.015625  '], time:0.033283ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['-12.96875   ', '61.03125    ', '-27.015625  '], time:0.050774ms
                     out_f16wmma(mma4x2): ['-12.96875   ', '61.03125    ', '-27.015625  '], time:0.029092ms
             out_f16wmma(mma4x2+warp2x4): ['-12.96875   ', '61.03125    ', '-27.015625  '], time:0.024495ms
       out_f16wmma(mma4x2+warp2x4+async): ['-12.96875   ', '61.03125    ', '-27.015625  '], time:0.024762ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['-12.96875   ', '61.03125    ', '-27.015625  '], time:0.022731ms
                              out_f16_th: ['-12.96875   ', '61.0        ', '-27.015625  '], time:0.018306ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=1024, N=2048, K=512
                                 out_f16: ['1.05175781  ', '-15.90625   ', '43.09375    '], time:0.613160ms
                             out_f16(sk): ['1.05175781  ', '-15.90625   ', '43.09375    '], time:0.457511ms
                  out_f16x4pack(t4x4bcf): ['1.05175781  ', '-15.90625   ', '43.09375    '], time:0.073748ms
               out_f16x4pack(t4x4offset): ['1.05175781  ', '-15.90625   ', '43.09375    '], time:0.072427ms
                       out_f16x4(t8x8sk): ['1.05175781  ', '-15.90625   ', '43.09375    '], time:0.084753ms
                      out_f16x4(t8x8bcf): ['1.05175781  ', '-15.90625   ', '43.09375    '], time:0.079632ms
                   out_f16x4pack(t8x8sk): ['1.05175781  ', '-15.90625   ', '43.09375    '], time:0.078106ms
                      out_f16x4pack(bcf): ['1.05175781  ', '-15.90625   ', '43.09375    '], time:0.073862ms
               out_f16x4pack(bcf+offset): ['1.05175781  ', '-15.90625   ', '43.09375    '], time:0.074024ms
                      out_f16x8pack(bcf): ['1.05175781  ', '-15.90625   ', '43.09375    '], time:0.070696ms
               out_f16x8pack(bcf+offset): ['1.05175781  ', '-15.90625   ', '43.09375    '], time:0.070424ms
                 out_f16x8pack(bcf+dbuf): ['1.05175781  ', '-15.90625   ', '43.09375    '], time:0.065341ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['1.05175781  ', '-15.90625   ', '43.09375    '], time:0.059662ms
          out_f16x8pack(k16+dbuf+offset): ['1.05175781  ', '-15.90625   ', '43.09375    '], time:0.059495ms
           out_f16x8pack(k16+dbuf+async): ['1.05175781  ', '-15.90625   ', '43.09375    '], time:0.061226ms
                 out_f16x8pack(k32+dbuf): ['1.05175781  ', '-15.90625   ', '43.09375    '], time:0.062227ms
           out_f16x8pack(k32+dbuf+async): ['1.05175781  ', '-15.90625   ', '43.09375    '], time:0.060568ms
           out_f16x8pack(k32+dbuf+t16x8): ['1.05175781  ', '-15.90625   ', '43.09375    '], time:0.065980ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['1.05175781  ', '-15.90625   ', '43.09375    '], time:0.062442ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['1.06835938  ', '-15.8359375 ', '43.15625    '], time:0.098968ms
                     out_f16wmma(mma4x2): ['1.06835938  ', '-15.8359375 ', '43.15625    '], time:0.053892ms
             out_f16wmma(mma4x2+warp2x4): ['1.06835938  ', '-15.8359375 ', '43.15625    '], time:0.043492ms
       out_f16wmma(mma4x2+warp2x4+async): ['1.06835938  ', '-15.8359375 ', '43.15625    '], time:0.043840ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['1.06835938  ', '-15.8359375 ', '43.15625    '], time:0.039301ms
                              out_f16_th: ['1.06640625  ', '-15.8671875 ', '43.1875     '], time:0.031567ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=1024, N=2048, K=1024
                                 out_f16: ['-12.9453125 ', '-0.81494141 ', '-9.0078125  '], time:1.216793ms
                             out_f16(sk): ['-12.9453125 ', '-0.81494141 ', '-9.0078125  '], time:0.904732ms
                  out_f16x4pack(t4x4bcf): ['-12.9453125 ', '-0.81494141 ', '-9.0078125  '], time:0.145049ms
               out_f16x4pack(t4x4offset): ['-12.9453125 ', '-0.81494141 ', '-9.0078125  '], time:0.142646ms
                       out_f16x4(t8x8sk): ['-12.9453125 ', '-0.81494141 ', '-9.0078125  '], time:0.162644ms
                      out_f16x4(t8x8bcf): ['-12.9453125 ', '-0.81494141 ', '-9.0078125  '], time:0.155468ms
                   out_f16x4pack(t8x8sk): ['-12.9453125 ', '-0.81494141 ', '-9.0078125  '], time:0.151381ms
                      out_f16x4pack(bcf): ['-12.9453125 ', '-0.81494141 ', '-9.0078125  '], time:0.143986ms
               out_f16x4pack(bcf+offset): ['-12.9453125 ', '-0.81494141 ', '-9.0078125  '], time:0.144348ms
                      out_f16x8pack(bcf): ['-12.9453125 ', '-0.81494141 ', '-9.0078125  '], time:0.137858ms
               out_f16x8pack(bcf+offset): ['-12.9453125 ', '-0.81494141 ', '-9.0078125  '], time:0.137372ms
                 out_f16x8pack(bcf+dbuf): ['-12.9453125 ', '-0.81494141 ', '-9.0078125  '], time:0.127616ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['-12.9453125 ', '-0.81494141 ', '-9.0078125  '], time:0.116458ms
          out_f16x8pack(k16+dbuf+offset): ['-12.9453125 ', '-0.81494141 ', '-9.0078125  '], time:0.116086ms
           out_f16x8pack(k16+dbuf+async): ['-12.9453125 ', '-0.81494141 ', '-9.0078125  '], time:0.119481ms
                 out_f16x8pack(k32+dbuf): ['-12.9453125 ', '-0.81494141 ', '-9.0078125  '], time:0.121422ms
           out_f16x8pack(k32+dbuf+async): ['-12.9453125 ', '-0.81494141 ', '-9.0078125  '], time:0.117879ms
           out_f16x8pack(k32+dbuf+t16x8): ['-12.9453125 ', '-0.81494141 ', '-9.0078125  '], time:0.128512ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['-12.9453125 ', '-0.81494141 ', '-9.0078125  '], time:0.120835ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['-12.890625  ', '-0.80322266 ', '-8.96875    '], time:0.194211ms
                     out_f16wmma(mma4x2): ['-12.890625  ', '-0.80322266 ', '-8.96875    '], time:0.103297ms
             out_f16wmma(mma4x2+warp2x4): ['-12.890625  ', '-0.80322266 ', '-8.96875    '], time:0.080924ms
       out_f16wmma(mma4x2+warp2x4+async): ['-12.890625  ', '-0.80322266 ', '-8.96875    '], time:0.081215ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['-12.890625  ', '-0.80322266 ', '-8.96875    '], time:0.071983ms
                              out_f16_th: ['-12.90625   ', '-0.79150391 ', '-8.9296875  '], time:0.057764ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=1024, N=4096, K=256
                                 out_f16: ['-18.375     ', '1.76269531  ', '-3.64648438 '], time:0.606341ms
                             out_f16(sk): ['-18.375     ', '1.76269531  ', '-3.64648438 '], time:0.454860ms
                  out_f16x4pack(t4x4bcf): ['-18.375     ', '1.76269531  ', '-3.64648438 '], time:0.070624ms
               out_f16x4pack(t4x4offset): ['-18.375     ', '1.76269531  ', '-3.64648438 '], time:0.069318ms
                       out_f16x4(t8x8sk): ['-18.375     ', '1.76269531  ', '-3.64648438 '], time:0.065789ms
                      out_f16x4(t8x8bcf): ['-18.375     ', '1.76269531  ', '-3.64648438 '], time:0.058060ms
                   out_f16x4pack(t8x8sk): ['-18.375     ', '1.76269531  ', '-3.64648438 '], time:0.058141ms
                      out_f16x4pack(bcf): ['-18.375     ', '1.76269531  ', '-3.64648438 '], time:0.053849ms
               out_f16x4pack(bcf+offset): ['-18.375     ', '1.76269531  ', '-3.64648438 '], time:0.053740ms
                      out_f16x8pack(bcf): ['-18.375     ', '1.76269531  ', '-3.64648438 '], time:0.052452ms
               out_f16x8pack(bcf+offset): ['-18.375     ', '1.76269531  ', '-3.64648438 '], time:0.052152ms
                 out_f16x8pack(bcf+dbuf): ['-18.375     ', '1.76269531  ', '-3.64648438 '], time:0.049257ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['-18.375     ', '1.76269531  ', '-3.64648438 '], time:0.046206ms
          out_f16x8pack(k16+dbuf+offset): ['-18.375     ', '1.76269531  ', '-3.64648438 '], time:0.046039ms
           out_f16x8pack(k16+dbuf+async): ['-18.375     ', '1.76269531  ', '-3.64648438 '], time:0.047054ms
                 out_f16x8pack(k32+dbuf): ['-18.375     ', '1.76269531  ', '-3.64648438 '], time:0.047631ms
           out_f16x8pack(k32+dbuf+async): ['-18.375     ', '1.76269531  ', '-3.64648438 '], time:0.046644ms
           out_f16x8pack(k32+dbuf+t16x8): ['-18.375     ', '1.76269531  ', '-3.64648438 '], time:0.050278ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['-18.375     ', '1.76269531  ', '-3.64648438 '], time:0.048027ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['-18.421875  ', '1.76855469  ', '-3.63671875 '], time:0.097446ms
                     out_f16wmma(mma4x2): ['-18.421875  ', '1.76855469  ', '-3.63671875 '], time:0.053015ms
             out_f16wmma(mma4x2+warp2x4): ['-18.421875  ', '1.76855469  ', '-3.63671875 '], time:0.044241ms
       out_f16wmma(mma4x2+warp2x4+async): ['-18.421875  ', '1.76855469  ', '-3.63671875 '], time:0.038147ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['-18.421875  ', '1.76855469  ', '-3.63671875 '], time:0.034943ms
                              out_f16_th: ['-18.40625   ', '1.77636719  ', '-3.64453125 '], time:0.025592ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=1024, N=4096, K=512
                                 out_f16: ['-5.63671875 ', '13.6328125  ', '-9.1875     '], time:1.198769ms
                             out_f16(sk): ['-5.63671875 ', '13.6328125  ', '-9.1875     '], time:0.893950ms
                  out_f16x4pack(t4x4bcf): ['-5.63671875 ', '13.6328125  ', '-9.1875     '], time:0.139127ms
               out_f16x4pack(t4x4offset): ['-5.63671875 ', '13.6328125  ', '-9.1875     '], time:0.136619ms
                       out_f16x4(t8x8sk): ['-5.63671875 ', '13.6328125  ', '-9.1875     '], time:0.120821ms
                      out_f16x4(t8x8bcf): ['-5.63671875 ', '13.6328125  ', '-9.1875     '], time:0.111866ms
                   out_f16x4pack(t8x8sk): ['-5.63671875 ', '13.6328125  ', '-9.1875     '], time:0.110240ms
                      out_f16x4pack(bcf): ['-5.63671875 ', '13.6328125  ', '-9.1875     '], time:0.103559ms
               out_f16x4pack(bcf+offset): ['-5.63671875 ', '13.6328125  ', '-9.1875     '], time:0.103326ms
                      out_f16x8pack(bcf): ['-5.63671875 ', '13.6328125  ', '-9.1875     '], time:0.100265ms
               out_f16x8pack(bcf+offset): ['-5.63671875 ', '13.6328125  ', '-9.1875     '], time:0.099654ms
                 out_f16x8pack(bcf+dbuf): ['-5.63671875 ', '13.6328125  ', '-9.1875     '], time:0.094352ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['-5.63671875 ', '13.6328125  ', '-9.1875     '], time:0.087776ms
          out_f16x8pack(k16+dbuf+offset): ['-5.63671875 ', '13.6328125  ', '-9.1875     '], time:0.087452ms
           out_f16x8pack(k16+dbuf+async): ['-5.63671875 ', '13.6328125  ', '-9.1875     '], time:0.089841ms
                 out_f16x8pack(k32+dbuf): ['-5.63671875 ', '13.6328125  ', '-9.1875     '], time:0.090537ms
           out_f16x8pack(k32+dbuf+async): ['-5.63671875 ', '13.6328125  ', '-9.1875     '], time:0.088868ms
           out_f16x8pack(k32+dbuf+t16x8): ['-5.63671875 ', '13.6328125  ', '-9.1875     '], time:0.095072ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['-5.63671875 ', '13.6328125  ', '-9.1875     '], time:0.090508ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['-5.6640625  ', '13.59375    ', '-9.2265625  '], time:0.191298ms
                     out_f16wmma(mma4x2): ['-5.6640625  ', '13.59375    ', '-9.2265625  '], time:0.099921ms
             out_f16wmma(mma4x2+warp2x4): ['-5.6640625  ', '13.59375    ', '-9.2265625  '], time:0.074449ms
       out_f16wmma(mma4x2+warp2x4+async): ['-5.6640625  ', '13.59375    ', '-9.2265625  '], time:0.064263ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['-5.6640625  ', '13.59375    ', '-9.2265625  '], time:0.057569ms
                              out_f16_th: ['-5.6796875  ', '13.5859375  ', '-9.2265625  '], time:0.045180ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=1024, N=4096, K=1024
                                 out_f16: ['33.40625    ', '-47.28125   ', '13.09375    '], time:2.384095ms
                             out_f16(sk): ['33.40625    ', '-47.28125   ', '13.09375    '], time:1.772037ms
                  out_f16x4pack(t4x4bcf): ['33.40625    ', '-47.28125   ', '13.09375    '], time:0.276551ms
               out_f16x4pack(t4x4offset): ['33.40625    ', '-47.28125   ', '13.09375    '], time:0.271640ms
                       out_f16x4(t8x8sk): ['33.40625    ', '-47.28125   ', '13.09375    '], time:0.230780ms
                      out_f16x4(t8x8bcf): ['33.40625    ', '-47.28125   ', '13.09375    '], time:0.219493ms
                   out_f16x4pack(t8x8sk): ['33.40625    ', '-47.28125   ', '13.09375    '], time:0.215173ms
                      out_f16x4pack(bcf): ['33.40625    ', '-47.28125   ', '13.09375    '], time:0.202866ms
               out_f16x4pack(bcf+offset): ['33.40625    ', '-47.28125   ', '13.09375    '], time:0.202212ms
                      out_f16x8pack(bcf): ['33.40625    ', '-47.28125   ', '13.09375    '], time:0.196414ms
               out_f16x8pack(bcf+offset): ['33.40625    ', '-47.28125   ', '13.09375    '], time:0.194912ms
                 out_f16x8pack(bcf+dbuf): ['33.40625    ', '-47.28125   ', '13.09375    '], time:0.184503ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['33.40625    ', '-47.28125   ', '13.09375    '], time:0.171151ms
          out_f16x8pack(k16+dbuf+offset): ['33.40625    ', '-47.28125   ', '13.09375    '], time:0.170736ms
           out_f16x8pack(k16+dbuf+async): ['33.40625    ', '-47.28125   ', '13.09375    '], time:0.175338ms
                 out_f16x8pack(k32+dbuf): ['33.40625    ', '-47.28125   ', '13.09375    '], time:0.176711ms
           out_f16x8pack(k32+dbuf+async): ['33.40625    ', '-47.28125   ', '13.09375    '], time:0.173063ms
           out_f16x8pack(k32+dbuf+t16x8): ['33.40625    ', '-47.28125   ', '13.09375    '], time:0.186300ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['33.40625    ', '-47.28125   ', '13.09375    '], time:0.176334ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['33.375      ', '-47.1875    ', '13.1640625  '], time:0.379415ms
                     out_f16wmma(mma4x2): ['33.375      ', '-47.1875    ', '13.1640625  '], time:0.193920ms
             out_f16wmma(mma4x2+warp2x4): ['33.375      ', '-47.1875    ', '13.1640625  '], time:0.134759ms
       out_f16wmma(mma4x2+warp2x4+async): ['33.375      ', '-47.1875    ', '13.1640625  '], time:0.116634ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['33.375      ', '-47.1875    ', '13.1640625  '], time:0.103126ms
                              out_f16_th: ['33.59375    ', '-47.25      ', '13.1796875  '], time:0.084305ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=2048, N=1024, K=256
                                 out_f16: ['-35.40625   ', '23.265625   ', '-17.828125  '], time:0.309739ms
                             out_f16(sk): ['-35.40625   ', '23.265625   ', '-17.828125  '], time:0.232854ms
                  out_f16x4pack(t4x4bcf): ['-35.40625   ', '23.265625   ', '-17.828125  '], time:0.038009ms
               out_f16x4pack(t4x4offset): ['-35.40625   ', '23.265625   ', '-17.828125  '], time:0.037336ms
                       out_f16x4(t8x8sk): ['-35.40625   ', '23.265625   ', '-17.828125  '], time:0.044756ms
                      out_f16x4(t8x8bcf): ['-35.40625   ', '23.265625   ', '-17.828125  '], time:0.041194ms
                   out_f16x4pack(t8x8sk): ['-35.40625   ', '23.265625   ', '-17.828125  '], time:0.040617ms
                      out_f16x4pack(bcf): ['-35.40625   ', '23.265625   ', '-17.828125  '], time:0.038404ms
               out_f16x4pack(bcf+offset): ['-35.40625   ', '23.265625   ', '-17.828125  '], time:0.038352ms
                      out_f16x8pack(bcf): ['-35.40625   ', '23.265625   ', '-17.828125  '], time:0.036693ms
               out_f16x8pack(bcf+offset): ['-35.40625   ', '23.265625   ', '-17.828125  '], time:0.036521ms
                 out_f16x8pack(bcf+dbuf): ['-35.40625   ', '23.265625   ', '-17.828125  '], time:0.034108ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['-35.40625   ', '23.265625   ', '-17.828125  '], time:0.031309ms
          out_f16x8pack(k16+dbuf+offset): ['-35.40625   ', '23.265625   ', '-17.828125  '], time:0.031219ms
           out_f16x8pack(k16+dbuf+async): ['-35.40625   ', '23.265625   ', '-17.828125  '], time:0.032020ms
                 out_f16x8pack(k32+dbuf): ['-35.40625   ', '23.265625   ', '-17.828125  '], time:0.032616ms
           out_f16x8pack(k32+dbuf+async): ['-35.40625   ', '23.265625   ', '-17.828125  '], time:0.031824ms
           out_f16x8pack(k32+dbuf+t16x8): ['-35.40625   ', '23.265625   ', '-17.828125  '], time:0.034928ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['-35.40625   ', '23.265625   ', '-17.828125  '], time:0.033135ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['-35.25      ', '23.21875    ', '-17.84375   '], time:0.050092ms
                     out_f16wmma(mma4x2): ['-35.25      ', '23.21875    ', '-17.84375   '], time:0.029101ms
             out_f16wmma(mma4x2+warp2x4): ['-35.25      ', '23.21875    ', '-17.84375   '], time:0.024171ms
       out_f16wmma(mma4x2+warp2x4+async): ['-35.25      ', '23.21875    ', '-17.84375   '], time:0.024681ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['-35.25      ', '23.21875    ', '-17.84375   '], time:0.022678ms
                              out_f16_th: ['-35.28125   ', '23.171875   ', '-17.84375   '], time:0.018239ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=2048, N=1024, K=512
                                 out_f16: ['16.8125     ', '20.0625     ', '22.109375   '], time:0.611677ms
                             out_f16(sk): ['16.8125     ', '20.0625     ', '22.109375   '], time:0.456862ms
                  out_f16x4pack(t4x4bcf): ['16.8125     ', '20.0625     ', '22.109375   '], time:0.073504ms
               out_f16x4pack(t4x4offset): ['16.8125     ', '20.0625     ', '22.109375   '], time:0.072174ms
                       out_f16x4(t8x8sk): ['16.8125     ', '20.0625     ', '22.109375   '], time:0.084033ms
                      out_f16x4(t8x8bcf): ['16.8125     ', '20.0625     ', '22.109375   '], time:0.079384ms
                   out_f16x4pack(t8x8sk): ['16.8125     ', '20.0625     ', '22.109375   '], time:0.077415ms
                      out_f16x4pack(bcf): ['16.8125     ', '20.0625     ', '22.109375   '], time:0.073833ms
               out_f16x4pack(bcf+offset): ['16.8125     ', '20.0625     ', '22.109375   '], time:0.073853ms
                      out_f16x8pack(bcf): ['16.8125     ', '20.0625     ', '22.109375   '], time:0.070457ms
               out_f16x8pack(bcf+offset): ['16.8125     ', '20.0625     ', '22.109375   '], time:0.070148ms
                 out_f16x8pack(bcf+dbuf): ['16.8125     ', '20.0625     ', '22.109375   '], time:0.065165ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['16.8125     ', '20.0625     ', '22.109375   '], time:0.059557ms
          out_f16x8pack(k16+dbuf+offset): ['16.8125     ', '20.0625     ', '22.109375   '], time:0.059338ms
           out_f16x8pack(k16+dbuf+async): ['16.8125     ', '20.0625     ', '22.109375   '], time:0.061054ms
                 out_f16x8pack(k32+dbuf): ['16.8125     ', '20.0625     ', '22.109375   '], time:0.061870ms
           out_f16x8pack(k32+dbuf+async): ['16.8125     ', '20.0625     ', '22.109375   '], time:0.060415ms
           out_f16x8pack(k32+dbuf+t16x8): ['16.8125     ', '20.0625     ', '22.109375   '], time:0.065889ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['16.8125     ', '20.0625     ', '22.109375   '], time:0.062232ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['16.8125     ', '19.984375   ', '22.15625    '], time:0.097713ms
                     out_f16wmma(mma4x2): ['16.8125     ', '19.984375   ', '22.15625    '], time:0.053744ms
             out_f16wmma(mma4x2+warp2x4): ['16.8125     ', '19.984375   ', '22.15625    '], time:0.043111ms
       out_f16wmma(mma4x2+warp2x4+async): ['16.8125     ', '19.984375   ', '22.15625    '], time:0.043545ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['16.8125     ', '19.984375   ', '22.15625    '], time:0.039206ms
                              out_f16_th: ['16.828125   ', '20.03125    ', '22.140625   '], time:0.031424ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=2048, N=1024, K=1024
                                 out_f16: ['-58.375     ', '-25.890625  ', '-0.04962158 '], time:1.215239ms
                             out_f16(sk): ['-58.375     ', '-25.890625  ', '-0.04962158 '], time:0.904570ms
                  out_f16x4pack(t4x4bcf): ['-58.375     ', '-25.890625  ', '-0.04962158 '], time:0.145087ms
               out_f16x4pack(t4x4offset): ['-58.375     ', '-25.890625  ', '-0.04962158 '], time:0.142665ms
                       out_f16x4(t8x8sk): ['-58.375     ', '-25.890625  ', '-0.04962158 '], time:0.162268ms
                      out_f16x4(t8x8bcf): ['-58.375     ', '-25.890625  ', '-0.04962158 '], time:0.155859ms
                   out_f16x4pack(t8x8sk): ['-58.375     ', '-25.890625  ', '-0.04962158 '], time:0.150948ms
                      out_f16x4pack(bcf): ['-58.375     ', '-25.890625  ', '-0.04962158 '], time:0.144429ms
               out_f16x4pack(bcf+offset): ['-58.375     ', '-25.890625  ', '-0.04962158 '], time:0.144320ms
                      out_f16x8pack(bcf): ['-58.375     ', '-25.890625  ', '-0.04962158 '], time:0.137944ms
               out_f16x8pack(bcf+offset): ['-58.375     ', '-25.890625  ', '-0.04962158 '], time:0.137277ms
                 out_f16x8pack(bcf+dbuf): ['-58.375     ', '-25.890625  ', '-0.04962158 '], time:0.127568ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['-58.375     ', '-25.890625  ', '-0.04962158 '], time:0.116444ms
          out_f16x8pack(k16+dbuf+offset): ['-58.375     ', '-25.890625  ', '-0.04962158 '], time:0.116048ms
           out_f16x8pack(k16+dbuf+async): ['-58.375     ', '-25.890625  ', '-0.04962158 '], time:0.119395ms
                 out_f16x8pack(k32+dbuf): ['-58.375     ', '-25.890625  ', '-0.04962158 '], time:0.121017ms
           out_f16x8pack(k32+dbuf+async): ['-58.375     ', '-25.890625  ', '-0.04962158 '], time:0.117860ms
           out_f16x8pack(k32+dbuf+t16x8): ['-58.375     ', '-25.890625  ', '-0.04962158 '], time:0.128183ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['-58.375     ', '-25.890625  ', '-0.04962158 '], time:0.120959ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['-58.375     ', '-25.90625   ', '-0.08874512 '], time:0.193443ms
                     out_f16wmma(mma4x2): ['-58.375     ', '-25.90625   ', '-0.08874512 '], time:0.103273ms
             out_f16wmma(mma4x2+warp2x4): ['-58.375     ', '-25.90625   ', '-0.08874512 '], time:0.080471ms
       out_f16wmma(mma4x2+warp2x4+async): ['-58.375     ', '-25.90625   ', '-0.08874512 '], time:0.081129ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['-58.375     ', '-25.90625   ', '-0.08874512 '], time:0.072947ms
                              out_f16_th: ['-58.375     ', '-25.90625   ', '-0.09619141 '], time:0.057721ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=2048, N=2048, K=256
                                 out_f16: ['-18.609375  ', '3.57421875  ', '39.625      '], time:0.606441ms
                             out_f16(sk): ['-18.609375  ', '3.57421875  ', '39.625      '], time:0.455413ms
                  out_f16x4pack(t4x4bcf): ['-18.609375  ', '3.57421875  ', '39.625      '], time:0.070772ms
               out_f16x4pack(t4x4offset): ['-18.609375  ', '3.57421875  ', '39.625      '], time:0.069413ms
                       out_f16x4(t8x8sk): ['-18.609375  ', '3.57421875  ', '39.625      '], time:0.067253ms
                      out_f16x4(t8x8bcf): ['-18.609375  ', '3.57421875  ', '39.625      '], time:0.058079ms
                   out_f16x4pack(t8x8sk): ['-18.609375  ', '3.57421875  ', '39.625      '], time:0.058904ms
                      out_f16x4pack(bcf): ['-18.609375  ', '3.57421875  ', '39.625      '], time:0.053906ms
               out_f16x4pack(bcf+offset): ['-18.609375  ', '3.57421875  ', '39.625      '], time:0.053844ms
                      out_f16x8pack(bcf): ['-18.609375  ', '3.57421875  ', '39.625      '], time:0.052543ms
               out_f16x8pack(bcf+offset): ['-18.609375  ', '3.57421875  ', '39.625      '], time:0.052328ms
                 out_f16x8pack(bcf+dbuf): ['-18.609375  ', '3.57421875  ', '39.625      '], time:0.049324ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['-18.609375  ', '3.57421875  ', '39.625      '], time:0.046191ms
          out_f16x8pack(k16+dbuf+offset): ['-18.609375  ', '3.57421875  ', '39.625      '], time:0.046134ms
           out_f16x8pack(k16+dbuf+async): ['-18.609375  ', '3.57421875  ', '39.625      '], time:0.047202ms
                 out_f16x8pack(k32+dbuf): ['-18.609375  ', '3.57421875  ', '39.625      '], time:0.047693ms
           out_f16x8pack(k32+dbuf+async): ['-18.609375  ', '3.57421875  ', '39.625      '], time:0.046687ms
           out_f16x8pack(k32+dbuf+t16x8): ['-18.609375  ', '3.57421875  ', '39.625      '], time:0.050125ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['-18.609375  ', '3.57421875  ', '39.625      '], time:0.047784ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['-18.5625    ', '3.57617188  ', '39.5        '], time:0.096774ms
                     out_f16wmma(mma4x2): ['-18.5625    ', '3.57617188  ', '39.5        '], time:0.053339ms
             out_f16wmma(mma4x2+warp2x4): ['-18.5625    ', '3.57617188  ', '39.5        '], time:0.044169ms
       out_f16wmma(mma4x2+warp2x4+async): ['-18.5625    ', '3.57617188  ', '39.5        '], time:0.038280ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['-18.5625    ', '3.57617188  ', '39.5        '], time:0.035343ms
                              out_f16_th: ['-18.609375  ', '3.58007812  ', '39.53125    '], time:0.025606ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=2048, N=2048, K=512
                                 out_f16: ['-32.6875    ', '-5.71484375 ', '-28.171875  '], time:1.198831ms
                             out_f16(sk): ['-32.6875    ', '-5.71484375 ', '-28.171875  '], time:0.895257ms
                  out_f16x4pack(t4x4bcf): ['-32.6875    ', '-5.71484375 ', '-28.171875  '], time:0.139270ms
               out_f16x4pack(t4x4offset): ['-32.6875    ', '-5.71484375 ', '-28.171875  '], time:0.136724ms
                       out_f16x4(t8x8sk): ['-32.6875    ', '-5.71484375 ', '-28.171875  '], time:0.122323ms
                      out_f16x4(t8x8bcf): ['-32.6875    ', '-5.71484375 ', '-28.171875  '], time:0.111880ms
                   out_f16x4pack(t8x8sk): ['-32.6875    ', '-5.71484375 ', '-28.171875  '], time:0.111170ms
                      out_f16x4pack(bcf): ['-32.6875    ', '-5.71484375 ', '-28.171875  '], time:0.103440ms
               out_f16x4pack(bcf+offset): ['-32.6875    ', '-5.71484375 ', '-28.171875  '], time:0.103326ms
                      out_f16x8pack(bcf): ['-32.6875    ', '-5.71484375 ', '-28.171875  '], time:0.100336ms
               out_f16x8pack(bcf+offset): ['-32.6875    ', '-5.71484375 ', '-28.171875  '], time:0.099821ms
                 out_f16x8pack(bcf+dbuf): ['-32.6875    ', '-5.71484375 ', '-28.171875  '], time:0.094533ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['-32.6875    ', '-5.71484375 ', '-28.171875  '], time:0.087891ms
          out_f16x8pack(k16+dbuf+offset): ['-32.6875    ', '-5.71484375 ', '-28.171875  '], time:0.087614ms
           out_f16x8pack(k16+dbuf+async): ['-32.6875    ', '-5.71484375 ', '-28.171875  '], time:0.090036ms
                 out_f16x8pack(k32+dbuf): ['-32.6875    ', '-5.71484375 ', '-28.171875  '], time:0.090804ms
           out_f16x8pack(k32+dbuf+async): ['-32.6875    ', '-5.71484375 ', '-28.171875  '], time:0.088954ms
           out_f16x8pack(k32+dbuf+t16x8): ['-32.6875    ', '-5.71484375 ', '-28.171875  '], time:0.095024ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['-32.6875    ', '-5.71484375 ', '-28.171875  '], time:0.090494ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['-32.6875    ', '-5.7890625  ', '-28.21875   '], time:0.189743ms
                     out_f16wmma(mma4x2): ['-32.6875    ', '-5.7890625  ', '-28.21875   '], time:0.100169ms
             out_f16wmma(mma4x2+warp2x4): ['-32.6875    ', '-5.7890625  ', '-28.21875   '], time:0.074544ms
       out_f16wmma(mma4x2+warp2x4+async): ['-32.6875    ', '-5.7890625  ', '-28.21875   '], time:0.064483ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['-32.6875    ', '-5.7890625  ', '-28.21875   '], time:0.057878ms
                              out_f16_th: ['-32.6875    ', '-5.78515625 ', '-28.203125  '], time:0.045185ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=2048, N=2048, K=1024
                                 out_f16: ['-38.21875   ', '43.9375     ', '-57.4375    '], time:2.384048ms
                             out_f16(sk): ['-38.21875   ', '43.9375     ', '-57.4375    '], time:1.774373ms
                  out_f16x4pack(t4x4bcf): ['-38.21875   ', '43.9375     ', '-57.4375    '], time:0.276599ms
               out_f16x4pack(t4x4offset): ['-38.21875   ', '43.9375     ', '-57.4375    '], time:0.271668ms
                       out_f16x4(t8x8sk): ['-38.21875   ', '43.9375     ', '-57.4375    '], time:0.232229ms
                      out_f16x4(t8x8bcf): ['-38.21875   ', '43.9375     ', '-57.4375    '], time:0.219541ms
                   out_f16x4pack(t8x8sk): ['-38.21875   ', '43.9375     ', '-57.4375    '], time:0.216312ms
                      out_f16x4pack(bcf): ['-38.21875   ', '43.9375     ', '-57.4375    '], time:0.202646ms
               out_f16x4pack(bcf+offset): ['-38.21875   ', '43.9375     ', '-57.4375    '], time:0.202289ms
                      out_f16x8pack(bcf): ['-38.21875   ', '43.9375     ', '-57.4375    '], time:0.196514ms
               out_f16x8pack(bcf+offset): ['-38.21875   ', '43.9375     ', '-57.4375    '], time:0.195346ms
                 out_f16x8pack(bcf+dbuf): ['-38.21875   ', '43.9375     ', '-57.4375    '], time:0.184565ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['-38.21875   ', '43.9375     ', '-57.4375    '], time:0.171204ms
          out_f16x8pack(k16+dbuf+offset): ['-38.21875   ', '43.9375     ', '-57.4375    '], time:0.170846ms
           out_f16x8pack(k16+dbuf+async): ['-38.21875   ', '43.9375     ', '-57.4375    '], time:0.175514ms
                 out_f16x8pack(k32+dbuf): ['-38.21875   ', '43.9375     ', '-57.4375    '], time:0.177193ms
           out_f16x8pack(k32+dbuf+async): ['-38.21875   ', '43.9375     ', '-57.4375    '], time:0.173178ms
           out_f16x8pack(k32+dbuf+t16x8): ['-38.21875   ', '43.9375     ', '-57.4375    '], time:0.186229ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['-38.21875   ', '43.9375     ', '-57.4375    '], time:0.176249ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['-37.8125    ', '44.09375    ', '-57.28125   '], time:0.375648ms
                     out_f16wmma(mma4x2): ['-37.8125    ', '44.09375    ', '-57.28125   '], time:0.194054ms
             out_f16wmma(mma4x2+warp2x4): ['-37.8125    ', '44.09375    ', '-57.28125   '], time:0.155973ms
       out_f16wmma(mma4x2+warp2x4+async): ['-37.8125    ', '44.09375    ', '-57.28125   '], time:0.116806ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['-37.8125    ', '44.09375    ', '-57.28125   '], time:0.103312ms
                              out_f16_th: ['-37.8125    ', '44.0625     ', '-57.25      '], time:0.084386ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=2048, N=4096, K=256
                                 out_f16: ['-29.328125  ', '-7.12109375 ', '-0.53515625 '], time:1.205297ms
                             out_f16(sk): ['-29.328125  ', '-7.12109375 ', '-0.53515625 '], time:0.903292ms
                  out_f16x4pack(t4x4bcf): ['-29.328125  ', '-7.12109375 ', '-0.53515625 '], time:0.131102ms
               out_f16x4pack(t4x4offset): ['-29.328125  ', '-7.12109375 ', '-0.53515625 '], time:0.128646ms
                       out_f16x4(t8x8sk): ['-29.328125  ', '-7.12109375 ', '-0.53515625 '], time:0.124044ms
                      out_f16x4(t8x8bcf): ['-29.328125  ', '-7.12109375 ', '-0.53515625 '], time:0.108495ms
                   out_f16x4pack(t8x8sk): ['-29.328125  ', '-7.12109375 ', '-0.53515625 '], time:0.108433ms
                      out_f16x4pack(bcf): ['-29.328125  ', '-7.12109375 ', '-0.53515625 '], time:0.100098ms
               out_f16x4pack(bcf+offset): ['-29.328125  ', '-7.12109375 ', '-0.53515625 '], time:0.099754ms
                      out_f16x8pack(bcf): ['-29.328125  ', '-7.12109375 ', '-0.53515625 '], time:0.096822ms
               out_f16x8pack(bcf+offset): ['-29.328125  ', '-7.12109375 ', '-0.53515625 '], time:0.096073ms
                 out_f16x8pack(bcf+dbuf): ['-29.328125  ', '-7.12109375 ', '-0.53515625 '], time:0.093141ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['-29.328125  ', '-7.12109375 ', '-0.53515625 '], time:0.088382ms
          out_f16x8pack(k16+dbuf+offset): ['-29.328125  ', '-7.12109375 ', '-0.53515625 '], time:0.088272ms
           out_f16x8pack(k16+dbuf+async): ['-29.328125  ', '-7.12109375 ', '-0.53515625 '], time:0.090036ms
                 out_f16x8pack(k32+dbuf): ['-29.328125  ', '-7.12109375 ', '-0.53515625 '], time:0.089822ms
           out_f16x8pack(k32+dbuf+async): ['-29.328125  ', '-7.12109375 ', '-0.53515625 '], time:0.088663ms
           out_f16x8pack(k32+dbuf+t16x8): ['-29.328125  ', '-7.12109375 ', '-0.53515625 '], time:0.094099ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['-29.328125  ', '-7.12109375 ', '-0.53515625 '], time:0.089703ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['-29.453125  ', '-7.12890625 ', '-0.56640625 '], time:0.189900ms
                     out_f16wmma(mma4x2): ['-29.453125  ', '-7.12890625 ', '-0.56640625 '], time:0.101275ms
             out_f16wmma(mma4x2+warp2x4): ['-29.453125  ', '-7.12890625 ', '-0.56640625 '], time:0.080466ms
       out_f16wmma(mma4x2+warp2x4+async): ['-29.453125  ', '-7.12890625 ', '-0.56640625 '], time:0.072355ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['-29.453125  ', '-7.12890625 ', '-0.56640625 '], time:0.065742ms
                              out_f16_th: ['-29.4375    ', '-7.11328125 ', '-0.55175781 '], time:0.046864ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=2048, N=4096, K=512
                                 out_f16: ['6.99609375  ', '54.3125     ', '4.67578125  '], time:2.384925ms
                             out_f16(sk): ['6.99609375  ', '54.3125     ', '4.67578125  '], time:1.779170ms
                  out_f16x4pack(t4x4bcf): ['6.99609375  ', '54.3125     ', '4.67578125  '], time:0.259366ms
               out_f16x4pack(t4x4offset): ['6.99609375  ', '54.3125     ', '4.67578125  '], time:0.254574ms
                       out_f16x4(t8x8sk): ['6.99609375  ', '54.3125     ', '4.67578125  '], time:0.230522ms
                      out_f16x4(t8x8bcf): ['6.99609375  ', '54.3125     ', '4.67578125  '], time:0.211601ms
                   out_f16x4pack(t8x8sk): ['6.99609375  ', '54.3125     ', '4.67578125  '], time:0.208082ms
                      out_f16x4pack(bcf): ['6.99609375  ', '54.3125     ', '4.67578125  '], time:0.194921ms
               out_f16x4pack(bcf+offset): ['6.99609375  ', '54.3125     ', '4.67578125  '], time:0.194206ms
                      out_f16x8pack(bcf): ['6.99609375  ', '54.3125     ', '4.67578125  '], time:0.188456ms
               out_f16x8pack(bcf+offset): ['6.99609375  ', '54.3125     ', '4.67578125  '], time:0.186882ms
                 out_f16x8pack(bcf+dbuf): ['6.99609375  ', '54.3125     ', '4.67578125  '], time:0.181394ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['6.99609375  ', '54.3125     ', '4.67578125  '], time:0.170593ms
          out_f16x8pack(k16+dbuf+offset): ['6.99609375  ', '54.3125     ', '4.67578125  '], time:0.170231ms
           out_f16x8pack(k16+dbuf+async): ['6.99609375  ', '54.3125     ', '4.67578125  '], time:0.174313ms
                 out_f16x8pack(k32+dbuf): ['6.99609375  ', '54.3125     ', '4.67578125  '], time:0.174580ms
           out_f16x8pack(k32+dbuf+async): ['6.99609375  ', '54.3125     ', '4.67578125  '], time:0.171957ms
           out_f16x8pack(k32+dbuf+t16x8): ['6.99609375  ', '54.3125     ', '4.67578125  '], time:0.181713ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['6.99609375  ', '54.3125     ', '4.67578125  '], time:0.173817ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['6.953125    ', '54.28125    ', '4.6640625   '], time:0.373082ms
                     out_f16wmma(mma4x2): ['6.953125    ', '54.28125    ', '4.6640625   '], time:0.192552ms
             out_f16wmma(mma4x2+warp2x4): ['6.953125    ', '54.28125    ', '4.6640625   '], time:0.136008ms
       out_f16wmma(mma4x2+warp2x4+async): ['6.953125    ', '54.28125    ', '4.6640625   '], time:0.124269ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['6.953125    ', '54.28125    ', '4.6640625   '], time:0.108714ms
                              out_f16_th: ['6.96484375  ', '54.3125     ', '4.67578125  '], time:0.090094ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=2048, N=4096, K=1024
                                 out_f16: ['-56.5625    ', '-12.15625   ', '103.3125    '], time:4.742117ms
                             out_f16(sk): ['-56.5625    ', '-12.15625   ', '103.3125    '], time:3.525944ms
                  out_f16x4pack(t4x4bcf): ['-56.5625    ', '-12.15625   ', '103.3125    '], time:0.515971ms
               out_f16x4pack(t4x4offset): ['-56.5625    ', '-12.15625   ', '103.3125    '], time:0.506635ms
                       out_f16x4(t8x8sk): ['-56.5625    ', '-12.15625   ', '103.3125    '], time:0.442162ms
                      out_f16x4(t8x8bcf): ['-56.5625    ', '-12.15625   ', '103.3125    '], time:0.417786ms
                   out_f16x4pack(t8x8sk): ['-56.5625    ', '-12.15625   ', '103.3125    '], time:0.407367ms
                      out_f16x4pack(bcf): ['-56.5625    ', '-12.15625   ', '103.3125    '], time:0.383983ms
               out_f16x4pack(bcf+offset): ['-56.5625    ', '-12.15625   ', '103.3125    '], time:0.382237ms
                      out_f16x8pack(bcf): ['-56.5625    ', '-12.15625   ', '103.3125    '], time:0.370898ms
               out_f16x8pack(bcf+offset): ['-56.5625    ', '-12.15625   ', '103.3125    '], time:0.367818ms
                 out_f16x8pack(bcf+dbuf): ['-56.5625    ', '-12.15625   ', '103.3125    '], time:0.357151ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['-56.5625    ', '-12.15625   ', '103.3125    '], time:0.335803ms
          out_f16x8pack(k16+dbuf+offset): ['-56.5625    ', '-12.15625   ', '103.3125    '], time:0.335407ms
           out_f16x8pack(k16+dbuf+async): ['-56.5625    ', '-12.15625   ', '103.3125    '], time:0.343256ms
                 out_f16x8pack(k32+dbuf): ['-56.5625    ', '-12.15625   ', '103.3125    '], time:0.343990ms
           out_f16x8pack(k32+dbuf+async): ['-56.5625    ', '-12.15625   ', '103.3125    '], time:0.338531ms
           out_f16x8pack(k32+dbuf+t16x8): ['-56.5625    ', '-12.15625   ', '103.3125    '], time:0.357404ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['-56.5625    ', '-12.15625   ', '103.3125    '], time:0.342174ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['-56.84375   ', '-12.2421875 ', '103.0625    '], time:0.739069ms
                     out_f16wmma(mma4x2): ['-56.84375   ', '-12.2421875 ', '103.0625    '], time:0.375371ms
             out_f16wmma(mma4x2+warp2x4): ['-56.84375   ', '-12.2421875 ', '103.0625    '], time:0.247140ms
       out_f16wmma(mma4x2+warp2x4+async): ['-56.84375   ', '-12.2421875 ', '103.0625    '], time:0.227537ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['-56.84375   ', '-12.2421875 ', '103.0625    '], time:0.195603ms
                              out_f16_th: ['-56.8125    ', '-12.2265625 ', '103.125     '], time:0.168357ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=4096, N=1024, K=256
                                 out_f16: ['-28.484375  ', '-9.0625     ', '15.8671875  '], time:0.606337ms
                             out_f16(sk): ['-28.484375  ', '-9.0625     ', '15.8671875  '], time:0.455542ms
                  out_f16x4pack(t4x4bcf): ['-28.484375  ', '-9.0625     ', '15.8671875  '], time:0.070791ms
               out_f16x4pack(t4x4offset): ['-28.484375  ', '-9.0625     ', '15.8671875  '], time:0.069504ms
                       out_f16x4(t8x8sk): ['-28.484375  ', '-9.0625     ', '15.8671875  '], time:0.065732ms
                      out_f16x4(t8x8bcf): ['-28.484375  ', '-9.0625     ', '15.8671875  '], time:0.058069ms
                   out_f16x4pack(t8x8sk): ['-28.484375  ', '-9.0625     ', '15.8671875  '], time:0.058184ms
                      out_f16x4pack(bcf): ['-28.484375  ', '-9.0625     ', '15.8671875  '], time:0.053992ms
               out_f16x4pack(bcf+offset): ['-28.484375  ', '-9.0625     ', '15.8671875  '], time:0.053816ms
                      out_f16x8pack(bcf): ['-28.484375  ', '-9.0625     ', '15.8671875  '], time:0.052595ms
               out_f16x8pack(bcf+offset): ['-28.484375  ', '-9.0625     ', '15.8671875  '], time:0.052247ms
                 out_f16x8pack(bcf+dbuf): ['-28.484375  ', '-9.0625     ', '15.8671875  '], time:0.049267ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['-28.484375  ', '-9.0625     ', '15.8671875  '], time:0.046144ms
          out_f16x8pack(k16+dbuf+offset): ['-28.484375  ', '-9.0625     ', '15.8671875  '], time:0.046101ms
           out_f16x8pack(k16+dbuf+async): ['-28.484375  ', '-9.0625     ', '15.8671875  '], time:0.047131ms
                 out_f16x8pack(k32+dbuf): ['-28.484375  ', '-9.0625     ', '15.8671875  '], time:0.047574ms
           out_f16x8pack(k32+dbuf+async): ['-28.484375  ', '-9.0625     ', '15.8671875  '], time:0.046635ms
           out_f16x8pack(k32+dbuf+t16x8): ['-28.484375  ', '-9.0625     ', '15.8671875  '], time:0.050182ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['-28.484375  ', '-9.0625     ', '15.8671875  '], time:0.047731ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['-28.484375  ', '-9.0390625  ', '15.859375   '], time:0.095749ms
                     out_f16wmma(mma4x2): ['-28.484375  ', '-9.0390625  ', '15.859375   '], time:0.053320ms
             out_f16wmma(mma4x2+warp2x4): ['-28.484375  ', '-9.0390625  ', '15.859375   '], time:0.043993ms
       out_f16wmma(mma4x2+warp2x4+async): ['-28.484375  ', '-9.0390625  ', '15.859375   '], time:0.038056ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['-28.484375  ', '-9.0390625  ', '15.859375   '], time:0.035138ms
                              out_f16_th: ['-28.46875   ', '-9.0390625  ', '15.84375    '], time:0.025487ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=4096, N=1024, K=512
                                 out_f16: ['-8.8359375  ', '-14.046875  ', '-18.734375  '], time:1.198897ms
                             out_f16(sk): ['-8.8359375  ', '-14.046875  ', '-18.734375  '], time:0.895262ms
                  out_f16x4pack(t4x4bcf): ['-8.8359375  ', '-14.046875  ', '-18.734375  '], time:0.139356ms
               out_f16x4pack(t4x4offset): ['-8.8359375  ', '-14.046875  ', '-18.734375  '], time:0.136843ms
                       out_f16x4(t8x8sk): ['-8.8359375  ', '-14.046875  ', '-18.734375  '], time:0.120826ms
                      out_f16x4(t8x8bcf): ['-8.8359375  ', '-14.046875  ', '-18.734375  '], time:0.111818ms
                   out_f16x4pack(t8x8sk): ['-8.8359375  ', '-14.046875  ', '-18.734375  '], time:0.110593ms
                      out_f16x4pack(bcf): ['-8.8359375  ', '-14.046875  ', '-18.734375  '], time:0.103731ms
               out_f16x4pack(bcf+offset): ['-8.8359375  ', '-14.046875  ', '-18.734375  '], time:0.103421ms
                      out_f16x8pack(bcf): ['-8.8359375  ', '-14.046875  ', '-18.734375  '], time:0.100379ms
               out_f16x8pack(bcf+offset): ['-8.8359375  ', '-14.046875  ', '-18.734375  '], time:0.099711ms
                 out_f16x8pack(bcf+dbuf): ['-8.8359375  ', '-14.046875  ', '-18.734375  '], time:0.094395ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['-8.8359375  ', '-14.046875  ', '-18.734375  '], time:0.087819ms
          out_f16x8pack(k16+dbuf+offset): ['-8.8359375  ', '-14.046875  ', '-18.734375  '], time:0.087538ms
           out_f16x8pack(k16+dbuf+async): ['-8.8359375  ', '-14.046875  ', '-18.734375  '], time:0.089936ms
                 out_f16x8pack(k32+dbuf): ['-8.8359375  ', '-14.046875  ', '-18.734375  '], time:0.090570ms
           out_f16x8pack(k32+dbuf+async): ['-8.8359375  ', '-14.046875  ', '-18.734375  '], time:0.088878ms
           out_f16x8pack(k32+dbuf+t16x8): ['-8.8359375  ', '-14.046875  ', '-18.734375  '], time:0.095172ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['-8.8359375  ', '-14.046875  ', '-18.734375  '], time:0.090537ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['-8.78125    ', '-14.0625    ', '-18.796875  '], time:0.187411ms
                     out_f16wmma(mma4x2): ['-8.78125    ', '-14.0625    ', '-18.796875  '], time:0.100183ms
             out_f16wmma(mma4x2+warp2x4): ['-8.78125    ', '-14.0625    ', '-18.796875  '], time:0.074439ms
       out_f16wmma(mma4x2+warp2x4+async): ['-8.78125    ', '-14.0625    ', '-18.796875  '], time:0.064349ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['-8.78125    ', '-14.0625    ', '-18.796875  '], time:0.057793ms
                              out_f16_th: ['-8.796875   ', '-14.0546875 ', '-18.78125   '], time:0.045075ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=4096, N=1024, K=1024
                                 out_f16: ['9.859375    ', '-44.78125   ', '18.03125    '], time:2.383838ms
                             out_f16(sk): ['9.859375    ', '-44.78125   ', '18.03125    '], time:1.776695ms
                  out_f16x4pack(t4x4bcf): ['9.859375    ', '-44.78125   ', '18.03125    '], time:0.276794ms
               out_f16x4pack(t4x4offset): ['9.859375    ', '-44.78125   ', '18.03125    '], time:0.271797ms
                       out_f16x4(t8x8sk): ['9.859375    ', '-44.78125   ', '18.03125    '], time:0.230827ms
                      out_f16x4(t8x8bcf): ['9.859375    ', '-44.78125   ', '18.03125    '], time:0.219555ms
                   out_f16x4pack(t8x8sk): ['9.859375    ', '-44.78125   ', '18.03125    '], time:0.215716ms
                      out_f16x4pack(bcf): ['9.859375    ', '-44.78125   ', '18.03125    '], time:0.203094ms
               out_f16x4pack(bcf+offset): ['9.859375    ', '-44.78125   ', '18.03125    '], time:0.202479ms
                      out_f16x8pack(bcf): ['9.859375    ', '-44.78125   ', '18.03125    '], time:0.196457ms
               out_f16x8pack(bcf+offset): ['9.859375    ', '-44.78125   ', '18.03125    '], time:0.195308ms
                 out_f16x8pack(bcf+dbuf): ['9.859375    ', '-44.78125   ', '18.03125    '], time:0.184584ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['9.859375    ', '-44.78125   ', '18.03125    '], time:0.171156ms
          out_f16x8pack(k16+dbuf+offset): ['9.859375    ', '-44.78125   ', '18.03125    '], time:0.170746ms
           out_f16x8pack(k16+dbuf+async): ['9.859375    ', '-44.78125   ', '18.03125    '], time:0.175409ms
                 out_f16x8pack(k32+dbuf): ['9.859375    ', '-44.78125   ', '18.03125    '], time:0.176625ms
           out_f16x8pack(k32+dbuf+async): ['9.859375    ', '-44.78125   ', '18.03125    '], time:0.173020ms
           out_f16x8pack(k32+dbuf+t16x8): ['9.859375    ', '-44.78125   ', '18.03125    '], time:0.186310ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['9.859375    ', '-44.78125   ', '18.03125    '], time:0.176206ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['9.7421875   ', '-44.8125    ', '18.015625   '], time:0.372000ms
                     out_f16wmma(mma4x2): ['9.7421875   ', '-44.8125    ', '18.015625   '], time:0.193892ms
             out_f16wmma(mma4x2+warp2x4): ['9.7421875   ', '-44.8125    ', '18.015625   '], time:0.134826ms
       out_f16wmma(mma4x2+warp2x4+async): ['9.7421875   ', '-44.8125    ', '18.015625   '], time:0.116773ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['9.7421875   ', '-44.8125    ', '18.015625   '], time:0.103359ms
                              out_f16_th: ['9.765625    ', '-44.8125    ', '17.984375   '], time:0.084429ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=4096, N=2048, K=256
                                 out_f16: ['22.328125   ', '-7.375      ', '-16.421875  '], time:1.205440ms
                             out_f16(sk): ['22.328125   ', '-7.375      ', '-16.421875  '], time:0.904360ms
                  out_f16x4pack(t4x4bcf): ['22.328125   ', '-7.375      ', '-16.421875  '], time:0.131235ms
               out_f16x4pack(t4x4offset): ['22.328125   ', '-7.375      ', '-16.421875  '], time:0.128794ms
                       out_f16x4(t8x8sk): ['22.328125   ', '-7.375      ', '-16.421875  '], time:0.128303ms
                      out_f16x4(t8x8bcf): ['22.328125   ', '-7.375      ', '-16.421875  '], time:0.108514ms
                   out_f16x4pack(t8x8sk): ['22.328125   ', '-7.375      ', '-16.421875  '], time:0.109921ms
                      out_f16x4pack(bcf): ['22.328125   ', '-7.375      ', '-16.421875  '], time:0.099945ms
               out_f16x4pack(bcf+offset): ['22.328125   ', '-7.375      ', '-16.421875  '], time:0.099769ms
                      out_f16x8pack(bcf): ['22.328125   ', '-7.375      ', '-16.421875  '], time:0.097132ms
               out_f16x8pack(bcf+offset): ['22.328125   ', '-7.375      ', '-16.421875  '], time:0.096197ms
                 out_f16x8pack(bcf+dbuf): ['22.328125   ', '-7.375      ', '-16.421875  '], time:0.093303ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['22.328125   ', '-7.375      ', '-16.421875  '], time:0.088539ms
          out_f16x8pack(k16+dbuf+offset): ['22.328125   ', '-7.375      ', '-16.421875  '], time:0.088477ms
           out_f16x8pack(k16+dbuf+async): ['22.328125   ', '-7.375      ', '-16.421875  '], time:0.090141ms
                 out_f16x8pack(k32+dbuf): ['22.328125   ', '-7.375      ', '-16.421875  '], time:0.090036ms
           out_f16x8pack(k32+dbuf+async): ['22.328125   ', '-7.375      ', '-16.421875  '], time:0.088735ms
           out_f16x8pack(k32+dbuf+t16x8): ['22.328125   ', '-7.375      ', '-16.421875  '], time:0.093803ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['22.328125   ', '-7.375      ', '-16.421875  '], time:0.089636ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['22.265625   ', '-7.390625   ', '-16.5       '], time:0.188503ms
                     out_f16wmma(mma4x2): ['22.265625   ', '-7.390625   ', '-16.5       '], time:0.101905ms
             out_f16wmma(mma4x2+warp2x4): ['22.265625   ', '-7.390625   ', '-16.5       '], time:0.080490ms
       out_f16wmma(mma4x2+warp2x4+async): ['22.265625   ', '-7.390625   ', '-16.5       '], time:0.072622ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['22.265625   ', '-7.390625   ', '-16.5       '], time:0.066009ms
                              out_f16_th: ['22.265625   ', '-7.3828125  ', '-16.5       '], time:0.046916ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=4096, N=2048, K=512
                                 out_f16: ['8.234375    ', '4.91796875  ', '35.0        '], time:2.384825ms
                             out_f16(sk): ['8.234375    ', '4.91796875  ', '35.0        '], time:1.781507ms
                  out_f16x4pack(t4x4bcf): ['8.234375    ', '4.91796875  ', '35.0        '], time:0.259504ms
               out_f16x4pack(t4x4offset): ['8.234375    ', '4.91796875  ', '35.0        '], time:0.254655ms
                       out_f16x4(t8x8sk): ['8.234375    ', '4.91796875  ', '35.0        '], time:0.234642ms
                      out_f16x4(t8x8bcf): ['8.234375    ', '4.91796875  ', '35.0        '], time:0.211630ms
                   out_f16x4pack(t8x8sk): ['8.234375    ', '4.91796875  ', '35.0        '], time:0.209303ms
                      out_f16x4pack(bcf): ['8.234375    ', '4.91796875  ', '35.0        '], time:0.194678ms
               out_f16x4pack(bcf+offset): ['8.234375    ', '4.91796875  ', '35.0        '], time:0.194192ms
                      out_f16x8pack(bcf): ['8.234375    ', '4.91796875  ', '35.0        '], time:0.188408ms
               out_f16x8pack(bcf+offset): ['8.234375    ', '4.91796875  ', '35.0        '], time:0.187092ms
                 out_f16x8pack(bcf+dbuf): ['8.234375    ', '4.91796875  ', '35.0        '], time:0.181570ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['8.234375    ', '4.91796875  ', '35.0        '], time:0.170741ms
          out_f16x8pack(k16+dbuf+offset): ['8.234375    ', '4.91796875  ', '35.0        '], time:0.170326ms
           out_f16x8pack(k16+dbuf+async): ['8.234375    ', '4.91796875  ', '35.0        '], time:0.174546ms
                 out_f16x8pack(k32+dbuf): ['8.234375    ', '4.91796875  ', '35.0        '], time:0.175085ms
           out_f16x8pack(k32+dbuf+async): ['8.234375    ', '4.91796875  ', '35.0        '], time:0.172029ms
           out_f16x8pack(k32+dbuf+t16x8): ['8.234375    ', '4.91796875  ', '35.0        '], time:0.181694ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['8.234375    ', '4.91796875  ', '35.0        '], time:0.173836ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['8.2578125   ', '4.9140625   ', '35.09375    '], time:0.370479ms
                     out_f16wmma(mma4x2): ['8.2578125   ', '4.9140625   ', '35.09375    '], time:0.193028ms
             out_f16wmma(mma4x2+warp2x4): ['8.2578125   ', '4.9140625   ', '35.09375    '], time:0.136247ms
       out_f16wmma(mma4x2+warp2x4+async): ['8.2578125   ', '4.9140625   ', '35.09375    '], time:0.124264ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['8.2578125   ', '4.9140625   ', '35.09375    '], time:0.108676ms
                              out_f16_th: ['8.234375    ', '4.921875    ', '35.09375    '], time:0.086040ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=4096, N=2048, K=1024
                                 out_f16: ['-0.37573242 ', '-0.44165039 ', '22.765625   '], time:4.741616ms
                             out_f16(sk): ['-0.37573242 ', '-0.44165039 ', '22.765625   '], time:3.530798ms
                  out_f16x4pack(t4x4bcf): ['-0.37573242 ', '-0.44165039 ', '22.765625   '], time:0.516210ms
               out_f16x4pack(t4x4offset): ['-0.37573242 ', '-0.44165039 ', '22.765625   '], time:0.506673ms
                       out_f16x4(t8x8sk): ['-0.37573242 ', '-0.44165039 ', '22.765625   '], time:0.446162ms
                      out_f16x4(t8x8bcf): ['-0.37573242 ', '-0.44165039 ', '22.765625   '], time:0.417891ms
                   out_f16x4pack(t8x8sk): ['-0.37573242 ', '-0.44165039 ', '22.765625   '], time:0.408673ms
                      out_f16x4pack(bcf): ['-0.37573242 ', '-0.44165039 ', '22.765625   '], time:0.383644ms
               out_f16x4pack(bcf+offset): ['-0.37573242 ', '-0.44165039 ', '22.765625   '], time:0.382633ms
                      out_f16x8pack(bcf): ['-0.37573242 ', '-0.44165039 ', '22.765625   '], time:0.370951ms
               out_f16x8pack(bcf+offset): ['-0.37573242 ', '-0.44165039 ', '22.765625   '], time:0.368061ms
                 out_f16x8pack(bcf+dbuf): ['-0.37573242 ', '-0.44165039 ', '22.765625   '], time:0.357280ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['-0.37573242 ', '-0.44165039 ', '22.765625   '], time:0.335937ms
          out_f16x8pack(k16+dbuf+offset): ['-0.37573242 ', '-0.44165039 ', '22.765625   '], time:0.335493ms
           out_f16x8pack(k16+dbuf+async): ['-0.37573242 ', '-0.44165039 ', '22.765625   '], time:0.343485ms
                 out_f16x8pack(k32+dbuf): ['-0.37573242 ', '-0.44165039 ', '22.765625   '], time:0.344944ms
           out_f16x8pack(k32+dbuf+async): ['-0.37573242 ', '-0.44165039 ', '22.765625   '], time:0.338612ms
           out_f16x8pack(k32+dbuf+t16x8): ['-0.37573242 ', '-0.44165039 ', '22.765625   '], time:0.358973ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['-0.37573242 ', '-0.44165039 ', '22.765625   '], time:0.342155ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['-0.35668945 ', '-0.39526367 ', '22.8125     '], time:0.735283ms
                     out_f16wmma(mma4x2): ['-0.35668945 ', '-0.39526367 ', '22.8125     '], time:0.375495ms
             out_f16wmma(mma4x2+warp2x4): ['-0.35668945 ', '-0.39526367 ', '22.8125     '], time:0.247149ms
       out_f16wmma(mma4x2+warp2x4+async): ['-0.35668945 ', '-0.39526367 ', '22.8125     '], time:0.227785ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['-0.35668945 ', '-0.39526367 ', '22.8125     '], time:0.195947ms
                              out_f16_th: ['-0.35961914 ', '-0.40844727 ', '22.765625   '], time:0.168276ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=4096, N=4096, K=256
                                 out_f16: ['1.29980469  ', '-4.984375   ', '-8.9375     '], time:2.402277ms
                             out_f16(sk): ['1.29980469  ', '-4.984375   ', '-8.9375     '], time:1.799273ms
                  out_f16x4pack(t4x4bcf): ['1.29980469  ', '-4.984375   ', '-8.9375     '], time:0.252466ms
               out_f16x4pack(t4x4offset): ['1.29980469  ', '-4.984375   ', '-8.9375     '], time:0.247731ms
                       out_f16x4(t8x8sk): ['1.29980469  ', '-4.984375   ', '-8.9375     '], time:0.230279ms
                      out_f16x4(t8x8bcf): ['1.29980469  ', '-4.984375   ', '-8.9375     '], time:0.208111ms
                   out_f16x4pack(t8x8sk): ['1.29980469  ', '-4.984375   ', '-8.9375     '], time:0.206265ms
                      out_f16x4pack(bcf): ['1.29980469  ', '-4.984375   ', '-8.9375     '], time:0.191441ms
               out_f16x4pack(bcf+offset): ['1.29980469  ', '-4.984375   ', '-8.9375     '], time:0.190635ms
                      out_f16x8pack(bcf): ['1.29980469  ', '-4.984375   ', '-8.9375     '], time:0.184526ms
               out_f16x8pack(bcf+offset): ['1.29980469  ', '-4.984375   ', '-8.9375     '], time:0.182753ms
                 out_f16x8pack(bcf+dbuf): ['1.29980469  ', '-4.984375   ', '-8.9375     '], time:0.180392ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['1.29980469  ', '-4.984375   ', '-8.9375     '], time:0.171375ms
          out_f16x8pack(k16+dbuf+offset): ['1.29980469  ', '-4.984375   ', '-8.9375     '], time:0.171103ms
           out_f16x8pack(k16+dbuf+async): ['1.29980469  ', '-4.984375   ', '-8.9375     '], time:0.174689ms
                 out_f16x8pack(k32+dbuf): ['1.29980469  ', '-4.984375   ', '-8.9375     '], time:0.174336ms
           out_f16x8pack(k32+dbuf+async): ['1.29980469  ', '-4.984375   ', '-8.9375     '], time:0.171928ms
           out_f16x8pack(k32+dbuf+t16x8): ['1.29980469  ', '-4.984375   ', '-8.9375     '], time:0.180473ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['1.29980469  ', '-4.984375   ', '-8.9375     '], time:0.173211ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['1.28613281  ', '-4.98828125 ', '-8.9453125  '], time:0.377154ms
                     out_f16wmma(mma4x2): ['1.28613281  ', '-4.98828125 ', '-8.9453125  '], time:0.198479ms
             out_f16wmma(mma4x2+warp2x4): ['1.28613281  ', '-4.98828125 ', '-8.9453125  '], time:0.155177ms
       out_f16wmma(mma4x2+warp2x4+async): ['1.28613281  ', '-4.98828125 ', '-8.9453125  '], time:0.133972ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['1.28613281  ', '-4.98828125 ', '-8.9453125  '], time:0.117846ms
                              out_f16_th: ['1.27539062  ', '-4.98828125 ', '-8.9375     '], time:0.088706ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=4096, N=4096, K=512
                                 out_f16: ['12.078125   ', '-24.46875   ', '-11.28125   '], time:4.751401ms
                             out_f16(sk): ['12.078125   ', '-24.46875   ', '-11.28125   '], time:3.544755ms
                  out_f16x4pack(t4x4bcf): ['12.078125   ', '-24.46875   ', '-11.28125   '], time:0.500393ms
               out_f16x4pack(t4x4offset): ['12.078125   ', '-24.46875   ', '-11.28125   '], time:0.490813ms
                       out_f16x4(t8x8sk): ['12.078125   ', '-24.46875   ', '-11.28125   '], time:0.440726ms
                      out_f16x4(t8x8bcf): ['12.078125   ', '-24.46875   ', '-11.28125   '], time:0.409417ms
                   out_f16x4pack(t8x8sk): ['12.078125   ', '-24.46875   ', '-11.28125   '], time:0.400891ms
                      out_f16x4pack(bcf): ['12.078125   ', '-24.46875   ', '-11.28125   '], time:0.375352ms
               out_f16x4pack(bcf+offset): ['12.078125   ', '-24.46875   ', '-11.28125   '], time:0.373926ms
                      out_f16x8pack(bcf): ['12.078125   ', '-24.46875   ', '-11.28125   '], time:0.361300ms
               out_f16x8pack(bcf+offset): ['12.078125   ', '-24.46875   ', '-11.28125   '], time:0.358438ms
                 out_f16x8pack(bcf+dbuf): ['12.078125   ', '-24.46875   ', '-11.28125   '], time:0.355268ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['12.078125   ', '-24.46875   ', '-11.28125   '], time:0.336571ms
          out_f16x8pack(k16+dbuf+offset): ['12.078125   ', '-24.46875   ', '-11.28125   '], time:0.336156ms
           out_f16x8pack(k16+dbuf+async): ['12.078125   ', '-24.46875   ', '-11.28125   '], time:0.343776ms
                 out_f16x8pack(k32+dbuf): ['12.078125   ', '-24.46875   ', '-11.28125   '], time:0.343266ms
           out_f16x8pack(k32+dbuf+async): ['12.078125   ', '-24.46875   ', '-11.28125   '], time:0.339074ms
           out_f16x8pack(k32+dbuf+t16x8): ['12.078125   ', '-24.46875   ', '-11.28125   '], time:0.355330ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['12.078125   ', '-24.46875   ', '-11.28125   '], time:0.341277ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['12.15625    ', '-24.40625   ', '-11.359375  '], time:0.742059ms
                     out_f16wmma(mma4x2): ['12.15625    ', '-24.40625   ', '-11.359375  '], time:0.380239ms
             out_f16wmma(mma4x2+warp2x4): ['12.15625    ', '-24.40625   ', '-11.359375  '], time:0.266757ms
       out_f16wmma(mma4x2+warp2x4+async): ['12.15625    ', '-24.40625   ', '-11.359375  '], time:0.236378ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['12.15625    ', '-24.40625   ', '-11.359375  '], time:0.201802ms
                              out_f16_th: ['12.1640625  ', '-24.4375    ', '-11.3515625 '], time:0.167875ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=4096, N=4096, K=1024
                                 out_f16: ['49.90625    ', '19.109375   ', '-29.59375   '], time:9.455042ms
                             out_f16(sk): ['49.90625    ', '19.109375   ', '-29.59375   '], time:7.028675ms
                  out_f16x4pack(t4x4bcf): ['49.90625    ', '19.109375   ', '-29.59375   '], time:0.995789ms
               out_f16x4pack(t4x4offset): ['49.90625    ', '19.109375   ', '-29.59375   '], time:0.977216ms
                       out_f16x4(t8x8sk): ['49.90625    ', '19.109375   ', '-29.59375   '], time:0.855165ms
                      out_f16x4(t8x8bcf): ['49.90625    ', '19.109375   ', '-29.59375   '], time:0.810685ms
                   out_f16x4pack(t8x8sk): ['49.90625    ', '19.109375   ', '-29.59375   '], time:0.788088ms
                      out_f16x4pack(bcf): ['49.90625    ', '19.109375   ', '-29.59375   '], time:0.744276ms
               out_f16x4pack(bcf+offset): ['49.90625    ', '19.109375   ', '-29.59375   '], time:0.741920ms
                      out_f16x8pack(bcf): ['49.90625    ', '19.109375   ', '-29.59375   '], time:0.715337ms
               out_f16x8pack(bcf+offset): ['49.90625    ', '19.109375   ', '-29.59375   '], time:0.709853ms
                 out_f16x8pack(bcf+dbuf): ['49.90625    ', '19.109375   ', '-29.59375   '], time:0.705132ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['49.90625    ', '19.109375   ', '-29.59375   '], time:0.665822ms
          out_f16x8pack(k16+dbuf+offset): ['49.90625    ', '19.109375   ', '-29.59375   '], time:0.664926ms
           out_f16x8pack(k16+dbuf+async): ['49.90625    ', '19.109375   ', '-29.59375   '], time:0.680118ms
                 out_f16x8pack(k32+dbuf): ['49.90625    ', '19.109375   ', '-29.59375   '], time:0.679498ms
           out_f16x8pack(k32+dbuf+async): ['49.90625    ', '19.109375   ', '-29.59375   '], time:0.670924ms
           out_f16x8pack(k32+dbuf+t16x8): ['49.90625    ', '19.109375   ', '-29.59375   '], time:0.704384ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['49.90625    ', '19.109375   ', '-29.59375   '], time:0.675421ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['49.71875    ', '18.953125   ', '-29.578125  '], time:1.469207ms
                     out_f16wmma(mma4x2): ['49.71875    ', '18.953125   ', '-29.578125  '], time:0.744109ms
             out_f16wmma(mma4x2+warp2x4): ['49.71875    ', '18.953125   ', '-29.578125  '], time:0.487657ms
       out_f16wmma(mma4x2+warp2x4+async): ['49.71875    ', '18.953125   ', '-29.578125  '], time:0.442595ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['49.71875    ', '18.953125   ', '-29.578125  '], time:0.373306ms
                              out_f16_th: ['49.875      ', '18.96875    ', '-29.640625  '], time:0.333500ms
--------------------------------------------------------------------------------------------------------------
```
