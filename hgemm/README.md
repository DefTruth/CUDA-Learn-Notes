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
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_async_offset(WMMA API, Tensor Cores, Tile MMA, Tile Warp, Copy Async, Pad)
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(WMMA API, Tensor Cores, Tile MMA, Tile Warp, Copy Async, Double Buffers)
- [X] hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_offset(WMMA API, Tensor Cores, Tile MMA, Tile Warp, Copy Async, Double Buffers, Pad)  
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
                                             M=4096, N=4096, K=256
                                 out_f16: ['21.34375    ', '12.5859375  ', '15.125      '], time:2.401705ms
                             out_f16(sk): ['21.34375    ', '12.5859375  ', '15.125      '], time:1.801600ms
                  out_f16x4pack(t4x4bcf): ['21.34375    ', '12.5859375  ', '15.125      '], time:0.252500ms
               out_f16x4pack(t4x4offset): ['21.34375    ', '12.5859375  ', '15.125      '], time:0.247712ms
                       out_f16x4(t8x8sk): ['21.34375    ', '12.5859375  ', '15.125      '], time:0.230112ms
                      out_f16x4(t8x8bcf): ['21.34375    ', '12.5859375  ', '15.125      '], time:0.207977ms
                   out_f16x4pack(t8x8sk): ['21.34375    ', '12.5859375  ', '15.125      '], time:0.206332ms
                      out_f16x4pack(bcf): ['21.34375    ', '12.5859375  ', '15.125      '], time:0.191288ms
               out_f16x4pack(bcf+offset): ['21.34375    ', '12.5859375  ', '15.125      '], time:0.190663ms
                      out_f16x8pack(bcf): ['21.34375    ', '12.5859375  ', '15.125      '], time:0.184417ms
               out_f16x8pack(bcf+offset): ['21.34375    ', '12.5859375  ', '15.125      '], time:0.182781ms
                 out_f16x8pack(bcf+dbuf): ['21.34375    ', '12.5859375  ', '15.125      '], time:0.180416ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['21.34375    ', '12.5859375  ', '15.125      '], time:0.171366ms
          out_f16x8pack(k16+dbuf+offset): ['21.34375    ', '12.5859375  ', '15.125      '], time:0.171094ms
           out_f16x8pack(k16+dbuf+async): ['21.34375    ', '12.5859375  ', '15.125      '], time:0.174646ms
                 out_f16x8pack(k32+dbuf): ['21.34375    ', '12.5859375  ', '15.125      '], time:0.174341ms
           out_f16x8pack(k32+dbuf+async): ['21.34375    ', '12.5859375  ', '15.125      '], time:0.171895ms
           out_f16x8pack(k32+dbuf+t16x8): ['21.34375    ', '12.5859375  ', '15.125      '], time:0.180469ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['21.34375    ', '12.5859375  ', '15.125      '], time:0.173192ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['21.328125   ', '12.6796875  ', '15.15625    '], time:0.377154ms
                     out_f16wmma(mma4x2): ['21.328125   ', '12.6796875  ', '15.15625    '], time:0.198345ms
             out_f16wmma(mma4x2+warp2x4): ['21.328125   ', '12.6796875  ', '15.15625    '], time:0.155196ms
       out_f16wmma(mma4x2+warp2x4+async): ['21.328125   ', '12.6796875  ', '15.15625    '], time:0.133786ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['21.328125   ', '12.6796875  ', '15.15625    '], time:0.117764ms
        out_f16wmma(mma4x2+warp2x4+dbuf): ['21.328125   ', '12.6796875  ', '15.15625    '], time:0.133781ms
 out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['21.328125   ', '12.6796875  ', '15.15625    '], time:0.114479ms
                              out_f16_th: ['21.34375    ', '12.6875     ', '15.1484375  '], time:0.088682ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=4096, N=4096, K=512
                                 out_f16: ['-6.7734375  ', '8.1796875   ', '-8.96875    '], time:4.750853ms
                             out_f16(sk): ['-6.7734375  ', '8.1796875   ', '-8.96875    '], time:3.545322ms
                  out_f16x4pack(t4x4bcf): ['-6.7734375  ', '8.1796875   ', '-8.96875    '], time:0.500407ms
               out_f16x4pack(t4x4offset): ['-6.7734375  ', '8.1796875   ', '-8.96875    '], time:0.491042ms
                       out_f16x4(t8x8sk): ['-6.7734375  ', '8.1796875   ', '-8.96875    '], time:0.440555ms
                      out_f16x4(t8x8bcf): ['-6.7734375  ', '8.1796875   ', '-8.96875    '], time:0.409460ms
                   out_f16x4pack(t8x8sk): ['-6.7734375  ', '8.1796875   ', '-8.96875    '], time:0.400872ms
                      out_f16x4pack(bcf): ['-6.7734375  ', '8.1796875   ', '-8.96875    '], time:0.375319ms
               out_f16x4pack(bcf+offset): ['-6.7734375  ', '8.1796875   ', '-8.96875    '], time:0.373988ms
                      out_f16x8pack(bcf): ['-6.7734375  ', '8.1796875   ', '-8.96875    '], time:0.361319ms
               out_f16x8pack(bcf+offset): ['-6.7734375  ', '8.1796875   ', '-8.96875    '], time:0.358438ms
                 out_f16x8pack(bcf+dbuf): ['-6.7734375  ', '8.1796875   ', '-8.96875    '], time:0.355258ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['-6.7734375  ', '8.1796875   ', '-8.96875    '], time:0.336514ms
          out_f16x8pack(k16+dbuf+offset): ['-6.7734375  ', '8.1796875   ', '-8.96875    '], time:0.336127ms
           out_f16x8pack(k16+dbuf+async): ['-6.7734375  ', '8.1796875   ', '-8.96875    '], time:0.343747ms
                 out_f16x8pack(k32+dbuf): ['-6.7734375  ', '8.1796875   ', '-8.96875    '], time:0.343237ms
           out_f16x8pack(k32+dbuf+async): ['-6.7734375  ', '8.1796875   ', '-8.96875    '], time:0.339074ms
           out_f16x8pack(k32+dbuf+t16x8): ['-6.7734375  ', '8.1796875   ', '-8.96875    '], time:0.355449ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['-6.7734375  ', '8.1796875   ', '-8.96875    '], time:0.341363ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['-6.87890625 ', '8.1328125   ', '-8.8515625  '], time:0.742221ms
                     out_f16wmma(mma4x2): ['-6.87890625 ', '8.1328125   ', '-8.8515625  '], time:0.380182ms
             out_f16wmma(mma4x2+warp2x4): ['-6.87890625 ', '8.1328125   ', '-8.8515625  '], time:0.266891ms
       out_f16wmma(mma4x2+warp2x4+async): ['-6.87890625 ', '8.1328125   ', '-8.8515625  '], time:0.236592ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['-6.87890625 ', '8.1328125   ', '-8.8515625  '], time:0.201979ms
        out_f16wmma(mma4x2+warp2x4+dbuf): ['-6.87890625 ', '8.1328125   ', '-8.8515625  '], time:0.235939ms
 out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['-6.87890625 ', '8.1328125   ', '-8.8515625  '], time:0.196977ms
                              out_f16_th: ['-6.86328125 ', '8.1484375   ', '-8.84375    '], time:0.167861ms
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
                                             M=4096, N=4096, K=1024
                                 out_f16: ['50.0        ', '-48.21875   ', '5.85546875  '], time:9.450979ms
                             out_f16(sk): ['50.0        ', '-48.21875   ', '5.85546875  '], time:7.027154ms
                  out_f16x4pack(t4x4bcf): ['50.0        ', '-48.21875   ', '5.85546875  '], time:0.995831ms
               out_f16x4pack(t4x4offset): ['50.0        ', '-48.21875   ', '5.85546875  '], time:0.977545ms
                       out_f16x4(t8x8sk): ['50.0        ', '-48.21875   ', '5.85546875  '], time:0.854812ms
                      out_f16x4(t8x8bcf): ['50.0        ', '-48.21875   ', '5.85546875  '], time:0.810866ms
                   out_f16x4pack(t8x8sk): ['50.0        ', '-48.21875   ', '5.85546875  '], time:0.788083ms
                      out_f16x4pack(bcf): ['50.0        ', '-48.21875   ', '5.85546875  '], time:0.742159ms
               out_f16x4pack(bcf+offset): ['50.0        ', '-48.21875   ', '5.85546875  '], time:0.739799ms
                      out_f16x8pack(bcf): ['50.0        ', '-48.21875   ', '5.85546875  '], time:0.712948ms
               out_f16x8pack(bcf+offset): ['50.0        ', '-48.21875   ', '5.85546875  '], time:0.707469ms
                 out_f16x8pack(bcf+dbuf): ['50.0        ', '-48.21875   ', '5.85546875  '], time:0.703216ms
----------------------------------------------------Async-----------------------------------------------------
                 out_f16x8pack(k16+dbuf): ['50.0        ', '-48.21875   ', '5.85546875  '], time:0.665779ms
          out_f16x8pack(k16+dbuf+offset): ['50.0        ', '-48.21875   ', '5.85546875  '], time:0.665040ms
           out_f16x8pack(k16+dbuf+async): ['50.0        ', '-48.21875   ', '5.85546875  '], time:0.680223ms
                 out_f16x8pack(k32+dbuf): ['50.0        ', '-48.21875   ', '5.85546875  '], time:0.679569ms
           out_f16x8pack(k32+dbuf+async): ['50.0        ', '-48.21875   ', '5.85546875  '], time:0.670943ms
           out_f16x8pack(k32+dbuf+t16x8): ['50.0        ', '-48.21875   ', '5.85546875  '], time:0.704427ms
     out_f16x8pack(k32+dbuf+t16x8+async): ['50.0        ', '-48.21875   ', '5.85546875  '], time:0.675478ms
-----------------------------------------------------WMMA-----------------------------------------------------
                     out_f16wmma(+naive): ['49.9375     ', '-48.34375   ', '5.9765625   '], time:1.469316ms
                     out_f16wmma(mma4x2): ['49.9375     ', '-48.34375   ', '5.9765625   '], time:0.744023ms
             out_f16wmma(mma4x2+warp2x4): ['49.9375     ', '-48.34375   ', '5.9765625   '], time:0.487695ms
       out_f16wmma(mma4x2+warp2x4+async): ['49.9375     ', '-48.34375   ', '5.9765625   '], time:0.442576ms
out_f16wmma(mma4x2+warp2x4+async+offset): ['49.9375     ', '-48.34375   ', '5.9765625   '], time:0.373168ms
        out_f16wmma(mma4x2+warp2x4+dbuf): ['49.9375     ', '-48.34375   ', '5.9765625   '], time:0.440197ms
 out_f16wmma(mma4x2+warp2x4+dbuf+offset): ['49.9375     ', '-48.34375   ', '5.9765625   '], time:0.361319ms
                              out_f16_th: ['49.9375     ', '-48.3125    ', '5.98828125  '], time:0.333457ms
--------------------------------------------------------------------------------------------------------------
```
