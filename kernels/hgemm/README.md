
## ⚡️⚡️Toy-HGEMM: Achieve the 98%~100% TFLOPS of cuBLAS 🎉🎉

![toy-hgemm-library](https://github.com/user-attachments/assets/962bda14-b494-4423-b8eb-775da9f5503d)

[📖Toy-HGEMM Library⚡️⚡️](./kernels/hgemm) is a library that write many HGEMM kernels from scratch using Tensor Cores with WMMA, MMA PTX and CuTe API, thus, can achieve `98%~100%` performance of **cuBLAS**. The codes here are source from 📖[CUDA-Learn-Notes](https://github.com/DefTruth/CUDA-Learn-Notes)  ![](https://img.shields.io/github/stars/DefTruth/CUDA-Learn-Notes.svg?style=social) and exported as a standalone library, please checkout [CUDA-Learn-Notes](https://github.com/DefTruth/CUDA-Learn-Notes) for latest updates. Welcome to 🌟👆🏻star this repo to support me, many thanks ~ 🎉🎉

<div id="hgemm-sgemm"></div>  

<div align='center'>
  <img src='https://github.com/user-attachments/assets/71927ac9-72b3-4ce9-b0e2-788b5885bc99' height="170px" width="270px">
  <img src='https://github.com/user-attachments/assets/05ef4f5e-d999-48ea-b58e-782cffb24e85' height="170px" width="270px">
  <img src='https://github.com/user-attachments/assets/9472e970-c083-4b31-9252-3eeecc761078' height="170px" width="270px">
</div> 


Currently, on NVIDIA L20, RTX 4090 and RTX 3080 Laptop, compared with cuBLAS's default Tensor Cores math algorithm `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, the `HGEMM (WMMA/MMA/CuTe)` implemented in this repo (`blue`🔵) can achieve `98%~100%` of its (`orange`🟠) performance. Please check [toy-hgemm library⚡️⚡️](./kernels/hgemm) for more details.

|📚Feature |📚Feature |📚Feature |📚Feature|
|:---:|:---:|:---:|:---:|
|✔️CUDA/**Tensor Cores**|✔️Loop over K|✔️Tile Block(BMxBK)|✔️Tile Threads(T 8x8)|
|✔️WMMA(m16n16k16)|✔️MMA(m16n8k16)|✔️Pack LDST(128 bits)|✔️SMEM Padding|
|✔️Copy Async|✔️Tile MMAs|✔️Tile Warps|✔️**Multi Stages(2~4)**|  
|✔️Register Double Buffers|✔️**Block Swizzle**|✔️**Warp Swizzle**|✔️**SMEM Swizzle**(CuTe/MMA)|
|✔️Collective Store(Shfl)|✔️Layout NN|✔️Layout TN|✔️SGEMM FP32/TF32|

## ©️Citations🎉🎉

```BibTeX
@misc{hgemm-tensorcores-mma@2024,
  title={hgemm-tensorcores-mma: Write HGEMM from scratch using Tensor Cores with WMMA, MMA PTX and CuTe API.},
  url={https://github.com/DefTruth/hgemm-tensorcores-mma},
  note={Open-source software available at https://github.com/DefTruth/hgemm-tensorcores-mma},
  author={DefTruth etc},
  year={2024}
}
```

## 📖 HGEMM CUDA Kernels in Toy-HGEMM Library 🎉🎉 

<div id="kernels"></div>  

```C++  
void hgemm_naive_f16(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_sliced_k_f16(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k_f16x4(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k_f16x4_pack(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k_f16x4_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k_f16x4_pack_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k_f16x8_pack_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_cublas_tensor_op_nn(torch::Tensor a, torch::Tensor b, torch::Tensor c); 
void hgemm_cublas_tensor_op_tn(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_wmma_m16n16k16_naive(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_wmma_m16n16k16_mma4x2(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_wmma_m16n16k16_mma4x2_warp2x4(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);                                                        
void hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_m16n8k16_naive(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_mma_m16n8k16_mma2x4_warp4x4(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_stages_block_swizzle_tn_cute(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
```

## 📖 Contents

- [📖 Prerequisites](#prerequisites)
- [📖 Installation](#install)
- [📖 Python Testing](#test)
- [📖 C++ Testing](#test-cpp)
- [📖 NVIDIA L20 bench](#perf-l20)
- [📖 NVIDIA RTX 4090 bench](#perf-4090)
- [📖 NVIDIA RTX 3080 Laptop bench](#perf-3080)
- [📖 Docs](#opt-docs)
- [📖 References](#ref)

## 📖 Prerequisites
<div id="prerequisites"></div>  

- PyTorch >= 2.0, CUDA >= 12.0
- Recommended: PyTorch 2.5.1, CUDA 12.5

## 📖 Installation  

<div id="install"></div>  

The HGEMM implemented in this repo can be install as a python library, namely, `toy-hgemm` library (optional). 
```bash
cd kernels/hgemm
git submodule update --init --recursive --force # Fetch `CUTLASS` submodule， needed
python3 setup.py bdist_wheel && cd dist && python3 -m pip install *.whl # pip uninstall toy-hgemm -y 
```

## 📖 Python Testing

<div id="test"></div>  

**CUTLASS**: Fetch `CUTLASS` submodule. Currently, I use `v3.5.1` for HGEMM CuTe kernel.
```bash
git submodule update --init --recursive --force
```

You can test many custom HGEMM kernel via Python script and figure out the difference in their performance.

```bash
# You can test Ada or Ampere only, also, Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada # for Ada only
export TORCH_CUDA_ARCH_LIST=Ampere # for Ampere only
python3 hgemm.py --wmma # test defalut wmma kernels for all MNK
python3 hgemm.py --mma  # test defalut mma kernels for all MNK
python3 hgemm.py --M 16384 --N 16384 --K 8192 --i 10 --wmma # test default wmma kernels for specific MNK
python3 hgemm.py --M 16384 --N 16384 --K 8192 --i 10 --mma # test default mma kernels for specific MNK
python3 hgemm.py --wmma-all # test all wmma kernels for all MNK
python3 hgemm.py --mma-all # test all mma kernels for all MNK
python3 hgemm.py --cuda-all --wmma-all --mma-all # test all kernels for all MNK
python3 hgemm.py --cute-tn --no-default # test cute hgemm kernels with smem swizzle for all MNK
```
If you want to draw a TFLOPS curve, you need to install `matplotlib` first and set the --plot-flops (or --plot) option.
```bash
python3 -m pip install matplotlib
# Specify topk to plot only the top k kernels with the best performance.
python3 hgemm.py --mma-all --plot --topk 8
# test default mma kernels & cute hgemm kernels with smem swizzle for all MNK
python3 hgemm.py --cute-tn --mma --plot 
```

## 📖 C++ Testing

<div id="test-cpp"></div>  

The HGEMM benchmark also supports C++ testing. Currently, it supports comparisons between the following implementations:

- MMA HGEMM NN implemented in this repository
- CuTe HGEMM TN implemented in this repository
- cuBLAS HGEMM TN use default Tensor Cores math algorithm

Performance data obtained from C++ binary tests tend to be slightly better than those from Python tests. This difference may be attributed to additional overhead introduced by the PyTorch Python bindings.
```bash
make
./hgemm_mma_stage.bin
# NVIDIA L20
ALGO = MMA16816 HGEMM NN MMA=2x4 WARP=4x4x2 STAGES=2 BLOCK SWIZZLE=2048
M N K =  12544  12544  12544, Time =   0.03445555   0.03446098   0.03447399 s, AVG Performance =   114.5541 Tflops
M N K =  15360  15360  15360, Time =   0.06307226   0.06307789   0.06308864 s, AVG Performance =   114.9017 Tflops
M N K =  15616  15616  15616, Time =   0.06612480   0.06612798   0.06613094 s, AVG Performance =   115.1739 Tflops
M N K =  15872  15872  15872, Time =   0.06969549   0.06970215   0.06971290 s, AVG Performance =   114.7305 Tflops
M N K =  16128  16128  16128, Time =   0.07295078   0.07295406   0.07295693 s, AVG Performance =   115.0064 Tflops
M N K =  16384  16384  16384, Time =   0.07663001   0.07663534   0.07664947 s, AVG Performance =   114.7785 Tflops

./hgemm_cute.bin
# NVIDIA L20
ALGO = CuTe HGEMM, TN, STAGES=2, SMEM SWIZZLE=<3, 3, 3>, BLOCK SWIZZLE=2048
M N K =  12544  12544  12544, Time =   0.03413504   0.03414354   0.03415450 s, AVG Performance =   115.6191 Tflops
M N K =  15360  15360  15360, Time =   0.06227354   0.06228111   0.06228992 s, AVG Performance =   116.3717 Tflops
M N K =  15616  15616  15616, Time =   0.06492467   0.06493727   0.06496666 s, AVG Performance =   117.2858 Tflops
M N K =  15872  15872  15872, Time =   0.06843085   0.06843873   0.06844723 s, AVG Performance =   116.8485 Tflops
M N K =  16128  16128  16128, Time =   0.07200256   0.07200881   0.07201792 s, AVG Performance =   116.5161 Tflops
M N K =  16384  16384  16384, Time =   0.07564493   0.07565752   0.07567462 s, AVG Performance =   116.2620 Tflops

./hgemm_cublas.bin
# NVIDIA L20
ALGO = cuBLAS CUBLAS_GEMM_DEFAULT_TENSOR_OP TN
M N K =  12544  12544  12544, Time =   0.03472691   0.03472968   0.03473408 s, AVG Performance =   113.6678 Tflops
M N K =  15360  15360  15360, Time =   0.06332416   0.06333143   0.06334157 s, AVG Performance =   114.4417 Tflops
M N K =  15616  15616  15616, Time =   0.06649446   0.06650184   0.06651699 s, AVG Performance =   114.5264 Tflops
M N K =  15872  15872  15872, Time =   0.06977024   0.06977659   0.06978355 s, AVG Performance =   114.6081 Tflops
M N K =  16128  16128  16128, Time =   0.07319142   0.07320709   0.07326925 s, AVG Performance =   114.6089 Tflops
M N K =  16384  16384  16384, Time =   0.07668429   0.07669371   0.07670784 s, AVG Performance =   114.6912 Tflops
```

## 📖 Benchmark  

<div id="perf-l20"></div>  

### 📖 NVIDIA L20  
<!--
目前最优的实现，在L20上（理论Tensor Cores FP16算力为 119.5 TFLOPS），整体上能达到cuBLAS大概`99~100+%`左右的性能。使用WMMA API能达到cuBLAS大概`95%~98%`左右的性能(105-113 TFLOPS vs 105-115 TFLOPS)，使用MMA API能达到115 TFLOPS，部分 case 会超越 cuBLAS。CuTe 版本的 HGEMM 实现了 Block Swizzle（L2 Cache friendly）和 SMEM Swizzle（bank conflicts free），性能最优，大规模矩阵乘能达到 116-117 TFLOPS，是 cuBLAS 大概`98%~100%+`左右的性能，很多case会超越cuBLAS。目前通过 SMEM Padding 和 SMEM Swizzle 的方式缓解 bank conflicts。对于 NN layout，使用 SMEM Padding 缓解 bank conflicts；对于 TN layout，通过 CUTLASS/CuTe 的 SMEM Swizzle 消除 bank conflicts。
-->
The current best implementation, on the L20 (with a theoretical Tensor Cores FP16 performance of 119.5 TFLOPS), achieves performance that is approximately 99~100+% of cuBLAS.

- Using the WMMA API, it can achieve around 95%~98% of cuBLAS performance (105-113 TFLOPS vs 105-115 TFLOPS).
- Using the MMA API, it can reach 115 TFLOPS, surpassing cuBLAS in some cases.
- The CuTe version of HGEMM implements Block Swizzle (L2 Cache friendly) and SMEM Swizzle (bank conflicts free), achieving the best performance. For large-scale matrix multiplication, it can reach 116-117 TFLOPS, which is approximately 98%~100%+ of cuBLAS performance, and it outperforms cuBLAS in many cases.

Currently, SMEM Padding and SMEM Swizzle are used to mitigate bank conflicts:

- For the NN layout, SMEM Padding is used to alleviate bank conflicts.
- For the TN layout, CUTLASS/CuTe's SMEM Swizzle is used to eliminate bank conflicts.

<div id="NV-L20"></div>


![NVIDIA_L20_NN+TN+v2](https://github.com/user-attachments/assets/71927ac9-72b3-4ce9-b0e2-788b5885bc99)

  
The command for testing all MNK setups (Tip: Performance data for each MNK tested individually is more accurate.)
```bash
python3 hgemm.py --cute-tn --mma --plot
```

### 📖 NVIDIA GeForce RTX 4090

<div id="perf-4090"></div>  

<!--
在NVIDIA RTX 4090上(FP16 Tensor Cores算力为330 TFLOPS)，WMMA(m16n16k16)性能表现比MMA(m16n8k16)要更好，大分部MNK下，本仓库的实现能达到cuBLAS 95%~99%的性能，某些case能超过cuBLAS。就本仓库的实现而言，在RTX 4090上，大规模矩阵乘(MNK>=8192)，WMMA表现更优，小规模矩阵乘，MMA表现更优。
-->

On the NVIDIA RTX 4090 (with an FP16 Tensor Cores performance of 330 TFLOPS), the WMMA (m16n16k16) implementation shows better performance compared to MMA (m16n8k16). For most MNK configurations, this repository's implementation achieves 95%~99% of cuBLAS performance, and in certain cases, it can surpass cuBLAS. Specifically:

- For large-scale matrix multiplications (MNK >= 8192), the WMMA implementation performs better.
- For small-scale matrix multiplications, the MMA implementation is more efficient.


![NVIDIA_GeForce_RTX_4090_NN+TN+v4](https://github.com/user-attachments/assets/05ef4f5e-d999-48ea-b58e-782cffb24e85)

```bash
python3 hgemm.py --cute-tn --mma --wmma-all --plot
```

### 📖 NVIDIA GeForce RTX 3080 Laptop   

<div id="perf-3080"></div>  

<!--
在NVIDIA GeForce RTX 3080 Laptop上测试，使用mma4x4_warp4x4（16 WMMA m16n16k16 ops, warp tile 64x64）以及Thread block swizzle，大部分case能持平甚至超过cuBLAS，使用Windows WSL2 + RTX 3080 Laptop进行测试。
-->
Testing was conducted on a NVIDIA GeForce RTX 3080 Laptop using the mma4x4_warp4x4 configuration (which includes 16 WMMA m16n16k16 operations with a warp tile size of 64x64) along with Thread block swizzle. In most cases, this setup matches or even exceeds cuBLAS performance. The tests were performed using Windows WSL2 + RTX 3080 Laptop.

![image](https://github.com/user-attachments/assets/9472e970-c083-4b31-9252-3eeecc761078)

```bash
python3 hgemm.py --wmma-all --plot
```

<details>
<summary> 🔑️ Performance Optimization Notes(TODO)</summary>    

## 📖 Performance Optimization Notes

<div id="opt-docs"></div>  

### PyTorch HGEMM Profile

在Ada架构下，PyTorch 2.4对FP16使用matmul时，会调用:
```C++
ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn_kernel
```
内部实际使用HMMA(Tensor Cores)进行计算，在3080上profile发现使用:
```C++
sm80_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize96x64x32_stage3_warpsize2x2x1_tensor16x8x16_kernel
```
因此，只有实现使用Tensor Cores的HGEMM，才有可能接近PyTorch/cuBLAS的性能。
```bash
ncu -o hgemm.prof -f python3 bench/prof.py
nsys profile --stats=true -t cuda,osrt,nvtx -o hgemm.prof --force-overwrite true python3 prof.py
```
- SASS (L20)

```C
// ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn_kernel
310	00007f41 37d5b850	      LDSM.16.M88.4 R192, [R169+UR8+0x2000] 
311	00007f41 37d5b860	      LDSM.16.M88.4 R196, [R169+UR8+0x2800]
336	00007f41 37d5b9f0	      HMMA.1688.F32 R112, R182, R196, R112
...
```

### SMEM Padding  

#### Bank Conflicts的产生
  
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
目前通过 SMEM Padding 和 SMEM swizzle的方式缓解bank conflicts。对于 NN layout，使用 SMEM Padding 缓解 bank conflicts；对于 TN layout，通过cutlass cute的 SMEM Swizzle 消除 bank conflicts。

### 双缓冲 Double Buffers

本仓库实现的HGEMM Double Buffers策略如下：1）主循环从bk = 1 开始，第一次数据加载在主循环之前，最后一次计算在主循环之后，这是pipeline 的特点决定的；2）由于计算和下一次访存使用的Shared Memory不同，因此主循环中每次循环只需要一次__syncthreads()即可，对比非double buffers版本，总共节省了 ((K + BK - 1) / BK) - 1 次block内的同步操作。比如，bk=1时，HFMA计算使用的是s_a[0]和s_b[0]，因此，和s_a[1]和s_b[1]的加载是没有依赖关系的。HFMA计算，从global内存到s_a[1]和s_b[1]和HFMA计算可以并行。s_a[1]和s_b[1]用于加载下一块BK需要的数据到共享内存；3）由于GPU不能向CPU那样支持乱序执行，主循环中需要先将下一次循环计算需要的Gloabal Memory中的数据load 到寄存器，然后进行本次计算，之后再将load到寄存器中的数据写到Shared Memory，这样在LDG指令向Global Memory做load时，不会影响后续HFMA及其它运算指令的 launch 执行，也就达到了Double Buffers的目的，具体代码见[hgemm.cu](./hgemm.cu)。


### Tile Block

TODO

### Tile Thread

TODO

### Pack LDST 128 bits

TODO

### Async Copy

TODO

### Multi Stages

TODO

### Tensor Cores(WMMA/MMA)

TODO

### Tile MMA/Warp

TODO 

### Thread Block Swizze 

TODO

### Warp Swizzle

TODO

### Reg Double Buffers

TODO

### Collective Store(Reg Reuse&Warp Shuffle)

TODO

### SMEM Swizzle/Permuted

TODO

</details>

## 📖 References 

<div id="ref"></div>  

- [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal)
- [tiny-flash-attention](https://github.com/66RING/tiny-flash-attention)
- [cute-gemm](https://github.com/reed-lau/cute-gemm)
- [cutlass_flash_atten_fp8](https://github.com/weishengying/cutlass_flash_atten_fp8)
- [cuda_learning](https://github.com/ifromeast/cuda_learning)
- [cuda_hgemm](https://github.com/Bruce-Lee-LY/cuda_hgemm)
- [cuda-tensorcore-hgemm](https://github.com/nicolaswilde/cuda-tensorcore-hgemm)
- [How_to_optimize_in_GPU](https://github.com/Liu-xiandong/How_to_optimize_in_GPU/tree/master/sgemv)
- [cute_gemm](https://github.com/weishengying/cute_gemm)
- [cutlass](https://github.com/NVIDIA/cutlass)
