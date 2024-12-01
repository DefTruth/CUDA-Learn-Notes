## ⚡️⚡️Toy-HGEMM Library: Achieve the performance of cuBLAS

![toy-hgemm-library](https://github.com/user-attachments/assets/4a2114e0-0341-4996-ac51-9b756ab13738)

[📖Toy-HGEMM Library⚡️⚡️](./kernels/hgemm) is library that write HGEMM kernels from scratch using Tensor Cores with WMMA, MMA PTX and CuTe API, thus, can achieve `98%~100%` performance of **cuBLAS**. The codes here are sources from 📖[CUDA-Learn-Notes](https://github.com/DefTruth/CUDA-Learn-Notes)  ![](https://img.shields.io/github/stars/DefTruth/CUDA-Learn-Notes.svg?style=social) as a story, please checkout 📖[CUDA-Learn-Notes](https://github.com/DefTruth/CUDA-Learn-Notes) for latest updates.

<div id="hgemm-sgemm"></div>  

<div align='center'>
  <img src='https://github.com/user-attachments/assets/71927ac9-72b3-4ce9-b0e2-788b5885bc99' height="170px" width="270px">
  <img src='https://github.com/user-attachments/assets/05ef4f5e-d999-48ea-b58e-782cffb24e85' height="170px" width="270px">
  <img src='https://github.com/user-attachments/assets/9472e970-c083-4b31-9252-3eeecc761078' height="170px" width="270px">
</div> 


Currently, on NVIDIA L20, RTX 4090 and RTX 3080 Laptop, compared with cuBLAS's default Tensor Cores math algorithm `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, the `HGEMM (WMMA/MMA/CuTe)` implemented in this repo (`blue`🔵) can achieve `98%~100%` of its (`orange`🟠) performance. Please check [toy-hgemm library⚡️⚡️](./kernels/hgemm) for more details.

|CUDA Cores|Sliced K (Loop over K)|Tile Block (BMxBN)|Tile Thread (t 8x8)|
|:---:|:---:|:---:|:---:|
|✔️|✔️|✔️|✔️|
|WMMA (m16n16k16)|MMA (m16n8k16)|Pack LDS T(pack 128 bits)|SMEM Padding|
|✔️|✔️|✔️|✔️|
|Copy Async (cp.async.cg/ca)|Tile MMA (More Threads)|Tile Warp (More Values)|Multi Stages(2/3/4/5)|  
|✔️|✔️|✔️|✔️|
|Register Double Buffers|Block Swizzle (Zigzag N)|Warp Swizzle (Zigzag N)| SMEM Swizzle (CUTLASS/CuTe)|
|✔️|✔️|✔️|✔️|
|Collective Store (Warp Shuffle & Reg Reuse)|Row Major (NN)|Col Major (TN)|SGEMM FP32/TF32|
|✔️|✔️|✔️|✔️|

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
```

## 📖 目录

- [📖 安装](#install)
- [📖 测试](#test)
- [📖 NVIDIA L20 性能数据](#perf-l20)
- [📖 NVIDIA RTX 4090 性能数据](#perf-4090)
- [📖 NVIDIA RTX 3080 Laptop 性能数据](#perf-3080)
- [📖 性能优化笔记](#opt-docs)
- [📖 参考文献](#ref)

## 📖 安装  

<div id="install"></div>  

本仓库实现的HGEMM可以作为一个python库使用（可选）
```bash
cd kernels/hgemm
git submodule update --init --recursive --force # 更新cutlass, 必须
python3 setup.py bdist_wheel && cd dist && python3 -m pip install *.whl # pip uninstall toy-hgemm -y 卸载
```

## 📖 测试

<div id="test"></div>  

**CUTLASS**: 更新CUTLASS依赖库
```bash
git submodule update --init --recursive --force
```

**Python**: 支持Python脚本直接测试

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长: Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada 
python3 hgemm.py --wmma # test defalut wmma kernels for all MNK
python3 hgemm.py --mma  # test defalut mma kernels for all MNK
python3 hgemm.py --M 16384 --N 16384 --K 8192 --i 10 --wmma # test default wmma kernels for specific MNK
python3 hgemm.py --M 16384 --N 16384 --K 8192 --i 10 --mma # test default mma kernels for specific MNK
python3 hgemm.py --wmma-all # test all wmma kernels for all MNK
python3 hgemm.py --mma-all # test all mma kernels for all MNK
python3 hgemm.py --cuda-all --wmma-all --mma-all # test all kernels for all MNK
python3 hgemm.py --cute-tn --no-default # test cute hgemm kernels with smem swizzle for all MNK
```
如果需要绘制TFLOPS曲线图，需要先安装matplotlib，并指定--plot-flops（或--plot）选项:
```bash
python3 -m pip install matplotlib
# topk指定只绘制性能最好的topk个kernel
python3 hgemm.py --mma-all --plot --topk 8
# test default mma kernels & cute hgemm kernels with smem swizzle for all MNK
python3 hgemm.py --cute-tn --mma --plot 
```

**C++**: HGEMM benchmark也支持C++测试，目前支持本仓库实现的 MMA HGEMM NN, CuTe HGEMM TN 和 cuBLAS HGEMM TN 进行对比，C++ bin方式测试的性能数据会略优于Python测试方式，可能是PyTorch Python binding引入了一定的额外开销。
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

## 📖 目前性能  

<div id="perf-l20"></div>  

### NVIDIA L20  

目前最优的实现，在L20上（理论Tensor Cores FP16算力为 119.5 TFLOPS），整体上能达到cuBLAS大概`99~100+%`左右的性能。使用WMMA API能达到cuBLAS大概`95%~98%`左右的性能(105-113 TFLOPS vs 105-115 TFLOPS)，使用MMA API能达到115 TFLOPS，部分 case 会超越 cuBLAS。CuTe 版本的 HGEMM 实现了 Block Swizzle（L2 Cache friendly）和 SMEM Swizzle（bank conflicts free），性能最优，大规模矩阵乘能达到 116-117 TFLOPS，是 cuBLAS 大概`98%~100%+`左右的性能，很多case会超越cuBLAS。目前通过 SMEM Padding 和 SMEM Swizzle 的方式缓解 bank conflicts。对于 NN layout，使用 SMEM Padding 缓解 bank conflicts；对于 TN layout，通过 CUTLASS/CuTe 的 SMEM Swizzle 消除 bank conflicts。

<div id="NV-L20"></div>


![NVIDIA_L20_NN+TN+v2](https://github.com/user-attachments/assets/71927ac9-72b3-4ce9-b0e2-788b5885bc99)

  
全量MNK测试命令（提示: 每个MNK单独测试的性能数据更准确）
```bash
python3 hgemm.py --cute-tn --mma --plot
```

### NVIDIA GeForce RTX 4090

<div id="perf-4090"></div>  

在NVIDIA RTX 4090上(FP16 Tensor Cores算力为330 TFLOPS)，WMMA(m16n16k16)性能表现比MMA(m16n8k16)要更好，大分部MNK下，本仓库的实现能达到cuBLAS 95%~99%的性能，某些case能超过cuBLAS。就本仓库的实现而言，在RTX 4090上，大规模矩阵乘(MNK>=8192)，WMMA表现更优，小规模矩阵乘，MMA表现更优。


![NVIDIA_GeForce_RTX_4090_NN+TN+v4](https://github.com/user-attachments/assets/05ef4f5e-d999-48ea-b58e-782cffb24e85)

```bash
python3 hgemm.py --cute-tn --mma --wmma-all --plot
```

### NVIDIA GeForce RTX 3080 Laptop   

<div id="perf-3080"></div>  

在NVIDIA GeForce RTX 3080 Laptop上测试，使用mma4x4_warp4x4（16 WMMA m16n16k16 ops, warp tile 64x64）以及Thread block swizzle，大部分case能持平甚至超过cuBLAS，使用Windows WSL2 + RTX 3080 Laptop进行测试。


![image](https://github.com/user-attachments/assets/9472e970-c083-4b31-9252-3eeecc761078)

```bash
python3 hgemm.py --wmma-all --plot
```

<details>
<summary> 🔑️ 性能优化笔记(TODO) ！Click here! </summary>    

## 📖 性能优化笔记

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

- [CUDA编程概念】一、什么是bank conflict？](https://zhuanlan.zhihu.com/p/659142274)
- [解决 bank conflict](https://github.com/PaddleJitLab/CUDATutorial/blob/develop/docs/09_optimize_reduce/02_bank_conflict/README.md)
- [Bank Conflict free 的几种方式](https://zhuanlan.zhihu.com/p/722286440)
- [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [CUDA（三）：通用矩阵乘法：从入门到熟练](https://zhuanlan.zhihu.com/p/657632577)
- [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal)
- [tiny-flash-attention](https://github.com/66RING/tiny-flash-attention)
- [cute-gemm](https://github.com/reed-lau/cute-gemm)
- [cutlass_flash_atten_fp8](https://github.com/weishengying/cutlass_flash_atten_fp8)
- [cuda_learning](https://github.com/ifromeast/cuda_learning)
- [cuda_hgemm](https://github.com/Bruce-Lee-LY/cuda_hgemm)
- [cuda-tensorcore-hgemm](https://github.com/nicolaswilde/cuda-tensorcore-hgemm)
