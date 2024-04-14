![cuda-learn-note](https://github.com/DefTruth/CUDA-Learn-Note/assets/31974251/882271fe-ab60-4b0e-9440-2e0fa3c0fb6f)   

<div align='center'>
  <img src=https://img.shields.io/badge/Language-CUDA-brightgreen.svg >
  <img src=https://img.shields.io/github/watchers/DefTruth/cuda-learn-note?color=9cc >
  <img src=https://img.shields.io/github/forks/DefTruth/cuda-learn-note.svg?style=social >
  <img src=https://img.shields.io/github/stars/DefTruth/cuda-learn-note.svg?style=social >
  <img src=https://img.shields.io/badge/Release-v0.3-brightgreen.svg >
  <img src=https://img.shields.io/badge/License-GPLv3.0-turquoise.svg >
 </div>   

📒**CUDA-Learn-Notes**: CUDA 笔记 / 大模型手撕CUDA / C++笔记，更新随缘: flash_attn、sgemm、sgemv、warp reduce、block reduce、dot、elementwise、softmax、layernorm、rmsnorm、histogram、relu、sigmoid etc.
<!--
<p align="center"> <a > 🌟如果觉得有用，不妨给个🌟👆🏻Star支持一下吧~ </a> </p>
-->

## 其他项目 🔥🔥 

|🛠[lite.ai.toolkit](https://github.com/DefTruth/lite.ai.toolkit) | 💎[torchlm](https://github.com/DefTruth/torchlm) | 📒[statistic-learning-R-note](https://github.com/DefTruth/statistic-learning-R-note) | 🎉[cuda-learn-note](https://github.com/DefTruth/cuda-learn-note) | 📖[Awesome-LLM-Inference](https://github.com/DefTruth/Awesome-LLM-Inference) |   
|:---:|:---:|:---:|:---:|:---:|
|![](https://img.shields.io/github/stars/DefTruth/lite.ai.toolkit.svg?style=social) ![](https://img.shields.io/github/downloads/DefTruth/lite.ai.toolkit/total?color=ccf&label=downloads&logo=github&logoColor=lightgrey)| ![](https://img.shields.io/github/stars/DefTruth/torchlm.svg?style=social)   ![](https://static.pepy.tech/personalized-badge/torchlm?period=total&units=international_system&left_color=grey&right_color=blue&left_text=downloads)| ![](https://img.shields.io/github/stars/DefTruth/statistic-learning-R-note.svg?style=social) ![](https://img.shields.io/github/downloads/DefTruth/statistic-learning-R-note/total?color=ccf&label=downloads&logo=github&logoColor=lightgrey) |![](https://img.shields.io/github/stars/DefTruth/cuda-learn-note.svg?style=social) ![](https://img.shields.io/github/issues/DefTruth/cuda-learn-note?color=9cc)|  ![](https://img.shields.io/github/stars/DefTruth/Awesome-LLM-Inference.svg?style=social) ![](https://img.shields.io/github/downloads/DefTruth/Awesome-LLM-Inference/total?color=ccf&label=downloads&logo=github&logoColor=lightgrey)|

## 0x00 前言
前段时间参加了一些**LLM AI Infra**面试，基本都要手撕**CUDA**⚡️，于是整体复习了一下**CUDA**优化的内容，也整理了一些高频题的写法。笔记分享在这里，不定期更新。关于**LLM AI Infra**，也推荐我整理的: 📖[Awesome-LLM-Inference](https://github.com/DefTruth/Awesome-LLM-Inference)  ![](https://img.shields.io/github/stars/DefTruth/Awesome-LLM-Inference.svg?style=social)

 
## 0x01 📖目录
<div id="kernellist"></div>  

- [x] 📖 [sgemm_naive_f32_kernel](#sgemm)
- [x] 📖 [sgemm_block_tile_k_tile_vec4_f32_kernel](#sgemm)
- [x] 📖 [sgemv_k32_f32_kernel](#sgemv)
- [x] 📖 [sgemv_k128_f32_kernel](#sgemv)
- [x] 📖 [sgemv_k16_f32_kernel](#sgemv)
- [x] 📖 [warp_reduce_sum/max_f32_kernel](#warpreduce)
- [x] 📖 [block_reduce_sum/max_f32_kernel](#warpreduce)
- [x] 📖 [block_all_reduce_f32_kernel](#blockallreduce)
- [x] 📖 [block_all_reduce_vec4_f32_kernel](#blockallreduce)
- [x] 📖 [dot_product_f32_kernel](#dot)
- [x] 📖 [dot_product_vec4_f32_kernel](#dot)
- [x] 📖 [elementwise_f32_kernel](#elementwise)
- [x] 📖 [elementwise_vec4_f32_kernel](#elementwise)
- [x] 📖 [histogram_i32_kernel](#histogram)
- [x] 📖 [histogram_vec4_i32_kernel](#histogram)
- [x] 📖 [softmax_f32_kernel (grid level memory fence)](#softmax)
- [x] 📖 [softmax_vec4_f32_kernel (grid level memory fence)](#softmax)
- [ ] 📖 [safe_softmax_f32_kernel (per token)](#softmax)
- [x] 📖 [sigmoid_f32_kernel](#sigmoid)
- [x] 📖 [sigmoid_vec4_f32_kernel](#sigmoid)
- [ ] 📖 [safe_sigmoid_f32_kernel](#sigmoid)
- [x] 📖 [relu_f32_kernel](#relu)
- [x] 📖 [relu_vec4_f32_kernel](#relu)
- [x] 📖 [layer_norm_f32_kernel (per token)](#layernorm)
- [x] 📖 [layer_norm_vec4_f32_kernel (per token)](#layernorm)
- [ ] 📖 [layer_norm_vec4_f16_kernel (per token)](#layernorm)
- [x] 📖 [rms_norm_f32_kernel (per token)](#rmsnorm)
- [x] 📖 [rms_norm_vec4_f32_kernel (per token)](#rmsnorm)
- [ ] 📖 [rms_norm_vec4_f16_kernel (per token)](#rmsnorm)
- [x] 📖 [flash_attn_1_fwd_f32_kernel](./flash_attn_1_fwd_f32.cu)
- [ ] 📖 flash_attn_2_fwd_f32_kernel
- [ ] 📖 flash_attn_2_fwd_f16_kernel
- [ ] 📖 flash_attn_2_fwd_b16_kernel
- [ ] 📖 flash_attn_2_fwd_f8_kernel
- [ ] 📖 flash_attn_2_split_kv_f16_kernel
- [ ] 📖 flash_attn_2_split_kv_b16_kernel
- [ ] 📖 flash_attn_2_split_kv_f8_kernel
- [ ] 📖 online_softmax_f32_kernel
- [ ] 📖 online_softmax_f16_kernel
- [ ] 📖 online_softmax_b16_kernel
- [ ] 📖 hgemm_f16_kernel
- [ ] 📖 sgemm_dbuf_f32_kernel

## 0x02 sgemm naive, sgemm + block-tile + k-tile + vec4  ([©️back👆🏻](#kernellist))  
<div id="sgemm"></div>  

```c++
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

// SGEMM: Block Tile + K Tile, with smem
// Block Tile (BM, BN) + K Tile (BK=32)
// grid((N + BN - 1) / BN, (M + BM - 1) / BM), block(BN, BM)
// a: MxK, b: KxN, c: MxN, compute: c = a * b, all row major  
__global__ void sgemm(float* a, float* b, float* c, int M, int N, int K) {
  // [1] Block Tile: 32x32的block处理c上一块32x32的元素计算
  // [2]     K Tile: 使用共享内存，并将K分块为BK大小的块
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;
  __shared__ float s_a[BM][BK], s_b[BK][BN]; 

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx; // tid within the block
  // load values to shared memory, 32x32 threads working together 
  // to fetch data along the row direction of a and b both for s_a 
  // and s_b 32x32x4x2=8KB, we use 32x32 threads within block to 
  // load 32x32 elements from global memory to shared memory, namely, 
  // each thread will load 1 element.
  int load_smem_a_m = tid / 32; // 0~31, tid / 32, tid / BM, threadIdx.y
  int load_smem_a_k = tid % 32; // 0~31, tid % 32, tid % BK, threadIdx.x
  int load_smem_b_k = tid / 32; // 0~31, tid / 32, tid / BK, threadIdx.y
  int load_smem_b_n = tid % 32; // 0~31, tid % 32, tid % BN, threadIdx.x
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  // if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;
  
  float sum = 0.f;
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_addr];
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_addr];
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      int comp_smem_a_m = load_smem_a_m;
      int comp_smem_b_n = load_smem_b_n;
      sum += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
    }
    __syncthreads();
  }
  int store_gmem_c_m = load_gmem_a_m;
  int store_gmem_c_n = load_gmem_b_n;
  int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
  c[store_gmem_c_addr] = sum;
}

// SGEMM: Block Tile + Thread Tile + K Tile + Vec4, with smem
// BK:TILE_K=8 BM=BN=128
// TM=TN=8 增加计算密度 BM/TM=16 BN/TN=16
// dim3 blockDim(BN/TN, BM/TM);
// dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM)
__global__ void sgemm_thread_tile_vec4(
  float* a, float* b, float* c, int M, int N, int K) {
  // [1]  Block Tile: 一个16x16的block处理C上大小为128X128的一个目标块
  // [2] Thread Tile: 每个thread负责计算TM*TN(8*8)个元素，增加计算密度
  // [3]      K Tile: 将K分块，每块BK大小，迭代(K+BK-1/BK)次，
  //                  每次计算TM*TN个元素各自的部分乘累加
  // [4]   Vectorize: 减少load和store指令，使用float4
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8; 
  constexpr int TM = 8;
  constexpr int TN = 8;

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx; // tid within the block
  __shared__ float s_a[BM][BK], s_b[BK][BN]; // 2*128*8*4=8KB
  
  // 0. 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
  // 对于s_a每行8个数据，每个线程读取4个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2; // tid/2 (128/8)*(128/8)=256 threads per block, tid/2->[0,128), BM=128 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 4;  // (tid%2 == 0) ? 0 : 4, col of s_a 0,4
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=8 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读4个数据，需要32个线程；总共8行，需要32x8=256个线程
  int load_smem_b_k = tid / 32; // tid/32, row of s_b 256/32=8 行 0~7
  int load_smem_b_n = (tid % 32) * 4;  // (tid % 32) * 4, col of s_b 0,4,...,124
  // 1. 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  
  float r_c[TM][TN] = {0.0}; // 8x8
  // 2. 先对K进行分块，每块BK大小
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    // 加载数据到共享内存smem s_a BM*BK 128*8 vectorize float4
    int load_gmem_a_k = bk * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    FLOAT4(s_a[load_smem_a_m][load_smem_a_k]) = FLOAT4(a[load_gmem_a_addr]);
    // 加载数据到共享内存smem s_b BK*BN 8*128 vectorize float4
    int load_gmem_b_k = bk * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 
    FLOAT4(s_b[load_smem_b_k][load_smem_b_n]) = FLOAT4(b[load_gmem_b_addr]); 
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < BK; k++) {
      // 3. 每个线程负责计算BM*BN(12x128)中的TM*TN(8x8)个元素
      #pragma unroll
      for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
          // k from 0~7，0 ~ BK, ty and tx range from 0 to 15, 16x8=128
          int comp_smem_a_m = ty * TM + m;  // 128*8 128/TM(8)=16 M方向 16线程
          int comp_smem_b_n = tx * TN + n;  // 8*128 128/TN(8)=16 N方向 16线程
          r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
        }
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int m = 0; m < TM; ++m) {
    int store_gmem_c_m = by * BM + ty * TM + m;
    #pragma unroll
    for (int n = 0; n < TN; n += 4) {
      int store_gmem_c_n = bx * BN + tx * TN + n;
      int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
      FLOAT4(c[store_gmem_c_addr]) = FLOAT4(r_c[m][n]);
    }
  }
}
```
这里gemm的实现比较简单，只使用了CUDA Cores，并且只实现Block Tile + K Tile以及Block Tile + K Tile+Thread Tile+向量化的版本。主要在于如何加载gmem中的数据到smem，也就是把全局内存中的数据索引mapping到共享内存中的。核心思维：把一个block中的线程id按照线性来理解，然后把这个线性的id和全局内存索引以及共享内存索引进行匹配。比如Block Tile + K Tile的实现，block内一共32x32个Threads，需要加载到smem的数据也是32x32，那么，最简单的做法，只需要每个线程加载一个互不重复数据即可。NOTE，本文的gemm kernel修改自：[紫气东来：CUDA（三）：通用矩阵乘法：从入门到熟练](https://zhuanlan.zhihu.com/p/657632577)


## 0x03 warp/block reduce sum/max  ([©️back👆🏻](#kernellist))
<div id="warpreduce"></div>  

```C++
// Warp Reduce Sum
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Warp Reduce Max
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// Block reduce sum/max/min device helper for Layer/RMS Norm/Softmax etc.
// grid 1D block 1D, grid(N/128), block(128)
template<const int NUM_THREADS=128>
__device__ __forceinline__ float block_reduce_sum(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];
  
  val = warp_reduce_sum<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = val;
  __syncthreads();
  val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  val = warp_reduce_sum<NUM_WARPS>(val);
  return val;
}

template<const int NUM_THREADS=128>
__device__ __forceinline__ float block_reduce_max(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];
  
  val = warp_reduce_max<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = val;
  __syncthreads();
  val = (lane < NUM_WARPS) ? shared[lane] : -FLT_MAX;
  val = warp_reduce_max<NUM_WARPS>(val);
  return val;
}
```
warp reduce几乎已经成为大部分reduce kernel的标准写法了，比如vLLM中，就是这种经典的写法。所以，先搞懂warp reduce（也就是搞懂各种warp functions的用法），再去写其他kernel，思路就会容易很多。需要注意的是，warp函数处理的是寄存器上的数据，也就是说，此时，没必要先加载数据到smem，再进行reduce，直接加载到寄存器即可（以前犯过这个小错误...）。Warp Functions建议参考：[jhang：CUDA编程入门之Warp-Level Primitives](https://zhuanlan.zhihu.com/p/572820783)

## 0x04 block all reduce + vec4  ([©️back👆🏻](#kernellist))
<div id="blockallreduce"></div>  

```c++
// Block All Reduce Sum
// grid(N/128), block(128)
// a: Nx1, y=sum(a)
template<const int NUM_THREADS = 128>
__global__ void block_all_reduce_sum(float* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  // keep the data in register is enougth for warp operaion.
  float sum = (idx < N) ? a[idx] : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum = warp_reduce_sum<WARP_SIZE>(sum);
  // warp leaders store the data to shared memory.
  if (lane == 0) reduce_smem[warp] = sum;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) sum = warp_reduce_sum<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, sum);
}

// Block All Reduce Sum + float4
// grid(N/128), block(128/4)
// a: Nx1, y=sum(a)
template<const int NUM_THREADS = 128/4>
__global__ void block_all_reduce_sum_vec4(float* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  float4 reg_a = FLOAT4(a[idx]);
  // keep the data in register is enougth for warp operaion.
  float sum = (idx < N) ? (reg_a.x + reg_a.y + reg_a.z + reg_a.w) : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum = warp_reduce_sum<WARP_SIZE>(sum);
  // warp leaders store the data to shared memory.
  if (lane == 0) reduce_smem[warp] = sum;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) sum = warp_reduce_sum<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, sum);
}
```
block all reduce是在warp reduce的基础上进行的，reduce_smem这部分的共享内存申请无法避免，这是用来同步每个warp之间得到局部结果。注意，最后，还需要atomicAdd做一个block级别的原子操作，以得到全局的和。float4向量化优化访存，可以减缓WarpScheduler发送指令的压力。

## 0x05 sgemv k32/k128/k16 kernel   ([©️back👆🏻](#kernellist))
<div id="sgemv"></div>  

```C++
// SGEMV: Warp SGEMV K32
// 假设K为32的倍数，每个warp负责一行
// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void sgemv_k32(float* a, float* x, float* y, int M, int K) {
  int tx = threadIdx.x;         // 0~31
  int ty = threadIdx.y;         // 0~4
  int bx = blockIdx.x;          // 0~M/4
  int lane = tx % WARP_SIZE;    // 0~31
  int m = bx * blockDim.y + ty; // (0~M/4) * 4 + (0~3)
  if (m < M) {
    float sum = 0.0f;
    int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;
    #pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
      // 若NUM_WARPS>=2，先将当前行的数据累加到第一个warp中
      int k = w * WARP_SIZE + lane;
      sum += a[m * K + k] * x[k];
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);
    if (lane == 0) y[m] = sum;
  }
}

// SGEMV: Warp SGEMV K128 + Vec4
// 假设K为128的倍数 float4
// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void sgemv_k128(float* a, float* x, float* y, int M, int K) {
  // 每个线程负责4个元素，一个warp覆盖128个元素
  int tx = threadIdx.x;         // 0~31
  int ty = threadIdx.y;         // 0~3
  int bx = blockIdx.x;          // 0~M/4
  int lane = tx % WARP_SIZE;    // 0~31
  int m = blockDim.y * bx + ty; // (0~M/4) * 4 + (0~3)
  
  if (m < M) {
    float sum = 0.0f;
    // process 4*WARP_SIZE elements per warp.
    int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
    #pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
      int k = (w * WARP_SIZE + lane) * 4;
      float4 reg_x = FLOAT4(x[k]);
      float4 reg_a = FLOAT4(a[m * K + k]);
      sum += (reg_a.x * reg_x.x + reg_a.y * reg_x.y 
            + reg_a.z * reg_x.z + reg_a.w * reg_x.w);
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);
    if(lane == 0) y[m] = sum;
  }
}

// SGEMV: Warp SGEMV K16
// 假设K为16 < 32,每个warp负责2行，每行有16个元素
// NUM_THREADS=128, NUM_WARPS=NUM_THREADS/WARP_SIZE;
// NUM_ROWS=NUM_WARPS * ROW_PER_WARP, grid(M/NUM_ROWS), block(32,NUM_WARPS)
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
template<const int ROW_PER_WARP = 2> 
__global__ void sgemv_k16(float* A, float* x, float* y, int M, int K) {
  constexpr int K_WARP_SIZE = (WARP_SIZE + ROW_PER_WARP - 1) / ROW_PER_WARP;
  int tx = threadIdx.x;       // 0~31
  int ty = threadIdx.y;       // 0~NUM_WARPS
  int bx = blockIdx.x;        // 0~M/NUM_ROWS (NUM_ROWS=NUM_WARPS * ROW_PER_WARP)
  int lane = tx % WARP_SIZE;  // 0~31
  int k = lane % K_WARP_SIZE; // 0~15
  // gloabl row of a: MxK and y:Mx1, blockDim.y=NUM_WARPS
  int m = (blockDim.y * bx + ty) * ROW_PER_WARP + lane / K_WARP_SIZE;
  if (m < M) {
    float sum = A[m * K + k] * x[k];
    sum = warp_reduce_sum<K_WARP_SIZE>(sum);
    // 注意是k == 0，而不是lane == 0
    if(k == 0) y[m] = sum; 
  }
}
```
估计有些大佬倒立都能写sgemv的各种优化版了，核心思路其实也是基于warp reduce，考虑K的不同情况进行优化。本文的sgemv kernel修改自：[有了琦琦的棍子：深入浅出GPU优化系列：gemv优化](https://zhuanlan.zhihu.com/p/494144694)

## 0x06 dot product, dot product + vec4  ([©️back👆🏻](#kernellist))
<div id="dot"></div>  

```c++
// Dot Product
// grid(N/128), block(128)
// a: Nx1, b: Nx1, y=sum(elementwise_mul(a,b))
template<const int NUM_THREADS = 128>
__global__ void dot(float* a, float* b, float* y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  // keep the data in register is enougth for warp operaion.
  float prod = (idx < N) ? a[idx] * b[idx] : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  prod = warp_reduce_sum<WARP_SIZE>(prod);
  // warp leaders store the data to shared memory.
  if (lane == 0) reduce_smem[warp] = prod;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) prod = warp_reduce_sum<NUM_WARPS>(prod);
  if (tid == 0) atomicAdd(y, prod);
}

// Dot Product + Vec4
// grid(N/128), block(128/4)
// a: Nx1, b: Nx1, y=sum(elementwise_mul(a,b))
template<const int NUM_THREADS = 128/4>
__global__ void dot_vec4(float* a, float* b, float* y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  float4 reg_a = FLOAT4(a[idx]);
  float4 reg_b = FLOAT4(b[idx]);
  float prod = (idx < N) ? (reg_a.x * reg_b.x + reg_a.y * reg_b.y 
                          + reg_a.z * reg_b.z + reg_a.w * reg_b.w) : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  prod = warp_reduce_sum<WARP_SIZE>(prod);
  // warp leaders store the data to shared memory.
  if (lane == 0) reduce_smem[warp] = prod;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) prod = warp_reduce_sum<NUM_WARPS>(prod);
  if (tid == 0) atomicAdd(y, prod);
}
```
dot product kernel的核心就是block reduce，不多说了。

## 0x07 elementwise, elementwise + vec4  ([©️back👆🏻](#kernellist))
<div id="elementwise"></div>  

```c++
// ElementWise Add  
// grid(N/128), block(128)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add(float* a, float* b, float* c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) c[idx] = a[idx] + b[idx];
}

// ElementWise Add + Vec4
// grid(N/128), block(128/4)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_vec4(float* a, float* b, float* c, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FLOAT4(c[idx]) = reg_c;
  }
}
```
elementwise可以考虑加点向量化进行访存优化。

## 0x08 histogram, histogram + vec4  
<div id="histogram"></div>  

```c++
// Histogram
// grid(N/128), block(128)
// a: Nx1, y: count histogram
__global__ void histogram(int* a, int* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) atomicAdd(&(y[a[idx]]), 1);
}

// Histogram + Vec4
// grid(N/128), block(128/4)
// a: Nx1, y: count histogram
__global__ void histogram_vec4(int* a, int* y, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    int4 reg_a = INT4(a[idx]);
    atomicAdd(&(y[reg_a.x]), 1);
    atomicAdd(&(y[reg_a.y]), 1);
    atomicAdd(&(y[reg_a.z]), 1);
    atomicAdd(&(y[reg_a.w]), 1);
  }
}
```
统计频数直方图，很简单，两行代码搞定。

## 0x09 softmax, softmax + vec4 (grid level memory fence)   ([©️back👆🏻](#kernellist))
<div id="softmax"></div>  

```c++
// Softmax x: N, y: N
// grid(N/128), block(K=128)
template<const int NUM_THREADS = 128>
__global__ void softmax(float* x, float* y, float* total, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid; 
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  
  float sum = (idx < N) ? expf(x[idx]) : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  sum = warp_reduce_sum<WARP_SIZE>(sum);
  if (lane == 0) reduce_smem[warp] = sum;
  __syncthreads();
  // compute the final sum in each warp
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  sum = warp_reduce_sum<NUM_WARPS>(sum); // sum(e^x_0,...,e^x_n-1)
  // get the total sum of all blocks.
  if (tid == 0) atomicAdd(total, sum);
  __threadfence(); // grid level memory fence 注意这里需要网格级别的内存同步
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if (idx < N) y[idx] = block_smem[tid] / (*total); 
}

// Softmax x: N, y: N
// grid(N/128), block(K=128)
template<const int NUM_THREADS = 128>
__global__ void softmax_v2(float* x, float* y, float* total, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid; 
  
  float exp_val = (idx < N) ? expf(x[idx]) : 0.0f;
  float sum = block_reduce_sum<NUM_THREADS>(exp_val);
  // get the total sum of all blocks.
  if (tid == 0) atomicAdd(total, sum);
  __threadfence(); // grid level memory fence  注意这里需要网格级别的内存同步
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if (idx < N) y[idx] = exp_val / (*total); 
}

// Softmax Vec4 x: N, y: N
// grid(N/128), block(128/4)
template<const int NUM_THREADS = 128/4>
__global__ void softmax_v2_vec4(float* x, float* y, float* total, int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * blockDim.x + tid) * 4; 
  
  float4 reg_x = FLOAT4(x[idx]);
  float4 reg_exp;
  reg_exp.x = (idx < N) ? expf(reg_x.x) : 0.0f;
  reg_exp.y = (idx < N) ? expf(reg_x.y) : 0.0f;
  reg_exp.z = (idx < N) ? expf(reg_x.z) : 0.0f;
  reg_exp.w = (idx < N) ? expf(reg_x.w) : 0.0f;
  float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
  float sum = block_reduce_sum<NUM_THREADS>(exp_val);
  // get the total sum of all blocks.
  if (tid == 0) atomicAdd(total, sum);
  __threadfence(); // grid level memory fence  注意这里需要网格级别的内存同步
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if (idx < N) {
    float4 reg_y;
    reg_y.x = reg_exp.x / (*total);
    reg_y.y = reg_exp.y / (*total);
    reg_y.z = reg_exp.z / (*total);
    reg_y.w = reg_exp.w / (*total);
    FLOAT4(y[idx]) = reg_y; 
  }
}
```
softmax稍微要注意的就是内存同步的问题，这里，你需要做一个网格级别的同步，而不能仅仅是block级别，否则拿不到全局的exp sum作为分母项。因此使用 __threadfence 这个网格及内存同步操作。不过效率我还没测过，实在要高效的话，可能得整成FA2那样的 1-pass + online softmax的实现。不过，如果是面试的话，就不要太为难自己了...，但是FA1/FA2的论文很经典，强烈建议多读几遍。

## 0x0a sigmoid, sigmoid + vec4   ([©️back👆🏻](#kernellist))
<div id="sigmoid"></div>  

```c++
// Sigmoid x: N, y: N y=1/(1+exp(-x))
// grid(N/128), block(K=128) 
__global__ void sigmoid(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = 1.0f / (1.0f + expf(-x[idx]));
}

// Sigmoid x: N, y: N y=1/(1+exp(-x)) Vec4
// grid(N/128), block(128/4)
__global__ void sigmoid_vec4(float* x, float* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = 1.0f / (1.0f + expf(-reg_x.x));
    reg_y.y = 1.0f / (1.0f + expf(-reg_x.y));
    reg_y.z = 1.0f / (1.0f + expf(-reg_x.z));
    reg_y.w = 1.0f / (1.0f + expf(-reg_x.w));
    FLOAT4(y[idx]) = reg_y;
  }
}
```

## 0x0b relu, relu + vec4   ([©️back👆🏻](#kernellist))
<div id="relu"></div>  

```c++
// Relu x: N, y: N y=max(0,x)
// grid(N/128), block(K=128) 
__global__ void relu(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = fmaxf(0.0f, x[idx]);
}

// Relu x: N, y: N y=max(0,x) Vec4
// grid(N/128/4), block(128/4) 
__global__ void relu_vec4(float* x, float* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = fmaxf(0.0f, reg_x.x);
    reg_y.y = fmaxf(0.0f, reg_x.y);
    reg_y.z = fmaxf(0.0f, reg_x.z);
    reg_y.w = fmaxf(0.0f, reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}
```

## 0x0c layer_norm, layer_norm + vec4   ([©️back👆🏻](#kernellist))
<div id="layernorm"></div>  

```c++
// Layer Norm: x: NxK(K=128<1024), y': NxK, y'=x-mean(x)/std(x) each row
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g + b (g: scale, b: bias)
template<const int NUM_THREADS=128>
__global__ void layer_norm(float* x, float* y, float g, float b, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x; // 0..N-1
  int idx = bid * blockDim.x + threadIdx.x;
  const float epsilon = 1e-5f;

  __shared__ float s_mean; // shared within block
  __shared__ float s_variance; // shared within block
  float value = (idx < N * K) ? x[idx] : 0.0f; // load once only
  float sum = block_reduce_sum<NUM_THREADS>(value);
  if (tid == 0) s_mean = sum / (float) K;
  // wait for s_mean in shared memory to be ready for all threads
  __syncthreads();
  float variance = (value - s_mean) * (value - s_mean);
  variance = block_reduce_sum<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float) K + epsilon);
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();
  if (idx < N * K) y[idx] = ((value - s_mean) * s_variance) * g + b;
}

// Layer Norm Vec4: x: NxK(K=128<1024), y': NxK, y'=x-mean(x)/std(x) each row
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
// grid(N*K/K), block(K/4<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g + b (g: scale, b: bias)
template<const int NUM_THREADS=128/4>
__global__ void layer_norm_vec4(float* x, float* y, float g, float b, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x; // 0..N-1
  int idx = (bid * blockDim.x + threadIdx.x) * 4;
  const float epsilon = 1e-5f;

  __shared__ float s_mean; // shared within block
  __shared__ float s_variance; // shared within block
  float4 reg_x = FLOAT4(x[idx])
  float value = (idx < N * K) ? (reg_x.x + reg_x.y 
                               + reg_x.z + reg_x.w) : 0.0f;
  float sum = block_reduce_sum<NUM_THREADS>(value);
  if (tid == 0) s_mean = sum / (float) K;
  // wait for s_mean in shared memory to be ready for all threads
  __syncthreads();
  float4 reg_x_hat;
  reg_x_hat.x = reg_x.x - s_mean;
  reg_x_hat.y = reg_x.y - s_mean;
  reg_x_hat.z = reg_x.z - s_mean;
  reg_x_hat.w = reg_x.w - s_mean;
  float variance = reg_x_hat.x * reg_x_hat.x + reg_x_hat.y * reg_x_hat.y 
                 + reg_x_hat.z * reg_x_hat.z + reg_x_hat.w * reg_x_hat.w;
  variance = block_reduce_sum<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float) K + epsilon);
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();
  float4 reg_y;
  reg_y.x = reg_x_hat.x * s_variance * g + b;
  reg_y.y = reg_x_hat.y * s_variance * g + b;
  reg_y.z = reg_x_hat.z * s_variance * g + b;
  reg_y.w = reg_x_hat.w * s_variance * g + b;
  if (idx < N * K) FLOAT4(y[idx]) = reg_y;
}
```
layer norm实现的核心同样也是block reduce和warp reduce，然后再整点向量化...

## 0x0d rms_norm, rms_norm + vec4   ([©️back👆🏻](#kernellist))
<div id="rmsnorm"></div>  

```c++
// RMS Norm: x: NxK(K=128<1024), y': NxK, y'=x/rms(x) each row
// 1/rms(x) = rsqrtf( sum(x^2)/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g (g: scale)
template<const int NUM_THREADS=128>
__global__ void rms_norm(float* x, float* y, float g, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x; // 0..N-1
  int idx = bid * blockDim.x + threadIdx.x;
  const float epsilon = 1e-5f;

  __shared__ float s_variance; // shared within block
  float value = (idx < N * K) ? x[idx] : 0.0f; // load once only
  float variance = value * value;
  variance = block_reduce_sum<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float) K + epsilon);
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads(); 
  if (idx < N * K) y[idx] = (value * s_variance) * g;
}

// RMS Norm Vec4: x: NxK(K=128<1024), y': NxK, y'=x/rms(x) each row
// 1/rms(x) = rsqrtf( sum(x^2)/K ) each row
// grid(N*K/K), block(K/4<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g (g: scale)
template<const int NUM_THREADS=128/4>
__global__ void rms_norm_vec4(float* x, float* y, float g, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x; // 0..N-1
  int idx = (bid * blockDim.x + threadIdx.x) * 4;
  const float epsilon = 1e-5f;

  __shared__ float s_variance; // shared within block
  float4 reg_x = FLOAT4(x[idx]);
  float variance = (idx < N * K) ? (reg_x.x * reg_x.x + reg_x.y * reg_x.y 
                                  + reg_x.z * reg_x.z + reg_x.w * reg_x.w) : 0.0f;
  variance = block_reduce_sum<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float) K + epsilon);
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads(); 
  float4 reg_y;
  reg_y.x = reg_x.x * s_variance * g;
  reg_y.y = reg_x.y * s_variance * g;
  reg_y.z = reg_x.z * s_variance * g;
  reg_y.w = reg_x.w * s_variance * g;
  if (idx < N * K) FLOAT4(y[idx]) = reg_y;
}
```
rms norm实现的核心同样也是block reduce和warp reduce...，然后再加点float4向量化什么的。

## 0x0e NMS  ([©️back👆🏻](#kernellist))
<div id="NMS"></div>  

```c++
struct Box {
  float x1, y1, x2, y2, score;
  float area() const {return (std::abs(x2 - x1 + 1)) * std::abs(y2 - y1 + 1); }
  float iou_of(const Box& other) const{
    float inner_x1 = x1 > other.x1 ? x1 : other.x1;
    float inner_y1 = y1 > other.y1 ? y1 : other.y1;
    float inner_x2 = x2 < other.x2 ? x2 : other.x2;
    float inner_y2 = y2 < other.y2 ? y2 : other.y2;
    float inner_h = inner_y2 - inner_y1 + 1.0f;
    float inner_w = inner_x2 - inner_x1 + 1.0f;
    float inner_area = inner_h * inner_w;
    return (inner_area / (area() + tbox.area() - inner_area));
  }
}
void hard_nms(std::vector<Box> &input, std::vector<Box> &output, float iou_threshold){
  if (input.empty()) return;
  std::sort(input.begin(), input.end(),[](Box& a, Box& b) { return a.score > b.score; });
  int box_num = input.size();
  std::vector<int> merged(box_num, 0);
  for (int i = 0; i < box_num; ++i) {
    if (merged[i]) continue;
    merged[i] = 1;
    for (int j = i + 1; j < box_num; ++j) {
      if (merged[j]) continue;
      float iou = input[i].iou_of(input[j]);
      if (iou > iou_threshold) merged[j] = 1;
    }
    output.push_back(input[i]);
  }
}
```
CV相关的经常会要手撕NMS，也记录下。

## 0x0f 总结  ([©️back👆🏻](#kernellist))
可以发现，大部分kernel的基本写法都是依赖warp reduce和block reduce的，基本上只要熟练应用warp functions各种场景的写法，应该问题不大；softmax需要考虑网格级同步的问题，或者online softmax以及FlashAttention；sgemm的优化是个很大的课题，不是案例中写的这么简单，但是入门的话，基本就是tiling的思想以及如何做索引之间的mapping；sgemv的优化则主要考虑K不同的值（因为M为1了），比如K=16,64,128等情况下，如何按照warp来处理；relu、sigmoid等都是elementwise的操作，很好实现，可以再考虑加点向量化优化访存；layer norm和rms norm在数学上其实也是挺清晰简单的，落实到cuda kernel时，只要按照逐个token来处理，headdim没有超过1024的情况下（一个block最多可以放1024个threads），可以放到一个block处理，这样并行化就很好写。当然，核心还是warp reduce和block reduce；NMS是乱入的，没有CUDA版本，别问了...

## ©️License
GNU General Public License v3.0

## References  
- [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal): Flash Attention in ~100 lines of CUDA (forward pass only)

## 🎉Contribute
🌟如果觉得有用，不妨给个🌟👆🏻Star支持一下吧~

<div align='center'>
<a href="https://star-history.com/#DefTruth/Awesome-LLM-Inference&Date">
  <picture align='center'>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=DefTruth/cuda-learn-note&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=DefTruth/cuda-learn-note&type=Date" />
    <img width=450 height=300 alt="Star History Chart" src="https://api.star-history.com/svg?repos=DefTruth/cuda-learn-note&type=Date" />
  </picture>
</a>  
</div>

