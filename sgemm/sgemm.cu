#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/types.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

// modified from: https://zhuanlan.zhihu.com/p/657632577
// -------------------------------------- FP32 -------------------------------------- 
// SGEMM naive: compute one c[i,j] element per threads, all row major
__global__ void sgemm_naive_f32_kernel(float* a, float* b, float* c, int M, int N, int K) {

  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;

  if (m < M && n < N) {
    float psum = 0.0;
    #pragma unroll
    for (int k = 0; k < K; k++) {
      // m row in a matrix, n col in b matrix
      psum += a[m * K + k] * b[k * N + n];
    }
    c[m * N + n] = psum; // c[m,n]
  }
}

// SGEMM: Block Tile + K Tile, with smem
// Block Tile (BM, BN) + K Tile (BK=32)
// grid((N + BN - 1) / BN, (M + BM - 1) / BM), block(BN, BM)
// a: MxK, b: KxN, c: MxN, compute: c = a * b, all row major  
template<const int BM=32, const int BN=32, const int BK=32>
__global__ void sgemm_sliced_k_f32_kernel(float* a, float* b, float* c, int M, int N, int K) {
  // [1] Block Tile: 32x32的block处理c上一块32x32的元素计算
  // [2]     K Tile: 使用共享内存，并将K分块为BK大小的块
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
template<const int BM=128, const int BN=128, const int BK=8, const int TM=8, const int TN=8>
__global__ void sgemm_t_8x8_sliced_k_f32x4_kernel(float* a, float* b, float* c, int M, int N, int K) {
  // [1]  Block Tile: 一个16x16的block处理C上大小为128X128的一个目标块
  // [2] Thread Tile: 每个thread负责计算TM*TN(8*8)个元素，增加计算密度
  // [3]      K Tile: 将K分块，每块BK大小，迭代(K+BK-1/BK)次，
  //                  每次计算TM*TN个元素各自的部分乘累加
  // [4]   Vectorize: 减少load和store指令，使用float4
  
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

template<const int BM=128, const int BN=128, const int BK=8, const int TM=8, const int TN=8, const int OFFSET=0>
__global__ void sgemm_t_8x8_sliced_k_f32x4_bcf_kernel(
  float* a, float* b, float* c, const int M, const int N, const int K) {
  
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ float s_a[BK][BM + OFFSET];
  __shared__ float s_b[BK][BN + OFFSET];
  // __shared__ float s_a[BK][BM + 4];
  // __shared__ float s_b[BK][BN + 4];

  float r_load_a[TM/2]; // 4
  float r_load_b[TN/2]; // 4
  float r_comp_a[TM];
  float r_comp_b[TN];
  float r_c[TM][TN] = {0.0};

  // mapping tid to s_a[BK][BM], for each orginal m-th row, load 4 + 4 K-dim 
  // row major values from A matrix, and store it in COL major s_a[BK][BM].
  int load_a_smem_m = tid / 2; // tid / 2，(0,1,2,...,128)
  // (0b00000000 & 0b00000001) << 2 = 0
  // (0b00000001 & 0b00000001) << 2 = 4
  // (0b00000010 & 0b00000001) << 2 = 0
  // (0b00000011 & 0b00000001) << 2 = 4
  int load_a_smem_k = (tid & 1) << 2; // (0,4)
  // mapping tid to s_b[BK][BN], for each orginal k-th row, load 4 + 4 N-dim 
  // row major values from B matrix, and store it in ROW major s_b[BK][BN].
  int load_b_smem_k = tid / 32; // 0~8
  // (0b00000000 & 0b00011111) << 2 = 0
  // (0b00000001 & 0b00011111) << 2 = 4
  // (0b00000010 & 0b00011111) << 2 = 8
  // (0b00000011 & 0b00011111) << 2 = 12
  int load_b_smem_n = (tid & 31) << 2; // (0,4,8,12,...,124)

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  if (load_a_gmem_m >= M || load_b_gmem_n >= N) return;

  for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
    FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

    // 0. bank layout analysis: s_a[8][128]
    // 4 bytes per bank(32 banks, total 128 bytes, 32 float values), 
    // 1 float per bank. smem banks layout for s_a[8][128]:
    // 8*(128/32)=32 bank layers, 4 layers per k-th row.
    // [k=0][m=  [0],   [1],   [2],...,    [31]]
    // layer_0   [b0],  [b1],  [b2],...,   [b31]
    // [k=0][m=  [32],  [33],  [34],...,   [63]]
    // layer_1   [b0],  [b1],  [b2],...,   [b31]
    // [k=0][m=  [64],  [65],  [66],...,   [95]]
    // layer_2   [b0],  [b1],  [b2],...,   [b31]
    // [k=0][m=  [96],  [97],  [98],...,   [127]]
    // layer_3   [b0],  [b1],  [b2],...,   [b31]
    // ...       ...               ...
    // [k=7][m=  [0],   [1],   [2],...,    [31]]
    // layer_28  [b0],  [b1],  [b2],...,   [b31]
    // [k=7][m=  [32],  [33],  [34],...,   [63]]
    // layer_29  [b0],  [b1],  [b2],...,   [b31]
    // [k=7][m=  [64],  [65],  [66],...,   [95]]
    // layer_30  [b0],  [b1],  [b2],...,   [b31]
    // [k=7][m=  [96],  [97],  [98],...,   [127]]
    // layer_31  [b0],  [b1],  [b2],...,   [b31]
    // 1. bank conficts analysis: s_a[8][128]
    // tid 0   -> m 0,   k 0 -> all access bank 0  (layer_0/4/8/12)
    // tid 1   -> m 0,   k 4 -> all access bank 0  (layer_16/20/24/28)
    // tid 2   -> m 1,   k 0 -> all access bank 1  (layer_0/4/8/12)
    // tid 3   -> m 1,   k 4 -> all access bank 1  (layer_16/20/24/28)
    // tid 4   -> m 2,   k 0 -> all access bank 2  (layer_0/4/8/12)
    // tid 5   -> m 2,   k 4 -> all access bank 2  (layer_16/20/24/28)
    // tid 6   -> m 3,   k 0 -> all access bank 3  (layer_0/4/8/12)
    // tid 7   -> m 3,   k 4 -> all access bank 3  (layer_16/20/24/28)
    // ...        ...           ...                ...
    // tid 28  -> m 14,  k 0 -> all access bank 14 (layer_0/4/8/12)
    // tid 29  -> m 14,  k 4 -> all access bank 14 (layer_16/20/24/28)
    // tid 30  -> m 15,  k 0 -> all access bank 15 (layer_0/2/4/6)
    // tid 31  -> m 15,  k 4 -> all access bank 15 (layer_16/20/24/28)
    // conclusion: we still have bank conflicts for smem_a write access, 
    // each 2 consecutive threads within warp access the same bank! 
    // thus, we still need 2 memory issues as least per warp.
    s_a[load_a_smem_k    ][load_a_smem_m] = r_load_a[0]; // e.g layer_0  b0
    s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1]; // e.g layer_4  b0
    s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2]; // e.g layer_8  b0
    s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3]; // e.g layer_12 b0
    // 2. bank layout analysis: s_b[8][128] same as s_a[8][128]
    // 3. bank conficts analysis: s_b[8][128]
    // tid 0   -> k 0, n 0   -> all access bank 0~3   (layer_0)
    // tid 1   -> k 0, n 4   -> all access bank 4~7   (layer_0)
    // tid 2   -> k 0, n 8   -> all access bank 7~11  (layer_0)
    // tid 7   -> k 0, n 28  -> all access bank 28~31 (layer_0)
    // tid 8   -> k 0, n 32  -> all access bank 0~3   (layer_1)
    // ...        ...         ...                 ...
    // tid 15  -> k 0, n 60  -> all access bank 28~31 (layer_1)
    // tid 16  -> k 0, n 64  -> all access bank 0~3   (layer_2)
    // ...        ...         ...                 ...
    // tid 31  -> k 0, n 124 -> all access bank 28~31 (layer_3)
    // conclusion: we still have bank conflicts within warp, 
    // 0/8/16/24 -> bank 0~3, 1/9/17/25 -> bank 4~7, etc. 
    // thus, we still need 4 memory issues at least per warp.
    FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

    __syncthreads();

    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      // bank conflicts analysis, tx/ty 0~15, 0~7 bank 4*8=32 bytes
      // tid 0~15 access bank 0~3,  tid 16~31 access bank 4~7, etc.
      // tid 0,  tk 0 -> ty 0 -> [0][0+0~3],[0][64+0~3] -> bank 0~3(layer_0/2),   
      // tid 0,  tk 7 -> ty 0 -> [7][0+0~3],[0][64+0~3] -> bank 0~3(layer_28/30), 
      // tid 15, tk 0 -> ty 0 -> [0][0+0~3],[0][64+0~3] -> bank 0~3(layer_0/2),   
      // tid 15, tk 7 -> ty 0 -> [7][0+0~3],[0][64+0~3] -> bank 0~3(layer_28/30), 
      // tid 16, tk 0 -> ty 1 -> [0][0+4~7],[0][64+4~7] -> bank 4~7(layer_0/2),   
      // tid 16, tk 7 -> ty 1 -> [7][0+4~7],[0][64+4~7] -> bank 4~7(layer_28/30), 
      // tid 31, tk 0 -> ty 1 -> [0][0+4~7],[0][64+4~7] -> bank 4~7(layer_0/2),   
      // tid 31, tk 7 -> ty 1 -> [7][0+4~7],[0][64+4~7] -> bank 4~7(layer_28/30), 
      // tid 255,tk 0 -> ty 15 -> [0][0+60~63],[0][64+60~63] -> bank 28~31(layer_1/3),   
      // tid 255,tk 7 -> ty 15 -> [7][0+60~63],[0][64+60~63] -> bank 28~31(layer_29/31), 
      FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][ty * TM / 2         ]);
      FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][ty * TM / 2 + BM / 2]);
      // if (tid == < 32 && bx == 0 && by == 0) {
      //   printf("tid: %d, tx: %d, ty: %d, [%d][%d]\n", tid, tx, ty, tk, ty * TM / 2);
      //   printf("tid: %d, tx: %d, ty: %d, [%d][%d]\n", tid, tx, ty, tk, ty * TM / 2 + BM / 2);
      // }
      // conclusion: still have bank conflicts, need 16 memory issues ?

      // tid 0/8/16/24  access bank 0~3,  tid 1/9/17/25  access bank 4~7, 
      // tid 2/10/18/26 access bank 8~11, tid 7/15/23/31 access bank 28~31, etc.
      // tid 0, tk 0 -> tx 0 -> [0][0+0~3],[0][64+0~3] -> bank 0~3(layer_0/2),    
      // tid 0, tk 7 -> tx 0 -> [7][0+0~3],[0][64+0~3] -> bank 0~3(layer_28/30), 
      // tid 1, tk 0 -> tx 1 -> [0][0+4~7],[0][64+4~7] -> bank 4~7(layer_0/2),    
      // tid 1, tk 7 -> tx 1 -> [7][0+4~7],[0][64+4~7] -> bank 4~7(layer_28/30), 
      FLOAT4(r_comp_b[0]) = FLOAT4(s_b[tk][tx * TN / 2         ]);
      FLOAT4(r_comp_b[4]) = FLOAT4(s_b[tk][tx * TN / 2 + BN / 2]);
      // conclusion: still have some bank conflicts, need 4 memory issues.

      #pragma unroll
      for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
          r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }
    // sync per BK.
    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
  }
  #pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
  }
}

template<const int BM=128, const int BN=128, const int BK=8, const int TM=8, const int TN=8, const int OFFSET=0>
__global__ void sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel(
  float* a, float* b, float* c, const int M, const int N, const int K) {

  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ float s_a[2][BK][BM + OFFSET];
  __shared__ float s_b[2][BK][BN + OFFSET];

  float r_load_a[TM/2];
  float r_load_b[TN/2];
  float r_comp_a[TM];
  float r_comp_b[TN];
  float r_c[TM][TN] = {0.0};

  // mapping tid to s_a[BK][BM], for each orginal m-th row, load 4 + 4 K-dim 
  // row major values from A matrix, and store it in COL major s_a[BK][BM].
  int load_a_smem_m = tid / 2; // tid / 2，(0,1,2,...,128)
  // (0b00000000 & 0b00000001) << 2 = 0
  // (0b00000001 & 0b00000001) << 2 = 4
  // (0b00000010 & 0b00000001) << 2 = 0
  // (0b00000011 & 0b00000001) << 2 = 4
  int load_a_smem_k = (tid & 1) << 2; // (0,4)
  // mapping tid to s_b[BK][BN], for each orginal k-th row, load 4 + 4 N-dim 
  // row major values from B matrix, and store it in ROW major s_b[BK][BN].
  int load_b_smem_k = tid / 32; // 0~8
  // (0b00000000 & 0b00011111) << 2 = 0
  // (0b00000001 & 0b00011111) << 2 = 4
  // (0b00000010 & 0b00011111) << 2 = 8
  // (0b00000011 & 0b00011111) << 2 = 12
  int load_b_smem_n = (tid & 31) << 2; // (0,4,8,12,...,124)

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  // 1）主循环从bk = 1 开始，第一次数据加载在主循环之前，最后一次计算在主循环之后，这是pipeline 的特点决定的；
  // 2）由于计算和下一次访存使用的Shared Memory不同，因此主循环中每次循环只需要一次__syncthreads()即可
  // 3）由于GPU不能向CPU那样支持乱序执行，主循环中需要先将下一次循环计算需要的Gloabal Memory中的数据load 
  // 到寄存器，然后进行本次计算，之后再将load到寄存器中的数据写到Shared Memory，这样在LDG指令向Global 
  // Memory做load时，不会影响后续FFMA及其它运算指令的 launch 执行，也就达到了Double Buffering的目的。
  
  // bk = 0 is loading here, buffer 0

  {
    int load_a_gmem_k = load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
    FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

    s_a[0][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
    s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
  }
  // Without this synchronization, accuracy may occasionally be abnormal.
  __syncthreads(); 

  // bk start from 1，需要注意的是，虽然 bk 从 1 开始，但实际上 bk=1时，使用的是
  // 第0块BK中的数据（已经加载到共享内存s_a[0]和s_b[0]）；bk=2时，实际计算的是第1块
  // BK中的数据。其余以此类推，这个循环结束后，剩下最后一块BK大小的数据需要计算。
  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {

    int smem_sel = (bk - 1) & 1;
    int smem_sel_next = bk & 1;

    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
    FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2     ]);
      FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
      FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2     ]);
      FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

      #pragma unroll
      for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
          r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
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
    FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

    __syncthreads();
  }
  
  // 计算剩下最后一块BK
  #pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2     ]);
    FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
    FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2     ]);
    FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][tx * TN / 2 + BN / 2]);

    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
      #pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
        r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }

  #pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
  }
  #pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
  }
}

// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
if(((T).options().dtype() != (th_type))) {                   \
  std::cout << "Tensor Info:" << (T).options() << std::endl; \
  throw std::runtime_error("values must be "#th_type);       \
}

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)           \
if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
  throw std::runtime_error("Tensor size mismatch!");  \
}

// SGEMM naive: compute one c[i,j] element per threads, all row major
void sgemm_naive_f32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 32;
  constexpr int BN = 32;

  dim3 block(BN, BM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_naive_f32_kernel<<<grid, block>>>(
    reinterpret_cast<float*>(a.data_ptr()),
    reinterpret_cast<float*>(b.data_ptr()),
    reinterpret_cast<float*>(c.data_ptr()),
    M, N, K
  );
}

// SGEMM: Block Tile + K Tile, with smem
// Block Tile (BM, BN) + K Tile (BK=32)
// grid((N + BN - 1) / BN, (M + BM - 1) / BM), block(BN, BM)
// a: MxK, b: KxN, c: MxN, compute: c = a * b, all row major  
void sgemm_sliced_k_f32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;

  dim3 block(BN, BM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_sliced_k_f32_kernel<BM, BN, BK><<<grid, block>>>(
    reinterpret_cast<float*>(a.data_ptr()),
    reinterpret_cast<float*>(b.data_ptr()),
    reinterpret_cast<float*>(c.data_ptr()),
    M, N, K
  );
}

// SGEMM: Block Tile + Thread Tile + K Tile + Vec4, with smem
// BK:TILE_K=8 BM=BN=128
// TM=TN=8 增加计算密度 BM/TM=16 BN/TN=16
// dim3 blockDim(BN/TN, BM/TM);
// dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM)
void sgemm_t_8x8_sliced_k_f32x4(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8; 
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN/TN, BM/TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x8_sliced_k_f32x4_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
    reinterpret_cast<float*>(a.data_ptr()),
    reinterpret_cast<float*>(b.data_ptr()),
    reinterpret_cast<float*>(c.data_ptr()),
    M, N, K
  );
}

void sgemm_t_8x8_sliced_k_f32x4_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8; 
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN/TN, BM/TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x8_sliced_k_f32x4_bcf_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
    reinterpret_cast<float*>(a.data_ptr()),
    reinterpret_cast<float*>(b.data_ptr()),
    reinterpret_cast<float*>(c.data_ptr()),
    M, N, K
  );
}

void sgemm_t_8x8_sliced_k_f32x4_bcf_offset(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8; 
  constexpr int TM = 8;
  constexpr int TN = 8;
  constexpr int OFFSET = 4;

  dim3 block(BN/TN, BM/TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x8_sliced_k_f32x4_bcf_kernel<BM, BN, BK, TM, TN, OFFSET><<<grid, block>>>(
    reinterpret_cast<float*>(a.data_ptr()),
    reinterpret_cast<float*>(b.data_ptr()),
    reinterpret_cast<float*>(c.data_ptr()),
    M, N, K
  );
}

void sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8; 
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN/TN, BM/TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
    reinterpret_cast<float*>(a.data_ptr()),
    reinterpret_cast<float*>(b.data_ptr()),
    reinterpret_cast<float*>(c.data_ptr()),
    M, N, K
  );
}

void sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8; 
  constexpr int TM = 8;
  constexpr int TN = 8;
  constexpr int OFFSET = 4;

  dim3 block(BN/TN, BM/TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel<BM, BN, BK, TM, TN, OFFSET><<<grid, block>>>(
    reinterpret_cast<float*>(a.data_ptr()),
    reinterpret_cast<float*>(b.data_ptr()),
    reinterpret_cast<float*>(c.data_ptr()),
    M, N, K
  );
}

// from sgemm_async.cu
void sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
// from sgemm_wmma_tf32_stage.cu
void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stage2(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stage2_offset(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stage3(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stage3_offset(torch::Tensor a, torch::Tensor b, torch::Tensor c);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(sgemm_naive_f32)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_sliced_k_f32)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k_f32x4_bcf)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k_f32x4_bcf_offset)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_wmma_m16n16k8_mma4x2_warp2x4_stage2)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_wmma_m16n16k8_mma4x2_warp2x4_stage2_offset)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_wmma_m16n16k8_mma4x2_warp2x4_stage3)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_wmma_m16n16k8_mma4x2_warp2x4_stage3_offset)
}
