#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <torch/types.h>
#include <torch/extension.h>
using namespace nvcuda;

#define WARP_SIZE 32
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2*>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
// Support A and B matrix with row-major inorder to compare with the kernels using CUDA Cores in
// hgemm.cu and hgemm_async.cu. 

HOST_DEVICE_INLINE 
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// only 1 warp per block(32 threads), m16n16k16. A, B, C: all row_major.
template<const int WMMA_M=16, const int WMMA_N=16, const int WMMA_K=16>
__global__ void hgemm_wmma_m16n16k16_naive_kernel(half* A, half* B, half* C, 
                                                  int M, int N, int K) {
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  const int load_gmem_a_m = blockIdx.y * WMMA_M;
  const int load_gmem_b_n = blockIdx.x * WMMA_N;
  if (load_gmem_a_m >= M && load_gmem_b_n >= N) return;
  
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
  wmma::fill_fragment(C_frag, 0.0);
  
  #pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;
    
    wmma::load_matrix_sync(A_frag, A + load_gmem_a_m * K + k * WMMA_K,    K);
    wmma::load_matrix_sync(B_frag, B + (k * WMMA_K) * N  + load_gmem_b_n, N);

    wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    
    __syncthreads(); 
  }
  wmma::store_matrix_sync(C + load_gmem_a_m * N + load_gmem_b_n, C_frag, N, 
                          wmma::mem_row_major);
}

// m16n16k16 wmma  + tile MMA with smem,  A, B, C: all row_major.
template<const int WMMA_M=16, const int WMMA_N=16, const int WMMA_K=16, 
         const int WMMA_TILE_M=4, const int WMMA_TILE_N=2>
__global__ void hgemm_wmma_m16n16k16_mma4x2_kernel(
  half* A, half* B, half* C, int M, int N, int K) {
  // 256 threads(8 warps) per block.
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  constexpr int BM = WMMA_M * WMMA_TILE_M; // 16x4=64
  constexpr int BN = WMMA_N * WMMA_TILE_N; // 16x2=32
  constexpr int BK = WMMA_K; // 16
  __shared__ half s_a[BM][BK], s_b[WMMA_K][BN]; // 64x16x2=2KB, 16x32x2=1KB
  
  // 要保证相同的warp下thread执行相同的指令
  // warp_id 0 -> warp_m 0, warp_n 0
  // warp_id 1 -> warp_m 0, warp_n 1
  // warp_id 2 -> warp_m 1, warp_n 0
  // warp_id 3 -> warp_m 1, warp_n 1
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id / 2; // 0,1,2,3
  const int warp_n = warp_id % 2; // 0,1
  
  // 256线程分别load s_a=64x16, s_b=16x32
  // 64*16/256=4, half4, 16x32/256=2, half2
  // s_a, 64*16, 每个线程load 4 half, 每行需要4线程，64行，共256线程
  const int load_smem_a_m = tid / 4; // 0~63
  const int load_smem_a_k = (tid % 4) * 4; // 0,4,12,...
  // s_b, 16x32, 每个线程load 2 half, 每行需要8线程，32行，共256线程
  const int load_smem_b_k = tid / 16; // 0~16
  const int load_smem_b_n = (tid % 16) * 2; // 0,2,4,...,32
  const int load_gmem_a_m = by * BM + load_smem_a_m; // global m
  const int load_gmem_b_n = bx * BN + load_smem_b_n; // global n

  if (load_gmem_a_m >= M && load_gmem_b_n >= N) return;
 
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
  wmma::fill_fragment(C_frag, 0.0);
  
  #pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 
    // 64 bits sync memory issues gmem_a -> smem_a.
    LDST64BITS(s_a[load_smem_a_m][load_smem_a_k]) = (
      LDST64BITS(A[load_gmem_a_addr]));
    // 32 bits sync memory issues gmem_b -> smem_b.
    LDST32BITS(s_b[load_smem_b_k][load_smem_b_n]) = (
      LDST32BITS(B[load_gmem_b_addr]));
    __syncthreads();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;
  
    wmma::load_matrix_sync(A_frag, &s_a[warp_m * WMMA_M][0], BK); // BM*BK, BK=WMMA_K
    wmma::load_matrix_sync(B_frag, &s_b[0][warp_n * WMMA_N], BN); // BK=BN, BK=WMMA_K

    wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

    __syncthreads(); 
  }

  const int store_gmem_a_m = by * BM + warp_m * WMMA_M;
  const int store_gmem_a_n = bx * BN + warp_n * WMMA_N;
  wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag, N, 
                          wmma::mem_row_major);
}

// m16n16k16 wmma  + tile MMA with smem,  A, B, C: all row_major.
template<const int WMMA_M=16, const int WMMA_N=16, const int WMMA_K=16, 
         const int WMMA_TILE_M=4, const int WMMA_TILE_N=2, 
         const int WARP_TILE_M=2, const int WARP_TILE_N=4>
__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel(
  half* A, half* B, half* C, int M, int N, int K) {
  // 256 threads(8 warps) per block.
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
  constexpr int BK = WMMA_K; // 16
  __shared__ half s_a[BM][BK], s_b[BK][BN]; // 16x128x2=4KB
 
  // 要保证相同的warp下thread执行相同的指令
  // warp_id 0 -> warp_m 0, warp_n 0
  // warp_id 1 -> warp_m 0, warp_n 1
  // warp_id 2 -> warp_m 1, warp_n 0
  // warp_id 3 -> warp_m 1, warp_n 1
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id / 2; // 0,1,2,3
  const int warp_n = warp_id % 2; // 0,1
  
  // 0. 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
  // 对于s_a每行16个数据，每个线程读取8个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共16行，需要16x16=256个线程
  int load_smem_b_k = tid / 16; // row 0~15
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
  // 1. 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  wmma::fragment<wmma::accumulator, 
                 WMMA_M, WMMA_N, WMMA_K, 
                 half> C_frag[WARP_TILE_M][WARP_TILE_N];
  
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }
  
  #pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {

    int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 
    LDST128BITS(s_b[load_smem_b_k][load_smem_b_n]) = (
      LDST128BITS(B[load_gmem_b_addr]));
    LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) = (
      LDST128BITS(A[load_gmem_a_addr]));
    __syncthreads(); 
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> B_frag[WARP_TILE_N];

    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      wmma::load_matrix_sync(A_frag[i], &s_a[warp_smem_a_m][0], BK); // BM*BK, BK=WMMA_K
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_M) + j * WMMA_N;
      wmma::load_matrix_sync(B_frag[j], &s_b[0][warp_smem_b_n], BN); // BM*BK, BK=WMMA_K
    }

    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }
    __syncthreads(); 
  }
  
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int store_gmem_a_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_gmem_a_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag[i][j], N, 
                              wmma::mem_row_major);
    }
  }
}

// copy async
template<const int WMMA_M=16, const int WMMA_N=16, const int WMMA_K=16, 
         const int WMMA_TILE_M=4, const int WMMA_TILE_N=2, 
         const int WARP_TILE_M=2, const int WARP_TILE_N=4,
         const int OFFSET=0>
__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4_async_kernel(
  half* A, half* B, half* C, int M, int N, int K) {
  // 256 threads(8 warps) per block.
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
  constexpr int BK = WMMA_K; // 16
  // 16x128x2=4KB, 4+4=8KB, padding to reduce bank conflicts.
  __shared__ half s_a[BM][BK+OFFSET], s_b[BK][BN+OFFSET]; 
 
  // 要保证相同的warp下thread执行相同的指令
  // warp_id 0 -> warp_m 0, warp_n 0
  // warp_id 1 -> warp_m 0, warp_n 1
  // warp_id 2 -> warp_m 1, warp_n 0
  // warp_id 3 -> warp_m 1, warp_n 1
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id / 2; // 0,1,2,3
  const int warp_n = warp_id % 2; // 0,1
  
  // 0. 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
  // 对于s_a每行16个数据，每个线程读取8个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共16行，需要16x16=256个线程
  int load_smem_b_k = tid / 16; // row 0~15
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
  // 1. 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  wmma::fragment<wmma::accumulator, 
                 WMMA_M, WMMA_N, WMMA_K, 
                 half> C_frag[WARP_TILE_M][WARP_TILE_N];
  
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }
  
  #pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {

    int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
      &s_a[load_smem_a_m][load_smem_a_k]);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
    CP_ASYNC_COMMIT_GROUP();

    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
      &s_b[load_smem_b_k][load_smem_b_n]);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    CP_ASYNC_COMMIT_GROUP();

    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads(); 
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> B_frag[WARP_TILE_N];

    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      wmma::load_matrix_sync(A_frag[i], &s_a[warp_smem_a_m][0], BK+OFFSET); // BM*BK, BK=WMMA_K
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_M) + j * WMMA_N;
      wmma::load_matrix_sync(B_frag[j], &s_b[0][warp_smem_b_n], BN+OFFSET); // BM*BK, BK=WMMA_K
    }

    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }
    __syncthreads(); 
  }
  
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int store_gmem_a_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_gmem_a_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag[i][j], N, 
                              wmma::mem_row_major);
    }
  }
}

// Double buffers
template<const int WMMA_M=16, const int WMMA_N=16, const int WMMA_K=16, 
         const int WMMA_TILE_M=4, const int WMMA_TILE_N=2, 
         const int WARP_TILE_M=2, const int WARP_TILE_N=4,
         const int OFFSET=0>
__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel(
  half* A, half* B, half* C, int M, int N, int K) {
  // 256 threads(8 warps) per block.
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
  constexpr int BK = WMMA_K; // 16
  // 16x128x2=4KB, 4+4=8KB, padding to reduce bank conflicts.
  __shared__ half s_a[2][BM][BK+OFFSET], s_b[2][BK][BN+OFFSET]; 
 
  // 要保证相同的warp下thread执行相同的指令
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id / 2; // 0,1,2,3
  const int warp_n = warp_id % 2; // 0,1
  
  // 0. 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
  // 对于s_a每行16个数据，每个线程读取8个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共16行，需要16x16=256个线程
  int load_smem_b_k = tid / 16; // row 0~15
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
  // 1. 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  wmma::fragment<wmma::accumulator, 
                 WMMA_M, WMMA_N, WMMA_K, 
                 half> C_frag[WARP_TILE_M][WARP_TILE_N];
  
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }

  // k = 0 is loading here, buffer 0
  {
    int load_gmem_a_k = load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
      &s_a[0][load_smem_a_m][load_smem_a_k]);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
      &s_b[0][load_smem_b_k][load_smem_b_n]);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);
  }
  __syncthreads(); 
  
  #pragma unroll
  for (int k = 1; k < NUM_K_TILES; ++k) { // start from 1
    int smem_sel = (k - 1) & 1; // k 1->0, k 2->1, k 3->0, ...
    int smem_sel_next = k & 1;  // k 1->1, k 2->0, k 3->1, ...

    int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
      &s_a[smem_sel_next][load_smem_a_m][load_smem_a_k]);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
      &s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> B_frag[WARP_TILE_N];

    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][0], BK+OFFSET); 
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_M) + j * WMMA_N;
      wmma::load_matrix_sync(B_frag[j], &s_b[smem_sel][0][warp_smem_b_n], BN+OFFSET);
    }

    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);

    __syncthreads(); 
  }

  // processing last k tile
  {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> B_frag[WARP_TILE_N];

    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      wmma::load_matrix_sync(A_frag[i], &s_a[1][warp_smem_a_m][0], BK+OFFSET); 
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_M) + j * WMMA_N;
      wmma::load_matrix_sync(B_frag[j], &s_b[1][0][warp_smem_b_n], BN+OFFSET);
    }

    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }
  }
  
  // finally, store back to C matrix.
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int store_gmem_a_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_gmem_a_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag[i][j], N, 
                              wmma::mem_row_major);
    }
  }
}

// Chunk K(WARP_TILE_K): 减少__syncthreads同步次数，由于m16n16k16要求必须k=16
// 因此可以通过chunk方式，支持更大的k，比如k=32, chunk=2
template<const int WMMA_M=16, const int WMMA_N=16, const int WMMA_K=16, 
         const int WMMA_TILE_M=4, const int WMMA_TILE_N=2, 
         const int WARP_TILE_M=2, const int WARP_TILE_N=4,
         const int WARP_TILE_K=2, const int OFFSET=0>
__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_async_kernel(
  half* A, half* B, half* C, int M, int N, int K) {
  // 256 threads(8 warps) per block.
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K * WARP_TILE_K); // K/32
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
  constexpr int BK = WMMA_K * WARP_TILE_K; // 16*2=32
  // 32x128x2=8KB, 8+8=16KB, padding to reduce bank conflicts.
  __shared__ half s_a[BM][BK+OFFSET], s_b[BK][BN+OFFSET]; 
 
  // 要保证相同的warp下thread执行相同的指令
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id / 2; // 0,1,2,3
  const int warp_n = warp_id % 2; // 0,1
  
  // 0. 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
  // 对于s_a每行32个数据，每个线程读取16个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 16; // col 0,16
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=32 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读16个数据，需要8个线程；总共32行，需要8x32=256个线程
  int load_smem_b_k = tid / 8; // row 0~31
  int load_smem_b_n = (tid % 8) * 16; // col 0,16,32,...,112
  // 1. 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  wmma::fragment<wmma::accumulator, 
                 WMMA_M, WMMA_N, WMMA_K, 
                 half> C_frag[WARP_TILE_M][WARP_TILE_N];
  
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }
  
  #pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {

    int load_gmem_a_k = k * (WMMA_K * WARP_TILE_K) + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * (WMMA_K * WARP_TILE_K) + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
      &s_a[load_smem_a_m][load_smem_a_k]);
    // load 16 half values per threads
    #pragma unroll
    for (int i = 0; i < 16; i += 8) {
      CP_ASYNC_CG(load_smem_a_ptr + i * 2, &A[load_gmem_a_addr + i], 16);
    }

    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
      &s_b[load_smem_b_k][load_smem_b_n]);
    #pragma unroll
    for (int j = 0; j < 16; j += 8) {
      CP_ASYNC_CG(load_smem_b_ptr + j * 2, &B[load_gmem_b_addr + j], 16);
    }

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads(); 

    #pragma unroll
    for (int warp_k = 0; warp_k < WARP_TILE_K; ++warp_k) {
      const int warp_smem_k = warp_k * WMMA_K; // 0,16
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                    wmma::row_major> A_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                     wmma::row_major> B_frag[WARP_TILE_N];

      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
        const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        wmma::load_matrix_sync(A_frag[i], &s_a[warp_smem_a_m][warp_smem_k], BK+OFFSET);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
        const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_M) + j * WMMA_N;
        wmma::load_matrix_sync(B_frag[j], &s_b[warp_smem_k][warp_smem_b_n], BN+OFFSET); 
      }

      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
        }
      }
    }
    __syncthreads(); 
  }
  
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int store_gmem_a_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_gmem_a_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag[i][j], N, 
                              wmma::mem_row_major);
    }
  }
}

template<const int WMMA_M=16, const int WMMA_N=16, const int WMMA_K=16, 
         const int WMMA_TILE_M=4, const int WMMA_TILE_N=2, 
         const int WARP_TILE_M=2, const int WARP_TILE_N=4,
         const int WARP_TILE_K=2, const int OFFSET=0>
__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_dbuf_async_kernel(
  half* A, half* B, half* C, int M, int N, int K) {
  // 256 threads(8 warps) per block.
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K * WARP_TILE_K); // K/32
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
  constexpr int BK = WMMA_K * WARP_TILE_K; // 16*2=32
  // 32x128x2=8KB, 8+8=16KB, 16x2=32KB, padding to reduce bank conflicts.
  __shared__ half s_a[2][BM][BK+OFFSET], s_b[2][BK][BN+OFFSET]; 
 
  // 要保证相同的warp下thread执行相同的指令
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id / 2; // 0,1,2,3
  const int warp_n = warp_id % 2; // 0,1
  
  // 0. 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
  // 对于s_a每行32个数据，每个线程读取16个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 16; // col 0,16
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=32 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读16个数据，需要8个线程；总共32行，需要8x32=256个线程
  int load_smem_b_k = tid / 8; // row 0~31
  int load_smem_b_n = (tid % 8) * 16; // col 0,16,32,...,112
  // 1. 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  wmma::fragment<wmma::accumulator, 
                 WMMA_M, WMMA_N, WMMA_K, 
                 half> C_frag[WARP_TILE_M][WARP_TILE_N];
  
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }

  // k = 0 is loading here, buffer 0
  {
    int load_gmem_a_k = load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
      &s_a[0][load_smem_a_m][load_smem_a_k]);
    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
      &s_b[0][load_smem_b_k][load_smem_b_n]);
    #pragma unroll
    for (int i = 0; i < 16; i += 8) {
      CP_ASYNC_CG(load_smem_a_ptr + i * 2, &A[load_gmem_a_addr + i], 16);
      CP_ASYNC_CG(load_smem_b_ptr + i * 2, &B[load_gmem_b_addr + i], 16);
    }

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0); 
  }
  __syncthreads();
  
  #pragma unroll
  for (int k = 1; k < NUM_K_TILES; ++k) {
    int smem_sel = (k - 1) & 1; // k 1->0, k 2->1, k 3->0, ...
    int smem_sel_next = k & 1;  // k 1->1, k 2->0, k 3->1, ...

    int load_gmem_a_k = k * (WMMA_K * WARP_TILE_K) + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * (WMMA_K * WARP_TILE_K) + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
      &s_a[smem_sel_next][load_smem_a_m][load_smem_a_k]);
    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
      &s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]);
    #pragma unroll
    for (int i = 0; i < 16; i += 8) {
      CP_ASYNC_CG(load_smem_a_ptr + i * 2, &A[load_gmem_a_addr + i], 16);
      CP_ASYNC_CG(load_smem_b_ptr + i * 2, &B[load_gmem_b_addr + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();

    #pragma unroll
    for (int warp_k = 0; warp_k < WARP_TILE_K; ++warp_k) {
      const int warp_smem_k = warp_k * WMMA_K; // 0,16
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                    wmma::row_major> A_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                     wmma::row_major> B_frag[WARP_TILE_N];

      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
        const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][warp_smem_k], BK+OFFSET);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
        const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_M) + j * WMMA_N;
        wmma::load_matrix_sync(B_frag[j], &s_b[smem_sel][warp_smem_k][warp_smem_b_n], BN+OFFSET); 
      }

      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
        }
      }
    }

    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads(); 
  }
  
  // processing last k tile
  {
    #pragma unroll
    for (int warp_k = 0; warp_k < WARP_TILE_K; ++warp_k) {
      const int warp_smem_k = warp_k * WMMA_K; // 0,16
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                    wmma::row_major> A_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                     wmma::row_major> B_frag[WARP_TILE_N];

      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
        const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        wmma::load_matrix_sync(A_frag[i], &s_a[1][warp_smem_a_m][warp_smem_k], BK+OFFSET);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
        const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_M) + j * WMMA_N;
        wmma::load_matrix_sync(B_frag[j], &s_b[1][warp_smem_k][warp_smem_b_n], BN+OFFSET); 
      }

      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
        }
      }
    }
  }

  // finally, store back to C matrix.
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int store_gmem_a_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_gmem_a_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag[i][j], N, 
                              wmma::mem_row_major);
    }
  }
}

// 16 warps, mma4x4_warp2x2x2
template<const int WMMA_M=16, const int WMMA_N=16, const int WMMA_K=16, 
         const int WMMA_TILE_M=4, const int WMMA_TILE_N=4, 
         const int WARP_TILE_M=2, const int WARP_TILE_N=2,
         const int WARP_TILE_K=2, const int OFFSET=0>
__global__ void hgemm_wmma_m16n16k16_mma4x4_warp2x2x2_dbuf_async_kernel(
  half* A, half* B, half* C, int M, int N, int K) {
  // 512 threads(16 warps) per block.
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K * WARP_TILE_K); // K/32
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x4*2=128
  constexpr int BK = WMMA_K * WARP_TILE_K; // 16*2=32
  // 32x128x2=8KB, 8+8=16KB, 16x2=32KB, padding to reduce bank conflicts.
  __shared__ half s_a[2][BM][BK+OFFSET], s_b[2][BK][BN+OFFSET]; 
 
  // 要保证相同的warp下thread执行相同的指令
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // 0~15 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id / WMMA_TILE_N; // 0,1,2,3
  const int warp_n = warp_id % WMMA_TILE_M; // 0,1,2,3
  // 128x128需要8x8=64次m16n16k16 MMA计算, 每个warp执行4个MMA，16 warps
  
  // 0. 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
  // 对于s_a每行32个数据，每个线程读取8个，需要4个线程；总共128行，需要128x4刚好512线程
  int load_smem_a_m = tid / 4; // row 0~127
  int load_smem_a_k = (tid % 4) * 8; // col 0,8,24,32
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=32 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共32行，需要16x32=512个线程
  int load_smem_b_k = tid / 16; // row 0~31
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,16,...,128
  // 1. 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  wmma::fragment<wmma::accumulator, 
                 WMMA_M, WMMA_N, WMMA_K, 
                 half> C_frag[WARP_TILE_M][WARP_TILE_N];
  
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }

  // k = 0 is loading here, buffer 0
  {
    int load_gmem_a_k = load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
      &s_a[0][load_smem_a_m][load_smem_a_k]);
    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
      &s_b[0][load_smem_b_k][load_smem_b_n]);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0); 
  }
  __syncthreads();
  
  #pragma unroll
  for (int k = 1; k < NUM_K_TILES; ++k) {
    int smem_sel = (k - 1) & 1; // k 1->0, k 2->1, k 3->0, ...
    int smem_sel_next = k & 1;  // k 1->1, k 2->0, k 3->1, ...

    int load_gmem_a_k = k * (WMMA_K * WARP_TILE_K) + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * (WMMA_K * WARP_TILE_K) + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
      &s_a[smem_sel_next][load_smem_a_m][load_smem_a_k]);
    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
      &s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    CP_ASYNC_COMMIT_GROUP();

    #pragma unroll
    for (int warp_k = 0; warp_k < WARP_TILE_K; ++warp_k) {
      const int warp_smem_k = warp_k * WMMA_K; // 0,16
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                    wmma::row_major> A_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                     wmma::row_major> B_frag[WARP_TILE_N];

      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
        const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][warp_smem_k], BK+OFFSET);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
        const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_M) + j * WMMA_N;
        wmma::load_matrix_sync(B_frag[j], &s_b[smem_sel][warp_smem_k][warp_smem_b_n], BN+OFFSET); 
      }

      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
        }
      }
    }

    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads(); 
  }
  
  // processing last k tile
  {
    #pragma unroll
    for (int warp_k = 0; warp_k < WARP_TILE_K; ++warp_k) {
      const int warp_smem_k = warp_k * WMMA_K; // 0,16
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                    wmma::row_major> A_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                     wmma::row_major> B_frag[WARP_TILE_N];

      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
        const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        wmma::load_matrix_sync(A_frag[i], &s_a[1][warp_smem_a_m][warp_smem_k], BK+OFFSET);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
        const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_M) + j * WMMA_N;
        wmma::load_matrix_sync(B_frag[j], &s_b[1][warp_smem_k][warp_smem_b_n], BN+OFFSET); 
      }

      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
        }
      }
    }
  }

  // finally, store back to C matrix.
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int store_gmem_a_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_gmem_a_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag[i][j], N, 
                              wmma::mem_row_major);
    }
  }
}

// m32n8k16/m8n32k16 kernel
template<const int WMMA_M=32, const int WMMA_N=8, const int WMMA_K=16, 
         const int WMMA_TILE_M=2, const int WMMA_TILE_N=4, 
         const int WARP_TILE_M=2, const int WARP_TILE_N=4,
         const int OFFSET=0>
__global__ void hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async_kernel(
  half* A, half* B, half* C, int M, int N, int K) {
  // 256 threads(8 warps) per block.
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 32x2*2=128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 8x4*4=128
  constexpr int BK = WMMA_K; // 16
  // 16x128x2=4KB, 4+4=8KB, padding to reduce bank conflicts.
  __shared__ half s_a[2][BM][BK+OFFSET], s_b[2][BK][BN+OFFSET]; 
 
  // 要保证相同的warp下thread执行相同的指令
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id / 4; // 0,1
  const int warp_n = warp_id % 4; // 0,1,2,3
  
  // 0. 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
  // 对于s_a每行16个数据，每个线程读取8个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共16行，需要16x16=256个线程
  int load_smem_b_k = tid / 16; // row 0~15
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
  // 1. 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  wmma::fragment<wmma::accumulator, 
                 WMMA_M, WMMA_N, WMMA_K, 
                 half> C_frag[WARP_TILE_M][WARP_TILE_N];
  
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }

  // k = 0 is loading here, buffer 0
  {
    int load_gmem_a_k = load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
      &s_a[0][load_smem_a_m][load_smem_a_k]);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
      &s_b[0][load_smem_b_k][load_smem_b_n]);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);
  }
  __syncthreads(); 
  
  #pragma unroll
  for (int k = 1; k < NUM_K_TILES; ++k) { // start from 1
    int smem_sel = (k - 1) & 1; // k 1->0, k 2->1, k 3->0, ...
    int smem_sel_next = k & 1;  // k 1->1, k 2->0, k 3->1, ...

    int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
      &s_a[smem_sel_next][load_smem_a_m][load_smem_a_k]);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
      &s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> B_frag[WARP_TILE_N];

    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][0], BK+OFFSET); 
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_M) + j * WMMA_N;
      wmma::load_matrix_sync(B_frag[j], &s_b[smem_sel][0][warp_smem_b_n], BN+OFFSET);
    }

    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);

    __syncthreads(); 
  }

  // processing last k tile
  {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> B_frag[WARP_TILE_N];

    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      wmma::load_matrix_sync(A_frag[i], &s_a[1][warp_smem_a_m][0], BK+OFFSET); 
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_M) + j * WMMA_N;
      wmma::load_matrix_sync(B_frag[j], &s_b[1][0][warp_smem_b_n], BN+OFFSET);
    }

    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }
  }
  
  // finally, store back to C matrix.
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int store_gmem_a_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_gmem_a_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag[i][j], N, 
                              wmma::mem_row_major);
    }
  }
}

// TODO: regristers double buffers
template<const int WMMA_M=16, const int WMMA_N=16, const int WMMA_K=16, 
         const int WMMA_TILE_M=4, const int WMMA_TILE_N=2, 
         const int WARP_TILE_M=2, const int WARP_TILE_N=4,
         const int WARP_TILE_K=2, const int OFFSET=0>
__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_rbuf_async_kernel(
  half* A, half* B, half* C, int M, int N, int K) {
  // 256 threads(8 warps) per block.
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K * WARP_TILE_K); // K/32
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
  constexpr int BK = WMMA_K * WARP_TILE_K; // 16*2=32
  // 32x128x2=8KB, 8+8=16KB, 16x2=32KB, padding to reduce bank conflicts.
  __shared__ half s_a[2][BM][BK+OFFSET], s_b[2][BK][BN+OFFSET]; 
 
  // 要保证相同的warp下thread执行相同的指令
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id / 2; // 0,1,2,3
  const int warp_n = warp_id % 2; // 0,1
  
  // 0. 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
  // 对于s_a每行32个数据，每个线程读取16个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 16; // col 0,16
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=32 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读16个数据，需要8个线程；总共32行，需要8x32=256个线程
  int load_smem_b_k = tid / 8; // row 0~31
  int load_smem_b_n = (tid % 8) * 16; // col 0,16,32,...,112
  // 1. 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  wmma::fragment<wmma::accumulator, 
                 WMMA_M, WMMA_N, WMMA_K, 
                 half> C_frag[WARP_TILE_M][WARP_TILE_N];
  
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }

  // k = 0 is loading here, buffer 0
  {
    int load_gmem_a_k = load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
      &s_a[0][load_smem_a_m][load_smem_a_k]);
    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
      &s_b[0][load_smem_b_k][load_smem_b_n]);
    #pragma unroll
    for (int i = 0; i < 16; i += 8) {
      CP_ASYNC_CG(load_smem_a_ptr + i * 2, &A[load_gmem_a_addr + i], 16);
      CP_ASYNC_CG(load_smem_b_ptr + i * 2, &B[load_gmem_b_addr + i], 16);
    }

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0); 
  }
  __syncthreads();
  
  #pragma unroll
  for (int k = 1; k < NUM_K_TILES; ++k) {
    int smem_sel = (k - 1) & 1; // k 1->0, k 2->1, k 3->0, ...
    int smem_sel_next = k & 1;  // k 1->1, k 2->0, k 3->1, ...

    int load_gmem_a_k = k * (WMMA_K * WARP_TILE_K) + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * (WMMA_K * WARP_TILE_K) + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
      &s_a[smem_sel_next][load_smem_a_m][load_smem_a_k]);
    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
      &s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]);
    #pragma unroll
    for (int i = 0; i < 16; i += 8) {
      CP_ASYNC_CG(load_smem_a_ptr + i * 2, &A[load_gmem_a_addr + i], 16);
      CP_ASYNC_CG(load_smem_b_ptr + i * 2, &B[load_gmem_b_addr + i], 16);
    }

    // regristers double buffers
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> A_frag[2][WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> B_frag[2][WARP_TILE_N];

    int reg_store_idx = 0;
    int reg_load_idx = 1;

    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      wmma::load_matrix_sync(A_frag[reg_store_idx][i], &s_a[smem_sel][warp_smem_a_m][0], 
                             BK+OFFSET);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_M) + j * WMMA_N;
      wmma::load_matrix_sync(B_frag[reg_store_idx][j], &s_b[smem_sel][0][warp_smem_b_n], 
                             BN+OFFSET); 
    }
    
    // warp_k start from 1
    #pragma unroll
    for (int warp_k = 1; warp_k < WARP_TILE_K; ++warp_k) {
      reg_store_idx ^= 1; // 0 -> 1
      reg_load_idx ^= 1; // 1 -> 0

      const int warp_smem_k = warp_k * WMMA_K; // 0,16

      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
        const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        wmma::load_matrix_sync(A_frag[reg_store_idx][i], 
                               &s_a[smem_sel][warp_smem_a_m][warp_smem_k], 
                               BK+OFFSET);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
        const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_M) + j * WMMA_N;
        wmma::load_matrix_sync(B_frag[reg_store_idx][j], 
                               &s_b[smem_sel][warp_smem_k][warp_smem_b_n], 
                               BN+OFFSET); 
      }

      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          wmma::mma_sync(C_frag[i][j], A_frag[reg_load_idx][i], 
                         B_frag[reg_load_idx][j], C_frag[i][j]);
        }
      }
    }

    // process last warp_k
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[reg_store_idx][i], 
                       B_frag[reg_store_idx][j], C_frag[i][j]);
      }
    }

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads(); 
  }
  
  // processing last k tile
  {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> A_frag[2][WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> B_frag[2][WARP_TILE_N];

    int reg_store_idx = 0;
    int reg_load_idx = 1;

    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      wmma::load_matrix_sync(A_frag[reg_store_idx][i], &s_a[1][warp_smem_a_m][0], 
                             BK+OFFSET);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_M) + j * WMMA_N;
      wmma::load_matrix_sync(B_frag[reg_store_idx][j], &s_b[1][0][warp_smem_b_n], 
                             BN+OFFSET); 
    }
    
    // warp_k start from 1
    #pragma unroll
    for (int warp_k = 1; warp_k < WARP_TILE_K; ++warp_k) {
      reg_store_idx ^= 1; // 0 -> 1
      reg_load_idx ^= 1; // 1 -> 0

      const int warp_smem_k = warp_k * WMMA_K; // 0,16

      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
        const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        wmma::load_matrix_sync(A_frag[reg_store_idx][i], 
                               &s_a[1][warp_smem_a_m][warp_smem_k], 
                               BK+OFFSET);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
        const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_M) + j * WMMA_N;
        wmma::load_matrix_sync(B_frag[reg_store_idx][j], 
                               &s_b[1][warp_smem_k][warp_smem_b_n], 
                               BN+OFFSET); 
      }

      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          wmma::mma_sync(C_frag[i][j], A_frag[reg_load_idx][i], 
                         B_frag[reg_load_idx][j], C_frag[i][j]);
        }
      }
    }

    // process last warp_k
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[reg_store_idx][i], 
                       B_frag[reg_store_idx][j], C_frag[i][j]);
      }
    }
  }

  // finally, store back to C matrix.
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int store_gmem_a_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_gmem_a_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag[i][j], N, 
                              wmma::mem_row_major);
    }
  }
}


// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)   \
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

// 1 warp per block(32 threads), m16n16k16. A, B, C: all row_major.
void hgemm_wmma_m16n16k16_naive(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16; 

  dim3 block(WARP_SIZE);
  dim3 grid(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));
 
  hgemm_wmma_m16n16k16_naive_kernel<
    WMMA_M, WMMA_N, WMMA_K><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

void hgemm_wmma_m16n16k16_mma4x2(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2; 
  constexpr int NUM_THREADS= (
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 2 * 32 = 256

  dim3 block(NUM_THREADS);
  dim3 grid(div_ceil(N, WMMA_N * WMMA_TILE_N), 
            div_ceil(M, WMMA_M * WMMA_TILE_M));
 
  hgemm_wmma_m16n16k16_mma4x2_kernel<
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

void hgemm_wmma_m16n16k16_mma4x2_warp2x4(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2; 
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  constexpr int NUM_THREADS= (
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 2 * 32 = 256

  dim3 block(NUM_THREADS);
  dim3 grid(div_ceil(N, WMMA_M * WMMA_TILE_M * WARP_TILE_M), 
            div_ceil(M, WMMA_N * WMMA_TILE_N * WARP_TILE_N));
 
  hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel<
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, 
    WARP_TILE_M, WARP_TILE_N><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

// copy async
void hgemm_wmma_m16n16k16_mma4x2_warp2x4_async(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2; 
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  constexpr int NUM_THREADS= (
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 2 * 32 = 256

  dim3 block(NUM_THREADS);
  dim3 grid(div_ceil(N, WMMA_M * WMMA_TILE_M * WARP_TILE_M), 
            div_ceil(M, WMMA_N * WMMA_TILE_N * WARP_TILE_N));
 
  hgemm_wmma_m16n16k16_mma4x2_warp2x4_async_kernel<
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, 
    WARP_TILE_M, WARP_TILE_N><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

// padding, bank conflicts reduce.
void hgemm_wmma_m16n16k16_mma4x2_warp2x4_async_offset(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2; 
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  constexpr int NUM_THREADS= (
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 2 * 32 = 256

  dim3 block(NUM_THREADS);
  dim3 grid(div_ceil(N, WMMA_M * WMMA_TILE_M * WARP_TILE_M), 
            div_ceil(M, WMMA_N * WMMA_TILE_N * WARP_TILE_N));
  
  // padding offset 8
  hgemm_wmma_m16n16k16_mma4x2_warp2x4_async_kernel<
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, 
    WARP_TILE_M, WARP_TILE_N, 8><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

// double buffer
void hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2; 
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  constexpr int NUM_THREADS= (
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 2 * 32 = 256

  dim3 block(NUM_THREADS);
  dim3 grid(div_ceil(N, WMMA_M * WMMA_TILE_M * WARP_TILE_M), 
            div_ceil(M, WMMA_N * WMMA_TILE_N * WARP_TILE_N));
 
  hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel<
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, 
    WARP_TILE_M, WARP_TILE_N><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

// double buffer, padding
void hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_offset(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2; 
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  constexpr int NUM_THREADS= (
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 2 * 32 = 256

  dim3 block(NUM_THREADS);
  dim3 grid(div_ceil(N, WMMA_M * WMMA_TILE_M * WARP_TILE_M), 
            div_ceil(M, WMMA_N * WMMA_TILE_N * WARP_TILE_N));
 
  hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel<
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, 
    WARP_TILE_M, WARP_TILE_N, 8><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

// warp2x4x2, warp tile k, smem 16KB
void hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_async(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2; 
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  constexpr int WARP_TILE_K = 2;
  constexpr int NUM_THREADS= (
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 2 * 32 = 256

  dim3 block(NUM_THREADS);
  dim3 grid(div_ceil(N, WMMA_M * WMMA_TILE_M * WARP_TILE_M), 
            div_ceil(M, WMMA_N * WMMA_TILE_N * WARP_TILE_N));
 
  hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_async_kernel<
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, 
    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

void hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_async_offset(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2; 
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  constexpr int WARP_TILE_K = 2;
  constexpr int NUM_THREADS= (
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 2 * 32 = 256

  dim3 block(NUM_THREADS);
  dim3 grid(div_ceil(N, WMMA_M * WMMA_TILE_M * WARP_TILE_M), 
            div_ceil(M, WMMA_N * WMMA_TILE_N * WARP_TILE_N));
 
  hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_async_kernel<
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, 
    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, 8><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

// mma4x2_warp2x4x2, smem 32KB
void hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_dbuf_async(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2; 
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  constexpr int WARP_TILE_K = 2;
  constexpr int NUM_THREADS= (
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 2 * 32 = 256

  dim3 block(NUM_THREADS);
  dim3 grid(div_ceil(N, WMMA_M * WMMA_TILE_M * WARP_TILE_M), 
            div_ceil(M, WMMA_N * WMMA_TILE_N * WARP_TILE_N));
 
  hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_dbuf_async_kernel<
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, 
    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

void hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_dbuf_async_offset(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2; 
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  constexpr int WARP_TILE_K = 2;
  constexpr int NUM_THREADS= (
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 2 * 32 = 256

  dim3 block(NUM_THREADS);
  dim3 grid(div_ceil(N, WMMA_M * WMMA_TILE_M * WARP_TILE_M), 
            div_ceil(M, WMMA_N * WMMA_TILE_N * WARP_TILE_N));
 
  hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_dbuf_async_kernel<
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, 
    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, 8><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

// mma4x4_warp2x2x2, smem 32KB
void hgemm_wmma_m16n16k16_mma4x4_warp2x2x2_dbuf_async(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 4; 
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 2;
  constexpr int WARP_TILE_K = 2;
  constexpr int NUM_THREADS= (
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 4 * 32 = 512

  dim3 block(NUM_THREADS);
  dim3 grid(div_ceil(N, WMMA_M * WMMA_TILE_M * WARP_TILE_M), 
            div_ceil(M, WMMA_N * WMMA_TILE_N * WARP_TILE_N));
 
  hgemm_wmma_m16n16k16_mma4x4_warp2x2x2_dbuf_async_kernel<
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, 
    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

void hgemm_wmma_m16n16k16_mma4x4_warp2x2x2_dbuf_async_offset(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 4; 
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 2;
  constexpr int WARP_TILE_K = 2;
  constexpr int NUM_THREADS= (
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 4 * 32 = 512

  dim3 block(NUM_THREADS);
  dim3 grid(div_ceil(N, WMMA_M * WMMA_TILE_M * WARP_TILE_M), 
            div_ceil(M, WMMA_N * WMMA_TILE_N * WARP_TILE_N));
 
  hgemm_wmma_m16n16k16_mma4x4_warp2x2x2_dbuf_async_kernel<
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, 
    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, 8><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

// m32n8k16
void hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int WMMA_M = 32;
  constexpr int WMMA_N = 8;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 2;
  constexpr int WMMA_TILE_N = 4; 
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  constexpr int NUM_THREADS= (
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 2 * 4 * 32 = 256

  dim3 block(NUM_THREADS);
  dim3 grid(div_ceil(N, WMMA_M * WMMA_TILE_M * WARP_TILE_M), 
            div_ceil(M, WMMA_N * WMMA_TILE_N * WARP_TILE_N));
 
  hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async_kernel<
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, 
    WARP_TILE_M, WARP_TILE_N><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

void hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async_offset(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int WMMA_M = 32;
  constexpr int WMMA_N = 8;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 2;
  constexpr int WMMA_TILE_N = 4; 
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  constexpr int NUM_THREADS= (
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 2 * 4 * 32 = 256

  dim3 block(NUM_THREADS);
  dim3 grid(div_ceil(N, WMMA_M * WMMA_TILE_M * WARP_TILE_M), 
            div_ceil(M, WMMA_N * WMMA_TILE_N * WARP_TILE_N));
 
  hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async_kernel<
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, 
    WARP_TILE_M, WARP_TILE_N, 8><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

// regristers double buffers
void hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_rbuf_async(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2; 
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  constexpr int WARP_TILE_K = 2;
  constexpr int NUM_THREADS= (
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 2 * 32 = 256

  dim3 block(NUM_THREADS);
  dim3 grid(div_ceil(N, WMMA_M * WMMA_TILE_M * WARP_TILE_M), 
            div_ceil(M, WMMA_N * WMMA_TILE_N * WARP_TILE_N));
 
  hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_rbuf_async_kernel<
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, 
    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

void hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_rbuf_async_offset(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2; 
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  constexpr int WARP_TILE_K = 2;
  constexpr int NUM_THREADS= (
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 2 * 32 = 256

  dim3 block(NUM_THREADS);
  dim3 grid(div_ceil(N, WMMA_M * WMMA_TILE_M * WARP_TILE_M), 
            div_ceil(M, WMMA_N * WMMA_TILE_N * WARP_TILE_N));
 
  hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_rbuf_async_kernel<
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, 
    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K><<<grid, block>>>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}
