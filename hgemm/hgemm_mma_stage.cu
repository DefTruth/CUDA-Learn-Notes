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
// gmem -> smem
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
// smem -> gmem: requires sm_90 or higher.
#define CP_ASYNC_BULK_COMMIT_GROUP() asm volatile("cp.async.bulk.commit_group;\n" ::)
#define CP_ASYNC_BULK_WAIT_ALL() asm volatile("cp.async.bulk.wait_all;\n" ::)
#define CP_ASYNC_BULK_WAIT_GROUP(n) asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_BULK(dst, src, bytes) asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
// ldmatrix
#define LDMATRIX_X1(R, addr) asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))
#define LDMATRIX_X2(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
#define LDMATRIX_X1_T(R, addr) asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
// stmatrix: requires sm_90 or higher.
#define STMATRIX_X1(addr, R) asm volatile("stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\n" :: "r"(addr), "r"(R))
#define STMATRIX_X2(addr, R0, R1) asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n" :: "r"(addr), "r"(R0), "r"(R1))
#define STMATRIX_X4(addr, R0, R1, R2, R3) asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(addr), "r"(R0), "r"(R1), "r"(R2), "r"(R3))
#define STMATRIX_X1_T(addr, R) asm volatile("stmatrix.sync.aligned.x1.trans.m8n8.shared.b16 [%0], {%1};\n" :: "r"(addr), "r"(R))
#define STMATRIX_X2_T(addr, R0, R1) asm volatile("stmatrix.sync.aligned.x2.trans.m8n8.shared.b16 [%0], {%1, %2};\n" :: "r"(addr), "r"(R0), "r"(R1))
#define STMATRIX_X4_T(addr, R0, R1, R2, R3) asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(addr), "r"(R0), "r"(R1), "r"(R2), "r"(R3))
// mma m16n8k16
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" : "=r"(RD0), "=r"(RD1) : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

HOST_DEVICE_INLINE 
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle
template<const int MMA_M=16, 
         const int MMA_N=8, 
         const int MMA_K=16,
         const int MMA_TILE_M=2,
         const int MMA_TILE_N=4,
         const int WARP_TILE_M=4,
         const int WARP_TILE_N=4,
         const int A_PAD=0, 
         const int B_PAD=0,
         const int K_STAGE=2, 
         const bool BLOCK_SWIZZLE=false>
__global__ void  __launch_bounds__(256) 
hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_kernel(
  half* A, half* B, half* C, int M, int N, int K) {
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int) BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
  constexpr int BK = MMA_K; // 16

  __shared__ half s_a[K_STAGE][BM][BK+A_PAD]; // 128*16*2=4KB
  __shared__ half s_b[K_STAGE][BK][BN+B_PAD]; // 16*128*2=4KB, 16*(128+16)*2=4.5KB
  constexpr int s_a_stage_offset = BM * (BK + A_PAD);
  constexpr int s_b_stage_offset = BK * (BN + B_PAD);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2; // 0,1
  const int warp_n = warp_id / 2; // 0,1,2,3

  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  int load_smem_b_k = tid / 16; // row 0~15
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      RC[i][j][0] = 0;
      RC[i][j][1] = 0;
    }
  }

  // may avoid cvta overhead ? only cvta smem base ptr once for cp.async.
  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  #pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (k * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (k * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE-2); // s2->0, s3->1, s4->2
  __syncthreads(); 

  #pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    // gmem -> smem
    // s2/4 can use bitwise ops but s3 can not, so, we use mod
    // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
    // s3: (k + 1) % 3
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    CP_ASYNC_COMMIT_GROUP();
    
    // ldmatrix for s_a, ldmatrix.trans for s_b.
    uint32_t RA[WARP_TILE_M][4];
    uint32_t RB[WARP_TILE_N][2];

    // smem -> reg
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_a_ptr = __cvta_generic_to_shared(
        &s_a[smem_sel][lane_smem_a_m][lane_smem_a_k]);
      LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;  // 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr = __cvta_generic_to_shared(
        &s_b[smem_sel][lane_smem_b_k][lane_smem_b_n]);
      LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
    }
    
    // MMA compute
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        HMMA16816(RC[i][j][0], RC[i][j][1], 
                  RA[i][0], RA[i][1], RA[i][2], RA[i][3], 
                  RB[j][0], RB[j][1], 
                  RC[i][j][0], RC[i][j][1]);
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE-2);
    __syncthreads(); 
  }

  // make sure all memory issues ready.
  if ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads(); 
  }

  // processing last (K_STAGE-1) k iters.
  {
    #pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      // ldmatrix for s_a, ldmatrix.trans for s_b.
      uint32_t RA[WARP_TILE_M][4];
      uint32_t RB[WARP_TILE_N][2];

      // smem -> reg
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
        int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
        uint32_t lane_smem_a_ptr = __cvta_generic_to_shared(
          &s_a[stage_sel][lane_smem_a_m][lane_smem_a_k]);
        LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int lane_smem_b_k = lane_id % 16;  // 0~15
        int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
        uint32_t lane_smem_b_ptr = __cvta_generic_to_shared(
          &s_b[stage_sel][lane_smem_b_k][lane_smem_b_n]);
        LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
      }

      // MMA compute
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          HMMA16816(RC[i][j][0], RC[i][j][1], 
                    RA[i][0], RA[i][1], RA[i][2], RA[i][3], 
                    RB[j][0], RB[j][1], 
                    RC[i][j][0], RC[i][j][1]);
        }
      }
    }
  }

  // reg -> gmem, MMA_MxMMA_N=16x8
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      // mapping lane smem index -> global index.
      // [16][8], https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
      // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
      // [0~7][0~3 u32 -> 0~7 f16], [8~15][0~3 u32 -> 0~7 f16]
      int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
      int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n + (lane_id % 4) * 2;
      int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
      int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
      LDST32BITS(C[store_gmem_c_addr_0]) = LDST32BITS(RC[i][j][0]); 
      LDST32BITS(C[store_gmem_c_addr_1]) = LDST32BITS(RC[i][j][1]); 
      // TODO: How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
      // thus, we only need 8 memory issues with 128 bits after shfl_sync.
      // uint32_t r_store_c_0[4], r_store_c_1[4];
      // r_store_c_0[0] = RC[i][j][0]; r_store_c_1[0] = RC[i][j][1];
      // r_store_c_0[1] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
      // r_store_c_0[2] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
      // r_store_c_0[3] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
      // r_store_c_1[1] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
      // r_store_c_1[2] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
      // r_store_c_1[3] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
      // if (lane_id % 4 == 0) {
      //   LDST128BITS(C[store_gmem_c_addr_0]) = LDST128BITS(r_store_c_0[0]); 
      //   LDST128BITS(C[store_gmem_c_addr_1]) = LDST128BITS(r_store_c_1[0]); 
      // } 
    }
  }
}

// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle, dsmem
template<const int MMA_M=16, 
         const int MMA_N=8, 
         const int MMA_K=16,
         const int MMA_TILE_M=2,
         const int MMA_TILE_N=4,
         const int WARP_TILE_M=4,
         const int WARP_TILE_N=4,
         const int A_PAD=0, 
         const int B_PAD=0,
         const int K_STAGE=2, 
         const bool BLOCK_SWIZZLE=false,
         const bool COLLECTIVE_STORE=false>
__global__ void  __launch_bounds__(256) 
hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel(
  half* A, half* B, half* C, int M, int N, int K) {
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  // COLLECTIVE_STORE true/false control use stmatrix or not.
  const int bx = ((int) BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
  constexpr int BK = MMA_K; // 16

  extern __shared__ half smem[]; 
  half* s_a = smem;
  half* s_b = smem + K_STAGE * BM * (BK + A_PAD);
  constexpr int s_a_stage_offset = BM * (BK + A_PAD);
  constexpr int s_b_stage_offset = BK * (BN + B_PAD);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2; // 0,1
  const int warp_n = warp_id / 2; // 0,1,2,3

  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  int load_smem_b_k = tid / 16; // row 0~15
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      RC[i][j][0] = 0;
      RC[i][j][1] = 0;
    }
  }

  // may avoid cvta overhead ? only cvta smem base ptr once for cp.async.
  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  #pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (k * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (k * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE-2); // s2->0, s3->1, s4->2
  __syncthreads(); 

  #pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    // gmem -> smem
    // s2/4 can use bitwise ops but s3 can not, so, we use mod
    // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
    // s3: (k + 1) % 3
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    CP_ASYNC_COMMIT_GROUP();
    
    // ldmatrix for s_a, ldmatrix.trans for s_b.
    uint32_t RA[WARP_TILE_M][4];
    uint32_t RB[WARP_TILE_N][2];

    // smem -> reg
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + (smem_sel * s_a_stage_offset + 
                           lane_smem_a_m * (BK + A_PAD) + 
                           lane_smem_a_k) * sizeof(half)
      );
      LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;  // 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + (smem_sel * s_b_stage_offset + 
                           lane_smem_b_k * (BN + B_PAD) + 
                           lane_smem_b_n) * sizeof(half)
      );
      LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
    }
    
    // MMA compute
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        HMMA16816(RC[i][j][0], RC[i][j][1], 
                  RA[i][0], RA[i][1], RA[i][2], RA[i][3], 
                  RB[j][0], RB[j][1], 
                  RC[i][j][0], RC[i][j][1]);
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE-2);
    __syncthreads(); 
  }

  // make sure all memory issues ready.
  if ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads(); 
  }

  // processing last (K_STAGE-1) k iters.
  {
    #pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      // ldmatrix for s_a, ldmatrix.trans for s_b.
      uint32_t RA[WARP_TILE_M][4];
      uint32_t RB[WARP_TILE_N][2];

      // smem -> reg
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
        int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
        uint32_t lane_smem_a_ptr = (
          smem_a_base_ptr + (stage_sel * s_a_stage_offset + 
                             lane_smem_a_m * (BK + A_PAD) + 
                             lane_smem_a_k) * sizeof(half)
        );
        LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int lane_smem_b_k = lane_id % 16;  // 0~15
        int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
        uint32_t lane_smem_b_ptr = (
          smem_b_base_ptr + (stage_sel * s_b_stage_offset + 
                             lane_smem_b_k * (BN + B_PAD) + 
                             lane_smem_b_n) * sizeof(half)
        );
        LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
      }

      // MMA compute
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          HMMA16816(RC[i][j][0], RC[i][j][1], 
                    RA[i][0], RA[i][1], RA[i][2], RA[i][3], 
                    RB[j][0], RB[j][1], 
                    RC[i][j][0], RC[i][j][1]);
        }
      }
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 90)
  if (COLLECTIVE_STORE) {
    // reg -> smem(stmatrix) -> gmem(cp.async.bulk), MMA_MxMMA_N=16x8
    // NOTE: need [MMA_M][MMA_N] per warp to avoid overlap between warps.
    __shared__ half s_c[MMA_TILE_M][MMA_TILE_N][MMA_M][MMA_N]; // (2*4)*16*8*2=2KB
    uint32_t smem_c_base_ptr = __cvta_generic_to_shared(&s_c[warp_m][warp_n]);
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // store (i,j) warp tile -> smem c, 16x8
        uint32_t lane_smem_c_ptr = (
          smem_c_base_ptr + (lane_id % 16) * MMA_N * sizeof(half)); // (0~15)*8
        STMATRIX_X2(lane_smem_c_ptr, RC[i][j][0], RC[i][j][1]);
        // smem -> gmem, may use cp.async.bulk.global.share::cta?
        int store_warp_gmem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int store_warp_gmem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int store_lane_gmem_c_m = by * BM + store_warp_gmem_c_m;
        int store_lane_gmem_c_n = bx * BN + store_warp_gmem_c_n;
        // send 16 memory issues with 128 bits within lower half lanes.
        // TODO: use cp.async.bulk and wait outside the inner loop.
        if (lane_id < 16) {
          int store_gmem_c_addr = (store_lane_gmem_c_m + lane_id) * N + store_lane_gmem_c_n;
          LDST128BITS(C[store_gmem_c_addr]) = LDST128BITS(
            s_c[warp_m][warp_n][lane_id][0]);
        }
        __syncwarp();
      }
    }
  } else {
    // reg -> gmem, MMA_MxMMA_N=16x8
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
        int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n + (lane_id % 4) * 2;
        int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
        int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
        LDST32BITS(C[store_gmem_c_addr_0]) = LDST32BITS(RC[i][j][0]); 
        LDST32BITS(C[store_gmem_c_addr_1]) = LDST32BITS(RC[i][j][1]); 
      }
    }
  }
#else 
#warning "for sm<90, can not use stmatrix, force disable collective store!"
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      // mapping lane smem index -> global index.
      // [16][8], https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
      // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
      // [0~7][0~3 u32 -> 0~7 f16], [8~15][0~3 u32 -> 0~7 f16]
      int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
      int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n + (lane_id % 4) * 2;
      int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
      int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
      LDST32BITS(C[store_gmem_c_addr_0]) = LDST32BITS(RC[i][j][0]); 
      LDST32BITS(C[store_gmem_c_addr_1]) = LDST32BITS(RC[i][j][1]); 
      // TODO: How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
      // thus, we only need 8 memory issues with 128 bits after shfl_sync.
      // uint32_t r_store_c_0[4], r_store_c_1[4];
      // r_store_c_0[0] = RC[i][j][0]; r_store_c_1[0] = RC[i][j][1];
      // r_store_c_0[1] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
      // r_store_c_0[2] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
      // r_store_c_0[3] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
      // r_store_c_1[1] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
      // r_store_c_1[2] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
      // r_store_c_1[3] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
      // if (lane_id % 4 == 0) {
      //   LDST128BITS(C[store_gmem_c_addr_0]) = LDST128BITS(r_store_c_0[0]); 
      //   LDST128BITS(C[store_gmem_c_addr_1]) = LDST128BITS(r_store_c_1[0]); 
      // } 
    }
  }
#endif
}

// K32, 将K维度按照stage维度折半进行保存stages=3, warp_tile_k=2, [3*2][BM][16], 减少bank conflicts.
// 128x128, mma2x4, warp4x4(64,32,32), stages, block swizzle, dsmem, k32 with reg double buffers
template<const int MMA_M=16, 
         const int MMA_N=8, 
         const int MMA_K=16,
         const int MMA_TILE_M=2,
         const int MMA_TILE_N=4,
         const int WARP_TILE_M=4,
         const int WARP_TILE_N=4,
         const int WARP_TILE_K=2,
         const int A_PAD=0, 
         const int B_PAD=0,
         const int K_STAGE=2, 
         const bool BLOCK_SWIZZLE=false>
__global__ void  __launch_bounds__(256) 
hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel(
  half* A, half* B, half* C, int M, int N, int K) {
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int) BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K * WARP_TILE_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
  constexpr int BK = MMA_K; // 16x2=32

  extern __shared__ half smem[]; 
  half* s_a = smem;
  half* s_b = smem + K_STAGE * BM * (BK + A_PAD) * WARP_TILE_K;
  constexpr int s_a_stage_offset = BM * (BK + A_PAD); // 128x16 
  constexpr int s_b_stage_offset = BK * (BN + B_PAD); // 16x128
  constexpr int s_a_mma_k_store_offset = K_STAGE * BM * (BK + A_PAD);
  constexpr int s_b_mma_k_store_offset = K_STAGE * BK * (BN + B_PAD);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2; // 0,1
  const int warp_n = warp_id / 2; // 0,1,2,3

  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  int load_smem_b_k = tid / 16; // row 0~15
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,16,...
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      RC[i][j][0] = 0;
      RC[i][j][1] = 0;
    }
  }

  // may avoid cvta overhead ? only cvta smem base ptr once for cp.async.
  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  #pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * BK * WARP_TILE_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK * WARP_TILE_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (k * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16); // MMA_K 0
    uint32_t load_smem_a_mma_k_ptr = (
      smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + 
      (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + 
      load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_mma_k_ptr, &A[load_gmem_a_addr + 16], 16); // MMA_K 1

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (k * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    
    int load_gmem_b_k_mma_k = k * BK * WARP_TILE_K + MMA_K + load_smem_b_k;
    int load_gmem_b_addr_mma_k = load_gmem_b_k_mma_k * N + load_gmem_b_n; 
    uint32_t load_smem_b_mma_k_ptr = (
      smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
      (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + 
      load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_mma_k_ptr, &B[load_gmem_b_addr_mma_k], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE-2); // s2->0, s3->1, s4->2
  __syncthreads(); 

  uint32_t RA[2][WARP_TILE_M][4];
  uint32_t RB[2][WARP_TILE_N][2];

  int reg_store_idx = 0;
  int reg_load_idx = 1;

  { 
    // ldmatrix for s_a, ldmatrix.trans for s_b.
    // smem -> reg buffers 0, first MMA_K, 0~15
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + 
        (0 * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + 
        lane_smem_a_k) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;  // 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + 
        (0 * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) + 
        lane_smem_b_n) * sizeof(half)
      );
      // TODO: may use .x4.trans to load 4 matrix for reg double buffers at once?
      LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                    lane_smem_b_ptr);
      // int lane_smem_b_k = lane_id % 16;  // 0~15, 0~15
      // int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      // uint32_t lane_smem_b_ptr = (
      //   smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) * (lane_id / 16) +
      //   (0 * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) + 
      //   lane_smem_b_n) * sizeof(half)
      // );
      // // TRICK: I use .x4.trans to load 4 matrix for reg double buffers at once.
      // LDMATRIX_X4_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
      //               RB[reg_load_idx][j][0],  RB[reg_load_idx][j][1],
      //               lane_smem_b_ptr);
    }
  }
  
  #pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    reg_store_idx ^= 1; // 0->1
    reg_load_idx ^= 1; // 1->0
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    // stage gmem -> smem
    int load_gmem_a_k = k * BK * WARP_TILE_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK * WARP_TILE_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16); // MMA_K 0
    uint32_t load_smem_a_mma_k_ptr = (
      smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + 
      (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + 
      load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_mma_k_ptr, &A[load_gmem_a_addr + 16], 16); // MMA_K 1

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    
    int load_gmem_b_k_mma_k = k * BK * WARP_TILE_K + MMA_K + load_smem_b_k;
    int load_gmem_b_addr_mma_k = load_gmem_b_k_mma_k * N + load_gmem_b_n; 
    uint32_t load_smem_b_mma_k_ptr = (
      smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
      (smem_sel_next * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + 
      load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_mma_k_ptr, &B[load_gmem_b_addr_mma_k], 16);
    CP_ASYNC_COMMIT_GROUP();
    
    // ldmatrix for s_a, ldmatrix.trans for s_b.
    // smem -> reg buffers 1, second MMA_K, 16~31
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + 
        (smem_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + 
        lane_smem_a_k) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16; // 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
        (smem_sel * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) + 
        lane_smem_b_n) * sizeof(half)
      );
      // TODO: may use .x4.trans to load 4 matrix for reg double buffers at once?
      LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                    lane_smem_b_ptr);
    }
    
    // MMA compute, first MMA_K
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // Warp swizzle: Right -> Left -> Right -> Left
        size_t j_s = (i % 2) ? (WARP_TILE_N - j - 1) : j;
        HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                  RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                  RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                  RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                  RC[i][j_s][0], RC[i][j_s][1]);
      }
    }
    
    reg_store_idx ^= 1; // 1 -> 0
    reg_load_idx ^= 1; // 0 -> 1
    // MMA compute, second MMA_K 
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // Warp swizzle: Right -> Left -> Right -> Left
        size_t j_s = (i % 2) ? (WARP_TILE_N - j - 1) : j;
        HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                  RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                  RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                  RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                  RC[i][j_s][0], RC[i][j_s][1]);
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE-2);
    __syncthreads(); 

    // load next k iters to reg buffers.
    // smem -> reg buffers 0, first MMA_K, 0~15
    // int smem_sel_reg = (k + 2) % K_STAGE; // vs smem_sel k=2->(0)1, k=3->(1)2
    int smem_sel_reg = (smem_sel + 1) % K_STAGE; // vs smem_sel k=2->(0)1, k=3->(1)2
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + (smem_sel_reg * s_a_stage_offset + 
                           lane_smem_a_m * (BK + A_PAD) + 
                           lane_smem_a_k) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;  // 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + (smem_sel_reg * s_b_stage_offset + 
                           lane_smem_b_k * (BN + B_PAD) + 
                           lane_smem_b_n) * sizeof(half)
      );
      // TODO: may use .x4.trans to load 4 matrix for reg double buffers at once?
      LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                    lane_smem_b_ptr);
      // int lane_smem_b_k = lane_id % 16;  // 0~15, 0~15
      // int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      // uint32_t lane_smem_b_ptr = (
      //   smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) * (lane_id / 16) +
      //   (smem_sel_reg * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) + 
      //   lane_smem_b_n) * sizeof(half)
      // );
      // // may use .x4.trans to load 4 matrix for reg double buffers at once?
      // LDMATRIX_X4_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
      //               RB[reg_load_idx][j][0],  RB[reg_load_idx][j][1],
      //               lane_smem_b_ptr);
    }
  }

  // make sure all memory issues ready.
  if ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads(); 
  }

  // processing last (K_STAGE-1) k iters.
  {
    #pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      reg_store_idx ^= 1; // 0->1
      reg_load_idx ^= 1; // 1->0

      int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      // ldmatrix for s_a, ldmatrix.trans for s_b.
      // smem -> reg buffers 1, second MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
        int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
        uint32_t lane_smem_a_ptr = (
          smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) +
          (stage_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + 
          lane_smem_a_k) * sizeof(half)
        );
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                    lane_smem_a_ptr);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int lane_smem_b_k = lane_id % 16; // 0~15
        int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
        uint32_t lane_smem_b_ptr = (
          smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
          (stage_sel * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) + 
          lane_smem_b_n) * sizeof(half)
        );
        // TODO: may use .x4.trans to load 4 matrix for reg double buffers at once?
        LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                      lane_smem_b_ptr);
      }

      // MMA compute, first MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          // Warp swizzle: Right -> Left -> Right -> Left
          size_t j_s = (i % 2) ? (WARP_TILE_N - j - 1) : j;
          HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                    RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                    RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                    RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                    RC[i][j_s][0], RC[i][j_s][1]);
        }
      }

      reg_store_idx ^= 1; // 1 -> 0
      reg_load_idx ^= 1; // 0 -> 1

      // MMA compute, second MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          // Warp swizzle: Right -> Left -> Right -> Left
          size_t j_s = (i % 2) ? (WARP_TILE_N - j - 1) : j;
          HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                    RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                    RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                    RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                    RC[i][j_s][0], RC[i][j_s][1]);
        }
      }
      
      // load next k iters to reg buffers.
      // smem -> reg buffers 0, first MMA_K, 0~15
      // int stage_sel_reg = ((NUM_K_TILES - K_STAGE + k) % K_STAGE); 
      int stage_sel_reg = (stage_sel + 1) % K_STAGE; 
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
        int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
        uint32_t lane_smem_a_ptr = (
          smem_a_base_ptr + (stage_sel_reg * s_a_stage_offset + 
                             lane_smem_a_m * (BK + A_PAD) + 
                             lane_smem_a_k) * sizeof(half)
        );
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                    lane_smem_a_ptr);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int lane_smem_b_k = lane_id % 16;  // 0~15
        int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
        uint32_t lane_smem_b_ptr = (
          smem_b_base_ptr + (stage_sel_reg * s_b_stage_offset + 
                             lane_smem_b_k * (BN + B_PAD) + 
                             lane_smem_b_n) * sizeof(half)
        );
        LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                      lane_smem_b_ptr);
        // int lane_smem_b_k = lane_id % 16;  // 0~15, 0~15
        // int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
        // uint32_t lane_smem_b_ptr = (
        //   smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) * (lane_id / 16) +
        //   (stage_sel_reg * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) + 
        //   lane_smem_b_n) * sizeof(half)
        // );
        // // may use .x4.trans to load 4 matrix for reg double buffers at once?
        // LDMATRIX_X4_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
        //               RB[reg_load_idx][j][0],  RB[reg_load_idx][j][1],
        //               lane_smem_b_ptr);
      }
    }
  }

  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      // mapping lane smem index -> global index.
      // [16][8], https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
      // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
      // [0~7][0~3 u32 -> 0~7 f16], [8~15][0~3 u32 -> 0~7 f16]
      int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
      int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n + (lane_id % 4) * 2;
      int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
      int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
      // LDST32BITS(C[store_gmem_c_addr_0]) = LDST32BITS(RC[i][j][0]); 
      // LDST32BITS(C[store_gmem_c_addr_1]) = LDST32BITS(RC[i][j][1]); 
      // TODO: How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
      // thus, we only need 8 memory issues with 128 bits after shfl_sync.
      uint32_t r_store_c_0[4], r_store_c_1[4];
      r_store_c_0[0] = RC[i][j][0]; r_store_c_1[0] = RC[i][j][1];
      r_store_c_0[1] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
      r_store_c_0[2] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
      r_store_c_0[3] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
      r_store_c_1[1] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
      r_store_c_1[2] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
      r_store_c_1[3] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
      if (lane_id % 4 == 0) {
        LDST128BITS(C[store_gmem_c_addr_0]) = LDST128BITS(r_store_c_0[0]); 
        LDST128BITS(C[store_gmem_c_addr_1]) = LDST128BITS(r_store_c_1[0]); 
      } 
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

// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle
#define LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_KERNEL(stages, stride)    \
{                                                                           \
  const int N_SWIZZLE = (N + (stride) - 1) / (stride);                      \
  dim3 block(NUM_THREADS);                                                  \
  dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE,                  \
             div_ceil(M, BM),                                               \
             N_SWIZZLE);                                                    \
  hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_kernel<                          \
    MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                            \
    WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD,                                 \
    (stages), true><<<grid, block>>>(                                       \
    reinterpret_cast<half*>(a.data_ptr()),                                  \
    reinterpret_cast<half*>(b.data_ptr()),                                  \
    reinterpret_cast<half*>(c.data_ptr()),                                  \
    M, N, K                                                                 \
  );                                                                        \
}

#define LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_KERNEL(stages)         \
{                                                                           \
  dim3 block(NUM_THREADS);                                                  \
  dim3 grid(div_ceil(N, BN), div_ceil(M, BM));                              \
  hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_kernel<                          \
    MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                            \
    WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD,                                 \
    (stages), false><<<grid, block>>>(                                      \
    reinterpret_cast<half*>(a.data_ptr()),                                  \
    reinterpret_cast<half*>(b.data_ptr()),                                  \
    reinterpret_cast<half*>(c.data_ptr()),                                  \
    M, N, K                                                                 \
  );                                                                        \
}

// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle
void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages(
  torch::Tensor a, torch::Tensor b, torch::Tensor c, 
  int stages, bool swizzle, int swizzle_stride) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  constexpr int MMA_TILE_M = 2;
  constexpr int MMA_TILE_N = 4; 
  constexpr int WARP_TILE_M = 4;
  constexpr int WARP_TILE_N = 4;
  // s_a 4  ways bank conflicts within warp, after pad 8  -> 4 ways bank conflicts.
  // s_b 16 ways bank conflicts within warp, after pad 8  -> 8 ways bank conflicts.
  // s_b 16 ways bank conflicts within warp, after pad 16 -> 4 ways bank conflicts.
  // so, the best padding policy for s_a and s_b is A_PAD=0/8, B_PAD=16. Thus, 
  // improve B_PAD consume 8x~ less smem than A_PAD, 16xB_PAD vs 128xA_PAD.
  constexpr int A_PAD = 0;  // 0,8,16
  constexpr int B_PAD = 16; // 0,8,16
  constexpr int NUM_THREADS= (
    MMA_TILE_M * MMA_TILE_N * WARP_SIZE); // 2 * 4 * 32 = 256
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;    
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;    
  // constexpr int BK = MMA_K;   
  // s2: 2*128*(16)*2=8KB,  2*16*(128+16)*2=9KB,    ~17KB
  // s3: 3*128*(16)*2=12KB, 3*16*(128+16)*2=13.5KB, ~26KB
  // s4: 4*128*(16)*2=16KB, 4*16*(128+16)*2=18KB,   ~34KB                            
  // s5: 5*128*(16)*2=20KB, 5*16*(128+16)*2=22.5KB, ~43KB    
  if (swizzle) {
    assert(swizzle_stride % 256 == 0);
    switch (stages)
    {
    case 2: // ~17KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_KERNEL(2, swizzle_stride);
      break;
    case 3: // ~26KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_KERNEL(3, swizzle_stride);
      break;
    case 4: // ~34KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_KERNEL(4, swizzle_stride);
      break;
    case 5: // ~43KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_KERNEL(5, swizzle_stride);
      break;
    default:
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_KERNEL(2, swizzle_stride);
      break;
    }
  } else {
    switch (stages)
    {
    case 2:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_KERNEL(2);
      break;
    case 3:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_KERNEL(3);
      break;
    case 4:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_KERNEL(4);
      break;
    case 5:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_KERNEL(5);
      break;
    default:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_KERNEL(2);
      break;
    }
  }
}

// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle, dsmem
#define LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(stages, stride)   \
{                                                                                \
  const int smem_max_size = (                                                    \
    (stages) * BM * (BK + A_PAD) * sizeof(half) +                                \
    (stages) * BK * (BN + B_PAD) * sizeof(half));                                \
  cudaFuncSetAttribute(                                                          \
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel<                       \
      MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                               \
      WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, (stages), true>,                   \
    cudaFuncAttributeMaxDynamicSharedMemorySize,                                 \
    98304);                                                                      \
  const int N_SWIZZLE = (N + (stride) - 1) / (stride);                           \
  dim3 block(NUM_THREADS);                                                       \
  dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE,                       \
             div_ceil(M, BM),                                                    \
             N_SWIZZLE);                                                         \
  hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel<                         \
    MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                                 \
    WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, (stages), true><<<                   \
    grid, block, smem_max_size>>>(                                               \
    reinterpret_cast<half*>(a.data_ptr()),                                       \
    reinterpret_cast<half*>(b.data_ptr()),                                       \
    reinterpret_cast<half*>(c.data_ptr()),                                       \
    M, N, K                                                                      \
  );                                                                             \
}

#define LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(stages)     \
{                                                                             \
  const int smem_max_size = (                                                 \
    (stages) * BM * (BK + A_PAD) * sizeof(half) +                             \
    (stages) * BK * (BN + B_PAD) * sizeof(half));                             \
  cudaFuncSetAttribute(                                                       \
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel<                    \
      MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                            \
      WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, (stages), false>,               \
    cudaFuncAttributeMaxDynamicSharedMemorySize,                              \
    98304);                                                                   \
  dim3 block(NUM_THREADS);                                                    \
  dim3 grid(div_ceil(N, BN), div_ceil(M, BM));                                \
  hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel<                      \
    MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                              \
    WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, (stages), false><<<               \
    grid, block, smem_max_size>>>(                                            \
    reinterpret_cast<half*>(a.data_ptr()),                                    \
    reinterpret_cast<half*>(b.data_ptr()),                                    \
    reinterpret_cast<half*>(c.data_ptr()),                                    \
    M, N, K                                                                   \
  );                                                                          \
}

// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle, dsmem
void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem(
  torch::Tensor a, torch::Tensor b, torch::Tensor c, 
  int stages, bool swizzle, int swizzle_stride) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  constexpr int MMA_TILE_M = 2;
  constexpr int MMA_TILE_N = 4; 
  constexpr int WARP_TILE_M = 4;
  constexpr int WARP_TILE_N = 4;
  // s_a 4  ways bank conflicts within warp, after pad 8  -> 4 ways bank conflicts.
  // s_b 16 ways bank conflicts within warp, after pad 8  -> 8 ways bank conflicts.
  // s_b 16 ways bank conflicts within warp, after pad 16 -> 4 ways bank conflicts.
  // so, the best padding policy for s_a and s_b is A_PAD=0/8, B_PAD=16. Thus, 
  // improve B_PAD consume 8x~ less smem than A_PAD, 16xB_PAD vs 128xA_PAD.
  constexpr int A_PAD = 0;  // 0,8,16
  constexpr int B_PAD = 16; // 0,8,16
  constexpr int NUM_THREADS= (
    MMA_TILE_M * MMA_TILE_N * WARP_SIZE); // 2 * 4 * 32 = 256
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;    
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;    
  constexpr int BK = MMA_K;   
  // s2: 2*128*(16)*2=8KB,  2*16*(128+16)*2=9KB,    ~17KB
  // s3: 3*128*(16)*2=12KB, 3*16*(128+16)*2=13.5KB, ~26KB
  // s4: 4*128*(16)*2=16KB, 4*16*(128+16)*2=18KB,   ~34KB                            
  // s5: 5*128*(16)*2=20KB, 5*16*(128+16)*2=22.5KB, ~43KB    
  if (swizzle) {
    assert(swizzle_stride % 256 == 0);
    switch (stages)
    {
    case 2: // ~17KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(2, swizzle_stride);
      break;
    case 3: // ~26KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(3, swizzle_stride);
      break;
    case 4: // ~34KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(4, swizzle_stride);
      break;
    case 5: // ~43KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(5, swizzle_stride);
      break;
    default:
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(2, swizzle_stride);
      break;
    }
  } else {
    switch (stages)
    {
    case 2:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(2);
      break;
    case 3:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(3);
      break;
    case 4:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(4);
      break;
    case 5:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(5);
      break;
    default:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(2);
      break;
    }
  }
}


// 128x128, mma2x4, warp4x4x2(64,32,32), stages, block swizzle, dsmem, reg double buffers
#define LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(stages, stride) \
{                                                                                \
  const int smem_max_size = (                                                    \
    (stages) * BM * (BK + A_PAD) * WARP_TILE_K * sizeof(half) +                  \
    (stages) * BK * (BN + B_PAD) * WARP_TILE_K * sizeof(half));                  \
  cudaFuncSetAttribute(                                                          \
    hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel<                     \
      MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                               \
      WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, A_PAD, B_PAD, (stages), true>,      \
    cudaFuncAttributeMaxDynamicSharedMemorySize,                                 \
    98304);                                                                      \
  const int N_SWIZZLE = (N + (stride) - 1) / (stride);                           \
  dim3 block(NUM_THREADS);                                                       \
  dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE,                       \
             div_ceil(M, BM),                                                    \
             N_SWIZZLE);                                                         \
  hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel<                       \
    MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                                 \
    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, A_PAD, B_PAD, (stages), true><<<      \
    grid, block, smem_max_size>>>(                                               \
    reinterpret_cast<half*>(a.data_ptr()),                                       \
    reinterpret_cast<half*>(b.data_ptr()),                                       \
    reinterpret_cast<half*>(c.data_ptr()),                                       \
    M, N, K                                                                      \
  );                                                                             \
}

#define LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(stages)   \
{                                                                             \
  const int smem_max_size = (                                                 \
    (stages) * BM * (BK + A_PAD) * WARP_TILE_K * sizeof(half) +               \
    (stages) * BK * (BN + B_PAD) * WARP_TILE_K * sizeof(half));               \
  cudaFuncSetAttribute(                                                       \
    hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel<                  \
      MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                            \
      WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, A_PAD, B_PAD, (stages), false>,  \
    cudaFuncAttributeMaxDynamicSharedMemorySize,                              \
    98304);                                                                   \
  dim3 block(NUM_THREADS);                                                    \
  dim3 grid(div_ceil(N, BN), div_ceil(M, BM));                                \
  hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel<                    \
    MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                              \
    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, A_PAD, B_PAD, (stages), false><<<  \
    grid, block, smem_max_size>>>(                                            \
    reinterpret_cast<half*>(a.data_ptr()),                                    \
    reinterpret_cast<half*>(b.data_ptr()),                                    \
    reinterpret_cast<half*>(c.data_ptr()),                                    \
    M, N, K                                                                   \
  );                                                                          \
}

// 128x128, mma2x4, warp4x4x2(64,32,32), stages, block swizzle, dsmem, reg double buffers
void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem(
  torch::Tensor a, torch::Tensor b, torch::Tensor c, 
  int stages, bool swizzle, int swizzle_stride) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  constexpr int MMA_TILE_M = 2;
  constexpr int MMA_TILE_N = 4; 
  constexpr int WARP_TILE_M = 4;
  constexpr int WARP_TILE_N = 4;
  constexpr int WARP_TILE_K = 2;
  // s_a 4  ways bank conflicts within warp, after pad 8  -> 4 ways bank conflicts.
  // s_b 16 ways bank conflicts within warp, after pad 8  -> 8 ways bank conflicts.
  // s_b 16 ways bank conflicts within warp, after pad 16 -> 4 ways bank conflicts.
  // so, the best padding policy for s_a and s_b is A_PAD=0/8, B_PAD=16. Thus, 
  // improve B_PAD consume 8x~ less smem than A_PAD, 16xB_PAD vs 128xA_PAD.
  constexpr int A_PAD = 0;  // 0,8,16
  constexpr int B_PAD = 16; // 0,8,16
  constexpr int NUM_THREADS= (
    MMA_TILE_M * MMA_TILE_N * WARP_SIZE); // 2 * 4 * 32 = 256
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;    
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;    
  constexpr int BK = MMA_K;   
  // s2: 2*128*(32)*2=16KB, 2*32*(128+16)*2=18KB, ~35KB
  // s3: 3*128*(32)*2=24KB, 3*32*(128+16)*2=27KB, ~51KB
  // s4: 4*128*(32)*2=32KB, 4*32*(128+16)*2=36KB, ~68KB                            
  // s5: 5*128*(32)*2=40KB, 5*32*(128+16)*2=45KB, ~85KB    
  if (swizzle) {
    assert(swizzle_stride % 256 == 0);
    switch (stages)
    {
    case 2: // ~35KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(2, swizzle_stride);
      break;
    case 3: // ~51KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(3, swizzle_stride);
      break;
    case 4: // ~68KB  
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(4, swizzle_stride);
      break;
    case 5: // ~85KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(5, swizzle_stride);
      break;
    default:
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(2, swizzle_stride);
      break;
    }
  } else {
    switch (stages)
    {
    case 2:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(2);
      break;
    case 3:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(3);
      break;
    case 4:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(4);
      break;
    case 5:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(5);
      break;
    default:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(2);
      break;
    }
  }
}
