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

// i: row index; j: col index. 
// e.g kColStride = 16, kStep = 8 -> load 8 half as 128 bits memory issue.
template<const int kColStride = 16, const int kStep = 8>
static __device__ __forceinline__ int swizzle_permuted_j(int i, int j) {
  // for col_stride > 16, we have to permute it using col major ZigZag order.
  // e.g, A smem logical layout [Br,d]=[Br,64] -> store layout [4][Br][16].
  static_assert(kColStride <= 16, "kColStride must <= 16");
  // swizzle: ((int(j / kStep) ^ int(i / 4)) % int(kColStride / kStep)) * kStep;
  static_assert(kStep == 4 || kStep == 8, "kStep must be 8 or 4.");
  static_assert(kColStride % kStep == 0, "kColStride must be multiple of kStep.");
  if constexpr (kStep == 8) {
    return (((j >> 3) ^ (i >> 2)) % (kColStride >> 3)) << 3;
  } else {
    static_assert(kStep == 4);
    return (((j >> 2) ^ (i >> 2)) % (kColStride >> 2)) << 2;
  }
}

// i: row index; j: col index
template<const int kMmaAtomK = 16>
static __device__ __forceinline__ int swizzle_permuted_A_j(int i, int j) {
  // -------------------
  // -col 0~16, step 8--
  // -------------------
  // | row 0  | (0, 8) |
  // | row 1  | (0, 8) |
  // | row 2  | (0, 8) |
  // | row 3  | (0, 8) |
  // -------------------
  // | row 4  | (8, 0) |
  // | row 5  | (8, 0) |
  // | row 6  | (8, 0) |
  // | row 7  | (8, 0) |
  // -------------------
  // | row 8  | (0, 8) |
  // | row 9  | (0, 8) |
  // | row 10 | (0, 8) |
  // | row 11 | (0, 8) |
  // -------------------
  // | row 12 | (8, 0) |
  // | row 13 | (8, 0) |
  // | row 14 | (8, 0) |
  // | row 15 | (8, 0) |
  // -------------------
  return swizzle_permuted_j<kMmaAtomK, 8>(i, j);
}

// i: row index; j: col index
template<const int kMmaAtomK = 16>
static __device__ __forceinline__ int swizzle_permuted_B_j(int i, int j) {
  // -------------------
  // -col 0~16, step 8--
  // -------------------
  // | row 0  | (0, 8) |
  // | row 1  | (0, 8) |
  // | row 2  | (0, 8) |
  // | row 3  | (0, 8) |
  // -------------------
  // | row 4  | (8, 0) |
  // | row 5  | (8, 0) |
  // | row 6  | (8, 0) |
  // | row 7  | (8, 0) |
  // -------------------
  // | row 8  | (0, 8) |
  // | row 9  | (0, 8) |
  // | row 10 | (0, 8) |
  // | row 11 | (0, 8) |
  // -------------------
  // | row 12 | (8, 0) |
  // | row 13 | (8, 0) |
  // | row 14 | (8, 0) |
  // | row 15 | (8, 0) |
  // -------------------
  return swizzle_permuted_j<kMmaAtomK, 8>(i, j);
}

// NN: A/B/C All row major
// TN: A row major MxK, B col major NxK, C row major MxN
// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle, dsmem
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
         const bool BLOCK_SWIZZLE=true,
         const bool WARP_SWIZZLE=true>
__global__ void  __launch_bounds__(256) 
hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4_kernel(
  const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C, 
  int M, int N, int K) {
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
  constexpr int s_b_stage_offset = BN * (BK + B_PAD); // 128*16
  constexpr int s_a_mma_k_store_offset = K_STAGE * BM * (BK + A_PAD);
  constexpr int s_b_mma_k_store_offset = K_STAGE * BN * (BK + B_PAD);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2; // 0,1
  const int warp_n = warp_id / 2; // 0,1,2,3

  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  int load_smem_b_n = tid / 2; // row 0~127
  int load_smem_b_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

  uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      RC[i][j][0] = 0;
      RC[i][j][1] = 0;
    }
  }

  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  #pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * BK * WARP_TILE_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK * WARP_TILE_K + load_smem_b_k; // global col of b
    int load_gmem_b_addr = load_gmem_b_n * K + load_gmem_b_k; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (k * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         swizzle_permuted_A_j<MMA_K>(
                          load_smem_a_m, load_smem_a_k)) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16); // MMA_K 0
    uint32_t load_smem_a_mma_k_ptr = (
      smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + 
      (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + 
      swizzle_permuted_A_j<MMA_K>(load_smem_a_m, load_smem_a_k)) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_mma_k_ptr, &A[load_gmem_a_addr + 16], 16); // MMA_K 1

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (k * s_b_stage_offset + 
                         load_smem_b_n * (BK + B_PAD) + 
                         load_smem_b_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    uint32_t load_smem_b_mma_k_ptr = (
      smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
      (k * s_b_stage_offset + load_smem_b_n * (BK + B_PAD) + 
      load_smem_b_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_mma_k_ptr, &B[load_gmem_b_addr + 16], 16);

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
        swizzle_permuted_A_j<MMA_K>(lane_smem_a_m, lane_smem_a_k)) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_n = warp_smem_b_n + lane_id % 8; // 0~7, MMA_N=8
      int lane_smem_b_k = ((lane_id / 8) % 2) * 8; // 0,8
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + 
        (0 * s_b_stage_offset + lane_smem_b_n * (BK + B_PAD) + 
        lane_smem_b_k) * sizeof(half)
      );
      // may use .x4.trans to load 4 matrix for reg double buffers at once?
      LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                  lane_smem_b_ptr);
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
    int load_gmem_b_k = k * BK * WARP_TILE_K + load_smem_b_k; // global col of b
    int load_gmem_b_addr = load_gmem_b_n * K + load_gmem_b_k; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         swizzle_permuted_A_j<MMA_K>(
                          load_smem_a_m, load_smem_a_k)) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16); // MMA_K 0
    uint32_t load_smem_a_mma_k_ptr = (
      smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + 
      (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + 
      swizzle_permuted_A_j<MMA_K>(load_smem_a_m, load_smem_a_k)) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_mma_k_ptr, &A[load_gmem_a_addr + 16], 16); // MMA_K 1

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + 
                         load_smem_b_n * (BK + B_PAD) + 
                         load_smem_b_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    
    uint32_t load_smem_b_mma_k_ptr = (
      smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
      (smem_sel_next * s_b_stage_offset + load_smem_b_n * (BK + B_PAD) + 
      load_smem_b_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_mma_k_ptr, &B[load_gmem_b_addr + 16], 16);
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
        swizzle_permuted_A_j<MMA_K>(lane_smem_a_m, lane_smem_a_k)) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_n = warp_smem_b_n + lane_id % 8; // 0~7, MMA_N=8
      int lane_smem_b_k = ((lane_id / 8) % 2) * 8; // 0,8
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
        (smem_sel * s_b_stage_offset + lane_smem_b_n * (BK + B_PAD) + 
        lane_smem_b_k) * sizeof(half)
      );
      // TODO: use .x4.trans to load 4 matrix for reg double buffers at once?
      LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                  lane_smem_b_ptr);
    }
    
    // MMA compute, first MMA_K
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // Warp swizzle: Right -> Left -> Right -> Left
        int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
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
        int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
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
                           swizzle_permuted_A_j<MMA_K>(
                            lane_smem_a_m, lane_smem_a_k)) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_n = warp_smem_b_n + lane_id % 8; // 0~7, MMA_N=8
      int lane_smem_b_k = ((lane_id / 8) % 2) * 8; // 0,8
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + 
        (smem_sel_reg * s_b_stage_offset + lane_smem_b_n * (BK + B_PAD) + 
        lane_smem_b_k) * sizeof(half)
      );
      LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                  lane_smem_b_ptr);
    }
  }

  // make sure all memory issues ready.
  if constexpr ((K_STAGE - 2) > 0) {
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
          swizzle_permuted_A_j<MMA_K>(lane_smem_a_m, lane_smem_a_k)) * sizeof(half)
        );
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                    lane_smem_a_ptr);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int lane_smem_b_n = warp_smem_b_n + lane_id % 8; // 0~7, MMA_N=8
        int lane_smem_b_k = ((lane_id / 8) % 2) * 8; // 0,8
        uint32_t lane_smem_b_ptr = (
          smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
          (stage_sel * s_b_stage_offset + lane_smem_b_n * (BK + B_PAD) + 
          lane_smem_b_k) * sizeof(half)
        );
        // may use .x4.trans to load 4 matrix for reg double buffers at once?
        LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                    lane_smem_b_ptr);
      }

      // MMA compute, first MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          // Warp swizzle: Right -> Left -> Right -> Left
          int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
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
          int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
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
                             swizzle_permuted_A_j<MMA_K>(
                              lane_smem_a_m, lane_smem_a_k)) * sizeof(half)
        );
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                    lane_smem_a_ptr);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int lane_smem_b_n = warp_smem_b_n + lane_id % 8; // 0~7, MMA_N=8
        int lane_smem_b_k = ((lane_id / 8) % 2) * 8; // 0,8
        uint32_t lane_smem_b_ptr = (
          smem_b_base_ptr + 
          (stage_sel_reg * s_b_stage_offset + lane_smem_b_n * (BK + B_PAD) + 
          lane_smem_b_k) * sizeof(half)
        );
        // may use .x4.trans to load 4 matrix for reg double buffers at once?
        LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                    lane_smem_b_ptr);
      }
    }
  }

  // collective store with reg reuse & warp shuffle
  for (int i = 0; i < WARP_TILE_M; ++i) {
    // reuse RA[2][4][4] reg here, this may boost 0.3~0.5 TFLOPS up.
    // may not put 'if' in N loop, it will crash the 'pragma unroll' hint ?
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
      // thus, we only need 8 memory issues with 128 bits after shfl_sync.
      RA[0][j][0] = RC[i][j][0]; 
      RA[1][j][0] = RC[i][j][1];
      RA[0][j][1] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
      RA[0][j][2] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
      RA[0][j][3] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
      RA[1][j][1] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
      RA[1][j][2] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
      RA[1][j][3] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
    }

    if (lane_id % 4 == 0) {
      int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n;
        int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
        int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
        LDST128BITS(C[store_gmem_c_addr_0]) = LDST128BITS(RA[0][j][0]); 
        LDST128BITS(C[store_gmem_c_addr_1]) = LDST128BITS(RA[1][j][0]); 
      }
    }
  }
}

// build cpp binary
#ifndef NO_MMA_HGEMM_BIN

#include "utils.h"

// 128x128, mma2x4, warp4x4(64,32,16), stages, block&smem swizzle, dsmem, TN
#define LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_TN_X4_KERNEL(stages, stride) \
{                                                                                      \
  const int smem_max_size = (                                                          \
    (stages) * BM * (BK + A_PAD) * WARP_TILE_K * sizeof(half) +                        \
    (stages) * BN * (BK + B_PAD) * WARP_TILE_K * sizeof(half));                        \
  cudaFuncSetAttribute(                                                                \
    hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4_kernel<             \
      MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                                     \
      WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, A_PAD, B_PAD, (stages), true>,            \
    cudaFuncAttributeMaxDynamicSharedMemorySize,                                       \
    98304);                                                                            \
  const int N_SWIZZLE = (N + (stride) - 1) / (stride);                                 \
  dim3 block(NUM_THREADS);                                                             \
  dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE,                             \
             div_ceil(M, BM),                                                          \
             N_SWIZZLE);                                                               \
  hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4_kernel<               \
    MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                                       \
    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, A_PAD, B_PAD, (stages), true><<<            \
    grid, block, smem_max_size>>>(                                                     \
    a, b, c,                                                                           \
    M, N, K                                                                            \
  );                                                                                   \
}

template <const int K_STAGE = 2, 
          const int BLOCK_SWIZZLE_STRIDE = 2048>
void lanunch_hgemm_mma_m16n8k16_swizzle_tn_x4(
  half* a, half* b, half* c, int M, int N, int K) {
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  constexpr int MMA_TILE_M = 2;
  constexpr int MMA_TILE_N = 4; 
  constexpr int WARP_TILE_M = 4;
  constexpr int WARP_TILE_N = 4;
  constexpr int WARP_TILE_K = 2;
  constexpr int A_PAD = 0; // apply smem swizzle
  constexpr int B_PAD = 8;
  constexpr int NUM_THREADS= (
    MMA_TILE_M * MMA_TILE_N * WARP_SIZE); // 2 * 4 * 32 = 256
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;    
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;    
  constexpr int BK = MMA_K;   
  LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_TN_X4_KERNEL(
    K_STAGE, BLOCK_SWIZZLE_STRIDE);
}

#ifdef HGEMM_MMA_DEBUG  
#include <iostream>
#endif

int main(int argc, char *argv[]) {
  // ncu --metrics sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm ./hgemm_mma_stage_tn_swizzle.80.debug.bin 1024 1024 1024 0 1
#ifdef HGEMM_MMA_DEBUG  
  const int test_num = 1;
#else
  const int test_num = 64;
#endif
  int M_list[test_num];
  int N_list[test_num];
  int K_list[test_num];
  
  for (int i = 0; i < test_num; i++) {
    M_list[i] = (i + 1) * 256;
    N_list[i] = (i + 1) * 256;
    K_list[i] = (i + 1) * 256;
  }
  
#ifdef HGEMM_MMA_DEBUG  
  if (argc > 1) M_list[0] = std::stoi(argv[1]);
  if (argc > 2) N_list[0] = std::stoi(argv[2]);
  if (argc > 3) K_list[0] = std::stoi(argv[3]);
#endif
  
#ifdef HGEMM_MMA_DEBUG  
  int outer_repeat = 1, inner_repeat = 1, warmup = 1;
  if (argc > 4) warmup = std::stoi(argv[4]);
  if (argc > 5) inner_repeat = std::stoi(argv[5]);
#else
  int outer_repeat = 10, inner_repeat = 1, warmup = 1;
#endif
  
  printf("ALGO = MMA16816 HGEMM TN MMA=2x4 WARP=4x4x2 STAGES=2, BLOCK SWIZZLE=2048, A SMEM SWIZZLE, B Pad = 8\n");
#ifndef HGEMM_MMA_DEBUG  
  for (int j = 0; j < 5; j++) {
    int M = M_list[j], N = N_list[j], K = K_list[j];
    float max_error = gemm_error_check_tn<half>(
      lanunch_hgemm_mma_m16n8k16_swizzle_tn_x4<2, 2048>, 
      M, N, K);
    printf("M N K = %6d %6d %6d, ", M, N, K);
    printf("Max Error = %f\n", max_error);
  }
#endif
  
  for (int j = 0; j < test_num; j++) {
    int M = M_list[j], N = N_list[j], K = K_list[j];
   
    double max_sec = 0.0;
    double min_sec = DBL_MAX;
    double total_sec = 0.0;
  
    for (int k = 0; k < outer_repeat; k++) {
      double this_sec = perf_gemm<half>(
        lanunch_hgemm_mma_m16n8k16_swizzle_tn_x4<2, 2048>, 
        M, N, K, inner_repeat, warmup);
      max_sec = max(max_sec, this_sec);
      min_sec = min(min_sec, this_sec);
      total_sec += this_sec;
    }
      
    // 1 TFLOPS = 10^12 FLOPS
    // ref: https://imgtec.eetrend.com/blog/2021/100062210.html.
    double avg_sec = total_sec / outer_repeat;
    double avg_Tflops = ((double)M) * N * K * 2 * 1e-12 / avg_sec;
  
    printf("M N K = %6d %6d %6d, W = %1d, R = %2d ", M, N, K, warmup, inner_repeat);
    printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
    printf("AVG Performance = %10.4lf Tflops\n", avg_Tflops);
  }
  
  return 0;
}

#else
// --------------------- PyTorch bindings for custom kernel -----------------------
#include <torch/types.h>
#include <torch/extension.h>
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

// 128x128, mma2x4, warp4x4(64,32,16), stages, block&smem swizzle, dsmem, TN
#define LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_TN_X4_KERNEL(stages, stride)\
{                                                                                     \
  const int smem_max_size = (                                                         \
    (stages) * BM * (BK + A_PAD) * WARP_TILE_K * sizeof(half) +                       \
    (stages) * BN * (BK + B_PAD) * WARP_TILE_K * sizeof(half));                       \
  cudaFuncSetAttribute(                                                               \
    hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4_kernel<            \
      MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                                    \
      WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, A_PAD, B_PAD, (stages), true>,           \
    cudaFuncAttributeMaxDynamicSharedMemorySize,                                      \
  98304);                                                                             \
  const int N_SWIZZLE = (N + (stride) - 1) / (stride);                                \
  dim3 block(NUM_THREADS);                                                            \
  dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE,                            \
             div_ceil(M, BM),                                                         \
             N_SWIZZLE);                                                              \
  hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4_kernel<              \
    MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                                      \
    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, A_PAD, B_PAD, (stages), true><<<           \
    grid, block, smem_max_size>>>(                                                    \
    reinterpret_cast<half*>(a.data_ptr()),                                            \
    reinterpret_cast<half*>(b.data_ptr()),                                            \
    reinterpret_cast<half*>(c.data_ptr()),                                            \
    M, N, K                                                                           \
  );                                                                                  \
}

#define LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_TN_X4_KERNEL(stages)\
{                                                                                \
  const int smem_max_size = (                                                    \
    (stages) * BM * (BK + A_PAD) * WARP_TILE_K * sizeof(half) +                  \
    (stages) * BN * (BK + B_PAD) * WARP_TILE_K * sizeof(half));                  \
  cudaFuncSetAttribute(                                                          \
    hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4_kernel<       \
      MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                               \
      WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, A_PAD, B_PAD, (stages), true>,      \
    cudaFuncAttributeMaxDynamicSharedMemorySize,                                 \
  98304);                                                                        \
  dim3 block(NUM_THREADS);                                                       \
  dim3 grid(div_ceil(N, BN), div_ceil(M, BM));                                   \
  hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4_kernel<         \
    MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                                 \
    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, A_PAD, B_PAD, (stages), false><<<     \
    grid, block, smem_max_size>>>(                                               \
    reinterpret_cast<half*>(a.data_ptr()),                                       \
    reinterpret_cast<half*>(b.data_ptr()),                                       \
    reinterpret_cast<half*>(c.data_ptr()),                                       \
    M, N, K                                                                      \
  );                                                                             \
}

// 128x128, mma2x4, warp4x4(64,32,16), stages, block&smem swizzle, dsmem
void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4(
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
  constexpr int A_PAD = 0; // 0,8,16
  constexpr int B_PAD = 8; // 0,8,16
  constexpr int NUM_THREADS= (
    MMA_TILE_M * MMA_TILE_N * WARP_SIZE); // 2 * 4 * 32 = 256
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;    
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;    
  constexpr int BK = MMA_K;   
    
  if (swizzle) {
    // assert(swizzle_stride % 256 == 0);
    switch (stages)
    {
    case 2: 
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_TN_X4_KERNEL(2, swizzle_stride);
      break;
    case 3: 
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_TN_X4_KERNEL(3, swizzle_stride);
      break;
    case 4: 
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_TN_X4_KERNEL(4, swizzle_stride);
      break;
    case 5: 
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_TN_X4_KERNEL(5, swizzle_stride);
      break;
    default:
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_TN_X4_KERNEL(2, swizzle_stride);
      break;
    }
  } else {
    switch (stages)
    {
    case 2:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_TN_X4_KERNEL(2);
      break;
    case 3:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_TN_X4_KERNEL(3);
      break;
    case 4:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_TN_X4_KERNEL(4);
      break;
    case 5:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_TN_X4_KERNEL(5);
      break;
    default:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_TN_X4_KERNEL(2);
      break;
    }
  }
}

#endif