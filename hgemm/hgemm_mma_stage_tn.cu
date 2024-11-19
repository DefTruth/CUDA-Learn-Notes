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
         const int A_PAD=0, 
         const int B_PAD=0,
         const int K_STAGE=2, 
         const bool BLOCK_SWIZZLE=false>
__global__ void  __launch_bounds__(256) 
hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel(
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
  constexpr int s_a_stage_offset = BM * (BK + A_PAD); // BMxBK 128*16
  constexpr int s_b_stage_offset = BN * (BK + B_PAD); // BNxBK 128*16

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2; // 0,1
  const int warp_n = warp_id / 2; // 0,1,2,3

  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  int load_smem_b_n = tid / 2; // row 0~127
  int load_smem_b_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of c

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
    int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK + load_smem_b_k; // global col of b
    int load_gmem_b_addr = load_gmem_b_n * K + load_gmem_b_k; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (k * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (k * s_b_stage_offset + 
                         load_smem_b_n * (BK + B_PAD) + 
                         load_smem_b_k) * sizeof(half)
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
    int load_gmem_b_k = k * BK + load_smem_b_k; // global col of b
    int load_gmem_b_addr = load_gmem_b_n * K + load_gmem_b_k; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + 
                         load_smem_b_n * (BK + B_PAD) + 
                         load_smem_b_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
    
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
      int lane_smem_b_n = warp_smem_b_n + lane_id % 8; // 0~7, MMA_N=8
      int lane_smem_b_k = ((lane_id / 8) % 2) * 8; // 0,8
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + (smem_sel * s_b_stage_offset + 
                           lane_smem_b_n * (BK + B_PAD) + 
                           lane_smem_b_k) * sizeof(half)
      );
      LDMATRIX_X2(RB[j][0], RB[j][1], lane_smem_b_ptr);
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
      uint32_t RA[WARP_TILE_M][4];
      uint32_t RB[WARP_TILE_N][2];

      int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      // ldmatrix for s_a, ldmatrix.trans for s_b.
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
        int lane_smem_b_n = warp_smem_b_n + lane_id % 8; // 0~7, MMA_N=8
        int lane_smem_b_k = ((lane_id / 8) % 2) * 8; // 0,8
        uint32_t lane_smem_b_ptr = (
          smem_b_base_ptr + (stage_sel * s_b_stage_offset + 
                            lane_smem_b_n * (BK + B_PAD) + 
                            lane_smem_b_k) * sizeof(half)
        );
        LDMATRIX_X2(RB[j][0], RB[j][1], lane_smem_b_ptr);
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

  {
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
      // thus, we only need 8 memory issues with 128 bits after shfl_sync.
      // may reuse RA[4][4] as RC0 ? only new RC1[4][4].
      uint32_t RC0[WARP_TILE_N][4];
      uint32_t RC1[WARP_TILE_N][4];
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
        // thus, we only need 8 memory issues with 128 bits after shfl_sync.
        RC0[j][0] = RC[i][j][0];
        RC1[j][0] = RC[i][j][1];
        RC0[j][1] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
        RC0[j][2] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
        RC0[j][3] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
        RC1[j][1] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
        RC1[j][2] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
        RC1[j][3] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
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
          LDST128BITS(C[store_gmem_c_addr_0]) = LDST128BITS(RC0[j][0]); 
          LDST128BITS(C[store_gmem_c_addr_1]) = LDST128BITS(RC1[j][0]); 
        }
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

// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle, dsmem, TN
#define LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(stages, stride)   \
{                                                                                   \
  const int smem_max_size = (                                                       \
    (stages) * BM * (BK + A_PAD) * sizeof(half) +                                   \
    (stages) * BN * (BK + B_PAD) * sizeof(half));                                   \
  cudaFuncSetAttribute(                                                             \
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel<                       \
      MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                                  \
      WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, (stages), true>,                      \
    cudaFuncAttributeMaxDynamicSharedMemorySize,                                    \
    98304);                                                                         \
  const int N_SWIZZLE = (N + (stride) - 1) / (stride);                              \
  dim3 block(NUM_THREADS);                                                          \
  dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE,                          \
             div_ceil(M, BM),                                                       \
             N_SWIZZLE);                                                            \
  hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel<                         \
    MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                                    \
    WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, (stages), true><<<                      \
    grid, block, smem_max_size>>>(                                                  \
    reinterpret_cast<half*>(a.data_ptr()),                                          \
    reinterpret_cast<half*>(b.data_ptr()),                                          \
    reinterpret_cast<half*>(c.data_ptr()),                                          \
    M, N, K                                                                         \
  );                                                                                \
}

#define LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(stages)  \
{                                                                             \
  const int smem_max_size = (                                                 \
    (stages) * BM * (BK + A_PAD) * sizeof(half) +                             \
    (stages) * BN * (BK + B_PAD) * sizeof(half));                             \
  cudaFuncSetAttribute(                                                       \
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel<                 \
      MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                            \
      WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, (stages), false>,               \
    cudaFuncAttributeMaxDynamicSharedMemorySize,                              \
    98304);                                                                   \
  dim3 block(NUM_THREADS);                                                    \
  dim3 grid(div_ceil(N, BN), div_ceil(M, BM));                                \
  hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel<                   \
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
void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn(
  torch::Tensor a, torch::Tensor b, torch::Tensor c, 
  int stages, bool swizzle, int swizzle_stride) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(0); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, N, K)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  constexpr int MMA_TILE_M = 2;
  constexpr int MMA_TILE_N = 4; 
  constexpr int WARP_TILE_M = 4;
  constexpr int WARP_TILE_N = 4;
  constexpr int A_PAD = 0; // 0,8,16
  constexpr int B_PAD = 0; // 0,8,16
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
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(2, swizzle_stride);
      break;
    case 3: 
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(3, swizzle_stride);
      break;
    case 4: 
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(4, swizzle_stride);
      break;
    case 5: 
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(5, swizzle_stride);
      break;
    default:
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(2, swizzle_stride);
      break;
    }
  } else {
    switch (stages)
    {
    case 2:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(2);
      break;
    case 3:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(3);
      break;
    case 4:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(4);
      break;
    case 5:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(5);
      break;
    default:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(2);
      break;
    }
  }
}
