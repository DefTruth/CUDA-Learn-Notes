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

// stage2/3/4 (stage2=double buffers+copy async), 128x128, warp2x4(32,64,16)
// 1. When using shared memory exceeds 48 KB, dynamic shared memory needs to be used,
// i.e., declare a block of dynamic shared memory with extern shared half smem[];. 
// When calling the kernel, the size of the dynamic shared memory needs to be specified, 
// and smem addressing should be used in a one-dimensional array manner. 
// 2. Improve L2 Cache locality (Thread Block Swizzle): https://zhuanlan.zhihu.com/p/555339335
// 3. __launch_bounds__: avoid error 'too many resources required for launch'
// reference: https://blog.csdn.net/feng__shuai/article/details/124395023
template<const int WMMA_M=16, 
         const int WMMA_N=16, 
         const int WMMA_K=16, 
         const int WMMA_TILE_M=4, 
         const int WMMA_TILE_N=2, 
         const int WARP_TILE_M=2, 
         const int WARP_TILE_N=4,
         const int A_PAD=0, 
         const int B_PAD=0, 
         const int K_STAGE=2, 
         const bool BLOCK_SWIZZLE=false>
__global__ void  __launch_bounds__(256) 
hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_kernel(
  half* A, half* B, half* C, int M, int N, int K) {
  // 256 threads(8 warps) per block.
  // const int bx = blockIdx.x;
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int) BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
  constexpr int BK = WMMA_K; // 16
  // s2: 2*128*(16+8)*2=12KB, 2*16*(128+8)*2=8.50KB,  ~21KB
  // s3: 3*128*(16+8)*2=18KB, 3*16*(128+8)*2=12.75KB, ~31KB
  // s4: 4*128*(16+8)*2=24KB, 4*16*(128+8)*2=17KB,    ~41KB
  __shared__ half s_a[K_STAGE][BM][BK + A_PAD], s_b[K_STAGE][BK][BN + B_PAD]; 
  constexpr int s_a_stage_offset = BM * (BK + A_PAD);
  constexpr int s_b_stage_offset = BK * (BN + B_PAD);
 
  // 要保证相同的warp下thread执行相同的指令
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // tid >> 5; // 0~7 warp_id within block
  const int warp_m =  warp_id / 2; // warp_id >> 1; // 0,1,2,3
  const int warp_n = warp_id % 2; // 0,1
  
  // 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=16 按行读取 A行主序
  // 对于s_a每行16个数据，每个线程读取8个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2; // tid >> 1; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共16行，需要16x16=256个线程
  int load_smem_b_k = tid / 16; // tid >> 4; // row 0~15
  int load_smem_b_n =  (tid % 16) * 8; // ((tid & 0xF) << 3); // col 0,8,...,120
  // 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> 
  C_frag[WARP_TILE_M][WARP_TILE_N];
  
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }

  // may avoid cvta overhead ? only cvta smem base ptr once for cp.async.
  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  #pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
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
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) { 
    // s2/4 can use bitwise ops but s3 can not, so, we use mod
    // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
    // s3: (k + 1) % 3
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
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

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> B_frag[WARP_TILE_N];
    
    // compute stage 0
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][0], BK + A_PAD); 
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::load_matrix_sync(B_frag[j], &s_b[smem_sel][0][warp_smem_b_n], BN + B_PAD);
    }

    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
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
      const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                     wmma::row_major> A_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                     wmma::row_major> B_frag[WARP_TILE_N];
    
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
        const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        wmma::load_matrix_sync(A_frag[i], &s_a[stage_sel][warp_smem_a_m][0], BK + A_PAD); 
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
        const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
        wmma::load_matrix_sync(B_frag[j], &s_b[stage_sel][0][warp_smem_b_n], BN + B_PAD);
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

// stage2/3/4 (stage2=double buffers+copy async), 128x128, warp2x4(32,64,16)
// 1. When using shared memory exceeds 48 KB, dynamic shared memory needs to be used,
// i.e., declare a block of dynamic shared memory with extern shared half smem[];. 
// When calling the kernel, the size of the dynamic shared memory needs to be specified, 
// and smem addressing should be used in a one-dimensional array manner. 
// 2. Improve L2 Cache locality (Thread Block Swizzle): https://zhuanlan.zhihu.com/p/555339335
// 3. __launch_bounds__: avoid error 'too many resources required for launch'
// reference: https://blog.csdn.net/feng__shuai/article/details/124395023
template<const int WMMA_M=16, 
         const int WMMA_N=16, 
         const int WMMA_K=16, 
         const int WMMA_TILE_M=4, 
         const int WMMA_TILE_N=2, 
         const int WARP_TILE_M=2, 
         const int WARP_TILE_N=4,
         const int A_PAD=0, 
         const int B_PAD=0, 
         const int K_STAGE=2,
         const bool BLOCK_SWIZZLE=false>
__global__ void __launch_bounds__(256) 
hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel(
  half* A, half* B, half* C, int M, int N, int K) {
  // 256 threads(8 warps) per block.
  // const int bx = blockIdx.x;
   // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int) BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
  constexpr int BK = WMMA_K; // 16
  // s2: 2*128*(16+8)*2=12KB, 2*16*(128+8)*2=8.50KB,  ~21KB
  // s3: 3*128*(16+8)*2=18KB, 3*16*(128+8)*2=12.75KB, ~31KB
  // s4: 4*128*(16+8)*2=24KB, 4*16*(128+8)*2=17KB,    ~41KB
  // s5: 5*128*(16+8)*2=30KB, 5*16*(128+8)*2=21.25KB, ~52KB > 48KB
  extern __shared__ half smem[]; 
  half* s_a = smem;
  half* s_b = smem + K_STAGE * BM * (BK + A_PAD);
  constexpr int s_a_stage_offset = BM * (BK + A_PAD);
  constexpr int s_b_stage_offset = BK * (BN + B_PAD);

  // 要保证相同的warp下thread执行相同的指令
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int warp_m = warp_id / 2; // 0,1,2,3
  const int warp_n = warp_id % 2; // 0,1
  
  // 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=16 按行读取 A行主序
  // 对于s_a每行16个数据，每个线程读取8个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共16行，需要16x16=256个线程
  int load_smem_b_k = tid / 16; // row 0~15
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
  // 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> 
  C_frag[WARP_TILE_M][WARP_TILE_N];
  
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }

  // only cvta smem base ptr once for cp.async.
  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  #pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
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
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) { 
    // s2/4 can use bitwise ops but s3 can not, so, we use mod
    // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
    // s3: (k + 1) % 3
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    // load stage 2, k start from 2
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

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> B_frag[WARP_TILE_N];
    
    // compute stage 0
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
      int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      half* load_smem_a_frag_ptr = (s_a + smem_sel * s_a_stage_offset + 
                                    warp_smem_a_m * (BK + A_PAD) 
                                    + 0); // BK=WMMA_K=16
      wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD); 
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
      int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      half* load_smem_b_frag_ptr = (s_b + smem_sel * s_b_stage_offset + 
                                    0 * (BN + B_PAD) + 
                                    warp_smem_b_n); // BK=WMMA_K=16
      wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
    }

    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
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
      const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                     wmma::row_major> A_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                     wmma::row_major> B_frag[WARP_TILE_N];
    
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
        int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        half* load_smem_a_frag_ptr = (s_a + stage_sel * s_a_stage_offset + 
                                      warp_smem_a_m * (BK + A_PAD) 
                                      + 0); // BK=WMMA_K=16
        wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD); 
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
        int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
        half* load_smem_b_frag_ptr = (s_b + stage_sel * s_b_stage_offset + 
                                      0 * (BN + B_PAD) + 
                                      warp_smem_b_n); // BK=WMMA_K=16
        wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
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

// stage with 256x256 block, warp4x4(64,64,16), dynamic smem
// __launch_bounds__: avoid error 'too many resources required for launch'
// reference: https://blog.csdn.net/feng__shuai/article/details/124395023
template<const int WMMA_M=16, 
         const int WMMA_N=16,
         const int WMMA_K=16, 
         const int WMMA_TILE_M=4, 
         const int WMMA_TILE_N=4, 
         const int WARP_TILE_M=4, 
         const int WARP_TILE_N=4,
         const int A_PAD=0, 
         const int B_PAD=0, 
         const int K_STAGE=2, 
         const bool BLOCK_SWIZZLE=false>
__global__ void __launch_bounds__(512) 
hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem_kernel(
  half* A, half* B, half* C, int M, int N, int K) {
  // 512 threads(16 warps) per block / 256 threads, 8 warps
  // const int bx = blockIdx.x;
  // BLOCK_SWIZZLE 0/1 控制是否使用 block swizzle
  const int bx = ((int) BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*4=256
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x4*4=256
  constexpr int BK = WMMA_K; // 16
  extern __shared__ half smem[]; 
  half* s_a = smem;
  half* s_b = smem + K_STAGE * BM * (BK + A_PAD);
  constexpr int s_a_stage_offset = BM * (BK + A_PAD);
  constexpr int s_b_stage_offset = BK * (BN + B_PAD);

  // 要保证相同的warp下thread执行相同的指令
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // 0~15 warp_id within block
  const int warp_m = warp_id / 4; // 0,1,2,3
  const int warp_n = warp_id % 4; // 0,1,2,3
  
  // 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=256 BK=16 按行读取 A行主序
  // 对于s_a每行16个数据，每个线程读取8个，需要2个线程；总共256行，需要刚好256x2=512线程
  int load_smem_a_m = tid / 2; // row 0~255
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0, 8
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=256 按行读取 B行主序
  // 对于s_b每行256个数据，每个线程读8个数据，需要32个线程；总共16行，需要32x16=512个线程
  int load_smem_b_k = tid / 32; // row 0~15
  int load_smem_b_n = (tid % 32) * 8; // col 0,8,...,256
  // 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> 
  C_frag[WARP_TILE_M][WARP_TILE_N];
  
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }

  // only cvta smem base ptr once for cp.async.
  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  #pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
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
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) { 
    // s2/4 can use bitwise ops but s3 can not, so, we use mod
    // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
    // s3: (k + 1) % 3
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 
    
    // load stage 2, k start from 2
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

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> B_frag[WARP_TILE_N];
    
    // compute stage 0
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
      int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      half* load_smem_a_frag_ptr = (s_a + smem_sel * s_a_stage_offset + 
                                    warp_smem_a_m * (BK + A_PAD) 
                                    + 0); // BK=WMMA_K=16
      wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD); 
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
      int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      half* load_smem_b_frag_ptr = (s_b + smem_sel * s_b_stage_offset + 
                                    0 * (BN + B_PAD) + 
                                    warp_smem_b_n); // BK=WMMA_K=16
      wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
    }

    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
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
      const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                     wmma::row_major> A_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                     wmma::row_major> B_frag[WARP_TILE_N];
    
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
        int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        half* load_smem_a_frag_ptr = (s_a + stage_sel * s_a_stage_offset + 
                                      warp_smem_a_m * (BK + A_PAD) 
                                      + 0); // BK=WMMA_K=16
        wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD); 
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
        int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
        half* load_smem_b_frag_ptr = (s_b + stage_sel * s_b_stage_offset + 
                                      0 * (BN + B_PAD) + 
                                      warp_smem_b_n); // BK=WMMA_K=16
        wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
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

// 128x128, Stages + K32 + Reg Buffers, mma4x2, warp2x4x2(32,64,32)
// stage2/3/4 (stage2=double buffers+copy async)
// 1. When using shared memory exceeds 48 KB, dynamic shared memory needs to be used,
// i.e., declare a block of dynamic shared memory with extern shared half smem[];. 
// When calling the kernel, the size of the dynamic shared memory needs to be specified, 
// and smem addressing should be used in a one-dimensional array manner. 
// 2. Improve L2 Cache locality (Thread Block Swizzle): https://zhuanlan.zhihu.com/p/555339335
// 3. __launch_bounds__: avoid error 'too many resources required for launch'
// reference: https://blog.csdn.net/feng__shuai/article/details/124395023
template<const int WMMA_M=16, 
         const int WMMA_N=16, 
         const int WMMA_K=16, 
         const int WMMA_TILE_M=4, 
         const int WMMA_TILE_N=2, 
         const int WARP_TILE_M=2, 
         const int WARP_TILE_N=4,
         const int WARP_TILE_K=2, 
         const int A_PAD=0, 
         const int B_PAD=0, 
         const int K_STAGE=2,
         const bool BLOCK_SWIZZLE=false>
__global__ void __launch_bounds__(256) 
hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_stages_dsmem_kernel(
  half* A, half* B, half* C, int M, int N, int K) {
  // 256 threads(8 warps) per block.
  // const int bx = blockIdx.x;
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int) BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K * WARP_TILE_K);
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
  constexpr int BK = WMMA_K * WARP_TILE_K; // 16*2=32
  // s2: 2*128*(32)*2=16KB, 2*32*(128+16)*2=18KB, ~42KB
  // s3: 3*128*(32)*2=24KB, 3*32*(128+16)*2=27KB, ~51KB
  // s4: 4*128*(32)*2=32KB, 4*32*(128+16)*2=36KB, ~68KB
  // s4: 5*128*(32)*2=40KB, 5*32*(128+16)*2=45KB, ~85KB
  extern __shared__ half smem[]; 
  half* s_a = smem;
  half* s_b = smem + K_STAGE * BM * (BK + A_PAD);
  constexpr int s_a_stage_offset = BM * (BK + A_PAD);
  constexpr int s_b_stage_offset = BK * (BN + B_PAD);

  // 要保证相同的warp下thread执行相同的指令
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int warp_m = warp_id / 2; // 0,1,2,3
  const int warp_n = warp_id % 2; // 0,1
  
  // 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=32 按行读取 A行主序
  // 对于s_a每行32个数据，每个线程读取16个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 16; // col 0,16
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=32 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读16个数据，需要8个线程；总共32行，需要32x16=256个线程
  int load_smem_b_k = tid / 8; // row 0~31
  int load_smem_b_n = (tid % 8) * 16; // col 0,16,...,127
  // 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> 
  C_frag[WARP_TILE_M][WARP_TILE_N];
  
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }

  // only cvta smem base ptr once for cp.async.
  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  #pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * (WMMA_K * WARP_TILE_K) + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * (WMMA_K * WARP_TILE_K) + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 
    
    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (k * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (k * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );

    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
    CP_ASYNC_CG(load_smem_a_ptr + 16, &A[load_gmem_a_addr + 8], 16);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);   
    CP_ASYNC_CG(load_smem_b_ptr + 16, &B[load_gmem_b_addr + 8], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE-2); // s2->0, s3->1, s4->2
  __syncthreads(); 

  #pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) { 
    // s2/4 can use bitwise ops but s3 can not, so, we use mod
    // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
    // s3: (k + 1) % 3
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * (WMMA_K * WARP_TILE_K) + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * (WMMA_K * WARP_TILE_K) + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    // load stage 2, k start from 2
    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );

    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
    CP_ASYNC_CG(load_smem_a_ptr + 16, &A[load_gmem_a_addr + 8], 16);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);   
    CP_ASYNC_CG(load_smem_b_ptr + 16, &B[load_gmem_b_addr + 8], 16);

    CP_ASYNC_COMMIT_GROUP();

    // WARP_TILE_K=2
    for (int warp_k = 0; warp_k < WARP_TILE_K; ++warp_k) {
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                     wmma::row_major> A_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                     wmma::row_major> B_frag[WARP_TILE_N];
      const int warp_smem_k = warp_k * WMMA_K; // 0,16

      // compute stage 0
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
        int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        half* load_smem_a_frag_ptr = (s_a + smem_sel * s_a_stage_offset + 
                                      warp_smem_a_m * (BK + A_PAD) + 
                                      warp_smem_k);
        wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD); 
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
        int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
        half* load_smem_b_frag_ptr = (s_b + smem_sel * s_b_stage_offset + 
                                      warp_smem_k * (BN + B_PAD) + 
                                      warp_smem_b_n);
        wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
      }

      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
        }
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
      const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);

      #pragma unroll
      for (int warp_k = 0; warp_k < WARP_TILE_K; ++warp_k) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                      wmma::row_major> A_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                      wmma::row_major> B_frag[WARP_TILE_N];
        const int warp_smem_k = warp_k * WMMA_K; // 0,16

        // compute stage 0
        #pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
          // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
          int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
          half* load_smem_a_frag_ptr = (s_a + stage_sel * s_a_stage_offset + 
                                        warp_smem_a_m * (BK + A_PAD) + 
                                        warp_smem_k);
          wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD); 
        }

        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
          int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
          half* load_smem_b_frag_ptr = (s_b + stage_sel * s_b_stage_offset + 
                                        warp_smem_k * (BN + B_PAD) + 
                                        warp_smem_b_n);
          wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
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

// 128x128, Stages + K32 + Reg Buffers, mma4x4, warp2x2x2(32,32,32)
// stage2/3/4 (stage2=double buffers+copy async)
// 1. When using shared memory exceeds 48 KB, dynamic shared memory needs to be used,
// i.e., declare a block of dynamic shared memory with extern shared half smem[];. 
// When calling the kernel, the size of the dynamic shared memory needs to be specified, 
// and smem addressing should be used in a one-dimensional array manner. 
// 2. Improve L2 Cache locality (Thread Block Swizzle): https://zhuanlan.zhihu.com/p/555339335
// 3. __launch_bounds__: avoid error 'too many resources required for launch'
// reference: https://blog.csdn.net/feng__shuai/article/details/124395023
template<const int WMMA_M=16, 
         const int WMMA_N=16, 
         const int WMMA_K=16, 
         const int WMMA_TILE_M=4, 
         const int WMMA_TILE_N=4, 
         const int WARP_TILE_M=2, 
         const int WARP_TILE_N=2,
         const int WARP_TILE_K=2, 
         const int A_PAD=0, 
         const int B_PAD=0, 
         const int K_STAGE=2,
         const bool BLOCK_SWIZZLE=false>
__global__ void __launch_bounds__(512) 
hgemm_wmma_m16n16k16_mma4x4_warp2x2x2_stages_dsmem_kernel(
  half* A, half* B, half* C, int M, int N, int K) {
  // 512 threads(16 warps) per block.
  // const int bx = blockIdx.x;
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int) BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K * WARP_TILE_K);
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x4*2=128
  constexpr int BK = WMMA_K * WARP_TILE_K; // 16*2=32
  // s2: 2*128*(32)*2=16KB, 2*32*(128+16)*2=18KB, ~42KB
  // s3: 3*128*(32)*2=24KB, 3*32*(128+16)*2=27KB, ~51KB
  // s4: 4*128*(32)*2=32KB, 4*32*(128+16)*2=36KB, ~68KB
  // s5: 5*128*(32)*2=40KB, 5*32*(128+16)*2=45KB, ~85KB
  extern __shared__ half smem[]; 
  half* s_a = smem;
  half* s_b = smem + K_STAGE * BM * (BK + A_PAD);
  constexpr int s_a_stage_offset = BM * (BK + A_PAD);
  constexpr int s_b_stage_offset = BK * (BN + B_PAD);

  // 要保证相同的warp下thread执行相同的指令
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int warp_m = warp_id / 4; // 0,1,2,3
  const int warp_n = warp_id % 4; // 0,1,2,3
  
  // 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=32 按行读取 A行主序
  // 对于s_a每行32个数据，每个线程读取8个，需要4个线程；总共128行，需要128x4刚好512线程
  int load_smem_a_m = tid / 4; // row 0~127
  int load_smem_a_k = (tid % 4) * 8; // col 0,8,16,24
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=32 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共32行，需要32x16=256个线程
  int load_smem_b_k = tid / 16; // row 0~31
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
  // 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> 
  C_frag[WARP_TILE_M][WARP_TILE_N];
  
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }

  // only cvta smem base ptr once for cp.async.
  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  #pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * (WMMA_K * WARP_TILE_K) + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * (WMMA_K * WARP_TILE_K) + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 
    
    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (k * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (k * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );

    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);   

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE-2); // s2->0, s3->1, s4->2
  __syncthreads(); 

  #pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) { 
    // s2/4 can use bitwise ops but s3 can not, so, we use mod
    // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
    // s3: (k + 1) % 3
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * (WMMA_K * WARP_TILE_K) + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * (WMMA_K * WARP_TILE_K) + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    // load stage 2, k start from 2
    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );

    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);   

    CP_ASYNC_COMMIT_GROUP();

    // WARP_TILE_K=2
    for (int warp_k = 0; warp_k < WARP_TILE_K; ++warp_k) {
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                     wmma::row_major> A_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                     wmma::row_major> B_frag[WARP_TILE_N];
      const int warp_smem_k = warp_k * WMMA_K; // 0,16

      // compute stage 0
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
        int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        half* load_smem_a_frag_ptr = (s_a + smem_sel * s_a_stage_offset + 
                                      warp_smem_a_m * (BK + A_PAD) + 
                                      warp_smem_k);
        wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD); 
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
        int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
        half* load_smem_b_frag_ptr = (s_b + smem_sel * s_b_stage_offset + 
                                      warp_smem_k * (BN + B_PAD) + 
                                      warp_smem_b_n);
        wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
      }

      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
        }
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
      const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);

      #pragma unroll
      for (int warp_k = 0; warp_k < WARP_TILE_K; ++warp_k) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                      wmma::row_major> A_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                      wmma::row_major> B_frag[WARP_TILE_N];
        const int warp_smem_k = warp_k * WMMA_K; // 0,16

        // compute stage 0
        #pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
          // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
          int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
          half* load_smem_a_frag_ptr = (s_a + stage_sel * s_a_stage_offset + 
                                        warp_smem_a_m * (BK + A_PAD) + 
                                        warp_smem_k);
          wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD); 
        }

        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
          int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
          half* load_smem_b_frag_ptr = (s_b + stage_sel * s_b_stage_offset + 
                                        warp_smem_k * (BN + B_PAD) + 
                                        warp_smem_b_n);
          wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
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

// TODO: 256x128, Stages + K32 + Reg Buffers, mma4x2, warp4x4x2(64,64,16)
template<const int WMMA_M=16, 
         const int WMMA_N=16, 
         const int WMMA_K=16, 
         const int WMMA_TILE_M=4, 
         const int WMMA_TILE_N=2, 
         const int WARP_TILE_M=4, 
         const int WARP_TILE_N=4,
         const int WARP_TILE_K=1, 
         const int A_PAD=0, 
         const int B_PAD=0, 
         const int K_STAGE=2,
         const bool BLOCK_SWIZZLE=false>
__global__ void __launch_bounds__(256) 
hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem_kernel(
  half* A, half* B, half* C, int M, int N, int K) {
  // 256 threads(8 warps) per block.
  // const int bx = blockIdx.x;
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int) BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K * WARP_TILE_K);
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*4=256
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
  constexpr int BK = WMMA_K * WARP_TILE_K; // 16*2=32
  // s2: 2*128*(32)*2=16KB, 2*32*(128+16)*2=18KB, ~42KB
  // s3: 3*128*(32)*2=24KB, 3*32*(128+16)*2=27KB, ~51KB
  // s4: 4*128*(32)*2=32KB, 4*32*(128+16)*2=36KB, ~68KB
  // s4: 5*128*(32)*2=40KB, 5*32*(128+16)*2=45KB, ~85KB
  extern __shared__ half smem[]; 
  half* s_a = smem;
  half* s_b = smem + K_STAGE * BM * (BK + A_PAD);
  constexpr int s_a_stage_offset = BM * (BK + A_PAD);
  constexpr int s_b_stage_offset = BK * (BN + B_PAD);

  // 要保证相同的warp下thread执行相同的指令
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int warp_m = warp_id / 2; // 0,1,2,3
  const int warp_n = warp_id % 2; // 0,1
  
  // 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=256 BK=32 按行读取 A行主序
  // 对于s_a每行16个数据，每个线程读取16个，需要1个线程；总共256行，刚好256线程
  int load_smem_a_m = tid; // row 0~255
  int load_smem_a_k = 0; // col 0,16
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共16行，需要16x16=256个线程
  int load_smem_b_k = tid / 16; // row 0~15
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
  // 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> 
  C_frag[WARP_TILE_M][WARP_TILE_N];
  
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }

  // only cvta smem base ptr once for cp.async.
  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  #pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * (WMMA_K * WARP_TILE_K) + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * (WMMA_K * WARP_TILE_K) + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 
    
    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (k * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (k * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );

    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);   
    CP_ASYNC_CG(load_smem_a_ptr + 16, &A[load_gmem_a_addr + 8], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE-2); // s2->0, s3->1, s4->2
  __syncthreads(); 

  #pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) { 
    // s2/4 can use bitwise ops but s3 can not, so, we use mod
    // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
    // s3: (k + 1) % 3
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * (WMMA_K * WARP_TILE_K) + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * (WMMA_K * WARP_TILE_K) + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    // load stage 2, k start from 2
    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );

    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);   
    CP_ASYNC_CG(load_smem_a_ptr + 16, &A[load_gmem_a_addr + 8], 16);

    CP_ASYNC_COMMIT_GROUP();

    for (int warp_k = 0; warp_k < WARP_TILE_K; ++warp_k) {
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                     wmma::row_major> A_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                     wmma::row_major> B_frag[WARP_TILE_N];
      const int warp_smem_k = warp_k * WMMA_K; // 0,16

      // compute stage 0
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
        int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        half* load_smem_a_frag_ptr = (s_a + smem_sel * s_a_stage_offset + 
                                      warp_smem_a_m * (BK + A_PAD) + 
                                      warp_smem_k);
        wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD); 
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
        int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
        half* load_smem_b_frag_ptr = (s_b + smem_sel * s_b_stage_offset + 
                                      warp_smem_k * (BN + B_PAD) + 
                                      warp_smem_b_n);
        wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
      }

      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
        }
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
      const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);

      #pragma unroll
      for (int warp_k = 0; warp_k < WARP_TILE_K; ++warp_k) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                      wmma::row_major> A_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                      wmma::row_major> B_frag[WARP_TILE_N];
        const int warp_smem_k = warp_k * WMMA_K; // 0,16

        // compute stage 0
        #pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
          // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
          int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
          half* load_smem_a_frag_ptr = (s_a + stage_sel * s_a_stage_offset + 
                                        warp_smem_a_m * (BK + A_PAD) + 
                                        warp_smem_k);
          wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD); 
        }

        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
          int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
          half* load_smem_b_frag_ptr = (s_b + stage_sel * s_b_stage_offset + 
                                        warp_smem_k * (BN + B_PAD) + 
                                        warp_smem_b_n);
          wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
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

// TODO: Warp swizzle/permute support ? (MMA, not WMMA)

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

// 128x128 warp2x4(32,64) w/o dynamic smem
#define LAUNCH_161616_STAGE_SWIZZLE_KERNEL(stages, stride)   \
{                                                            \
  const int N_SWIZZLE = (N + (stride) - 1) / (stride);       \
  dim3 block(NUM_THREADS);                                   \
  dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE,   \
             div_ceil(M, BM),                                \
             N_SWIZZLE);                                     \
  hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_kernel<         \
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,        \
    WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD,                  \
    (stages), true><<<grid, block>>>(                        \
    reinterpret_cast<half*>(a.data_ptr()),                   \
    reinterpret_cast<half*>(b.data_ptr()),                   \
    reinterpret_cast<half*>(c.data_ptr()),                   \
    M, N, K                                                  \
  );                                                         \
}

#define LAUNCH_161616_STAGE_NO_SWIZZLE_KERNEL(stages)        \
{                                                            \
  dim3 block(NUM_THREADS);                                   \
  dim3 grid(div_ceil(N, BN), div_ceil(M, BM));               \
  hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_kernel<         \
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,        \
    WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD,                  \
    (stages), false><<<grid, block>>>(                       \
    reinterpret_cast<half*>(a.data_ptr()),                   \
    reinterpret_cast<half*>(b.data_ptr()),                   \
    reinterpret_cast<half*>(c.data_ptr()),                   \
    M, N, K                                                  \
  );                                                         \
}

// 128x128 warp2x4(32,64) stage 2/3/4 w/o block swizzle across N dim, static smem < 48KB
void hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages(
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
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2; 
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  // s_a 4  ways bank conflicts within warp, after pad 8  -> 4 ways bank conflicts.
  // s_b 16 ways bank conflicts within warp, after pad 8  -> 8 ways bank conflicts.
  // s_b 16 ways bank conflicts within warp, after pad 16 -> 4 ways bank conflicts.
  // so, the best padding policy for s_a and s_b is A_PAD=0/8, B_PAD=16. Thus, 
  // improve B_PAD consume 8x~ less smem than A_PAD, 16xB_PAD vs 128xA_PAD.
  constexpr int A_PAD = 0;  // 0,8,16
  constexpr int B_PAD = 16; // 0,8,16
  constexpr int NUM_THREADS= (
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 2 * 4 * 32 = 256
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;    
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;    
  constexpr int BK = WMMA_K;   
  // s2: 2*128*(16)*2=8KB,  2*16*(128+16)*2=9KB,    ~17KB
  // s3: 3*128*(16)*2=12KB, 3*16*(128+16)*2=13.5KB, ~26KB
  // s4: 4*128*(16)*2=16KB, 4*16*(128+16)*2=18KB,   ~34KB                            
  // s5: 5*128*(16)*2=20KB, 5*16*(128+16)*2=22.5KB, ~43KB    
  if (swizzle) {
    assert(swizzle_stride % 256 == 0);
    switch (stages)
    {
    case 2: // ~17KB
      LAUNCH_161616_STAGE_SWIZZLE_KERNEL(2, swizzle_stride);
      break;
    case 3: // ~26KB
      LAUNCH_161616_STAGE_SWIZZLE_KERNEL(3, swizzle_stride);
      break;
    case 4: // ~34KB
      LAUNCH_161616_STAGE_SWIZZLE_KERNEL(4, swizzle_stride);
      break;
    case 5: // ~43KB
      LAUNCH_161616_STAGE_SWIZZLE_KERNEL(5, swizzle_stride);
      break;
    default:
      LAUNCH_161616_STAGE_SWIZZLE_KERNEL(2, swizzle_stride);
      break;
    }
  } else {
    switch (stages)
    {
    case 2:
      LAUNCH_161616_STAGE_NO_SWIZZLE_KERNEL(2);
      break;
    case 3:
      LAUNCH_161616_STAGE_NO_SWIZZLE_KERNEL(3);
      break;
    case 4:
      LAUNCH_161616_STAGE_NO_SWIZZLE_KERNEL(4);
      break;
    default:
      LAUNCH_161616_STAGE_NO_SWIZZLE_KERNEL(2);
      break;
    }
  }
}

// 128x128 warp2x4(32,64) w dynamic smem, 98304=96KB < Ampere, Ada, Hopper ...
#define LAUNCH_161616_STAGE_SWIZZLE_DSMEM_KERNEL(stages, stride)  \
{                                                                 \
  const int smem_max_size = (                                     \
    (stages) * BM * (BK + A_PAD) * sizeof(half) +                 \
    (stages) * BK * (BN + B_PAD) * sizeof(half));                 \
  cudaFuncSetAttribute(                                           \
    hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel<      \
      WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,           \
      WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, (stages), true>,    \
    cudaFuncAttributeMaxDynamicSharedMemorySize,                  \
    98304);                                                       \
  const int N_SWIZZLE = (N + (stride) - 1) / (stride);            \
  dim3 block(NUM_THREADS);                                        \
  dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE,        \
             div_ceil(M, BM),                                     \
             N_SWIZZLE);                                          \
  hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel<        \
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,             \
    WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, (stages), true><<<    \
    grid, block, smem_max_size>>>(                                \
    reinterpret_cast<half*>(a.data_ptr()),                        \
    reinterpret_cast<half*>(b.data_ptr()),                        \
    reinterpret_cast<half*>(c.data_ptr()),                        \
    M, N, K                                                       \
  );                                                              \
}

#define LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_KERNEL(stages)    \
{                                                              \
  const int smem_max_size = (                                  \
    (stages) * BM * (BK + A_PAD) * sizeof(half) +              \
    (stages) * BK * (BN + B_PAD) * sizeof(half));              \
  cudaFuncSetAttribute(                                        \
    hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel<   \
      WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,        \
      WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, (stages), false>,\
    cudaFuncAttributeMaxDynamicSharedMemorySize,               \
    98304);                                                    \
  dim3 block(NUM_THREADS);                                     \
  dim3 grid(div_ceil(N, BN), div_ceil(M, BM));                 \
  hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel<     \
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,          \
    WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, (stages), false><<<\
    grid, block, smem_max_size>>>(                             \
    reinterpret_cast<half*>(a.data_ptr()),                     \
    reinterpret_cast<half*>(b.data_ptr()),                     \
    reinterpret_cast<half*>(c.data_ptr()),                     \
    M, N, K                                                    \
  );                                                           \
}

// 128x128 warp2x4(32,64) stage 2/3/4 + dynamic smem, w/o block swizzle across N dim
void hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem(
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
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2; 
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  // s_a 4  ways bank conflicts within warp, after pad 8  -> 4 ways bank conflicts.
  // s_b 16 ways bank conflicts within warp, after pad 8  -> 8 ways bank conflicts.
  // s_b 16 ways bank conflicts within warp, after pad 16 -> 4 ways bank conflicts.
  // so, the best padding policy for s_a and s_b is A_PAD=0/8, B_PAD=16. Thus, 
  // improve B_PAD consume 8x~ less smem than A_PAD, 16xB_PAD vs 128xA_PAD.
  constexpr int A_PAD = 0;  // 0,8,16
  constexpr int B_PAD = 16; // 0,8,16
  constexpr int NUM_THREADS= (
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 2 * 4 * 32 = 256
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;    
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;    
  constexpr int BK = WMMA_K;   
  // s2: 2*128*(16)*2=8KB,  2*16*(128+16)*2=9KB,    ~17KB
  // s3: 3*128*(16)*2=12KB, 3*16*(128+16)*2=13.5KB, ~26KB
  // s4: 4*128*(16)*2=16KB, 4*16*(128+16)*2=18KB,   ~34KB                            
  // s5: 5*128*(16)*2=20KB, 5*16*(128+16)*2=22.5KB, ~43KB         
  // s6: 6*128*(16)*2=24KB, 6*16*(128+16)*2=27KB,   ~51KB > 48KB
  if (swizzle) {
    assert(swizzle_stride % 256 == 0);
    switch (stages)
    {
    case 2: // ~17KB
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_KERNEL(2, swizzle_stride);
      break;
    case 3: // ~26KB
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_KERNEL(3, swizzle_stride);
      break;
    case 4: // ~34K
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_KERNEL(4, swizzle_stride);
      break;
    case 5: // ~43KB
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_KERNEL(5, swizzle_stride);
      break;
    case 6: // ~51KB
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_KERNEL(6, swizzle_stride);
      break;
    default:
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_KERNEL(2, swizzle_stride);
      break;
    }
  } else {
    switch (stages)
    {
    case 2:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_KERNEL(2);
      break;
    case 3:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_KERNEL(3);
      break;
    case 4:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_KERNEL(4);
      break;
    case 5:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_KERNEL(5);
      break;
    case 6:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_KERNEL(6);
      break;
    default:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_KERNEL(2);
      break;
    }
  }
}

// 256x256 warp4x4(64,64,32) w dynamic smem, 98304=96KB < Ampere, Ada, Hopper ...
#define LAUNCH_161616_STAGE_SWIZZLE_DSMEM_256x256_KERNEL(stages, stride)  \
{                                                                         \
  const int smem_max_size = (                                             \
    (stages) * BM * (BK + A_PAD) * sizeof(half) +                         \
    (stages) * BK * (BN + B_PAD) * sizeof(half));                         \
  cudaFuncSetAttribute(                                                   \
    hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem_kernel<              \
      WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,                   \
      WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, (stages), true>,            \
    cudaFuncAttributeMaxDynamicSharedMemorySize,                          \
    98304);                                                               \
  const int N_SWIZZLE = (N + (stride) - 1) / (stride);                    \
  dim3 block(NUM_THREADS);                                                \
  dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE,                \
             div_ceil(M, BM),                                             \
             N_SWIZZLE);                                                  \
  hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem_kernel<                \
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,                     \
    WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, (stages), true><<<            \
    grid, block, smem_max_size>>>(                                        \
    reinterpret_cast<half*>(a.data_ptr()),                                \
    reinterpret_cast<half*>(b.data_ptr()),                                \
    reinterpret_cast<half*>(c.data_ptr()),                                \
    M, N, K                                                               \
  );                                                                      \
}

#define LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_256x256_KERNEL(stages)       \
{                                                                         \
  const int smem_max_size = (                                             \
    (stages) * BM * (BK + A_PAD) * sizeof(half) +                         \
    (stages) * BK * (BN + B_PAD) * sizeof(half));                         \
  cudaFuncSetAttribute(                                                   \
    hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem_kernel<              \
      WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,                   \
      WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, (stages), false>,           \
    cudaFuncAttributeMaxDynamicSharedMemorySize,                          \
    98304);                                                               \
  dim3 block(NUM_THREADS);                                                \
  dim3 grid(div_ceil(N, BN), div_ceil(M, BM));                            \
  hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem_kernel<                \
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,                     \
    WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, (stages), false><<<           \
    grid, block, smem_max_size>>>(                                        \
    reinterpret_cast<half*>(a.data_ptr()),                                \
    reinterpret_cast<half*>(b.data_ptr()),                                \
    reinterpret_cast<half*>(c.data_ptr()),                                \
    M, N, K                                                               \
  );                                                                      \
}

// 256x256 warp4x4(64,64,32) stage 2/3/4 + dynamic smem, w/o block swizzle across N dim
void hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem(
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
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 4; 
  constexpr int WARP_TILE_M = 4;
  constexpr int WARP_TILE_N = 4;
  // s_a 4  ways bank conflicts within warp, after pad 8  -> 4 ways bank conflicts.
  // s_b 16 ways bank conflicts within warp, after pad 8  -> 8 ways bank conflicts.
  // s_b 16 ways bank conflicts within warp, after pad 16 -> 4 ways bank conflicts.
  // so, the best padding policy for s_a and s_b is A_PAD=0/8, B_PAD=16. Thus, 
  // improve B_PAD consume 16x~ less smem than A_PAD, 16xB_PAD vs 256xA_PAD.
  constexpr int A_PAD = 0;
  constexpr int B_PAD = 16;
  constexpr int NUM_THREADS= (
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 4 * 32 = 512
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 256
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 256
  constexpr int BK = WMMA_K;      
  // s2: 2*256*(16)*2=16KB, 2*16*(256+16)*2=17KB,   ~33KB
  // s3: 3*256*(16)*2=24KB, 3*16*(256+16)*2=25.5KB, ~50KB > 48KB
  // s4: 4*256*(16)*2=32KB, 4*16*(256+16)*2=34KB,   ~66KB    
  if (swizzle) {
    assert(swizzle_stride % 256 == 0);
    switch (stages)
    {
    case 2: // ~33KB
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_256x256_KERNEL(2, swizzle_stride);
      break;
    case 3: // ~50KB
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_256x256_KERNEL(3, swizzle_stride);
      break;
    case 4: // ~66KB
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_256x256_KERNEL(4, swizzle_stride);
      break;
    default:
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_256x256_KERNEL(2, swizzle_stride);
      break;
    }
  } else {
    switch (stages)
    {
    case 2:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_256x256_KERNEL(2);
      break;
    case 3:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_256x256_KERNEL(3);
      break;
    case 4:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_256x256_KERNEL(4);
      break;
    default:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_256x256_KERNEL(2);
      break;
    }
  }
}

// 128x128 warp2x4x2(32,64,32) w dynamic smem, 98304=96KB < Ampere, Ada, Hopper ...
#define LAUNCH_161616_STAGE_SWIZZLE_DSMEM_K32_KERNEL(stages, stride)\
{                                                                 \
  const int smem_max_size = (                                     \
    (stages) * BM * (BK + A_PAD) * sizeof(half) +                 \
    (stages) * BK * (BN + B_PAD) * sizeof(half));                 \
  cudaFuncSetAttribute(                                           \
    hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_stages_dsmem_kernel<    \
      WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,           \
      WARP_TILE_M, WARP_TILE_N, WARP_TILE_K,                      \
      A_PAD, B_PAD, (stages), true>,                              \
    cudaFuncAttributeMaxDynamicSharedMemorySize,                  \
    98304);                                                       \
  const int N_SWIZZLE = (N + (stride) - 1) / (stride);            \
  dim3 block(NUM_THREADS);                                        \
  dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE,        \
             div_ceil(M, BM),                                     \
             N_SWIZZLE);                                          \
  hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_stages_dsmem_kernel<      \
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,             \
    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K,                        \
    A_PAD, B_PAD, (stages), true><<<                              \
    grid, block, smem_max_size>>>(                                \
    reinterpret_cast<half*>(a.data_ptr()),                        \
    reinterpret_cast<half*>(b.data_ptr()),                        \
    reinterpret_cast<half*>(c.data_ptr()),                        \
    M, N, K                                                       \
  );                                                              \
}

#define LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_K32_KERNEL(stages)\
{                                                              \
  const int smem_max_size = (                                  \
    (stages) * BM * (BK + A_PAD) * sizeof(half) +              \
    (stages) * BK * (BN + B_PAD) * sizeof(half));              \
  cudaFuncSetAttribute(                                        \
    hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_stages_dsmem_kernel< \
      WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,        \
      WARP_TILE_M, WARP_TILE_N, WARP_TILE_K,                   \
      A_PAD, B_PAD, (stages), false>,                          \
    cudaFuncAttributeMaxDynamicSharedMemorySize,               \
    98304);                                                    \
  dim3 block(NUM_THREADS);                                     \
  dim3 grid(div_ceil(N, BN), div_ceil(M, BM));                 \
  hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_stages_dsmem_kernel<   \
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,          \
    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K,                     \
    A_PAD, B_PAD, (stages), false><<<                          \
    grid, block, smem_max_size>>>(                             \
    reinterpret_cast<half*>(a.data_ptr()),                     \
    reinterpret_cast<half*>(b.data_ptr()),                     \
    reinterpret_cast<half*>(c.data_ptr()),                     \
    M, N, K                                                    \
  );                                                           \
}

// 128x128 warp2x4x2(32,64,32)
void hgemm_wmma_m16n16k16_mma4x2_warp2x4x2_stages_dsmem(
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
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2; 
  constexpr int WARP_TILE_M = 2;
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
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 2 * 4 * 32 = 256
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;    
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;    
  constexpr int BK = WMMA_K * WARP_TILE_K;   
  
  if (swizzle) {
    assert(swizzle_stride % 256 == 0);
    switch (stages)
    {
    case 2: 
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_K32_KERNEL(2, swizzle_stride);
      break;
    case 3: 
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_K32_KERNEL(3, swizzle_stride);
      break;
    case 4: 
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_K32_KERNEL(4, swizzle_stride);
      break;
    case 5: 
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_K32_KERNEL(5, swizzle_stride);
      break;
    default:
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_K32_KERNEL(2, swizzle_stride);
      break;
    }
  } else {
    switch (stages)
    {
    case 2:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_K32_KERNEL(2);
      break;
    case 3:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_K32_KERNEL(3);
      break;
    case 4:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_K32_KERNEL(4);
      break;
    case 5:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_K32_KERNEL(5);
      break;
    default:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_K32_KERNEL(2);
      break;
    }
  }
}

// 128x128 warp2x2x2(32,32,32) w dynamic smem, 98304=96KB < Ampere, Ada, Hopper ...
#define LAUNCH_161616_STAGE_SWIZZLE_DSMEM_K32_MMA4x4_KERNEL(stages, stride)\
{                                                                          \
  const int smem_max_size = (                                              \
    (stages) * BM * (BK + A_PAD) * sizeof(half) +                          \
    (stages) * BK * (BN + B_PAD) * sizeof(half));                          \
  cudaFuncSetAttribute(                                                    \
    hgemm_wmma_m16n16k16_mma4x4_warp2x2x2_stages_dsmem_kernel<             \
      WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,                    \
      WARP_TILE_M, WARP_TILE_N, WARP_TILE_K,                               \
      A_PAD, B_PAD, (stages), true>,                                       \
    cudaFuncAttributeMaxDynamicSharedMemorySize,                           \
    98304);                                                                \
  const int N_SWIZZLE = (N + (stride) - 1) / (stride);                     \
  dim3 block(NUM_THREADS);                                                 \
  dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE,                 \
             div_ceil(M, BM),                                              \
             N_SWIZZLE);                                                   \
  hgemm_wmma_m16n16k16_mma4x4_warp2x2x2_stages_dsmem_kernel<               \
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,                      \
    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K,                                 \
    A_PAD, B_PAD, (stages), true><<<                                       \
    grid, block, smem_max_size>>>(                                         \
    reinterpret_cast<half*>(a.data_ptr()),                                 \
    reinterpret_cast<half*>(b.data_ptr()),                                 \
    reinterpret_cast<half*>(c.data_ptr()),                                 \
    M, N, K                                                                \
  );                                                                       \
}

#define LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_K32_MMA4x4_KERNEL(stages) \
{                                                                      \
  const int smem_max_size = (                                          \
    (stages) * BM * (BK + A_PAD) * sizeof(half) +                      \
    (stages) * BK * (BN + B_PAD) * sizeof(half));                      \
  cudaFuncSetAttribute(                                                \
    hgemm_wmma_m16n16k16_mma4x4_warp2x2x2_stages_dsmem_kernel<         \
      WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,                \
      WARP_TILE_M, WARP_TILE_N, WARP_TILE_K,                           \
      A_PAD, B_PAD, (stages), false>,                                  \
    cudaFuncAttributeMaxDynamicSharedMemorySize,                       \
    98304);                                                            \
  dim3 block(NUM_THREADS);                                             \
  dim3 grid(div_ceil(N, BN), div_ceil(M, BM));                         \
  hgemm_wmma_m16n16k16_mma4x4_warp2x2x2_stages_dsmem_kernel<           \
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,                  \
    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K,                             \
    A_PAD, B_PAD, (stages), false><<<                                  \
    grid, block, smem_max_size>>>(                                     \
    reinterpret_cast<half*>(a.data_ptr()),                             \
    reinterpret_cast<half*>(b.data_ptr()),                             \
    reinterpret_cast<half*>(c.data_ptr()),                             \
    M, N, K                                                            \
  );                                                                   \
}

void hgemm_wmma_m16n16k16_mma4x4_warp2x2x2_stages_dsmem(
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
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 4; 
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 2;
  constexpr int WARP_TILE_K = 2;
  // s_a 4  ways bank conflicts within warp, after pad 8  -> 4 ways bank conflicts.
  // s_b 16 ways bank conflicts within warp, after pad 8  -> 8 ways bank conflicts.
  // s_b 16 ways bank conflicts within warp, after pad 16 -> 4 ways bank conflicts.
  // so, the best padding policy for s_a and s_b is A_PAD=0/8, B_PAD=16. Thus, 
  // improve B_PAD consume 8x~ less smem than A_PAD, 16xB_PAD vs 128xA_PAD.
  constexpr int A_PAD = 0;  // 0,8,16
  constexpr int B_PAD = 16; // 0,8,16
  constexpr int NUM_THREADS= (
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 4 * 32 = 512
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;    
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;    
  constexpr int BK = WMMA_K * WARP_TILE_K;   
  
  if (swizzle) {
    assert(swizzle_stride % 256 == 0);
    switch (stages)
    {
    case 2: 
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_K32_MMA4x4_KERNEL(2, swizzle_stride);
      break;
    case 3: 
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_K32_MMA4x4_KERNEL(3, swizzle_stride);
      break;
    case 4: 
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_K32_MMA4x4_KERNEL(4, swizzle_stride);
      break;
    case 5: 
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_K32_MMA4x4_KERNEL(5, swizzle_stride);
      break;
    default:
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_K32_MMA4x4_KERNEL(2, swizzle_stride);
      break;
    }
  } else {
    switch (stages)
    {
    case 2:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_K32_MMA4x4_KERNEL(2);
      break;
    case 3:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_K32_MMA4x4_KERNEL(3);
      break;
    case 4:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_K32_MMA4x4_KERNEL(4);
      break;
    case 5:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_K32_MMA4x4_KERNEL(5);
      break;
    default:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_K32_MMA4x4_KERNEL(2);
      break;
    }
  }
}

// 256x128 warp4x4(64,64,16) w dynamic smem, 98304=96KB < Ampere, Ada, Hopper ...
#define LAUNCH_161616_STAGE_SWIZZLE_DSMEM_WARP4X4_KERNEL(stages, stride)   \
{                                                                          \
  const int smem_max_size = (                                              \
    (stages) * BM * (BK + A_PAD) * sizeof(half) +                          \
    (stages) * BK * (BN + B_PAD) * sizeof(half));                          \
  cudaFuncSetAttribute(                                                    \
    hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem_kernel<               \
      WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,                    \
      WARP_TILE_M, WARP_TILE_N, WARP_TILE_K,                               \
      A_PAD, B_PAD, (stages), true>,                                       \
    cudaFuncAttributeMaxDynamicSharedMemorySize,                           \
    98304);                                                                \
  const int N_SWIZZLE = (N + (stride) - 1) / (stride);                     \
  dim3 block(NUM_THREADS);                                                 \
  dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE,                 \
             div_ceil(M, BM),                                              \
             N_SWIZZLE);                                                   \
  hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem_kernel<                 \
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,                      \
    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K,                                 \
    A_PAD, B_PAD, (stages), true><<<                                       \
    grid, block, smem_max_size>>>(                                         \
    reinterpret_cast<half*>(a.data_ptr()),                                 \
    reinterpret_cast<half*>(b.data_ptr()),                                 \
    reinterpret_cast<half*>(c.data_ptr()),                                 \
    M, N, K                                                                \
  );                                                                       \
}

#define LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_WARP4x4_KERNEL(stages)    \
{                                                                      \
  const int smem_max_size = (                                          \
    (stages) * BM * (BK + A_PAD) * sizeof(half) +                      \
    (stages) * BK * (BN + B_PAD) * sizeof(half));                      \
  cudaFuncSetAttribute(                                                \
    hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem_kernel<           \
      WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,                \
      WARP_TILE_M, WARP_TILE_N, WARP_TILE_K,                           \
      A_PAD, B_PAD, (stages), false>,                                  \
    cudaFuncAttributeMaxDynamicSharedMemorySize,                       \
    98304);                                                            \
  dim3 block(NUM_THREADS);                                             \
  dim3 grid(div_ceil(N, BN), div_ceil(M, BM));                         \
  hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem_kernel<             \
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,                  \
    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K,                             \
    A_PAD, B_PAD, (stages), false><<<                                  \
    grid, block, smem_max_size>>>(                                     \
    reinterpret_cast<half*>(a.data_ptr()),                             \
    reinterpret_cast<half*>(b.data_ptr()),                             \
    reinterpret_cast<half*>(c.data_ptr()),                             \
    M, N, K                                                            \
  );                                                                   \
}

void hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem(
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
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2; 
  constexpr int WARP_TILE_M = 4;
  constexpr int WARP_TILE_N = 4;
  constexpr int WARP_TILE_K = 1;
  // s_a 4  ways bank conflicts within warp, after pad 8  -> 4 ways bank conflicts.
  // s_b 16 ways bank conflicts within warp, after pad 8  -> 8 ways bank conflicts.
  // s_b 16 ways bank conflicts within warp, after pad 16 -> 4 ways bank conflicts.
  // so, the best padding policy for s_a and s_b is A_PAD=0/8, B_PAD=16. Thus, 
  // improve B_PAD consume 8x~ less smem than A_PAD, 16xB_PAD vs 128xA_PAD.
  constexpr int A_PAD = 0;  // 0,8,16
  constexpr int B_PAD = 16; // 0,8,16
  constexpr int NUM_THREADS= (
    WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 2 * 32 = 256
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;    
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;    
  constexpr int BK = WMMA_K * WARP_TILE_K;   
  
  if (swizzle) {
    assert(swizzle_stride % 256 == 0);
    switch (stages)
    {
    case 2: 
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_WARP4X4_KERNEL(2, swizzle_stride);
      break;
    case 3: 
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_WARP4X4_KERNEL(3, swizzle_stride);
      break;
    case 4: 
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_WARP4X4_KERNEL(4, swizzle_stride);
      break;
    case 5: 
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_WARP4X4_KERNEL(5, swizzle_stride);
      break;
    default:
      LAUNCH_161616_STAGE_SWIZZLE_DSMEM_WARP4X4_KERNEL(2, swizzle_stride);
      break;
    }
  } else {
    switch (stages)
    {
    case 2:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_WARP4x4_KERNEL(2);
      break;
    case 3:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_WARP4x4_KERNEL(3);
      break;
    case 4:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_WARP4x4_KERNEL(4);
      break;
    case 5:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_WARP4x4_KERNEL(5);
      break;
    default:
      LAUNCH_161616_STAGE_NO_SWIZZLE_DSMEM_WARP4x4_KERNEL(2);
      break;
    }
  }
}