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

#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))


template<const int BM=64, const int BN=64, const int BK=16, 
         const int TM=8,  const int TN=4,  const int OFFSET=0>
__global__ void sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_kernel(
  float* a, float* b, float* c, const int M, const int N, const int K) {
  // block(BN/TN, BM/TM) -> (x=16,y=8), 128 threads
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  // 2*(16*64*4)=8KB, 8+8=16KB, 128KB/16=8 blocks
  __shared__ float s_a[2][BK][BM + OFFSET];
  __shared__ float s_b[2][BK][BN + OFFSET];
  
  float r_load_a[8]; // load 8 values per thread
  float r_comp_a[TM];
  float r_comp_b[TN];
  float r_c[TM][TN] = {0.0}; // 8x4
  
  // 128 threads, tx: 0~15, ty: 0~7
  int load_a_smem_m = tid / 2; // (0,1,2,...,63)
  int load_a_smem_k = (tid % 2 == 0) ? 0 : 8; // (0,8)
  int load_b_smem_k = tid / 8; // 0~15
  int load_b_smem_n = (tid % 8) * 8; // (0,8,16,...,56)
  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  {
    int load_a_gmem_k = load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

    #pragma unroll
    for(int i = 0; i < 8; i += 4) {
      FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n + i]) = (
        FLOAT4(b[load_b_gmem_addr + i]));
      FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
    }
    #pragma unroll
    for(int i = 0; i < 8; ++i) {
      s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
    }
  }
  __syncthreads(); 

  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {

    int smem_sel = (bk - 1) & 1;
    int smem_sel_next = bk & 1;

    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    #pragma unroll
    for(int i = 0; i < 8; i += 4) {
      FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n + i]) = (
        FLOAT4(b[load_b_gmem_addr + i]));
      FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
    }
   
    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      // ty: 0~7, (0~7)*8=64; tx: 0~15, (0~15)*4=64
      FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM     ]);
      FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM + 4 ]);
      FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN     ]);

      #pragma unroll
      for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }

    #pragma unroll
    for(int i = 0; i < 8; ++i) {
      s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
    }
    
    __syncthreads();
  }
  
  #pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM     ]);
    FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM + 4 ]);
    FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN     ]);

    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
      #pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }

  #pragma unroll
  for (int i = 0; i < TM; i++) {
    int store_c_gmem_m = by * BM + ty * TM + i;
    int store_c_gmem_n = bx * BN + tx * TN;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
  }
}


template<const int BM=64, const int BN=64, const int BK=16, 
         const int TM=8,  const int TN=4,  const int OFFSET=0>
__global__ void sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async_kernel(
  float* a, float* b, float* c, const int M, const int N, const int K) {
  // block(BN/TN, BM/TM) -> (x=16,y=8), 128 threads
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  // 2*(16*64*4)=8KB, 8+8=16KB, 128KB/16=8 blocks
  __shared__ float s_a[2][BK][BM + OFFSET];
  __shared__ float s_b[2][BK][BN + OFFSET];
  
  float r_load_a[8]; // load 8 values per thread
  float r_comp_a[TM];
  float r_comp_b[TN];
  float r_c[TM][TN] = {0.0}; // 8x4
  
  // 128 threads, tx: 0~15, ty: 0~7
  int load_a_smem_m = tid / 2; // (0,1,2,...,63)
  int load_a_smem_k = (tid % 2 == 0) ? 0 : 8; // (0,8)
  int load_b_smem_k = tid / 8; // 0~15
  int load_b_smem_n = (tid % 8) * 8; // (0,8,16,...,56)
  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  {
    int load_a_gmem_k = load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    uint32_t load_b_smem_ptr = __cvta_generic_to_shared(
      &s_b[0][load_b_smem_k][load_b_smem_n]);
    // 2 cp.async issue, 16 bytes = 4 float.
    #pragma unroll
    for(int i = 0; i < 8; i += 4) {
      CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();

    #pragma unroll
    for(int i = 0; i < 8; i += 4) {
      FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
    }
    #pragma unroll
    for(int i = 0; i < 8; ++i) {
      s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
    }
    CP_ASYNC_WAIT_GROUP(0);
  }
  __syncthreads(); 

  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {

    int smem_sel = (bk - 1) & 1;
    int smem_sel_next = bk & 1;

    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

    uint32_t load_b_smem_ptr = __cvta_generic_to_shared(
      &s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]);
    // 2 cp.async issue, 16 bytes = 4 float.
    #pragma unroll
    for(int i = 0; i < 8; i += 4) {
      CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();

    #pragma unroll
    for(int i = 0; i < 8; i += 4) {
      FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
    }
    
    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM     ]);
      FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM + 4 ]);
      FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN     ]);

      #pragma unroll
      for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }

    #pragma unroll
    for(int i = 0; i < 8; ++i) {
      s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
    }

    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }
  
  #pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM     ]);
    FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM + 4 ]);
    FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN     ]);

    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
      #pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }

  #pragma unroll
  for (int i = 0; i < TM; i++) {
    int store_c_gmem_m = by * BM + ty * TM + i;
    int store_c_gmem_n = bx * BN + tx * TN;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
  }
}


template<const int BM=128, const int BN=128, const int BK=16, 
         const int TM=8,   const int TN=8,   const int OFFSET=0>
__global__ void sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_kernel(
  float* a, float* b, float* c, const int M, const int N, const int K) {
  // block(BN/TN, BM/TM) -> (x=16,y=16), 256 threads
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ float s_a[2][BK][BM + OFFSET];
  __shared__ float s_b[2][BK][BN + OFFSET];
  
  float r_load_a[8]; // load 8 values per thread
  float r_comp_a[TM];
  float r_comp_b[TN];
  float r_c[TM][TN] = {0.0}; // 8x8
  
  // 256 threads, tx: 0~15, ty: 0~7
  int load_a_smem_m = tid / 2; // (0,1,2,...,128)
  int load_a_smem_k = (tid % 2 == 0) ? 0 : 8; // (0,8)
  int load_b_smem_k = tid / 16; // 0~15
  int load_b_smem_n = (tid % 16) * 8; // (0,8,16,...,128)
  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  {
    int load_a_gmem_k = load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    
    #pragma unroll
    for(int i = 0; i < 8; i += 4) {
      FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n + i]) = (
        FLOAT4(b[load_b_gmem_addr + i]));
      FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
    }
    #pragma unroll
    for(int i = 0; i < 8; ++i) {
      s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
    }
  }
  __syncthreads(); 

  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {

    int smem_sel = (bk - 1) & 1;
    int smem_sel_next = bk & 1;

    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

    #pragma unroll
    for(int i = 0; i < 8; i += 4) {
      FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n + i]) = (
        FLOAT4(b[load_b_gmem_addr + i]));
      FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
    }
    
    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2         ]);
      FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
      FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2         ]);
      FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

      #pragma unroll
      for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }

    #pragma unroll
    for(int i = 0; i < 8; ++i) {
      s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
    }

    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }
  
  #pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2         ]);
    FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
    FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2         ]);
    FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][tx * TN / 2 + BN / 2]);

    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
      #pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }

  #pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr         ]) = FLOAT4(r_c[i][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
  }
  #pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr         ]) = FLOAT4(r_c[i + TM / 2][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
  }
}


template<const int BM=128, const int BN=128, const int BK=16, 
         const int TM=8,   const int TN=8,   const int OFFSET=0>
__global__ void sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async_kernel(
  float* a, float* b, float* c, const int M, const int N, const int K) {
  // block(BN/TN, BM/TM) -> (x=16,y=16), 256 threads
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ float s_a[2][BK][BM + OFFSET];
  __shared__ float s_b[2][BK][BN + OFFSET];
  
  float r_load_a[8]; // load 8 values per thread
  float r_comp_a[TM];
  float r_comp_b[TN];
  float r_c[TM][TN] = {0.0}; // 8x8
  
  // 256 threads, tx: 0~15, ty: 0~7
  int load_a_smem_m = tid / 2; // (0,1,2,...,128)
  int load_a_smem_k = (tid % 2 == 0) ? 0 : 8; // (0,8)
  int load_b_smem_k = tid / 16; // 0~15
  int load_b_smem_n = (tid % 16) * 8; // (0,8,16,...,128)
  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  {
    int load_a_gmem_k = load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    uint32_t load_b_smem_ptr = __cvta_generic_to_shared(
      &s_b[0][load_b_smem_k][load_b_smem_n]);
    // 2 cp.async issue, 16 bytes = 4 float.
    #pragma unroll
    for(int i = 0; i < 8; i += 4) {
      CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();

    #pragma unroll
    for(int i = 0; i < 8; i += 4) {
      FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
    }
    #pragma unroll
    for(int i = 0; i < 8; ++i) {
      s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
    }
    CP_ASYNC_WAIT_GROUP(0);
  }
  __syncthreads(); 

  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {

    int smem_sel = (bk - 1) & 1;
    int smem_sel_next = bk & 1;

    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

    uint32_t load_b_smem_ptr = __cvta_generic_to_shared(
      &s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]);
    // 2 cp.async issue, 16 bytes = 4 float.
    #pragma unroll
    for(int i = 0; i < 8; i += 4) {
      CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();

    #pragma unroll
    for(int i = 0; i < 8; i += 4) {
      FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
    }
    
    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2         ]);
      FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
      FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2         ]);
      FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

      #pragma unroll
      for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }

    #pragma unroll
    for(int i = 0; i < 8; ++i) {
      s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
    }

    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }
  
  #pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2         ]);
    FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
    FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2         ]);
    FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][tx * TN / 2 + BN / 2]);

    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
      #pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }

  #pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr         ]) = FLOAT4(r_c[i][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
  }
  #pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr         ]) = FLOAT4(r_c[i + TM / 2][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
  }
}

// t 8x16, k 16, w/o async
template<const int BM=128, const int BN=256, const int BK=16, 
         const int TM=8,   const int TN=16,  const int OFFSET=0>
__global__ void sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_kernel(
  float* a, float* b, float* c, const int M, const int N, const int K) {
  // block(BN/TN, BM/TM) -> (x=16,y=16), 256 threads
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  // (16*128*4)=8KB (16*256*4)=16KB 24KB
  __shared__ float s_a[2][BK][BM + OFFSET];
  __shared__ float s_b[2][BK][BN + OFFSET];
  
  float r_load_a[8];
  float r_comp_a[TM]; // 8
  float r_comp_b[TN]; // 16
  float r_c[TM][TN] = {0.0}; // 8x16
  
  // 256 threads, tx: 0~15, ty: 0~7
  // a: load 8 values per thread
  int load_a_smem_m = tid / 2; // (0,1,2,...,128)
  int load_a_smem_k = (tid % 2 == 0) ? 0 : 8; // (0,8)
  // b: load 16 values per thread
  int load_b_smem_k = tid / 16; // 0~15
  int load_b_smem_n = (tid % 16) * 16; // (0,16,32,...,256)
  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  {
    // gmem -> smem
    int load_a_gmem_k = load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

    // b: load 16 values per thread
    #pragma unroll
    for(int i = 0; i < 16; i += 4) {
      FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n + i]) = (
        FLOAT4(b[load_b_gmem_addr + i]));
    }
    // a: load 8 values per thread
    #pragma unroll
    for(int i = 0; i < 8; i += 4) {
      FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
    }
    #pragma unroll
    for(int i = 0; i < 8; ++i) {
      // online transpose
      s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
    }
  }
  __syncthreads(); 

  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
    // gmem -> smem
    int smem_sel = (bk - 1) & 1;
    int smem_sel_next = bk & 1;

    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    
    // b: load 16 values per thread
    #pragma unroll
    for(int i = 0; i < 16; i += 4) {
      FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n + i]) = (
        FLOAT4(b[load_b_gmem_addr + i]));
    }
    
    // a: load 8 values per thread
    #pragma unroll
    for(int i = 0; i < 8; i += 4) {
      FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
    }
    
    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      // smem -> regristers  
      #pragma unroll
      for (int r = 0; r < 8; r += 4) {
        FLOAT4(r_comp_a[r]) = FLOAT4(s_a[smem_sel][tk][ty * TM + r]);
      }
      #pragma unroll
      for (int r = 0; r < 16; r += 4) {
        FLOAT4(r_comp_b[r]) = FLOAT4(s_b[smem_sel][tk][tx * TN + r]);
      }
    
      #pragma unroll
      for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }
    
    #pragma unroll
    for(int i = 0; i < 8; ++i) {
      // online transpose
      s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
    }
    __syncthreads();
  }
  
  #pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    // smem -> regristers  
    #pragma unroll
    for (int r = 0; r < 8; r += 4) {
      FLOAT4(r_comp_a[r]) = FLOAT4(s_a[1][tk][ty * TM + r]);
    }
    #pragma unroll
    for (int r = 0; r < 16; r += 4) {
      FLOAT4(r_comp_b[r]) = FLOAT4(s_b[1][tk][tx * TN + r]);
    }
    
    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
      #pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }

  #pragma unroll
  for (int m = 0; m < TM; m++) {
    #pragma unroll
    for (int n = 0; n < TN; n += 4) {
      int store_c_gmem_m = by * BM + ty * TM + m;
      int store_c_gmem_n = bx * BN + tx * TN + n;
      int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
      FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[m][n]);
    }
  }
}

// t 8x16, k 16, async
template<const int BM=128, const int BN=256, const int BK=16, 
         const int TM=8,   const int TN=16,  const int OFFSET=0>
__global__ void sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async_kernel(
  float* a, float* b, float* c, const int M, const int N, const int K) {
  // block(BN/TN, BM/TM) -> (x=16,y=16), 256 threads
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  // (16*128*4)=8KB (16*256*4)=16KB 24KB
  __shared__ float s_a[2][BK][BM + OFFSET];
  __shared__ float s_b[2][BK][BN + OFFSET];
  
  float r_load_a[8];
  float r_comp_a[TM]; // 8
  float r_comp_b[TN]; // 16
  float r_c[TM][TN] = {0.0}; // 8x16
  
  // 256 threads, tx: 0~15, ty: 0~7
  // a: load 8 values per thread
  int load_a_smem_m = tid / 2; // (0,1,2,...,128)
  int load_a_smem_k = (tid % 2 == 0) ? 0 : 8; // (0,8)
  // b: load 16 values per thread
  int load_b_smem_k = tid / 16; // 0~15
  int load_b_smem_n = (tid % 16) * 16; // (0,16,32,...,256)
  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  {
    // gmem -> smem
    int load_a_gmem_k = load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

    // b: load 16 values per thread
    uint32_t load_b_smem_ptr = __cvta_generic_to_shared(
      &s_b[0][load_b_smem_k][load_b_smem_n]);
    // 4 cp.async issue, 16 bytes = 4 float, 4x4=16
    #pragma unroll
    for(int i = 0; i < 16; i += 4) {
      CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();
    
    // a: load 8 values per thread
    #pragma unroll
    for(int i = 0; i < 8; i += 4) {
      FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
    }
    #pragma unroll
    for(int i = 0; i < 8; ++i) {
      // online transpose
      s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
    }
    CP_ASYNC_WAIT_GROUP(0);
  }
  __syncthreads(); 

  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
    // gmem -> smem
    int smem_sel = (bk - 1) & 1;
    int smem_sel_next = bk & 1;

    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    
    // b: load 16 values per thread
    uint32_t load_b_smem_ptr = __cvta_generic_to_shared(
      &s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]);
    // 4 cp.async issue, 16 bytes = 4 float, 4x4=16
    #pragma unroll
    for(int i = 0; i < 16; i += 4) {
      CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();
    
    // a: load 8 values per thread
    #pragma unroll
    for(int i = 0; i < 8; i += 4) {
      FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
    }
    
    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      // smem -> regristers  
      #pragma unroll
      for (int r = 0; r < 8; r += 4) {
        FLOAT4(r_comp_a[r]) = FLOAT4(s_a[smem_sel][tk][ty * TM + r]);
      }
      #pragma unroll
      for (int r = 0; r < 16; r += 4) {
        FLOAT4(r_comp_b[r]) = FLOAT4(s_b[smem_sel][tk][tx * TN + r]);
      }
    
      #pragma unroll
      for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }
    
    #pragma unroll
    for(int i = 0; i < 8; ++i) {
      // online transpose
      s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
    }

    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }
  
  #pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    // smem -> regristers  
    #pragma unroll
    for (int r = 0; r < 8; r += 4) {
      FLOAT4(r_comp_a[r]) = FLOAT4(s_a[1][tk][ty * TM + r]);
    }
    #pragma unroll
    for (int r = 0; r < 16; r += 4) {
      FLOAT4(r_comp_b[r]) = FLOAT4(s_b[1][tk][tx * TN + r]);
    }
    
    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
      #pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }

  #pragma unroll
  for (int m = 0; m < TM; m++) {
    #pragma unroll
    for (int n = 0; n < TN; n += 4) {
      int store_c_gmem_m = by * BM + ty * TM + m;
      int store_c_gmem_n = bx * BN + tx * TN + n;
      int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
      FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[m][n]);
    }
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

// 8x4, k16
void sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 16; 
  constexpr int TM = 8;
  constexpr int TN = 4;

  dim3 block(BN/TN, BM/TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_kernel<
    BM, BN, BK, TM, TN><<<grid, block>>>(
    reinterpret_cast<float*>(a.data_ptr()),
    reinterpret_cast<float*>(b.data_ptr()),
    reinterpret_cast<float*>(c.data_ptr()),
    M, N, K
  );
}

void sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 16; 
  constexpr int TM = 8;
  constexpr int TN = 4;

  dim3 block(BN/TN, BM/TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async_kernel<
    BM, BN, BK, TM, TN><<<grid, block>>>(
    reinterpret_cast<float*>(a.data_ptr()),
    reinterpret_cast<float*>(b.data_ptr()),
    reinterpret_cast<float*>(c.data_ptr()),
    M, N, K
  );
}

// 8x8, k16
void sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
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
  constexpr int BK = 16; 
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN/TN, BM/TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_kernel<
    BM, BN, BK, TM, TN><<<grid, block>>>(
    reinterpret_cast<float*>(a.data_ptr()),
    reinterpret_cast<float*>(b.data_ptr()),
    reinterpret_cast<float*>(c.data_ptr()),
    M, N, K
  );
}


void sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
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
  constexpr int BK = 16; 
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN/TN, BM/TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async_kernel<
    BM, BN, BK, TM, TN><<<grid, block>>>(
    reinterpret_cast<float*>(a.data_ptr()),
    reinterpret_cast<float*>(b.data_ptr()),
    reinterpret_cast<float*>(c.data_ptr()),
    M, N, K
  );
}

// 8x16, k16
void sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
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
  constexpr int BN = 256;
  constexpr int BK = 16; 
  constexpr int TM = 8;
  constexpr int TN = 16;

  dim3 block(BN/TN, BM/TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_kernel<
    BM, BN, BK, TM, TN><<<grid, block>>>(
    reinterpret_cast<float*>(a.data_ptr()),
    reinterpret_cast<float*>(b.data_ptr()),
    reinterpret_cast<float*>(c.data_ptr()),
    M, N, K
  );
}


void sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
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
  constexpr int BN = 256;
  constexpr int BK = 16; 
  constexpr int TM = 8;
  constexpr int TN = 16;

  dim3 block(BN/TN, BM/TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async_kernel<
    BM, BN, BK, TM, TN><<<grid, block>>>(
    reinterpret_cast<float*>(a.data_ptr()),
    reinterpret_cast<float*>(b.data_ptr()),
    reinterpret_cast<float*>(c.data_ptr()),
    M, N, K
  );
}
