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

// -------------------------------------- FP32 -------------------------------------- 
// Warp Reduce Sum
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Block reduce sum/max/min device helper for Layer/RMS Norm/Softmax etc.
// grid 1D block 1D, grid(N/256), block(256)
template<const int NUM_THREADS=256>
__device__ __forceinline__ float block_reduce_sum_f32(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];
  
  val = warp_reduce_sum_f32<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = val;
  __syncthreads();
  val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  val = warp_reduce_sum_f32<NUM_WARPS>(val);
  return val;
}

// Softmax x: N, y: N
// grid(N/256), block(K=256)
template<const int NUM_THREADS = 256>
__global__ void softmax_f32(float* x, float* y, float* total, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid; 
  
  float exp_val = (idx < N) ? expf(x[idx]) : 0.0f;
  float sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
  // get the total sum of all blocks.
  if (tid == 0) atomicAdd(total, sum);
  __threadfence(); // grid level memory fence
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if (idx < N) y[idx] = exp_val / (*total); 
}

// Softmax Vec4 x: N, y: N
// grid(N/256), block(256/4)
template<const int NUM_THREADS = 256/4>
__global__ void softmax_f32x4(float* x, float* y, float* total, int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * blockDim.x + tid) * 4; 
  
  float4 reg_x = FLOAT4(x[idx]);
  float4 reg_exp;
  reg_exp.x = (idx < N) ? expf(reg_x.x) : 0.0f;
  reg_exp.y = (idx < N) ? expf(reg_x.y) : 0.0f;
  reg_exp.z = (idx < N) ? expf(reg_x.z) : 0.0f;
  reg_exp.w = (idx < N) ? expf(reg_x.w) : 0.0f;
  float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
  float sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
  // get the total sum of all blocks.
  if (tid == 0) atomicAdd(total, sum);
  __threadfence(); // grid level memory fence
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

// TODO: support per-token w/o __threadfence