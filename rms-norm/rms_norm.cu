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

// RMS Norm: x: NxK(K=256<1024), y': NxK, y'=x/rms(x) each row
// 1/rms(x) = rsqrtf( sum(x^2)/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g (g: scale)
template<const int NUM_THREADS=256>
__global__ void rms_norm_f32(float* x, float* y, float g, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x; // 0..N-1
  int idx = bid * blockDim.x + threadIdx.x;
  const float epsilon = 1e-5f;

  __shared__ float s_variance; // shared within block
  float value = (idx < N * K) ? x[idx] : 0.0f; // load once only
  float variance = value * value;
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float) K + epsilon);
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads(); 
  if (idx < N * K) y[idx] = (value * s_variance) * g;
}

// RMS Norm Vec4: x: NxK(K=256<1024), y': NxK, y'=x/rms(x) each row
// 1/rms(x) = rsqrtf( sum(x^2)/K ) each row
// grid(N*K/K), block(K/4<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g (g: scale)
template<const int NUM_THREADS=256/4>
__global__ void rms_norm_f32x4(float* x, float* y, float g, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x; // 0..N-1
  int idx = (bid * blockDim.x + threadIdx.x) * 4;
  const float epsilon = 1e-5f;

  __shared__ float s_variance; // shared within block
  float4 reg_x = FLOAT4(x[idx]);
  float variance = (idx < N * K) ? (reg_x.x * reg_x.x + reg_x.y * reg_x.y 
                                  + reg_x.z * reg_x.z + reg_x.w * reg_x.w) : 0.0f;
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
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
