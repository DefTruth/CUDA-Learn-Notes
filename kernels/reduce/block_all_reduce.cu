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
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

// FP16/BF16 CUDA Cores/Tensor Cores: 
// https://resources.nvidia.com/en-us-tensor-core 
// Non MatMul FP16/BF16 -> CUDA Cores
//     MatMul FP16/BF16 -> Tensor Cores
//       Non MatMul FP8 -> Not supported
//           MatMul FP8 -> Tensor Cores

// CUDA温故(0x00): 一步步学习block all reduce: 从FP32到FP16/BF16，再到FP8
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

// Block All Reduce Sum
// grid(N/256), block(256)
// a: Nx1, y=sum(a)
template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  // keep the data in register is enough for warp operaion.
  float sum = (idx < N) ? a[idx] : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
  // warp leaders store the data to shared memory.
  if (lane == 0) reduce_smem[warp] = sum;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, sum);
}

// Block All Reduce Sum + float4
// grid(N/256), block(256/4)
// a: Nx1, y=sum(a)
template<const int NUM_THREADS = 256/4>
__global__ void block_all_reduce_sum_f32x4_f32_kernel(float* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  float4 reg_a = FLOAT4(a[idx]);
  // keep the data in register is enough for warp operaion.
  float sum = (idx < N) ? (reg_a.x + reg_a.y + reg_a.z + reg_a.w) : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
  // warp leaders store the data to shared memory.
  if (lane == 0) reduce_smem[warp] = sum;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, sum);
}

// -------------------------------------- FP16 -------------------------------------- 
// Warp Reduce Sum: Half
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
    // val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {
  float val_f32 = __half2float(val);
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
  }
  return val_f32;
}

// Block All Reduce Sum: Half
// grid(N/256), block(256)
// a: Nx1, y=sum(a)
template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f16_f16_kernel(half* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  // keep the data in register is enough for warp operaion.
  half sum_f16 = (idx < N) ? a[idx] : __float2half(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
  // warp leaders store the data to shared memory.
  // use float to keep sum from each block and reduce 
  // with fp32 inter warps.
  if (lane == 0) reduce_smem[warp] = __half2float(sum_f16);
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, sum);
}

template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f16_f32_kernel(half* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  // keep the data in register is enough for warp operaion.
  half sum_f16 = (idx < N) ? a[idx] : __float2half(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  float sum_f32 = warp_reduce_sum_f16_f32<WARP_SIZE>(sum_f16);
  // warp leaders store the data to shared memory.
  // use float to keep sum from each block and reduce 
  // with fp32 inter warps.
  if (lane == 0) reduce_smem[warp] = sum_f32;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, sum);
}

template<const int NUM_THREADS = 256/2>
__global__ void block_all_reduce_sum_f16x2_f32_kernel(half* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 2; // 2 half elements per thread
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  // keep the data in register is enough for warp operaion.
  half2 reg_a = HALF2(a[idx]);
  half sum_f16 = (idx < N) ? __hadd(reg_a.x, reg_a.y) : __float2half(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  float sum_f32 = warp_reduce_sum_f16_f32<WARP_SIZE>(sum_f16);
  // warp leaders store the data to shared memory.
  // use float to keep sum from each block and reduce 
  // with fp32 inter warps.
  if (lane == 0) reduce_smem[warp] = sum_f32;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, sum);
}

template<const int NUM_THREADS = 256/2>
__global__ void block_all_reduce_sum_f16x2_f16_kernel(half* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 2; // 2 half elements per thread
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  // keep the data in register is enough for warp operaion.
  half2 reg_a = HALF2(a[idx]);
  half sum_f16 = (idx < N) ? __hadd(reg_a.x, reg_a.y) : __float2half(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
  // warp leaders store the data to shared memory.
  // use float to keep sum from each block and reduce 
  // with fp32 inter warps.
  if (lane == 0) reduce_smem[warp] = __half2float(sum_f16);
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, sum);
}

template<const int NUM_THREADS = 256/8>
__global__ void block_all_reduce_sum_f16x8_pack_f16_kernel(half* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 8; // 8 half elements per thread
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  // temporary register(memory), .local space in ptx, addressable
  half pack_a[8]; // 8x16 bits=128 bits.
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
  const half z = __float2half(0.0f);

  half sum_f16 = z;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    sum_f16 += (((idx + i ) < N) ? pack_a[i] : z);
  }

  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
  // warp leaders store the data to shared memory.
  // use float to keep sum from each block and reduce 
  // with fp32 inter warps.
  if (lane == 0) reduce_smem[warp] = __half2float(sum_f16);
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, sum);
}

template<const int NUM_THREADS = 256/8>
__global__ void block_all_reduce_sum_f16x8_pack_f32_kernel(half* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 8; // 8 half elements per thread
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  // temporary register(memory), .local space in ptx, addressable
  half pack_a[8]; // 8x16 bits=128 bits.
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits

  float sum_f32 = 0.0f;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    sum_f32 += (((idx + i ) < N) ? __half2float(pack_a[i]) : 0.0f);
  }

  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum_f32 = warp_reduce_sum_f32<WARP_SIZE>(sum_f32);
  // warp leaders store the data to shared memory.
  // use float to keep sum from each block and reduce 
  // with fp32 inter warps.
  if (lane == 0) reduce_smem[warp] = sum_f32;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, sum);
}

// -------------------------------------- BF16 -------------------------------------- 
// Warp Reduce Sum: Half
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ __nv_bfloat16 warp_reduce_sum_bf16_bf16(
  __nv_bfloat16 val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_bf16_f32(
  __nv_bfloat16 val) {
  float val_f32 = __bfloat162float(val);
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
  }
  return val_f32;
}

// Block All Reduce Sum: BF16
// grid(N/256), block(256)
// a: Nx1, y=sum(a)
template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_bf16_bf16_kernel(
  __nv_bfloat16* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ __nv_bfloat16 reduce_smem[NUM_WARPS];

  // keep the data in register is enough for warp operaion.
  __nv_bfloat16 sum_bf16 = (idx < N) ? a[idx] : __float2bfloat16(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum_bf16 = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum_bf16);
  // warp leaders store the data to shared memory.
  // use float to keep sum from each block and reduce 
  // with fp32 inter warps.
  if (lane == 0) reduce_smem[warp] = sum_bf16;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  __nv_bfloat16 sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2bfloat16(0.0f);
  if (warp == 0) sum = warp_reduce_sum_bf16_bf16<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, __bfloat162float(sum));
}

template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_bf16_f32_kernel(
  __nv_bfloat16* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  // keep the data in register is enough for warp operaion.
  __nv_bfloat16 sum_bf16 = (idx < N) ? a[idx] : __float2bfloat16(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  float sum_f32 = warp_reduce_sum_bf16_f32<WARP_SIZE>(sum_bf16);
  // warp leaders store the data to shared memory.
  // use float to keep sum from each block and reduce 
  // with fp32 inter warps.
  if (lane == 0) reduce_smem[warp] = sum_f32;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, sum);
}

template<const int NUM_THREADS = 256/2>
__global__ void block_all_reduce_sum_bf16x2_bf16_kernel(
  __nv_bfloat16* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 2; // 2 bf16 elements per thread
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ __nv_bfloat16 reduce_smem[NUM_WARPS];

  // keep the data in register is enough for warp operaion.
  __nv_bfloat162 reg_a = BFLOAT2(a[idx]);
  __nv_bfloat16 sum_bf16 = (idx < N) ? __hadd(reg_a.x, reg_a.y) : __float2bfloat16(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum_bf16 = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum_bf16);
  // warp leaders store the data to shared memory.
  // use float to keep sum from each block and reduce 
  // with fp32 inter warps.
  if (lane == 0) reduce_smem[warp] = sum_bf16;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  __nv_bfloat16 sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2bfloat16(0.0f);
  if (warp == 0) sum = warp_reduce_sum_bf16_bf16<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, __bfloat162float(sum));
}

template<const int NUM_THREADS = 256/2>
__global__ void block_all_reduce_sum_bf16x2_f32_kernel(
  __nv_bfloat16* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 2; // 2 bf16 elements per thread
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  // keep the data in register is enough for warp operaion.
  __nv_bfloat162 reg_a = BFLOAT2(a[idx]);
  __nv_bfloat16 sum_bf16 = (idx < N) ? __hadd(reg_a.x, reg_a.y) : __float2bfloat16(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  float sum_f32 = warp_reduce_sum_bf16_f32<WARP_SIZE>(sum_bf16);
  // warp leaders store the data to shared memory.
  // use float to keep sum from each block and reduce 
  // with fp32 inter warps.
  if (lane == 0) reduce_smem[warp] = sum_f32;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, sum);
}

template<const int NUM_THREADS = 256/8>
__global__ void block_all_reduce_sum_bf16x8_pack_bf16_kernel(
  __nv_bfloat16* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 8; // 8 bf16 elements per thread
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ __nv_bfloat16 reduce_smem[NUM_WARPS];
  // temporary register(memory), .local space in ptx, addressable
  __nv_bfloat16 pack_a[8]; // 8x16 bits=128 bits.
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
  const __nv_bfloat16 z = __float2bfloat16(0.0f);

  __nv_bfloat16 sum_bf16 = z;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    sum_bf16 += (((idx + i ) < N) ? pack_a[i] : z);
  }

  // keep the data in register is enough for warp operaion.
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum_bf16 = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum_bf16);
  // warp leaders store the data to shared memory.
  // use float to keep sum from each block and reduce 
  // with fp32 inter warps.
  if (lane == 0) reduce_smem[warp] = sum_bf16;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  __nv_bfloat16 sum = (lane < NUM_WARPS) ? reduce_smem[lane] : z;
  if (warp == 0) sum = warp_reduce_sum_bf16_bf16<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, __bfloat162float(sum));
}

template<const int NUM_THREADS = 256/8>
__global__ void block_all_reduce_sum_bf16x8_pack_f32_kernel(
  __nv_bfloat16* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 8; // 8 bf16 elements per thread
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  // temporary register(memory), .local space in ptx, addressable
  __nv_bfloat16 pack_a[8]; // 8x16 bits=128 bits.
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
  const __nv_bfloat16 z = __float2bfloat16(0.0f);

  __nv_bfloat16 sum_bf16 = z;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    sum_bf16 += (((idx + i ) < N) ? pack_a[i] : z);
  }

  // keep the data in register is enough for warp operaion.
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  float sum_f32 = warp_reduce_sum_bf16_f32<WARP_SIZE>(sum_bf16);
  // warp leaders store the data to shared memory.
  // use float to keep sum from each block and reduce 
  // with fp32 inter warps.
  if (lane == 0) reduce_smem[warp] = sum_f32;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, sum);
}

// -------------------------------------- FP8 -------------------------------------- 
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_fp8_e4m3_f16(
  __nv_fp8_storage_t val) {
  // typedef unsigned char __nv_fp8_storage_t;
  // __half &operator=(const __half_raw &hr);
  half val_f16 = __nv_cvt_fp8_to_halfraw(val, __NV_E4M3);
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_f16 = __hadd(val_f16, __shfl_xor_sync(0xffffffff, val_f16, mask));
  }
  return val_f16;
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_fp8_e5m2_f16(
  __nv_fp8_storage_t val) {
  // typedef unsigned char __nv_fp8_storage_t;
  // __half &operator=(const __half_raw &hr);
  half val_f16 = __nv_cvt_fp8_to_halfraw(val, __NV_E5M2);
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_f16 = __hadd(val_f16, __shfl_xor_sync(0xffffffff, val_f16, mask));
  }
  return val_f16;
}

template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_fp8_e4m3_f16_kernel(
  __nv_fp8_storage_t* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ half reduce_smem[NUM_WARPS];

  // keep the data in register is enough for warp operaion.
  __nv_fp8_storage_t sum_f8 = (idx < N) ? a[idx] : __nv_cvt_float_to_fp8(
    0.0f, __NV_SATFINITE, __NV_E4M3);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  half sum_f16 = warp_reduce_sum_fp8_e4m3_f16<WARP_SIZE>(sum_f8);
  // warp leaders store the data to shared memory.
  // use float to keep sum from each block and reduce 
  // with fp16 inter warps.
  if (lane == 0) reduce_smem[warp] = sum_f16;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  half sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2half(0.0f);
  if (warp == 0) sum = warp_reduce_sum_f16_f16<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, __half2float(sum));
}

template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_fp8_e5m2_f16_kernel(
  __nv_fp8_storage_t* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ half reduce_smem[NUM_WARPS];

  // keep the data in register is enough for warp operaion.
  __nv_fp8_storage_t sum_f8 = (idx < N) ? a[idx] : __nv_cvt_float_to_fp8(
    0.0f, __NV_SATFINITE, __NV_E5M2);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  half sum_f16 = warp_reduce_sum_fp8_e5m2_f16<WARP_SIZE>(sum_f8);
  // warp leaders store the data to shared memory.
  // use float to keep sum from each block and reduce 
  // with fp16 inter warps.
  if (lane == 0) reduce_smem[warp] = sum_f16;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  half sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2half(0.0f);
  if (warp == 0) sum = warp_reduce_sum_f16_f16<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, __half2float(sum));
}

template<const int NUM_THREADS = 256/16>
__global__ void block_all_reduce_sum_fp8_e4m3x16_pack_f16_kernel(
  __nv_fp8_storage_t* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 16;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ half reduce_smem[NUM_WARPS];
  __nv_fp8_storage_t pack_a[16]; // 16x8 bits=128 bits.
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits

  half sum_f16 = __float2half(0.0f);
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    sum_f16 += __nv_cvt_fp8_to_halfraw(pack_a[i], __NV_E4M3);
  }
  // keep the data in register is enough for warp operaion.
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
  // warp leaders store the data to shared memory.
  // use float to keep sum from each block and reduce 
  // with fp16 inter warps.
  if (lane == 0) reduce_smem[warp] = sum_f16;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  half sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2half(0.0f);
  if (warp == 0) sum = warp_reduce_sum_f16_f16<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, __half2float(sum));
}

template<const int NUM_THREADS = 256/16>
__global__ void block_all_reduce_sum_fp8_e5m2x16_pack_f16_kernel(
  __nv_fp8_storage_t* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 16;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ half reduce_smem[NUM_WARPS];
  __nv_fp8_storage_t pack_a[16]; // 16x8 bits=128 bits.
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits

  half sum_f16 = __float2half(0.0f);
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    sum_f16 += __nv_cvt_fp8_to_halfraw(pack_a[i], __NV_E5M2);
  }
  // keep the data in register is enough for warp operaion.
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
  // warp leaders store the data to shared memory.
  // use float to keep sum from each block and reduce 
  // with fp16 inter warps.
  if (lane == 0) reduce_smem[warp] = sum_f16;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  half sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2half(0.0f);
  if (warp == 0) sum = warp_reduce_sum_f16_f16<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, __half2float(sum));
}

// -------------------------------------- INT8 -------------------------------------- 
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ int32_t warp_reduce_sum_i8_i32(int8_t val) {
  int32_t val_i32 = static_cast<int32_t>(val);
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_i32 += __shfl_xor_sync(0xffffffff, val_i32, mask);
  }
  return val_i32;
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ int32_t warp_reduce_sum_i32_i32(int32_t val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_i8_i32_kernel(
  int8_t* a, int32_t* y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ int32_t reduce_smem[NUM_WARPS];

  // keep the data in register is enough for warp operaion.
  int8_t sum_i8 = (idx < N) ? a[idx] : 0;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  int32_t sum_i32 = warp_reduce_sum_i8_i32<WARP_SIZE>(sum_i8);
  if (lane == 0) reduce_smem[warp] = sum_i32;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  int32_t sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0;
  if (warp == 0) sum = warp_reduce_sum_i32_i32<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, sum);
}

template<const int NUM_THREADS = 256/16>
__global__ void block_all_reduce_sum_i8x16_pack_i32_kernel(
  int8_t* a, int32_t* y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 16;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ int32_t reduce_smem[NUM_WARPS];
  int8_t pack_a[16]; // 16x8=128 bits
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits

  // keep the data in register is enough for warp operaion.
  int32_t sum_i32 = 0;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    sum_i32 += (static_cast<int32_t>(pack_a[i]));
  }

  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum_i32 = warp_reduce_sum_i32_i32<WARP_SIZE>(sum_i32);
  if (lane == 0) reduce_smem[warp] = sum_i32;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  int32_t sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0;
  if (warp == 0) sum = warp_reduce_sum_i32_i32<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, sum);
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

#define LANUCH_REDUCE_KERNEL(NT, packed_type, acc_type, element_type, out_type)   \
block_all_reduce_sum_##packed_type##_##acc_type##_kernel<(NT)><<<grid, block>>>(  \
  reinterpret_cast<element_type*>(x.data_ptr()),                                  \
  reinterpret_cast<out_type*>(y.data_ptr()), N);  

#define DISPATCH_REDUCE_KERNEL(K, packed_type, acc_type, element_type, n_elements, out_type) \
  const int NT = (K)/(n_elements);                                                           \
  dim3 block(NT);                                                                            \
  dim3 grid((S));                                                                            \
  switch (NT)                                                                                \
  {                                                                                          \
  case 32:                                                                                   \
    LANUCH_REDUCE_KERNEL(32, packed_type, acc_type, element_type, out_type)                  \
    break;                                                                                   \
  case 64:                                                                                   \
    LANUCH_REDUCE_KERNEL(64, packed_type, acc_type, element_type, out_type)                  \
    break;                                                                                   \
  case 128:                                                                                  \
    LANUCH_REDUCE_KERNEL(128, packed_type, acc_type, element_type, out_type)                 \
    break;                                                                                   \
  case 256:                                                                                  \
    LANUCH_REDUCE_KERNEL(256, packed_type, acc_type, element_type, out_type)                 \
    break;                                                                                   \
  case 512:                                                                                  \
    LANUCH_REDUCE_KERNEL(512, packed_type, acc_type, element_type, out_type)                 \
    break;                                                                                   \
  case 1024:                                                                                 \
    LANUCH_REDUCE_KERNEL(1024, packed_type, acc_type, element_type, out_type)                \
    break;                                                                                   \
  default:                                                                                   \
    throw std::runtime_error(                                                                \
      "only support (K)/(n_elements): 32/64/128/256/512/1024");                              \
    break;                                                                                   \
  } 

#define TORCH_BINDING_REDUCE(packed_type, acc_type, th_type, element_type, n_elements, out_type) \
torch::Tensor block_all_reduce_sum_##packed_type##_##acc_type(torch::Tensor x) {                 \
  CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                                         \
  auto y_th_type = (th_type) == torch::kInt8 ? torch::kInt32 : torch::kFloat32;                  \
  auto options = torch::TensorOptions().dtype(y_th_type).device(torch::kCUDA, 0);                \
  auto y = torch::zeros({1}, options);                                                           \
  const int ndim = x.dim();                                                                      \
  if (ndim != 2) {                                                                               \
    int N = 1;                                                                                   \
    for (int i = 0; i < ndim; ++i) { N *= x.size(i); }                                           \
    dim3 block(1024 / (n_elements));                                                             \
    dim3 grid((N + 1024 - 1) / 1024);                                                            \
    block_all_reduce_sum_##packed_type##_##acc_type##_kernel<                                    \
      1024 / (n_elements)><<<grid, block>>>(                                                     \
      reinterpret_cast<element_type*>(x.data_ptr()),                                             \
      reinterpret_cast<out_type*>(y.data_ptr()), N);                                             \
  } else {                                                                                       \
    const int S = x.size(0);                                                                     \
    const int K = x.size(1);                                                                     \
    const int N = S * K;                                                                         \
    if ((K/(n_elements)) <= 1024) {                                                              \
      DISPATCH_REDUCE_KERNEL(K, packed_type, acc_type, element_type, n_elements, out_type)       \
    } else {                                                                                     \
      int N = 1;                                                                                 \
      for (int i = 0; i < ndim; ++i) { N *= x.size(i); }                                         \
      dim3 block(1024 / (n_elements));                                                           \
      dim3 grid((N + 1024 - 1) / 1024);                                                          \
      block_all_reduce_sum_##packed_type##_##acc_type##_kernel<                                  \
        1024 / (n_elements)><<<grid, block>>>(                                                   \
        reinterpret_cast<element_type*>(x.data_ptr()),                                           \
        reinterpret_cast<out_type*>(y.data_ptr()), N);                                           \
    }                                                                                            \
  }                                                                                              \
  return y;                                                                                      \
}

// packed_type, acc_type, th_type, element_type, n_elements_per_pack, out_type
TORCH_BINDING_REDUCE(f32,              f32,  torch::kFloat32,       float,              1,  float)
TORCH_BINDING_REDUCE(f32x4,            f32,  torch::kFloat32,       float,              4,  float)
TORCH_BINDING_REDUCE(f16,              f16,  torch::kHalf,          half,               1,  float)
TORCH_BINDING_REDUCE(f16,              f32,  torch::kHalf,          half,               1,  float)
TORCH_BINDING_REDUCE(f16x2,            f16,  torch::kHalf,          half,               2,  float)
TORCH_BINDING_REDUCE(f16x2,            f32,  torch::kHalf,          half,               2,  float)
TORCH_BINDING_REDUCE(f16x8_pack,       f16,  torch::kHalf,          half,               8,  float)
TORCH_BINDING_REDUCE(f16x8_pack,       f32,  torch::kHalf,          half,               8,  float)
TORCH_BINDING_REDUCE(bf16,             bf16, torch::kBFloat16,      __nv_bfloat16,      1,  float)
TORCH_BINDING_REDUCE(bf16,             f32,  torch::kBFloat16,      __nv_bfloat16,      1,  float)
TORCH_BINDING_REDUCE(bf16x2,           bf16, torch::kBFloat16,      __nv_bfloat16,      2,  float)
TORCH_BINDING_REDUCE(bf16x2,           f32,  torch::kBFloat16,      __nv_bfloat16,      2,  float)
TORCH_BINDING_REDUCE(bf16x8_pack,      bf16, torch::kBFloat16,      __nv_bfloat16,      8,  float)
TORCH_BINDING_REDUCE(bf16x8_pack,      f32,  torch::kBFloat16,      __nv_bfloat16,      8,  float)
TORCH_BINDING_REDUCE(fp8_e4m3,         f16,  torch::kFloat8_e4m3fn, __nv_fp8_storage_t, 1,  float)
TORCH_BINDING_REDUCE(fp8_e4m3x16_pack, f16,  torch::kFloat8_e4m3fn, __nv_fp8_storage_t, 16, float)
TORCH_BINDING_REDUCE(fp8_e5m2,         f16,  torch::kFloat8_e5m2,   __nv_fp8_storage_t, 1,  float)
TORCH_BINDING_REDUCE(fp8_e5m2x16_pack, f16,  torch::kFloat8_e5m2,   __nv_fp8_storage_t, 16, float)
TORCH_BINDING_REDUCE(i8,               i32,  torch::kInt8,          int8_t,             1,  int32_t)
TORCH_BINDING_REDUCE(i8x16_pack,       i32,  torch::kInt8,          int8_t,             16, int32_t)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f32_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f32x4_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x2_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x2_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x8_pack_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x8_pack_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16_bf16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x2_bf16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x2_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x8_pack_bf16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x8_pack_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e4m3_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e4m3x16_pack_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e5m2_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e5m2x16_pack_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_i8_i32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_i8x16_pack_i32)
}
