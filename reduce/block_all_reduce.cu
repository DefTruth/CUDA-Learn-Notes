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
  // keep the data in register is enougth for warp operaion.
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
  // keep the data in register is enougth for warp operaion.
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
  // keep the data in register is enougth for warp operaion.
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
  // keep the data in register is enougth for warp operaion.
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

  // keep the data in register is enougth for warp operaion.
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

  // keep the data in register is enougth for warp operaion.
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

  // keep the data in register is enougth for warp operaion.
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

  // keep the data in register is enougth for warp operaion.
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

  // keep the data in register is enougth for warp operaion.
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

template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_bf16x2_f32_kernel(
  __nv_bfloat16* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 2; // 2 bf16 elements per thread
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  // keep the data in register is enougth for warp operaion.
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

  // keep the data in register is enougth for warp operaion.
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

  // keep the data in register is enougth for warp operaion.
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

  // keep the data in register is enougth for warp operaion.
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

// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
if(((T).options().dtype() != (th_type))) {                   \
  std::cout << "Tensor Info:" << (T).options() << std::endl; \
  throw std::runtime_error("values must be "#th_type);       \
}

#define CHECK_TORCH_TENSOR_SHAPE(T, S0) \
if (((T).size(0) != (S0))) { throw std::runtime_error("Tensor size mismatch!"); }

#define TORCH_BINDING_BLOCK_ALL_REDUCE(packed_type, acc_type, th_type, element_type, n_elements) \
torch::Tensor block_all_reduce_sum_##packed_type##_##acc_type(torch::Tensor a) {                 \
  CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                                         \
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(                           \
    torch::kCUDA, 0);                                                                            \
  auto sum = torch::zeros({1}, options);                                                         \
  const int N = a.size(0);                                                                       \
  static const int NUM_THREADS_PER_BLOCK = 256 / (n_elements);                                   \
  const int NUM_BLOCKS = (N + 256 - 1) / 256;                                                    \
  dim3 block(NUM_THREADS_PER_BLOCK);                                                             \
  dim3 grid(NUM_BLOCKS);                                                                         \
  block_all_reduce_sum_##packed_type##_##acc_type##_kernel<                                      \
    NUM_THREADS_PER_BLOCK><<<grid, block>>>(                                                     \
      reinterpret_cast<element_type*>(a.data_ptr()), sum.data_ptr<float>(), N);                  \
  return sum;                                                                                    \
}

#define TORCH_BINDING_BLOCK_ALL_REDUCE_I(packed_type, acc_type, th_type, element_type, n_elements) \
torch::Tensor block_all_reduce_sum_##packed_type##_##acc_type(torch::Tensor a) {                   \
  CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                                           \
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(                               \
    torch::kCUDA, 0);                                                                              \
  auto sum = torch::zeros({1}, options);                                                           \
  const int N = a.size(0);                                                                         \
  static const int NUM_THREADS_PER_BLOCK = 256 / (n_elements);                                     \
  const int NUM_BLOCKS = (N + 256 - 1) / 256;                                                      \
  dim3 block(NUM_THREADS_PER_BLOCK);                                                               \
  dim3 grid(NUM_BLOCKS);                                                                           \
  block_all_reduce_sum_##packed_type##_##acc_type##_kernel<                                        \
    NUM_THREADS_PER_BLOCK><<<grid, block>>>(                                                       \
      reinterpret_cast<element_type*>(a.data_ptr()), sum.data_ptr<int32_t>(), N);                  \
  return sum;                                                                                      \
}

// packed_type, acc_type, th_type, element_type, n_elements_per_pack
TORCH_BINDING_BLOCK_ALL_REDUCE(f32,      f32,  torch::kFloat32,       float,              1)
TORCH_BINDING_BLOCK_ALL_REDUCE(f32x4,    f32,  torch::kFloat32,       float,              4)
TORCH_BINDING_BLOCK_ALL_REDUCE(f16,      f16,  torch::kHalf,          half,               1)
TORCH_BINDING_BLOCK_ALL_REDUCE(f16,      f32,  torch::kHalf,          half,               1)
TORCH_BINDING_BLOCK_ALL_REDUCE(f16x2,    f16,  torch::kHalf,          half,               2)
TORCH_BINDING_BLOCK_ALL_REDUCE(f16x2,    f32,  torch::kHalf,          half,               2)
TORCH_BINDING_BLOCK_ALL_REDUCE(bf16,     bf16, torch::kBFloat16,      __nv_bfloat16,      1)
TORCH_BINDING_BLOCK_ALL_REDUCE(bf16,     f32,  torch::kBFloat16,      __nv_bfloat16,      1)
TORCH_BINDING_BLOCK_ALL_REDUCE(bf16x2,   bf16, torch::kBFloat16,      __nv_bfloat16,      2)
TORCH_BINDING_BLOCK_ALL_REDUCE(bf16x2,   f32,  torch::kBFloat16,      __nv_bfloat16,      2)
TORCH_BINDING_BLOCK_ALL_REDUCE(fp8_e4m3, f16,  torch::kFloat8_e4m3fn, __nv_fp8_storage_t, 1)
TORCH_BINDING_BLOCK_ALL_REDUCE(fp8_e5m2, f16,  torch::kFloat8_e5m2,   __nv_fp8_storage_t, 1)
TORCH_BINDING_BLOCK_ALL_REDUCE_I(i8,     i32,  torch::kInt8,          int8_t,             1)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f32_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f32x4_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x2_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x2_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16_bf16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x2_bf16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x2_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e4m3_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e5m2_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_i8_i32)
}
