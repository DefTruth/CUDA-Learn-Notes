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
__global__ void rms_norm_f32_kernel(float* x, float* y, float g, int N, int K) {
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
__global__ void rms_norm_f32x4_kernel(float* x, float* y, float g, int N, int K) {
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

// -------------------------------------- FP16 -------------------------------------- 
// Warp Reduce Sum: Half
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    // val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
    val += __shfl_xor_sync(0xffffffff, val, mask);
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

template<const int NUM_THREADS=256>
__device__ half block_reduce_sum_f16_f16(half val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ half shared[NUM_WARPS];
  // reduce using half dtype within warps
  val = warp_reduce_sum_f16_f16<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = val;
  __syncthreads();
  val = (lane < NUM_WARPS) ? shared[lane] : __float2half(0.0f);
  val = warp_reduce_sum_f16_f16<NUM_WARPS>(val);
  return val; // half
}

template<const int NUM_THREADS=256>
__device__ float block_reduce_sum_f16_f32(half val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];
  // reduce using float dtype within warps
  float val_f32 = warp_reduce_sum_f16_f32<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = val_f32;
  __syncthreads();
  val_f32 = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  val_f32 = warp_reduce_sum_f32<NUM_WARPS>(val_f32);
  return val_f32; // float
}

template<const int NUM_THREADS=256>
__global__ void rms_norm_f16_f16_kernel(half* x, half* y, float g, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x; // 0..N-1
  int idx = bid * blockDim.x + threadIdx.x;
  const half epsilon = __float2half(1e-5f);
  const half g_      = __float2half(g);
  const half K_      = __int2half_rn(K);

  __shared__ half s_variance; // shared within block
  half value = (idx < N * K) ? x[idx] : __float2half(0.0f); // load once only
  half variance = value * value;
  variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
  if (tid == 0) s_variance = hrsqrt(variance / (K_ + epsilon));
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads(); 
  if (idx < N * K) y[idx] = (value * s_variance) * g_;
}

template<const int NUM_THREADS=256>
__global__ void rms_norm_f16x2_f16_kernel(half* x, half* y, float g, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x; // 0..N-1
  int idx = (bid * blockDim.x + threadIdx.x) * 2;
  const half epsilon = __float2half(1e-5f);
  const half g_      = __float2half(g);
  const half K_      = __int2half_rn(K);

  __shared__ half s_variance; // shared within block
  half2 reg_x = HALF2(x[idx]);
  half variance = (idx < N * K) ? (reg_x.x * reg_x.x 
                                 + reg_x.y * reg_x.y): __float2half(0.0f);
  variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
  if (tid == 0) s_variance = hrsqrt(variance / (K_ + epsilon));
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads(); 
  half2 reg_y;
  reg_y.x = reg_x.x * s_variance * g_;
  reg_y.y = reg_x.y * s_variance * g_;
  if (idx < N * K) HALF2(y[idx]) = reg_y;
}

template<const int NUM_THREADS=256>
__global__ void rms_norm_f16x8_f16_kernel(half* x, half* y, float g, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x; // 0..N-1
  int idx = (bid * blockDim.x + threadIdx.x) * 2;
  const half epsilon = __float2half(1e-5f);
  const half g_      = __float2half(g);
  const half K_      = __int2half_rn(K);

  __shared__ half s_variance; // shared within block
  // manual unroll, load 128 bits in 1 memory issue.
  half2 reg_x_0 = HALF2(x[idx + 0]);
  half2 reg_x_1 = HALF2(x[idx + 2]);
  half2 reg_x_2 = HALF2(x[idx + 4]);
  half2 reg_x_3 = HALF2(x[idx + 6]);
  half variance = (((idx + 0) < N * K) ? (reg_x_0.x * reg_x_0.x 
                                        + reg_x_0.y * reg_x_0.y) 
                                        : __float2half(0.0f));
  variance     += (((idx + 2) < N * K) ? (reg_x_1.x * reg_x_1.x 
                                        + reg_x_1.y * reg_x_1.y) 
                                       : __float2half(0.0f));
  variance     += (((idx + 4) < N * K) ? (reg_x_2.x * reg_x_2.x 
                                        + reg_x_2.y * reg_x_2.y) 
                                        : __float2half(0.0f));
  variance     += (((idx + 6) < N * K) ? (reg_x_3.x * reg_x_3.x 
                                        + reg_x_3.y * reg_x_3.y) 
                                        : __float2half(0.0f));
  variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
  if (tid == 0) s_variance = hrsqrt(variance / (K_ + epsilon));
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads(); 
  // manual unroll
  half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
  reg_y_0.x = reg_x_0.x * s_variance * g_;
  reg_y_0.y = reg_x_0.y * s_variance * g_;
  reg_y_1.x = reg_x_1.x * s_variance * g_;
  reg_y_1.y = reg_x_1.y * s_variance * g_;
  reg_y_2.x = reg_x_2.x * s_variance * g_;
  reg_y_2.y = reg_x_2.y * s_variance * g_;
  reg_y_3.x = reg_x_3.x * s_variance * g_;
  reg_y_3.y = reg_x_3.y * s_variance * g_;
  if ((idx + 0) < N * K) { HALF2(y[idx + 0]) = reg_y_0; }
  if ((idx + 2) < N * K) { HALF2(y[idx + 2]) = reg_y_1; }
  if ((idx + 4) < N * K) { HALF2(y[idx + 4]) = reg_y_2; }
  if ((idx + 6) < N * K) { HALF2(y[idx + 6]) = reg_y_3; }
}

template<const int NUM_THREADS=256>
__global__ void rms_norm_f16_f32_kernel(half* x, half* y, float g, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x; // 0..N-1
  int idx = bid * blockDim.x + threadIdx.x;
  const float epsilon = 1e-5f;

  __shared__ float s_variance; // shared within block
  float value = (idx < N * K) ? __half2float(x[idx]) : 0.0f; // load once only
  float variance = value * value;
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / ((float) K + epsilon));
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads(); 
  if (idx < N * K) {
    y[idx] = __float2half((value * s_variance) * g);
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

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)                                \
if (((T2).size(0) != (T1).size(0)) || ((T2).size(1) != (T1).size(1))) { \
  throw std::runtime_error("Tensor size mismatch!");                    \
}

#define LANUCH_RMS_NORM_F32_KERNEL(K)      \
rms_norm_f32_kernel<(K)><<<grid, block>>>( \
  reinterpret_cast<float*>(x.data_ptr()),  \
  reinterpret_cast<float*>(y.data_ptr()),  \
  g, N, (K));  

#define DISPATCH_RMS_NORM_F32_KERNEL(N, K)   \
  dim3 block((K));                           \
  dim3 grid((N));                            \     
  switch ((K))                               \
  {                                          \
  case 64:                                   \
    LANUCH_RMS_NORM_F32_KERNEL(64)           \
    break;                                   \
  case 128:                                  \
    LANUCH_RMS_NORM_F32_KERNEL(128)          \
    break;                                   \
  case 256:                                  \
    LANUCH_RMS_NORM_F32_KERNEL(256)          \
    break;                                   \
  case 512:                                  \
    LANUCH_RMS_NORM_F32_KERNEL(512)          \
    break;                                   \
  case 1024:                                 \
    LANUCH_RMS_NORM_F32_KERNEL(1024)         \
    break;                                   \
  default:                                   \
    throw std::runtime_error(                \
      "only support K: 64/128/256/512/1024");\
    break;                                   \
  } 

#define LANUCH_RMS_NORM_F32x4_KERNEL(K)        \
rms_norm_f32x4_kernel<(K)/4><<<grid, block>>>( \
  reinterpret_cast<float*>(x.data_ptr()),      \
  reinterpret_cast<float*>(y.data_ptr()),      \
  g, N, (K));  

#define DISPATCH_RMS_NORM_F32x4_KERNEL(N, K) \
  dim3 block((K)/4);                         \
  dim3 grid((N));                            \     
  switch ((K))                               \
  {                                          \
  case 64:                                   \
    LANUCH_RMS_NORM_F32x4_KERNEL(64)         \
    break;                                   \
  case 128:                                  \
    LANUCH_RMS_NORM_F32x4_KERNEL(128)        \
    break;                                   \
  case 256:                                  \
    LANUCH_RMS_NORM_F32x4_KERNEL(256)        \
    break;                                   \
  case 512:                                  \
    LANUCH_RMS_NORM_F32x4_KERNEL(512)        \
    break;                                   \
  case 1024:                                 \
    LANUCH_RMS_NORM_F32x4_KERNEL(1024)       \
    break;                                   \
  default:                                   \
    throw std::runtime_error(                \
      "only support K: 64/128/256/512/1024");\
    break;                                   \
  } 

void rms_norm_f32(torch::Tensor x, torch::Tensor y, float g) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_RMS_NORM_F32_KERNEL(N, K)
}

void rms_norm_f32x4(torch::Tensor x, torch::Tensor y, float g) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_RMS_NORM_F32x4_KERNEL(N, K)
}

// fp16
#define LANUCH_RMS_NORM_F16F16_KERNEL(K)         \
rms_norm_f16_f16_kernel<(K)><<<grid, block>>>(   \
  reinterpret_cast<half*>(x.data_ptr()),         \
  reinterpret_cast<half*>(y.data_ptr()),         \
  g, N, (K));  

#define DISPATCH_RMS_NORM_F16F16_KERNEL(N, K)   \
  dim3 block((K));                              \
  dim3 grid((N));                               \     
  switch ((K))                                  \
  {                                             \
  case 64:                                      \
    LANUCH_RMS_NORM_F16F16_KERNEL(64)           \
    break;                                      \
  case 128:                                     \
    LANUCH_RMS_NORM_F16F16_KERNEL(128)          \
    break;                                      \
  case 256:                                     \
    LANUCH_RMS_NORM_F16F16_KERNEL(256)          \
    break;                                      \
  case 512:                                     \
    LANUCH_RMS_NORM_F16F16_KERNEL(512)          \
    break;                                      \
  case 1024:                                    \
    LANUCH_RMS_NORM_F16F16_KERNEL(1024)         \
    break;                                      \
  default:                                      \
    throw std::runtime_error(                   \
      "only support K: 64/128/256/512/1024");   \
    break;                                      \
  } 

#define LANUCH_RMS_NORM_F16F32_KERNEL(K)         \
rms_norm_f16_f32_kernel<(K)><<<grid, block>>>( \
  reinterpret_cast<half*>(x.data_ptr()),         \
  reinterpret_cast<half*>(y.data_ptr()),         \
  g, N, (K));  

#define DISPATCH_RMS_NORM_F16F32_KERNEL(N, K)   \
  dim3 block((K));                              \
  dim3 grid((N));                               \     
  switch ((K))                                  \
  {                                             \
  case 64:                                      \
    LANUCH_RMS_NORM_F16F32_KERNEL(64)           \
    break;                                      \
  case 128:                                     \
    LANUCH_RMS_NORM_F16F32_KERNEL(128)          \
    break;                                      \
  case 256:                                     \
    LANUCH_RMS_NORM_F16F32_KERNEL(256)          \
    break;                                      \
  case 512:                                     \
    LANUCH_RMS_NORM_F16F32_KERNEL(512)          \
    break;                                      \
  case 1024:                                    \
    LANUCH_RMS_NORM_F16F32_KERNEL(1024)         \
    break;                                      \
  default:                                      \
    throw std::runtime_error(                   \
      "only support K: 64/128/256/512/1024");   \
    break;                                      \
  } 

#define LANUCH_RMS_NORM_F16x2F16_KERNEL(K)           \
rms_norm_f16x2_f16_kernel<(K)/2><<<grid, block>>>( \
  reinterpret_cast<half*>(x.data_ptr()),             \
  reinterpret_cast<half*>(y.data_ptr()),             \
  g, N, (K));  

#define DISPATCH_RMS_NORM_F16x2F16_KERNEL(N, K)   \
  dim3 block((K)/2);                              \
  dim3 grid((N));                                 \     
  switch ((K))                                    \
  {                                               \
  case 64:                                        \
    LANUCH_RMS_NORM_F16x2F16_KERNEL(64)           \
    break;                                        \
  case 128:                                       \
    LANUCH_RMS_NORM_F16x2F16_KERNEL(128)          \
    break;                                        \
  case 256:                                       \
    LANUCH_RMS_NORM_F16x2F16_KERNEL(256)          \
    break;                                        \
  case 512:                                       \
    LANUCH_RMS_NORM_F16x2F16_KERNEL(512)          \
    break;                                        \
  case 1024:                                      \
    LANUCH_RMS_NORM_F16x2F16_KERNEL(1024)         \
    break;                                        \
  default:                                        \
    throw std::runtime_error(                     \
      "only support K: 64/128/256/512/1024");     \
    break;                                        \
  } 

#define LANUCH_RMS_NORM_F16x8F16_KERNEL(K)           \
rms_norm_f16x8_f16_kernel<(K)/8><<<grid, block>>>( \
  reinterpret_cast<half*>(x.data_ptr()),             \
  reinterpret_cast<half*>(y.data_ptr()),             \
  g, N, (K));  

#define DISPATCH_RMS_NORM_F16x8F16_KERNEL(N, K)   \
  dim3 block((K)/8);                              \
  dim3 grid((N));                                 \     
  switch ((K))                                    \
  {                                               \
  case 64:                                        \
    LANUCH_RMS_NORM_F16x8F16_KERNEL(64)           \
    break;                                        \
  case 128:                                       \
    LANUCH_RMS_NORM_F16x8F16_KERNEL(128)          \
    break;                                        \
  case 256:                                       \
    LANUCH_RMS_NORM_F16x8F16_KERNEL(256)          \
    break;                                        \
  case 512:                                       \
    LANUCH_RMS_NORM_F16x8F16_KERNEL(512)          \
    break;                                        \
  case 1024:                                      \
    LANUCH_RMS_NORM_F16x8F16_KERNEL(1024)         \
    break;                                        \
  default:                                        \
    throw std::runtime_error(                     \
      "only support K: 64/128/256/512/1024");     \
    break;                                        \
  } 

void rms_norm_f16_f16(torch::Tensor x, torch::Tensor y, float g) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_RMS_NORM_F16F16_KERNEL(N, K)
}

void rms_norm_f16x2_f16(torch::Tensor x, torch::Tensor y, float g) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_RMS_NORM_F16x2F16_KERNEL(N, K)
}

void rms_norm_f16x8_f16(torch::Tensor x, torch::Tensor y, float g) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_RMS_NORM_F16x8F16_KERNEL(N, K)
}

void rms_norm_f16_f32(torch::Tensor x, torch::Tensor y, float g) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_RMS_NORM_F16F32_KERNEL(N, K)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(rms_norm_f32)
  TORCH_BINDING_COMMON_EXTENSION(rms_norm_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(rms_norm_f16_f16)
  TORCH_BINDING_COMMON_EXTENSION(rms_norm_f16x2_f16)
  TORCH_BINDING_COMMON_EXTENSION(rms_norm_f16x8_f16)
  TORCH_BINDING_COMMON_EXTENSION(rms_norm_f16_f32)
}
