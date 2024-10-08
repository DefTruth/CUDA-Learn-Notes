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

// -------------------------------------- FP32 -------------------------------------- 
// DS required for Online Softmax
struct __align__(8) MD { float m; float d; }; 
// Warp Reduce for Online Softmax
template<const int kWarpSize = WARP_SIZE >
__device__ __forceinline__ MD warp_reduce_md_op(MD value) {
  unsigned int mask = 0xffffffff;
  #pragma unroll
  for(int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
    MD other;
    other.m = __shfl_xor_sync(mask, value.m, stride);
    other.d = __shfl_xor_sync(mask, value.d, stride);

    bool value_bigger = (value.m > other.m);
    MD bigger_m = value_bigger ? value : other;
    MD smaller_m = value_bigger ? other : value;
    
    value.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    value.m = bigger_m.m;
  }
  return value;

// Warp Reduce Sum
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Warp Reduce Max
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// grid 1D block 1D, grid(N/256), block(256)
template<const int NUM_THREADS=256>
__device__ float block_reduce_sum_f32(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];
  
  float value = warp_reduce_sum_f32<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  value = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  value = warp_reduce_sum_f32<NUM_WARPS>(value);  
  // WRAN: need to broadcast value to all threads within warp
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

template<const int NUM_THREADS=256>
__device__ float block_reduce_max_f32(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];
  
  float value = warp_reduce_max_f32<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  value = (lane < NUM_WARPS) ? shared[lane] : -FLT_MAX;
  value = warp_reduce_max_f32<NUM_WARPS>(value);
  // WRAN: need to broadcast value to all threads within warp
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

// Softmax x: N, y: N
// grid(N/256), block(K=256)
template<const int NUM_THREADS = 256>
__global__ void softmax_f32_kernel(float* x, float* y, float* total, int N) {
  
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid; 
  
  float exp_val = (idx < N) ? expf(x[idx]) : 0.0f;
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
  // get the total sum of all blocks.
  if (tid == 0) atomicAdd(total, exp_sum);
  __threadfence(); // grid level memory fence
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  // printf("N: %d, idx: %d, bid: %d, tid: %d, exp_val: %f, exp_sum: %f, total: %f\n", 
  //         N,     idx, blockIdx.x,  tid,     exp_val,     exp_sum,     *total);
  if (idx < N) y[idx] = exp_val / (*total); 
}

// Softmax Vec4 x: N, y: N
// grid(N/256), block(256/4)
template<const int NUM_THREADS = 256/4>
__global__ void softmax_f32x4_kernel(float* x, float* y, float* total, int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * blockDim.x + tid) * 4; 
  
  float4 reg_x = FLOAT4(x[idx]);
  float4 reg_exp;
  reg_exp.x = (idx + 0 < N) ? expf(reg_x.x) : 0.0f;
  reg_exp.y = (idx + 1 < N) ? expf(reg_x.y) : 0.0f;
  reg_exp.z = (idx + 2 < N) ? expf(reg_x.z) : 0.0f;
  reg_exp.w = (idx + 3 < N) ? expf(reg_x.w) : 0.0f;
  float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
  // get the total sum of all blocks.
  if (tid == 0) atomicAdd(total, exp_sum);
  __threadfence(); // grid level memory fence
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if (idx + 3 < N) {
    float4 reg_y;
    reg_y.x = reg_exp.x / (*total);
    reg_y.y = reg_exp.y / (*total);
    reg_y.z = reg_exp.z / (*total);
    reg_y.w = reg_exp.w / (*total);
    FLOAT4(y[idx]) = reg_y; 
  }
}

// NOTE: softmax per-token
// Softmax x: (S,h), y: (S,h)
// grid(S*h/h), block(h), assume h<=1024
// one token per thread block, only support 64<=h<=1024 and 2^n
// HEAD_SIZE/KV_LEN=NUM_THREADS
template<const int NUM_THREADS = 256>
__global__ void softmax_f32_per_token_kernel(float* x, float* y, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid; 
  
  float exp_val = (idx < N) ? expf(x[idx]) : 0.0f;
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val); // block sum
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  // printf("N: %d, idx: %d, tid: %d, exp_val: %f, exp_sum: %f\n", 
  //         N, idx, tid, exp_val, exp_sum);
  if (idx < N) y[idx] = exp_val / exp_sum;
}

template<const int NUM_THREADS = 256/4>
__global__ void softmax_f32x4_per_token_kernel(float* x, float* y, int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * blockDim.x + tid) * 4;

  float4 reg_x = FLOAT4(x[idx]);
  float4 reg_exp;
  reg_exp.x = (idx + 0 < N) ? expf(reg_x.x) : 0.0f;
  reg_exp.y = (idx + 1 < N) ? expf(reg_x.y) : 0.0f;
  reg_exp.z = (idx + 2 < N) ? expf(reg_x.z) : 0.0f;
  reg_exp.w = (idx + 3 < N) ? expf(reg_x.w) : 0.0f;

  float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val); // block sum
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if (idx + 3 < N) {
    float4 reg_y;
    reg_y.x = reg_exp.x / (exp_sum);
    reg_y.y = reg_exp.y / (exp_sum);
    reg_y.z = reg_exp.z / (exp_sum);
    reg_y.w = reg_exp.w / (exp_sum);
    FLOAT4(y[idx]) = reg_y; 
  }
}

// safe_softmax per token
template<const int NUM_THREADS = 256>
__global__ void safe_softmax_f32_per_token_kernel(float* x, float* y, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid; 
  
  float val = (idx < N) ? x[idx] : -FLT_MAX;
  float max_val = block_reduce_max_f32<NUM_THREADS>(val); // block max
  float exp_val = (idx < N) ? expf(x[idx] - max_val) : 0.0f;
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val); // block sum
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if (idx < N) y[idx] = exp_val / exp_sum; 
}

template<const int NUM_THREADS = 256/4>
__global__ void safe_softmax_f32x4_per_token_kernel(float* x, float* y, int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * blockDim.x + tid) * 4;

  float4 reg_x = FLOAT4(x[idx]);
  reg_x.x = (idx + 0 < N) ? reg_x.x : -FLT_MAX;
  reg_x.y = (idx + 1 < N) ? reg_x.y : -FLT_MAX;
  reg_x.z = (idx + 2 < N) ? reg_x.z : -FLT_MAX;
  reg_x.w = (idx + 3 < N) ? reg_x.w : -FLT_MAX;
  float val =      reg_x.x;
  val = fmaxf(val, reg_x.y);
  val = fmaxf(val, reg_x.z);
  val = fmaxf(val, reg_x.w);
  float max_val = block_reduce_max_f32<NUM_THREADS>(val); // block max

  float4 reg_exp;
  reg_exp.x = (idx + 0 < N) ? expf(reg_x.x - max_val) : 0.0f;
  reg_exp.y = (idx + 1 < N) ? expf(reg_x.y - max_val) : 0.0f;
  reg_exp.z = (idx + 2 < N) ? expf(reg_x.z - max_val) : 0.0f;
  reg_exp.w = (idx + 3 < N) ? expf(reg_x.w - max_val) : 0.0f;

  float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val); // block sum
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if (idx + 3 < N) {
    float4 reg_y;
    reg_y.x = reg_exp.x / (exp_sum);
    reg_y.y = reg_exp.y / (exp_sum);
    reg_y.z = reg_exp.z / (exp_sum);
    reg_y.w = reg_exp.w / (exp_sum);
    FLOAT4(y[idx]) = reg_y; 
  }
}

template<const int NUM_THREADS = 256>
__global__ void safe_softmax_f16_f32_per_token_kernel(half* x, half* y, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid; 
  
  float val = (idx < N) ? __half2float(x[idx]) : -FLT_MAX;
  float max_val = block_reduce_max_f32<NUM_THREADS>(val); // block max
  float exp_val = (idx < N) ? expf(val - max_val) : 0.0f;
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val); // block sum
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if (idx < N) y[idx] = __float2half_rn(exp_val / exp_sum); 
}

template<const int NUM_THREADS = 256>
__global__ void safe_softmax_f16x2_f32_per_token_kernel(half* x, half* y, int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * blockDim.x + tid) * 2; 
  
  float2 reg_x = __half22float2(HALF2(x[idx]));
  float max_val = -FLT_MAX;
  max_val = ((idx + 0) < N) ? fmaxf(reg_x.x, max_val): -FLT_MAX;
  max_val = ((idx + 1) < N) ? fmaxf(reg_x.y, max_val): -FLT_MAX;
  max_val = block_reduce_max_f32<NUM_THREADS>(max_val); // block max

  float2 reg_exp;
  reg_exp.x = ((idx + 0) < N) ? expf(reg_x.x - max_val) : 0.0f;
  reg_exp.y = ((idx + 1) < N) ? expf(reg_x.y - max_val) : 0.0f;

  float exp_val = reg_exp.x + reg_exp.y;
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val); // block sum

  float2 reg_y;
  reg_y.x = reg_exp.x / (exp_sum);
  reg_y.y = reg_exp.y / (exp_sum);

  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if ((idx + 1) < N) HALF2(y[idx]) = __float22half2_rn(reg_y); 
}

template<const int NUM_THREADS = 256>
__global__ void safe_softmax_f16x8_pack_f32_per_token_kernel(half* x, half* y, int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * blockDim.x + tid) * 8; 
  // temporary register(memory), .local space in ptx, addressable
  half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits
  
  float max_val = -FLT_MAX;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    max_val = fmaxf(__half2float(pack_x[i]), max_val);
  }
  max_val = block_reduce_max_f32<NUM_THREADS>(max_val); // block max

  float exp_sum = 0.0f;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float exp_val = expf(__half2float(pack_x[i]) - max_val);
    exp_sum += (((idx + i) < N) ? exp_val : 0.0f);
  }
  exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_sum); // block sum

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    // e^x_i/sum(e^x_0,...,e^x_n-1) 
    float exp_val = expf(__half2float(pack_x[i]) - max_val);
    pack_y[i] = __float2half_rn(exp_val / exp_sum);
  }
  // reinterpret as float4 and store 128 bits in 1 memory issue.
  if ((idx + 7) < N) { LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]); }
  // TODO: support non 8-multiple K here
}

template<const int NUM_THREADS = 256 >
__global__ void online_softmax_f32_per_token_kernel(const float* x, float* y, int N) {
  // reference: https://arxiv.org/pdf/1805.02867 (Online normalizer calculation for softmax)
  int local_tid = threadIdx.x;
  int global_tid = blockIdx.x * NUM_THREADS + threadIdx.x;
  const int WAPR_NUM = NUM_THREADS / WARP_SIZE;
  int warp_id = local_tid / WARP_SIZE;
  int lane_id = local_tid % WARP_SIZE;
  MD val;
  val.m = global_tid < N ? x[global_tid] : -FLT_MAX;
  val.d = global_tid < N ? 1.0f : 0.0f;

  __shared__ MD shared[WAPR_NUM]; 
  MD res = warp_reduce_md_op<WARP_SIZE>(val);

  if (lane_id == 0) shared[warp_id] = res; 
  __syncthreads();

  if (local_tid < WARP_SIZE) {
    MD block_res = shared[local_tid];
    block_res = warp_reduce_md_op<WAPR_NUM>(block_res); 
    if (local_tid == 0) {
      shared[0] = block_res; 
    }
  }
  __syncthreads();

  MD final_res = shared[0];
  float d_total_inverse = __fdividef(1.0f, final_res.d);
  if (global_tid < N) {
    y[global_tid] = __expf(x[global_tid] - final_res.m) * d_total_inverse;
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

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)               \
assert((T1).dim() == (T2).dim());                      \
for (int i = 0; i < (T1).dim(); ++i) {                 \
  if ((T2).size(i) != (T1).size(i)) {                  \
    throw std::runtime_error("Tensor size mismatch!"); \
  }                                                    \
}

// grid memory fence
#define TORCH_BINDING_SOFTMAX(packed_type, th_type, element_type, n_elements)    \
void softmax_##packed_type(torch::Tensor x, torch::Tensor y) {                   \
  CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                         \
  CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                         \
  auto options = torch::TensorOptions().dtype((th_type)).device(torch::kCUDA, 0);\
  const int N = x.size(0);                                                       \
  CHECK_TORCH_TENSOR_SHAPE(x, y)                                                 \
  auto total = torch::zeros({1}, options);                                       \
  dim3 block(256);                                                               \
  dim3 grid(((N + 256 - 1) / 256) / (n_elements));                               \
  softmax_##packed_type##_kernel<256><<<grid, block>>>(                          \
      reinterpret_cast<element_type*>(x.data_ptr()),                             \
      reinterpret_cast<element_type*>(y.data_ptr()),                             \
      reinterpret_cast<element_type*>(total.data_ptr()), N);                     \
}

// softmax per token
#define LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(H)       \
softmax_f32_per_token_kernel<(H)><<<grid, block>>>(  \
      reinterpret_cast<float*>(x.data_ptr()),        \
      reinterpret_cast<float*>(y.data_ptr()),        \
      N);  

#define DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H) \
  dim3 block((H));                                  \
  dim3 grid((S));                                   \
  switch ((H))                                      \
  {                                                 \
  case 32:                                          \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(32)         \
    break;                                          \
  case 64:                                          \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(64)         \
    break;                                          \
  case 128:                                         \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(128)        \
    break;                                          \
  case 256:                                         \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(256)        \
    break;                                          \
  case 512:                                         \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(512)        \
    break;                                          \
  case 1024:                                        \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(1024)       \
    break;                                          \
  default:                                          \
    throw std::runtime_error(                       \
      "only support H: 64/128/256/512/1024");       \
    break;                                          \
  } 

#define LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(H)  \
softmax_f32x4_per_token_kernel<(H)/4><<<          \
      grid, block>>>(                             \
      reinterpret_cast<float*>(x.data_ptr()),     \
      reinterpret_cast<float*>(y.data_ptr()),     \
      N);  

#define DISPATCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H) \
  const int NT = (H)/4;                               \
  dim3 block(NT);                                     \
  dim3 grid((S));                                     \
  switch (H)                                          \
  {                                                   \
  case 32:                                            \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(32)         \
    break;                                            \
  case 64:                                            \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(64)         \
    break;                                            \
  case 128:                                           \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(128)        \
    break;                                            \
  case 256:                                           \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(256)        \
    break;                                            \
  case 512:                                           \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(512)        \
    break;                                            \
  case 1024:                                          \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(1024)       \
    break;                                            \
  case 2048:                                          \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(2048)       \
    break;                                            \
  case 4096:                                          \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(4096)       \
    break;                                            \
  default:                                            \
    throw std::runtime_error(                         \
      "only support H: 64/128/.../1024*4");           \
    break;                                            \
  } 

// safe softmax per token
#define LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(H)       \
safe_softmax_f32_per_token_kernel<(H)><<<grid, block>>>(  \
      reinterpret_cast<float*>(x.data_ptr()),             \
      reinterpret_cast<float*>(y.data_ptr()),             \
      N);  

#define DISPATCH_SATE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H) \
  dim3 block((H));                                       \
  dim3 grid((S));                                        \
  switch ((H))                                           \
  {                                                      \
  case 32:                                               \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(32)         \
    break;                                               \
  case 64:                                               \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(64)         \
    break;                                               \
  case 128:                                              \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(128)        \
    break;                                               \
  case 256:                                              \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(256)        \
    break;                                               \
  case 512:                                              \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(512)        \
    break;                                               \
  case 1024:                                             \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(1024)       \
    break;                                               \
  default:                                               \
    throw std::runtime_error(                            \
      "only support H: 64/128/256/512/1024");            \
    break;                                               \
  } 

// online softmax per token
#define LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(H)       \
online_softmax_f32_per_token_kernel<(H)><<<grid, block>>>(  \
      reinterpret_cast<float*>(x.data_ptr()),               \
      reinterpret_cast<float*>(y.data_ptr()),               \
      N);  

#define DISPATCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H) \
  dim3 block((H));                                       \
  dim3 grid((S));                                        \
  switch ((H))                                           \
  {                                                      \
  case 32:                                               \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(32)       \
    break;                                               \
  case 64:                                               \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(64)       \
    break;                                               \
  case 128:                                              \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(128)      \
    break;                                               \
  case 256:                                              \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(256)      \
    break;                                               \
  case 512:                                              \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(512)      \
    break;                                               \
  case 1024:                                             \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(1024)     \
    break;                                               \
  default:                                               \
    throw std::runtime_error(                            \
      "only support H: 64/128/256/512/1024");            \
    break;                                               \
  } 
#define LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(H)   \
safe_softmax_f32x4_per_token_kernel<(H)/4><<<           \
      grid, block>>>(                                   \
      reinterpret_cast<float*>(x.data_ptr()),           \
      reinterpret_cast<float*>(y.data_ptr()),           \
      N);  

#define DISPATCH_SATE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H) \
  const int NT = (H)/4;                                    \
  dim3 block(NT);                                          \
  dim3 grid((S));                                          \
  switch (H)                                               \
  {                                                        \
  case 32:                                                 \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(32)         \
    break;                                                 \
  case 64:                                                 \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(64)         \
    break;                                                 \
  case 128:                                                \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(128)        \
    break;                                                 \
  case 256:                                                \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(256)        \
    break;                                                 \
  case 512:                                                \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(512)        \
    break;                                                 \
  case 1024:                                               \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(1024)       \
    break;                                                 \
  case 2048:                                               \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(2048)       \
    break;                                                 \
  case 4096:                                               \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(4096)       \
    break;                                                 \
  default:                                                 \
    throw std::runtime_error(                              \
      "only support H: 64/128/.../1024*4");                \
    break;                                                 \
  } 

#define LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(H)       \
safe_softmax_f16_f32_per_token_kernel<(H)><<<grid, block>>>(  \
      reinterpret_cast<half*>(x.data_ptr()),                  \
      reinterpret_cast<half*>(y.data_ptr()),                  \
      N);  

#define DISPATCH_SATE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(S, H) \
  dim3 block((H));                                           \
  dim3 grid((S));                                            \
  switch ((H))                                               \
  {                                                          \
  case 32:                                                   \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(32)         \
    break;                                                   \
  case 64:                                                   \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(64)         \
    break;                                                   \
  case 128:                                                  \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(128)        \
    break;                                                   \
  case 256:                                                  \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(256)        \
    break;                                                   \
  case 512:                                                  \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(512)        \
    break;                                                   \
  case 1024:                                                 \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(1024)       \
    break;                                                   \
  default:                                                   \
    throw std::runtime_error(                                \
      "only support H: 64/128/256/512/1024");                \
    break;                                                   \
  } 

#define LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(H)        \
safe_softmax_f16x2_f32_per_token_kernel<(H)/2><<<grid, block>>>( \
      reinterpret_cast<half*>(x.data_ptr()),                     \
      reinterpret_cast<half*>(y.data_ptr()),                     \
      N);  

#define DISPATCH_SATE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(S, H) \
  const int NT = (H)/2;                                        \
  dim3 block(NT);                                              \
  dim3 grid((S));                                              \
  switch (H)                                                   \
  {                                                            \
  case 32:                                                     \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(32)         \
    break;                                                     \
  case 64:                                                     \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(64)         \
    break;                                                     \
  case 128:                                                    \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(128)        \
    break;                                                     \
  case 256:                                                    \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(256)        \
    break;                                                     \
  case 512:                                                    \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(512)        \
    break;                                                     \
  case 1024:                                                   \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(1024)       \
    break;                                                     \
  case 2048:                                                   \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(2048)       \
    break;                                                     \
  default:                                                     \
    throw std::runtime_error(                                  \
      "only support H: 64/128/.../1024*2");                    \
    break;                                                     \
  } 

#define LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(H)        \
safe_softmax_f16x8_pack_f32_per_token_kernel<(H)/8><<<grid, block>>>( \
      reinterpret_cast<half*>(x.data_ptr()),                          \
      reinterpret_cast<half*>(y.data_ptr()),                          \
      N);  

#define DISPATCH_SATE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(S, H) \
  const int NT = (H)/8;                                             \
  dim3 block(NT);                                                   \
  dim3 grid((S));                                                   \
  switch (H)                                                        \
  {                                                                 \
  case 32:                                                          \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(32)         \
    break;                                                          \
  case 64:                                                          \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(64)         \
    break;                                                          \
  case 128:                                                         \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(128)        \
    break;                                                          \
  case 256:                                                         \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(256)        \
    break;                                                          \
  case 512:                                                         \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(512)        \
    break;                                                          \
  case 1024:                                                        \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(1024)       \
    break;                                                          \
  case 2048:                                                        \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(2048)       \
    break;                                                          \
  case 4096:                                                        \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(4096)       \
    break;                                                          \
  case 8192:                                                        \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(8192)       \
    break;                                                          \
  default:                                                          \
    throw std::runtime_error(                                       \
      "only support H: 64/128/.../1024*8");                         \
    break;                                                          \
  } 

// per token fp32
void softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)                       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)                   
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len
  const int N = S * H; 
  DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
}

void softmax_f32x4_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)                       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)                                                                                                            
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len
  const int N = S * H; 
  DISPATCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)                       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)                                                                                                                              
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len
  const int N = S * H; 
  DISPATCH_SATE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f32x4_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)                       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)                                                                                                                      
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len
  const int N = S * H; 
  DISPATCH_SATE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)
}

// per token fp16
void safe_softmax_f16_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)                       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)                                                                                                                              
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len
  const int N = S * H; 
  DISPATCH_SATE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f16x2_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)                       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)                                                                                                                              
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len
  const int N = S * H; 
  DISPATCH_SATE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f16x8_pack_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)                       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)                                                                                                                              
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len
  const int N = S * H; 
  DISPATCH_SATE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(S, H)
}

void online_softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)                       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)                                                                                                                              
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len
  const int N = S * H; 
  DISPATCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
}

// grid memory fence fp32
TORCH_BINDING_SOFTMAX(f32,   torch::kFloat32, float, 1)
TORCH_BINDING_SOFTMAX(f32x4, torch::kFloat32, float, 4)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32)
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32x4_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32x4_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16x2_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16x8_pack_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(online_softmax_f32_per_token)
}
