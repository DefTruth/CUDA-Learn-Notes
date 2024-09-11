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
  reg_exp.x = (idx < N) ? expf(reg_x.x) : 0.0f;
  reg_exp.y = (idx < N) ? expf(reg_x.y) : 0.0f;
  reg_exp.z = (idx < N) ? expf(reg_x.z) : 0.0f;
  reg_exp.w = (idx < N) ? expf(reg_x.w) : 0.0f;
  float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
  // get the total sum of all blocks.
  if (tid == 0) atomicAdd(total, exp_sum);
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
  reg_exp.x = (idx < N) ? expf(reg_x.x) : 0.0f;
  reg_exp.y = (idx < N) ? expf(reg_x.y) : 0.0f;
  reg_exp.z = (idx < N) ? expf(reg_x.z) : 0.0f;
  reg_exp.w = (idx < N) ? expf(reg_x.w) : 0.0f;

  float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val); // block sum
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if (idx < N) {
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
  reg_x.x = (idx < N) ? reg_x.x : -FLT_MAX;
  reg_x.y = (idx < N) ? reg_x.y : -FLT_MAX;
  reg_x.z = (idx < N) ? reg_x.z : -FLT_MAX;
  reg_x.w = (idx < N) ? reg_x.w : -FLT_MAX;
  float val =      reg_x.x;
  val = fmaxf(val, reg_x.y);
  val = fmaxf(val, reg_x.z);
  val = fmaxf(val, reg_x.w);
  float max_val = block_reduce_max_f32<NUM_THREADS>(val); // block max

  float4 reg_exp;
  reg_exp.x = (idx < N) ? expf(reg_x.x - max_val) : 0.0f;
  reg_exp.y = (idx < N) ? expf(reg_x.y - max_val) : 0.0f;
  reg_exp.z = (idx < N) ? expf(reg_x.z - max_val) : 0.0f;
  reg_exp.w = (idx < N) ? expf(reg_x.w - max_val) : 0.0f;

  float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val); // block sum
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if (idx < N) {
    float4 reg_y;
    reg_y.x = reg_exp.x / (exp_sum);
    reg_y.y = reg_exp.y / (exp_sum);
    reg_y.z = reg_exp.z / (exp_sum);
    reg_y.w = reg_exp.w / (exp_sum);
    FLOAT4(y[idx]) = reg_y; 
  }
}

// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

// naive softmax
#define TORCH_BINDING_SOFTMAX(packed_type, th_type, element_type, n_elements)    \
torch::Tensor softmax_##packed_type(torch::Tensor x) {                           \
  if((x.options().dtype() != (th_type))) {                                       \
    std::cout << "x Tensor Info:" << x.options() << std::endl;                   \
    throw std::runtime_error("values must be "#th_type);                         \
  }                                                                              \
  auto options = torch::TensorOptions().dtype((th_type)).device(                 \
    torch::kCUDA, 0);                                                            \
  const int N = x.size(0);                                                       \
  auto y = torch::zeros({N}, options);                                           \
  auto total = torch::zeros({1}, options);                                       \
  static const int NUM_THREADS_PER_BLOCK = 256 / (n_elements);                   \
  const int NUM_BLOCKS = (N + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;\
  dim3 block(NUM_THREADS_PER_BLOCK);                                             \
  dim3 grid(NUM_BLOCKS);                                                         \
  softmax_##packed_type##_kernel<NUM_THREADS_PER_BLOCK><<<grid, block>>>(        \
      reinterpret_cast<element_type*>(x.data_ptr()),                             \
      reinterpret_cast<element_type*>(y.data_ptr()),                             \
      reinterpret_cast<element_type*>(total.data_ptr()), N);                     \
  return y;                                                                      \
}

#define TORCH_BINDING_SOFTMAX_V2(packed_type, th_type, element_type, n_elements) \
void softmax_##packed_type##_v2(torch::Tensor x, torch::Tensor y) {              \
  if((x.options().dtype() != (th_type))) {                                       \
    std::cout << "x Tensor Info:" << x.options() << std::endl;                   \
    throw std::runtime_error("values must be "#th_type);                         \
  }                                                                              \
  auto options = torch::TensorOptions().dtype((th_type)).device(                 \
    torch::kCUDA, 0);                                                            \
  const int N = x.size(0);                                                       \
  if (y.size(0) != N) {throw std::runtime_error("y size mismatch!"); }           \
  auto total = torch::zeros({1}, options);                                       \
  static const int NUM_THREADS_PER_BLOCK = 256 / (n_elements);                   \
  const int NUM_BLOCKS = (N + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;\
  dim3 block(NUM_THREADS_PER_BLOCK);                                             \
  dim3 grid(NUM_BLOCKS);                                                         \
  softmax_##packed_type##_kernel<NUM_THREADS_PER_BLOCK><<<grid, block>>>(        \
      reinterpret_cast<element_type*>(x.data_ptr()),                             \
      reinterpret_cast<element_type*>(y.data_ptr()),                             \
      reinterpret_cast<element_type*>(total.data_ptr()), N);                     \
}

TORCH_BINDING_SOFTMAX(f32,       torch::kFloat32,    float,    1)
TORCH_BINDING_SOFTMAX(f32x4,     torch::kFloat32,    float,    4)
TORCH_BINDING_SOFTMAX_V2(f32,    torch::kFloat32,    float,    1)
TORCH_BINDING_SOFTMAX_V2(f32x4,  torch::kFloat32,    float,    4)

// softmax per token
#define LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(_H_)      \
softmax_f32_per_token_kernel<(_H_)><<<grid, block>>>( \
      reinterpret_cast<float*>(x.data_ptr()),         \
      reinterpret_cast<float*>(y.data_ptr()),         \
      N);  

#define DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H) \
  dim3 block((H));                                  \
  dim3 grid((S));                                   \     
  switch ((H))                                      \
  {                                                 \
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

#define LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(_H_)         \
softmax_f32x4_per_token_kernel<(_H_)/4><<<                 \
      grid, block>>>(                                      \
      reinterpret_cast<float*>(x.data_ptr()),              \
      reinterpret_cast<float*>(y.data_ptr()),              \
      N);  

#define DISPATCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)      \
  dim3 block((H)/4);                                       \
  dim3 grid((S));                                          \
  switch ((H))                                             \
  {                                                        \
  case 64:                                                 \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(64)              \
    break;                                                 \
  case 128:                                                \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(128)             \
    break;                                                 \
  case 256:                                                \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(256)             \
    break;                                                 \
  case 512:                                                \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(512)             \
    break;                                                 \
  case 1024:                                               \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(1024)            \
    break;                                                 \
  default:                                                 \
    throw std::runtime_error(                              \
      "only support H: 64/128/256/512/1024");              \
    break;                                                 \
  } 

// safe softmax per token
#define LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(_H_)      \
safe_softmax_f32_per_token_kernel<(_H_)><<<grid, block>>>( \
      reinterpret_cast<float*>(x.data_ptr()),              \
      reinterpret_cast<float*>(y.data_ptr()),              \
      N);  

#define DISPATCH_SATE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H) \
  dim3 block((H));                                       \
  dim3 grid((S));                                        \
  switch ((H))                                           \
  {                                                      \
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

#define LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(_H_)    \
safe_softmax_f32x4_per_token_kernel<(_H_)/4><<<            \
      grid, block>>>(                                      \
      reinterpret_cast<float*>(x.data_ptr()),              \
      reinterpret_cast<float*>(y.data_ptr()),              \
      N);  

#define DISPATCH_SATE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H) \
  dim3 block((H)/4);                                       \
  dim3 grid((S));                                          \
  switch ((H))                                             \
  {                                                        \
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
  default:                                                 \
    throw std::runtime_error(                              \
      "only support H: 64/128/256/512/1024");              \
    break;                                                 \
  } 

// softmax per token
torch::Tensor softmax_f32_per_token(torch::Tensor x) {
  if((x.options().dtype() != (torch::kFloat32))) {             
    std::cout << "x Tensor Info:" << x.options() << std::endl;                   
    throw std::runtime_error("values must be torch::kFloat32");                        
  }                                                                              
  auto options = torch::TensorOptions().dtype((torch::kFloat32)).device(                 
    torch::kCUDA, 0);                                                            
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len  
  const int N = S * H;                                                 
  auto y = torch::zeros({S, H}, options).contiguous(); // [S,H]

  DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
  return y;                     
}

// no copy for y Tensor
void softmax_f32_per_token_v2(torch::Tensor x, torch::Tensor y) {
  if((x.options().dtype() != (torch::kFloat32))) {             
    std::cout << "x Tensor Info:" << x.options() << std::endl;                   
    throw std::runtime_error("values must be torch::kFloat32");                        
  }                                                
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len
  const int N = S * H; 
  if ((y.size(0) != S) || (y.size(1) != H)) {
    throw std::runtime_error("y Tensor size mismatch!");
  }

  DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
}

torch::Tensor softmax_f32x4_per_token(torch::Tensor x) {
  if((x.options().dtype() != (torch::kFloat32))) {                                       
    std::cout << "x Tensor Info:" << x.options() << std::endl;                   
    throw std::runtime_error("values must be torch::kFloat32");                        
  }                                                                              
  auto options = torch::TensorOptions().dtype((torch::kFloat32)).device(                 
    torch::kCUDA, 0);                                                            
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len  
  const int N = S * H; 
  auto y = torch::zeros({S, H}, options).contiguous(); // [S,H]

  DISPATCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)                        
  return y;                     
}

// no copy for y Tensor
void softmax_f32x4_per_token_v2(torch::Tensor x, torch::Tensor y) {
  if((x.options().dtype() != (torch::kFloat32))) {             
    std::cout << "x Tensor Info:" << x.options() << std::endl;                   
    throw std::runtime_error("values must be torch::kFloat32");                        
  }                                                                                                                                  
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len
  const int N = S * H; 
  if ((y.size(0) != S) || (y.size(1) != H)) {
    throw std::runtime_error("y Tensor size mismatch!");
  }

  DISPATCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)
}

// safe_softmax per token
torch::Tensor safe_softmax_f32_per_token(torch::Tensor x) {
  if((x.options().dtype() != (torch::kFloat32))) {                                       
    std::cout << "x Tensor Info:" << x.options() << std::endl;                   
    throw std::runtime_error("values must be torch::kFloat32");                        
  }                                                                              
  auto options = torch::TensorOptions().dtype((torch::kFloat32)).device(                 
    torch::kCUDA, 0);                                                            
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len
  const int N = S * H; 
  auto y = torch::zeros({S, H}, options).contiguous(); // [S,H]

  DISPATCH_SATE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)                        
  return y;                     
}

// no copy for y Tensor
void safe_softmax_f32_per_token_v2(torch::Tensor x, torch::Tensor y) {
  if((x.options().dtype() != (torch::kFloat32))) {             
    std::cout << "x Tensor Info:" << x.options() << std::endl;                   
    throw std::runtime_error("values must be torch::kFloat32");                        
  }                                                                                                                                  
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len
  const int N = S * H; 
  if ((y.size(0) != S) || (y.size(1) != H)) {
    throw std::runtime_error("y Tensor size mismatch!");
  }

  DISPATCH_SATE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
}

torch::Tensor safe_softmax_f32x4_per_token(torch::Tensor x) {
  if((x.options().dtype() != (torch::kFloat32))) {                                       
    std::cout << "x Tensor Info:" << x.options() << std::endl;                   
    throw std::runtime_error("values must be torch::kFloat32");                        
  }                                                                              
  auto options = torch::TensorOptions().dtype((torch::kFloat32)).device(                 
    torch::kCUDA, 0);                                                            
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len 
  const int N = S * H; 
  auto y = torch::zeros({S, H}, options).contiguous(); // [S,H]

  DISPATCH_SATE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)                        
  return y;                     
}

// no copy for y Tensor
void safe_softmax_f32x4_per_token_v2(torch::Tensor x, torch::Tensor y) {
  if((x.options().dtype() != (torch::kFloat32))) {             
    std::cout << "x Tensor Info:" << x.options() << std::endl;                   
    throw std::runtime_error("values must be torch::kFloat32");                        
  }                                                                                                                                  
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len
  const int N = S * H; 
  if ((y.size(0) != S) || (y.size(1) != H)) {
    throw std::runtime_error("y Tensor size mismatch!");
  }

  DISPATCH_SATE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32)
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32_v2)
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32x4_v2)
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32_per_token_v2)
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32x4_per_token)
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32x4_per_token_v2)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32_per_token_v2)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32x4_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32x4_per_token_v2)
}
