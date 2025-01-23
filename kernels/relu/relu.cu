#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/types.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

// -------------------------------------- FP32 -------------------------------------- 
// Relu x: N, y: N y=max(0,x)
// grid(N/256), block(K=256) 
__global__ void relu_f32_kernel(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = fmaxf(0.0f, x[idx]);
}

// Relu x: N, y: N y=max(0,x) Vec4
// grid(N/256/4), block(256/4) 
__global__ void relu_f32x4_kernel(float* x, float* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = fmaxf(0.0f, reg_x.x);
    reg_y.y = fmaxf(0.0f, reg_x.y);
    reg_y.z = fmaxf(0.0f, reg_x.z);
    reg_y.w = fmaxf(0.0f, reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}

// -------------------------------------- FP16 -------------------------------------- 
__global__ void relu_f16_kernel(half* x, half* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = __hmax(__float2half(0.0f), x[idx]);
}

__global__ void relu_f16x2_kernel(half* x, half* y, int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 reg_x = HALF2(x[idx]);
    half2 reg_y = HALF2(y[idx]);
    reg_y.x = __hmax(__float2half(0.0f), reg_x.x);
    reg_y.y = __hmax(__float2half(0.0f), reg_x.y);
    HALF2(y[idx]) = reg_y;
  }
}

__global__ void relu_f16x8_kernel(half* x, half* y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half2 reg_x_0 = HALF2(x[idx + 0]);
  half2 reg_x_1 = HALF2(x[idx + 2]);
  half2 reg_x_2 = HALF2(x[idx + 4]);
  half2 reg_x_3 = HALF2(x[idx + 6]);
  half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
  reg_y_0.x = __hmax(__float2half(0.0f), reg_x_0.x);
  reg_y_0.y = __hmax(__float2half(0.0f), reg_x_0.y);
  reg_y_1.x = __hmax(__float2half(0.0f), reg_x_1.x);
  reg_y_1.y = __hmax(__float2half(0.0f), reg_x_1.y);
  reg_y_2.x = __hmax(__float2half(0.0f), reg_x_2.x);
  reg_y_2.y = __hmax(__float2half(0.0f), reg_x_2.y);
  reg_y_3.x = __hmax(__float2half(0.0f), reg_x_3.x);
  reg_y_3.y = __hmax(__float2half(0.0f), reg_x_3.y);
  if ((idx + 0) < N) { HALF2(y[idx + 0]) = reg_y_0; }
  if ((idx + 2) < N) { HALF2(y[idx + 2]) = reg_y_1; }
  if ((idx + 4) < N) { HALF2(y[idx + 4]) = reg_y_2; }
  if ((idx + 6) < N) { HALF2(y[idx + 6]) = reg_y_3; }
}

__global__ void relu_f16x8_pack_kernel(half* x, half* y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  const half2 z2 = {__float2half(0.0f), __float2half(0.0f)};
  // temporary register(memory), .local space in ptx, addressable
  half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits

  #pragma unroll
  for (int i = 0; i < 8; i += 2) {
    // __hmax2 for half2 x 4
    HALF2(pack_y[i]) = __hmax2(HALF2(pack_x[i]), z2);
  } 
  // reinterpret as float4 and store 128 bits in 1 memory issue.
  if ((idx + 7) < N) { LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]); }
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

#define TORCH_BINDING_RELU(packed_type, th_type, element_type, n_elements)       \
void relu_##packed_type(torch::Tensor x, torch::Tensor y) {                      \
  CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                         \
  CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                         \
  const int ndim = x.dim();                                                      \
  if (ndim != 2) {                                                               \
    int N = 1;                                                                   \
    for (int i = 0; i < ndim; ++i) { N *= x.size(i); }                           \
    dim3 block(256 / (n_elements));                                              \
    dim3 grid((N + 256 - 1) / 256);                                              \
    relu_##packed_type##_kernel<<<grid, block>>>(                                \
      reinterpret_cast<element_type*>(x.data_ptr()),                             \
      reinterpret_cast<element_type*>(y.data_ptr()), N);                         \
  } else {                                                                       \
    const int S = x.size(0);                                                     \
    const int K = x.size(1);                                                     \
    const int N = S * K;                                                         \
    if ((K/(n_elements)) <= 1024) {                                              \
      dim3 block(K/(n_elements));                                                \
      dim3 grid(S);                                                              \
      relu_##packed_type##_kernel<<<grid, block>>>(                              \
        reinterpret_cast<element_type*>(x.data_ptr()),                           \
        reinterpret_cast<element_type*>(y.data_ptr()), N);                       \
    } else {                                                                     \
      int N = 1;                                                                 \
      for (int i = 0; i < ndim; ++i) { N *= x.size(i); }                         \
      dim3 block(256 / (n_elements));                                            \
      dim3 grid((N + 256 - 1) / 256);                                            \
      relu_##packed_type##_kernel<<<grid, block>>>(                              \
        reinterpret_cast<element_type*>(x.data_ptr()),                           \
        reinterpret_cast<element_type*>(y.data_ptr()), N);                       \
    }                                                                            \
  }                                                                              \
}


TORCH_BINDING_RELU(f32,        torch::kFloat32,    float,    1)
TORCH_BINDING_RELU(f32x4,      torch::kFloat32,    float,    4)
TORCH_BINDING_RELU(f16,        torch::kHalf,       half,     1)
TORCH_BINDING_RELU(f16x2,      torch::kHalf,       half,     2)
TORCH_BINDING_RELU(f16x8,      torch::kHalf,       half,     8)
TORCH_BINDING_RELU(f16x8_pack, torch::kHalf,       half,     8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(relu_f32)
  TORCH_BINDING_COMMON_EXTENSION(relu_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16x8_pack)
}
