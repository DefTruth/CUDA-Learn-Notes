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

// 定义全局 LAMBD 值
#define LAMBD 0.5f

// 定义 CHECK_TORCH_TENSOR_DTYPE 宏
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
  if (((T).options().dtype() != (th_type))) {              \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("Tensor dtype must be " #th_type); \
  }

// 定义 TORCH_BINDING_COMMON_EXTENSION 宏
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func));



// HARDSHRINK 计算函数
// -------------------------------------- FP32 --------------------------------------
__device__ __forceinline__ float hardshrink(float x) {
  if (x > LAMBD || x < -LAMBD) {
    return x;
  } else {
    return 0;
  }
}


// -------------------------------------- FP16 --------------------------------------
__device__ __forceinline__ half hardshrink_half(half x) {
  if(x > __float2half(LAMBD) || x < __float2half(-LAMBD)) {
    return x;
  } else {
    return __float2half(0.f);
  }
}


// CUDA 核函数
// -------------------------------------- FP32 --------------------------------------
__global__ void hardshrink_f32_kernel(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = hardshrink(x[idx]);
}

__global__ void hardshrink_f32x4_kernel(float* x, float* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = hardshrink(reg_x.x);
    reg_y.y = hardshrink(reg_x.y);
    reg_y.z = hardshrink(reg_x.z);
    reg_y.w = hardshrink(reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}

// -------------------------------------- FP16 --------------------------------------
__global__ void hardshrink_f16_kernel(half* x, half* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = hardshrink_half(x[idx]);
}


__global__ void hardshrink_f16x2_kernel(half* x, half* y, int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 reg_x = HALF2(x[idx]);
    half2 reg_y;
    reg_y.x = hardshrink_half(reg_x.x);
    reg_y.y = hardshrink_half(reg_x.y);
    HALF2(y[idx]) = reg_y;
  }
}



__global__ void hardshrink_f16x8_kernel(half* x, half* y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half2 reg_x_0 = HALF2(x[idx + 0]);
  half2 reg_x_1 = HALF2(x[idx + 2]);
  half2 reg_x_2 = HALF2(x[idx + 4]);
  half2 reg_x_3 = HALF2(x[idx + 6]);
  half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
  reg_y_0.x = hardshrink_half(reg_x_0.x);
  reg_y_0.y = hardshrink_half(reg_x_0.y);
  reg_y_1.x = hardshrink_half(reg_x_1.x);
  reg_y_1.y = hardshrink_half(reg_x_1.y);
  reg_y_2.x = hardshrink_half(reg_x_2.x);
  reg_y_2.y = hardshrink_half(reg_x_2.y);
  reg_y_3.x = hardshrink_half(reg_x_3.x);
  reg_y_3.y = hardshrink_half(reg_x_3.y);
  if ((idx + 0) < N) { HALF2(y[idx + 0]) = reg_y_0; }
  if ((idx + 2) < N) { HALF2(y[idx + 2]) = reg_y_1; }
  if ((idx + 4) < N) { HALF2(y[idx + 4]) = reg_y_2; }
  if ((idx + 6) < N) { HALF2(y[idx + 6]) = reg_y_3; }
}



__global__ void hardshrink_f16x8_pack_kernel(half* x, half* y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half pack_x[8], pack_y[8];
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

  #pragma unroll
  for (int i = 0; i < 8; i++) {
    pack_y[i] = hardshrink_half(pack_x[i]);
  }
  if ((idx + 7) < N) { LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]); }
}

// PyTorch 绑定代码
#define TORCH_BINDING_HARDSHRINK(packed_type, th_type, element_type, n_elements)      \
void hardshrink_##packed_type(torch::Tensor x, torch::Tensor y) {                     \
  CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
  CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
  const int ndim = x.dim();                                                  \
  if (ndim != 2) {                                                           \
    int N = 1;                                                             \
    for (int i = 0; i < ndim; ++i) { N *= x.size(i); }                     \
    dim3 block(256 / (n_elements));                                        \
    dim3 grid((N + 256 - 1) / 256);                                        \
    hardshrink_##packed_type##_kernel<<<grid, block>>>(                           \
        reinterpret_cast<element_type*>(x.data_ptr()),                     \
        reinterpret_cast<element_type*>(y.data_ptr()), N);                 \
  } else {                                                                   \
    const int S = x.size(0);                                               \
    const int K = x.size(1);                                               \
    const int N = S * K;                                                   \
    if ((K/(n_elements)) <= 1024) {                                        \
      dim3 block(K/(n_elements));                                        \
      dim3 grid(S);                                                      \
      hardshrink_##packed_type##_kernel<<<grid, block>>>(                       \
        reinterpret_cast<element_type*>(x.data_ptr()),                 \
        reinterpret_cast<element_type*>(y.data_ptr()), N);             \
  } else {                                                               \
    int N = 1;                                                         \
    for (int i = 0; i < ndim; ++i) { N *= x.size(i); }                 \
    dim3 block(256 / (n_elements));                                    \
    dim3 grid((N + 256 - 1) / 256);                                    \
    hardshrink_##packed_type##_kernel<<<grid, block>>>(                       \
    reinterpret_cast<element_type*>(x.data_ptr()),                 \
    reinterpret_cast<element_type*>(y.data_ptr()), N);             \
    }                                                                      \
    }                                                                          \
}

TORCH_BINDING_HARDSHRINK(f32,        torch::kFloat32,    float,    1)
TORCH_BINDING_HARDSHRINK(f32x4,      torch::kFloat32,    float,    4)
TORCH_BINDING_HARDSHRINK(f16,        torch::kHalf,       half,     1)
TORCH_BINDING_HARDSHRINK(f16x2,      torch::kHalf,       half,     2)
TORCH_BINDING_HARDSHRINK(f16x8,      torch::kHalf,       half,     8)
TORCH_BINDING_HARDSHRINK(f16x8_pack, torch::kHalf,       half,     8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
TORCH_BINDING_COMMON_EXTENSION(hardshrink_f32)
TORCH_BINDING_COMMON_EXTENSION(hardshrink_f32x4)
TORCH_BINDING_COMMON_EXTENSION(hardshrink_f16)
TORCH_BINDING_COMMON_EXTENSION(hardshrink_f16x2)
TORCH_BINDING_COMMON_EXTENSION(hardshrink_f16x8)
TORCH_BINDING_COMMON_EXTENSION(hardshrink_f16x8_pack)
}