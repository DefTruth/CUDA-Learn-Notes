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
#define MAX_EXP_F32  88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)
#define SQRT_2_PI M_SQRT2 * M_2_SQRTPI * 0.5f
#define HALF_1 __float2half(1.0f)
#define HALF_2 __float2half(2.0f)
#define HALF_DIV2 __float2half(0.5f)
// to clear the error among self defined gelu and pytorch gelu. Calculate $\sqrt{\frac{\pi}{2}}$ by $\sqrt{2 * \pi} / 2$
#define HALF_SQRT_2_PI __float2half(M_SQRT2) * __float2half(M_2_SQRTPI) * HALF_DIV2
#define HALF_V_APP __float2half(0.044715f)

#define HALF_GELU_OPS gelu_tanh_approximate
#define GELU_OPS gelu_tanh_approximate

// There is no half presicion operation like sinh, cosh, tanh. [Half Math Functions](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__FUNCTIONS.html#group__CUDA__MATH____HALF__FUNCTIONS)
// $$ tanh(x) = \frac{exp^{2x} - 1}{exp^{2x} + 1}$$ 
// But ops above will introduce error. 
// pytorch transform type while do tanh operator which include in the [pytorch/c10/util/BFloat16-math.h](https://github.com/pytorch/pytorch/blob/main/c10/util/BFloat16-math.h)
__inline__ __device__ half gelu_tanh_approximate(half x){
  half x_cube = x * x * x;
  // compute mid value : inner = 0.7978845608 * (x + 0.044715 * x * x * x)
  half inner = HALF_SQRT_2_PI * (x + HALF_V_APP * x_cube);
  // compute tanh
  return HALF_DIV2 * x * (HALF_1 + ((hexp(inner * HALF_2) - HALF_1) / (hexp(inner * HALF_2) + HALF_1))); 
}

__inline__ __device__ float gelu_tanh_approximate(float x){
  return 0.5f * x * (1.0f + tanhf(SQRT_2_PI * (x + 0.044715f * x * x * x)));
}

__inline__ __device__ float gelu_none_approximate(float x){
  return x * 0.5 * (1 + erff(x  * M_SQRT1_2));
}

// -------------------------------------- FP32 -------------------------------------- 
// GELU tanh approximate: x, y:x 0.5 * x * (1.0 + tanh(0.7978845608 * x * (1.0 + 0.044715 * x * x)))
// grid(N/256), block(K=256) 
__global__ void gelu_f32_kernel(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float v = fminf(fmaxf(x[idx], MIN_EXP_F32), MAX_EXP_F32);
    y[idx] = GELU_OPS(v);
  }
}

// GELU tanh approximate; Vec4
// grid(N/256), block(256/4)
__global__ void gelu_f32x4_kernel(float* x, float* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  float4 reg_x = FLOAT4(x[idx]);
  float4 reg_y;
    
  reg_x.x = fminf(fmaxf(reg_x.x, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.y = fminf(fmaxf(reg_x.y, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.z = fminf(fmaxf(reg_x.z, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.w = fminf(fmaxf(reg_x.w, MIN_EXP_F32), MAX_EXP_F32);

  reg_y.x = GELU_OPS(reg_x.x);
  reg_y.y = GELU_OPS(reg_x.y);
  reg_y.z = GELU_OPS(reg_x.z);
  reg_y.w = GELU_OPS(reg_x.w);

  if ((idx + 0) < N) { FLOAT4(y[idx]) = reg_y; }
}

// -------------------------------------- FP16 -------------------------------------- 
// GELU approximate: x, y:x 0.5 * x * (1.0 + tanh(0.7978845608 (x + 0.044715 * x * x * x))) Vec4
__global__ void gelu_f16_kernel(half* x, half* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    half v = x[idx];
    v = __hmin(__hmax(v, MIN_EXP_F16), MAX_EXP_F16);
    
    y[idx] = HALF_GELU_OPS(v);
  }
}

__global__ void gelu_f16x2_kernel(half* x, half* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

  half2 reg_x = HALF2(x[idx]);
  half2 reg_y;
  reg_x.x = __hmin(__hmax(reg_x.x, MIN_EXP_F16), MAX_EXP_F16);
  reg_x.y = __hmin(__hmax(reg_x.y, MIN_EXP_F16), MAX_EXP_F16);

  reg_y.x = HALF_GELU_OPS(reg_x.x);
  reg_y.y = HALF_GELU_OPS(reg_x.y);
  if ((idx + 0) < N) { HALF2(y[idx]) = reg_y; }
}

// unpack f16x8
__global__ void gelu_f16x8_kernel(half* x, half* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

  half2 reg_x_0 = HALF2(x[idx + 0]);
  half2 reg_x_1 = HALF2(x[idx + 2]);
  half2 reg_x_2 = HALF2(x[idx + 4]);
  half2 reg_x_3 = HALF2(x[idx + 6]);

  reg_x_0.x = __hmin(__hmax(reg_x_0.x, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_0.y = __hmin(__hmax(reg_x_0.y, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_1.x = __hmin(__hmax(reg_x_1.x, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_1.y = __hmin(__hmax(reg_x_1.y, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_2.x = __hmin(__hmax(reg_x_2.x, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_2.y = __hmin(__hmax(reg_x_2.y, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_3.x = __hmin(__hmax(reg_x_3.x, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_3.y = __hmin(__hmax(reg_x_3.y, MIN_EXP_F16), MAX_EXP_F16);

  half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;

  reg_x_0.x = HALF_GELU_OPS(reg_x_0.x);
  reg_x_0.y = HALF_GELU_OPS(reg_x_0.y);
  reg_x_1.x = HALF_GELU_OPS(reg_x_1.x);
  reg_x_1.y = HALF_GELU_OPS(reg_x_1.y);
  reg_x_2.x = HALF_GELU_OPS(reg_x_2.x);
  reg_x_2.y = HALF_GELU_OPS(reg_x_2.y);
  reg_x_3.x = HALF_GELU_OPS(reg_x_3.x);
  reg_x_3.y = HALF_GELU_OPS(reg_x_3.y);

  if ((idx + 0) < N) { HALF2(y[idx + 0]) = reg_x_0; }
  if ((idx + 2) < N) { HALF2(y[idx + 2]) = reg_x_1; }
  if ((idx + 4) < N) { HALF2(y[idx + 4]) = reg_x_2; }
  if ((idx + 6) < N) { HALF2(y[idx + 6]) = reg_x_3; }
}

// pack f16x8
__global__ void gelu_f16x8_pack_kernel(half* x, half* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  
  // temporary register(memory), .local space in ptx, addressable
  half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits
  
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    half v = __hmin(__hmax(pack_x[i], MIN_EXP_F16), MAX_EXP_F16);
    pack_y[i] = HALF_GELU_OPS(v);
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

#define TORCH_BINDING_GELU(packed_type, th_type, element_type, n_elements)       \
void gelu_##packed_type(torch::Tensor x, torch::Tensor y) {                      \
  CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                         \
  CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                         \
  const int ndim = x.dim();                                                      \
  if (ndim != 2) {                                                               \
    int N = 1;                                                                   \
    for (int i = 0; i < ndim; ++i) { N *= x.size(i); }                           \
    dim3 block(256 / (n_elements));                                              \
    dim3 grid((N + 256 - 1) / 256);                                              \
    gelu_##packed_type##_kernel<<<grid, block>>>(                                \
      reinterpret_cast<element_type*>(x.data_ptr()),                             \
      reinterpret_cast<element_type*>(y.data_ptr()), N);                         \
  } else {                                                                       \
    const int S = x.size(0);                                                     \
    const int K = x.size(1);                                                     \
    const int N = S * K;                                                         \
    if ((K/(n_elements)) <= 1024) {                                              \
      dim3 block(K/(n_elements));                                                \
      dim3 grid(S);                                                              \
      gelu_##packed_type##_kernel<<<grid, block>>>(                              \
        reinterpret_cast<element_type*>(x.data_ptr()),                           \
        reinterpret_cast<element_type*>(y.data_ptr()), N);                       \
    } else {                                                                     \
      int N = 1;                                                                 \
      for (int i = 0; i < ndim; ++i) { N *= x.size(i); }                         \
      dim3 block(256 / (n_elements));                                            \
      dim3 grid((N + 256 - 1) / 256);                                            \
      gelu_##packed_type##_kernel<<<grid, block>>>(                              \
        reinterpret_cast<element_type*>(x.data_ptr()),                           \
        reinterpret_cast<element_type*>(y.data_ptr()), N);                       \
    }                                                                            \
  }                                                                              \
}


TORCH_BINDING_GELU(f32,        torch::kFloat32,    float,    1)
TORCH_BINDING_GELU(f32x4,      torch::kFloat32,    float,    4)
TORCH_BINDING_GELU(f16,        torch::kHalf,       half,     1)
TORCH_BINDING_GELU(f16x2,      torch::kHalf,       half,     2)
TORCH_BINDING_GELU(f16x8,      torch::kHalf,       half,     8)
TORCH_BINDING_GELU(f16x8_pack, torch::kHalf,       half,     8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(gelu_f32)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16x8_pack)
}
