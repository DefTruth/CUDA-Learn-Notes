#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <torch/types.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define MAX_EXP_INPUT 88.3762626647949f
#define MIN_EXP_INPUT -88.3762626647949f

// -------------------------------------- FP32 -------------------------------------- 
// Sigmoid x: N, y: N y=1/(1+exp(-x))
// grid(N/256), block(K=256) 
__global__ void sigmoid_f32_kernel(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float v = x[idx];
    v = fminf(fmaxf(v, MIN_EXP_INPUT), MAX_EXP_INPUT);
    y[idx] = 1.0f / (1.0f + expf(-v));
  }
}

// Sigmoid x: N, y: N y=1/(1+exp(-x)) Vec4
// grid(N/256), block(256/4)
__global__ void sigmoid_f32x4_kernel(float* x, float* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    
    reg_x.x = fminf(fmaxf(reg_x.x, MIN_EXP_INPUT), MAX_EXP_INPUT);
    reg_x.y = fminf(fmaxf(reg_x.y, MIN_EXP_INPUT), MAX_EXP_INPUT);
    reg_x.z = fminf(fmaxf(reg_x.z, MIN_EXP_INPUT), MAX_EXP_INPUT);
    reg_x.w = fminf(fmaxf(reg_x.w, MIN_EXP_INPUT), MAX_EXP_INPUT);

    reg_y.x = 1.0f / (1.0f + expf(-reg_x.x));
    reg_y.y = 1.0f / (1.0f + expf(-reg_x.y));
    reg_y.z = 1.0f / (1.0f + expf(-reg_x.z));
    reg_y.w = 1.0f / (1.0f + expf(-reg_x.w));
    FLOAT4(y[idx]) = reg_y;
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

#define CHECK_TORCH_TENSOR_SHAPE(T, S0) \
if (((T).size(0) != (S0))) { throw std::runtime_error("Tensor size mismatch!"); }

#define TORCH_BINDING_SIGMOID(packed_type, th_type, element_type, n_elements)    \
torch::Tensor sigmoid_##packed_type(torch::Tensor x) {                           \
  CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                         \
  auto options = torch::TensorOptions().dtype((th_type)).device(                 \
    torch::kCUDA, 0);                                                            \
  const int N = x.size(0);                                                       \
  auto y = torch::zeros({N}, options);                                           \
  static const int NUM_THREADS_PER_BLOCK = 256 / (n_elements);                   \
  const int NUM_BLOCKS = (N + 256 - 1) / 256;                                    \
  dim3 block(NUM_THREADS_PER_BLOCK);                                             \
  dim3 grid(NUM_BLOCKS);                                                         \
  sigmoid_##packed_type##_kernel<<<grid, block>>>(                               \
      reinterpret_cast<element_type*>(x.data_ptr()),                             \
      reinterpret_cast<element_type*>(y.data_ptr()), N);                         \
  return y;                                                                      \
}

#define TORCH_BINDING_SIGMOID_V2(packed_type, th_type, element_type, n_elements) \
void sigmoid_##packed_type##_v2(torch::Tensor x, torch::Tensor y) {              \
  CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                         \
  CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                         \
  const int N = x.size(0);                                                       \
  CHECK_TORCH_TENSOR_SHAPE(y, N)                                                 \
  static const int NUM_THREADS_PER_BLOCK = 256 / (n_elements);                   \
  const int NUM_BLOCKS = (N + 256 - 1) / 256;                                    \
  dim3 block(NUM_THREADS_PER_BLOCK);                                             \
  dim3 grid(NUM_BLOCKS);                                                         \
  sigmoid_##packed_type##_kernel<<<grid, block>>>(                               \
      reinterpret_cast<element_type*>(x.data_ptr()),                             \
      reinterpret_cast<element_type*>(y.data_ptr()), N);                         \
}

TORCH_BINDING_SIGMOID(f32,       torch::kFloat32,    float,    1)
TORCH_BINDING_SIGMOID(f32x4,     torch::kFloat32,    float,    4)
TORCH_BINDING_SIGMOID_V2(f32,    torch::kFloat32,    float,    1)
TORCH_BINDING_SIGMOID_V2(f32x4,  torch::kFloat32,    float,    4)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f32)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f32_v2)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f32x4_v2)
}
