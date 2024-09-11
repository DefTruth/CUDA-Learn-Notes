#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cuda_runtime.h>
#include <torch/types.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

// Histogram
// grid(N/256), block(256)
// a: Nx1, y: count histogram, a >= 1
__global__ void histogram_i32_kernel(int* a, int* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) atomicAdd(&(y[a[idx]]), 1);
}

// Histogram + Vec4
// grid(N/256), block(256/4)
// a: Nx1, y: count histogram, a >= 1
__global__ void histogram_i32x4_kernel(int* a, int* y, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    int4 reg_a = INT4(a[idx]);
    atomicAdd(&(y[reg_a.x]), 1);
    atomicAdd(&(y[reg_a.y]), 1);
    atomicAdd(&(y[reg_a.z]), 1);
    atomicAdd(&(y[reg_a.w]), 1);
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

#define TORCH_BINDING_HIST(packed_type, th_type, element_type, n_elements)       \
torch::Tensor histogram_##packed_type(torch::Tensor a) {                         \
  CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                         \
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(             \
    torch::kCUDA, 0);                                                            \
  const int N = a.size(0);                                                       \
  std::tuple<torch::Tensor, torch::Tensor> max_a = torch::max(a, 0);             \
  torch::Tensor max_val = std::get<0>(max_a).cpu();                              \
  const int M = max_val.item().to<int>();                                        \
  auto y = torch::zeros({M+1}, options);                                         \
  static const int NUM_THREADS_PER_BLOCK = 256 / (n_elements);                   \
  const int NUM_BLOCKS = (N + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;\
  dim3 block(NUM_THREADS_PER_BLOCK);                                             \
  dim3 grid(NUM_BLOCKS);                                                         \
  histogram_##packed_type##_kernel<<<grid, block>>>(                             \
      reinterpret_cast<element_type*>(a.data_ptr()),                             \
      reinterpret_cast<element_type*>(y.data_ptr()), N);                         \
  return y;                                                                      \
}

TORCH_BINDING_HIST(i32,   torch::kInt32, int, 1)
TORCH_BINDING_HIST(i32x4, torch::kInt32, int, 4)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(histogram_i32)
  TORCH_BINDING_COMMON_EXTENSION(histogram_i32x4)
}