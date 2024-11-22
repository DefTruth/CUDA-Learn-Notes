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

#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define BLOCK_SIZE 256
#define theta 10000.0f

__global__ void rope_f32_kernel(float* x, float* out, int seq_len, int N){ 
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float x1 = x[idx * 2];
  float x2 = x[idx * 2 + 1];
  int token_pos = idx / N; 
  int token_idx = idx % N;
  float exp_v = 1.0f / powf(theta, token_idx / (N * 2));
  float sin_v = sinf(token_pos / exp_v);
  float cos_v = cosf(token_pos / exp_v);
  float out1 = x1 * cos_v - x2 * sin_v;
  float out2 = x1 * sin_v + x2 * cos_v;
  out[idx * 2] = out1;
  out[idx * 2 + 1] = out2;
}

// another index method of rope.
__global__ void rope_f32_v2_kernel(float* x, float* out, int seq_len, int N){ 
  int token_pos = blockIdx.x;
  int tid = threadIdx.x;
  float x1 = x[token_pos * N * 2 + tid * 2];
  float x2 = x[token_pos * N * 2 + tid * 2 + 1];
  float exp_v = 1.0f / powf(theta, (int)(tid / 2) / (N * 2));
  float sin_v = sinf(token_pos / exp_v);
  float cos_v = cosf(token_pos / exp_v);
  float out1 = x1 * cos_v - x2 * sin_v;
  float out2 = x1 * sin_v + x2 * cos_v;
  out[token_pos * N * 2 + tid * 2] = out1;
  out[token_pos * N * 2 + tid * 2 + 1] = out2;
}

__global__ void rope_f32x4_pack_kernel(float* x, float* out, int seq_len, int N){ 
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float4 x_v = FLOAT4(x[idx * 4]);
  int token_pos = idx / N; 
  int token_idx = idx % N;
  float exp_f_v = 1.0f / powf(theta, token_idx * 2 / (N * 4));
  float exp_s_v = 1.0f / powf(theta, ((token_idx * 2) + 1) / (N * 4));
  float sin_f_v = sinf(token_pos / exp_f_v);
  float cos_f_v = cosf(token_pos / exp_f_v);
  float sin_s_v = sinf(token_pos / exp_s_v);
  float cos_s_v = cosf(token_pos / exp_s_v);
  float4 out_v;
  out_v.x = x_v.x * cos_f_v - x_v.y * sin_f_v;
  out_v.y = x_v.x * sin_f_v + x_v.y * cos_f_v;
  out_v.z = x_v.z * cos_s_v - x_v.w * sin_s_v;
  out_v.w = x_v.z * sin_s_v + x_v.w * cos_s_v; 
  FLOAT4(out[idx * 4]) = out_v;
}

// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
if (((T).options().dtype() != (th_type))) {                  \
  std::cout << "Tensor Info:" << (T).options() << std::endl; \
  throw std::runtime_error("values must be " #th_type);      \
}

void rope_f32(torch::Tensor x, torch::Tensor out) {
  CHECK_TORCH_TENSOR_DTYPE(x,   torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(out, torch::kFloat32)
  int seq_len = x.size(0);
  int hidden_size = x.size(1);
  int N = (int)(hidden_size/2);
  dim3 grid((seq_len * N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 block(BLOCK_SIZE);
  rope_f32_kernel<<<grid, block>>>(
    x.data_ptr<float>(), out.data_ptr<float>(), seq_len, N);
}

void rope_f32_v2(torch::Tensor x, torch::Tensor out) {
  CHECK_TORCH_TENSOR_DTYPE(x,   torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(out, torch::kFloat32)
  int seq_len = x.size(0);
  int hidden_size = x.size(1);
  int N = (int)(hidden_size/2);
  dim3 grid(seq_len);
  dim3 block(N);
  rope_f32_v2_kernel<<<grid, block>>>(
    x.data_ptr<float>(), out.data_ptr<float>(), seq_len, N);
}

void rope_f32x4_pack(torch::Tensor x, torch::Tensor out) {
  CHECK_TORCH_TENSOR_DTYPE(x,   torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(out, torch::kFloat32)
  int seq_len = x.size(0);
  int hidden_size = x.size(1);
  int N = (int)(hidden_size/4);
  dim3 grid((seq_len * N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 block(BLOCK_SIZE);
  rope_f32x4_pack_kernel<<<grid, block>>>(
    x.data_ptr<float>(), out.data_ptr<float>(), seq_len, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(rope_f32)
  TORCH_BINDING_COMMON_EXTENSION(rope_f32_v2)
  TORCH_BINDING_COMMON_EXTENSION(rope_f32x4_pack)
}
