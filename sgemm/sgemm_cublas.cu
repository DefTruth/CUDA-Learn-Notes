#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>

#include <torch/types.h>
#include <torch/extension.h>

#include "cublas_v2.h"


void cublas_sgemm(float *A, float *B, float *C,  size_t M, 
                  size_t N, size_t K) {

  cublasHandle_t handle = nullptr;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

  static float alpha = 1.0;
  static float beta = 0.0;

  cublasGemmEx(handle, 
               CUBLAS_OP_N, 
               CUBLAS_OP_N, 
               N, M, K, 
               &alpha, 
               B, CUDA_R_32F, N, 
               A, CUDA_R_32F, K, 
               &beta,  
               C, CUDA_R_32F, N, 
               CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT);
}

void cublas_sgemm_tf32(float *A, float *B, float *C,  size_t M, 
                       size_t N, size_t K) {

  cublasHandle_t handle = nullptr;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

  static float alpha = 1.0;
  static float beta = 0.0;

  cublasGemmEx(handle, 
               CUBLAS_OP_N, 
               CUBLAS_OP_N, 
               N, M, K, 
               &alpha, 
               B, CUDA_R_32F, N, 
               A, CUDA_R_32F, K, 
               &beta,  
               C, CUDA_R_32F, N, 
               CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
if(((T).options().dtype() != (th_type))) {                   \
  std::cout << "Tensor Info:" << (T).options() << std::endl; \
  throw std::runtime_error("values must be "#th_type);       \
}

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)           \
if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
  throw std::runtime_error("Tensor size mismatch!");  \
}

void sgemm_cublas(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  cublas_sgemm(
    reinterpret_cast<float*>(a.data_ptr()),
    reinterpret_cast<float*>(b.data_ptr()),
    reinterpret_cast<float*>(c.data_ptr()),
    M, N, K
  );
}

// cublas tensor op
void sgemm_cublas_tf32(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  cublas_sgemm_tf32(
    reinterpret_cast<float*>(a.data_ptr()),
    reinterpret_cast<float*>(b.data_ptr()),
    reinterpret_cast<float*>(c.data_ptr()),
    M, N, K
  );
}
