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
#include "cublas_v2.h"

static cublasHandle_t g_handle = nullptr;

void init_cublas_handle() {
  if (g_handle == nullptr) {
    cublasStatus_t status = cublasCreate(&g_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("Failed to create cuBLAS handle: %d", status);
      exit(EXIT_FAILURE);
    }
    status = cublasSetMathMode(g_handle, CUBLAS_TENSOR_OP_MATH);
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("Failed to set cuBLAS Math Mode: %d", status);
      exit(EXIT_FAILURE);
    }
  }
}

void destroy_cublas_handle() {
  if (g_handle != nullptr) {
    cublasStatus_t status = cublasDestroy(g_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("Failed to destroy cuBLAS handle: %d", status);
    }
    g_handle = nullptr;
  }
}

// NN: A/B/C All row major
void cublas_tensor_op_nn(half *A, half *B, half *C,  size_t M, size_t N, size_t K) {

  static half alpha = 1.0;
  static half beta = 0.0;

  if (g_handle == nullptr) {
    init_cublas_handle();
  }

  cublasGemmEx(g_handle, 
               CUBLAS_OP_N, 
               CUBLAS_OP_N, 
               N, M, K, 
               &alpha, 
               B, CUDA_R_16F, N, 
               A, CUDA_R_16F, K, 
               &beta,  
               C, CUDA_R_16F, N, 
               CUBLAS_COMPUTE_16F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// TN: A row major MxK, B col major NxK, C row major MxN
void cublas_tensor_op_tn(half *A, half *B, half *C,  size_t M, size_t N, size_t K) {

  static half alpha = 1.0;
  static half beta = 0.0;

  if (g_handle == nullptr) {
    init_cublas_handle();
  }

  cublasGemmEx(g_handle, 
               CUBLAS_OP_T, 
               CUBLAS_OP_N, 
               N, M, K, 
               &alpha, 
               B, CUDA_R_16F, K, 
               A, CUDA_R_16F, K, 
               &beta,  
               C, CUDA_R_16F, N, 
               CUBLAS_COMPUTE_16F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// build cpp binary
#ifndef NO_CUBLAS_HGEMM_BIN

// pass the cuBLAS handle from outside to avoid error.
void cublas_tensor_op_tn_v2(cublasHandle_t handle, 
                            half *A, half *B, half *C,  
                            size_t M, size_t N, size_t K) {
  half alpha = 1.0;
  half beta = 0.0;

  cublasGemmEx(handle, 
               CUBLAS_OP_T, 
               CUBLAS_OP_N, 
               N, M, K, 
               &alpha, 
               B, CUDA_R_16F, K, 
               A, CUDA_R_16F, K, 
               &beta,  
               C, CUDA_R_16F, N, 
               CUBLAS_COMPUTE_16F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

float perf_cublas_tn(int M, int N, int K, int repeat) {
  size_t size_a = M * K * sizeof(half);
  size_t size_b = K * N * sizeof(half);
  size_t size_c = M * N * sizeof(half);

  half *d_a, *d_b;
  half *d_c;
  cudaMalloc(&d_a, size_a);
  cudaMalloc(&d_b, size_b);
  cudaMalloc(&d_c, size_c);

  cublasHandle_t handle = nullptr;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

  // warmup
  for (int i = 0; i < 10; ++i) {
    cublas_tensor_op_tn_v2(handle, d_a, d_b, d_c, M, N, K);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);

  for (int i = 0; i < repeat; i++) {
    cublas_tensor_op_tn_v2(handle, d_a, d_b, d_c, M, N, K);
  }

  cudaEventRecord(end);
  cudaDeviceSynchronize();
  cudaEventSynchronize(end);

  float msec, sec;
  cudaEventElapsedTime(&msec, start, end);
  sec = msec / 1000.0 / repeat;

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cublasDestroy(handle);

  return sec;
}

int main(int argc, char *argv[]) {
  const int test_num = 64;
  int M_list[test_num];
  int N_list[test_num];
  int K_list[test_num];

  for (int i = 0; i < test_num; i++) {
    M_list[i] = (i + 1) * 256;
    N_list[i] = (i + 1) * 256;
    K_list[i] = (i + 1) * 256;
  }

  const int outer_repeat = 10, inner_repeat = 1;

  printf("ALGO = cuBLAS CUBLAS_GEMM_DEFAULT_TENSOR_OP TN\n");

  for (int j = 0; j < test_num; j++) {
    int M = M_list[j], N = N_list[j], K = K_list[j];

    double max_sec = 0.0;
    double min_sec = DBL_MAX;
    double total_sec = 0.0;

    for (int k = 0; k < outer_repeat; k++) {
      double this_sec = perf_cublas_tn(M, N, K, inner_repeat);
      max_sec = max(max_sec, this_sec);
      min_sec = min(min_sec, this_sec);
      total_sec += this_sec;
    }

    // 1 TFLOPS = 10^12 FLOPS
    // ref: https://imgtec.eetrend.com/blog/2021/100062210.html.
    double avg_sec = total_sec / outer_repeat;
    double avg_Tflops = ((double)M) * N * K * 2 * 1e-12 / avg_sec;

    printf("M N K = %6d %6d %6d, ", M, N, K);
    printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
    printf("AVG Performance = %10.4lf Tflops\n", avg_Tflops);
  }

  return 0;
}
// build torch python binding
#else
// --------------------- PyTorch bindings for custom kernel -----------------------
#include <torch/types.h>
#include <torch/extension.h>

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

// NN: A/B/C All row major
void hgemm_cublas_tensor_op_nn(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  cublas_tensor_op_nn(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}

// TN: A row major MxK, B col major KxN, C row major MxN
void hgemm_cublas_tensor_op_tn(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  cublas_tensor_op_tn(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}
#endif
