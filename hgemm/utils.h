#include <cuda_fp16.h>
#include <cstdlib>
#include <cuda.h>
#include <cublas_v2.h>

template <typename T>
float perf_gemm(
  void (*gpu_hgemm) (const T *, const T *, T *, int, int, int),
  int M, int N, int K, int repeat) {

  size_t size_a = M * K * sizeof(T);
  size_t size_b = K * N * sizeof(T);
  size_t size_c = M * N * sizeof(T);

  T *d_a, *d_b;
  T *d_c;
  cudaMalloc(&d_a, size_a);
  cudaMalloc(&d_b, size_b);
  cudaMalloc(&d_c, size_c);
  
  // warmup
  for (int i = 0; i < 10; ++i){
    gpu_hgemm(d_a, d_b, d_c, M, N, K);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    gpu_hgemm(d_a, d_b, d_c, M, N, K);
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

  return sec;
}


template <typename T>
float perf_gemm_swizzle(
  void (*gpu_hgemm) (const T *, const T *, T *, int, int, int, int),
  int M, int N, int K, int swizzle_stride, int repeat) {

  size_t size_a = M * K * sizeof(T);
  size_t size_b = K * N * sizeof(T);
  size_t size_c = M * N * sizeof(T);

  T *d_a, *d_b;
  T *d_c;
  cudaMalloc(&d_a, size_a);
  cudaMalloc(&d_b, size_b);
  cudaMalloc(&d_c, size_c);
  
  // warmup
  for (int i = 0; i < 10; ++i){
    gpu_hgemm(d_a, d_b, d_c, M, N, K, swizzle_stride);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    gpu_hgemm(d_a, d_b, d_c, M, N, K, swizzle_stride);
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

  return sec;
}


template <typename T>
float gemm_error_check_tn(
  void (*gpu_hgemm) (const T *, const T *, T *, int, int, int),
  int M, int N, int K) {

  size_t size_a = M * K * sizeof(T);
  size_t size_b = K * N * sizeof(T);
  size_t size_c = M * N * sizeof(T);

  T *h_a, *h_b, *h_c, *h_c_ref;
  T *d_a, *d_b, *d_c, *d_c_ref;

  h_a = (T *)malloc(size_a);
  h_b = (T *)malloc(size_b);
  h_c = (T *)malloc(size_c);
  h_c_ref = (T *)malloc(size_c);

  cudaMalloc(&d_a, size_a);
  cudaMalloc(&d_b, size_b);
  cudaMalloc(&d_c, size_c);
  cudaMalloc(&d_c_ref, size_c);

  srand(time(0));
  for (int i = 0; i < M * K; i++)
    h_a[i] = (T)((rand() % 200 - 100) * 0.01); // -1 ~ 1
  for (int i = 0; i < K * N; i++)
    h_b[i] = (T)((rand() % 200 - 100) * 0.01);

  cublasHandle_t handle;
  cublasCreate(&handle);
  half alpha = 1.f;
  half beta = 0.f;

  cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
  
  cublasHgemm(handle, 
              CUBLAS_OP_T, 
              CUBLAS_OP_N, 
              N, M, K,
              &alpha, 
              (half *)d_b, K, 
              (half *)d_a, K, 
              &beta, 
              (half *)d_c_ref, N);
        
  gpu_hgemm(d_a, d_b, d_c, M, N, K);

  cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c_ref, d_c_ref, size_c, cudaMemcpyDeviceToHost);

  float max_error = 0.0;
  for (int i = 0; i < M * N; i++) {
    float this_error = abs((float)h_c_ref[i] - (float)h_c[i]);
    max_error = max(max_error, this_error);
  }

  free(h_a); 
  free(h_b); 
  free(h_c); 
  free(h_c_ref);
  cudaFree(d_a); 
  cudaFree(d_b); 
  cudaFree(d_c); 
  cudaFree(d_c_ref);
  cublasDestroy(handle);
  
  return max_error;
}

template <typename T>
float gemm_error_check_tn_swizzle(
  void (*gpu_hgemm) (const T *, const T *, T *, int, int, int, int),
  int M, int N, int K, int swizzle_stride) {

  size_t size_a = M * K * sizeof(T);
  size_t size_b = K * N * sizeof(T);
  size_t size_c = M * N * sizeof(T);

  T *h_a, *h_b, *h_c, *h_c_ref;
  T *d_a, *d_b, *d_c, *d_c_ref;

  h_a = (T *)malloc(size_a);
  h_b = (T *)malloc(size_b);
  h_c = (T *)malloc(size_c);
  h_c_ref = (T *)malloc(size_c);

  cudaMalloc(&d_a, size_a);
  cudaMalloc(&d_b, size_b);
  cudaMalloc(&d_c, size_c);
  cudaMalloc(&d_c_ref, size_c);

  srand(time(0));
  for (int i = 0; i < M * K; i++)
    h_a[i] = (T)((rand() % 200 - 100) * 0.01); // -1 ~ 1
  for (int i = 0; i < K * N; i++)
    h_b[i] = (T)((rand() % 200 - 100) * 0.01);

  cublasHandle_t handle;
  cublasCreate(&handle);
  half alpha = 1.f;
  half beta = 0.f;

  cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
  
  cublasHgemm(handle, 
              CUBLAS_OP_T, 
              CUBLAS_OP_N, 
              N, M, K,
              &alpha, 
              (half *)d_b, K, 
              (half *)d_a, K, 
              &beta, 
              (half *)d_c_ref, N);
        
  gpu_hgemm(d_a, d_b, d_c, M, N, K, swizzle_stride);

  cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c_ref, d_c_ref, size_c, cudaMemcpyDeviceToHost);

  float max_error = 0.0;
  for (int i = 0; i < M * N; i++) {
    float this_error = abs((float)h_c_ref[i] - (float)h_c[i]);
    max_error = max(max_error, this_error);
  }

  free(h_a); 
  free(h_b); 
  free(h_c); 
  free(h_c_ref);
  cudaFree(d_a); 
  cudaFree(d_b); 
  cudaFree(d_c); 
  cudaFree(d_c_ref);
  cublasDestroy(handle);
  
  return max_error;
}

template <typename T>
float gemm_error_check_nn(
  void (*gpu_hgemm) (const T *, const T *, T *, int, int, int),
  int M, int N, int K) {

  size_t size_a = M * K * sizeof(T);
  size_t size_b = K * N * sizeof(T);
  size_t size_c = M * N * sizeof(T);

  T *h_a, *h_b, *h_c, *h_c_ref;
  T *d_a, *d_b, *d_c, *d_c_ref;

  h_a = (T *)malloc(size_a);
  h_b = (T *)malloc(size_b);
  h_c = (T *)malloc(size_c);
  h_c_ref = (T *)malloc(size_c);

  cudaMalloc(&d_a, size_a);
  cudaMalloc(&d_b, size_b);
  cudaMalloc(&d_c, size_c);
  cudaMalloc(&d_c_ref, size_c);

  srand(time(0));
  for (int i = 0; i < M * K; i++)
    h_a[i] = (T)((rand() % 200 - 100) * 0.01); // -1 ~ 1
  for (int i = 0; i < K * N; i++)
    h_b[i] = (T)((rand() % 200 - 100) * 0.01);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  half alpha = 1.f;
  half beta = 0.f;

  cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
  
  cublasGemmEx(handle, 
               CUBLAS_OP_N, 
               CUBLAS_OP_N, 
               N, M, K, 
               &alpha, 
               (half *)d_b, CUDA_R_16F, N, 
               (half *)d_a, CUDA_R_16F, K, 
               &beta,  
               (half *)d_c_ref, CUDA_R_16F, N, 
               CUBLAS_COMPUTE_16F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        
  gpu_hgemm(d_a, d_b, d_c, M, N, K);

  cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c_ref, d_c_ref, size_c, cudaMemcpyDeviceToHost);

  float max_error = 0.0;
  for (int i = 0; i < M * N; i++) {
    float this_error = abs((float)h_c_ref[i] - (float)h_c[i]);
    max_error = max(max_error, this_error);
  }

  free(h_a); 
  free(h_b); 
  free(h_c); 
  free(h_c_ref);
  cudaFree(d_a); 
  cudaFree(d_b); 
  cudaFree(d_c); 
  cudaFree(d_c_ref);
  cublasDestroy(handle);
  
  return max_error;
}
