#include <cuda_fp16.h>
#include <cstdlib>
#include <cuda.h>
#include <cublas_v2.h>
// modified from: https://github.com/weishengying/cute_gemm/blob/main/utils.h

#define OFFSET(row_idx, col_idx, stride_0, stride_1) \
  row_idx*stride_0 + col_idx*stride_1

#define PRINT(name, content) \
  print(name);               \
  print(" : ");              \
  print(content);            \
  print("\n");

#define PRINTTENSOR(name, content) \
  print(name);                     \
  print(" : ");                    \
  print_tensor(content);           \
  print("\n");
  
template<class T>
void cpu_hgemm(const T* A, const T* B, T* C,
         const int M, const int N, const int K) {
  // A(M,K):(K,1)     B(K,N):(1,K)
  for(int m = 0; m < M; m++) {
    for(int n = 0;  n < N; n++) {
      float tmp = 0.0;
      for(int k = 0; k < K; k++) {
        tmp += float(A[OFFSET(m, k, K, 1)]) * float(B[OFFSET(k, n, 1, K)]); 
      }
      C[OFFSET(m, n, N, 1)] = T(tmp);
    }
  }
  return;
}

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
float gemm_error_check(
  void (*gpu_hgemm) (const T *, const T *, T *, int, int, int),
  int M, int N, int K) {

  size_t size_a = M * K * sizeof(T);
  size_t size_b = K * N * sizeof(T);
  size_t size_c = M * N * sizeof(T);

  T *h_a, *h_b, *d_a, *d_b;
  T *h_c, *d_c, *h_d_c;

  h_a = (T *)malloc(size_a);
  h_b = (T *)malloc(size_b);
  h_c = (T *)malloc(size_c);
  cudaMalloc(&d_a, size_a);
  cudaMalloc(&d_b, size_b);
  cudaMalloc(&d_c, size_c);

  h_d_c = (T *)malloc(size_c);

  srand(time(0));
  for (int i = 0; i < M * K; i++)
    h_a[i] = (T)((rand() % 200 - 100) * 0.01); // -1 ~ 1
  for (int i = 0; i < K * N; i++)
    h_b[i] = (T)((rand() % 200 - 100) * 0.01);

  cpu_hgemm(h_a, h_b, h_c, M, N, K);

  cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

  gpu_hgemm(d_a, d_b, d_c, M, N, K);

  cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

  float max_error = 0.0;
  for (int i = 0; i < M * N; i++) {
    float this_error = abs((float)h_d_c[i] - (float)h_c[i]);
    max_error = max(max_error, this_error);
  }

  free(h_a); 
  free(h_b); 
  free(h_c); 
  cudaFree(d_a); 
  cudaFree(d_b); 
  cudaFree(d_c); 
  free(h_d_c);

  return max_error;
}

template <typename T>
float gemm_error_check_v2(
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
