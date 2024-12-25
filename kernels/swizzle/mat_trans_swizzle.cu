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
#include <iostream>


// reference: https://zhuanlan.zhihu.com/p/4746910252
// 转置前的矩阵存储在dev_A中，矩阵大小为M*N，转置后的数据存储在dev_B中
__global__ void mat_trans_smem_naive_kernel(int* dev_A, int M, int N, int* dev_B) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  // 每个block处理32*32的矩阵块
  __shared__ int s_data[32][32];

  if (row < M && col < N) {
    // 从全局内存中加载数据，转置后写到共享内存中
    s_data[threadIdx.x][threadIdx.y] = dev_A[row * N + col];
    __syncthreads();
    int n_col = blockIdx.y * blockDim.y + threadIdx.x;
    int n_row = blockIdx.x * blockDim.x + threadIdx.y;
    if (n_col < M && n_row < N) {
      // 从转置后的共享内存按行写到全局内存结果中
      dev_B[n_row * M + n_col] = s_data[threadIdx.y][threadIdx.x];
    }
  }
}

// reference: https://zhuanlan.zhihu.com/p/4746910252
__global__ void mat_trans_smem_padding_kernel(int* dev_A, int M, int N, int* dev_B) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // 每个block处理32*32的矩阵块，尾部padding来避免bank conflict
  __shared__ int s_data[32][33];

  if (row < M && col < N) {
    s_data[threadIdx.x][threadIdx.y] = dev_A[row * N + col];
    __syncthreads();
    int n_col = blockIdx.y * blockDim.y + threadIdx.x;
    int n_row = blockIdx.x * blockDim.x + threadIdx.y;
    if (n_col < M && n_row < N) {
      dev_B[n_row * M + n_col] = s_data[threadIdx.y][threadIdx.x];
    }
  }
}

// reference: https://zhuanlan.zhihu.com/p/4746910252
__global__ void mat_trans_smem_swizzle_kernel(int* dev_A, int M, int N, int* dev_B) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int s_data[32][32];

  if (row < M && col < N) {
    // 从全局内存读取数据写入共享内存的逻辑坐标(row=x,col=y)
    // 其映射的物理存储位置位置(row=x,col=x^y)
    s_data[threadIdx.x][threadIdx.x ^ threadIdx.y] = dev_A[row * N + col];
    __syncthreads();
    int n_col = blockIdx.y * blockDim.y + threadIdx.x;
    int n_row = blockIdx.x * blockDim.x + threadIdx.y;
    if (n_row < N && n_col < M) {
      // 从共享内存的逻辑坐标(row=y,col=x)读取数据
      // 其映射的物理存储位置(row=y,col=x^y)
      dev_B[n_row * M + n_col] = s_data[threadIdx.y][threadIdx.x ^ threadIdx.y];
    }
  }
}

int main(int argc, char *argv[]) {
  int M = 1024; 
  int N = 1024; 
  if (argc > 1) M = std::stoi(argv[1]);
  if (argc > 2) N = std::stoi(argv[2]);
  size_t size_a = M * N * sizeof(int);
  size_t size_b = M * N * sizeof(int);

  int* dev_A;
  int* dev_B;
  cudaMalloc(&dev_A, size_a);
  cudaMalloc(&dev_B, size_b);
  cudaDeviceSynchronize();

  dim3 block(32, 32);
  dim3 grid(N/32, M/32);

  mat_trans_smem_naive_kernel<<<grid, block>>>(dev_A, M, N, dev_B);
  cudaDeviceSynchronize();

  mat_trans_smem_padding_kernel<<<grid, block>>>(dev_A, M, N, dev_B);
  cudaDeviceSynchronize();

  mat_trans_smem_swizzle_kernel<<<grid, block>>>(dev_A, M, N, dev_B);
  cudaDeviceSynchronize();

  printf("Done.\n");
  cudaFree(dev_A);
  cudaFree(dev_B);

  return 0;
}
