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

#define WARP_SIZE 256
#define WARP_SIZE_S 16
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

// -------------------------------------- FP32 --------------------------------------
// col2row means read x[row][col] and write y[col][row]
// row2col means read x[col][row] and write y[row][col]
__global__ void mat_transpose_f32_col2row_kernel(
  float *x, float *y, const int row, const int col) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_row = global_idx / col;
  const int global_col = global_idx % col;
  if (global_idx < row * col) {
    y[global_col * row + global_row] = x[global_idx];
  }
}

__global__ void mat_transpose_f32_row2col_kernel(
  float *x, float *y, const int row, const int col) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_col = global_idx / row;
  const int global_row = global_idx % row;
  if (global_idx < row * col) {
    y[global_idx] = x[global_row * col + global_col];
  }
}

__global__ void mat_transpose_f32x4_col2row_kernel(
  float *x, float *y, const int row, const int col) {
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int global_col = (global_idx * 4) % col;
  int global_row = (global_idx * 4) / col;

  if (global_row < row && global_col + 3 < col) {
    float4 x_val = reinterpret_cast<float4 *>(x)[global_idx];

    y[global_col * row + global_row] = x_val.x;
    y[(global_col + 1) * row + global_row] = x_val.y;
    y[(global_col + 2) * row + global_row] = x_val.z;
    y[(global_col + 3) * row + global_row] = x_val.w;
  }
}
__global__ void mat_transpose_f32x4_row2col_kernel(
  float *x, float *y, const int row, const int col) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_col = (global_idx * 4) / row;
  const int global_row = (global_idx * 4) % row;

  if (global_row < row && global_col < col) {
    float4 x_val;
    x_val.x = x[global_row * col + global_col];
    x_val.y = x[(global_row + 1) * col + global_col];
    x_val.z = x[(global_row + 2) * col + global_col];
    x_val.w = x[(global_row + 3) * col + global_col];
    reinterpret_cast<float4 *>(y)[global_idx] = FLOAT4(x_val);
  }
}

// work for row == col
__global__ void mat_transpose_f32_diagonal2d_kernel(
  float *x, float *y, int row, int col) {
  const int block_y = blockIdx.x;
  const int block_x = (blockIdx.x + blockIdx.y) % gridDim.x;
  const int global_col = threadIdx.x + blockDim.x * block_x;
  const int global_row = threadIdx.y + blockDim.y * block_y;
  if (global_col < col && global_row < row) {
    y[global_row * col + global_col] = x[global_col * row + global_row];
  }
}

__global__ void mat_transpose_f32_col2row2d_kernel(
  float *x, float *y, const int row, const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_x < col && global_y < row) {
    y[global_x * row + global_y] = x[global_y * col + global_x];
  }
}

__global__ void mat_transpose_f32_row2col2d_kernel(
  float *x, float *y, const int row, const int col) {
  const int global_y = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_x = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_y < col && global_x < row) {
    y[global_y * row + global_x] = x[global_x * col + global_y];
  }
}

__global__ void mat_transpose_f32x4_col2row2d_kernel(
  float *x, float *y, const int row, const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_x * 4 + 3 < col && global_y < row) {
    float4 x_val = reinterpret_cast<float4 *>(x)[global_y * col / 4 + global_x];
    y[(global_x * 4) * row + global_y] = x_val.x;
    y[(global_x * 4 + 1) * row + global_y] = x_val.y;
    y[(global_x * 4 + 2) * row + global_y] = x_val.z;
    y[(global_x * 4 + 3) * row + global_y] = x_val.w;
  }
}
__global__ void mat_transpose_f32x4_row2col2d_kernel(
  float *x, float *y, const int row, const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_y * 4 + 3 < row && global_x < col) {
    float4 x_val;
    x_val.x = x[(global_y * 4) * col + global_x];
    x_val.y = x[(global_y * 4 + 1) * col + global_x];
    x_val.z = x[(global_y * 4 + 2) * col + global_x];
    x_val.w = x[(global_y * 4 + 3) * col + global_x];
    reinterpret_cast<float4 *>(y)[global_x * row / 4 + global_y] = FLOAT4(x_val);
  }
}

// TODO: may support shared memory optimize ?
__global__ void mat_transpose_f32x4_shared_col2row2d_kernel(
  float *x, float *y, const int row, const int col) {
  return;
}
__global__ void mat_transpose_f32x4_shared_row2col2d_kernel(
  float *x, float *y, const int row, const int col) {
  return;
}
__global__ void mat_transpose_f32x4_shared_bcf_col2row2d_kernel(
  float *x, float *y, const int row, const int col) {
  return;
}

// TODO: may support fp16 mat transpose ?

// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                   \
  if (((T).options().dtype() != (th_type)))                    \
  {                                                            \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type);      \
  }

#define TORCH_BINDING_MAT_TRANSPOSE(tag, th_type, element_type, n_pack) \
  void mat_transpose_##tag(torch::Tensor x, torch::Tensor y)            \
  {                                                                     \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                              \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                              \
    const int M = x.size(0);                                            \
    const int N = x.size(1);                                            \
    dim3 block(WARP_SIZE);                                              \
    dim3 grid(((N * M + WARP_SIZE - 1) / n_pack / WARP_SIZE));          \
    mat_transpose_##tag##_kernel<<<grid, block>>>(                      \
        reinterpret_cast<element_type *>(x.data_ptr()),                 \
        reinterpret_cast<element_type *>(y.data_ptr()), M, N);          \
  }

#define TORCH_BINDING_MAT_TRANSPOSE2D(tag, th_type, element_type, n_element_row, n_element_col) \
  void mat_transpose_##tag##2d(torch::Tensor x, torch::Tensor y)                                \
  {                                                                                             \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                                      \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                                      \
    const int M = x.size(0);                                                                    \
    const int N = x.size(1);                                                                    \
    dim3 block(WARP_SIZE_S, WARP_SIZE_S);                                                       \
    dim3 grid((N + WARP_SIZE_S - 1) / (WARP_SIZE_S * n_element_col),                            \
              (M + WARP_SIZE_S - 1) / (WARP_SIZE_S / n_element_row));                           \
    mat_transpose_##tag##2d_kernel <<<grid, block>>>(                                           \
      reinterpret_cast<element_type *>(x.data_ptr()),                                           \
      reinterpret_cast<element_type *>(y.data_ptr()), M, N);                                    \
  }

// 1d index
TORCH_BINDING_MAT_TRANSPOSE(f32_col2row, torch::kFloat32, float, 1)
TORCH_BINDING_MAT_TRANSPOSE(f32_row2col, torch::kFloat32, float, 1)
TORCH_BINDING_MAT_TRANSPOSE(f32x4_col2row, torch::kFloat32, float, 4)
TORCH_BINDING_MAT_TRANSPOSE(f32x4_row2col, torch::kFloat32, float, 4)
// 2d index. easier for diagonal 
TORCH_BINDING_MAT_TRANSPOSE2D(f32_col2row, torch::kFloat32, float, 1, 1)
TORCH_BINDING_MAT_TRANSPOSE2D(f32_row2col, torch::kFloat32, float, 1, 1)
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_col2row, torch::kFloat32, float, 1, 4)
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_row2col, torch::kFloat32, float, 4, 1)
// diagonal index method.
TORCH_BINDING_MAT_TRANSPOSE2D(f32_diagonal, torch::kFloat32, float, 1, 1)
// TODO: may support shared memory optimize ?
// TODO: may support fp16 mat transpose ?

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 1d index
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_col2row)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_col2row)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_row2col)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_row2col)
  // 2d index. easier for diagonal
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_col2row2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_col2row2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_row2col2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_row2col2d)
  // diagonal index method.
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_diagonal2d)
}
