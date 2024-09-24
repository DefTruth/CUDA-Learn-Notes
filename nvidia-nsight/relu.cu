#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

// -------------------------------------- FP32 -------------------------------------- 
// Relu x: N, y: N y=max(0,x)
// grid(N/256), block(K=256) 
__global__ void relu_f32_kernel(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = fmaxf(0.0f, x[idx]);
}

// Relu x: N, y: N y=max(0,x) Vec4
// grid(N/256/4), block(256/4) 
__global__ void relu_f32x4_kernel(float* x, float* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = fmaxf(0.0f, reg_x.x);
    reg_y.y = fmaxf(0.0f, reg_x.y);
    reg_y.z = fmaxf(0.0f, reg_x.z);
    reg_y.w = fmaxf(0.0f, reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}

// -------------------------------------- FP16 -------------------------------------- 
__global__ void relu_f16_kernel(half* x, half* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = __hmax(__float2half(0.0f), x[idx]);
}

__global__ void relu_f16x2_kernel(half* x, half* y, int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 reg_x = HALF2(x[idx]);
    half2 reg_y = HALF2(y[idx]);
    reg_y.x = __hmax(__float2half(0.0f), reg_x.x);
    reg_y.y = __hmax(__float2half(0.0f), reg_x.y);
    HALF2(y[idx]) = reg_y;
  }
}

__global__ void relu_f16x8_kernel(half* x, half* y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  // manual unroll and improve L2 cache hit rate.
  // Only   L2 cache: load 32  bytes in 1 memory issue (default)
  // Enable L1 cache: load 128 bytes in 1 memory issue (-Xptxas -dlcm=ca)
  // why try fp16x8 within 1 threads? ref: https://zhuanlan.zhihu.com/p/641639133
  // 0. first, tid_0 load 32 bytes in 1 memory issue and cache data into L2 cache.
  // 1. then, tid_1,...,tid_3 hit L2 cache and load data from L2 cache directly.
  half2 reg_x_0 = HALF2(x[idx + 0]);
  half2 reg_x_1 = HALF2(x[idx + 2]);
  half2 reg_x_2 = HALF2(x[idx + 4]);
  half2 reg_x_3 = HALF2(x[idx + 6]);
  half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
  reg_y_0.x = __hmax(__float2half(0.0f), reg_x_0.x);
  reg_y_0.y = __hmax(__float2half(0.0f), reg_x_0.y);
  reg_y_1.x = __hmax(__float2half(0.0f), reg_x_1.x);
  reg_y_1.y = __hmax(__float2half(0.0f), reg_x_1.y);
  reg_y_2.x = __hmax(__float2half(0.0f), reg_x_2.x);
  reg_y_2.y = __hmax(__float2half(0.0f), reg_x_2.y);
  reg_y_3.x = __hmax(__float2half(0.0f), reg_x_3.x);
  reg_y_3.y = __hmax(__float2half(0.0f), reg_x_3.y);
  if ((idx + 0) < N) { HALF2(y[idx + 0]) = reg_y_0; }
  if ((idx + 2) < N) { HALF2(y[idx + 2]) = reg_y_1; }
  if ((idx + 4) < N) { HALF2(y[idx + 4]) = reg_y_2; }
  if ((idx + 6) < N) { HALF2(y[idx + 6]) = reg_y_3; }
}

__global__ void relu_f16x8_pack_kernel(half* x, half* y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  const half2 z2 = {__float2half(0.0f), __float2half(0.0f)};
  // temporary register(memory), .local space in ptx, addressable
  half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits

  #pragma unroll
  for (int i = 0; i < 8; i += 2) {
    // __hmax2 for half2 x 4
    HALF2(pack_y[i]) = __hmax2(HALF2(pack_x[i]), z2);
  } 
  // reinterpret as float4 and store 128 bits in 1 memory issue.
  if ((idx + 7) < N) { LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]); }
}

