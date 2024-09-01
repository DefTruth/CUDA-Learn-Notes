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

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

// -------------------------------------- FP32 -------------------------------------- 
// Sigmoid x: N, y: N y=1/(1+exp(-x))
// grid(N/256), block(K=256) 
__global__ void sigmoid_f32(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = 1.0f / (1.0f + expf(-x[idx]));
}

// Sigmoid x: N, y: N y=1/(1+exp(-x)) Vec4
// grid(N/256), block(256/4)
__global__ void sigmoid_f32x4(float* x, float* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = 1.0f / (1.0f + expf(-reg_x.x));
    reg_y.y = 1.0f / (1.0f + expf(-reg_x.y));
    reg_y.z = 1.0f / (1.0f + expf(-reg_x.z));
    reg_y.w = 1.0f / (1.0f + expf(-reg_x.w));
    FLOAT4(y[idx]) = reg_y;
  }
}
