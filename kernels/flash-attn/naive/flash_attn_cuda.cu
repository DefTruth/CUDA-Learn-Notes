// Modified from: https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu  
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/types.h>
#include <torch/extension.h>

#define ENABLE_NOTE_LOG 0
#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])


__global__ void flash_attn_1_fwd_f32_kernel(
  const float* Q, 
  const float* K, 
  const float* V, 
  const int N, 
  const int d,
  const int Tc,
  const int Tr, 
  const int Bc, 
  const int Br, 
  const float scale,
  float* l, 
  float *m, 
  float* O) {
  int tx = threadIdx.x;
  int bx = blockIdx.x; 
  int by = blockIdx.y;  // batch and head index

  // Offset into Q,K,V,O,l,m - different for each batch and head
  int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
  int lm_offset  = (bx * gridDim.y * N) + (by * N);  // offset for l and m

  // Define SRAM for Q,K,V,S
  extern __shared__ float sram[];
  int tile_size = Bc * d;  // size of Qi, Kj, Vj
  float* Qi = sram;
  float* Kj = &sram[tile_size];
  float* Vj = &sram[tile_size * 2];
  float* S = &sram[tile_size * 3];

  for (int j = 0; j < Tc; j++) {

    // Load Kj, Vj to SRAM
    #pragma unroll
    for (int x = 0; x < d; x++) {
      Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
      Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
    }
    __syncthreads();  // such that the inner loop can use the correct Kj, Vj

    #pragma unroll
    for (int i = 0; i < Tr; i++)  {

      // Load Qi to SRAM, l and m to registers
      #pragma unroll
      for (int x = 0; x < d; x++) {
        Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
      }
      float row_m_prev = m[lm_offset + (Br * i) + tx];
      float row_l_prev = l[lm_offset + (Br * i) + tx];

      // S = QK^T, row_m = rowmax(S)
      float row_m = -INFINITY;
      #pragma unroll
      for (int y = 0; y < Bc; y++) {
        float sum = 0;
        #pragma unroll
        for (int x = 0; x < d; x++) {
          sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
        }
        sum *= scale;
        S[(Bc * tx) + y] = sum;

        if (sum > row_m)
          row_m = sum;
      }

      // P = exp(S - row_m), row_l = rowsum(P)
      float row_l = 0;
      #pragma unroll
      for (int y = 0; y < Bc; y++) {
        S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
        row_l += S[(Bc * tx) + y];
      }

      // Compute new m and l
      float row_m_new = max(row_m_prev, row_m);
      float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) \
                      + (__expf(row_m - row_m_new) * row_l);

      // Write O, l, m to HBM
      #pragma unroll
      for (int x = 0; x < d; x++) {
        float pv = 0;  // Pij * Vj
        #pragma unroll
        for (int y = 0; y < Bc; y++) {
          pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
        }
        O[qkv_offset + (tile_size * i) + (tx * d) + x] = \
          (1 / row_l_new) * ((row_l_prev * __expf(row_m_prev - row_m_new) \
          * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
          + (__expf(row_m - row_m_new) * pv));
      }
      m[lm_offset + (Br * i) + tx] = row_m_new;
      l[lm_offset + (Br * i) + tx] = row_l_new;
    }
    __syncthreads();
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

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)             \
if (((T2).size(0) != (T1).size(0)) ||                \
    ((T2).size(1) != (T1).size(1)) ||                \
    ((T2).size(2) != (T1).size(2)) ||                \
    ((T2).size(3) != (T1).size(3))) {                \
  throw std::runtime_error("Tensor size mismatch!"); \
}

void flash_attn_1_fwd_f32(
  torch::Tensor Q, 
  torch::Tensor K, 
  torch::Tensor V,
  torch::Tensor O) {
  CHECK_TORCH_TENSOR_DTYPE(Q, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(K, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(V, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(O, torch::kFloat32)
  // TODO: determine Bc, Br dynamically
  const int Bc = 32; 
  const int Br = 32;
  // batch_size, n_head, seq_len, head_dim (B,nh,N,d)
  const int B = Q.size(0); 
  const int nh = Q.size(1);
  const int N = Q.size(2); 
  const int d = Q.size(3);
  CHECK_TORCH_TENSOR_SHAPE(K, Q)
  CHECK_TORCH_TENSOR_SHAPE(V, Q)
  CHECK_TORCH_TENSOR_SHAPE(O, Q)
  const int Tc = ceil((float) N / Bc); 
  const int Tr = ceil((float) N / Br);
  const float scale = 1.0 / sqrt(d);
  
  // Initialize O, l, m to HBM
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
  auto l = torch::zeros({B, nh, N}, options); 
  auto m = torch::full({B, nh, N}, -INFINITY, options);
  
  // Calculate SRAM size needed per block
  const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
#if ENABLE_NOTE_LOG
  printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);
#endif 
  dim3 grid(B, nh);  // batch_size x num_heads
  dim3 block(Bc);  // Bc threads per block
  
  flash_attn_1_fwd_f32_kernel<<<grid, block, sram_size>>>(
    reinterpret_cast<float*>(Q.data_ptr()), 
    reinterpret_cast<float*>(K.data_ptr()), 
    reinterpret_cast<float*>(V.data_ptr()), 
    N, 
    d, 
    Tc, 
    Tr, 
    Bc, 
    Br, 
    scale,
    reinterpret_cast<float*>(l.data_ptr()), 
    reinterpret_cast<float*>(m.data_ptr()), 
    reinterpret_cast<float*>(O.data_ptr())
  );
}
  

