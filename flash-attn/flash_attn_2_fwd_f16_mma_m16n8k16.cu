// modified from: https://github.com/Byeong-Chan/flash-attention-minimal/blob/add_matmul_optimize/flash_optimize_matmul.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/types.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])

// Load matrix to REGISTER
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))

// half mma 16x8x16 (only support "ARCH >= SM_80")
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)                                                    \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                 : "=r"(RD0), "=r"(RD1)                                                                                \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

template<const int Bc, const int Br, const int d>
__global__  void flash_attn_2_fwd_f16_mma_m16n8k16_kernel(
  half* Q, half* K, half* V, const int N,
  const int Tc, const int Tr, const float scale,
  half* O) {
  // batch and head index
  int bx = blockIdx.x; int by = blockIdx.y;

  // warp and lane Id
  int warpId = threadIdx.x / 32;
  int laneId = threadIdx.x % 32;

  // Offset into Q, K, V, O - different for each batch and head
  int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh

  // Define SRAM for Q,K,V,O
  extern __shared__ half sram[];
  int tile_size = Br * d;  // size of Qi, Kj, Vj (Br == Bc)
  half* Qi = sram;
  half* Kj = sram + tile_size;
  half* Vj = sram + tile_size; // share with K

  // temporary register
  half reg[32];

  for (int i = 0; i < Tr; i++) {
    // Read Q from global memory to shared memory
    for (int x = threadIdx.x * 8; x < tile_size; x += 1024) {
      int dim_x = x % d;
      int dim_y = x / d;

      // fixed colum length(16) conversion for LDMATRIX
      int new_dim_x = dim_x % 16;
      int new_dim_y = (dim_y / 16 * (d / 16) * 16) + (dim_x / 16 * 16) + (dim_y % 16);

      FLOAT4(Qi[new_dim_y * 16 + new_dim_x]) = FLOAT4(Q[qkv_offset + (i * tile_size) + x]);
    }
    __syncthreads();

    // m_old, l_old
    float thread_max_old[2] = { -INFINITY, -INFINITY }; 
    float thread_sum_old[2] = { 0, 0 };

    // REGISTER for O
    float RO[d / 16][2][2][2] = { 0, };

    for (int j = 0; j < Tc; j++) {
      // m, l
      float thread_max[2] = { -INFINITY, -INFINITY }; 
      float thread_sum[2] = { 0, 0 };

      // REGISTER for mma
      uint32_t RC[Bc / 8][2] = { 0, };
      uint32_t RA[4];
      uint32_t RB[4];
      uint32_t RD[4];

      // Read K from global memory to shared memory
      for (int x = threadIdx.x * 8; x < tile_size; x += 1024) {
        int dim_x = x % d;
        int dim_y = x / d;

        int new_dim_x = dim_x % 16;
        int new_dim_y = (dim_y / 16 * (d / 16) * 16) + (dim_x / 16 * 16) + (dim_y % 16);

         FLOAT4(Kj[new_dim_y * 16 + new_dim_x]) = FLOAT4(K[qkv_offset + (j * tile_size) + x]);
      }
      __syncthreads();

      // Q @ K^T
      for (int k = 0; k < d / 16; k++) {
        // Bc x d to Bc / 4 x d (4 is warp size)
        uint32_t Qi_lane_addr = __cvta_generic_to_shared(&Qi[(warpId * 16 * d) + (laneId % 16) * 16 + (laneId / 16) * 8 + (k * 16 * 16)]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], Qi_lane_addr);

        for (int len = 0; len < Bc; len += 16) {
          uint32_t Kj_lane_addr = __cvta_generic_to_shared(&Kj[(len * d) + (laneId % 16) * 16 + (laneId / 16) * 8 + (k * 16 * 16)]);
          // be careful "not 0 1 2 3"
          LDMATRIX_X4(RB[0], RB[2], RB[1], RB[3], Kj_lane_addr);

          // 16x16x16 wmma *(16x8x16 mma 0)
          HMMA16816(RC[(len / 16) * 2 + 0][0], RC[(len / 16) * 2 + 0][1],
                    RA[0], RA[1], RA[2], RA[3],
                    RB[0], RB[1],
                    RC[(len / 16) * 2 + 0][0], RC[(len / 16) * 2 + 0][1]);

          // 16x16x16 wmma *(16x8x16 mma 1)
          HMMA16816(RC[(len / 16) * 2 + 1][0], RC[(len / 16) * 2 + 1][1],
                    RA[0], RA[1], RA[2], RA[3],
                    RB[2], RB[3],
                    RC[(len / 16) * 2 + 1][0], RC[(len / 16) * 2 + 1][1]);
        }
      }
      __syncthreads();

      // Read V from global memory to shared memory
      for (int x = threadIdx.x * 8; x < tile_size; x += 1024) {
        FLOAT4(reg[0]) = FLOAT4(V[qkv_offset + (j * tile_size) + x]);

        int dim_x = x % d;
        int dim_y = x / d;

        #pragma unroll
        for (int iter = 0; iter < 8; iter++) {
          int new_dim_y = ((dim_x + iter) / 16 * (Bc / 16) * 16) + (dim_y / 16 * 16) + ((dim_x + iter) % 16);
          int new_dim_x = dim_y % 16;

          Vj[new_dim_y * 16 + new_dim_x] = reg[iter];
        }
      }
      __syncthreads();

      // adapt from https://github.com/jundaf2/INT8-Flash-Attention-FMHA-Quantization/blob/main/inc/fmha_i8.cuh
      // Softmax phase (m, l calculate)
      // FETCHING REGISTER
      FLOAT4(reg[0])  = FLOAT4(RC[0][0]);
      FLOAT4(reg[8])  = FLOAT4(RC[2][0]);
      FLOAT4(reg[16]) = FLOAT4(RC[4][0]);
      FLOAT4(reg[24]) = FLOAT4(RC[6][0]);

      // thread level reduce max
      #pragma unroll
      for (int xi = 0; xi < Bc / 16; xi++) {
        #pragma unroll
        for (int tc_yi = 0; tc_yi < 2; tc_yi++) {
          #pragma unroll
          for (int tc_xi = 0; tc_xi < 2; tc_xi++) {
            float tmp_val1 = __half2float(reg[xi * 8 + tc_xi * 4 + tc_yi * 2 + 0]);
            float tmp_val2 = __half2float(reg[xi * 8 + tc_xi * 4 + tc_yi * 2 + 1]);
            float tmp_max_val = max(tmp_val1, tmp_val2) * scale;
            thread_max[tc_yi] = max(thread_max[tc_yi], tmp_max_val);
          }
        }
      }

      // warp level reduce max
      #pragma unroll
      for (int s = 2; s > 0; s >>= 1) {
        #pragma unroll
        for (int tc_yi = 0; tc_yi < 2; tc_yi++) {
          thread_max[tc_yi] = max(thread_max[tc_yi], __shfl_xor_sync(0xffffffff, thread_max[tc_yi], s, 4));
        }
      }
      
      // thread level reduce sum
      #pragma unroll
      for (int xi = 0; xi < Bc / 16; xi++) {
        #pragma unroll
        for (int tc_yi = 0; tc_yi < 2; tc_yi++) {
          #pragma unroll
          for (int tc_xi = 0; tc_xi < 2; tc_xi++) {
            float tmp_sum_val_0 = __expf(__half2float(reg[xi * 8 + tc_xi * 4 + tc_yi * 2 + 0]) * scale - thread_max[tc_yi]);
            float tmp_sum_val_1 = __expf(__half2float(reg[xi * 8 + tc_xi * 4 + tc_yi * 2 + 1]) * scale - thread_max[tc_yi]);
            reg[xi * 8 + tc_xi * 4 + tc_yi * 2 + 0] = __float2half(tmp_sum_val_0);
            reg[xi * 8 + tc_xi * 4 + tc_yi * 2 + 1] = __float2half(tmp_sum_val_1);
            thread_sum[tc_yi] += (tmp_sum_val_0 + tmp_sum_val_1);
          }
        }
      }

      // warp level reduce sum
      #pragma unroll
      for (int s = 2; s > 0; s >>= 1) {
        #pragma unroll
        for (int tc_yi = 0; tc_yi < 2; tc_yi++) {
          thread_sum[tc_yi] += __shfl_xor_sync(0xffffffff, thread_sum[tc_yi], s, 4);
        }
      }

      // FETCHING REGISTER for P
      FLOAT4(RC[0][0]) = FLOAT4(reg[0]);
      FLOAT4(RC[2][0]) = FLOAT4(reg[8]);
      FLOAT4(RC[4][0]) = FLOAT4(reg[16]);
      FLOAT4(RC[6][0]) = FLOAT4(reg[24]);

      // P @ V
      for (int k = 0; k < d / 16; k++) {
        RD[0] = RD[1] = RD[2] = RD[3] = 0;
        for (int len = 0; len < Bc; len += 16) {
          uint32_t Vj_lane_addr = __cvta_generic_to_shared(&Vj[(k * 16 * Bc) + (len * 16) + (laneId % 16) * 16 + (laneId / 16) * 8]);
          LDMATRIX_X4(RB[0], RB[2], RB[1], RB[3], Vj_lane_addr);

          HMMA16816(RD[0], RD[1],
                    RC[len / 16 * 2 + 0][0], RC[len / 16 * 2 + 0][1], RC[len / 16 * 2 + 1][0], RC[len / 16 * 2 + 1][1],
                    RB[0], RB[1],
                    RD[0], RD[1]);

          HMMA16816(RD[2], RD[3],
                    RC[len / 16 * 2 + 0][0], RC[len / 16 * 2 + 0][1], RC[len / 16 * 2 + 1][0], RC[len / 16 * 2 + 1][1],
                    RB[2], RB[3],
                    RD[2], RD[3]);
        }

        FLOAT4(reg[0]) =  FLOAT4(RD[0]);
        #pragma unroll
        for(int tc_yi = 0; tc_yi < 2; tc_yi++) {
          float thread_max_new = max(thread_max_old[tc_yi], thread_max[tc_yi]);
          float exp_max_old = __expf(thread_max_old[tc_yi] - thread_max_new);
          float exp_max = __expf(thread_max[tc_yi] - thread_max_new);
          float thread_sum_new = exp_max_old * thread_sum_old[tc_yi] + exp_max * thread_sum[tc_yi];
          #pragma unroll
          for(int tc_xi=0; tc_xi < 2; tc_xi++) {
            RO[k][tc_yi][tc_xi][0] =
              __frcp_rn(thread_sum_new) *
              (thread_sum_old[tc_yi] *
               exp_max_old * RO[k][tc_yi][tc_xi][0] +
               exp_max * __half2float(reg[tc_xi * 4 + tc_yi * 2 + 0]));

            RO[k][tc_yi][tc_xi][1] =
              __frcp_rn(thread_sum_new) *
              (thread_sum_old[tc_yi] *
               exp_max_old * RO[k][tc_yi][tc_xi][1] +
               exp_max * __half2float(reg[tc_xi * 4 + tc_yi * 2 + 1]));
          }
        }
      }

      // update m, l
      for(int tc_yi = 0; tc_yi < 2; tc_yi++) {
        float thread_max_new = max(thread_max_old[tc_yi], thread_max[tc_yi]);
        float exp_max_old = __expf(thread_max_old[tc_yi] - thread_max_new);
        float exp_max = __expf(thread_max[tc_yi] - thread_max_new);
        float thread_sum_new = exp_max_old * thread_sum_old[tc_yi] + exp_max * thread_sum[tc_yi];
        thread_sum_old[tc_yi] = thread_sum_new;
        thread_max_old[tc_yi] = thread_max_new;
      }
      __syncthreads();
    }

    // update O
    for (int k = 0; k < d / 16; k++) {
      #pragma unroll
      for(int tc_yi = 0; tc_yi < 2; tc_yi++) {
        #pragma unroll
        for(int tc_xi=0; tc_xi < 2; tc_xi++) {
          int lane_pos = qkv_offset + i * Br * d + (warpId * 16 * d) + (laneId / 4 + tc_yi * 8) * d + tc_xi * 8 + laneId % 4 * 2 + (k * 16);
          O[lane_pos + 0] = __float2half(RO[k][tc_yi][tc_xi][0]);
          O[lane_pos + 1] = __float2half(RO[k][tc_yi][tc_xi][1]);
        }
      }
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

torch::Tensor flash_attn_2_fwd_f16_mma_m16n8k16(
  torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  // TODO: determine Bc, Br dynamically
  CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf)
  const int Bc = 64; 
  const int Br = 64;

  const int B = Q.size(0); 
  const int nh = Q.size(1);
  const int N = Q.size(2); 
  const int d = Q.size(3);
  CHECK_TORCH_TENSOR_SHAPE(K, Q)
  CHECK_TORCH_TENSOR_SHAPE(V, Q)

  const int Tc = ceil((float) N / Bc); 
  const int Tr = ceil((float) N / Br);
  const float scale = 1.0 / sqrt(d);

  // Initialize O, l, m to HBM
  auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kHalf);
  auto O = torch::zeros_like(Q, options);

  // Calculate SRAM size needed per block
  const int sram_size = (2 * Br * d * sizeof(half));
  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);

  dim3 grid(B, nh);  // batch_size x num_heads
  dim3 block(128);   // 4 Warps per block

  // cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (d == 64) {
    flash_attn_2_fwd_f16_mma_m16n8k16_kernel<Bc, Br, 64><<<
    grid, block, sram_size>>>(
      reinterpret_cast<half*>(Q.data_ptr()),
      reinterpret_cast<half*>(K.data_ptr()),
      reinterpret_cast<half*>(V.data_ptr()),
      N, Tc, Tr, scale,
      reinterpret_cast<half*>(O.data_ptr())
    );
  }
  if (d == 128) {
    flash_attn_2_fwd_f16_mma_m16n8k16_kernel<Bc, Br, 128><<<
    grid, block, sram_size>>>(
      reinterpret_cast<half*>(Q.data_ptr()),
      reinterpret_cast<half*>(K.data_ptr()),
      reinterpret_cast<half*>(V.data_ptr()),
      N, Tc, Tr, scale,
      reinterpret_cast<half*>(O.data_ptr())
    );
  }
  return O;
}

void flash_attn_2_fwd_f16_mma_m16n8k16_v2(
  torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O) {
  // TODO: determine Bc, Br dynamically
  CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf)
  const int Bc = 64; 
  const int Br = 64;

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

  // Calculate SRAM size needed per block
  const int sram_size = (2 * Br * d * sizeof(half));
  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);

  dim3 grid(B, nh);  // batch_size x num_heads
  dim3 block(128);   // 4 Warps per block

  // cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (d == 64) {
    flash_attn_2_fwd_f16_mma_m16n8k16_kernel<Bc, Br, 64><<<
    grid, block, sram_size>>>(
      reinterpret_cast<half*>(Q.data_ptr()),
      reinterpret_cast<half*>(K.data_ptr()),
      reinterpret_cast<half*>(V.data_ptr()),
      N, Tc, Tr, scale,
      reinterpret_cast<half*>(O.data_ptr())
    );
  }
  if (d == 128) {
    flash_attn_2_fwd_f16_mma_m16n8k16_kernel<Bc, Br, 128><<<
    grid, block, sram_size>>>(
      reinterpret_cast<half*>(Q.data_ptr()),
      reinterpret_cast<half*>(K.data_ptr()),
      reinterpret_cast<half*>(V.data_ptr()),
      N, Tc, Tr, scale,
      reinterpret_cast<half*>(O.data_ptr())
    );
  }
}
