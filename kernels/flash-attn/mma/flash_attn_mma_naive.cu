// modified from: https://github.com/Byeong-Chan/flash-attention-minimal/blob/add_matmul_optimize/flash_optimize_matmul.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/types.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <mma.h>
#include "utils.h"


template<const int Bc, const int Br, const int d>
__global__  void flash_attn_mma_naive_kernel(
  half* Q, half* K, half* V, const int N,
  const int Tc, const int Tr, const float scale,
  half* O) {
  // batch and head index
  int bx = blockIdx.x; int by = blockIdx.y;

  // warp and lane Id
  int warpId = threadIdx.x / 32;
  int laneId = threadIdx.x % 32;
  int tid = threadIdx.x;

  // Offset into Q, K, V, O - different for each batch and head
  int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh

  // Define SRAM for Q,K,V,O
  extern __shared__ half sram[];
  int tile_size = Br * d;  // size of Qi, Kj, Vj (Br == Bc)
  half* Qi = sram;
  half* Kj = sram + tile_size;
  half* Vj = sram + tile_size; // share with K

  // temporary register(memory), .local space in ptx, addressable
  half reg[32];

  for (int i = 0; i < Tr; i++) {
    // Read Q from global memory to shared memory [Br,d]
    for (int x = threadIdx.x * 8; x < tile_size; x += 1024) {
      int dim_x = x % d;
      int dim_y = x / d;

      // fixed colum length(16) conversion for LDMATRIX
      int new_dim_x = dim_x % 16;
      int new_dim_y = (dim_y / 16 * (d / 16) * 16) + (dim_x / 16 * 16) + (dim_y % 16);

      LDST128BITS(Qi[new_dim_y * 16 + new_dim_x]) = LDST128BITS(Q[qkv_offset + (i * tile_size) + x]);
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

      // Read K from global memory to shared memory, 8*128=1024
      for (int x = threadIdx.x * 8; x < tile_size; x += 1024) {
        int dim_x = x % d; // d=64, 0~63, col
        int dim_y = x / d; // x=(0~127=64x2)*8,d=64 or 128
        // shared memory: Br*d=64x64, reshape [256,16], 变换后的row按照16递增
        // 变换后的col，则为0和8，表示两个MMA需要的8x8矩阵，按照K=16, N=16=2x8来布局。
        // 对于一个M16N16K16，当col>15后，属于新的MMA，因此按照K=16在行数递增
        // 满足ldmatrix.x4的加载要求的布局，加载4个8x8，也就是一个16x16的矩阵。
        // [Naive] Load K, g->s, tid: 0, x:0, (row,col):(0,0)->(0,0)
        // [Naive] Load K, g->s, tid: 1, x:8, (row,col):(0,8)->(0,8)
        // [Naive] Load K, g->s, tid: 8, x:64, (row,col):(1,0)->(1,0)
        // [Naive] Load K, g->s, tid: 9, x:72, (row,col):(1,8)->(1,8)
        // [Naive] Load K, g->s, tid: 10, x:80, (row,col):(1,16)->(17,0)
        // [Naive] Load K, g->s, tid: 0, x:1024, (row,col):(16,0)->(64,0)
        // x是8的倍数，因此这里结果为 0,8
        int new_dim_x = dim_x % 16; 
        int new_dim_y = ((dim_y / 16) * (d / 16) * 16) + (dim_x / 16 * 16) + (dim_y % 16);
        LDST128BITS(Kj[new_dim_y * 16 + new_dim_x]) = LDST128BITS(
          K[qkv_offset + (j * tile_size) + x]);
      }
      __syncthreads();

      // Q[Br,d] @ K^T[Br,d], tile_d=16, matmul K=16
      // 先 loop over K(d)
      for (int k = 0; k < d / 16; k++) {
        // Bc x d to Bc / 4 x d (4 is warp size)
        uint32_t Qi_lane_addr = __cvta_generic_to_shared(
          &Qi[(warpId * 16 * d) + (laneId % 16) * 16 + (laneId / 16) * 8 + (k * 16 * 16)]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], Qi_lane_addr);
        
        // tile_Bc=16=2x8, matmul N=8, 256 x 16
        // 先 loop over N(Bc)
        for (int len = 0; len < Bc; len += 16) {
          // (len * d): 每个迭代处理[16,d]的块
          // (laneId % 16) * 16: 16表示col=16，smem是按照col=16布局的，(laneId % 16),
          // 0~15则表示K16N16块中的行数. 
          // (laneId / 16) * 8: (0~1)*8, 0/8, 表示K16N16块中的列数
          // (k * 16 * 16): 表示K16N16的块大小，每次加载16x16大小的块
          //  T0|(0, 0),...,(0,7) | T16|(0, 8),...,(0,15) |
          //  T1|(1, 0),...,(1,7) | T17|(1, 8),...,(1,15) |
          //  T2|(2, 0),...,(2,7) | T18|(2, 8),...,(2,15) |
          //    |(., 0),...,(.,7) |    |(., 8),...,(.,15) |
          //    |(7, 0),...,(7,7) |    |(7, 8),...,(7,15) |
          //    |(8, 0),...,(8,7) |    |(8, 8),...,(8,15) |
          //    |(9, 0),...,(9,7) |    |(9, 8),...,(9,15) |
          //    |(., 0),...,(.,7) |    |(., 8),...,(.,15) |
          // T15|(15,0),...,(15,7)| T31|(15,8),...,(15,15)|
          uint32_t Kj_lane_addr = __cvta_generic_to_shared(
            &Kj[(len * d) + (laneId % 16) * 16 + (laneId / 16) * 8 + (k * 16 * 16)]);
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
      } // end for loop over d.
      
      __syncthreads();
    
      // Read V from global memory to shared memory
      for (int x = threadIdx.x * 8; x < tile_size; x += 1024) {
        LDST128BITS(reg[0]) = LDST128BITS(V[qkv_offset + (j * tile_size) + x]);

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
      LDST128BITS(reg[0])  = LDST128BITS(RC[0][0]);
      LDST128BITS(reg[8])  = LDST128BITS(RC[2][0]);
      LDST128BITS(reg[16]) = LDST128BITS(RC[4][0]);
      LDST128BITS(reg[24]) = LDST128BITS(RC[6][0]);
      
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
      LDST128BITS(RC[0][0]) = LDST128BITS(reg[0]);
      LDST128BITS(RC[2][0]) = LDST128BITS(reg[8]);
      LDST128BITS(RC[4][0]) = LDST128BITS(reg[16]);
      LDST128BITS(RC[6][0]) = LDST128BITS(reg[24]);

      // P[Br=M,Bc=K] @ V[Bc=K,d=N]
      // 先 loop over N(d), RD[4]
      for (int k = 0; k < d / 16; k++) { // 64/16=4
        RD[0] = RD[1] = RD[2] = RD[3] = 0;
        // 再 loop over K(Bc)
        for (int len = 0; len < Bc; len += 16) {
          uint32_t Vj_lane_addr = __cvta_generic_to_shared(&Vj[(k * 16 * Bc) + (len * 16) + (laneId % 16) * 16 + (laneId / 16) * 8]);
          LDMATRIX_X4(RB[0], RB[2], RB[1], RB[3], Vj_lane_addr);

          // RC[8][2] {0,1|2,3|4,5|6,7}[0|1] 
          // len = 0,  RC[0][0] [0][1] [1][0] [1][1]
          // len = 16, RC[2][0] [2][1] [3][0] [3][1]
          // len = 32, RC[4][0] [4][1] [5][0] [5][1]
          // len = 48, RC[6][0] [6][1] [7][0] [7][1]
          // RD[4]在K维度累加了4次
          HMMA16816(RD[0], RD[1],
                    RC[len / 16 * 2 + 0][0], RC[len / 16 * 2 + 0][1], RC[len / 16 * 2 + 1][0], RC[len / 16 * 2 + 1][1],
                    RB[0], RB[1],
                    RD[0], RD[1]);
          // RC[0][0] [0][1] [1][0] [1][1]
          HMMA16816(RD[2], RD[3],
                    RC[len / 16 * 2 + 0][0], RC[len / 16 * 2 + 0][1], RC[len / 16 * 2 + 1][0], RC[len / 16 * 2 + 1][1],
                    RB[2], RB[3],
                    RD[2], RD[3]);
        } // end for Bc
       
        LDST128BITS(reg[0]) =  LDST128BITS(RD[0]);
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
      } // end for d

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

void flash_attn_mma_naive(
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

  if (d == 64) {
    flash_attn_mma_naive_kernel<Bc, Br, 64><<<
    grid, block, sram_size>>>(
      reinterpret_cast<half*>(Q.data_ptr()),
      reinterpret_cast<half*>(K.data_ptr()),
      reinterpret_cast<half*>(V.data_ptr()),
      N, Tc, Tr, scale,
      reinterpret_cast<half*>(O.data_ptr())
    );
  }
  if (d == 128) {
    flash_attn_mma_naive_kernel<Bc, Br, 128><<<
    grid, block, sram_size>>>(
      reinterpret_cast<half*>(Q.data_ptr()),
      reinterpret_cast<half*>(K.data_ptr()),
      reinterpret_cast<half*>(V.data_ptr()),
      N, Tc, Tr, scale,
      reinterpret_cast<half*>(O.data_ptr())
    );
  }
}
