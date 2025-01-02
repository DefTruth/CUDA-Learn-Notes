#include "utils.h"

// Write FlashAttention-2 from scratch using Tensor Cores with MMA PTX instruction.
// The input is Q,K,V, 4D tensor with shape [batch_size, num_heads, seq_len, head_dim].
// The output is O, a 4D tensor with shape [batch_size, num_heads, seq_len, head_dim].

// The FlashAttention-2 algorithm is described in the following paper:
// https://arxiv.org/pdf/2307.08691

// Q,K,V,O: [batch_size, num_heads, seq_len, head_dim], [B,H,N,d]
// each block processes Q_tile with shape [Br,d] and full K,V with shape [N,d]

// Split Q across MMA(Warps) and keep access KV for all MMA(Warps),
// in order to reduce the comm between warps via smem and warp shuffle.

// MMA = m16n8k16, Br=16x4=64, Bc=8x8=64, layout: 4 warps
// |   64x64   |      warp_KV 0       |
// | warp_QP 0 | MMA 0 ... MMA 0 (x8) |
// | warp_QP 1 | MMA 1 ... MMA 1 (x8) |
// | warp_QP 2 | MMA 2 ... MMA 2 (x8) |
// | warp_QP 3 | MMA 3 ... MMA 3 (x8) |

// MMA = m16n8k16, Br=16x8=128, Bc=8x16=128, layout: 8 warps
// |  128x128  |      warp_KV 0        |
// | warp_QP 0 | MMA 0 ... MMA 0 (x16) |
// | warp_QP 1 | MMA 1 ... MMA 1 (x16) |
// | warp_QP 2 | MMA 2 ... MMA 2 (x16) |
// | warp_QP 3 | MMA 3 ... MMA 3 (x16) |
// | warp_QP 4 | MMA 4 ... MMA 4 (x16) |
// | warp_QP 5 | MMA 5 ... MMA 5 (x16) |
// | warp_QP 6 | MMA 6 ... MMA 6 (x16) |
// | warp_QP 7 | MMA 7 ... MMA 7 (x16) |

// MMA = m16n8k16, Br=16x8=128, Bc=8x8=64, layout: 8 warps
// |  128x64  |      warp_KV 0        |
// | warp_QP 0 | MMA 0 ... MMA 0 (x8) |
// | warp_QP 1 | MMA 1 ... MMA 1 (x8) |
// | warp_QP 2 | MMA 2 ... MMA 2 (x8) |
// | warp_QP 3 | MMA 3 ... MMA 3 (x8) |
// | warp_QP 4 | MMA 4 ... MMA 4 (x8) |
// | warp_QP 5 | MMA 5 ... MMA 5 (x8) |
// | warp_QP 6 | MMA 6 ... MMA 6 (x8) |
// | warp_QP 7 | MMA 7 ... MMA 7 (x8) |

// Fine-grained tiling at the MMA level for Q and K results in a constant SRAM usage of
// 64 * kMmaAtomK for Q and K. For V, the SRAM complexity is O(kMmaAtomK * d), leading to
// an overall SRAM complexity of O(kMmaAtomK * d). Consequently, this approach allows us to
// extend D (head dimension) up to 1024. Performance optimizations are ongoing. 
// Stay tuned for updates ~

template<
         const int kHeadDim,              // Headdim, 32,64,128     
         const int kMmaAtomM,             // MMA Atom M, 16
         const int kMmaAtomN,             // MMA Atom N, 8
         const int kMmaAtomK,             // MMA Atom K, 16
         const int kMmaTileSeqLenQ,       // 4, more MMA(warp), M=16*4=64, Q@K^T=[Br(M), d(K)]@[d(K),  Bc(N)]  
         const int kMmaTileSeqLenK,       // 1, more MMA(warp), N=8*1 =8,  Q@K^T=[Br(M), d(K)]@[d(K),  Bc(N)]    
         const int kMmaTileSeqLenP,       // 4, more MMA(warp), M=16*4=64, P@V  =[Br(M),Bc(K)]@[Bc(K), d(N) ]
         const int kMmaTileHeadDimV,      // 1, more MMA(warp), N=8*1 =8,  P@V  =[Br(M),Bc(K)]@[Bc(K), d(N) ]       
         const int kWarpTileSeqLenQ,      // 1, more values, M, Br=64*1=64, matmul M 
         const int kWarpTileSeqLenK,      // 8, more values, N, Bc=8*8 =64, matmul N
         const int kWarpTileSeqLenP,      // 1, more values, M, Br=64*1=64, matmul M
         const int kWarpTileHeadDimV,     // 8, more values, N, d=8*(1|2|3|4|...)=8|...|32|64|96|128|...
         const int kOStorageAccFloat32,   // 0/1, MMA Acc always be fp32, but O storage can be fp32 or half.
         const int kStage,                // 1,2
         const int kPadQ,                 // Pad Q/K/V 0,8
         const int kPadK,             
         const int kPadV             
         >
__global__ void __launch_bounds__(
  WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK) 
flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_kernel(half* Q, 
                                                        half* K, 
                                                        half* V, 
                                                        half* O, 
                                                        int QKV_seqlen,
                                                        int QKV_head) {
  // Matmul Layout: Q[Br,d]@K^T[d,Bc] NT, P[Br,Bc]@V[Bc,d] NN.
  // NOTE: K[Bc,d] with row major means K^T[d,Bc] in col major.
  static_assert(kMmaAtomM == 16 && kMmaAtomN == 8 && kMmaAtomK == 16); // m16n8k16
  static_assert(kMmaTileSeqLenQ  <= 8 && kMmaTileSeqLenK  == 1);  // Q@K^T
  static_assert(kMmaTileSeqLenP  <= 8 && kMmaTileHeadDimV == 1);  // P@V
  static_assert(kWarpTileSeqLenQ == 1 && kWarpTileSeqLenK <= 16); // Q@K^T
  // kWarpTileHeadDimV: d=8*(1|2|3|4|...) = 8|...|32|64|96|128|..., etc.
  // e.g, kWarpTileHeadDimV = 8 -> d = 8*8 = 64; 16 -> d = 8*16 = 128.
  static_assert(kWarpTileSeqLenP == 1 && kWarpTileHeadDimV == (
    kHeadDim / (kMmaAtomN * kMmaTileHeadDimV))); // P@V
  static_assert(kOStorageAccFloat32 == 0 || kOStorageAccFloat32 == 1);
  static_assert(kStage < 3 && kStage > 0); 
  static_assert(kPadQ >= 0 && kPadQ % 8 == 0); // 0,8,16
  static_assert(kPadK >= 0 && kPadK % 8 == 0); // 0,8,16
  static_assert(kPadV >= 0 && kPadV % 8 == 0); // 0,8,16
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ; // 16*4*1=64
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK; //  8*1*8=64
  static_assert(Br >= Bc); // for shared memory reuse.
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK; // 32*4*1=128, num threads
  // Now, N must be mutliples of Bc(32/64) for KV tiling across seqlen.
  const int Tc = div_ceil(QKV_seqlen, Bc); // Tc K_tile[Bc,d]
  const float scale = 1.0f / sqrt((float) kHeadDim);
  
  // grid(div_ceil(QKV_seqlen, Br), QKV_batch * QKV_head), (x,y,z)
  const int QKV_batch_id = blockIdx.y / QKV_head; // Batch size
  const int QKV_head_id  = blockIdx.y % QKV_head; // Head num
  const int Q_tile_id    = blockIdx.x;            // Q tile_id, range [0, Tr]
  const int O_tile_id    = Q_tile_id;             // O tile_id, same as Q.
  const int tid          = threadIdx.x;           // within block
  const int warp_id      = tid / WARP_SIZE;       // 0~7 warp_id within block
  const int lane_id      = tid % WARP_SIZE;       // 0~31
  const int warp_QP      = warp_id;               // 0,1,2,3 or 0~7
  const int warp_KV      = 0;                     // 0
  // MMA Layout [Br,Bc]=[64,64], MMA = m16n8k16, Br=16x4=64, Bc=8x8=64, layout: 4 warps
  // |   64x64   |      warp_KV 0       |
  // | warp_QP 0 | MMA 0 ... MMA 0 (x8) |
  // | warp_QP 1 | MMA 1 ... MMA 1 (x8) |
  // | warp_QP 2 | MMA 2 ... MMA 2 (x8) |
  // | warp_QP 3 | MMA 3 ... MMA 3 (x8) |
  // MMA Layout [Br,Bc]=[128,128], MMA = m16n8k16, Br=16x8=128, Bc=8x16=128, layout: 8 warps
  // |  128x128  |      warp_KV 0        |
  // | warp_QP 0 | MMA 0 ... MMA 0 (x16) |
  // | warp_QP 1 | MMA 1 ... MMA 1 (x16) |
  // | warp_QP 2 | MMA 2 ... MMA 2 (x16) |
  // | warp_QP 3 | MMA 3 ... MMA 3 (x16) |
  // | warp_QP 4 | MMA 4 ... MMA 4 (x16) |
  // | warp_QP 5 | MMA 5 ... MMA 5 (x16) |
  // | warp_QP 6 | MMA 6 ... MMA 6 (x16) |
  // | warp_QP 7 | MMA 7 ... MMA 7 (x16) |
  const int Q_gmem_offset = ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) + 
                             (QKV_head_id * QKV_seqlen * kHeadDim)); // Q [seqlen,d]
  const int K_gmem_offset = ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) + 
                             (QKV_head_id * QKV_seqlen * kHeadDim)); // K [seqlen,d]                           
  const int V_gmem_offset = Q_gmem_offset; // V [seqlen,d]
  const int O_gmem_offset = Q_gmem_offset; // O [seqlen,d]

  // Mapping Q gmem -> tid -> smem, Q[Br,kMmaAtomK]=[64/128,16], 128/256 threads.
  int load_smem_Q_Br = (tid / (kNumThreads / Br)); // Br 64, tid / 2, row 0~64
  int load_smem_Q_d  = (tid % (kNumThreads / Br)) * (kMmaAtomK / (kNumThreads / Br)); // (tid % 2) * 8, 0,8,...
  // Mapping K gmem -> tid -> smem, K[Bc,kMmaAtomK]=[64/128,16], 128 threads.
  int load_smem_K_Bc = (tid / (kNumThreads / Bc)); // Bc 64, tid / 2, row 0~64
  int load_smem_K_d  = (tid % (kNumThreads / Bc)) * (kMmaAtomK / (kNumThreads / Bc)); // (tid % 2) * 8, 0,8,...
  // Mapping V gmem -> tid -> smem, V[Bc,d_tile=16]=[64,16], 128 threads.
  int load_smem_V_Bc = (tid / (kNumThreads / Bc)); // kMmaAtomN*2 16, tid / 8, row 0~15
  int load_smem_V_d  = (tid % (kNumThreads / Bc)) * ((kMmaAtomN * 2) / (kNumThreads / Bc)); // (tid % 2) * 8, 0,8
  // global Q row of current head for tile [Br,d] per block.
  int load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br; 
  if (load_gmem_Q_Br >= QKV_seqlen) return;

  // Shared memory for Q,K,V, we don not need additional smem for O 
  // collective store which perform via registers reuse and warp shuffle.
  extern __shared__ half smem[];
  // Split Q + Shared KV SMEM + Fine grain tiling, only need O(1) SRAM complexity.
  constexpr int Q_tile_size = Br * (kMmaAtomK     + kPadQ); // Q[Br,16], 64*16*2=2048 bytes, 2M
  constexpr int K_tile_size = Bc * (kMmaAtomK     + kPadK); // K[Bc,16], 2M
  constexpr int V_tile_size = Bc * (kMmaAtomN * 2 + kPadV); // V[Bc,16], 2M
  half* Q_tile_smem = smem; // 8M/16M
  half* K_tile_smem = Q_tile_smem + kStage * Q_tile_size; // 8M/16M
  half* V_tile_smem = Q_tile_smem; // V may reuse all Q+K smem after Q@K^T.
  uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
  uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
  uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

  // --------------------- Registers/SMEM for thread block -------------------------
  // block m_old, l_old, store in lane, use float to keep precision.
  float lane_block_row_max_old[kWarpTileSeqLenQ][2]; // [1][2]
  float lane_block_row_sum_old[kWarpTileSeqLenQ][2]; // [1][2]
  fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_max_old, -INFINITY);
  fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_sum_old, 0.0f);

  // ---------------------- Registers for S=Q@K^T/O=P@V ----------------------------
  // registers for QKV, S=Q[Br,d]@K[Bc,d]=[Br,Bc] and O=P[Br,Bc]@V[Bc,d]=[Br,d].
  uint32_t R_Q[kWarpTileSeqLenQ][ 4]; // [1][4]
  uint32_t R_K[kWarpTileSeqLenK][ 2]; // [8][2]
  uint32_t R_V[2]; // [2], S=Q@K, only use 2 32bits registers.
  // registers for current tile_K_seqlen within, [64,64] = S_tile[Br,Bc]
  // = Q_tile[Br,d] * K[Bc,d], each thread hold 2x32 bits regs.
  uint32_t R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][ 4]; // [1][8][4], acc f32.
  uint32_t R_O[4]; // registers for O=PV[Br,d]=P@V, [4], only use 4 32bits registers.
  // registers final Output [D]=final rescale(R_O), [2][2/4][2], 8 or 16 regs.
  // 0/1, MMA Acc always be fp32, but O storage(R_D) can be fp32 or half.
  // FP16 can provide precision to approximately 3-4 decimal places. Thus, if the 
  // error does not exceed 1e-3, using FP16 storage is sufficient for most applications.
  uint32_t R_D[kWarpTileSeqLenP][kWarpTileHeadDimV][(kOStorageAccFloat32) ? 4 : 2]; 
  fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV, ((kOStorageAccFloat32) ? 4 : 2)>(R_D, 0);
  
  // <loop over K seqlen>: for K^T[d,seqlen] with K^T_tile[d,Bc]
  // tile_K_seqlen: compute S_tile[Br,Bc] = Q@K^T = Q_tile[Br,d] * K^T[d,Bc]
  #pragma unroll 1
  for (int tile_K_seqlen = 0; tile_K_seqlen < Tc; ++tile_K_seqlen) { 
    // TODO: process last tile_K_seqlen ? pad to multiple of 8.
    
    // Q/K g2s
    if constexpr (kStage > 1) {
      #pragma unroll
      for (int stage = 0; stage < (kStage - 1); ++stage) {
        // Q g2s
        int load_gmem_Q_d = (stage * kMmaAtomK) + load_smem_Q_d; // 0,8
        int load_gmem_Q_addr = (
          Q_gmem_offset + load_gmem_Q_Br * kHeadDim + load_gmem_Q_d);
        uint32_t load_smem_Q_ptr = (
          smem_Q_base_ptr + (stage * Q_tile_size + 
                             load_smem_Q_Br * (kMmaAtomK + kPadQ) + 
                             load_smem_Q_d) * sizeof(half));
        #pragma unroll
        for (int i = 0; i < (kMmaAtomK / (kNumThreads / Br)); i += 8) {
          CP_ASYNC_CG(load_smem_Q_ptr + i * 2, &Q[load_gmem_Q_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
        
        // K g2s
        int load_gmem_K_Bc = (tile_K_seqlen * Bc) + load_smem_K_Bc; // < seqlen
        int load_gmem_K_d  = (stage * kMmaAtomK) + load_smem_K_d; // K [Bc,16] from [seqlen,d]
        int load_gmem_K_addr = (
          K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
        uint32_t load_smem_K_ptr = (
          smem_K_base_ptr + (stage * K_tile_size + 
                             load_smem_K_Bc * (kMmaAtomK + kPadK) + 
                             load_smem_K_d) * sizeof(half)
        );
        #pragma unroll
        for (int i = 0; i < (kMmaAtomK / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
      } // end for stage

      CP_ASYNC_WAIT_GROUP(kStage - 2); // s2->0, s3->1, s4->2
      __syncthreads(); 
    } // end if kStage > 1

    // <loop over K d>: tile_K_d, kMmaAtomK = 16, K_tile_d[kMmaAtomK,Bc]
    // Matmul with NT layout, Q row major, K^T col major. 
    // NOTE: K[Bc,d] with row major means K^T[d,Bc] in col major.
    // S_tile[Br,Bc]=Q_tile[Br,d]@K[Bc,d]
    // <HGEMM in shared memory>
    fill_3D_regs<uint32_t, kWarpTileSeqLenQ, kWarpTileSeqLenK, 4>(R_S, 0);
    #pragma unroll
    for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
      // s2 tn 0->0, 1->1, 2->0; s3 tn 0->0, 1->1, 2->2, 3->0;
      int smem_sel      = (tile_K_d) % kStage;   
      // s2 tn 0->1, 1->0, 2->1; s3 tn 0->2, 1->0, 2->1, 3->2;  
      int smem_sel_next = (tile_K_d + (kStage - 1)) % kStage;

      // stages for Q, K
      if constexpr (kStage > 1) {
        if ((tile_K_d + 1) < (kHeadDim / kMmaAtomK)) {
          // next Q tile g2s
          int load_gmem_Q_d = ((tile_K_d + 1) * kMmaAtomK) + load_smem_Q_d;
          int load_gmem_Q_addr = (
            Q_gmem_offset + load_gmem_Q_Br * kHeadDim + load_gmem_Q_d);
          uint32_t load_smem_Q_ptr = (
            smem_Q_base_ptr + (smem_sel_next * Q_tile_size + 
                               load_smem_Q_Br * (kMmaAtomK + kPadQ) + 
                               load_smem_Q_d) * sizeof(half));
          #pragma unroll
          for (int i = 0; i < (kMmaAtomK / (kNumThreads / Br)); i += 8) {
            CP_ASYNC_CG(load_smem_Q_ptr + i * 2, &Q[load_gmem_Q_addr + i], 16);
          }
          CP_ASYNC_COMMIT_GROUP();

          // next K tile g2s
          int load_gmem_K_Bc = tile_K_seqlen * Bc + load_smem_K_Bc; // < seqlen
          int load_gmem_K_d  = ((tile_K_d + 1) * kMmaAtomK) + load_smem_K_d; // K [Bc,16] from [seqlen,d]
          int load_gmem_K_addr = (
            K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
          uint32_t load_smem_K_ptr = (
            smem_K_base_ptr + (smem_sel_next * K_tile_size + 
                               load_smem_K_Bc * (kMmaAtomK + kPadK) + 
                               load_smem_K_d) * sizeof(half)
          );
          #pragma unroll
          for (int i = 0; i < (kMmaAtomK / (kNumThreads / Bc)); i += 8) {
            CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
          }
          CP_ASYNC_COMMIT_GROUP();
        } 
      } else {
        // sync load curr Q, K g2s
        // curr Q tile g2s
        int load_gmem_Q_d = (tile_K_d * kMmaAtomK) + load_smem_Q_d;
        int load_gmem_Q_addr = (
          Q_gmem_offset + load_gmem_Q_Br * kHeadDim + load_gmem_Q_d);
        uint32_t load_smem_Q_ptr = (
          smem_Q_base_ptr + (smem_sel * Q_tile_size + 
                             load_smem_Q_Br * (kMmaAtomK + kPadQ) + 
                             load_smem_Q_d) * sizeof(half));
        #pragma unroll
        for (int i = 0; i < (kMmaAtomK / (kNumThreads / Br)); i += 8) {
          CP_ASYNC_CG(load_smem_Q_ptr + i * 2, &Q[load_gmem_Q_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

        // curr K tile g2s
        int load_gmem_K_Bc = (tile_K_seqlen * Bc) + load_smem_K_Bc; // < seqlen
        int load_gmem_K_d  = (tile_K_d * kMmaAtomK) + load_smem_K_d; // K [Bc,16] from [seqlen,d]
        int load_gmem_K_addr = (
          K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
        uint32_t load_smem_K_ptr = (
          smem_K_base_ptr + (smem_sel * K_tile_size + 
                             load_smem_K_Bc * (kMmaAtomK + kPadK) + 
                             load_smem_K_d) * sizeof(half)
        );
        #pragma unroll
        for (int i = 0; i < (kMmaAtomK / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
        // Wait curr Q, K tile ready.
        CP_ASYNC_WAIT_GROUP(0); 
        __syncthreads(); 
      } // end if kStage > 1

      // Q s2r
      static_assert(kWarpTileSeqLenQ == 1);
      { // kWarpTileSeqLenQ = 1, Q[Br,d]=[M,K]
        int warp_smem_Q_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + 0 * kMmaAtomM;
        int lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16; // 0~15
        int lane_smem_Q_d  = (lane_id / 16) * 8; // 0,8
        uint32_t lane_smem_Q_ptr = (
            smem_Q_base_ptr + (smem_sel * Q_tile_size + 
                               lane_smem_Q_Br * (kMmaAtomK + kPadQ) + 
                               lane_smem_Q_d) * sizeof(half)
        );
        LDMATRIX_X4(R_Q[0][0], R_Q[0][1], R_Q[0][2], R_Q[0][3], 
                    lane_smem_Q_ptr); // now, R_Q[1][4]
      }

      // smem -> reg, load k16n8 from smem K, offset d according tile_K_d.
      // ldmatrix.x2 for K_tile_smem, [Bc,kMmaAtomK] from [Bc,d]=[K,N]
      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        // load k16n8 via ldmatrix.x2 from K_tile_smem[Bc,d]. 
        // K[Bc,d] with row major means K^T[d,Bc] in col major.
        int warp_smem_K_Bc = warp_KV * (kMmaAtomN * kWarpTileSeqLenK) + j * kMmaAtomN;
        int lane_smem_K_Bc = warp_smem_K_Bc + lane_id % 8; // 0~7
        int lane_smem_K_d  = ((lane_id / 8) % 2) * 8; // 0,8
        uint32_t lane_smem_K_ptr = (
            smem_K_base_ptr + (smem_sel * K_tile_size + 
                               lane_smem_K_Bc * (kMmaAtomK + kPadK) + 
                               lane_smem_K_d) * sizeof(half)
        );
        LDMATRIX_X2(R_K[j][0], R_K[j][1], lane_smem_K_ptr); // R_K
      } // end for kWarpTileSeqLenK
      if constexpr (kStage < 2) {
        // Wait Q, K s2r ready if kStage < 2 in order to avoid 
        // the next Q, K tile g2s overwrite.
        __syncthreads();
      }
      
      // MMA compute
      static_assert(kWarpTileSeqLenQ == 1);
      { // kWarpTileSeqLenQ = 1
        #pragma unroll
        for (int j = 0; j < kWarpTileSeqLenK; ++j) { // 8, 16, 32, ...
          // MMA always accumulate with F32 dtype for high precision.
          HMMA16816F32(R_S[0][j][0], R_S[0][j][1], R_S[0][j][2], R_S[0][j][3],
                       R_Q[0][0],    R_Q[0][1],    R_Q[0][2],    R_Q[0][3], 
                       R_K[j][0],    R_K[j][1], 
                       R_S[0][j][0], R_S[0][j][1], R_S[0][j][2], R_S[0][j][3]);
        }
      }

      if constexpr (kStage > 1) {
        // Wait next Q, K tile g2s ready.
        CP_ASYNC_WAIT_GROUP(kStage - 2);
        __syncthreads(); 
      }

    } // end loop over d, S=Q@K^T
    __syncthreads();

    // MMA = m16n8k16, Br=16x4=64, Bc=8x8=64, layout: 4 warps
    // |   64x64   |      warp_KV 0       |
    // | warp_QP 0 | MMA 0 ... MMA 0 (x8) |
    // | warp_QP 1 | MMA 1 ... MMA 1 (x8) |
    // | warp_QP 2 | MMA 2 ... MMA 2 (x8) |
    // | warp_QP 3 | MMA 3 ... MMA 3 (x8) |

    // Online safe softmax, warp/block reduce max/sum, row wise
    float lane_row_max_new[kWarpTileSeqLenQ][2]; // [1][2]
    float lane_row_sum_new[kWarpTileSeqLenQ][2]; // [1][2]
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_max_new, -INFINITY);
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_sum_new, 0.0f);

    static_assert(kWarpTileSeqLenQ == 1);
    // Row max for [Br,Bc] tile, Thread -> Warp -> Block.
    { // kWarpTileSeqLenQ = 1
      // Thread level reduce max across kWarpTileSeqLenK dim, namely Bc.
      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        // reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
        // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
        // The layout of the fragments held by different threads for C. (m16n8k16)
        // Row\Col  0    1    2    3    4    5    6    7
        // 0        T0: {c0, c1}  T1: {c0, c1}  T2: {c0, c1}  T3: {c0, c1}
        // 1        T4: {c0, c1}  T5: {c0, c1}  T6: {c0, c1}  T7: {c0, c1}
        // 2        ...
        // ...
        // 7        T28: {c0, c1}  T29: {c0, c1}  T30: {c0, c1}  T31: {c0, c1}
        // 8        T0: {c2, c3}   T1: {c2, c3}   T2: {c2, c3}   T3: {c2, c3}
        // 9        T4: {c2, c3}   T5: {c2, c3}   T6: {c2, c3}   T7: {c2, c3}
        // 10       ...
        // ...
        // 15       T28: {c2, c3}  T29: {c2, c3}  T30: {c2, c3}  T31: {c2, c3}
        // R_S[][][4] 4 32bit registers with each contains 1 F32 element.
        // (x,y) 0~7->{c0, c1}, (z,w)->8~15 {c2, c3}
        float* t_fptr_S_0_1 = reinterpret_cast<float*>(&(R_S[0][j][0])); 
        // This should be the row max after S = (Q @ K^T) / sqrt(d)
        float tmp_max_0 = max(t_fptr_S_0_1[0], t_fptr_S_0_1[1]) * scale;
        float tmp_max_1 = max(t_fptr_S_0_1[2], t_fptr_S_0_1[3]) * scale;
        lane_row_max_new[0][0] = max(lane_row_max_new[0][0], tmp_max_0);
        lane_row_max_new[0][1] = max(lane_row_max_new[0][1], tmp_max_1);
      } // end for kWarpTileSeqLenK

      // Warp level reduce max, warp_size = 4
      // Each thread contains the maximum of 2 rows of Br, 
      // and only the values of T0, T4, ..., T28 are used.
      lane_row_max_new[0][0] = warp_reduce_max<float, 4>(lane_row_max_new[0][0]);
      lane_row_max_new[0][1] = warp_reduce_max<float, 4>(lane_row_max_new[0][1]);
    } // end for kWarpTileSeqLenQ

    static_assert(kWarpTileSeqLenQ == 1);
    // Exp sum and mul scale_factor for [Br,Bc] tile, Thread -> Warp -> Block.
    { // kWarpTileSeqLenQ = 1
      // Use latest global row max without update.
      // Br 0, row_id, 0~7,  16~23, 32~39, 48~55; 
      float block_row_max_new_0 = lane_row_max_new[0][0]; 
      // Br 1, row_id, 8~15, 24~31, 40~47, 56~63;
      float block_row_max_new_1 = lane_row_max_new[0][1];
    
      float block_row_max_old_0 = lane_block_row_max_old[0][0];
      float block_row_max_old_1 = lane_block_row_max_old[0][1];
      // Apply m_new = max(m_old, m_new) here.
      block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);

      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        // R_S[][][4] 4 32bit registers with each contains 1 F32 element.
        // (x,y) 0~7->{c0, c1}, (z,w)->8~15 {c2, c3}
        float* t_fptr_S_0_1 = reinterpret_cast<float*>(&(R_S[0][j][0])); 
        half*  t_hptr_S_0_1 = reinterpret_cast< half*>(&(R_S[0][j][0])); 
        // P = Exp(S - m_new), fmaf(x, y, z) = x * y + z in registers;
        t_fptr_S_0_1[0] = __expf(__fmaf_rn(t_fptr_S_0_1[0], scale, - block_row_max_new_0));
        t_fptr_S_0_1[1] = __expf(__fmaf_rn(t_fptr_S_0_1[1], scale, - block_row_max_new_0));
        t_fptr_S_0_1[2] = __expf(__fmaf_rn(t_fptr_S_0_1[2], scale, - block_row_max_new_1));
        t_fptr_S_0_1[3] = __expf(__fmaf_rn(t_fptr_S_0_1[3], scale, - block_row_max_new_1));
        lane_row_sum_new[0][0] += (t_fptr_S_0_1[0] + t_fptr_S_0_1[1]);
        lane_row_sum_new[0][1] += (t_fptr_S_0_1[2] + t_fptr_S_0_1[3]);
        // Update R_S for P[Br,Bc] = Exp(S-m), point wise.
        // Also convert F32 -> half for P@V MMA, reuse R_S as P.
        t_hptr_S_0_1[0] = __float2half_rn(t_fptr_S_0_1[0]);
        t_hptr_S_0_1[1] = __float2half_rn(t_fptr_S_0_1[1]);
        t_hptr_S_0_1[2] = __float2half_rn(t_fptr_S_0_1[2]);
        t_hptr_S_0_1[3] = __float2half_rn(t_fptr_S_0_1[3]);
      } // end for kWarpTileSeqLenK

      // Warp level reduce sum, warp_size = 4
      lane_row_sum_new[0][0] = warp_reduce_sum<float, 4>(lane_row_sum_new[0][0]);
      lane_row_sum_new[0][1] = warp_reduce_sum<float, 4>(lane_row_sum_new[0][1]);
    }
    
    // Prefetch V g2s before row max/sum for P@V if kStage > 1
    static_assert(kWarpTileSeqLenP == 1);
    { // kWarpTileSeqLenP = 1
      if constexpr (kStage > 1) {
        #pragma unroll
        for (int stage = 0; stage < (kStage - 1); ++stage) {
          int load_gmem_V_Bc = (tile_K_seqlen * Bc) + load_smem_V_Bc; // < seqlen
          int load_gmem_V_d  = (stage * kMmaAtomN * 2) + load_smem_V_d; // V [Bc,16] from [seqlen,d]
          int load_gmem_V_addr = (
            V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
          uint32_t load_smem_V_ptr = (
            smem_V_base_ptr + (stage * V_tile_size + 
                               load_smem_V_Bc * (kMmaAtomN * 2 + kPadV) + 
                               load_smem_V_d) * sizeof(half)
          );
          #pragma unroll
          for (int i = 0; i < (kMmaAtomN * 2 / (kNumThreads / Bc)); i += 8) {
            CP_ASYNC_CG(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 16);
          }
          CP_ASYNC_COMMIT_GROUP();
        }
      }
    }

    // according to the A matrix layout for MMA m16n8k16 instruction. 
    // reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
    // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    // The layout of the fragments held by different threads for A matrix with .f16.
    // R\C  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
    // 0    T0: {a0, a1}  T1: {a0, a1}  T2: {a0, a1}  T3: {a0, a1}  T0: {a4, a5}  T1: {a4, a5}  T2: {a4, a5}  T3: {a4, a5}
    // 1    T4: {a0, a1}  T5: {a0, a1}  T6: {a0, a1}  T7: {a0, a1}  T4: {a4, a5}  T5: {a4, a5}  T6: {a4, a5}  T7: {a4, a5}
    // 2    (dashed arrow pointing right)
    // ...
    // 7    T28: {a0, a1}  T29: {a0, a1}  T30: {a0, a1}  T31: {a0, a1}  T28: {a4, a5}  T29: {a4, a5}  T30: {a4, a5}  T31: {a4, a5}
    // 8    T0: {a2, a3}   T1: {a2, a3}   T2: {a2, a3}   T3: {a2, a3}   T0: {a6, a7}   T1: {a6, a7}   T2: {a6, a7}   T3: {a6, a7}
    // 9    T4: {a2, a3}   T5: {a2, a3}   T6: {a2, a3}   T7: {a2, a3}   T4: {a6, a7}   T5: {a6, a7}   T6: {a6, a7}   T7: {a6, a7}
    // 10   (dashed arrow pointing right)
    // ...
    // 15   T28: {a2, a3}  T29: {a2, a3}  T30: {a2, a3}  T31: {a2, a3}  T28: {a6, a7}  T29: {a6, a7}  T30: {a6, a7}  T31: {a6, a7}

    static_assert(kWarpTileSeqLenP == 1);
    {
      // <Prefetch max/sum values>
      // m = max(m_old, m_new), l = exp(m_old - m) * l_old + l_new (FA2 paper)
      // Br 0, row_id, 0~7,  16~23, 32~39, 48~55; Br 1, row_id, 8~15, 24~31, 40~47, 56~63
      float block_row_max_new_0 = lane_row_max_new[0][0]; 
      float block_row_max_new_1 = lane_row_max_new[0][1];
      float block_row_sum_new_0 = lane_row_sum_new[0][0];
      float block_row_sum_new_1 = lane_row_sum_new[0][1];
        
      float block_row_max_old_0 = lane_block_row_max_old[0][0];
      float block_row_max_old_1 = lane_block_row_max_old[0][1];
      // NOTE: max(-inf, val) = val.
      block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);   
      // Avoid inf value while using m_old for rescaling O.
      block_row_max_old_0 = (tile_K_seqlen > 0 ? block_row_max_old_0 : 
                                                 block_row_max_new_0);                                       
      block_row_max_old_1 = (tile_K_seqlen > 0 ? block_row_max_old_1 : 
                                                 block_row_max_new_1);  
      // rescale factor for O and l, exp(m_old - m) for curr tile [Br,d].
      float rescale_o_factor_0 = __expf(block_row_max_old_0 - block_row_max_new_0);
      float rescale_o_factor_1 = __expf(block_row_max_old_1 - block_row_max_new_1);
      
      // Wait V g2s stages ready.
      if constexpr (kStage > 1) {
        CP_ASYNC_WAIT_GROUP(kStage - 2); // s2->0, s3->1, s4->2
        __syncthreads(); 
      }

      // <HGEMM in registers>
      #pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) { // 8, 16, 32, ...
        // Compute d tile, P[Br,Bc]@V[Bc,16] = O[Br,16]
        fill_1D_regs<uint32_t, 4>(R_O, 0); // must clear 

        int smem_sel_v = (j / 2) % kStage;   
        int smem_sel_v_next = ((j / 2) + (kStage - 1)) % kStage;
        // V g2s, V tile smem [Bc,kMmaAtomN*2]=[64,16]
        if (j % 2 == 0) { // 0,2,4,6,...// curr K tile g2s
          if constexpr (kStage > 1) {
            if (((j / 2) + 1) < (kWarpTileHeadDimV / 2)) {
              int load_gmem_V_Bc = (tile_K_seqlen * Bc) + load_smem_V_Bc; // < seqlen
              int load_gmem_V_d  = (((j / 2) + 1) * kMmaAtomN * 2) + load_smem_V_d; // V [Bc,16] from [seqlen,d]
              int load_gmem_V_addr = (
                V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
              uint32_t load_smem_V_ptr = (
                smem_V_base_ptr + (smem_sel_v_next * V_tile_size + 
                                   load_smem_V_Bc * (kMmaAtomN * 2 + kPadV) + 
                                   load_smem_V_d) * sizeof(half)
              );
              #pragma unroll
              for (int i = 0; i < (kMmaAtomN * 2 / (kNumThreads / Bc)); i += 8) {
                CP_ASYNC_CG(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 16);
              }
              CP_ASYNC_COMMIT_GROUP();
            } // end if < (kWarpTileHeadDimV / 2)
          } else {
            // no stages for V g2s
            int load_gmem_V_Bc = (tile_K_seqlen * Bc) + load_smem_V_Bc; // < seqlen
            int load_gmem_V_d  = ((j / 2) * kMmaAtomN * 2) + load_smem_V_d; // V [Bc,16] from [seqlen,d]
            int load_gmem_V_addr = (
              V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
            uint32_t load_smem_V_ptr = (
              smem_V_base_ptr + (smem_sel_v * V_tile_size + 
                                 load_smem_V_Bc * (kMmaAtomN * 2 + kPadV) + 
                                 load_smem_V_d) * sizeof(half)
            );
            #pragma unroll
            for (int i = 0; i < (kMmaAtomN * 2 / (kNumThreads / Bc)); i += 8) {
              CP_ASYNC_CG(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 16);
            }
            CP_ASYNC_COMMIT_GROUP();
            // Wait curr V tile g2s ready.
            CP_ASYNC_WAIT_GROUP(0); 
            __syncthreads(); 
          }
        }

        #pragma unroll
        for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); ++tile_V_Bc) {
          // Load k16n8 V from smem [Bc,8*2] -> regs, R_V, ldmatrix.x2.trans.
          int warp_smem_V_d  = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + (j % 2) * kMmaAtomN; // d, matmaul N
          int lane_smem_V_Bc = tile_V_Bc * kMmaAtomK + lane_id % 16; // 0~15; Bc, matmul K
          int lane_smem_V_d  = warp_smem_V_d; // 0
          uint32_t lane_smem_V_ptr = (
            smem_V_base_ptr + (smem_sel_v * V_tile_size + 
                               lane_smem_V_Bc * (kMmaAtomN * 2 + kPadV) + 
                               lane_smem_V_d) * sizeof(half)
          );
          LDMATRIX_X2_T(R_V[0], R_V[1], lane_smem_V_ptr); // R_V
          // Compute P[Br,Bc]@V[Bc,d] = O[Br,d]
          // For R_S[1][8][2], mapping the layout below of P matrix.
          // MMA = m16n8k16, Br=16x4=64, Bc=8x8=64, layout: 4 warps
          // |   64x64   |      warp_KV 0       |
          // | warp_QP 0 | MMA 0 ... MMA 0 (x8) |
          // | warp_QP 1 | MMA 1 ... MMA 1 (x8) |
          // | warp_QP 2 | MMA 2 ... MMA 2 (x8) |
          // | warp_QP 3 | MMA 3 ... MMA 3 (x8) |
          // tile_V_Bc = 0, all curr MMAs(0~4) need slice P[:,  0:16], 0, 1; stored in all MMAs.
          // tile_V_Bc = 1, all curr MMAs(0~4) need slice P[:, 16:32], 2, 3; stored in all MMAs.
          // tile_V_Bc = 2, all curr MMAs(0~4) need slice P[:, 32:48], 4, 5; stored in all MMAs. 
          // tile_V_Bc = 3, all curr MMAs(0~4) need slice P[:, 48:64], 6, 7; stored in all MMAs. 
          int w = tile_V_Bc * 2; // MMA(Warp) selected, 0, 2, 4, 6
          // MMA always accumulate with F32 dtype for high precision.
          HMMA16816F32(R_O[0], R_O[1], R_O[2], R_O[3],
                       R_S[0][w][0], R_S[0][w][1], R_S[0][w + 1][0],  R_S[0][w + 1][1], 
                       R_V[0], R_V[1],
                       R_O[0], R_O[1], R_O[2], R_O[3]); 
        } // end for V Bc.
        if constexpr (kStage < 2) {
          // Wait curr P@V tile ready if kStage < 2 in order to avoid 
          // the next V tile g2s overwrite.
          __syncthreads();
        }

        // Now, we get [Br,8] slice of [Br,d], each warp(MMA) contains m16n8.
        // 0. Rescale O: Online rescaling O each tile_K_seqlen step, need m_new, m_old.
        // m = max(m_old, m_new), O_new[Br,d] = exp(m_old - m) * O_old + P@V
        // use exp(m_old - m_new), not 1/(m_old - m_new).
        // O_new[Br,d] = exp(m_old - m_new) * O_old + P@V
        float* t_fptr_O_0_1 = reinterpret_cast<float*>(&(R_O[0])); 
        if constexpr (kOStorageAccFloat32) {
          // (x,y) 0~7->{c0, c1}, (z,w)->8~15 {c2, c3}
          float* t_fptr_D_0_1 = reinterpret_cast<float*>(&(R_D[0][j][0])); // kWarpTileSeqLenP=1
          t_fptr_D_0_1[0] = __fmaf_rn(rescale_o_factor_0, t_fptr_D_0_1[0], t_fptr_O_0_1[0]);
          t_fptr_D_0_1[1] = __fmaf_rn(rescale_o_factor_0, t_fptr_D_0_1[1], t_fptr_O_0_1[1]);
          t_fptr_D_0_1[2] = __fmaf_rn(rescale_o_factor_1, t_fptr_D_0_1[2], t_fptr_O_0_1[2]);
          t_fptr_D_0_1[3] = __fmaf_rn(rescale_o_factor_1, t_fptr_D_0_1[3], t_fptr_O_0_1[3]);
        } else {
          half* t_hptr_D_0_1 = reinterpret_cast<half*>(&(R_D[0][j][0])); // kWarpTileSeqLenP=1
          t_hptr_D_0_1[0] = __float2half_rn(__fmaf_rn(
            rescale_o_factor_0, __half2float(t_hptr_D_0_1[0]), t_fptr_O_0_1[0]));
          t_hptr_D_0_1[1] = __float2half_rn(__fmaf_rn(
            rescale_o_factor_0, __half2float(t_hptr_D_0_1[1]), t_fptr_O_0_1[1]));
          t_hptr_D_0_1[2] = __float2half_rn(__fmaf_rn(
            rescale_o_factor_1, __half2float(t_hptr_D_0_1[2]), t_fptr_O_0_1[2]));
          t_hptr_D_0_1[3] = __float2half_rn(__fmaf_rn(
            rescale_o_factor_1, __half2float(t_hptr_D_0_1[3]), t_fptr_O_0_1[3]));
        } // end for tile_V_Bc
        // TODO: Write/Load scaled O -> gmem directly for very large d, e.g 4k,8k,...
        // Thus, reduce the registers usage or SRAM size for R_D. Prefetch O 128bits
        // before P@V tile (>= Ampere) and copy async O 128bits r2g (>= Hopper).
        if constexpr (kStage > 1) {
          // Wait next V tile g2s ready.
          CP_ASYNC_WAIT_GROUP(kStage - 2); 
          __syncthreads();
        }
      } // end for kWarpTileHeadDimV. 
      // Now, we can update m, l after O has been scaled.
      // 1. First, update block row sum Exp for each lane which
      // need both m_new and m_old.
      float block_row_sum_old_0 = lane_block_row_sum_old[0][0];
      float block_row_sum_old_1 = lane_block_row_sum_old[0][1];
      // Update l = exp(m_old - m_new) * l_old + row_sum(P).
      lane_block_row_sum_old[0][0] = (__fmaf_rn(
        rescale_o_factor_0, block_row_sum_old_0, block_row_sum_new_0));
      lane_block_row_sum_old[0][1] = (__fmaf_rn(
        rescale_o_factor_1, block_row_sum_old_1, block_row_sum_new_1));
      // 2. Then, update block row max for each lane.
      lane_block_row_max_old[0][0] = block_row_max_new_0;
      lane_block_row_max_old[0][1] = block_row_max_new_1;
    } // end P@V
    __syncthreads(); 

  } // end loop over N
  __syncthreads();

  // Finaly, we still have to rescale O once more.
  // O_output(D) = ( 1/l_final ) * O_final (FA2 paper)
  static_assert(kWarpTileSeqLenP == 1);
  { // kWarpTileSeqLenP = 1
    float rescale_factor_0 = __frcp_rn(lane_block_row_sum_old[0][0]);
    float rescale_factor_1 = __frcp_rn(lane_block_row_sum_old[0][1]);
    #pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) { // 8, 16, 32, ...
      // Scaling in registers & convert F32 -> half for O collective store.
      if constexpr (kOStorageAccFloat32) {
        float* t_fptr_D_0_1 = reinterpret_cast<float*>(&(R_D[0][j][0])); 
        half*  t_hptr_D_0_1 = reinterpret_cast< half*>(&(R_D[0][j][0])); 
        t_hptr_D_0_1[0] = __float2half_rn(rescale_factor_0 * t_fptr_D_0_1[0]);
        t_hptr_D_0_1[1] = __float2half_rn(rescale_factor_0 * t_fptr_D_0_1[1]);
        t_hptr_D_0_1[2] = __float2half_rn(rescale_factor_1 * t_fptr_D_0_1[2]);
        t_hptr_D_0_1[3] = __float2half_rn(rescale_factor_1 * t_fptr_D_0_1[3]);
      } else {
        half* t_hptr_D_0_1 = reinterpret_cast<half*>(&(R_D[0][j][0])); 
        t_hptr_D_0_1[0] = __float2half_rn(rescale_factor_0 * __half2float(t_hptr_D_0_1[0]));
        t_hptr_D_0_1[1] = __float2half_rn(rescale_factor_0 * __half2float(t_hptr_D_0_1[1]));
        t_hptr_D_0_1[2] = __float2half_rn(rescale_factor_1 * __half2float(t_hptr_D_0_1[2]));
        t_hptr_D_0_1[3] = __float2half_rn(rescale_factor_1 * __half2float(t_hptr_D_0_1[3]));
      }
    } // end for kWarpTileHeadDimV
  } // end for kWarpTileSeqLenP = 1

  // Store O(D): Write O[Br,d] from regs -> gmem, collective store 
  // with reg reuse & warp shuffle. 
  static_assert(kWarpTileSeqLenP == 1);
  { // kWarpTileSeqLenP = 1
    #pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) { // 8
      // reuse R_Q[1][4], R_K[8][2] for collective store.
      uint32_t* t_uptr_Z_0 = reinterpret_cast<uint32_t*>(&(R_Q[0][0])); 
      uint32_t* t_uptr_Z_1 = reinterpret_cast<uint32_t*>(&(R_K[0][0])); 
      t_uptr_Z_0[0] = R_D[0][j][0]; 
      t_uptr_Z_1[0] = R_D[0][j][1]; 
      t_uptr_Z_0[1] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 1, 4);
      t_uptr_Z_0[2] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 2, 4);
      t_uptr_Z_0[3] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 3, 4);
      t_uptr_Z_1[1] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 1, 4);
      t_uptr_Z_1[2] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 2, 4);
      t_uptr_Z_1[3] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 3, 4);

      // st.global.v4 128 bits. [Br,d]
      if (lane_id % 4 == 0) {
        // (0/1)*32 + (0/1)*16=(0,16,32,48), + 0~7 -> 0~56
        int store_warp_regs_O_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenP ) + 0 * kMmaAtomM;
        int store_lane_gmem_O_Br = O_tile_id * Br + store_warp_regs_O_Br + lane_id / 4; // 0~7
        // (0~3)*16 + (0/1)*8=(0,8,16,24,...,48,56)
        int store_warp_regs_O_d = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
        int store_lane_gmem_O_d = store_warp_regs_O_d; // (0~3)*16+(0/8)
        int store_gmem_O_addr_0 = (
          O_gmem_offset + (store_lane_gmem_O_Br + 0) * kHeadDim + store_lane_gmem_O_d);
        int store_gmem_O_addr_1 = (
          O_gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim + store_lane_gmem_O_d);
        LDST128BITS(O[store_gmem_O_addr_0]) = LDST128BITS(t_uptr_Z_0[0]);
        LDST128BITS(O[store_gmem_O_addr_1]) = LDST128BITS(t_uptr_Z_1[0]);
      }
    } // end for kWarpTileHeadDimV
  } // kWarpTileSeqLenP = 1
}

// Launch kernel for flash_attn_mma_stages_split_q_tiling_qk
template<const int kHeadDim, const int kStage>
void launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32(
  torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O) {
  // Now: fixed tile BrxBc=128x128 for d>= 128, 64x64 for d<128.
  constexpr int kMmaAtomM = 16;
  constexpr int kMmaAtomN = 8;
  constexpr int kMmaAtomK = 16;
  constexpr int kMmaTileSeqLenQ  = (kHeadDim < 128) ? 4 : 8;
  constexpr int kMmaTileSeqLenK  = 1;
  constexpr int kMmaTileSeqLenP  = (kHeadDim < 128) ? 4 : 8;
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kWarpTileSeqLenQ = 1;
  constexpr int kWarpTileSeqLenK = (kHeadDim < 128) ? 8 : 16;
  constexpr int kWarpTileSeqLenP = 1;
  constexpr int kWarpTileHeadDimV = (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV)); // (d=64)8,(d=128)16,32,....
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ; // 16*4*1=64
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK; //  8*1*8=64
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK; // 32*4*1=128, num threads
  constexpr int kPadQ = 8;
  constexpr int kPadK = 8; 
  constexpr int kPadV = 8;
  // 0/1, MMA Acc always be fp16, but O storage can be fp32 or half.
  // FP16 can provide precision to approximately 3-4 decimal places.
  // Thus, if the error does not exceed 1e-3, using FP16 storage is 
  // sufficient for most applications.
  constexpr int kOStorageAccFloat32 = (kHeadDim < 256) ? 1 : 0;
  
  // static int kMaxSramPerBlock;
  // cudaDeviceGetAttribute(&kMaxSramPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  // Calculate SRAM size needed per block, Q,K,V smem size, V shared the QK smem.
  constexpr int QK_smem_size = (kStage * (Br * (kMmaAtomK + kPadQ)) + 
                                kStage * (Bc * (kMmaAtomK + kPadK)));
  // Now, for V_smem_size, need fixed smem size, e.g 64*16*2/1024=2M;
  // R_D registers, s=2, d=64, 16 regs; d=128, 32 regs; 
  // d=256, 64 regs; d=512, 128 regs; d=1024, 256 regs;
  constexpr int V_smem_size  = (kStage * (Bc * (kMmaAtomN * 2 + kPadV))); 
  // try to let V reuse all Q+K smem after Q@K^T, reduce smem usage.
  const int smem_max_size = max(QK_smem_size, V_smem_size) * sizeof(half);

  const int QKV_batch  = Q.size(0); 
  const int QKV_head   = Q.size(1);
  const int QKV_seqlen = Q.size(2); // QKV_seqlen
  assert(QKV_seqlen % max(Br, Bc) == 0); // multiple of max(Br, Bc)
  
  // TODO: How to apply block swizzle to improve L2 Cache hit rate?
  // NOTE: reorder (B,H,Tr) -> (Tr,B*H) seems can improve L2 Cache hit rate. 
  // This might be because SM schedules blocks starting from the x-dimension. 
  // Placing Tr at the forefront ensures that identical KV pairs are placed 
  // in consecutive scheduling queues, thereby improving L2 Cache hit rates.
  // Tr(=N/Br), batch_size x num_heads
  dim3 grid(div_ceil(QKV_seqlen, Br), QKV_batch * QKV_head); 
  dim3 block(kNumThreads); // 4/8 warps per block

  cudaFuncSetAttribute(
    flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_kernel<
      kHeadDim, 
      kMmaAtomM, 
      kMmaAtomN, 
      kMmaAtomK, 
      kMmaTileSeqLenQ, 
      kMmaTileSeqLenK, 
      kMmaTileSeqLenP, 
      kMmaTileHeadDimV, 
      kWarpTileSeqLenQ, 
      kWarpTileSeqLenK, 
      kWarpTileSeqLenP, 
      kWarpTileHeadDimV, 
      kOStorageAccFloat32,
      kStage, 
      kPadQ,
      kPadK,
      kPadV
    >,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    // kMaxSramPerBlock
    98304
  );

  flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_kernel<
    kHeadDim, 
    kMmaAtomM, 
    kMmaAtomN, 
    kMmaAtomK, 
    kMmaTileSeqLenQ,  
    kMmaTileSeqLenK,
    kMmaTileSeqLenP, 
    kMmaTileHeadDimV, 
    kWarpTileSeqLenQ, 
    kWarpTileSeqLenK, 
    kWarpTileSeqLenP, 
    kWarpTileHeadDimV, 
    kOStorageAccFloat32,
    kStage, 
    kPadQ,
    kPadK,
    kPadV
  ><<<grid, block, smem_max_size>>>(
    reinterpret_cast<half*>(Q.data_ptr()),
    reinterpret_cast<half*>(K.data_ptr()),
    reinterpret_cast<half*>(V.data_ptr()),
    reinterpret_cast<half*>(O.data_ptr()),
    QKV_seqlen,
    QKV_head
  );
}

void flash_attn_mma_stages_split_q_tiling_qkv_acc_f32(torch::Tensor Q, 
                                                      torch::Tensor K, 
                                                      torch::Tensor V, 
                                                      torch::Tensor O, 
                                                      int stages) {
  CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf) // Q [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf) // K [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf) // V [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf) // O [B,H,N,D]
  const int d = Q.size(3); // B, H, N, d

  if (stages > 1) {
    switch (d)
    {
    case 32:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32<32,   2>(Q, K, V, O);
      break;
    case 64:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32<64,   2>(Q, K, V, O);
      break;
    case 96:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32<96,   2>(Q, K, V, O);
      break;
    case 128:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32<128,  2>(Q, K, V, O);
      break;
    case 256:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32<256,  2>(Q, K, V, O);
      break;
    case 512:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32<512,  2>(Q, K, V, O);
      break;
    case 1024:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32<1024, 2>(Q, K, V, O);
      break;
    default:
      throw std::runtime_error("headdim not support!");
      break;
    }
  } else {
    switch (d)
    {
    case 32:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32<32,   1>(Q, K, V, O);
      break;
    case 64:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32<64,   1>(Q, K, V, O);
      break;
    case 96:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32<96,   1>(Q, K, V, O);
      break;
    case 128:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32<128,  1>(Q, K, V, O);
      break;
    case 256:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32<256,  1>(Q, K, V, O);
      break;
    case 512:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32<512,  1>(Q, K, V, O);
      break;
    case 1024:
      launch_flash_attn_mma_stages_split_q_tiling_qkv_acc_f32<1024, 1>(Q, K, V, O);
      break;
    default:
      throw std::runtime_error("headdim not support!");
      break;
    }
  }
}
