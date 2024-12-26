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
         const int kHeadDim,          // Headdim, 32,64,128     
         const int kMmaAtomM,         // MMA Atom M, 16
         const int kMmaAtomN,         // MMA Atom N, 8
         const int kMmaAtomK,         // MMA Atom K, 16
         const int kMmaTileSeqLenQ,   // 4, more MMA(warp), M=16*4=64, Q@K^T=[Br(M), d(K)]@[d(K),  Bc(N)]  
         const int kMmaTileSeqLenK,   // 1, more MMA(warp), N=8*1 =8,  Q@K^T=[Br(M), d(K)]@[d(K),  Bc(N)]    
         const int kMmaTileSeqLenP,   // 4, more MMA(warp), M=16*4=64, P@V  =[Br(M),Bc(K)]@[Bc(K), d(N) ]
         const int kMmaTileHeadDimV,  // 1, more MMA(warp), N=8*1 =8,  P@V  =[Br(M),Bc(K)]@[Bc(K), d(N) ]       
         const int kWarpTileSeqLenQ,  // 1, more values, M, Br=64*1=64, matmul M 
         const int kWarpTileSeqLenK,  // 8, more values, N, Bc=8*8 =64, matmul N
         const int kWarpTileSeqLenP,  // 1, more values, M, Br=64*1=64, matmul M
         const int kWarpTileHeadDimV, // 8, more values, N, d=8*(1|2|3|4|...)=8|...|32|64|96|128|...
         const int kStage, 
         const int kPad
         >
__global__ void __launch_bounds__(
  WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK) 
flash_attn_mma_stages_split_q_tiling_qk_kernel(half* Q, 
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
  static_assert(kStage < 3 && kStage > 0); 
  static_assert(kPad >= 0 && kPad % 8 == 0); // 0,8,16
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
  // TODO: Mapping V gmem -> tid -> smem, V[kMmaAtomK,kMmaAtomN]=[16,64/128], 128 threads.
  // Mapping V gmem -> tid -> smem, V[kMmaAtomK,d]=[16,64/128], 128 threads.
  int load_smem_V_Bc = (tid / (kNumThreads / kMmaAtomK)); // kMmaAtomK 16, tid / 8, row 0~15
  int load_smem_V_d  = (tid % (kNumThreads / kMmaAtomK)) * (kHeadDim / (kNumThreads / kMmaAtomK)); // (tid % 8) * 8, 0,8,56...
  // global Q row of current head for tile [Br,d] per block.
  int load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br; 
  if (load_gmem_Q_Br >= QKV_seqlen) return;
  constexpr bool kIsVCanLoadIn128b = (kHeadDim / (kNumThreads / kMmaAtomK)) % 8 == 0;
  constexpr bool kIsVCanLoadIn64b  = (kHeadDim / (kNumThreads / kMmaAtomK)) % 4 == 0;
  static_assert(kIsVCanLoadIn128b || kIsVCanLoadIn64b, "V can't load in 128b or 64b."); // 32,64,128,192,256,...

  // Shared memory for Q,K,V, we don not need additional smem for O 
  // collective store which perform via registers reuse and warp shuffle.
  extern __shared__ half smem[];
  // Split Q + Shared KV SMEM + Fine grain tiling, only need O(1) SRAM complexity.
  constexpr int Q_tile_size = Br * (kMmaAtomK + kPad); // Q[Br,16], 64*16*2=2048 bytes, 2M
  constexpr int K_tile_size = Bc * (kMmaAtomK + kPad); // K[Bc,16], 2M
  constexpr int V_tile_size = kMmaAtomK * (kHeadDim + kPad); // V[16,d], 2M
  // TODO: optimize QKV kStage smem store layout as in HGEMM.
  half* Q_tile_smem = smem; // 8M/16M
  half* K_tile_smem = Q_tile_smem + kStage * Q_tile_size; // 8M/16M
  half* V_tile_smem = Q_tile_smem; // V may reuse all Q+K smem after Q@K^T.
  // stage 1, Q/K smem = 64*16*2/1024=2M, V smem =16*d(64|128|...)*2/1024=2M/4M/..
  // stage 1, total smem = max(QK_smem, V_smem) = 4M if d <= 64 else V_smem.
  // stage 1, V shared QK smem, Br=Bc=64,  d=64:  2M+(2M) =4M,  +Pad(2M)  = 6M
  // stage 1, V shared QK smem, Br=Bc=128, d=64:  4M+4M   =8M,  +Pad(2M)  = 10M
  // stage 2, V shared QK smem, Br=Bc=64,  d=64:  4M+(4M) =8M,  +Pad(2M)  = 10M
  // stage 2, V shared QK smem, Br=Bc=128, d=64:  8M+8M   =16M,  +Pad(2M) = 18M
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
  uint32_t R_V[kWarpTileHeadDimV][2]; // [8][2]
  // NOTE: For R_V[kWarpTileHeadDimV][2], kWarpTileHeadDimV will increase with d.
  // so, for large d, R_V will need more registers and cause performance down.
  // We have to find a way to apply MMA level tiling for V(R_V) for large d.
  // registers for current tile_K_seqlen within, [64,64] = S_tile[Br,Bc]
  // = Q_tile[Br,d] * K[Bc,d], each thread hold 2x32 bits regs.
  uint32_t R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][ 2]; // [1][8][2]
  // registers for tile_K_seqlen O=PV[Br,d]=P@V, [2][2/4][2], 8 or 16 regs.
  uint32_t R_O[kWarpTileSeqLenP][kWarpTileHeadDimV][2]; // [1][8][2]
  // registers final Output [D]=final rescale(R_O), [2][2/4][2], 8 or 16 regs.
  uint32_t R_D[kWarpTileSeqLenP][kWarpTileHeadDimV][2]; // [1][8][2]
  fill_3D_regs<uint32_t, kWarpTileSeqLenQ, kWarpTileSeqLenK,  2>(R_S, 0);
  fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV, 2>(R_D, 0);
  fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV, 2>(R_O, 0);
  
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
                             load_smem_Q_Br * (kMmaAtomK + kPad) + 
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
                             load_smem_K_Bc * (kMmaAtomK + kPad) + 
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
    fill_3D_regs<uint32_t, kWarpTileSeqLenQ, kWarpTileSeqLenK, 2>(R_S, 0);
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
                               load_smem_Q_Br * (kMmaAtomK + kPad) + 
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
                               load_smem_K_Bc * (kMmaAtomK + kPad) + 
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
                             load_smem_Q_Br * (kMmaAtomK + kPad) + 
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
                             load_smem_K_Bc * (kMmaAtomK + kPad) + 
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
      #pragma unroll
      for (int i = 0; i < kWarpTileSeqLenQ; ++i) { // Q[Br,d]=[M,K]
        int warp_smem_Q_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
        int lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16; // 0~15
        int lane_smem_Q_d  = (lane_id / 16) * 8; // 0,8
        uint32_t lane_smem_Q_ptr = (
            smem_Q_base_ptr + (smem_sel * Q_tile_size + 
                               lane_smem_Q_Br * (kMmaAtomK + kPad) + 
                               lane_smem_Q_d) * sizeof(half)
        );
        LDMATRIX_X4(R_Q[i][0], R_Q[i][1], R_Q[i][2], R_Q[i][3], 
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
                               lane_smem_K_Bc * (kMmaAtomK + kPad) + 
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
      #pragma unroll
      for (int i = 0; i < kWarpTileSeqLenQ; ++i) {
        #pragma unroll
        for (int j = 0; j < kWarpTileSeqLenK; ++j) {
          HMMA16816(R_S[i][j][0], R_S[i][j][1], 
                    R_Q[i][0], R_Q[i][1], R_Q[i][2], R_Q[i][3], 
                    R_K[j][0], R_K[j][1], 
                    R_S[i][j][0], R_S[i][j][1]);
        }
      }

      if constexpr (kStage > 1) {
        // Wait next Q, K tile g2s ready.
        CP_ASYNC_WAIT_GROUP(kStage - 2);
        __syncthreads(); 
      }

    } // end loop over d, S=Q@K^T
    __syncthreads();

    // V g2s stages. (reuse Q+K smem) load [16,d] from [Bc,d]
    if constexpr (kStage > 1) {
      #pragma unroll
      for (int stage = 0; stage < (kStage - 1); ++stage) {
        // V g2s
        int load_gmem_V_Bc = (
          (tile_K_seqlen * Bc) + (stage * kMmaAtomK) + load_smem_V_Bc); // 0~15
        int load_gmem_V_d  = load_smem_V_d;
        int load_gmem_V_addr = (
          V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
        uint32_t load_smem_V_ptr = (
          smem_V_base_ptr + (stage * V_tile_size + 
                             load_smem_V_Bc * (kHeadDim + kPad) + 
                             load_smem_V_d) * sizeof(half)
        );
        // headdim must be multiple of 32, (kHeadDim/8)%8==0 for 128 bits ld.
        if constexpr (kIsVCanLoadIn128b) {
          // 64,128,192,256,...
          #pragma unroll
          for (int i = 0; i < (kHeadDim / (kNumThreads / kMmaAtomK)); i += 8) {
            CP_ASYNC_CG(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 16);
          }
        } else {
          // 32,96,160,224
          #pragma unroll
          for (int i = 0; i < (kHeadDim / (kNumThreads / kMmaAtomK)); i += 4) {
            CP_ASYNC_CA(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 8);
          }
        }
        CP_ASYNC_COMMIT_GROUP();
      } // end for stage
    }
  
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

    // Row max for [Br,Bc] tile, Thread -> Warp -> Block.
    #pragma unroll
    for (int i = 0; i < kWarpTileSeqLenQ; ++i) {
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
        float2 t_reg_S_0 = __half22float2(HALF2(R_S[i][j][0])); // 0~7  {c0, c1}
        float2 t_reg_S_1 = __half22float2(HALF2(R_S[i][j][1])); // 8~15 {c2, c3}
        // This should be the row max after S = (Q @ K^T) / sqrt(d)
        float tmp_max_0 = max(t_reg_S_0.x, t_reg_S_0.y) * scale;
        float tmp_max_1 = max(t_reg_S_1.x, t_reg_S_1.y) * scale;
        lane_row_max_new[i][0] = max(lane_row_max_new[i][0], tmp_max_0);
        lane_row_max_new[i][1] = max(lane_row_max_new[i][1], tmp_max_1);
      } // end for kWarpTileSeqLenK

      // Warp level reduce max, warp_size = 4
      // Each thread contains the maximum of 2 rows of Br, 
      // and only the values of T0, T4, ..., T28 are used.
      lane_row_max_new[i][0] = warp_reduce_max<float, 4>(lane_row_max_new[i][0]);
      lane_row_max_new[i][1] = warp_reduce_max<float, 4>(lane_row_max_new[i][1]);
    } // end for kWarpTileSeqLenQ
    __syncthreads();

    // Exp sum and mul scale_factor for [Br,Bc] tile, Thread -> Warp -> Block.
    #pragma unroll
    for (int i = 0; i < kWarpTileSeqLenQ; ++i) {
      // Use latest global row max without update.
      // Br 0, row_id, 0~7,  16~23, 32~39, 48~55; 
      float block_row_max_new_0 = lane_row_max_new[i][0]; 
      // Br 1, row_id, 8~15, 24~31, 40~47, 56~63;
      float block_row_max_new_1 = lane_row_max_new[i][1];
  
      float block_row_max_old_0 = lane_block_row_max_old[i][0];
      float block_row_max_old_1 = lane_block_row_max_old[i][1];
      // Apply m_new = max(m_old, m_new) here.
      block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);

      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        float2 t_reg_S_0 = __half22float2(HALF2(R_S[i][j][0])); // 0~7  {c0, c1}
        float2 t_reg_S_1 = __half22float2(HALF2(R_S[i][j][1])); // 8~15 {c2, c3}
        // P = Exp(S - m_new), fmaf(x, y, z) = x * y + z;
        t_reg_S_0.x = __expf(__fmaf_rn(t_reg_S_0.x, scale, - block_row_max_new_0));
        t_reg_S_0.y = __expf(__fmaf_rn(t_reg_S_0.y, scale, - block_row_max_new_0));
        t_reg_S_1.x = __expf(__fmaf_rn(t_reg_S_1.x, scale, - block_row_max_new_1));
        t_reg_S_1.y = __expf(__fmaf_rn(t_reg_S_1.y, scale, - block_row_max_new_1));
        lane_row_sum_new[i][0] += (t_reg_S_0.x + t_reg_S_0.y);
        lane_row_sum_new[i][1] += (t_reg_S_1.x + t_reg_S_1.y);
        // Update R_S for P[Br,Bc] = Exp(S-m), point wise.
        HALF2(R_S[i][j][0]) = __float22half2_rn(t_reg_S_0);
        HALF2(R_S[i][j][1]) = __float22half2_rn(t_reg_S_1);
      } // end for kWarpTileSeqLenK

      // Warp level reduce sum, warp_size = 4
      lane_row_sum_new[i][0] = warp_reduce_sum<float, 4>(lane_row_sum_new[i][0]);
      lane_row_sum_new[i][1] = warp_reduce_sum<float, 4>(lane_row_sum_new[i][1]);
    } // end for kWarpTileSeqLenQ
    __syncthreads();
    
    // <loop over V Bc>: P[Br,Bc]@V[Bc,d]=[Br,d]=[64,64/128], partion Attention.
    // Matmul with NN layout: P[Br,Bc] row major, V[Bc,d] row major.
    // Make sure to clear the states in R_O before MMA for P@V for each step.

    // NOTE: Values for P[Br,Bc] already in R_S registers, can we use these 
    // registers for P(A) matrix directly ? How to do that ?
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

    // Wait V g2s stages ready.
    if constexpr (kStage > 1) {
      CP_ASYNC_WAIT_GROUP(kStage - 2); // s2->0, s3->1, s4->2
      __syncthreads(); 
    }
    
    // <HGEMM in registers> P@V=[Br,Bc]@[Bc,d]
    fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV, 2>(R_O, 0);
    #pragma unroll
    for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); ++tile_V_Bc) {
      // s2 tn 0->0, 1->1, 2->0; s3 tn 0->0, 1->1, 2->2, 3->0;
      int smem_sel      = (tile_V_Bc) % kStage;   
      // s2 tn 0->1, 1->0, 2->1; s3 tn 0->2, 1->0, 2->1, 3->2;  
      int smem_sel_next = (tile_V_Bc + (kStage - 1)) % kStage;

      // stages for V
      if constexpr (kStage > 1) {
        if ((tile_V_Bc + 1) < (Bc / kMmaAtomK)) {
          //  next V tile g2s
          int load_gmem_V_Bc = (
            (tile_K_seqlen * Bc) + (tile_V_Bc + 1) * kMmaAtomK + load_smem_V_Bc); // 0~15
          int load_gmem_V_d  = load_smem_V_d;
          int load_gmem_V_addr = (
            V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
          uint32_t load_smem_V_ptr = (
            smem_V_base_ptr + (smem_sel_next * V_tile_size + 
                               load_smem_V_Bc * (kHeadDim + kPad) + 
                               load_smem_V_d) * sizeof(half)
          );
          // headdim must be multiple of 32, (kHeadDim/8)%8==0 for 128 bits ld.
          if constexpr (kIsVCanLoadIn128b) {
            // 64,128,192,256,...
            #pragma unroll
            for (int i = 0; i < (kHeadDim / (kNumThreads / kMmaAtomK)); i += 8) {
              CP_ASYNC_CG(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 16);
            }
          } else {
            // 32,96,160,224
            #pragma unroll
            for (int i = 0; i < (kHeadDim / (kNumThreads / kMmaAtomK)); i += 4) {
              CP_ASYNC_CA(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 8);
            }
          }
          CP_ASYNC_COMMIT_GROUP();
        }
      } else {
        // sync load curr V g2s
        int load_gmem_V_Bc = (
          (tile_K_seqlen * Bc) + (tile_V_Bc * kMmaAtomK) + load_smem_V_Bc); // 0~15
        int load_gmem_V_d = load_smem_V_d;
        int load_gmem_V_addr = (
          V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
        uint32_t load_smem_V_ptr = (
          smem_V_base_ptr + (smem_sel * V_tile_size + 
                             load_smem_V_Bc * (kHeadDim + kPad) + 
                             load_smem_V_d) * sizeof(half)
        );
        // headdim must be multiple of 32, (kHeadDim/8)%8==0 for 128 bits ld.
        if constexpr (kIsVCanLoadIn128b) {
          // 64,128,192,256,...
          #pragma unroll
          for (int i = 0; i < (kHeadDim / (kNumThreads / kMmaAtomK)); i += 8) {
            CP_ASYNC_CG(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 16);
          }
        } else {
          // 32,96,160,224
          #pragma unroll
          for (int i = 0; i < (kHeadDim / (kNumThreads / kMmaAtomK)); i += 4) {
            CP_ASYNC_CA(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 8);
          }
        }
        CP_ASYNC_COMMIT_GROUP();
        // Wait curr V tile ready.
        CP_ASYNC_WAIT_GROUP(0); 
        __syncthreads(); 
      }

      // Load k16n8 V from smem -> regs, R_KV, ldmatrix.x2.trans.
      #pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) { 
        int warp_smem_V_d  = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN; // d, matmaul N
        int lane_smem_V_Bc = lane_id % 16; // 0~15; Bc, matmul K
        int lane_smem_V_d  = warp_smem_V_d; // 0
        uint32_t lane_smem_V_ptr = (
          smem_V_base_ptr + (smem_sel * V_tile_size + 
                             lane_smem_V_Bc * (kHeadDim + kPad) + 
                             lane_smem_V_d) * sizeof(half)
        );
        LDMATRIX_X2_T(R_V[j][0], R_V[j][1], lane_smem_V_ptr); // R_V
      }
      if constexpr (kStage < 2) {
        // Wait V s2r ready if kStage < 2 in order to avoid 
        // the next V tile g2s overwrite.
        __syncthreads();
      }
      
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
      #pragma unroll
      for (int i = 0; i < kWarpTileSeqLenP; ++i) { // 1
        #pragma unroll
        for (int j = 0; j < kWarpTileHeadDimV; ++j) { // 8, 16, 32, ...
          HMMA16816(R_O[i][j][0], R_O[i][j][1], 
                    R_S[i][w][0], R_S[i][w][1], R_S[i][w + 1][0],  R_S[i][w + 1][1], 
                    R_V[j][0],    R_V[j][1],
                    R_O[i][j][0], R_O[i][j][1]);
        }
      }

      if constexpr (kStage > 1) {
        // Wait next V tile g2s ready.
        CP_ASYNC_WAIT_GROUP(kStage - 2); 
        __syncthreads();
      }

    } // end for V Bc.
    __syncthreads(); 

    // Rescale O -> Update row sum Exp -> then, Update row max.
    #pragma unroll
    for (int i = 0; i < kWarpTileSeqLenP; ++i) { // kWarpTileSeqLenQ=kWarpTileSeqLenP=1
      // m = max(m_old, m_new), l = exp(m_old - m) * l_old + l_new (FA2 paper)
      // Br 0, row_id, 0~7,  16~23, 32~39, 48~55; Br 1, row_id, 8~15, 24~31, 40~47, 56~63
      float block_row_max_new_0 = lane_row_max_new[i][0]; 
      float block_row_max_new_1 = lane_row_max_new[i][1];
      float block_row_sum_new_0 = lane_row_sum_new[i][0];
      float block_row_sum_new_1 = lane_row_sum_new[i][1];
      
      float block_row_max_old_0 = lane_block_row_max_old[i][0];
      float block_row_max_old_1 = lane_block_row_max_old[i][1];
      // NOTE: max(-inf, val) = val.
      block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);   
      // Avoid inf value while using m_old for rescaling O.
      block_row_max_old_0 = (tile_K_seqlen > 0 ? block_row_max_old_0 : 
                                                 block_row_max_new_0);                                       
      block_row_max_old_1 = (tile_K_seqlen > 0 ? block_row_max_old_1 : 
                                                 block_row_max_new_1);  

      // rescale factor for O and l, exp(m_old - m)
      float rescale_o_factor_0 = __expf(block_row_max_old_0 - block_row_max_new_0);
      float rescale_o_factor_1 = __expf(block_row_max_old_1 - block_row_max_new_1);
      // 0. Rescale O: Online rescaling O each tile_K_seqlen step, need m_new, m_old.
      // m = max(m_old, m_new), O_new[Br,d] = exp(m_old - m) * O_old + P@V
      #pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) { // 8, 16, 32, ...
        float2 t_reg_O_0 = __half22float2(HALF2(R_O[i][j][0])); // 0~7  {c0, c1}
        float2 t_reg_O_1 = __half22float2(HALF2(R_O[i][j][1])); // 8~15 {c2, c3}
        float2 t_reg_D_0 = __half22float2(HALF2(R_D[i][j][0])); // 0~7  {c0, c1}
        float2 t_reg_D_1 = __half22float2(HALF2(R_D[i][j][1])); // 8~15 {c2, c3}
        // Note that the formula in the FA2 paper is incorrect; here, 
        // the inverse of the exp function should not be taken, as it 
        // would result in an error during rescaling, namely, you have
        // use exp(m_old - m_new), not 1/(m_old - m_new).
        // O_new[Br,d] = exp(m_old - m_new) * O_old + P@V
        t_reg_D_0.x = __fmaf_rn(rescale_o_factor_0, t_reg_D_0.x, t_reg_O_0.x);
        t_reg_D_0.y = __fmaf_rn(rescale_o_factor_0, t_reg_D_0.y, t_reg_O_0.y);
        t_reg_D_1.x = __fmaf_rn(rescale_o_factor_1, t_reg_D_1.x, t_reg_O_1.x);
        t_reg_D_1.y = __fmaf_rn(rescale_o_factor_1, t_reg_D_1.y, t_reg_O_1.y);
        HALF2(R_D[i][j][0]) = __float22half2_rn(t_reg_D_0);
        HALF2(R_D[i][j][1]) = __float22half2_rn(t_reg_D_1);
      } // end for kWarpTileHeadDimV.

      // Now, we can update m, l after O has been scaled.
      // 1. First, update block row sum Exp for each lane which
      // need both m_new and m_old.
      float block_row_sum_old_0 = lane_block_row_sum_old[i][0];
      float block_row_sum_old_1 = lane_block_row_sum_old[i][1];
      // Update l = exp(m_old - m_new) * l_old + row_sum(P).
      lane_block_row_sum_old[i][0] = (__fmaf_rn(
        rescale_o_factor_0, block_row_sum_old_0, block_row_sum_new_0));
      lane_block_row_sum_old[i][1] = (__fmaf_rn(
        rescale_o_factor_1, block_row_sum_old_1, block_row_sum_new_1));
      // 2. Then, update block row max for each lane.
      lane_block_row_max_old[i][0] = block_row_max_new_0;
      lane_block_row_max_old[i][1] = block_row_max_new_1;
    }
  } // end loop over N
  __syncthreads();

  // Finaly, we still have to rescale O once more.
  // O_output(D) = ( 1/l_final ) * O_final (FA2 paper)
  // NOTE: Here, we choose to reuse R_O as final output 
  // in order to reduce regs usage.
  #pragma unroll
  for (int i = 0; i < kWarpTileSeqLenP; ++i) { // 1
    float rescale_factor_0 = __frcp_rn(lane_block_row_sum_old[i][0]);
    float rescale_factor_1 = __frcp_rn(lane_block_row_sum_old[i][1]);
    #pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) { // 8, 16, 32, ...
      float2 t_reg_D_0 = __half22float2(HALF2(R_D[i][j][0])); // 0~7  {c0, c1}
      float2 t_reg_D_1 = __half22float2(HALF2(R_D[i][j][1])); // 8~15 {c2, c3}
      t_reg_D_0.x = rescale_factor_0 * t_reg_D_0.x;
      t_reg_D_0.y = rescale_factor_0 * t_reg_D_0.y;
      t_reg_D_1.x = rescale_factor_1 * t_reg_D_1.x;
      t_reg_D_1.y = rescale_factor_1 * t_reg_D_1.y;
      HALF2(R_D[i][j][0]) = __float22half2_rn(t_reg_D_0);
      HALF2(R_D[i][j][1]) = __float22half2_rn(t_reg_D_1);
    }
  }

  // Store O(D): Write O[Br,d] from regs -> gmem, collective store 
  // with reg reuse & warp shuffle. need R_Z[2][4]. 
  // TODO: reuse Q smem for collective store: regs -> smem -> gmem
  #pragma unroll
  for (int i = 0; i < kWarpTileSeqLenP; ++i) { // 1
    #pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) { // 8
      // we have to use new R_Z regs for collective store.
      uint32_t R_Z[2][4];
      R_Z[0][0] = R_D[i][j][0]; R_Z[1][0] = R_D[i][j][1]; // warp_size 4
      R_Z[0][1] = __shfl_sync((0xffffffff), R_D[i][j][0], lane_id + 1, 4);
      R_Z[0][2] = __shfl_sync((0xffffffff), R_D[i][j][0], lane_id + 2, 4);
      R_Z[0][3] = __shfl_sync((0xffffffff), R_D[i][j][0], lane_id + 3, 4);
      R_Z[1][1] = __shfl_sync((0xffffffff), R_D[i][j][1], lane_id + 1, 4);
      R_Z[1][2] = __shfl_sync((0xffffffff), R_D[i][j][1], lane_id + 2, 4);
      R_Z[1][3] = __shfl_sync((0xffffffff), R_D[i][j][1], lane_id + 3, 4);
      // st.global.v4 128 bits. [Br,d]
      if (lane_id % 4 == 0) {
        // (0/1)*32 + (0/1)*16=(0,16,32,48), + 0~7 -> 0~56
        int store_warp_regs_O_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenP ) + i * kMmaAtomM;
        int store_lane_gmem_O_Br = O_tile_id * Br + store_warp_regs_O_Br + lane_id / 4; // 0~7
        // (0~3)*16 + (0/1)*8=(0,8,16,24,...,48,56)
        int store_warp_regs_O_d = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
        int store_lane_gmem_O_d = store_warp_regs_O_d; // (0~3)*16+(0/8)
        int store_gmem_O_addr_0 = (
          O_gmem_offset + (store_lane_gmem_O_Br + 0) * kHeadDim + store_lane_gmem_O_d);
        int store_gmem_O_addr_1 = (
          O_gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim + store_lane_gmem_O_d);
        LDST128BITS(O[store_gmem_O_addr_0]) = LDST128BITS(R_Z[0][0]);
        LDST128BITS(O[store_gmem_O_addr_1]) = LDST128BITS(R_Z[1][0]);
      }
    } // end for kWarpTileHeadDimV
  } // end for kWarpTileSeqLenQ
}

// Launch kernel for flash_attn_mma_stages_split_q_tiling_qk
template<const int kHeadDim, const int kStage>
void launch_flash_attn_mma_stages_split_q_tiling_qk(
  torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O) {
  // Now: fixed tile BrxBc=128x128 for d>= 128, 64x64 for d<128.
  // TODO: dynamic tile size for Br, Bc according to kHeadDim and shared memory size.
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
  constexpr int kPad = 8; // 0.25~0.5M
  
  // static int kMaxSramPerBlock;
  // cudaDeviceGetAttribute(&kMaxSramPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  // Calculate SRAM size needed per block, Q,K,V smem size, V shared the QK smem.
  constexpr int QK_smem_size = (kStage * (Br * (kMmaAtomK + kPad)) + 
                                kStage * (Bc * (kMmaAtomK + kPad)));
  // Now, for V_smem_size, s=2, d=64, 4M, 16 regs; d=128, 8M, 32 regs; 
  // d=256, 16M, 64 regs; d=512, 32M, 128 regs; d=1024, 64M, 256 regs;
  // TODO: Fully sub-tiling for d while perform P@V, kMmaAtomK * (kMmaAtomN)
  constexpr int V_smem_size  = (kStage * (kMmaAtomK * (kHeadDim + kPad))); 
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
    flash_attn_mma_stages_split_q_tiling_qk_kernel<
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
      kStage, 
      kPad
    >,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    // kMaxSramPerBlock
    98304
  );

  flash_attn_mma_stages_split_q_tiling_qk_kernel<
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
    kStage, 
    kPad
  ><<<grid, block, smem_max_size>>>(
    reinterpret_cast<half*>(Q.data_ptr()),
    reinterpret_cast<half*>(K.data_ptr()),
    reinterpret_cast<half*>(V.data_ptr()),
    reinterpret_cast<half*>(O.data_ptr()),
    QKV_seqlen,
    QKV_head
  );
}

void flash_attn_mma_stages_split_q_tiling_qk(torch::Tensor Q, 
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
      launch_flash_attn_mma_stages_split_q_tiling_qk<32,   2>(Q, K, V, O);
      break;
    case 64:
      launch_flash_attn_mma_stages_split_q_tiling_qk<64,   2>(Q, K, V, O);
      break;
    case 96:
      launch_flash_attn_mma_stages_split_q_tiling_qk<96,   2>(Q, K, V, O);
      break;
    case 128:
      launch_flash_attn_mma_stages_split_q_tiling_qk<128,  2>(Q, K, V, O);
      break;
    case 256:
      launch_flash_attn_mma_stages_split_q_tiling_qk<256,  2>(Q, K, V, O);
      break;
    case 512:
      launch_flash_attn_mma_stages_split_q_tiling_qk<512,  2>(Q, K, V, O);
      break;
    case 1024:
      launch_flash_attn_mma_stages_split_q_tiling_qk<1024, 2>(Q, K, V, O);
      break;
    default:
      throw std::runtime_error("headdim not support!");
      break;
    }
  } else {
    switch (d)
    {
    case 32:
      launch_flash_attn_mma_stages_split_q_tiling_qk<32,   1>(Q, K, V, O);
      break;
    case 64:
      launch_flash_attn_mma_stages_split_q_tiling_qk<64,   1>(Q, K, V, O);
      break;
    case 96:
      launch_flash_attn_mma_stages_split_q_tiling_qk<96,   1>(Q, K, V, O);
      break;
    case 128:
      launch_flash_attn_mma_stages_split_q_tiling_qk<128,  1>(Q, K, V, O);
      break;
    case 256:
      launch_flash_attn_mma_stages_split_q_tiling_qk<256,  1>(Q, K, V, O);
      break;
    case 512:
      launch_flash_attn_mma_stages_split_q_tiling_qk<512,  1>(Q, K, V, O);
      break;
    case 1024:
      launch_flash_attn_mma_stages_split_q_tiling_qk<1024, 1>(Q, K, V, O);
      break;
    default:
      throw std::runtime_error("headdim not support!");
      break;
    }
  }
}
