/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"

#include "cute/tensor_predicate.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_traits_sm90_im2col.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"

#include "cutlass/conv/convolution.h"
#include "cutlass/conv/convnd_problem_shape.hpp"
#include "cutlass/conv/dispatch_policy.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/util/packed_stride.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::conv::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  conv::Operator ConvOp,
  int Stages,
  int NumSpatialDims,
  class ClusterShape,
  class KernelSchedule,
  int PipelineAsyncMmaStages,
  class TileShape_,
  class ElementA_,
  class ElementB_,
  class TiledMma_,
  class TileTraitsA_,
  class TileTraitsB_>
struct CollectiveConv<
    MainloopSm90TmaGmmaWarpSpecializedImplicitGemm<
        ConvOp, Stages, NumSpatialDims, ClusterShape, KernelSchedule, PipelineAsyncMmaStages>,
    TileShape_,
    ElementA_,
    ElementB_,
    TiledMma_,
    TileTraitsA_,
    TileTraitsB_>
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecializedImplicitGemm<
      ConvOp, Stages, NumSpatialDims, ClusterShape, KernelSchedule, PipelineAsyncMmaStages>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = typename TileTraitsA_::GmemTiledCopy;
  using GmemTiledCopyB = typename TileTraitsB_::GmemTiledCopy;
  using SmemLayoutA = typename TileTraitsA_::SmemLayout;
  using SmemLayoutB = typename TileTraitsB_::SmemLayout;
  using ArchTag = typename DispatchPolicy::ArchTag;
  static constexpr int NumSpatialDimensions = DispatchPolicy::NumSpatialDimensions;
  static constexpr int NumTensorDimensions = NumSpatialDimensions + 2;
  // Deduce the kernel-facing stride tuple types based on the dispatch policy
  // (which is a function of the number of spatial dimensions, the algorithm, etc.)
  using StrideA = decltype(detail::sm90_dispatch_policy_to_stride_A<DispatchPolicy>());
  using StrideB = decltype(detail::sm90_dispatch_policy_to_stride_B<DispatchPolicy>());

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;

  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState  = typename cutlass::PipelineState<DispatchPolicy::Stages>;

  // TODO: move pipeline mode tiling into the collective setup phase instead
  static_assert(rank(SmemLayoutA{}) == 3, "SmemLayout must be rank 3 (M/N, K, PIPE)");
  static_assert((size<0>(TileShape{}) == size<0>(SmemLayoutA{})), "SmemLayout must be compatible with the tile shape.");
  static_assert((size<2>(TileShape{}) == size<1>(SmemLayoutA{})), "SmemLayout must be compatible with the tile shape.");

  static_assert(rank(SmemLayoutB{}) == 3, "SmemLayout must be rank 3 (M/N, K, PIPE)");
  static_assert((size<1>(TileShape{}) == size<0>(SmemLayoutB{})), "SmemLayout must be compatible with the tile shape.");
  static_assert((size<2>(TileShape{}) == size<1>(SmemLayoutB{})), "SmemLayout must be compatible with the tile shape.");

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 1 or more.");
  static_assert(cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");

  // The tma load mode of wgrad is tiled for tensor A and im2col for tensor B while the tma load mode of fprop and dgrad
  // kernel is im2col for tensor A and tiled for tensor B.
  static_assert((ConvOp == conv::Operator::kWgrad
             && (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>))
             || (ConvOp != conv::Operator::kWgrad
             && (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_IM2COL> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_IM2COL_MULTICAST>)),
      "GmemTiledCopyA - invalid SM90 TMA copy atom specified.");
  static_assert((ConvOp == conv::Operator::kWgrad
             && (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_IM2COL> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_IM2COL_MULTICAST>))
             || (ConvOp != conv::Operator::kWgrad
             && (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>)),
      "GmemTiledCopyB - invalid SM90 TMA copy atom specified.");

  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  using InternalElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  using InternalElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;

  struct SharedStorage
  {
    struct TensorStorage : cute::aligned_struct<128> {
      cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
    } tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS = DispatchPolicy::PipelineAsyncMmaStages;
  static constexpr uint32_t TmaTransactionBytes =
      (size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof(InternalElementA)))+
      (size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof(InternalElementB)));

  // Host side kernel arguments
  struct Arguments {
    using ProblemShape = ConvProblemShape<ConvOp, NumSpatialDimensions>;
    ProblemShape problem_shape{};
    ElementA const* ptr_A{nullptr};
    ElementB const* ptr_B{nullptr};
  };

private:
  // Note that for fprop and dgrad kernel, the tma load mode is im2col for tensor A and tiled for
  // tensor B while for wgrad kernel, the tma load mode is tiled for tensor A and im2col for tensor
  // B since operand A, B is swapped.

  // Get tma_load_a instantce.
  template <class TensorA>
  static constexpr auto
  get_tma_load_a_instance(TensorA const& tensor_a, typename Arguments::ProblemShape const& problem_shape) {
    if constexpr (ConvOp == conv::Operator::kFprop || ConvOp == conv::Operator::kDgrad) {
      // compute the upper and lower corners based on the conv padding
      auto lower_corner_whd = detail::compute_lower_corner_whd(problem_shape);
      auto upper_corner_whd = detail::compute_upper_corner_whd(problem_shape);
      auto lower_srt = detail::compute_lower_srt(problem_shape);

      // The calculation of gbasis strides for dgrad kernel needs perform negate for dilation values.
      cute::array<int32_t, NumSpatialDimensions> stride_srt{};
      for (int i = 0; i < NumSpatialDimensions; ++i) {
        stride_srt[i] = ConvOp == conv::Operator::kDgrad ?
            -problem_shape.dilation[NumSpatialDimensions-1-i] :
            problem_shape.dilation[NumSpatialDimensions-1-i];
      }
  
      return make_im2col_tma_copy(
          GmemTiledCopyA{},
          tensor_a,
          SmemLayoutA{}(_,_,_0{}),
          product_each(shape(SmemLayoutA{}(_,_,_0{}))),
          size<1>(ClusterShape{}),
          shape(lower_corner_whd),
          shape(upper_corner_whd),
          cute::reverse(shape(problem_shape.lower_padding)),
          cute::reverse(shape(problem_shape.upper_padding)),
          cute::reverse(shape(problem_shape.traversal_stride)),
          shape(lower_srt),
          shape(stride_srt));
    }
    // TMA tiled mode for tensor A in wgrad kernel.
    else if constexpr (ConvOp == conv::Operator::kWgrad) {
      return make_tma_copy(
          GmemTiledCopyA{},
          tensor_a,
          SmemLayoutA{}(_,_,_0{}),
          make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
          size<1>(ClusterShape{}));
    }
  }

  // Get tma_load_b instantce.
  template <class TensorB>
  static constexpr auto
  get_tma_load_b_instance(TensorB const& tensor_b, typename Arguments::ProblemShape const& problem_shape) {
    if constexpr (ConvOp == conv::Operator::kFprop || ConvOp == conv::Operator::kDgrad) {
      return make_tma_copy(
          GmemTiledCopyB{},
          tensor_b,
          SmemLayoutB{}(_,_,_0{}),
          make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
          size<0>(ClusterShape{}));
    }
    // TMA im2col mode for tensor B in wgrad kernel.
    else if constexpr (ConvOp == conv::Operator::kWgrad) {
      // compute the upper and lower corners based on the conv padding
      auto lower_corner_whd = detail::compute_lower_corner_whd(problem_shape);
      auto upper_corner_whd = detail::compute_upper_corner_whd(problem_shape);
      auto lower_srt = detail::compute_lower_srt(problem_shape);
  
      return make_im2col_tma_copy(
          GmemTiledCopyB{},
          tensor_b,
          SmemLayoutB{}(_,_,_0{}),
          product_each(shape(SmemLayoutB{}(_,_,_0{}))),
          size<0>(ClusterShape{}),
          shape(lower_corner_whd),
          shape(upper_corner_whd),
          cute::reverse(shape(problem_shape.lower_padding)),
          cute::reverse(shape(problem_shape.upper_padding)),
          cute::reverse(shape(problem_shape.traversal_stride)),
          shape(lower_srt),
          cute::reverse(shape(problem_shape.dilation)));
    }
  }

public:

  // Device side kernel params
  struct Params {
    using _Submode = decltype(take<0,NumTensorDimensions-1>(typename Arguments::ProblemShape::TensorExtent{}));
    using ProblemShape = cute::conditional_t<DispatchPolicy::ConvOp == conv::Operator::kWgrad,
        Shape<int,  _Submode, _Submode>,
        Shape<_Submode, int, _Submode>>;

    // Assumption: StrideA is congruent with Problem_MK
    // Select TMA load type according to convolution operator.
    using TensorShapeA = cute::conditional_t<ConvOp == conv::Operator::kWgrad,
        decltype(repeat_like(StrideA{}, int32_t(0))),
        decltype(make_shape(_Submode{}, int(0)))>;

    using TensorShapeB = cute::conditional_t<ConvOp == conv::Operator::kWgrad,
        decltype(make_shape(int(0), _Submode{})),
        decltype(repeat_like(StrideB{}, int32_t(0)))>;

    using TMA_A = decltype(get_tma_load_a_instance(
        make_tensor(
            make_gmem_ptr(static_cast<InternalElementA const*>(nullptr)),
            make_layout(TensorShapeA{}, StrideA{})),
        ConvProblemShape<ConvOp, NumSpatialDimensions>{}));

    using TMA_B = decltype(get_tma_load_b_instance(
        make_tensor(
            make_gmem_ptr(static_cast<InternalElementB const*>(nullptr)),
            make_layout(TensorShapeB{}, StrideB{})),
        ConvProblemShape<ConvOp, NumSpatialDimensions>{}));

    // Members
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    ProblemShape problem_shape;
  };

  //
  // Methods
  //

  // Lowers the host side user facing arguments to the kernel facing lauch params
  static constexpr Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;
    // from the flat problem shape arrays of ConvProblemShape<ConvOp, N>, create a rank-3 MNK problem shape tuple
    // tma desc creation depends on the original untransformed domain.

    // A extents.
    auto shape_A_orig = args.problem_shape.get_shape_A();
    // B extents.
    auto shape_B_orig = args.problem_shape.get_shape_B();

    // Fill inferred cute strides from flat stride arrays
    auto dA = make_cute_packed_stride(StrideA{}, args.problem_shape.stride_A, ConvOp);
    auto dB = make_cute_packed_stride(StrideB{}, args.problem_shape.stride_B, ConvOp);

    auto ptr_A = reinterpret_cast<InternalElementA const*>(args.ptr_A);
    auto ptr_B = reinterpret_cast<InternalElementB const*>(args.ptr_B);

    Tensor tensor_a = make_tensor(make_gmem_ptr(ptr_A), make_layout(shape_A_orig, dA));
    Tensor tensor_b = make_tensor(make_gmem_ptr(ptr_B), make_layout(shape_B_orig, dB));

    auto tma_load_a = get_tma_load_a_instance(tensor_a, args.problem_shape);
    auto tma_load_b = get_tma_load_b_instance(tensor_b, args.problem_shape);

    auto problem_shape_mnk = args.problem_shape.get_transformed_problem_shape_MNK();

    return {
      tma_load_a,
      tma_load_b,
      problem_shape_mnk
    };
  }

  template<class ProblemShape>
  CUTLASS_HOST_DEVICE static bool
  can_implement(
      ProblemShape const& problem_shape,
      Arguments const& args) {
    // Activation and Filter channel mode extents much match
    bool implementable = true;
    // channel mode is major
    implementable &= args.problem_shape.stride_A[NumTensorDimensions-1] == 1;
    implementable &= args.problem_shape.stride_B[NumTensorDimensions-1] == 1;

    constexpr int tma_alignment_bits = 128;
    // A extents.
    auto shape_A_orig = args.problem_shape.get_shape_A();
    // B extents.
    auto shape_B_orig = args.problem_shape.get_shape_B();
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(shape_A_orig, StrideA{});
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(shape_B_orig, StrideB{});

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
      return false;
    }

    // Check valid padding values for TMA_LOAD_IM2COL
    constexpr int padding_limit = (ProblemShape::RankS == 1) ? 65536 : (ProblemShape::RankS == 2 ? 256 : 16);
    for (int i = 0; i < problem_shape.RankS; ++i) {
      implementable = implementable && problem_shape.lower_padding[i] <= padding_limit && problem_shape.lower_padding[i] >= 0;
      implementable = implementable && problem_shape.upper_padding[i] <= padding_limit && problem_shape.upper_padding[i] >= 0;
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Padding values don't meet requirements for TMA LOAD IM2COL.\n");
      return false;
    }

    if (problem_shape.groups > 1) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: This kernel does not support conv groups > 1.\n");
      return false;
    }

    return true;
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class TensorA, class TMA_LOAD_A,
    class TensorB, class TMA_LOAD_B,
    class KTileIterator
  >
  CUTLASS_DEVICE void
  load(MainloopPipeline pipeline, 
       PipelineState smem_pipe_producer_state,
       TensorA const& gA, TMA_LOAD_A& tma_load_a,
       TensorB const& gB, TMA_LOAD_B& tma_load_b,
       KTileIterator k_tile_iter, int k_tile_count,
       int thad_idx,
       TensorStorage& shared_tensors) {
    int warp_idx = canonical_warp_idx_sync();
    int warp_idx_in_warp_group  = warp_idx % 4;
    int lane_predicate = cute::elect_one_sync();

    if (warp_idx_in_warp_group == 0 and lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)

      //
      // Prepare the TMA loads for A and B
      //

      dim3 cluster_local_block_id = cute::block_id_in_cluster();
      auto block_tma_a = tma_load_a.get_slice(cluster_local_block_id.y);
      auto block_tma_b = tma_load_b.get_slice(cluster_local_block_id.x);

      // Applies the mapping from block_tma_a
      Tensor tAgA = block_tma_a.partition_S(gA);                                                 // (TMA,TMA_M,TMA_K,k)
      Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgB = block_tma_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)
      Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

      uint16_t mcast_mask_a = 0;
      uint16_t mcast_mask_b = 0;

      // Issue TmaLoads
      // Maps the tile -> block, value
      if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_IM2COL_MULTICAST> ||
                    cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
        for (int n = 0; n < size<1>(block_layout); ++n) {
          mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x,n,Int<0>{}));
        }
      }

      if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_IM2COL_MULTICAST> ||
                    cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
        for (int m = 0; m < size<0>(block_layout); ++m) {
          mcast_mask_b |= (uint16_t(1) << block_layout(m,cluster_local_block_id.y,Int<0>{}));
        }
      }

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count) {
        // LOCK smem_pipe_producer_state for _writing_
        pipeline.producer_acquire(smem_pipe_producer_state);

        //
        // Copy gmem to smem for *k_tile_iter
        //

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_producer_state);

        int write_stage = smem_pipe_producer_state.index();
        copy(tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy(tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
        ++k_tile_iter;

        // Advance smem_pipe_producer_state
        ++smem_pipe_producer_state;
      }
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_producer_state) {
    int warp_idx = canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % 4;
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (warp_idx_in_warp_group == 0 and lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all 
       * Consumer UNLOCKs), or if the stage was never used
       * then would just be acquired since the phase was 
       * still inverted from make_producer_start_state
       */
      pipeline.producer_tail(smem_pipe_producer_state);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <class FrgTensorC>
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_consumer_state,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& mainloop_params) {
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)

    //
    // Define C accumulators and A/B partitioning
    //

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor tCsA = thread_mma.partition_A(sA);                                                 // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB);                                                 // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments/descriptors"
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);                                           // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);                                           // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));                                                         // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                         // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                                          // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                                       // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                         // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                         // PIPE

    //
    // PIPELINED MAIN LOOP
    //
    static_assert((0 <= K_PIPE_MMAS) && (K_PIPE_MMAS <  K_PIPE_MAX),
        "ERROR : Incorrect number of MMAs in flight");

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_consumer_state;

    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);

    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

    warpgroup_fence_operand(accum);
    CUTLASS_PRAGMA_UNROLL
    for (int k_tile_prologue = prologue_mma_count; k_tile_prologue > 0; --k_tile_prologue) {
      // WAIT on smem_pipe_consumer_state until its data are available (phase bit flips from rdPhaseBit value)
      pipeline.consumer_wait(smem_pipe_consumer_state);

      int read_stage = smem_pipe_consumer_state.index();
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }

      warpgroup_commit_batch();

      ++smem_pipe_consumer_state;
    }

    warpgroup_fence_operand(accum);
    // Mainloop GMMAs
    k_tile_count -= prologue_mma_count;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      // WAIT on smem_pipe_consumer_state until its data are available (phase bit flips from rdPhaseBit value)
      pipeline.consumer_wait(smem_pipe_consumer_state);

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_consumer_state.index();
      warpgroup_fence_operand(accum);
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M) x (V,N) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();

      /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_producer_state is consumed
      warpgroup_wait<K_PIPE_MMAS>();
      warpgroup_fence_operand(accum);

      // UNLOCK smem_pipe_release, done _computing_ on it
      pipeline.consumer_release(smem_pipe_release);

      // Advance smem_pipe_consumer_state and smem_pipe_release
      ++smem_pipe_consumer_state;
      ++smem_pipe_release;
    }

    warpgroup_fence_operand(accum);
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);
    
    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(smem_pipe_release);                 // UNLOCK smem_pipe_release, done _computing_ on it
      ++smem_pipe_release;
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::conv::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
