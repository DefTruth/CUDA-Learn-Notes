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

#include "cutlass/gemm/collective/builders/sm90_common.inl"

// SM90 Collective Builders should be used only starting CUDA 12.0
#if (__CUDACC_VER_MAJOR__ >= 12)
#define CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template<int CapacityBytes, class ElementA, class ElementB, class TileShapeMNK, int stages>
constexpr int
compute_stage_count_or_override(StageCount<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template<int CapacityBytes, class ElementA, class ElementB, class TileShapeMNK, int stages>
constexpr int
compute_stage_count_or_override(cute::Int<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template<int CapacityBytes, class ElementA, class ElementB, class TileShapeMNK, int carveout_bytes>
constexpr int
compute_stage_count_or_override(StageCountAutoCarveout<carveout_bytes> stage_count) {
  constexpr auto mainloop_pipeline_bytes = sizeof(typename cutlass::PipelineTmaAsync<1>::SharedStorage);
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr int stage_bytes =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    static_cast<int>(mainloop_pipeline_bytes);

  return (CapacityBytes - carveout_bytes) / stage_bytes;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity (with an optional scale matrix), or overrides with manual count. 
template<int CapacityBytes, class ElementA, class ElementB, class ElementScale, class ElementZero, class TileShapeMNK, int stages>
constexpr int
compute_stage_count_or_override_single_affine_transformed_input(StageCount<stages> stage_count) {
  return stages;
}

template <class Element>
constexpr int get_bits_for_possibly_void_element() { 
  if constexpr (cute::is_same_v<Element, void>) {
    return 0;
  } 
  else {
    return sizeof_bits<Element>::value;
  }
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity (with an optional scale matrix), or overrides with manual count. 
template<int CapacityBytes, class ElementA, class ElementB, class ElementScale, class ElementZero, class TileShapeMNK, int carveout_bytes>
constexpr int
compute_stage_count_or_override_single_affine_transformed_input(StageCountAutoCarveout<carveout_bytes> stage_count) {

  // 32 bytes to account for barriers etc.
  constexpr auto mainloop_pipeline_bytes = sizeof(typename cutlass::PipelineTmaAsync<1>::SharedStorage);
  constexpr int scale_zero_k_tile = 1;
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr auto s_bits = get_bits_for_possibly_void_element<ElementScale>();
  constexpr auto z_bits = get_bits_for_possibly_void_element<ElementZero>();

  constexpr auto scale_bytes = cutlass::bits_to_bytes(s_bits * size<0>(TileShapeMNK{}) * scale_zero_k_tile);
  constexpr auto zero_bytes  = cutlass::bits_to_bytes(z_bits * size<0>(TileShapeMNK{}) * scale_zero_k_tile);
  static_assert(scale_bytes % 128 == 0, "Scale bytes must be a multiple of 128");
  static_assert(zero_bytes  % 128 == 0, "Zero bytes must be a multiple of 128");

  // When scales are void, s_bits will be 0 so no smem will be allocated for scales. 
  constexpr int stage_bytes =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    static_cast<int>(scale_bytes + zero_bytes + mainloop_pipeline_bytes);

  return (CapacityBytes - carveout_bytes) / stage_bytes;
}

template <class ElementA, class LayoutA, class ElementB, class LayoutB>
constexpr bool
is_swapAB(){
  constexpr bool IsInputSizeTwoBytes = is_input_size_two_bytes<ElementA, ElementB>();
  constexpr bool IsLayoutAkBmn = cutlass::gemm::detail::is_k_major_A<LayoutA>() &&
                                 cutlass::gemm::detail::is_mn_major_B<LayoutB>();
  constexpr bool SwapAB = !IsInputSizeTwoBytes && IsLayoutAkBmn;
  return SwapAB;
}

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class KernelScheduleType>
constexpr bool
is_warpspecialized_transpose_B(){
  constexpr bool IsInputSizeTwoBytes = is_input_size_two_bytes<ElementA, ElementB>();
  constexpr bool IsLayoutAmnBmn = cutlass::gemm::detail::is_mn_major_A<LayoutA>() &&
                                  cutlass::gemm::detail::is_mn_major_B<LayoutB>();
  constexpr bool IsWarpSpecialized = cute::is_base_of_v<KernelTmaWarpSpecialized, KernelScheduleType>                ||
                                     cute::is_base_of_v<KernelTmaWarpSpecializedPingpong, KernelScheduleType>        ||
                                     cute::is_base_of_v<KernelTmaWarpSpecializedCooperative, KernelScheduleType>     || 
                                     cute::is_base_of_v<KernelCpAsyncWarpSpecialized, KernelScheduleType>            ||
                                     cute::is_base_of_v<KernelCpAsyncWarpSpecializedPingpong, KernelScheduleType>    ||
                                     cute::is_base_of_v<KernelCpAsyncWarpSpecializedCooperative, KernelScheduleType>;
  constexpr bool IsWarpSpecializedTransposeB = !IsInputSizeTwoBytes && IsLayoutAmnBmn && IsWarpSpecialized;
  return IsWarpSpecializedTransposeB;
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_TMA_WS_SS
template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecialized> ||
       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedPingpong> ||
       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperative> ||
       cute::is_same_v<KernelScheduleType, KernelPtrArrayTmaWarpSpecializedCooperative>) &&
       not detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif
  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
                "Should meet TMA alignment requirement\n");

  static constexpr bool IsArrayOfPointersGemm = (cute::is_same_v<KernelScheduleType, KernelPtrArrayTmaWarpSpecializedCooperative>);
  static constexpr bool IsFP8Input = detail::is_input_fp8<ElementA, ElementB>();
  static_assert(!IsFP8Input || (IsFP8Input && !IsArrayOfPointersGemm),
                "Kernel[Array/Group]TmaWarpSpecializedCooperative is only compatible with FP8 FastAccum version right now\n");

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_ss_tag_to_major_A<ElementAMma, GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_ss_tag_to_major_B<ElementBMma, GmemLayoutBTag>();

  using AtomLayoutMNK = cute::conditional_t<
      cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperative> || IsArrayOfPointersGemm,
      Layout<Shape<_2,_1,_1>>, Layout<Shape<_1,_1,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
      ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK, GmmaMajorA, GmmaMajorB>(), AtomLayoutMNK{}));

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA = decltype(detail::ss_smem_selector<
      GmmaMajorA, ElementAMma, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::ss_smem_selector<
      GmmaMajorB, ElementBMma, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<detail::sm90_smem_capacity_bytes,
      ElementAMma, ElementBMma, TileShape_MNK>(StageCountType{});
  using DispatchPolicy = cute::conditional_t<IsArrayOfPointersGemm,
      MainloopSm90ArrayTmaGmmaWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>,
      /* For FP8 use a separate mainloop compared to other datatypes */
      cute::conditional_t<IsFP8Input,
          MainloopSm90TmaGmmaWarpSpecializedFP8<PipelineStages, ClusterShape_MNK, KernelScheduleType>,
          MainloopSm90TmaGmmaWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>>>;

  using SmemCopyAtomA = void; 
  using SmemCopyAtomB = void; 

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_TMA_WS_RS
template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecialized> ||
       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedPingpong> ||
       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperative>) &&
      detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()> 
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
                "Should meet TMA alignment requirement\n");
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif
  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_rs_tag_to_major_A<GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_rs_tag_to_major_B<GmemLayoutBTag>();
  static constexpr bool SwapAB = detail::is_swapAB<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>();
  static constexpr bool IsWarpSpecializedTransposeB = detail::is_warpspecialized_transpose_B<
      ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag, KernelScheduleType>();

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  using AtomLayoutMNK = cute::conditional_t<cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperative>,
      Layout<Shape<_2,_1,_1>>, Layout<Shape<_1,_1,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::rs_op_selector<
      ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK, GMMA::Major::K, GMMA::Major::K>(), AtomLayoutMNK{}));

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA = decltype(detail::rs_smem_selector<GmmaMajorA, ElementAMma,
      decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{})), IsWarpSpecializedTransposeB>());
  using SmemLayoutAtomB = decltype(detail::rs_smem_selector<GmmaMajorB, ElementBMma,
      decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{})), IsWarpSpecializedTransposeB>());

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<detail::sm90_smem_capacity_bytes,
      ElementAMma, ElementBMma, TileShape_MNK>(StageCountType{});

  using DispatchPolicy = MainloopSm90TmaGmmaRmemAWarpSpecialized<
      PipelineStages, ClusterShape_MNK, KernelScheduleType>;

  using SmemCopyAtomA = cute::conditional_t<SwapAB, void, Copy_Atom<cute::DefaultCopy, ElementA>>;
  using SmemCopyAtomB = cute::conditional_t<SwapAB, Copy_Atom<cute::DefaultCopy, ElementB>, void>;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_TMA_WS_RS Mixed Scaled GEMM
template <
  class ElementPairA_,
  class GmemLayoutATag_,
  int AlignmentA,
  class ElementPairB_,
  class GmemLayoutBTag_,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementPairA_,
    GmemLayoutATag_,
    AlignmentA,
    ElementPairB_,
    GmemLayoutBTag_,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedMixedInput> ||
       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedPingpongMixedInput> ||
       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperativeMixedInput>)>
> {

private:
  using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementPairA_>;
  using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementPairB_>;
  using ZeroA = detail::deduce_mixed_width_dtype_t<2, ElementPairA_>;
  using ZeroB = detail::deduce_mixed_width_dtype_t<2, ElementPairB_>;
  static constexpr bool NeitherIsTuple = !cute::is_tuple<ElementPairA_>::value && !cute::is_tuple<ElementPairB_>::value;

public:
  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementPairA_>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementPairB_>;
  static_assert(cute::is_tuple<ElementPairA_>::value ^ cute::is_tuple<ElementPairB_>::value ||
               (NeitherIsTuple && (sizeof_bits<ElementA>::value != sizeof_bits<ElementB>::value)), 
    "Either A OR B must be a tuple or the widths of A and B must be different.");

  static constexpr bool IsANarrow = sizeof_bits<ElementA>::value < sizeof_bits<ElementB>::value;

  using GmemLayoutATag = GmemLayoutATag_;
  using GmemLayoutBTag = GmemLayoutBTag_;

  using ElementPairA = cute::conditional_t<IsANarrow && NeitherIsTuple, cute::tuple<ElementA>, ElementPairA_>;
  using ElementPairB = cute::conditional_t<!IsANarrow && NeitherIsTuple, cute::tuple<ElementB>, ElementPairB_>;

  static constexpr bool IsATransformed = cute::is_tuple<ElementPairA>::value;
  using ElementScale = cute::conditional_t<IsATransformed, ScaleA, ScaleB>;
  using ElementZero = cute::conditional_t<IsATransformed, ZeroA, ZeroB>;

  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
                "Should meet TMA alignment requirement\n");
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif
  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_rs_tag_to_major_A<GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_rs_tag_to_major_B<GmemLayoutBTag>();
  static constexpr bool IsWarpSpecializedTransposeB = detail::is_warpspecialized_transpose_B<
      ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag, KernelScheduleType>();
  static_assert(!IsWarpSpecializedTransposeB, "Mixed input GEMM does not support WS transpose B.");

  // If A is scaled, then we don't need to swap. Otherwise, we must ensure B goes to RF and we must swap the operands.
  static constexpr bool SwapAB = !IsATransformed;

  // When we relax the above assertion, we must handle setting the tile mma GmmaMajorB correctly.
  static constexpr cute::GMMA::Major TiledMmaGmmaMajorB = SwapAB ? GmmaMajorA : GmmaMajorB;

  using ElementMma = cute::conditional_t<IsATransformed, ElementB, ElementA>;
  using AtomLayoutMNK = cute::conditional_t<cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperativeMixedInput>,
      Layout<Shape<_2,_1,_1>>, Layout<Shape<_1,_1,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::rs_op_selector<
      ElementMma, ElementMma, ElementAccumulator, TileShape_MNK, GMMA::Major::K, TiledMmaGmmaMajorB>(), AtomLayoutMNK{}));

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA = decltype(detail::rs_smem_selector<GmmaMajorA, ElementA,
      decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{})), IsWarpSpecializedTransposeB>());
  using SmemLayoutAtomB = decltype(detail::rs_smem_selector<GmmaMajorB, ElementB,
      decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{})), IsWarpSpecializedTransposeB>());

  using RealElementA = cute::conditional_t<SwapAB, ElementB, ElementA>;
  using RealElementB = cute::conditional_t<SwapAB, ElementA, ElementB>;
  static constexpr int PipelineStages = detail::compute_stage_count_or_override_single_affine_transformed_input<detail::sm90_smem_capacity_bytes,
      RealElementA, RealElementB, ElementScale, ElementZero, TileShape_MNK>(StageCountType{});

  using SmemCopyAtomA = cute::conditional_t<SwapAB, void, Copy_Atom<cute::DefaultCopy, ElementA>>;
  using SmemCopyAtomB = cute::conditional_t<SwapAB, Copy_Atom<cute::DefaultCopy, ElementB>, void>;

  using DispatchPolicy = MainloopSm90TmaGmmaRmemAWarpSpecializedMixedInput<PipelineStages, ClusterShape_MNK, KernelScheduleType>;

  // We pack the scale data with the operand that will be optionally scaled and converted before MMA.
  using StrideA = TagToStrideA_t<GmemLayoutATag>;
  using StrideB = TagToStrideB_t<GmemLayoutBTag>;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementPairA,
      StrideA,
      ElementPairB,
      StrideB,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;

};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_TMA_WS_FP8_FAST_ACCUM_SS
template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedFP8FastAccum> ||
      cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedPingpongFP8FastAccum> ||
      cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperativeFP8FastAccum> ||
       cute::is_same_v<KernelScheduleType, KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum>>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
                "Not meet TMA alignment requirement yet\n");
  static_assert(detail::is_input_fp8<ElementA, ElementB>(),
                "Only FP8 datatypes are compatible with these kernel schedules\n");
  // Dispatch TN fp8 kernels only to TMA warp specialized FP8 builder
  static_assert(!detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>(),
                 "Not supported for fp8 non-TN warp specialized kernels yet\n");
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif

  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_ss_tag_to_major_A<ElementA, GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_ss_tag_to_major_B<ElementB, GmemLayoutBTag>();

  static constexpr bool IsArrayOfPointersGemm = (cute::is_same_v<KernelScheduleType, KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum>);
  using AtomLayoutMNK = cute::conditional_t<cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperativeFP8FastAccum> ||
                                            IsArrayOfPointersGemm,
      Layout<Shape<_2,_1,_1>>, Layout<Shape<_1,_1,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
      ElementA, ElementB, ElementAccumulator, TileShape_MNK, GmmaMajorA, GmmaMajorB>(), AtomLayoutMNK{}));

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA = decltype(detail::ss_smem_selector<
      GmmaMajorA, ElementA, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::ss_smem_selector<
      GmmaMajorB, ElementB, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<detail::sm90_smem_capacity_bytes,
      ElementA, ElementB, TileShape_MNK>(StageCountType{});
  using DispatchPolicy = cute::conditional_t<IsArrayOfPointersGemm,
      MainloopSm90ArrayTmaGmmaWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>,
      MainloopSm90TmaGmmaWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>>;

  using SmemCopyAtomA = void;
  using SmemCopyAtomB = void;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_TMA_SS
template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<cute::is_same_v<KernelScheduleType, KernelTma> &&
                     not detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
                "Should meet TMA alignment requirement\n");
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_ss_tag_to_major_A<ElementAMma, GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_ss_tag_to_major_B<ElementBMma, GmemLayoutBTag>();

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
      ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK, GmmaMajorA, GmmaMajorB>()));

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA = decltype(detail::ss_smem_selector<
      GmmaMajorA, ElementAMma, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::ss_smem_selector<
      GmmaMajorB, ElementBMma, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<detail::sm90_smem_capacity_bytes,
      ElementAMma, ElementBMma, TileShape_MNK>(StageCountType{});
  using DispatchPolicy = MainloopSm90TmaGmma<PipelineStages, ClusterShape_MNK>;

  using SmemCopyAtomA = void;
  using SmemCopyAtomB = void;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_CpAsync
template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct [[deprecated("Use one of KernelCpAsyncWarpSpecialized schedules instead")]]
CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      cute::is_same_v<KernelScheduleType, KernelMultistage>>
> {
  // Map to warp-specialized kernels for better performance
  using CollectiveOp = typename CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelCpAsyncWarpSpecialized
  >::CollectiveOp;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_CpAsync_WS_SS
template <
  class ElementA,
  class GmemLayoutATag,
  int   AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int   AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecialized> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedCooperative> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedPingpong>) &&
      not detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()
    >
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::cp_async_min_alignment_bytes>(),
                "Minimum alignment required for cp.async is 4B.");

  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_ss_tag_to_major_A<ElementA, GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_ss_tag_to_major_B<ElementB, GmemLayoutBTag>();

  using AtomLayoutMNK = cute::conditional_t<cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedCooperative>,
      Layout<Shape<cute::Int<(size<0>(TileShape_MNK{}) < 128) ? 1 : 2>,_1,_1>>, Layout<Shape<_1,_1,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
      ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK, GmmaMajorA, GmmaMajorB>(), AtomLayoutMNK{}));

  static constexpr int NumLoadWarpGroups = cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecialized> ? 2 : 1;

  using GmemTiledCopyA = decltype(detail::make_cp_async_gmem_tiled_copy<
      NumThreadsPerWarpGroup * NumLoadWarpGroups, ElementA, AlignmentA, TagToStrideA_t<GmemLayoutATag>,
      decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using GmemTiledCopyB = decltype(detail::make_cp_async_gmem_tiled_copy<
      NumThreadsPerWarpGroup * NumLoadWarpGroups, ElementB, AlignmentB, TagToStrideB_t<GmemLayoutBTag>,
      decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  using SmemLayoutAtomA = decltype(detail::ss_smem_selector<
      GmmaMajorA, ElementAMma, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::ss_smem_selector<
      GmmaMajorB, ElementBMma, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<
      detail::sm90_smem_capacity_bytes, ElementAMma, ElementBMma, TileShape_MNK>(StageCountType{});

  using DispatchPolicy = MainloopSm90CpAsyncGmmaWarpSpecialized<
      PipelineStages, ClusterShape_MNK, KernelScheduleType>;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      void,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      void,
      cute::identity
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_CpAsync_WS_RS
template <
  class ElementA,
  class GmemLayoutATag,
  int   AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int   AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecialized> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedCooperative> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedPingpong>) &&
      detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()
    >
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::cp_async_min_alignment_bytes>(),
                "Minimum alignment required for cp.async is 4B.");

  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_rs_tag_to_major_A<GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_rs_tag_to_major_B<GmemLayoutBTag>();
  static constexpr bool SwapAB = detail::is_swapAB<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>();
  static constexpr bool IsWarpSpecializedTransposeB = detail::is_warpspecialized_transpose_B<
      ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag, KernelScheduleType>();

  using AtomLayoutMNK = cute::conditional_t<cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedCooperative>,
      Layout<Shape<cute::Int<(size<0>(TileShape_MNK{}) < 128) ? 1 : 2>,_1,_1>>, Layout<Shape<_1,_1,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::rs_op_selector<
      ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK, GMMA::Major::K, GMMA::Major::K>(), AtomLayoutMNK{}));

  static constexpr int NumLoadWarpGroups = 1;

  using GmemTiledCopyA = decltype(detail::make_cp_async_gmem_tiled_copy<
      NumThreadsPerWarpGroup * NumLoadWarpGroups, ElementA, AlignmentA, TagToStrideA_t<GmemLayoutATag>,
      decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using GmemTiledCopyB = decltype(detail::make_cp_async_gmem_tiled_copy<
      NumThreadsPerWarpGroup * NumLoadWarpGroups, ElementB, AlignmentB, TagToStrideB_t<GmemLayoutBTag>,
      decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  using SmemLayoutAtomA = decltype(detail::rs_smem_selector<GmmaMajorA, ElementAMma, 
      decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{})), IsWarpSpecializedTransposeB>());
  using SmemLayoutAtomB = decltype(detail::rs_smem_selector<GmmaMajorB, ElementBMma,
      decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{})), IsWarpSpecializedTransposeB>());

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<
      detail::sm90_smem_capacity_bytes, ElementAMma, ElementBMma, TileShape_MNK>(StageCountType{});

  using DispatchPolicy = MainloopSm90CpAsyncGmmaRmemAWarpSpecialized<
      PipelineStages, ClusterShape_MNK, KernelScheduleType>;

  using SmemCopyAtomA = cute::conditional_t<SwapAB, void, Copy_Atom<cute::DefaultCopy, ElementA>>;
  using SmemCopyAtomB = cute::conditional_t<SwapAB, Copy_Atom<cute::DefaultCopy, ElementB>, void>;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA auto kernel schedule
template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<cute::is_same_v<KernelScheduleType, KernelScheduleAuto>>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif

using ExtractedElementA = detail::deduce_mixed_width_dtype_t<0, ElementA>;
using ExtractedElementB = detail::deduce_mixed_width_dtype_t<0, ElementB>;

static constexpr bool IsTmaCompatible = detail::is_aligned<
    ExtractedElementA, AlignmentA, ExtractedElementB, AlignmentB, detail::tma_alignment_bytes>();

// Users opt into scales via the builder by passing a tuple of Elements for the input that will be scaled. We detect
// scale support if ONLY one of the inputs have tuples to describe them.
static constexpr bool OnlyOneIsTuple = cute::is_tuple<ElementA>::value ^ cute::is_tuple<ElementB>::value;
static constexpr bool IsDifferentWidth = sizeof_bits<ExtractedElementA>::value != sizeof_bits<ExtractedElementB>::value;
static constexpr bool IsMixedWidthInput = IsDifferentWidth || (IsDifferentWidth && OnlyOneIsTuple);

#if ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 1)))
  // Persistent schedules perform best for CUDA Toolkits with version >= 12.1
  // KernelTmaWarpSpecializedCooperative requires TileShape_M to be at least 128
  using KernelTmaWarpSpecializedScheduleSameInput = cute::conditional_t<size<0>(TileShape_MNK{}) == Int<64>{},
      KernelTmaWarpSpecializedPingpong, KernelTmaWarpSpecializedCooperative>;

  using KernelTmaWarpSpecializedScheduleMixedInput = cute::conditional_t<size<0>(TileShape_MNK{}) == Int<64>{},
      KernelTmaWarpSpecializedPingpongMixedInput, KernelTmaWarpSpecializedCooperativeMixedInput>;

  using KernelTmaWarpSpecializedSchedule = cute::conditional_t<IsMixedWidthInput, KernelTmaWarpSpecializedScheduleMixedInput, KernelTmaWarpSpecializedScheduleSameInput>;
#else
  using KernelTmaWarpSpecializedSchedule = cute::conditional_t<IsMixedWidthInput, KernelTmaWarpSpecializedMixedInput, KernelTmaWarpSpecialized>;
#endif

  // Non-persistent schedule is a safer choice for CpAsync kernels due to register pressure
  using KernelCpAsyncWarpSpecializedSchedule = KernelCpAsyncWarpSpecialized;
  using KernelSchedule = cute::conditional_t<IsTmaCompatible, KernelTmaWarpSpecializedSchedule, KernelCpAsyncWarpSpecializedSchedule>;
  static_assert((cute::is_same_v<KernelSchedule, KernelTmaWarpSpecializedSchedule> && IsMixedWidthInput) || !IsMixedWidthInput, "Only TMA warp specialized kernels are supported for mixed width input.");
  using CollectiveOp = typename CollectiveBuilder<
      arch::Sm90,
      arch::OpClassTensorOp,
      ElementA,
      GmemLayoutATag,
      AlignmentA,
      ElementB,
      GmemLayoutBTag,
      AlignmentB,
      ElementAccumulator,
      TileShape_MNK,
      ClusterShape_MNK,
      StageCountType,
      KernelSchedule
    >::CollectiveOp;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
