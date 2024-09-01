/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cute/config.hpp>

#include <cute/atom/copy_atom.hpp>

#include <cute/algorithm/copy.hpp>

#include <cute/tensor.hpp>
#include <cute/tensor_predicate.hpp>

namespace cute
{

// cooperative_copy<NumThreads, MaxVecBits>(thr_idx, src, dst)
// Use NumThreads to copy src to dst with element vectorization up to MaxVecBits.
// @pre 0 <= @a tid < NumThreads
// @pre Tensors @a src and @a dst are aligned up to MaxVecBits.
//
template <uint32_t NumThreads, uint32_t MaxVecBits,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
cooperative_copy(uint32_t                     const& tid,
                 Tensor<SrcEngine, SrcLayout> const& src,
                 Tensor<DstEngine, DstLayout>      & dst)
{
  // Assumes the shapes are static, can generalize
  CUTE_STATIC_ASSERT_V(size(src) == size(dst));
  // Assumes the types are the same, can generalize
  static_assert(sizeof_bits_v<typename SrcEngine::value_type> == sizeof_bits_v<typename DstEngine::value_type>);
  static_assert(MaxVecBits == sizeof_bits_v<typename SrcEngine::value_type> ||
                MaxVecBits == 8 || MaxVecBits == 16 || MaxVecBits == 32 || MaxVecBits == 64 || MaxVecBits == 128,
                "Expected MaxVecBits to be value size or 8 or 16 or 32 or 64 or 128 for alignment and performance.");
  // Check that the tensors are likely shared across threads: either gmem or smem
  static_assert((is_gmem<SrcEngine>::value || is_smem<SrcEngine>::value),
                 "cooperative_copy expects shared gmem or smem source tensor.");
  static_assert((is_gmem<DstEngine>::value || is_smem<DstEngine>::value),
                 "cooperative_copy expects shared gmem or smem destination tensor.");

  // Precondition on tid in DEBUG
  assert(tid < NumThreads);

  // Fallback - slow path, naive copy, vectorization disabled
  if constexpr(size(SrcLayout{}) % NumThreads != 0) {
    int index = static_cast<int>(tid);
    CUTE_UNROLL
    for(int i = 0; i < ceil_div(size(SrcLayout{}), NumThreads); i++) {
      if(index < size(SrcLayout{})) {
        dst[index] = src[index];
      }
      index += NumThreads;
    }
  } else {
    // Fast path with vectorization

    // Precondition on pointer alignment in DEBUG
    assert(is_byte_aligned<max(MaxVecBits/8, 1u)>(raw_pointer_cast(src.data())));
    assert(is_byte_aligned<max(MaxVecBits/8, 1u)>(raw_pointer_cast(dst.data())));
    constexpr int elem_bits  = sizeof_bits_v<typename SrcEngine::value_type>;

    //
    // Determine val+thr vectorization based on src/dst size and number of threads
    // NOTE: This heuristic promotes parallelization over vectorization
    //

    // The number of elements that can be vectorized in values
    constexpr int common_elem = decltype(max_common_vector(src, dst))::value;
    constexpr int common_bits = common_elem * elem_bits;
    constexpr int total_elem = decltype(size(src))::value;
    constexpr int total_bits = total_elem * elem_bits;
    static_assert(total_bits % NumThreads == 0);
    constexpr int total_bits_per_thr = total_bits / NumThreads;
    // If there are too many threads to allow a full elem copy, trunc the thrs and use elem_bits
    constexpr int max_vec_bits_by_thr = cute::max(elem_bits, total_bits_per_thr);

    // Cap the vectorization to the common bits, the max_vec_bits_by_thr, and the MaxVecBits
    constexpr int vec_bits = cute::min(common_bits, max_vec_bits_by_thr, static_cast<int>(MaxVecBits));
    // Convert back to number of elements, safe_div
    static_assert((vec_bits % elem_bits) == 0);
    constexpr int vec_elem = vec_bits / elem_bits;

    // Use only part of threads if there's not enough work for all threads
    constexpr int vec_thrs = (total_elem % (vec_elem * NumThreads) == 0)
                            ? NumThreads
                            : (total_elem / vec_elem);
    static_assert(vec_thrs <= NumThreads);

    // The common layout of the two tensors that can be vectorized over threads
    // vidx -> coord
    auto common_layout = max_common_layout(get_nonswizzle_portion(src.layout()),
                                          get_nonswizzle_portion(dst.layout()));

    // Scale up the common_layout to cover the entire tensors
    // vidx -> coord
    auto full_perm = tile_to_shape(make_layout(common_layout), size(src));

    // Create the Tiler
    // ((vid,tid),iter)
    auto layout_vt = logical_divide(full_perm, Layout<Shape<Int<vec_elem>, Int<vec_thrs>>>{});

    // Apply and slice
    Tensor src_v = src.compose(layout_vt)(make_coord(_,tid),_);
    Tensor dst_v = dst.compose(layout_vt)(make_coord(_,tid),_);

    // Should account for vec_bits < 8 and/or vec_elem <= 1
    // And also account for subbyte types, which could cause race conditions
    // Want to ENFORCE sufficient vectorization in those cases
    static_assert((vec_bits >= 8), "No support for subbyte copying");
    using VecType = uint_bit_t<vec_bits>;

#if 0
    if (thread0()) {
      print("   "); print("cooperative_copy -- vec\n");
      print("   "); print("NumThreads: "); print(NumThreads); print("\n");
      print("   "); print("MaxVecBits: "); print(MaxVecBits); print("\n");
      print("   "); print("src: "); print(src); print("\n");
      print("   "); print("dst: "); print(dst); print("\n");
      print("   "); print("common_layout: "); print(common_layout); print("\n");
      print("   "); print("full_perm: "); print(full_perm); print("\n");
      print("   "); print("Used vector: ");  print(vec_elem); print("\n");
      print("   "); print("Used threads: "); print(vec_thrs); print("\n");
      print("   "); print("layout_vt: "); print(layout_vt); print("\n");
      print("   "); print("src.compose(layout_vt): "); print(src.compose(layout_vt)); print("\n");
      print("   "); print("dst.compose(layout_vt): "); print(dst.compose(layout_vt)); print("\n");
      print("   "); print("src_v: "); print(src_v); print("\n");
      print("   "); print("dst_v: "); print(dst_v); print("\n");
      print("   "); print("recast<VecType const>(src_v): "); print(recast<VecType const>(src_v)); print("\n");
      print("   "); print("recast<VecType const>(dst_v): "); print(recast<VecType const>(dst_v)); print("\n");
    }
#ifdef __CUDA_ARCH__
    __syncthreads();
#endif
#endif

    // If we're using all threads (static) or the tid is in in-range (dynamic)
    if (vec_thrs >= NumThreads or tid < vec_thrs) {
      return copy_if(TrivialPredTensor{}, recast<VecType const>(src_v), recast<VecType>(dst_v));
    }
  }
}

template <uint32_t NumThreads,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
cooperative_copy(uint32_t                     const& tid,
                 Tensor<SrcEngine, SrcLayout> const& src,
                 Tensor<DstEngine, DstLayout>      & dst)
{
  constexpr uint32_t MaxVecBits = sizeof_bits_v<typename SrcEngine::value_type>;
  return cooperative_copy<NumThreads, MaxVecBits>(tid, src, dst);
}

// Accept mutable temporaries
template <uint32_t NumThreads,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
cooperative_copy(uint32_t                     const& tid,
                 Tensor<SrcEngine, SrcLayout> const& src,
                 Tensor<DstEngine, DstLayout>     && dst)
{
  return cooperative_copy<NumThreads>(tid, src, dst);
}

// Accept mutable temporaries
template <uint32_t NumThreads,
          uint32_t MaxVecBits,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
cooperative_copy(uint32_t                     const& tid,
                 Tensor<SrcEngine, SrcLayout> const& src,
                 Tensor<DstEngine, DstLayout>     && dst)
{
  return cooperative_copy<NumThreads, MaxVecBits>(tid, src, dst);
}

} // end namespace cute
