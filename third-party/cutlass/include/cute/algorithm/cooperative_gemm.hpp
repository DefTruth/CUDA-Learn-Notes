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
#include <cute/util/type_traits.hpp>

#include <cute/atom/mma_atom.hpp>

#include <cute/algorithm/axpby.hpp>
#include <cute/algorithm/functional.hpp>
#include <cute/algorithm/gemm.hpp>

#include <cute/tensor.hpp>

namespace cute
{

//
// Cooperative Shared-Memory GEMMs
//

namespace detail {

// Predicated Cooperative GEMM
template <class... Args,
          class Alpha, class TA, class ALayout, class TB, class BLayout,
          class Beta,  class TC, class CLayout,
          class ALoadTransformOp, class BLoadTransformOp,
          class CLoadTransformOp, class CStoreTransformOp,
          __CUTE_REQUIRES(ALayout::rank == 2 && is_smem<TA>::value &&
                          BLayout::rank == 2 && is_smem<TB>::value &&
                          CLayout::rank == 2 && is_smem<TC>::value)>
CUTE_HOST_DEVICE
void
cooperative_gemm_predication(ThrMMA<Args...> const& thr_mma,
                             Alpha const& alpha,
                             Tensor<TA, ALayout> sA,
                             Tensor<TB, BLayout> sB,
                             Beta  const& beta,
                             Tensor<TC, CLayout> sC,
                             ALoadTransformOp  const& sA_load_op,  // transforms A values before use in GEMM
                             BLoadTransformOp  const& sB_load_op,  // transforms B values before use in GEMM
                             CLoadTransformOp  const& sC_load_op,  // transforms C values before use in GEMM
                             CStoreTransformOp const& sC_store_op) // transforms results before they are stored to C
{
  using TypeA = typename TA::value_type;
  using TypeB = typename TB::value_type;
  using TypeC = typename TC::value_type;

  // Original, static size of the problem
  auto M = size<0>(sC);
  auto N = size<1>(sC);
  auto K = size<1>(sA);

  // Block size of the compute tile
  auto BLK_M = tile_size<0>(thr_mma);
  auto BLK_N = tile_size<1>(thr_mma);
  auto BLK_K = tile_size<2>(thr_mma);

  //
  // MMA Partitioning
  //

  // Round the layout extents up to BLK_X to satisfy MMA partitioning safety
  Tensor rounded_sA = sA.compose(make_shape(round_up(M, BLK_M), round_up(K, BLK_K)));
  Tensor rounded_sB = sB.compose(make_shape(round_up(N, BLK_N), round_up(K, BLK_K)));
  Tensor rounded_sC = sC.compose(make_shape(round_up(M, BLK_M), round_up(N, BLK_N)));

  // Partition the sA and sB tiles across the threads for the MMA
  Tensor tCsA = thr_mma.partition_A(rounded_sA);                    // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(rounded_sB);                    // (MMA,MMA_N,MMA_K)
  Tensor tCsC = thr_mma.partition_C(rounded_sC);                    // (MMA,MMA_M,MMA_N)
  // Create register tensors for the MMA to operate on
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);                      // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);                      // (MMA,MMA_N,MMA_K)
  Tensor tCrC = thr_mma.make_fragment_C(tCsC);                      // (MMA,MMA_M,MMA_N)

#if 0
  if (thread0()) {
    print("  sA: "); print(  sA); print("\n");
    print("  sB: "); print(  sB); print("\n");
    print("  sC: "); print(  sC); print("\n");
    print("r_sA: "); print(rounded_sA); print("\n");
    print("r_sB: "); print(rounded_sB); print("\n");
    print("r_sC: "); print(rounded_sC); print("\n");
    print(thr_mma);
    print("tCsA: "); print(tCsA); print("\n");
    print("tCsB: "); print(tCsB); print("\n");
    print("tCsC: "); print(tCsC); print("\n");
    print("tCrA: "); print(tCrA); print("\n");
    print("tCrB: "); print(tCrB); print("\n");
    print("tCrC: "); print(tCrC); print("\n");
  }
#endif

  //
  // PREDICATION
  //

  // Create coordinate tensors for the problem
  Tensor cA = make_identity_tensor(shape(rounded_sA));              // (M,K) -> (m,k)
  Tensor cB = make_identity_tensor(shape(rounded_sB));              // (N,K) -> (n,k)

  // Repeat partitioning with thr_mma
  Tensor tCcA = thr_mma.partition_A(cA);                            // (MMA,MMA_M,MMA_K) -> (m,k)
  Tensor tCcB = thr_mma.partition_B(cB);                            // (MMA,MMA_N,MMA_K) -> (n,k)

  // Allocate the preds for MMA- and MMA_MN-modes
  Tensor tCpA = make_tensor<bool>(make_shape(size<0>(tCsA), size<1>(tCsA)));
  Tensor tCpB = make_tensor<bool>(make_shape(size<0>(tCsB), size<1>(tCsB)));

  // Populate the predicates on M and N
  CUTE_UNROLL
  for (int i = 0; i < size(tCpA); ++i) {
    tCpA(i) = elem_less(get<0>(tCcA(_,_,Int<0>{})(i)), shape<0>(sA));
  }
  CUTE_UNROLL
  for (int i = 0; i < size(tCpB); ++i) {
    tCpB(i) = elem_less(get<0>(tCcB(_,_,Int<0>{})(i)), shape<0>(sB));
  }

#if 0
  if (thread0()) {
    print("  cA: "); print(  cA); print("\n");
    print("  cB: "); print(  cB); print("\n");
    print("tCcA: "); print(tCcA); print("\n");
    print("tCcB: "); print(tCcB); print("\n");
    print_tensor(tCpA);
    print_tensor(tCpB);
  }
#endif

  //
  // PREFETCH k_block = 0
  //   Condition the k-predication on (static) k_block == K_BLOCK_MAX-1, the last k_block
  //   Assumes the MMA-tiling in K is trivial
  //

  constexpr int K_BLOCK_MAX = size<2>(tCrA);

  CUTE_UNROLL
  for (int m = 0; m < size<1>(tCrA); ++m) {     // Copy MMA_M
    CUTE_UNROLL
    for (int i = 0; i < size<0>(tCrA); ++i) {   // Copy MMA_I
      tCrA(i,m,0) = (tCpA(i,m) && (0 < K_BLOCK_MAX-1 || elem_less(get<1>(tCcA(i,m,0)), shape<1>(sA)))) ? sA_load_op(tCsA(i,m,0)) : TypeA{};
    }
  }
  CUTE_UNROLL
  for (int n = 0; n < size<1>(tCrB); ++n) {     // Copy MMA_N
    CUTE_UNROLL
    for (int i = 0; i < size<0>(tCrB); ++i) {   // Copy MMA_I
      tCrB(i,n,0) = (tCpB(i,n) && (0 < K_BLOCK_MAX-1 || elem_less(get<1>(tCcB(i,n,0)), shape<1>(sB)))) ? sB_load_op(tCsB(i,n,0)) : TypeB{};
    }
  }
  //
  // MAINLOOP
  //

  // Clear accumulators
  clear(tCrC);

  CUTE_UNROLL
  for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
  {
    if (k_block < K_BLOCK_MAX-1)   // static-if not the last k_block
    {
      int k_next = k_block + 1;    // Load k_next block

      //   Condition the k-predication on (static) k_block == K_BLOCK_MAX-1, the last k_block
      //   Assumes the MMA-tiling in K is trivial

      CUTE_UNROLL
      for (int m = 0; m < size<1>(tCrA); ++m) {       // Copy MMA_M
        CUTE_UNROLL
        for (int i = 0; i < size<0>(tCrA); ++i) {     // Copy MMA_I
          tCrA(i,m,k_next) = (tCpA(i,m) && (k_next < K_BLOCK_MAX-1 || elem_less(get<1>(tCcA(i,m,k_next)), shape<1>(sA)))) ? sA_load_op(tCsA(i,m,k_next)) : TypeA{};
        }
      }
      CUTE_UNROLL
      for (int n = 0; n < size<1>(tCrB); ++n) {       // Copy MMA_N
        CUTE_UNROLL
        for (int i = 0; i < size<0>(tCrB); ++i) {     // Copy MMA_I
          tCrB(i,n,k_next) = (tCpB(i,n) && (k_next < K_BLOCK_MAX-1 || elem_less(get<1>(tCcB(i,n,k_next)), shape<1>(sB)))) ? sB_load_op(tCsB(i,n,k_next)) : TypeB{};
        }
      }
    }
    // GEMM on k_block in registers
    gemm(thr_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
  }

  //
  // Epilogue
  //

  // Create coordinate tensors for the problem
  Tensor cC   = make_identity_tensor(shape(rounded_sC));             // (M,N) -> (m,n)
  // Repeat partitioning with thr_mma
  Tensor tCcC = thr_mma.partition_C(cC);                             // (MMA,MMA_M,MMA_N) -> (m,n)

  const bool isBetaZero = (beta == Beta{});

  // Custom axpby_if for now
  CUTE_UNROLL
  for (int i = 0; i < size(tCrC); ++i)
  {
    if (elem_less(tCcC(i), shape(sC)))
    {
      tCsC(i) = sC_store_op(isBetaZero ? alpha * static_cast<TypeC>(tCrC(i))
                                       : alpha * static_cast<TypeC>(tCrC(i)) +
                                          beta * static_cast<TypeC>(sC_load_op(tCsC(i))));
    }
  }
}

// Slow fallback path
template <class... Args,
          class Alpha, class TA, class ALayout, class TB, class BLayout,
          class Beta,  class TC, class CLayout,
          class ALoadTransformOp, class BLoadTransformOp,
          class CLoadTransformOp, class CStoreTransformOp,
          __CUTE_REQUIRES(ALayout::rank == 2 && is_smem<TA>::value &&
                          BLayout::rank == 2 && is_smem<TB>::value &&
                          CLayout::rank == 2 && is_smem<TC>::value)>
CUTE_HOST_DEVICE
void
cooperative_gemm_predication(uint32_t thread_idx,
                             TiledMMA<Args...> const& tiled_mma,
                             Alpha const& alpha,
                             Tensor<TA, ALayout> sA,
                             Tensor<TB, BLayout> sB,
                             Beta  const& beta,
                             Tensor<TC, CLayout> sC,
                             ALoadTransformOp  const& sA_load_op,  // transforms A values before use in GEMM
                             BLoadTransformOp  const& sB_load_op,  // transforms B values before use in GEMM
                             CLoadTransformOp  const& sC_load_op,  // transforms C values before use in GEMM
                             CStoreTransformOp const& sC_store_op) // transforms results before they are stored to C
{
  // ThrMMA
  auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
  cooperative_gemm_predication(thr_mma, alpha, sA, sB, beta, sC, sA_load_op, sB_load_op, sC_load_op, sC_store_op);
}

// Unpredicated Cooperative GEMM
template <class SmemCopyOpA, class SmemCopyOpB, class SmemCopyOpC,
          class... Args,
          class Alpha, class TA, class ALayout, class TB, class BLayout,
          class Beta,  class TC, class CLayout,
          class ALoadTransformOp, class BLoadTransformOp,
          class CLoadTransformOp, class CStoreTransformOp,
          __CUTE_REQUIRES(ALayout::rank == 2 && is_smem<TA>::value &&
                          BLayout::rank == 2 && is_smem<TB>::value &&
                          CLayout::rank == 2 && is_smem<TC>::value)>
CUTE_HOST_DEVICE
void
cooperative_gemm_no_predication(uint32_t thread_idx,
                                TiledMMA<Args...> const& tiled_mma,
                                Alpha const& alpha,
                                Tensor<TA, ALayout> sA,
                                Tensor<TB, BLayout> sB,
                                Beta  const& beta,
                                Tensor<TC, CLayout> sC,
                                ALoadTransformOp  const& sA_load_op,  // transforms A values before use in GEMM
                                BLoadTransformOp  const& sB_load_op,  // transforms B values before use in GEMM
                                CLoadTransformOp  const& sC_load_op,  // transforms C values before use in GEMM
                                CStoreTransformOp const& sC_store_op) // transforms results before they are stored to C
{
  using TypeA = typename TA::value_type;
  using TypeB = typename TB::value_type;
  using TypeC = typename TC::value_type;

  // ThrMMA
  auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

  //
  // MMA Partitioning
  //

  Tensor tCsC = thr_mma.partition_C(sC);
  // Create register tensors for the MMA to operate on
  Tensor tCrA  = thr_mma.partition_fragment_A(sA);                    // (MMA,MMA_M,MMA_K)
  Tensor tCrB  = thr_mma.partition_fragment_B(sB);                    // (MMA,MMA_N,MMA_K)
  Tensor tCrC  = thr_mma.make_fragment_C(tCsC);                       // (MMA,MMA_M,MMA_N)

  using CopyOpAType = SmemCopyOpA;
  using CopyOpBType = SmemCopyOpB;

  auto smem_tiled_copy_A = make_tiled_copy_A(Copy_Atom<CopyOpAType, TypeA>{}, thr_mma);
  auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
  Tensor tCsA            = smem_thr_copy_A.partition_S(sA);
  Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));             // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));             // CPY_K

  auto smem_tiled_copy_B = make_tiled_copy_B(Copy_Atom<CopyOpBType, TypeB>{}, thr_mma);
  auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
  Tensor tCsB            = smem_thr_copy_B.partition_S(sB);
  Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

#if 0
  if (thread0()) {
    print("  sA: "); print(sA); print("\n");
    print("  sB: "); print(sB); print("\n");
    print("  sC: "); print(sC); print("\n");
    print(thr_mma); print("\n");
    print("tCsC: "); print(tCsC); print("\n");
    print("tCrA: "); print(tCrA); print("\n");
    print("tCrB: "); print(tCrB); print("\n");
    print("tCrC: "); print(tCrC); print("\n");
    print(smem_thr_copy_A); print("\n");
    print("tCsA: "); print(tCsA); print("\n");
    print("tCrA_copy_view: "); print(tCrA_copy_view); print("\n");
    print(smem_thr_copy_B); print("\n");
    print("tCsB: "); print(tCsB); print("\n");
    print("tCrB_copy_view: "); print(tCrB_copy_view); print("\n");
  }
#endif

  //
  // PREFETCH
  //

  copy(smem_tiled_copy_A, tCsA(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
  copy(smem_tiled_copy_B, tCsB(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
  //
  // MAINLOOP
  //

  // Clear accumulators
  clear(tCrC);

  constexpr int K_BLOCK_MAX = size<2>(tCrA);

  CUTE_UNROLL
  for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
  {
    // static-if load the next k_block. No k-predication required on these loads.
    if (k_block < K_BLOCK_MAX-1)
    {
      // Load the next k_block
      int k_next = k_block + 1;       // statically unrolled
      copy(smem_tiled_copy_A, tCsA(_,_,k_next), tCrA_copy_view(_,_,k_next));
      copy(smem_tiled_copy_B, tCsB(_,_,k_next), tCrB_copy_view(_,_,k_next));
    }

    // Transform A and B, relying on the compiler to remove in case of identity ops
    cute::transform(tCrA(_,_,k_block), sA_load_op);
    cute::transform(tCrB(_,_,k_block), sB_load_op);

    // GEMM on k_block in registers
    gemm(thr_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
  }

  //
  // Epilogue
  //

  auto isBetaZero = [&] () {
    if constexpr (is_complex<Beta>::value) {
      return beta.real() == Int<0>{} && beta.imag() == Int<0>{};
    }
    else {
      return beta == Int<0>{};
    }
    CUTE_GCC_UNREACHABLE;
  } ();

  using CopyOpCType = SmemCopyOpC;
  Tensor tCrD = thr_mma.make_fragment_C(tCsC);
  if(!isBetaZero) {
    copy(CopyOpCType{}, tCsC, tCrD);
    // Transform C on/after load
    cute::transform(tCrD, sC_load_op);
  }
  // C = alpha * (A * B) + beta * C
  axpby(alpha, tCrC, beta, tCrD);
  // Transform C before/on store
  cute::transform(tCrD, sC_store_op);
  copy(CopyOpCType{}, tCrD, tCsC);
}

} // end namespace detail

template <class SmemCopyOpA, class SmemCopyOpB, class SmemCopyOpC,
          class... Args,
          class Alpha, class TA, class ALayout, class TB, class BLayout,
          class Beta,  class TC, class CLayout,
          class ALoadTransformOp = cute::identity, class BLoadTransformOp  = cute::identity,
          class CLoadTransformOp = cute::identity, class CStoreTransformOp = cute::identity,
          __CUTE_REQUIRES(ALayout::rank == 2 && is_smem<TA>::value &&
                          BLayout::rank == 2 && is_smem<TB>::value &&
                          CLayout::rank == 2 && is_smem<TC>::value)>
CUTE_HOST_DEVICE
void
cooperative_gemm(uint32_t thread_idx,
                 TiledMMA<Args...> const& tiled_mma,
                 Alpha const& alpha,
                 Tensor<TA, ALayout> sA,
                 Tensor<TB, BLayout> sB,
                 Beta  const& beta,
                 Tensor<TC, CLayout> sC,
                 ALoadTransformOp  const& sA_load_op  = {}, // transforms A values before use in GEMM
                 BLoadTransformOp  const& sB_load_op  = {}, // transforms B values before use in GEMM
                 CLoadTransformOp  const& sC_load_op  = {}, // transforms C values before use in GEMM
                 CStoreTransformOp const& sC_store_op = {}) // transforms results before they are stored to C
{
  CUTE_STATIC_ASSERT_V(size<0>(sA) == size<0>(sC));  // AM == CM
  CUTE_STATIC_ASSERT_V(size<0>(sB) == size<1>(sC));  // BN == CN
  CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));  // AK == BK

  using TypeA = typename TA::value_type;
  using TypeB = typename TB::value_type;
  using TypeC = typename TC::value_type;

  static_assert(is_convertible_v<decay_t<invoke_result_t<ALoadTransformOp, TypeA>>, TypeA>,
    "ALoadTransformOp functor must accept value of type TA::value_type and return value convertible to type TA::value_type");
  static_assert(is_convertible_v<decay_t<invoke_result_t<BLoadTransformOp, TypeB>>, TypeB>,
    "BLoadTransformOp functor must accept value of type TB::value_type and return value convertible to type TB::value_type");
  static_assert(is_convertible_v<decay_t<invoke_result_t<CLoadTransformOp, TypeC>>, TypeC>,
    "CLoadTransformOp functor must accept value of type TC::value_type and return value convertible to type TC::value_type");
  static_assert(is_convertible_v<decay_t<invoke_result_t<CStoreTransformOp, TypeC>>, TypeC>,
    "CStoreTransformOp functor must accept value of type TC::value_type and return value convertible to type TC::value_type");

  static constexpr bool compat = weakly_compatible(tile_shape(TiledMMA<Args...>{}),
                                                   make_shape(size<0>(sA), size<0>(sB), size<1>(sA)));
  if constexpr (compat) {
    detail::cooperative_gemm_no_predication<SmemCopyOpA, SmemCopyOpB, SmemCopyOpC>(
        thread_idx, tiled_mma, alpha, sA, sB, beta, sC,
        sA_load_op, sB_load_op, sC_load_op, sC_store_op
    );
  } else {
    detail::cooperative_gemm_predication(
      thread_idx, tiled_mma, alpha, sA, sB, beta, sC,
      sA_load_op, sB_load_op, sC_load_op, sC_store_op
    );
  }
}

template <class... Args,
          class Alpha, class TA, class ALayout, class TB, class BLayout,
          class Beta,  class TC, class CLayout,
          class ALoadTransformOp = cute::identity, class BLoadTransformOp  = cute::identity,
          class CLoadTransformOp = cute::identity, class CStoreTransformOp = cute::identity,
          __CUTE_REQUIRES(ALayout::rank == 2 && is_smem<TA>::value &&
                          BLayout::rank == 2 && is_smem<TB>::value &&
                          CLayout::rank == 2 && is_smem<TC>::value)>
CUTE_HOST_DEVICE
void
cooperative_gemm(uint32_t thread_idx,
                 TiledMMA<Args...> const& tiled_mma,
                 Alpha const& alpha,
                 Tensor<TA, ALayout> sA,
                 Tensor<TB, BLayout> sB,
                 Beta  const& beta,
                 Tensor<TC, CLayout> sC,
                 ALoadTransformOp  const& sA_load_op  = {}, // transforms A values before use in GEMM
                 BLoadTransformOp  const& sB_load_op  = {}, // transforms B values before use in GEMM
                 CLoadTransformOp  const& sC_load_op  = {}, // transforms C values before use in GEMM
                 CStoreTransformOp const& sC_store_op = {}) // transforms results before they are stored to C
{
  using CopyOpA = AutoVectorizingCopyWithAssumedAlignment<sizeof_bits_v<typename TA::value_type>>;
  using CopyOpB = AutoVectorizingCopyWithAssumedAlignment<sizeof_bits_v<typename TB::value_type>>;
  using CopyOpC = AutoVectorizingCopyWithAssumedAlignment<sizeof_bits_v<typename TC::value_type>>;
  cooperative_gemm<CopyOpA, CopyOpB, CopyOpC>(
      thread_idx, tiled_mma, alpha, sA, sB, beta, sC,
      sA_load_op, sB_load_op, sC_load_op, sC_store_op
  );
}

// Legacy overload of cute::gemm for backwards-compatibility
template <class... Args,
          class Alpha, class TA, class ALayout, class TB, class BLayout,
          class Beta,  class TC, class CLayout,
          class ALoadTransformOp = cute::identity, class BLoadTransformOp  = cute::identity,
          class CLoadTransformOp = cute::identity, class CStoreTransformOp = cute::identity,
          __CUTE_REQUIRES(ALayout::rank == 2 && is_smem<TA>::value &&
                          BLayout::rank == 2 && is_smem<TB>::value &&
                          CLayout::rank == 2 && is_smem<TC>::value)>
CUTE_HOST_DEVICE
void
gemm(ThrMMA<Args...> const& thr_mma,
     Alpha const& alpha,
     Tensor<TA, ALayout> sA,
     Tensor<TB, BLayout> sB,
     Beta  const& beta,
     Tensor<TC, CLayout> sC,
     ALoadTransformOp  const& sA_load_op  = {}, // transforms A values before use in GEMM
     BLoadTransformOp  const& sB_load_op  = {}, // transforms B values before use in GEMM
     CLoadTransformOp  const& sC_load_op  = {}, // transforms C values before use in GEMM
     CStoreTransformOp const& sC_store_op = {}) // transforms results before they are stored to C
{
  // Goes directly to the slow path to avoid getting thread_idx from thr_mma
  detail::cooperative_gemm_predication(
    thr_mma, alpha, sA, sB, beta, sC,
    sA_load_op, sB_load_op, sC_load_op, sC_store_op
  );
}

} // end namespace cute
