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

#include "cutlass_unit_test.h"

#include <cute/tensor.hpp>

#include "../cooperative_gemm_common.hpp"

using namespace cute;

TEST(SM70_CuTe_Volta, CooperativeGemm1_FloatFMA) {
  using value_type = float;

  constexpr uint32_t m = 64;
  constexpr uint32_t n = 32;
  constexpr uint32_t k = 16;

  constexpr uint32_t thread_block_size = 128;

  using tiled_mma_t =
      TiledMMA<
        MMA_Atom<UniversalFMA<value_type, value_type, value_type, value_type>>,
        Layout<Shape<_16, _8, _1>>
      >;

  test_cooperative_gemm_col_major_layout<m, n, k, thread_block_size, tiled_mma_t, value_type>();
}

TEST(SM70_CuTe_Volta, CooperativeGemm1_FloatFMA_Predication) {
  using value_type = float;

  constexpr uint32_t m = 88;
  constexpr uint32_t n = 20;
  constexpr uint32_t k = 12;

  constexpr uint32_t thread_block_size = 128;

  using tiled_mma_t =
      TiledMMA<
        MMA_Atom<UniversalFMA<value_type, value_type, value_type, value_type>>,
        Layout<Shape<_2, _64, _1>>
      >;

  test_cooperative_gemm_col_major_layout<m, n, k, thread_block_size, tiled_mma_t, value_type>();
}

TEST(SM70_CuTe_Volta, CooperativeGemm1_FloatFMA_Predication2) {
  using value_type = float;

  constexpr uint32_t m = 88;
  constexpr uint32_t n = 36;
  constexpr uint32_t k = 24;

  constexpr uint32_t thread_block_size = 128;

  using tiled_mma_t =
      TiledMMA<
        MMA_Atom<UniversalFMA<value_type, value_type, value_type, value_type>>,
        Layout<Shape<_4, _32, _1>>
      >;

  test_cooperative_gemm_col_major_layout<m, n, k, thread_block_size, tiled_mma_t, value_type>();
}

TEST(SM70_CuTe_Volta, CooperativeGemm1_FloatFMA_Predication3) {
  using value_type = float;

  constexpr uint32_t m = 67;
  constexpr uint32_t n = 13;
  constexpr uint32_t k = 11;

  constexpr uint32_t thread_block_size = 128;

  using tiled_mma_t =
      TiledMMA<
        MMA_Atom<UniversalFMA<value_type, value_type, value_type, value_type>>,
        Layout<Shape<_1, _128, _1>>
      >;

  test_cooperative_gemm_col_major_layout<m, n, k, thread_block_size, tiled_mma_t, value_type>();
}

TEST(SM70_CuTe_Volta, CooperativeGemm2_DoubleFMA) {
  using value_type = double;

  constexpr uint32_t m = 16;
  constexpr uint32_t n = 32;
  constexpr uint32_t k = 32;

  constexpr uint32_t thread_block_size = 128;

  using tiled_mma_t =
      TiledMMA<
        MMA_Atom<UniversalFMA<value_type, value_type, value_type, value_type>>,
        Layout<Shape<_16, _8, _1>>
      >;

  test_cooperative_gemm_col_major_layout<m, n, k, thread_block_size, tiled_mma_t, value_type>();
}

TEST(SM70_CuTe_Volta, CooperativeGemm3_Float_FMA_CustomPermutationMNK) {
  using value_type = float;

  constexpr uint32_t m = 32;
  constexpr uint32_t n = 32;
  constexpr uint32_t k = 32;

  constexpr uint32_t thread_block_size = 256;

  using tiled_mma_t = TiledMMA<
    MMA_Atom<
      UniversalFMA<value_type, value_type, value_type, value_type>
    >,
    Layout<
      Shape<_16, _16, _1>
    >,
    Tile<
      Layout<
        Shape<_16,_2>, Stride<_2,_1>
      >,               // 32x32x1 MMA with perm for load vectorization
      Layout<
        Shape<_16,_2>, Stride<_2,_1>
      >,
      Underscore
    >
  >;

  test_cooperative_gemm_col_major_layout<m, n, k, thread_block_size, tiled_mma_t, value_type>();
}

TEST(SM70_CuTe_Volta, CooperativeGemm4_Half_MMA) {
  using value_type = cutlass::half_t;

  constexpr uint32_t m = 32;
  constexpr uint32_t n = 32;
  constexpr uint32_t k = 32;

  constexpr uint32_t thread_block_size = 128;

  using tiled_mma_t = TiledMMA<
    MMA_Atom<SM70_8x8x4_F16F16F16F16_TN>,
    Layout<Shape<_4, _4, _1>>
  >;

  using smem_a_atom_layout_t = typename tiled_mma_t::AtomLayoutB_TV;
  using smem_b_atom_layout_t = typename tiled_mma_t::AtomLayoutA_TV;
  using smem_c_atom_layout_t = decltype(make_layout(make_shape(Int<m> {}, Int<n> {})));

  test_cooperative_gemm_col_major_layout<smem_a_atom_layout_t,
                                         smem_b_atom_layout_t,
                                         smem_c_atom_layout_t,
                                         m,
                                         n,
                                         k,
                                         thread_block_size,
                                         tiled_mma_t,
                                         value_type>();
}

TEST(SM70_CuTe_Volta, CooperativeGemm5_Half_MMA) {
  using value_type = cutlass::half_t;

  constexpr uint32_t m = 32;
  constexpr uint32_t n = 32;
  constexpr uint32_t k = 32;

  constexpr uint32_t thread_block_size = 128;

  using tiled_mma_t = TiledMMA<
    MMA_Atom<SM70_8x8x4_F16F16F16F16_TN>,
    Layout<Shape<_4, _4, _1>>
  >;

  using gmem_a_layout_t = decltype(make_layout(make_shape(Int<m>{}, Int<k>{})));
  using gmem_b_layout_t = decltype(make_layout(make_shape(Int<n>{}, Int<k>{}), GenColMajor{}));
  using gmem_c_layout_t = decltype(make_layout(make_shape(Int<m>{}, Int<n>{})));

  using smem_a_layout_t = decltype(make_layout(make_shape(Int<m>{}, Int<k>{})));
  using smem_b_layout_t = decltype(make_layout(make_shape(Int<n>{}, Int<k>{}), GenColMajor{}));
  using smem_c_layout_t = decltype(make_layout(make_shape(Int<m>{}, Int<n>{})));

  test_cooperative_gemm<gmem_a_layout_t,
                        gmem_b_layout_t,
                        gmem_c_layout_t,
                        smem_a_layout_t,
                        smem_b_layout_t,
                        smem_c_layout_t,
                        AutoVectorizingCopyWithAssumedAlignment<128>, // A
                        AutoVectorizingCopyWithAssumedAlignment<128>, // B
                        AutoVectorizingCopyWithAssumedAlignment<128>, // C
                        thread_block_size,
                        tiled_mma_t,
                        128,
                        value_type,
                        value_type,
                        value_type>();
}

TEST(SM70_CuTe_Volta, CooperativeGemm5_Half_MMA_Predicated) {
  using value_type = cutlass::half_t;

  constexpr uint32_t m = 31;
  constexpr uint32_t n = 27;
  constexpr uint32_t k = 17;

  constexpr uint32_t thread_block_size = 128;

  using tiled_mma_t = TiledMMA<
    MMA_Atom<SM70_8x8x4_F16F16F16F16_TN>,
    Layout<Shape<_4, _4, _1>>
  >;

  using gmem_a_layout_t = decltype(make_layout(make_shape(Int<m>{}, Int<k>{})));
  using gmem_b_layout_t = decltype(make_layout(make_shape(Int<n>{}, Int<k>{}), GenColMajor{}));
  using gmem_c_layout_t = decltype(make_layout(make_shape(Int<m>{}, Int<n>{})));

  using smem_a_layout_t = decltype(make_layout(make_shape(Int<m>{}, Int<k>{})));
  using smem_b_layout_t = decltype(make_layout(make_shape(Int<n>{}, Int<k>{}), GenColMajor{}));
  using smem_c_layout_t = decltype(make_layout(make_shape(Int<m>{}, Int<n>{})));

  test_cooperative_gemm<gmem_a_layout_t,
                        gmem_b_layout_t,
                        gmem_c_layout_t,
                        smem_a_layout_t,
                        smem_b_layout_t,
                        smem_c_layout_t,
                        AutoVectorizingCopyWithAssumedAlignment<16>, // A
                        AutoVectorizingCopyWithAssumedAlignment<16>, // B
                        AutoVectorizingCopyWithAssumedAlignment<16>, // C
                        thread_block_size,
                        tiled_mma_t,
                        16,
                        value_type,
                        value_type,
                        value_type>();
}

TEST(SM70_CuTe_Volta, CooperativeGemm6_Half_MAA_SwizzledSmemLayouts) {
  using value_type = cutlass::half_t;

  constexpr uint32_t m = 128;
  constexpr uint32_t n = 128;
  constexpr uint32_t k = 64;

  constexpr uint32_t thread_block_size = 128;

  using tiled_mma_t = TiledMMA<
    MMA_Atom<SM70_8x8x4_F16F16F16F16_TN>,
    Layout<Shape<_4, _4, _1>>
  >;

  using smem_a_atom_layout_t = decltype(
    composition(Swizzle<3,3,3>{},
                Layout<Shape < _8,_64>,
                       Stride<_64, _1>>{}));
  using smem_b_atom_layout_t = decltype(
    composition(Swizzle<3,3,3>{},
                Layout<Shape <_64, _8>,
                       Stride< _1,_64>>{}));
  using smem_c_atom_layout_t = decltype(make_layout(make_shape(Int<m>{}, Int<n>{}), GenRowMajor{}));

  using gmem_a_layout_t = decltype(make_layout(make_shape(Int<m> {}, Int<k> {}), GenRowMajor{}));
  using gmem_b_layout_t = decltype(make_layout(make_shape(Int<n> {}, Int<k> {}), GenColMajor{}));
  using gmem_c_layout_t = decltype(make_layout(make_shape(Int<m> {}, Int<n> {}), GenRowMajor{}));

  using smem_a_atom_layout_t = smem_a_atom_layout_t;
  using smem_a_layout_t = decltype(tile_to_shape(
      smem_a_atom_layout_t{},
      make_shape(shape<0>(gmem_a_layout_t{}), shape<1>(gmem_a_layout_t{})))
  );

  // Transposed
  using smem_b_atom_layout_t = smem_b_atom_layout_t;
  using smem_b_layout_t = decltype(tile_to_shape(
      smem_b_atom_layout_t{},
      make_shape(shape<0>(gmem_b_layout_t{}), shape<1>(gmem_b_layout_t{})))
  );

  using smem_c_atom_layout_t = smem_c_atom_layout_t;
  using smem_c_layout_t = decltype(tile_to_shape(
      smem_c_atom_layout_t{},
      make_shape(shape<0>(gmem_c_layout_t{}), shape<1>(gmem_c_layout_t{})))
  );

  test_cooperative_gemm<gmem_a_layout_t,
                        gmem_b_layout_t,
                        gmem_c_layout_t,
                        smem_a_layout_t,
                        smem_b_layout_t,
                        smem_c_layout_t,
                        AutoVectorizingCopyWithAssumedAlignment<128>, // A
                        AutoVectorizingCopyWithAssumedAlignment<128>, // B
                        AutoVectorizingCopyWithAssumedAlignment<128>, // C
                        thread_block_size,
                        tiled_mma_t,
                        128,
                        value_type,
                        value_type,
                        value_type>();
}

TEST(SM70_CuTe_Volta, CooperativeGemm7_TransformNegate_FMA) {
  using TA = float;
  using TB = float;
  using TC = double;

  constexpr uint32_t m = 32;
  constexpr uint32_t n = 32;
  constexpr uint32_t k = 32;

  constexpr uint32_t thread_block_size = 128;

  using tiled_mma_t = TiledMMA<
    MMA_Atom<UniversalFMA<TC, TA, TB, TC>>,
    Layout<Shape<_16, _8, _1>>
  >;

  auto aload  = cute::negate {};
  auto bload  = cute::negate {};
  auto cload  = cute::negate {};
  auto cstore = cute::negate {};

  test_cooperative_gemm_col_major_layout<m, n, k, thread_block_size, tiled_mma_t, 64, TA, TB, TC>(
      aload, bload, cload, cstore);
}

TEST(SM70_CuTe_Volta, CooperativeGemm7_TransformNegate_MMA) {
  using value_type = cutlass::half_t;

  constexpr uint32_t m = 32;
  constexpr uint32_t n = 32;
  constexpr uint32_t k = 32;

  constexpr uint32_t thread_block_size = 128;

  using tiled_mma_t = TiledMMA<
    MMA_Atom<SM70_8x8x4_F16F16F16F16_TN>,
    Layout<Shape<_4, _4, _1>>
  >;

  auto aload  = cute::negate {};
  auto bload  = cute::negate {};
  auto cload  = cute::negate {};
  auto cstore = cute::negate {};

  test_cooperative_gemm_col_major_layout<m, n, k, thread_block_size, tiled_mma_t, value_type>(
      aload, bload, cload, cstore);
}

template<class ConstantType>
struct increment_by_x {
  ConstantType x;

  template <class T>
  CUTE_HOST_DEVICE constexpr
  T operator()(const T& arg) const {
    return arg + x;
  }
};

template<class From, class To>
struct convert_to {
  CUTE_HOST_DEVICE constexpr
  To operator()(const From& arg) const {
    return static_cast<To>(arg);
  }
};

TEST(SM70_CuTe_Volta, CooperativeGemm7_TransformCustomOp_FMA) {
  using TA = float;
  using TB = float;
  using TC = double;

  constexpr uint32_t m = 32;
  constexpr uint32_t n = 32;
  constexpr uint32_t k = 32;

  constexpr uint32_t thread_block_size = 128;

  using tiled_mma_t = TiledMMA<
    MMA_Atom<UniversalFMA<TC, TA, TB, TC>>,
    Layout<Shape<_16, _8, _1>>
  >;

  auto aload  = increment_by_x<float>{1.111f};
  auto bload  = convert_to<float, double> {};
  auto cload  = cute::negate {};
  auto cstore = cute::negate {};

  test_cooperative_gemm_col_major_layout<m, n, k, thread_block_size, tiled_mma_t, 64, TA, TB, TC>(
      aload, bload, cload, cstore);
}
