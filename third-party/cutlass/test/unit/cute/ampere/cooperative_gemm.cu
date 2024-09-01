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

TEST(SM80_CuTe_Ampere, CooperativeGemm1_Half_MMA) {
  using value_type = cutlass::half_t;

  constexpr uint32_t m = 64;
  constexpr uint32_t n = 64;
  constexpr uint32_t k = 64;

  constexpr uint32_t thread_block_size = 128;

  using tiled_mma_t =
      TiledMMA<
        MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>,
        Layout<Shape<_2, _2, _1>>
      >;

  test_cooperative_gemm_col_major_layout<m, n, k, thread_block_size, tiled_mma_t, value_type>();
}

TEST(SM80_CuTe_Ampere, CooperativeGemm2_Double_MMA) {
  using value_type = double;

  constexpr uint32_t m = 64;
  constexpr uint32_t n = 64;
  constexpr uint32_t k = 64;

  constexpr uint32_t thread_block_size = 128;

  using tiled_mma_t =
      TiledMMA<
        MMA_Atom<SM80_8x8x4_F64F64F64F64_TN>,
         Layout<Shape<_2,_2,_1>>
      >;

  test_cooperative_gemm_col_major_layout<m, n, k, thread_block_size, tiled_mma_t, value_type>();
}

TEST(SM80_CuTe_Ampere, CooperativeGemm3_Half_MMA_CustomSmemLayouts) {
  using value_type = cutlass::half_t;

  constexpr uint32_t m = 128;
  constexpr uint32_t n = 128;
  constexpr uint32_t k = 128;

  constexpr uint32_t thread_block_size = 128;

  using tiled_mma_t =
    TiledMMA<
      MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>,
      Layout<Shape<_2, _2, _1>>, // 2x2x1 thread group
      Tile<_32, _32, _16> // 32x32x16 MMA for LDSM, 1x2x1 value group`
    >;

  using smem_a_atom_layout_t = Layout<Shape<_64, _8>, Stride< _1,_64>>;
  using smem_b_atom_layout_t = Layout<Shape< _8,_32>, Stride<_32, _1>>;
  using smem_c_atom_layout_t = decltype(make_layout(make_shape(Int<m>{}, Int<n>{})));

  test_cooperative_gemm_col_major_layout<smem_a_atom_layout_t,
                                         smem_b_atom_layout_t,
                                         smem_c_atom_layout_t,
                                         m,
                                         n,
                                         k,
                                         thread_block_size,
                                         tiled_mma_t,
                                         128,
                                         value_type,
                                         value_type,
                                         value_type>();
}

TEST(SM80_CuTe_Ampere, CooperativeGemm4_Half_MMA_SwizzledSmemLayouts) {
  using value_type = cutlass::half_t;

  constexpr uint32_t m = 128;
  constexpr uint32_t n = 128;
  constexpr uint32_t k = 128;

  constexpr uint32_t thread_block_size = 128;

  using tiled_mma_t =
    TiledMMA<
      MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>,
      Layout<Shape<_2, _2, _1>>, // 2x2x1 thread group
      Tile<_32, _32, _16> // 32x32x16 MMA for LDSM, 1x2x1 value group`
    >;

  // RowMajor
  using smem_rowmajor_atom_layout_t = decltype(
    composition(Swizzle<3,3,3>{},
                Layout<Shape < _8,_64>,
                       Stride<_64, _1>>{}));
  // ColMajor
  using smem_colmajor_atom_layout_t = decltype(
    composition(Swizzle<3,3,3>{},
                Layout<Shape <_64, _8>,
                       Stride< _1,_64>>{}));
  using smem_a_atom_layout_t = smem_rowmajor_atom_layout_t;
  using smem_b_atom_layout_t = smem_colmajor_atom_layout_t;
  using smem_c_atom_layout_t = decltype(make_layout(make_shape(Int<m>{}, Int<n>{}), GenRowMajor{}));

  using gmem_a_layout_t = decltype(make_layout(make_shape(Int<m> {}, Int<k> {}), GenRowMajor{}));
  using gmem_b_layout_t = decltype(make_layout(make_shape(Int<n> {}, Int<k> {}), GenColMajor{}));
  using gmem_c_layout_t = decltype(make_layout(make_shape(Int<m> {}, Int<n> {}), GenRowMajor{}));

  using smem_a_atom_layout_t = smem_a_atom_layout_t;
  using smem_a_layout_t = decltype(tile_to_shape(
      smem_a_atom_layout_t{},
      make_shape(shape<0>(gmem_a_layout_t{}), shape<1>(gmem_a_layout_t{})))
  );

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
                        SM75_U32x4_LDSM_N, // A
                        SM75_U16x8_LDSM_T, // B
                        AutoVectorizingCopyWithAssumedAlignment<128>, // C
                        thread_block_size,
                        tiled_mma_t,
                        128,
                        value_type,
                        value_type,
                        value_type>();
}

TEST(SM80_CuTe_Ampere, CooperativeGemm5_Double_MMA_SwizzledSmemLayouts) {
  using value_type = double;

  constexpr uint32_t m = 128;
  constexpr uint32_t n = 64;
  constexpr uint32_t k = 16;

  constexpr uint32_t thread_block_size = 128;

  using tiled_mma_t =
      TiledMMA<MMA_Atom<SM80_8x8x4_F64F64F64F64_TN>,        // Atom
               Layout<Shape<_2, _2, _1>>,                   // Atom layout
               Tile<Layout<Shape<_16, _2>, Stride<_2, _1>>, // 32x32x4 MMA with perm for load vectorization
                    Layout<Shape<_16, _2>, Stride<_2, _1>>,
                    Underscore>>;

  using smem_a_atom_layout_t = decltype(
      composition(Swizzle<2,2,2>{},
                  Layout<Shape <_16, _4>,
                         Stride< _1,_16>>{})); // M, K
  using smem_b_atom_layout_t = decltype(
      composition(Swizzle<2,2,2>{},
                  Layout<Shape <_16, _4>,
                         Stride< _1,_16>>{})); // N, K
  using smem_c_atom_layout_t = decltype(make_layout(make_shape(Int<m>{}, Int<n>{}), GenRowMajor{}));

  using gmem_a_layout_t = decltype(make_layout(make_shape(Int<m> {}, Int<k> {}), GenRowMajor{}));
  using gmem_b_layout_t = decltype(make_layout(make_shape(Int<n> {}, Int<k> {}), GenColMajor{}));
  using gmem_c_layout_t = decltype(make_layout(make_shape(Int<m> {}, Int<n> {}), GenRowMajor{}));

  using smem_a_atom_layout_t = smem_a_atom_layout_t;
  using smem_a_layout_t = decltype(tile_to_shape(
      smem_a_atom_layout_t{},
      make_shape(shape<0>(gmem_a_layout_t{}), shape<1>(gmem_a_layout_t{})))
  );
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

TEST(SM80_CuTe_Ampere, CooperativeGemm6_MixedPrecisionFP16FP32_MMA) {
  using TA = cutlass::half_t;
  using TB = cutlass::half_t;
  using TC = float;

  constexpr uint32_t m = 64;
  constexpr uint32_t n = 64;
  constexpr uint32_t k = 64;

  constexpr uint32_t thread_block_size = 128;

  using tiled_mma_t =
      TiledMMA<
        MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>,
        Layout<Shape<_2, _2, _1>>
      >;

  test_cooperative_gemm_col_major_layout<m, n, k, thread_block_size, tiled_mma_t, 128, TA, TB, TC>();
}

TEST(SM80_CuTe_Ampere, CooperativeGemm7_MixedPrecisionBF16FP32_MMA) {
  using TA = cutlass::bfloat16_t;
  using TB = cutlass::bfloat16_t;
  using TC = float;

  constexpr uint32_t m = 64;
  constexpr uint32_t n = 64;
  constexpr uint32_t k = 64;

  constexpr uint32_t thread_block_size = 128;

  using tiled_mma_t =
      TiledMMA<
        MMA_Atom<SM80_16x8x8_F32BF16BF16F32_TN>,
        Layout<Shape<_2, _2, _1>>
      >;

  test_cooperative_gemm_col_major_layout<m, n, k, thread_block_size, tiled_mma_t, 128, TA, TB, TC>();
}

TEST(SM80_CuTe_Ampere, CooperativeGemm8_MixedPrecisionTF32FP32_MMA) {
  using TA = cutlass::tfloat32_t;
  using TB = cutlass::tfloat32_t;
  using TC = float;

  constexpr uint32_t m = 64;
  constexpr uint32_t n = 64;
  constexpr uint32_t k = 64;

  constexpr uint32_t thread_block_size = 128;

  using tiled_mma_t =
      TiledMMA<
        MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>,
        Layout<Shape<_2, _2, _1>>
      >;

  test_cooperative_gemm_col_major_layout<m, n, k, thread_block_size, tiled_mma_t, 128, TA, TB, TC>();
}
