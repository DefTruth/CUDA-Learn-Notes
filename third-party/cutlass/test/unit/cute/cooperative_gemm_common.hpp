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

#include "cutlass_unit_test.h"
#include "cutlass/util/reference/host/tensor_compare.h"

#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

using namespace cute;

template<class ALayout,
         class BLayout,
         class CLayout,
         class SMemALayout,
         class SMemBLayout,
         class SMemCLayout,
         class SmemCopyOpA,
         class SmemCopyOpB,
         class SmemCopyOpC,
         uint32_t ThreadBlockSize,
         class TiledMma,
         uint32_t CopyMaxVecBits,
         class TA,
         class TB,
         class TC,
         class Alpha,
         class Beta,
         class ALoadTransform,
         class BLoadTransform,
         class CLoadTransform,
         class CStoreTransform>
__launch_bounds__(ThreadBlockSize) __global__ void
cooperative_gemm_kernel(TA const*   a,
                        TB const*   b,
                        TC*         c,
                        TC*         c_out,
                        Alpha const alpha,
                        Beta  const beta,
                        ALoadTransform  a_load_transform,
                        BLoadTransform  b_load_transform,
                        CLoadTransform  c_load_transform,
                        CStoreTransform c_store_transform)
{
    using namespace cute;

    Tensor g_a_tensor     = make_tensor(make_gmem_ptr(a), ALayout{});
    Tensor g_b_tensor     = make_tensor(make_gmem_ptr(b), BLayout{});
    Tensor g_c_tensor     = make_tensor(make_gmem_ptr(c), CLayout{});
    Tensor g_c_out_tensor = make_tensor(make_gmem_ptr(c_out), CLayout{});

    constexpr uint32_t copy_max_vec_bytes = CopyMaxVecBits / 8;

    extern __shared__ float4 smem_buf[];
    auto* smem_ptr = reinterpret_cast<unsigned char*>(smem_buf);
    auto* smem_ptr_a = smem_ptr;
    auto* smem_ptr_b = smem_ptr_a + round_up((sizeof(TA) * cosize(SMemALayout {})), copy_max_vec_bytes);
    auto* smem_ptr_c = smem_ptr_b + round_up((sizeof(TB) * cosize(SMemBLayout {})), copy_max_vec_bytes);

    Tensor s_a_tensor = make_tensor(make_smem_ptr<TA>(smem_ptr_a), SMemALayout{});
    Tensor s_b_tensor = make_tensor(make_smem_ptr<TB>(smem_ptr_b), SMemBLayout{});
    Tensor s_c_tensor = make_tensor(make_smem_ptr<TC>(smem_ptr_c), SMemCLayout{});

    cooperative_copy<ThreadBlockSize, CopyMaxVecBits>(threadIdx.x, g_a_tensor, s_a_tensor);
    cooperative_copy<ThreadBlockSize, CopyMaxVecBits>(threadIdx.x, g_b_tensor, s_b_tensor);
    cooperative_copy<ThreadBlockSize, CopyMaxVecBits>(threadIdx.x, g_c_tensor, s_c_tensor);

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    TiledMma tiled_mma;
    cooperative_gemm<SmemCopyOpA, SmemCopyOpB, SmemCopyOpC>(
      threadIdx.x, tiled_mma,
      alpha, s_a_tensor, s_b_tensor, beta, s_c_tensor,
      a_load_transform, b_load_transform, c_load_transform, c_store_transform
    );
    __syncthreads();

    cooperative_copy<ThreadBlockSize, CopyMaxVecBits>(threadIdx.x, s_c_tensor, g_c_out_tensor);
}

template<class ALayout, // logical shape (M, K)
         class BLayout, // logical shape (N, K)
         class CLayout, // logical shape (M, N)
         class SMemALayout, // logical shape (M, K)
         class SMemBLayout, // logical shape (N, K)
         class SMemCLayout, // logical shape (M, N)
         class SmemCopyOpA,
         class SmemCopyOpB,
         class SmemCopyOpC,
         uint32_t ThreadBlockSize,
         class TiledMma,
         uint32_t CopyMaxVecBits,
         class TA,
         class TB,
         class TC,
         class ALoadTransform  = cute::identity,
         class BLoadTransform  = cute::identity,
         class CLoadTransform  = cute::identity,
         class CStoreTransform = cute::identity>
void test_cooperative_gemm(ALoadTransform  const& a_load_transform  = {},
                           BLoadTransform  const& b_load_transform  = {},
                           CLoadTransform  const& c_load_transform  = {},
                           CStoreTransform const& c_store_transform = {})
{
  using gmem_a_layout_t = ALayout;
  using gmem_b_layout_t = BLayout;
  using gmem_c_layout_t = CLayout;

  using smem_a_layout_t = SMemALayout;
  using smem_b_layout_t = SMemBLayout;
  using smem_c_layout_t = SMemCLayout;

  static_assert(size<0>(gmem_a_layout_t{}) == size<0>(gmem_c_layout_t{}));  // AM == CM
  static_assert(size<0>(gmem_b_layout_t{}) == size<1>(gmem_c_layout_t{}));  // BN == CN
  static_assert(size<1>(gmem_a_layout_t{}) == size<1>(gmem_b_layout_t{}));  // AK == BK

  static_assert(size<0>(smem_a_layout_t{}) == size<0>(smem_c_layout_t{}));  // AM == CM
  static_assert(size<0>(smem_b_layout_t{}) == size<1>(smem_c_layout_t{}));  // BN == CN
  static_assert(size<1>(smem_a_layout_t{}) == size<1>(smem_b_layout_t{}));  // AK == BK

  static_assert(cute::size(gmem_a_layout_t {}) == cute::size(smem_a_layout_t {}));
  static_assert(cute::size(gmem_b_layout_t {}) == cute::size(smem_b_layout_t {}));
  static_assert(cute::size(gmem_c_layout_t {}) == cute::size(smem_c_layout_t {}));

#if 0
  print("   "); print("gmem:    "); print(gmem_layout_t{}); print("\n");
  print("   "); print("smem:    "); print(smem_layout_t{}); print("\n");
  print("   "); print("threads: "); print(ThreadBlockSize); print("\n");
#endif

  const auto alpha = static_cast<TC>(1.1);
  const auto beta  = static_cast<TC>(1.2);

  thrust::host_vector<TA> h_a(cosize(gmem_a_layout_t{}));
  thrust::host_vector<TB> h_b(cosize(gmem_b_layout_t{}));
  thrust::host_vector<TC> h_c(cosize(gmem_c_layout_t{}));
  thrust::host_vector<TC> h_c_out(cosize(gmem_c_layout_t{}));

  auto h_a_tensor = make_tensor(h_a.data(), gmem_a_layout_t{});
  auto h_b_tensor = make_tensor(h_b.data(), gmem_b_layout_t{});
  auto h_c_tensor = make_tensor(h_c.data(), gmem_c_layout_t{});
  size_t max_size   = std::max<size_t>({static_cast<size_t>(size(gmem_a_layout_t {})),
                                        static_cast<size_t>(size(gmem_b_layout_t {})),
                                        static_cast<size_t>(size(gmem_c_layout_t {}))});
  for (size_t i = 0; i < max_size; ++i) {
    double di = static_cast<double>(i);
    if(i < size(gmem_a_layout_t{})) {
      h_a_tensor(i) = static_cast<TA>(di / size(gmem_a_layout_t{}));
    }
    if(i < size(gmem_b_layout_t{})) {
      h_b_tensor(i) = static_cast<TA>(di / size(gmem_a_layout_t{}));
    }
    if(i < size(gmem_c_layout_t{})) {
      h_c_tensor(i) = static_cast<TC>((di*di) / size(gmem_a_layout_t{}));
    }
  }

  thrust::device_vector<TA> d_a(h_a);
  thrust::device_vector<TB> d_b(h_b);
  thrust::device_vector<TC> d_c(h_c);
  thrust::device_vector<TC> d_c_out(h_c_out.size(), TC(float(-1)));

  const size_t shared_memory_size =
    (sizeof(TA) * h_a.size()) + (sizeof(TB) * h_b.size()) + (sizeof(TC) * h_c.size());
  auto kernel = cooperative_gemm_kernel<
    gmem_a_layout_t, gmem_b_layout_t, gmem_c_layout_t,
    smem_a_layout_t, smem_b_layout_t, smem_c_layout_t,
    SmemCopyOpA, SmemCopyOpB, SmemCopyOpC,
    ThreadBlockSize, TiledMma, CopyMaxVecBits,
    TA, TB, TC, decltype(alpha), decltype(beta),
    ALoadTransform, BLoadTransform, CLoadTransform, CStoreTransform
  >;
  ASSERT_EQ(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_memory_size)), 0);

  kernel<<<1, ThreadBlockSize, shared_memory_size>>>(
    thrust::raw_pointer_cast(d_a.data()),
    thrust::raw_pointer_cast(d_b.data()),
    thrust::raw_pointer_cast(d_c.data()),
    thrust::raw_pointer_cast(d_c_out.data()),
    alpha,
    beta,
    a_load_transform,
    b_load_transform,
    c_load_transform,
    c_store_transform
  );
  cudaError_t result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    cudaError_t error = cudaGetLastError();
    FAIL() << "Error at kernel sync: " << cudaGetErrorString(error) << "\n";
  }

  thrust::host_vector<TC> h_c_ref(h_c.size(), static_cast<TC>(0.0));
  auto h_c_ref_tensor = make_tensor(h_c_ref.data(), gmem_c_layout_t{});
  // A * B
  for (int k = 0; k < size<1>(h_a_tensor); k++) {
    for (int m = 0; m < size<0>(h_a_tensor); m++) {
      for (int n = 0; n < size<0>(h_b_tensor); n++) {
          const auto a_value      = a_load_transform(h_a_tensor(m, k));
          const auto b_value      = b_load_transform(h_b_tensor(n, k));
          const auto a_value_fp64 = static_cast<double>(a_value);
          const auto b_value_fp64 = static_cast<double>(b_value);
          h_c_ref_tensor(m, n) += static_cast<TC>(a_value_fp64 * b_value_fp64);
      }
    }
  }
  // C = A*B + C
  for (int i = 0; i < size(h_c_ref_tensor); i++) {
    const auto ab_value_fp64 = static_cast<double>(h_c_ref_tensor(i));
    const auto c_value_fp64  = static_cast<double>(c_load_transform(h_c_tensor(i)));
    h_c_ref_tensor(i)        = c_store_transform(static_cast<TC>(alpha * ab_value_fp64 + beta * c_value_fp64));
  }

  h_c_out = d_c_out;
  auto h_c_out_tensor = make_tensor(h_c_out.data(), gmem_c_layout_t{});
  for (int i = 0; i < size(h_c_ref_tensor); i++) {
    double h_c_ref_i = h_c_ref_tensor(i);
    double h_c_out_i = h_c_out_tensor(i);
    double epsilon(0.1f);
    double nonzero_floor(std::numeric_limits<double>::min());
    bool passed = cutlass::relatively_equal(h_c_out_i, h_c_ref_i, epsilon, nonzero_floor);
    ASSERT_TRUE(passed) << i << " - result:" << h_c_out_i << " expected:" << h_c_ref_i;
  }
}

template<uint32_t M,
         uint32_t N,
         uint32_t K,
         uint32_t ThreadBlockSize,
         class TiledMMAType,
         uint32_t CopyMaxVecBits,
         class TA,
         class TB,
         class TC,
         class ALoadTransform  = cute::identity,
         class BLoadTransform  = cute::identity,
         class CLoadTransform  = cute::identity,
         class CStoreTransform = cute::identity>
void test_cooperative_gemm_col_major_layout(ALoadTransform  const& a_load_transform  = {},
                                            BLoadTransform  const& b_load_transform  = {},
                                            CLoadTransform  const& c_load_transform  = {},
                                            CStoreTransform const& c_store_transform = {})
{
  using gmem_a_layout_t = decltype(make_layout(make_shape(Int<M> {}, Int<K> {})));
  using gmem_b_layout_t = decltype(make_layout(make_shape(Int<N> {}, Int<K> {}), GenRowMajor{}));
  using gmem_c_layout_t = decltype(make_layout(make_shape(Int<M> {}, Int<N> {})));

  using smem_a_layout_t = decltype(make_layout(make_shape(Int<M> {}, Int<K> {})));
  using smem_b_layout_t = decltype(make_layout(make_shape(Int<N> {}, Int<K> {}), GenRowMajor{}));
  using smem_c_layout_t = decltype(make_layout(make_shape(Int<M> {}, Int<N> {})));

  test_cooperative_gemm<gmem_a_layout_t,
                        gmem_b_layout_t,
                        gmem_c_layout_t,
                        smem_a_layout_t,
                        smem_b_layout_t,
                        smem_c_layout_t,
                        AutoVectorizingCopyWithAssumedAlignment<sizeof_bits_v<TA>>,
                        AutoVectorizingCopyWithAssumedAlignment<sizeof_bits_v<TB>>,
                        AutoVectorizingCopyWithAssumedAlignment<sizeof_bits_v<TC>>,
                        ThreadBlockSize,
                        TiledMMAType,
                        CopyMaxVecBits,
                        TA,
                        TB,
                        TC>(a_load_transform, b_load_transform, c_load_transform, c_store_transform);
}

template<uint32_t M,
         uint32_t N,
         uint32_t K,
         uint32_t ThreadBlockSize,
         class TiledMMAType,
         class T,
         class ALoadTransform  = cute::identity,
         class BLoadTransform  = cute::identity,
         class CLoadTransform  = cute::identity,
         class CStoreTransform = cute::identity>
void test_cooperative_gemm_col_major_layout(ALoadTransform  const& a_load_transform  = {},
                                            BLoadTransform  const& b_load_transform  = {},
                                            CLoadTransform  const& c_load_transform  = {},
                                            CStoreTransform const& c_store_transform = {})
{
  test_cooperative_gemm_col_major_layout<M, N, K, ThreadBlockSize, TiledMMAType, cute::sizeof_bits_v<T>, T, T, T>(
      a_load_transform, b_load_transform, c_load_transform, c_store_transform);
}

template<class SMemAAtomLayout,
         class SMemBAtomLayout,
         class SMemCAtomLayout,
         uint32_t M,
         uint32_t N,
         uint32_t K,
         uint32_t ThreadBlockSize,
         class TiledMMAType,
         uint32_t CopyMaxVecBits,
         class TA,
         class TB,
         class TC,
         class ALoadTransform  = cute::identity,
         class BLoadTransform  = cute::identity,
         class CLoadTransform  = cute::identity,
         class CStoreTransform = cute::identity>
void test_cooperative_gemm_col_major_layout(ALoadTransform  const& a_load_transform  = {},
                                            BLoadTransform  const& b_load_transform  = {},
                                            CLoadTransform  const& c_load_transform  = {},
                                            CStoreTransform const& c_store_transform = {})
{
  using gmem_a_layout_t = decltype(make_layout(make_shape(Int<M> {}, Int<K> {})));
  using gmem_b_layout_t = decltype(make_layout(make_shape(Int<N> {}, Int<K> {}), GenRowMajor{}));
  using gmem_c_layout_t = decltype(make_layout(make_shape(Int<M> {}, Int<N> {})));

  using smem_a_atom_layout_t = SMemAAtomLayout;
  using smem_a_layout_t = decltype(tile_to_shape(
      smem_a_atom_layout_t{},
      make_shape(shape<0>(gmem_a_layout_t{}), shape<1>(gmem_a_layout_t{})))
  );

  using smem_b_atom_layout_t = SMemBAtomLayout;
  using smem_b_layout_t = decltype(tile_to_shape(
      smem_b_atom_layout_t{},
      make_shape(shape<0>(gmem_b_layout_t{}), shape<1>(gmem_b_layout_t{})))
  );

  using smem_c_atom_layout_t = SMemCAtomLayout;
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
                        AutoVectorizingCopyWithAssumedAlignment<sizeof_bits_v<TA>>,
                        AutoVectorizingCopyWithAssumedAlignment<sizeof_bits_v<TB>>,
                        AutoVectorizingCopyWithAssumedAlignment<sizeof_bits_v<TC>>,
                        ThreadBlockSize,
                        TiledMMAType,
                        CopyMaxVecBits,
                        TA,
                        TB,
                        TC>(a_load_transform, b_load_transform, c_load_transform, c_store_transform);
}

template<class SMemAAtomLayout,
         class SMemBAtomLayout,
         class SMemCAtomLayout,
         uint32_t M,
         uint32_t N,
         uint32_t K,
         uint32_t ThreadBlockSize,
         class TiledMMAType,
         class T,
         class ALoadTransform  = cute::identity,
         class BLoadTransform  = cute::identity,
         class CLoadTransform  = cute::identity,
         class CStoreTransform = cute::identity>
void test_cooperative_gemm_col_major_layout(ALoadTransform  const& a_load_transform  = {},
                                            BLoadTransform  const& b_load_transform  = {},
                                            CLoadTransform  const& c_load_transform  = {},
                                            CStoreTransform const& c_store_transform = {})
{
  test_cooperative_gemm_col_major_layout<SMemAAtomLayout,
                                         SMemBAtomLayout,
                                         SMemCAtomLayout,
                                         M,
                                         N,
                                         K,
                                         ThreadBlockSize,
                                         TiledMMAType,
                                         cute::sizeof_bits_v<T>,
                                         T,
                                         T,
                                         T>(a_load_transform, b_load_transform, c_load_transform, c_store_transform);
}
