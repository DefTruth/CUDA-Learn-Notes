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

#include "cutlass_unit_test.h"

#include <iostream>
#include <iomanip>
#include <utility>
#include <type_traits>
#include <vector>
#include <numeric>
#include <tuple>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include <cute/numeric/numeric_types.hpp>

using namespace cute;

namespace cooperative_copy_mode {
  struct global_shared {};
  struct global_global {};
  struct shared_shared {};
}

// gs --> global to/from shared
template <int MaxVecBits, class GMemLayout, class SMemLayout, uint32_t ThreadBlockSize, class T>
__device__ void
cooperative_copy_default_gs(T const* g_in, T* g_out)
{
  using namespace cute;
  extern __shared__ float4 smem_buf[];
  // Cast smem_buf to smem_uint8_ptr and move it by MaxVecBits bits
  // This is to make sure tests pass on pointer aligned to MaxVecBits bits
  uint8_t* smem_uint8_ptr = reinterpret_cast<uint8_t*>(smem_buf) + (MaxVecBits/8);
  T* smem = reinterpret_cast<T*>(smem_uint8_ptr);

  Tensor g_in_tensor  = make_tensor(make_gmem_ptr(g_in),  GMemLayout{});
  Tensor g_out_tensor = make_tensor(make_gmem_ptr(g_out), GMemLayout{});
  Tensor s_tensor     = make_tensor(make_smem_ptr(smem), SMemLayout{});

  cooperative_copy<ThreadBlockSize, MaxVecBits>(threadIdx.x, g_in_tensor, s_tensor);
  __syncthreads();

  if(thread0()) {
    for(int i = 0; i < size(s_tensor); ++i) {
      s_tensor(i) += T(i);
    }
  }
  __syncthreads();

  cooperative_copy<ThreadBlockSize, MaxVecBits>(threadIdx.x, s_tensor, g_out_tensor);
}

// ss --> shared to shared
template <int MaxVecBits, class Layout1, class Layout2, uint32_t ThreadBlockSize, class T>
__device__ void
cooperative_copy_default_ss(T const* g_in, T* g_out)
{
  using namespace cute;
  extern __shared__ float4 smem_buf[];
  // Cast smem_buf to smem_uint8_ptr and move it by MaxVecBits bits
  // This is to make sure tests pass on pointer aligned to MaxVecBits bits
  T* smem1 = reinterpret_cast<T*>(smem_buf);
  uint8_t* smem2_uint8_ptr = reinterpret_cast<uint8_t*>(smem_buf) + (MaxVecBits/8);
  T* smem2 = reinterpret_cast<T*>(smem2_uint8_ptr) + cute::cosize(Layout2{});

  Tensor g_in_tensor  = make_tensor(make_gmem_ptr(g_in), Layout1 {});
  Tensor g_out_tensor = make_tensor(make_gmem_ptr(g_out), Layout2 {});

  Tensor s1_tensor    = make_tensor(make_smem_ptr(smem1), Layout2 {});
  Tensor s2_tensor    = make_tensor(make_smem_ptr(smem2), Layout1 {});

  cooperative_copy<ThreadBlockSize,  cute::sizeof_bits_v<T>>(threadIdx.x, g_in_tensor, s1_tensor);
  __syncthreads();

  if(thread0()) {
    for(int i = 0; i < size(s1_tensor); ++i) {
      s1_tensor(i) += T(i);
    }
  }
  __syncthreads();

  cooperative_copy<ThreadBlockSize, MaxVecBits>(threadIdx.x, s1_tensor, s2_tensor);
  __syncthreads();

  cooperative_copy<ThreadBlockSize,  cute::sizeof_bits_v<T>>(threadIdx.x, s2_tensor, g_out_tensor);
}

// gg --> global to global
template <int MaxVecBits, class Layout1, class Layout2, uint32_t ThreadBlockSize, class T>
__device__ void
cooperative_copy_default_gg(T const* g_in, T* g_out)
{
  using namespace cute;

  Tensor g_in_tensor  = make_tensor(make_gmem_ptr(g_in), Layout1{});
  Tensor g_out_tensor = make_tensor(make_gmem_ptr(g_out), Layout2{});

  cooperative_copy<ThreadBlockSize, MaxVecBits>(threadIdx.x, g_in_tensor, g_out_tensor);
}

template <class Mode, int MaxVecBits, class Layout1, class Layout2, uint32_t ThreadBlockSize, class T>
__global__ void
cooperative_copy_default_kernel(T const* g_in, T* g_out)
{
  if constexpr(std::is_same_v<Mode, cooperative_copy_mode::global_shared>) {
    cooperative_copy_default_gs<MaxVecBits, Layout1, Layout2, ThreadBlockSize>(g_in, g_out);
  } else if constexpr (std::is_same_v<Mode, cooperative_copy_mode::global_global>) {
    cooperative_copy_default_gg<MaxVecBits, Layout1, Layout2, ThreadBlockSize>(g_in, g_out);
  } else if constexpr (std::is_same_v<Mode, cooperative_copy_mode::shared_shared>) {
    cooperative_copy_default_ss<MaxVecBits, Layout1, Layout2, ThreadBlockSize>(g_in, g_out);
  }
}

// Mode - defines memory types of src and dst in cooperative_copy operation
// MaxVecBits - defines max vectorization in cooperative_copy operation, and enforces that
//              alignment on used pointers to ensure correct testing
template <class Mode, int MaxVecBits, class Layout1, class Layout2, uint32_t ThreadBlockSize, class T>
void test_cooperative_copy_default()
{
  using value_type = T;
  static_assert(cute::size(Layout1{}) == cute::size(Layout2{}));

  using gmem_layout_in  = Layout1;
  using gmem_layout_out = std::conditional_t<std::is_same_v<Mode, cooperative_copy_mode::global_shared>, Layout1, Layout2>;

#if 0
  print("   "); print("layout1:  "); print(Layout1{}); print("\n");
  print("   "); print("layout2: "); print(Layout2{}); print("\n");
  print("   "); print("threads:  "); print(ThreadBlockSize); print("\n");
#endif

  if constexpr (MaxVecBits < cute::sizeof_bits_v<value_type>) {
    GTEST_SKIP() << "Skipping test since MaxVecBits (=" << MaxVecBits
                 << ") < cute::sizeof_bits_v<value_type> (=" << cute::sizeof_bits_v<value_type> << ")";
  } else {
    constexpr auto max_vec_bytes = MaxVecBits / 8;
    static_assert((max_vec_bytes % sizeof(T)) == 0);

    constexpr uint32_t count = cute::cosize(gmem_layout_in {});
    // Extra elements to force MaxVecBits alignment in global memory
    constexpr uint32_t extra_elements = max_vec_bytes / sizeof(value_type);

    // Allocate
    thrust::host_vector<value_type> h_in(count + extra_elements);
    thrust::host_vector<value_type> h_out(count + extra_elements);

    // Initialize
    Tensor h_in_tensor  = make_tensor((h_in.data() + extra_elements), gmem_layout_in {});
    Tensor h_out_tensor = make_tensor((h_out.data() + extra_elements), gmem_layout_out {});
    for (int i = 0; i < cute::size(h_in_tensor); ++i) {
      h_in_tensor(i)  = value_type(float(i));
      // For global-to-global copy need to compare against the same value
      h_out_tensor(i) = std::is_same_v<Mode, cooperative_copy_mode::global_global> ? value_type(float(i)) : value_type(float(2 * i));
    }

    // To GPU
    thrust::device_vector<value_type> d_in = h_in;
    thrust::device_vector<value_type> d_out(d_in.size(), value_type(float(-2)));

    // Adds (MaxVecBits/8) bytes to shared memory as we'll move pointer by that many bytes inside the kernel to enforce
    // alignment to (MaxVecBits/8) bytes
    size_t shared_memory_bytes = (sizeof(value_type) * count) + max_vec_bytes;
    shared_memory_bytes += std::is_same_v<Mode, cooperative_copy_mode::shared_shared> * (sizeof(value_type) * count);

    // Launch
    auto coop_copy = cooperative_copy_default_kernel<Mode, MaxVecBits, Layout1, Layout2, ThreadBlockSize, value_type>;
    ASSERT_EQ(cudaFuncSetAttribute(coop_copy, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_memory_bytes)), cudaSuccess);

    auto d_in_ptr = thrust::raw_pointer_cast(d_in.data() + extra_elements);
    auto d_out_ptr = thrust::raw_pointer_cast(d_out.data() + extra_elements);
    coop_copy<<<1, ThreadBlockSize, shared_memory_bytes>>>(d_in_ptr, d_out_ptr);

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      cudaError_t error = cudaGetLastError();
      FAIL() << "Error at kernel sync: " << cudaGetErrorString(error) << "\n";
    }

    // Validate
    thrust::host_vector<value_type> h_result        = d_out;
    Tensor                          h_result_tensor = make_tensor((h_result.data() + extra_elements), gmem_layout_out {});
    for (int i = 0; i < cute::size(h_in_tensor); ++i) {
      ASSERT_EQ(h_result_tensor(i), h_out_tensor(i))
          << i << " - result:" << h_result_tensor(i) << " expected:" << h_out_tensor(i);
    }
  }
}

template<class T>
class SM70_CuTe_Volta;

template<class Mode, class MaxVecBits>
class SM70_CuTe_Volta<std::tuple<Mode, MaxVecBits>>: public testing::Test
{
public:
  using mode = Mode;
  static constexpr int max_vec_bits = MaxVecBits::value;
};

typedef testing::Types<
  std::tuple<cooperative_copy_mode::global_shared, cute::Int<128>>,
  std::tuple<cooperative_copy_mode::global_shared, cute::Int<64>>,
  std::tuple<cooperative_copy_mode::global_shared, cute::Int<32>>,
  std::tuple<cooperative_copy_mode::global_shared, cute::Int<16>>,

  std::tuple<cooperative_copy_mode::global_global, cute::Int<128>>,
  std::tuple<cooperative_copy_mode::global_global, cute::Int<64>>,
  std::tuple<cooperative_copy_mode::global_global, cute::Int<32>>,
  std::tuple<cooperative_copy_mode::global_global, cute::Int<16>>,

  std::tuple<cooperative_copy_mode::shared_shared, cute::Int<128>>,
  std::tuple<cooperative_copy_mode::shared_shared, cute::Int<64>>,
  std::tuple<cooperative_copy_mode::shared_shared, cute::Int<32>>,
  std::tuple<cooperative_copy_mode::shared_shared, cute::Int<16>>,
> CooperativeCopyModeMaxVecBitsList;

TYPED_TEST_SUITE(SM70_CuTe_Volta, CooperativeCopyModeMaxVecBitsList);

TYPED_TEST(SM70_CuTe_Volta, CooperativeCopyDefault1D)
{
  using value_type = float;
  constexpr uint32_t count = 512;
  using gmem_layout_t = decltype(make_layout(make_shape(Int<count>{})));
  using smem_layout_t = decltype(make_layout(make_shape(Int<count>{})));
  constexpr uint32_t thread_block_size = 64;
  test_cooperative_copy_default<typename TestFixture::mode,
                                TestFixture::max_vec_bits,
                                gmem_layout_t,
                                smem_layout_t,
                                thread_block_size,
                                value_type>();
}

TYPED_TEST(SM70_CuTe_Volta, CooperativeCopyDefault1DFallback)
{
  using value_type = float;
  constexpr uint32_t count = 99;
  using gmem_layout_t = decltype(make_layout(make_shape(Int<count>{})));
  using smem_layout_t = decltype(make_layout(make_shape(Int<count>{})));
  constexpr uint32_t thread_block_size = 128;
  test_cooperative_copy_default<typename TestFixture::mode,
                                TestFixture::max_vec_bits,
                                gmem_layout_t,
                                smem_layout_t,
                                thread_block_size,
                                value_type>();
}

TYPED_TEST(SM70_CuTe_Volta, CooperativeCopyDefaultGSSG2D)
{
  using value_type = float;
  constexpr uint32_t x = 32;
  constexpr uint32_t y = 32;
  using gmem_layout_t = decltype(make_layout(make_shape(Int<x>{}, Int<y>{})));
  using smem_layout_t = decltype(make_layout(make_shape(Int<x>{}, Int<y>{})));
  constexpr uint32_t thread_block_size = 64;
  test_cooperative_copy_default<typename TestFixture::mode,
                                TestFixture::max_vec_bits,
                                gmem_layout_t,
                                smem_layout_t,
                                thread_block_size,
                                value_type>();
}

TYPED_TEST(SM70_CuTe_Volta, CooperativeCopyDefaultGSSG2DFallback)
{
  using value_type = float;
  constexpr uint32_t x = 37;
  constexpr uint32_t y = 37;
  using gmem_layout_t = decltype(make_layout(make_shape(Int<x>{}, Int<y>{})));
  using smem_layout_t = decltype(make_layout(make_shape(Int<x>{}, Int<y>{})));
  constexpr uint32_t thread_block_size = 64;
  test_cooperative_copy_default<typename TestFixture::mode,
                                TestFixture::max_vec_bits,
                                gmem_layout_t,
                                smem_layout_t,
                                thread_block_size,
                                value_type>();
}

TYPED_TEST(SM70_CuTe_Volta, CooperativeCopyDefaultGSSG2DCustomStride)
{
  using value_type = float;
  constexpr uint32_t x = 16;
  constexpr uint32_t y = 16;
  using gmem_layout_t = decltype(make_layout(make_shape(Int<x>{}, Int<y>{}), make_stride(Int<y>{}, Int<1>{})));
  using smem_layout_t = decltype(make_layout(make_shape(Int<x>{}, Int<y>{}), make_stride(Int<1>{}, Int<x>{})));
  constexpr uint32_t thread_block_size = 64;
  test_cooperative_copy_default<typename TestFixture::mode,
                                TestFixture::max_vec_bits,
                                gmem_layout_t,
                                smem_layout_t,
                                thread_block_size,
                                value_type>();
}

TYPED_TEST(SM70_CuTe_Volta, CooperativeCopyDefaultGSSG3D)
{
  using value_type = cute::half_t;
  constexpr uint32_t x = 8;
  constexpr uint32_t y = 8;
  constexpr uint32_t z = 16;
  using gmem_layout_t = decltype(make_layout(make_shape(Int<x>{}, Int<y>{}, Int<z>{})));
  using smem_layout_t = decltype(make_layout(make_shape(Int<x>{}, Int<y>{}, Int<z>{})));
  constexpr uint32_t thread_block_size = 64;
  test_cooperative_copy_default<typename TestFixture::mode,
                                TestFixture::max_vec_bits,
                                gmem_layout_t,
                                smem_layout_t,
                                thread_block_size,
                                value_type>();
}

TYPED_TEST(SM70_CuTe_Volta, CooperativeCopyDefaultGSSG3DFallback)
{
  using value_type = cute::half_t;
  constexpr uint32_t x = 44;
  constexpr uint32_t y = 24;
  constexpr uint32_t z = 14;
  using gmem_layout_t = decltype(make_layout(make_shape(Int<x>{}, Int<y>{}, Int<z>{})));
  using smem_layout_t = decltype(make_layout(make_shape(Int<x>{}, Int<y>{}, Int<z>{})));
  constexpr uint32_t thread_block_size = 128;
  test_cooperative_copy_default<typename TestFixture::mode,
                                TestFixture::max_vec_bits,
                                gmem_layout_t,
                                smem_layout_t,
                                thread_block_size,
                                value_type>();
}

TYPED_TEST(SM70_CuTe_Volta, CooperativeCopyDefaultGSSG2Dto3D)
{
  using value_type = double;
  constexpr uint32_t x = 16;
  constexpr uint32_t y = 16;
  constexpr uint32_t z = 4;
  using gmem_layout_t = decltype(make_layout(make_shape(Int<x>{}, Int<y*z>{})));
  using smem_layout_t = decltype(make_layout(make_shape(Int<z>{}, Int<y>{}, Int<x>{})));
  constexpr uint32_t thread_block_size = 64;
  test_cooperative_copy_default<typename TestFixture::mode,
                                TestFixture::max_vec_bits,
                                gmem_layout_t,
                                smem_layout_t,
                                thread_block_size,
                                value_type>();
}

TYPED_TEST(SM70_CuTe_Volta, CooperativeCopyDefaultGSSGCustom1)
{
  using value_type = double;
  using gmem_layout_t = decltype(make_layout(
    make_shape(Int<8>{}, make_shape(Int<2>{}, Int<2>{})),
    make_stride(Int<2>{}, make_shape(Int<1>{}, Int<16>{}))
  ));
  using smem_layout_t = decltype(make_layout(
    make_shape(Int<8>{}, Int<4>{}),
    make_stride(Int<4>{}, Int<1>{})
  ));
  constexpr uint32_t thread_block_size = 8;
  test_cooperative_copy_default<typename TestFixture::mode,
                                TestFixture::max_vec_bits,
                                gmem_layout_t,
                                smem_layout_t,
                                thread_block_size,
                                value_type>();
}

TYPED_TEST(SM70_CuTe_Volta, CooperativeCopyDefaultGSSGCustom2)
{
  using value_type = float;
  using gmem_layout_t = decltype(make_layout(
    make_shape(make_shape(Int<4>{}, Int<2>{}), make_shape(Int<2>{}, Int<2>{})),
    make_stride(make_shape(Int<4>{}, Int<1>{}), make_shape(Int<16>{}, Int<2>{}))
  ));
  using smem_layout_t = decltype(make_layout(
    make_shape(make_shape(Int<2>{}, Int<2>{}, Int<2>{}), make_shape(Int<2>{}, Int<2>{})),
    make_stride(make_shape(Int<16>{}, Int<4>{}, Int<1>{}), make_shape(Int<8>{}, Int<2>{}))
  ));
  constexpr uint32_t thread_block_size = 16;
  test_cooperative_copy_default<typename TestFixture::mode,
                                TestFixture::max_vec_bits,
                                gmem_layout_t,
                                smem_layout_t,
                                thread_block_size,
                                value_type>();
}

TYPED_TEST(SM70_CuTe_Volta, CooperativeCopyDefaultGSSGSwizzle1)
{
  using value_type = float;
  using gmem_layout_t = Layout<Shape<_8, _64>, Stride<_64, _1>>;
  using smem_layout_t = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
  constexpr uint32_t thread_block_size = 128;
  test_cooperative_copy_default<typename TestFixture::mode,
                                TestFixture::max_vec_bits,
                                gmem_layout_t,
                                smem_layout_t,
                                thread_block_size,
                                value_type>();
}

TYPED_TEST(SM70_CuTe_Volta, CooperativeCopyDefaultGSSGSwizzle2)
{
  using value_type = cute::half_t;
  using gmem_layout_t = decltype(make_layout(make_shape(Int<64>{}, Int<64>{})));
  using smem_atom_layout_t = decltype(composition(Swizzle<3, 2, 3> {}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
  using smem_layout_t = decltype(tile_to_shape(
      smem_atom_layout_t{},
      make_shape(shape<0>(gmem_layout_t{}), shape<1>(gmem_layout_t{})))
  );
  constexpr uint32_t thread_block_size = 128;
  test_cooperative_copy_default<typename TestFixture::mode,
                                TestFixture::max_vec_bits,
                                gmem_layout_t,
                                smem_layout_t,
                                thread_block_size,
                                value_type>();
}

TYPED_TEST(SM70_CuTe_Volta, CooperativeCopyDefaultGSSGSwizzle3)
{
  using value_type = cute::half_t;
  using gmem_layout_t = decltype(make_layout(make_shape(Int<64>{}, Int<64>{})));
  using smem_atom_layout_t = decltype(composition(Swizzle<2, 4, 3> {}, Layout<Shape<_16, _64>, Stride<_64, _1>>{}));
  using smem_layout_t = decltype(tile_to_shape(
      smem_atom_layout_t{},
      make_shape(shape<0>(gmem_layout_t{}), shape<1>(gmem_layout_t{})))
  );
  constexpr uint32_t thread_block_size = 128;
  test_cooperative_copy_default<typename TestFixture::mode,
                                TestFixture::max_vec_bits,
                                gmem_layout_t,
                                smem_layout_t,
                                thread_block_size,
                                value_type>();
}

TYPED_TEST(SM70_CuTe_Volta, CooperativeCopyDefaultGSSGSwizzle4)
{
  using value_type = cute::half_t;
  using gmem_atom_layout_t = decltype(composition(Swizzle<3, 2, 3> {}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
  using smem_layout_t = decltype(make_layout(make_shape(Int<64>{}, Int<64>{})));
  using gmem_layout_t = decltype(tile_to_shape(
      gmem_atom_layout_t{},
      make_shape(shape<0>(smem_layout_t{}), shape<1>(smem_layout_t{})))
  );
  constexpr uint32_t thread_block_size = 128;
  test_cooperative_copy_default<typename TestFixture::mode,
                                TestFixture::max_vec_bits,
                                gmem_layout_t,
                                smem_layout_t,
                                thread_block_size,
                                value_type>();
}
