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
/*! \file
    \brief Tests for device-wide GEMM interface
*/

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>

#include "../../common/cutlass_unit_test.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/complex.h"
#include "testbed_utils.h"

#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/gemm/gemm.h"

#include "cute/int_tuple.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/int.hpp"

namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

enum class ScalarLoc {
  ON_HOST = 0,
  ON_DEVICE = 1
};

enum class VectorBeta {
  DISABLED = 0,
  ENABLED = 1
};

enum class CheckEquality {
  EXACT = 0,
  RELATIVE = 1
};

namespace detail{

// Helper classes that take default data type when
// the Gemm::EpilogueOutputOp does not have ElementCompute
// and ElementScalar.
// (e.g. when Sm90TreeVisitor is used as FusionCallbacks)
template <typename Gemm, typename Default, typename = void>
struct ElementComputeType {
  using Type = Default;
};

template <typename Gemm, typename Default>
struct ElementComputeType<Gemm, Default, std::void_t<typename Gemm::EpilogueOutputOp::ElementCompute>> {
  using Type = typename Gemm::EpilogueOutputOp::ElementCompute;
};

template <typename Gemm, typename Default, typename = void>
struct ElementScalarType {
  using Type = Default;
};

template <typename Gemm, typename Default>
struct ElementScalarType<Gemm, Default, std::void_t<typename Gemm::EpilogueOutputOp::ElementScalar>> {
  using Type = typename Gemm::EpilogueOutputOp::ElementScalar;
};

// The maximum swizzle size to use
//
// This class, like Splits above makes it harder to confuse
// the order of arguments of the various run(...) functions in this file.
class MaxSwizzleSize {
public:
  MaxSwizzleSize() = default;

  template<class IntegralNotBool,
    __CUTE_REQUIRES((std::is_integral_v<IntegralNotBool> &&
      !cute::is_same_v<IntegralNotBool, bool>)) >
  explicit MaxSwizzleSize(IntegralNotBool max_swizzle_size) : max_swizzle_size_(max_swizzle_size) {}
  explicit operator int() const { return max_swizzle_size_; }
private:
  int max_swizzle_size_ = 1;
};

template <typename T>
auto make_iterator(T* ptr) {
  using namespace cute;
  if constexpr (cute::is_subbyte_v<T>) {
    return subbyte_iterator<T>(ptr);
  }
  else {
    return ptr;
  }
}

template<class T>
struct IsDefaultEpilogue {
  static constexpr bool value = false;
};

template<class ...args>
struct IsDefaultEpilogue<cutlass::epilogue::collective::DefaultEpilogue<args...>> {
  static constexpr bool value = true;
};

template<class ...args>
struct IsDefaultEpilogue<cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<args...>> {
  static constexpr bool value = true;
};

// The number of splits to test.
//
// This class makes it harder to confuse the order of arguments
// of the various run(...) functions in this file.  The constructor
// is explicit, so one can't just type 42 (or false, which the
// compiler unhelpfully turns into 0); one has to type Splits(42).
// Splits() picks the default number of splits, 1.
//
// The conversion-to-int operator (operator int()) MUST be explicit!
// Conversion to int MUST require static_cast<int>.
// Otherwise, that defeats a key purpose of this class,
// which is to catch common errors of confusing the order
// of function arguments.
class Splits {
public:
  Splits() = default;

  template<class IntegralNotBool,
    __CUTE_REQUIRES((std::is_integral_v<IntegralNotBool> &&
      !cute::is_same_v<IntegralNotBool, bool>)) >
  explicit Splits(IntegralNotBool splits) : splits_(splits) {}
  explicit operator int() const { return splits_; }
private:
  int splits_ = 1;
};

// The number of iterations to test.
//
// This class, like Splits above makes it harder to confuse
// the order of arguments of the various run(...) functions in this file.
// Iterations() picks the default number of iterations, 20.
class Iterations {
public:
  Iterations() = default;

  template<class IntegralNotBool,
    __CUTE_REQUIRES((std::is_integral_v<IntegralNotBool> &&
      !cute::is_same_v<IntegralNotBool, bool>)) >
  explicit Iterations(IntegralNotBool iterations) : iterations_(iterations) {}
  explicit operator int() const { return iterations_; }
private:
  int iterations_ = 20;
};

template <typename Element, typename Layout>
bool initialize_tensor(
  cutlass::TensorView<Element, Layout> view,
  cutlass::Distribution::Kind dist_kind,
  uint64_t seed) {

  if (dist_kind == cutlass::Distribution::Uniform) {
    double scope_max, scope_min;
    int bits_input = cutlass::sizeof_bits<Element>::value;

    if (bits_input == 1) {
      scope_max = 2;
      scope_min = 0;
    }
    else if (bits_input <= 8) {
        scope_max = 1;
        scope_min = -1;
    }
    else{
      scope_max = 4;
      scope_min = -4;
    }
    cutlass::reference::host::TensorFillRandomUniform(
      view, seed, scope_max, scope_min, 0);
  }

  else if (dist_kind == cutlass::Distribution::Identity) {
    cutlass::reference::host::TensorFillIdentity(view);
  }

  else if (dist_kind == cutlass::Distribution::Gaussian) {
    cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
  }

  else if (dist_kind == cutlass::Distribution::Sequential) {
    cutlass::reference::host::BlockFillSequential(
      view.data(), view.capacity());
  }

  else if (dist_kind == cutlass::Distribution::AllOnes) {
    cutlass::reference::host::TensorFill(view, Element(1));
  }

  else {
    EXPECT_TRUE(false) << "Not implemented";
    return false;
  }

  return true;
}

// Looks at Cute Stride to check Row / Column Major
template<typename Stride>
static constexpr bool is_row_or_col_major(){
  int stride_0 = int(cute::size<0>(Stride{}));
  int stride_1 = int(cute::size<1>(Stride{}));
  int depth = cute::depth(Stride{});
  return ((stride_0 == 1) || (stride_1 == 1)) && (depth == 1);
}


//
// Default MMA input Operands : A , B
//
template<
  class ScheduleType_, 
  class Gemm, 
  class ElementA_ = typename Gemm::GemmKernel::ElementA,
  class ElementB_ = typename Gemm::GemmKernel::ElementB> 
struct HostCollectiveMainloop {
  // Kernel data types
  using ElementA = ElementA_;
  using StrideA  = typename Gemm::GemmKernel::StrideA;
  using ElementB = ElementB_;
  using StrideB  = typename Gemm::GemmKernel::StrideB;
  using ScheduleType = typename Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Schedule;
  using LayoutTagA = cutlass::detail::StrideToLayoutTagA_t<StrideA>;
  using LayoutTagB = cutlass::detail::StrideToLayoutTagB_t<StrideB>;

  using ElementAccumulator = typename Gemm::GemmKernel::ElementAccumulator;
  using ElementScalingFactor = ElementAccumulator;
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
  using EpilogueOutputOp = typename Gemm::EpilogueOutputOp;

  using Arguments = typename Gemm::GemmKernel::MainloopArguments;

  cutlass::ComplexTransform TransformA = Gemm::kTransformA;
  cutlass::ComplexTransform TransformB = Gemm::kTransformB;

  StrideA stride_a;
  StrideB stride_b;

  typename LayoutTagA::Stride stride_factor_A;
  typename LayoutTagB::Stride stride_factor_B;

  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;

  cutlass::HostTensor<ElementA, LayoutTagA> tensor_A;
  cutlass::HostTensor<ElementB, LayoutTagB> tensor_B;
  // Whether to use relative equality checks
  CheckEquality check_relative_equality = CheckEquality::EXACT;

  uint64_t seed;
  static constexpr uint64_t kDefaultSeed = 4096;

  // Note: this limitation comes from testbed / not the library
  static_assert(is_row_or_col_major<StrideA>(),
    "ERROR : A Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<StrideB>(),
    "ERROR : B Layout is neither Row / Column Major)");

  HostCollectiveMainloop(
    CheckEquality check_relative_equality_ = CheckEquality::EXACT,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = kDefaultSeed,
    typename LayoutTagA::Stride stride_factor_A_ = typename LayoutTagA::Stride(),
    typename LayoutTagB::Stride stride_factor_B_ = typename LayoutTagB::Stride()
  ):
    stride_factor_A(stride_factor_A_),
    stride_factor_B(stride_factor_B_),
    init_A(init_A_), init_B(init_B_), seed(seed_),
    check_relative_equality(check_relative_equality_) { }

  template<class ProblemShapeType>
  bool initialize(ProblemShapeType problem_size) {
    //
    // Allocate the GEMM workspace
    //
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::size<0>(problem_shape_MNKL);
    auto N = cute::size<1>(problem_shape_MNKL);
    auto K = cute::size<2>(problem_shape_MNKL);
    auto L = cute::size<3>(problem_shape_MNKL);

    stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));

    // 2.x host tensor does not natively contain a batch stride or coord, so we spoof if by folding it into the outer mode
    auto a_coord = cutlass::make_Coord(M * L, K);
    // Cutlass has Row/Col major refers to MxK times KxN matrix product,
    // so the HostTensorB should be treated as KxN in "coord"'s view
    auto b_coord = cutlass::make_Coord(K, N * L);


    tensor_A.resize(a_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagA>::layout_factory(a_coord, stride_factor_A));
    tensor_B.resize(b_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagB>::layout_factory(b_coord, stride_factor_B));
 
    EXPECT_TRUE(initialize_tensor(tensor_A.host_view(), init_A, seed + 2022));
    EXPECT_TRUE(initialize_tensor(tensor_B.host_view(), init_B, seed + 2021));

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_A.host_view().at({0, 0}) = ElementA(1);
    tensor_B.host_view().at({0, 0}) = ElementB(1);

    tensor_A.sync_device();
    tensor_B.sync_device();

    return true;
  }

  Arguments to_args() {

    Arguments arguments = 
    {
      tensor_A.device_data(), stride_a, tensor_B.device_data(), stride_b
    };
    return arguments;
  }

  auto to_host_args(ProblemShapeType problem_size) {
    using namespace cute;
    //
    // Allocate the GEMM workspace
    //
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::size<0>(problem_shape_MNKL);
    auto N = cute::size<1>(problem_shape_MNKL);
    auto K = cute::size<2>(problem_shape_MNKL);
    auto L = cute::size<3>(problem_shape_MNKL);
    auto A = make_tensor(make_iterator(tensor_A.host_data()),
          make_layout(make_shape(M, K, L), stride_a));
    auto B = make_tensor(make_iterator(tensor_B.host_data()),
        make_layout(make_shape(N, K, L), stride_b));

    cutlass::reference::host::GettMainloopParams<ElementAccumulator, 
                                                 decltype(A), 
                                                 decltype(B)
                                                 > mainloop_params{};

    mainloop_params.A = A;
    mainloop_params.B = B;
    mainloop_params.transform_A = TransformA;
    mainloop_params.transform_B = TransformB;

    return mainloop_params;
  }

  void print_tensors(std::ofstream& file) {
    file << "A =\n" << tensor_A.host_view()
         << "\nB =\n" << tensor_B.host_view();
  }

  template <
    class Element,
    class Layout
  >
  bool equality_check(
    cutlass::TensorView<Element, Layout> const& lhs,
    cutlass::TensorView<Element, Layout> const& rhs) const {

    // Factors used for calculating relative equality. CUTLASS's relative-equality
    // checks in include/cutlass/relatively_equal.h  are inspired by
    // https://floating-point-gui.de/errors/comparison/. This reference suggests using
    // the minimum normal value of a given type as the nonzero_floor.
    Element epsilon(static_cast<Element>(0.1f));
    Element nonzero_floor(std::numeric_limits<Element>::min());

    if constexpr (!cutlass::is_complex<Element>::value) {
      if (check_relative_equality == CheckEquality::RELATIVE) {
        return cutlass::reference::host::TensorRelativelyEquals(
          lhs, rhs, epsilon, nonzero_floor);
      }
      else {
        return cutlass::reference::host::TensorEquals(lhs, rhs);
      }
    }
    else {
      return cutlass::reference::host::TensorEquals(lhs, rhs);
    }
  }

  bool compare_reference(
      cute::Shape<int,int,int,int> problem_shape_MNKL) {
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_A.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_B.host_view()), 0);

    bool passed = true;
    return passed;
  }
};

template<class Gemm>
struct HostCollectiveDefaultEpilogue {
  // fusion types are potentially void if the fusion is not supported
  // helper so we don't try to construct HostTensor with void type
  template <typename T, typename U = uint8_t>
  using non_void_t = cute::conditional_t<cute::is_void_v<T>, U, T>;

  using ScheduleType = typename Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Schedule;
  using kernel   = typename Gemm::GemmKernel;
  using Epilogue = typename kernel::CollectiveEpilogue;

  using ElementD = typename kernel::ElementD;
  using StrideD  = typename kernel::StrideD;
  using ElementC = non_void_t<typename kernel::ElementC, ElementD>;
  using StrideC  = typename kernel::StrideC;

  using FusionOp = typename Gemm::EpilogueOutputOp;

  static_assert(rank(StrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(rank(StrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  static_assert(is_row_or_col_major<StrideC>(),
    "ERROR : C Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<StrideD>(),
    "ERROR : D Layout is neither Row / Column Major)");

  // Deduce Cutlass Layouts (RowMajor & ColumnMajor)
  using LayoutTagC = cutlass::detail::StrideToLayoutTagC_t<StrideC>;
  using LayoutTagD = cutlass::detail::StrideToLayoutTagC_t<StrideD>;
  using LayoutTagScalar = cutlass::layout::PackedVectorLayout; // scalars are size-1 vectors
  using LayoutTagVector = cutlass::layout::PackedVectorLayout;

  using ElementAccumulator = typename kernel::ElementAccumulator;
  using ElementScalingFactor = ElementAccumulator;
  using ProblemShapeType = typename kernel::ProblemShape;
  using ElementCompute = typename ElementComputeType<Gemm, ElementAccumulator>::Type;
  using ElementScalar = typename ElementScalarType<Gemm, ElementCompute>::Type;

  using Arguments = typename Gemm::GemmKernel::EpilogueArguments;

  /// Initialization
  StrideC stride_c;
  StrideD stride_d;

  typename LayoutTagC::Stride stride_factor_C;
  typename LayoutTagD::Stride stride_factor_D;

  cutlass::HostTensor<ElementC, LayoutTagC> tensor_C;
  // Inputs
  ElementScalar alpha;
  ElementScalar beta;

  cutlass::HostTensor<ElementD, LayoutTagD> tensor_D;
  cutlass::HostTensor<ElementD, LayoutTagD> reference_D;

  // Whether to use relative equality checks
  CheckEquality check_relative_equality = CheckEquality::EXACT;
  // Are scalars copied to device memory before kernel launch
  ScalarLoc use_device_scalars = ScalarLoc::ON_HOST;
  // If per-row scale is enabled and this is true, beta is passed as a host scalar instead of device vector
  VectorBeta disable_vector_beta = VectorBeta::DISABLED;

  cutlass::Distribution::Kind init_C;
  uint64_t seed;
  static constexpr uint64_t kDefaultSeed = 4096;

  HostCollectiveDefaultEpilogue(
    CheckEquality check_relative_equality_ = CheckEquality::EXACT,
    ScalarLoc use_device_scalars_ = ScalarLoc::ON_HOST,
    VectorBeta disable_vector_beta_ = VectorBeta::DISABLED,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_scale_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_bias_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = kDefaultSeed
  ): init_C(init_C_), seed(seed_), 
     stride_factor_C(typename LayoutTagC::Stride()), 
     stride_factor_D(typename LayoutTagD::Stride()),
     check_relative_equality(check_relative_equality_),
     use_device_scalars(use_device_scalars_){ }

  bool initialize(ProblemShapeType problem_size, ElementScalar alpha_=1.f, ElementScalar beta_=0.f) {
    // Initialize Epilogue tensors
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    // 2.x host tensor does not natively contain a batch stride or coord, so we spoof if by folding it into the outer mode
    auto c_coord = cutlass::make_Coord(M * L, N);
    tensor_C.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagC>::layout_factory(c_coord, stride_factor_C));
    tensor_D.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D));
    reference_D.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D), false);
    EXPECT_TRUE(initialize_tensor(tensor_C.host_view(), init_C, seed + 2020));
    tensor_C.host_view().at({0, 0}) = ElementC(1);

    cutlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());
    tensor_C.sync_device();
    tensor_D.sync_device();

    alpha = alpha_;
    beta = beta_;

    return true;
  }

  template <
    class Element,
    class Layout
  >
  bool equality_check(
    cutlass::TensorView<Element, Layout> const& lhs,
    cutlass::TensorView<Element, Layout> const& rhs) const {

    // Factors used for calculating relative equality. CUTLASS's relative-equality
    // checks in include/cutlass/relatively_equal.h  are inspired by
    // https://floating-point-gui.de/errors/comparison/. This reference suggests using
    // the minimum normal value of a given type as the nonzero_floor.
    Element epsilon(static_cast<Element>(0.1f));
    Element nonzero_floor(std::numeric_limits<Element>::min());

    if constexpr (!cutlass::is_complex<Element>::value) {
      if (check_relative_equality == CheckEquality::RELATIVE) {
        return cutlass::reference::host::TensorRelativelyEquals(
          lhs, rhs, epsilon, nonzero_floor);
      }
      else {
        return cutlass::reference::host::TensorEquals(lhs, rhs);
      }
    }
    else {
      return cutlass::reference::host::TensorEquals(lhs, rhs);
    }
  }

  bool compare_reference(
      cute::Shape<int,int,int,int> problem_shape_MNKL,
      ElementScalar alpha,
      ElementScalar beta) {
    auto [M, N, K, L] = problem_shape_MNKL;

    tensor_D.sync_host();
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_C.host_view()), 0);

    if (tensor_D.size() > 1) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_D.host_view()), 0);
    }

    if (reference_D.size() > 1) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(reference_D.host_view()), 0);
    }

    bool passed = equality_check(reference_D.host_view(), tensor_D.host_view());
    if(!passed) {
      std::cout<<"D is incorrect"<<std::endl;  
    }
    return passed;
  }

  void print_tensors(std::ofstream& file) {
    file
    << "\nC =\n" << tensor_C.host_view()
    << "\n\nReference =\n" << reference_D.host_view()
    << "\n\nComputed =\n" << tensor_D.host_view();
  }

  Arguments to_args(ProblemShapeType problem_size) {
    Arguments arguments = 
      {
        {alpha, beta},
        tensor_C.device_data(), stride_c, tensor_D.device_data(), stride_d
      };

    return arguments;
  }

  auto to_host_args(ProblemShapeType problem_size) {
    using namespace cute;
    //
    // Allocate the GEMM workspace
    //
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::get<0>(problem_shape_MNKL);
    auto N = cute::get<1>(problem_shape_MNKL);
    auto K = cute::get<2>(problem_shape_MNKL);
    auto L = cute::get<3>(problem_shape_MNKL);
    auto coord_0 = cutlass::make_Coord(0);
    auto C = cute::make_tensor(detail::make_iterator(tensor_C.host_data()),
        cute::make_layout(cute::make_shape(M, N, L), stride_c));
    auto D = cute::make_tensor(detail::make_iterator(reference_D.host_data()),
        cute::make_layout(cute::make_shape(M, N, L), stride_d));

    cutlass::reference::host::GettEpilogueParams<
      ElementScalar,
      ElementScalar,
      ElementAccumulator,
      ElementCompute,
      decltype(C),
      decltype(D)>
        epilogue_params{};

    epilogue_params.C = C;
    epilogue_params.D = D;
    epilogue_params.alpha = alpha;
    epilogue_params.beta = beta;

    return epilogue_params;
  }
};

template<class Gemm>
struct HostCollectiveEpilogue {
  // fusion types are potentially void if the fusion is not supported
  // helper so we don't try to construct HostTensor with void type
  template <typename T, typename U = uint8_t>
  using non_void_t = cute::conditional_t<cute::is_void_v<T>, U, T>;

  using ScheduleType = typename Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Schedule;
  using kernel   = typename Gemm::GemmKernel;
  using Epilogue = typename kernel::CollectiveEpilogue;
  static_assert(IsDefaultEpilogue<Epilogue>::value == false, "Default Epilogue is not supported");

  using ElementD = typename kernel::ElementD;
  using StrideD  = typename kernel::StrideD;
  using ElementC = non_void_t<typename kernel::ElementC, ElementD>;
  using StrideC  = typename kernel::StrideC;

  static_assert(rank(StrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(rank(StrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  static_assert(is_row_or_col_major<StrideC>(),
    "ERROR : C Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<StrideD>(),
    "ERROR : D Layout is neither Row / Column Major)");

  // Deduce Cutlass Layouts (RowMajor & ColumnMajor)
  using LayoutTagC = cutlass::detail::StrideToLayoutTagC_t<StrideC>;
  using LayoutTagD = cutlass::detail::StrideToLayoutTagC_t<StrideD>;
  using LayoutTagScalar = cutlass::layout::PackedVectorLayout; // scalars are size-1 vectors
  using LayoutTagVector = cutlass::layout::PackedVectorLayout;

  using ElementAccumulator = typename kernel::ElementAccumulator;
  using ElementScalingFactor = ElementAccumulator;
  using ProblemShapeType = typename kernel::ProblemShape;

  //
  // FusionOperation derived types/queries
  //
  using EpiloguePolicy = typename Epilogue::DispatchPolicy;
  static constexpr bool IsLegacy =
  cute::is_same_v<
    EpiloguePolicy,
    cutlass::epilogue::Sm90TmaWarpSpecializedBiasElementwise<
      EpiloguePolicy::StagesC, EpiloguePolicy::StagesD, EpiloguePolicy::FragmentSize>
  >;

  using FusionOp = typename Gemm::EpilogueOutputOp;
  static_assert(cute::is_base_of_v<cutlass::epilogue::fusion::FusionOperation, FusionOp>);

  using ElementCompute    = typename FusionOp::ElementCompute;
  using ElementScalar     = typename FusionOp::ElementScalar;
  using ElementBias       = non_void_t<typename FusionOp::ElementBias>;
  using ElementAux        = non_void_t<typename FusionOp::ElementAux>;
  using ElementAmax       = non_void_t<typename FusionOp::ElementAmax>;
  using LayoutTagAux      = non_void_t<typename FusionOp::GmemLayoutTagAux, LayoutTagD>;
  using ActivationFunctor = non_void_t<typename FusionOp::ActivationFn,
                              cutlass::epilogue::thread::Identity<ElementCompute>>;

  static constexpr bool IsBiasEnabled        = FusionOp::IsPerRowBiasSupported;
  static constexpr bool IsDeBiasEnabled      = FusionOp::IsDePerRowBiasSupported;
  static constexpr bool IsPerRowScaleEnabled = FusionOp::IsPerRowScaleSupported;
  static constexpr bool IsScaleFactorEnabled = FusionOp::IsScaleFactorSupported;
  static constexpr bool IsAuxInEnabled       = FusionOp::IsAuxInSupported;
  static constexpr bool IsAuxOutEnabled      = FusionOp::IsAuxOutSupported;
  static constexpr bool IsAbsMaxEnabledD     = FusionOp::IsAbsMaxSupported &&
                                                (cute::is_same_v<ElementD, cutlass::float_e4m3_t> ||
                                                 cute::is_same_v<ElementD, cutlass::float_e5m2_t>);
  static constexpr bool IsAbsMaxEnabledAux   = IsAuxOutEnabled && FusionOp::IsAbsMaxSupported &&
                                                (cute::is_same_v<ElementAux, cutlass::float_e4m3_t> ||
                                                 cute::is_same_v<ElementAux, cutlass::float_e5m2_t>);

  using Arguments = typename Gemm::GemmKernel::EpilogueArguments;

  /// Initialization
  StrideC stride_c;
  StrideD stride_d;

  typename LayoutTagC::Stride stride_factor_C;
  typename LayoutTagD::Stride stride_factor_D;

  // Inputs
  cutlass::HostTensor<ElementScalar, LayoutTagScalar> alpha;
  cutlass::HostTensor<ElementScalar, LayoutTagScalar> beta;
  cutlass::HostTensor<ElementScalar, LayoutTagScalar> scale_A;
  cutlass::HostTensor<ElementScalar, LayoutTagScalar> scale_B;
  cutlass::HostTensor<ElementScalar, LayoutTagScalar> scale_C;
  cutlass::HostTensor<ElementScalar, LayoutTagScalar> scale_D;
  cutlass::HostTensor<ElementScalar, LayoutTagScalar> scale_Aux;
  cutlass::HostTensor<ElementBias  , LayoutTagVector> bias;
  cutlass::HostTensor<ElementC, LayoutTagC> tensor_C;
  cutlass::HostTensor<ElementCompute, LayoutTagScalar> norm_constant;

  // Outputs
  cutlass::HostTensor<ElementAmax, LayoutTagScalar> abs_max_Aux;
  cutlass::HostTensor<ElementAmax, LayoutTagScalar> abs_max_D;
  cutlass::HostTensor<ElementAux , LayoutTagAux   > tensor_Aux;
  cutlass::gemm::TagToStrideC_t<   LayoutTagAux   > stride_Aux;
  cutlass::HostTensor<ElementD, LayoutTagD> tensor_D;
  cutlass::HostTensor<ElementD, LayoutTagD> reference_D;

  // References
  cutlass::HostTensor<ElementBias, LayoutTagVector> reference_dbias;
  cutlass::HostTensor<ElementAux , LayoutTagAux   > reference_Aux;
  cutlass::HostTensor<ElementAmax, LayoutTagScalar> reference_abs_max_Aux;
  cutlass::HostTensor<ElementAmax, LayoutTagScalar> reference_abs_max_D;

  // Whether to use relative equality checks
  CheckEquality check_relative_equality = CheckEquality::EXACT;
  // Are scalars copied to device memory before kernel launch
  ScalarLoc use_device_scalars = ScalarLoc::ON_HOST;
  // If per-row scale is enabled and this is true, beta is passed as a host scalar instead of device vector
  VectorBeta disable_vector_beta = VectorBeta::DISABLED;

  // Random distribution with which to initialize the A/B/C/D/Aux scaling factors
  cutlass::Distribution::Kind init_scale = cutlass::Distribution::Uniform;
  // Random distribution with which to initialize the bias vector
  cutlass::Distribution::Kind init_bias = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind init_C;
  uint64_t seed;
  static constexpr uint64_t kDefaultSeed = 4096;

  HostCollectiveEpilogue(
    CheckEquality check_relative_equality_ = CheckEquality::EXACT,
    ScalarLoc use_device_scalars_ = ScalarLoc::ON_HOST,
    VectorBeta disable_vector_beta_ = VectorBeta::DISABLED,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_scale_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_bias_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = kDefaultSeed
  ): init_scale(init_scale_), init_bias(init_bias_), 
     init_C(init_C_), seed(seed_), 
     stride_factor_C(typename LayoutTagC::Stride()), 
     stride_factor_D(typename LayoutTagD::Stride()),
     check_relative_equality(check_relative_equality_),
     use_device_scalars(use_device_scalars_){ }

  bool initialize(ProblemShapeType problem_size, ElementScalar alpha_=1.f, ElementScalar beta_=0.f) {
    // Initialize Epilogue tensors
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::size<0>(problem_shape_MNKL);
    auto N = cute::size<1>(problem_shape_MNKL);
    auto K = cute::size<2>(problem_shape_MNKL);
    auto L = cute::size<3>(problem_shape_MNKL);

    stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    // 2.x host tensor does not natively contain a batch stride or coord, so we spoof if by folding it into the outer mode
    auto c_coord = cutlass::make_Coord(M * L, N);
    tensor_C.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagC>::layout_factory(c_coord, stride_factor_C));
    tensor_D.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D));
    reference_D.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D), false);
    EXPECT_TRUE(initialize_tensor(tensor_C.host_view(), init_C, seed + 2020));
    tensor_C.host_view().at({0, 0}) = ElementC(1);

    cutlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());
    tensor_C.sync_device();
    tensor_D.sync_device();

    auto scalar_coord = cutlass::make_Coord(1);
    auto col_vector_coord = cutlass::make_Coord(M);
    if constexpr (IsPerRowScaleEnabled) {
      alpha.resize(col_vector_coord);
      EXPECT_TRUE(initialize_tensor(alpha.host_view(), init_scale, seed + 2023));
      if (disable_vector_beta == VectorBeta::DISABLED) {
        beta.resize(scalar_coord, false);
        cutlass::reference::host::TensorFill(beta.host_view(), beta_);
      }
      else {
        beta.resize(col_vector_coord);
        EXPECT_TRUE(initialize_tensor(beta.host_view(), init_scale, seed + 2024));
      }
    }
    else {
      alpha.resize(scalar_coord, (use_device_scalars == ScalarLoc::ON_DEVICE));
      beta.resize(scalar_coord, (use_device_scalars == ScalarLoc::ON_DEVICE));
      cutlass::reference::host::TensorFill(alpha.host_view(), alpha_);
      cutlass::reference::host::TensorFill(beta.host_view(), beta_);
    }
    alpha.sync_device();
    beta.sync_device();

    if constexpr (IsScaleFactorEnabled) {
      scale_A.resize(scalar_coord, (use_device_scalars == ScalarLoc::ON_DEVICE));
      scale_B.resize(scalar_coord, (use_device_scalars == ScalarLoc::ON_DEVICE));
      scale_C.resize(scalar_coord, (use_device_scalars == ScalarLoc::ON_DEVICE));
      scale_D.resize(scalar_coord, (use_device_scalars == ScalarLoc::ON_DEVICE));
      EXPECT_TRUE(initialize_tensor(scale_A.host_view(), init_scale, seed + 2023));
      EXPECT_TRUE(initialize_tensor(scale_B.host_view(), init_scale, seed + 2024));
      EXPECT_TRUE(initialize_tensor(scale_C.host_view(), init_scale, seed + 2025));
      EXPECT_TRUE(initialize_tensor(scale_D.host_view(), init_scale, seed + 2026));
      scale_A.sync_device();
      scale_B.sync_device();
      scale_C.sync_device();
      scale_D.sync_device();
    }

    if constexpr (IsBiasEnabled) {
      bias.resize(col_vector_coord);
      EXPECT_TRUE(initialize_tensor(bias.host_view(), init_bias, seed + 2023));
      bias.sync_device();
    }

    if constexpr (IsDeBiasEnabled) {
      bias.resize(col_vector_coord);
      reference_dbias.resize(col_vector_coord);
      cutlass::reference::host::TensorFill(bias.host_view(), ElementBias(0));
      cutlass::reference::host::TensorFill(reference_dbias.host_view(), ElementBias(0));
      bias.sync_device();
    }

    if constexpr (IsAbsMaxEnabledD) {
      abs_max_D.resize(scalar_coord);
      // ensure in-place device reductions perform their own initialization
      cutlass::reference::host::TensorFill(abs_max_D.host_view(),
                                           CUTLASS_STL_NAMESPACE::numeric_limits<ElementAmax>::max());
      abs_max_D.sync_device();
      reference_abs_max_D.resize(scalar_coord);
      cutlass::reference::host::TensorFill(reference_abs_max_D.host_view(), ElementAmax(0));
    }

    if constexpr (IsAuxInEnabled) {
      auto aux_coord = cutlass::make_Coord(M * L, N);
      auto aux_layout = cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(aux_coord, typename LayoutTagAux::Stride{});
      tensor_Aux.resize(aux_coord, aux_layout);
      EXPECT_TRUE(initialize_tensor(tensor_Aux.host_view(), init_C, seed + 2023));
      tensor_Aux.sync_device();
      stride_Aux = cutlass::make_cute_packed_stride(cutlass::gemm::TagToStrideC_t<LayoutTagAux>{}, cute::make_shape(M, N, L));
    }

    if constexpr (IsAuxOutEnabled) {
      auto aux_coord = cutlass::make_Coord(M * L, N);
      auto aux_layout = cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(aux_coord, typename LayoutTagAux::Stride{});
      tensor_Aux.resize(aux_coord, aux_layout);
      reference_Aux.resize(aux_coord, aux_layout, false);
      tensor_Aux.sync_device();
      stride_Aux = cutlass::make_cute_packed_stride(cutlass::gemm::TagToStrideC_t<LayoutTagAux>{}, cute::make_shape(M, N, L));

      if constexpr (IsScaleFactorEnabled) {
        scale_Aux.resize(scalar_coord, (use_device_scalars == ScalarLoc::ON_DEVICE));
        EXPECT_TRUE(initialize_tensor(scale_Aux.host_view(), init_scale, seed + 2027));
        scale_Aux.sync_device();
      }

      if constexpr (IsAbsMaxEnabledAux) {
        abs_max_Aux.resize(scalar_coord);
        // ensure in-place device reductions perform their own initialization
        cutlass::reference::host::TensorFill(abs_max_Aux.host_view(),
                                             CUTLASS_STL_NAMESPACE::numeric_limits<ElementAmax>::max());
        abs_max_Aux.sync_device();
        reference_abs_max_Aux.resize(scalar_coord);
        cutlass::reference::host::TensorFill(reference_abs_max_Aux.host_view(), ElementAmax(0));
      }
    }

    return true;
  }

  template <
    class Element,
    class Layout
  >
  bool equality_check(
    cutlass::TensorView<Element, Layout> const& lhs,
    cutlass::TensorView<Element, Layout> const& rhs) const {

    // Factors used for calculating relative equality. CUTLASS's relative-equality
    // checks in include/cutlass/relatively_equal.h  are inspired by
    // https://floating-point-gui.de/errors/comparison/. This reference suggests using
    // the minimum normal value of a given type as the nonzero_floor.
    Element epsilon(static_cast<Element>(0.1f));
    Element nonzero_floor(std::numeric_limits<Element>::min());

    if constexpr (!cutlass::is_complex<Element>::value) {
      if (check_relative_equality == CheckEquality::RELATIVE) {
        return cutlass::reference::host::TensorRelativelyEquals(
          lhs, rhs, epsilon, nonzero_floor);
      }
      else {
        return cutlass::reference::host::TensorEquals(lhs, rhs);
      }
    }
    else {
      return cutlass::reference::host::TensorEquals(lhs, rhs);
    }
  }

  bool compare_reference(
      cute::Shape<int,int,int,int> problem_shape_MNKL,
      ElementScalar alpha,
      ElementScalar beta) {
    tensor_D.sync_host();
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_C.host_view()), 0);

    if (tensor_D.size() > 1) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_D.host_view()), 0);
    }

    if (reference_D.size() > 1) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(reference_D.host_view()), 0);
    }

    bool passed = equality_check(reference_D.host_view(), tensor_D.host_view());
    if(!passed) {
      std::cout<<"D is incorrect"<<std::endl;  
    }

    if constexpr (IsAbsMaxEnabledD) {
      abs_max_D.sync_host();
      passed &= equality_check(reference_abs_max_D.host_view(), abs_max_D.host_view());
    }

    if constexpr (IsDeBiasEnabled) {
      bias.sync_host();
      EXPECT_GT(cutlass::reference::host::TensorNorm(bias.host_view()), 0);
      EXPECT_GT(cutlass::reference::host::TensorNorm(reference_dbias.host_view()), 0);
      passed &= equality_check(reference_dbias.host_view(), bias.host_view());
    }

    if constexpr (IsAuxOutEnabled) {
      tensor_Aux.sync_host();
      EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_Aux.host_view()), 0);
      EXPECT_GT(cutlass::reference::host::TensorNorm(reference_Aux.host_view()), 0);
      passed &= equality_check(reference_Aux.host_view(), tensor_Aux.host_view());
      if(!passed) {
        std::cout<<"Aux is incorrect"<<std::endl;  
      }
      if constexpr (IsAbsMaxEnabledAux) {
        abs_max_Aux.sync_host();
        bool tmp =  equality_check(reference_abs_max_Aux.host_view(), abs_max_Aux.host_view());
        if(!tmp) {
          std::cout<<"AbsMax of Aux is incorrect"<<std::endl;  
        }
        passed &= tmp;
      }
    }

    return passed;
  }

  void print_tensors(std::ofstream& file) {
    auto coord_0 = cutlass::make_Coord(0);
    if constexpr (IsScaleFactorEnabled) {
      file
        << ", scale_a: " << scale_A.at(coord_0)
        << ", scale_b: " << scale_B.at(coord_0)
        << ", scale_c: " << scale_C.at(coord_0);
    }
    if constexpr (IsPerRowScaleEnabled) {
      file << "\n\nvalpha = \n" << alpha.host_view();
      file << "\n\nvbeta = \n" << beta.host_view();
    } else {
      file
        << ", alpha: " << alpha.at(coord_0) << ", beta: " << beta.at(coord_0);
    }
    file << "\n\n";

    if constexpr (IsAbsMaxEnabledD) {
      file << "scale_d: " << float(scale_D.at(coord_0));
      file << "\nReference abs_max_D :";
      file << " " << float(reference_abs_max_D.at(coord_0));

      file << "\nComputed abs_max_D :";
      file << " " << float(abs_max_D.at(coord_0));
      file << "\n\n";
    }

    if constexpr (IsAbsMaxEnabledAux) {
      file << "scale_aux: " << float(scale_Aux.at(coord_0));
      file << "\nReference abs_max_Aux :";
      file << " " << float(reference_abs_max_Aux.at(coord_0));

      file << "\nComputed abs_max_Aux :";
      file << " " << float(abs_max_Aux.at(coord_0));
      file << "\n\n";
    }

    if constexpr (IsBiasEnabled) {
      file << "\n\nBias = \n" << bias.host_view();
    }

    if constexpr (IsAuxInEnabled) {
      file << "\n\nAux Input = \n" << tensor_Aux.host_view();
    }

    if constexpr (IsDeBiasEnabled) {
      file << "\n\nReference dBias = \n" << reference_dbias.host_view();
      file << "\n\nComputed dBias = \n" << bias.host_view();
    }

    if constexpr (IsAuxOutEnabled) {
      file
        << "\n\nReference Aux =\n" << reference_Aux.host_view()
        << "\n\nComputed Aux =\n" << tensor_Aux.host_view();
    }
    file
    << "\nC =\n" << tensor_C.host_view()
    << "\n\nReference =\n" << reference_D.host_view()
    << "\n\nComputed =\n" << tensor_D.host_view();

  }

  Arguments to_args(ProblemShapeType problem_size) {
    auto coord_0 = cutlass::make_Coord(0);
    Arguments arguments = 
      {
        {},
        tensor_C.device_data(), stride_c, tensor_D.device_data(), stride_d
      };

    auto &fusion_args = arguments.thread;
    if constexpr (IsLegacy) {
      arguments.thread = {
        alpha.at(coord_0),
        beta.at(coord_0),
        alpha.device_data(),
        beta.device_data()
      };
      arguments.ptr_Bias = bias.device_data();
      arguments.ptr_T = tensor_Aux.device_data();
    }
    else {
      fusion_args.alpha = alpha.at(coord_0);
      fusion_args.beta = beta.at(coord_0);
      fusion_args.alpha_ptr = alpha.device_data();
      fusion_args.beta_ptr = beta.device_data(); // if disable_vector_beta is true this is nullptr

      if constexpr (IsScaleFactorEnabled) {
        fusion_args.scale_a = scale_A.at(coord_0);
        fusion_args.scale_b = scale_B.at(coord_0);
        fusion_args.scale_c = scale_C.at(coord_0);
        fusion_args.scale_d = scale_D.at(coord_0);
        fusion_args.scale_a_ptr = scale_A.device_data();
        fusion_args.scale_b_ptr = scale_B.device_data();
        fusion_args.scale_c_ptr = scale_C.device_data();
        fusion_args.scale_d_ptr = scale_D.device_data();
      }

      if constexpr (IsBiasEnabled) {
        fusion_args.bias_ptr = bias.device_data();
      }

      if constexpr (IsDeBiasEnabled) {
        fusion_args.dbias_ptr = bias.device_data();
      }

      // example of how to set kernel activation arguments
      // see ActivationFunctor::Arguments in activation.h for definition
      // if Arguments doesn't exist then fusion_args.activation is empty
      if constexpr (cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::ScaledGELU_taylor<ElementCompute>>) {
        fusion_args.activation.scale = ElementCompute(1);
      }

      // Treat Clamp as ReLU
      if constexpr (cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::Clamp<ElementCompute>>) {
        fusion_args.activation.lower_bound = 0;
        fusion_args.activation.upper_bound = std::numeric_limits<ElementCompute>::max();
      }

      if constexpr (IsAbsMaxEnabledD) {
        fusion_args.amax_D_ptr = abs_max_D.device_data();
      }

      if constexpr (IsAuxInEnabled) {
        fusion_args.aux_ptr = tensor_Aux.device_data();
        fusion_args.dAux = stride_Aux;
      }

      if constexpr (IsAuxOutEnabled) {
        fusion_args.aux_ptr = tensor_Aux.device_data();
        fusion_args.dAux = stride_Aux;
        if constexpr (IsScaleFactorEnabled) {
          fusion_args.scale_aux = scale_Aux.at(coord_0);
          fusion_args.scale_aux_ptr = scale_Aux.device_data();
        }
        if constexpr (IsAbsMaxEnabledAux) {
          fusion_args.amax_aux_ptr = abs_max_Aux.device_data();
        }
      }

    }

    return arguments;
  }

  auto to_host_args(ProblemShapeType problem_size) {
    using namespace cute;
    //
    // Allocate the GEMM workspace
    //
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::get<0>(problem_shape_MNKL);
    auto N = cute::get<1>(problem_shape_MNKL);
    auto K = cute::get<2>(problem_shape_MNKL);
    auto L = cute::get<3>(problem_shape_MNKL);
    auto coord_0 = cutlass::make_Coord(0);
    auto C = cute::make_tensor(detail::make_iterator(tensor_C.host_data()),
        cute::make_layout(cute::make_shape(M, N, L), stride_c));
    auto D = cute::make_tensor(detail::make_iterator(reference_D.host_data()),
        cute::make_layout(cute::make_shape(M, N, L), stride_d));
    auto Bias = cute::make_tensor(detail::make_iterator(IsDeBiasEnabled ? reference_dbias.host_data() : bias.host_data()),
        cute::make_layout(cute::make_shape(M, cute::_1{})));
    auto Aux = cute::make_tensor(detail::make_iterator(IsAuxInEnabled ? tensor_Aux.host_data() : reference_Aux.host_data()),
        cute::make_layout(cute::make_shape(M, N, L), stride_Aux));
    auto Valpha = cute::make_tensor(detail::make_iterator(alpha.host_data()),
        cute::make_layout(cute::make_shape(M, cute::_1{})));
    auto Vbeta = cute::make_tensor(detail::make_iterator(beta.host_data()),
        cute::make_layout(cute::make_shape(M, cute::_1{})));
    cutlass::reference::host::GettEpilogueParams<
      ElementScalar,
      ElementScalar,
      ElementAccumulator,
      ElementCompute,
      decltype(C),
      decltype(D),
      decltype(Bias),
      decltype(Aux),
      decltype(Valpha),
      decltype(Vbeta),
      ActivationFunctor
    > epilogue_params{};

    epilogue_params.C = C;
    epilogue_params.D = D;
    epilogue_params.alpha = alpha.at(coord_0);
    epilogue_params.beta = beta.at(coord_0);

    if constexpr (IsScaleFactorEnabled) {
      epilogue_params.scale_a = scale_A.at(coord_0);
      epilogue_params.scale_b = scale_B.at(coord_0);
      epilogue_params.scale_c = scale_C.at(coord_0);
      epilogue_params.scale_d = scale_D.at(coord_0);
    }

    if constexpr (IsBiasEnabled or IsDeBiasEnabled) {
      epilogue_params.Bias = Bias;
    }

    if constexpr (IsAbsMaxEnabledD) {
      epilogue_params.abs_max_D = reference_abs_max_D.host_data();
    }

    if constexpr (IsAuxInEnabled) {
      epilogue_params.Aux = Aux;
    }

    if constexpr (IsAuxOutEnabled) {
      epilogue_params.Aux = Aux;
      if constexpr (IsScaleFactorEnabled) {
        epilogue_params.scale_aux = scale_Aux.at(coord_0);
      }
      if constexpr (IsAbsMaxEnabledAux) {
        epilogue_params.abs_max_Aux = reference_abs_max_Aux.host_data();
      }
    }

    if constexpr (IsPerRowScaleEnabled) {
      epilogue_params.Valpha = Valpha;
      if (disable_vector_beta == VectorBeta::ENABLED) {
        epilogue_params.Vbeta = Vbeta;
      }
    }
    return epilogue_params;
  }
};

template <
  typename Gemm,
  template <class T> class ActivationFunctor_ = cutlass::epilogue::thread::Identity,
  bool force_legacy_epilogue = false,
  typename ElementA = typename Gemm::GemmKernel::ElementA,
  typename ElementB = typename Gemm::GemmKernel::ElementB
>
struct TestbedImpl {
  // Kernel data types
  using ScheduleType = typename Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Schedule;
  // All Collective MMA operands are defined by HostCollectiveMainloopType based on the schedule type
  using HostCollectiveMainloopType = HostCollectiveMainloop<ScheduleType, Gemm, ElementA, ElementB>;
  using CollectiveEpilogue = cute::conditional_t<IsDefaultEpilogue<typename Gemm::GemmKernel::CollectiveEpilogue>::value || force_legacy_epilogue, 
                                                HostCollectiveDefaultEpilogue<Gemm>, 
                                                HostCollectiveEpilogue<Gemm>>;
  
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
  using ElementAccumulator = typename Gemm::GemmKernel::ElementAccumulator;
  using ElementCompute = typename ElementComputeType<Gemm, ElementAccumulator>::Type;
  using ElementScalar = typename ElementScalarType<Gemm, ElementCompute>::Type;

  using LayoutTagA = typename HostCollectiveMainloopType::LayoutTagA;
  using LayoutTagB = typename HostCollectiveMainloopType::LayoutTagB;
  using LayoutTagC = typename CollectiveEpilogue::LayoutTagC;
  using LayoutTagD = typename CollectiveEpilogue::LayoutTagD;

  uint32_t sm_count;
  // Used to force multi-wave tests for persistent kernel schedules
  constexpr static int MaxSmCount = 16;
  static constexpr uint64_t kDefaultSeed = 4096;
  static constexpr uint32_t mma_promotion_interval = 4;
  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::RasterOrderOptions;
  using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;

  HostCollectiveMainloopType collective_mma_inputs;
  CollectiveEpilogue collective_epilogue;

  //
  // Methods
  //

  TestbedImpl(
    CheckEquality check_relative_equality_ = CheckEquality::EXACT,
    ScalarLoc use_device_scalars_ = ScalarLoc::ON_HOST,
    VectorBeta disable_vector_beta_ = VectorBeta::DISABLED,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_scale_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_bias_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = kDefaultSeed
  ): collective_mma_inputs(HostCollectiveMainloopType(check_relative_equality_, init_A_, init_B_, seed_)), 
     collective_epilogue(CollectiveEpilogue(check_relative_equality_, use_device_scalars_, disable_vector_beta_, init_C_, init_scale_, init_bias_, seed_)) { }

  TestbedImpl(
    typename LayoutTagA::Stride stride_factor_A_,
    typename LayoutTagB::Stride stride_factor_B_,
    typename LayoutTagC::Stride stride_factor_C_,
    typename LayoutTagD::Stride stride_factor_D_,
    CheckEquality check_relative_equality_ = CheckEquality::EXACT,
    ScalarLoc use_device_scalars_ = ScalarLoc::ON_HOST,
    VectorBeta disable_vector_beta_ = VectorBeta::DISABLED,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_scale_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_bias_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = kDefaultSeed
  ): collective_mma_inputs(HostCollectiveMainloopType(check_relative_equality_, stride_factor_A_, stride_factor_B_, init_A_, init_B_, seed_)),
     collective_epilogue(CollectiveEpilogue(check_relative_equality_, use_device_scalars_, disable_vector_beta_, init_C_, init_scale_, init_bias_, seed_)) { }

  /// Initializes data structures
  bool initialize(ProblemShapeType problem_size, ElementScalar alpha_=1.f, ElementScalar beta_=0.f) {
    collective_mma_inputs.initialize(problem_size);
    collective_epilogue.initialize(problem_size, alpha_, beta_);

    return true;
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(
      cute::Shape<int,int,int,int> problem_shape_MNKL,
      ElementScalar alpha,
      ElementScalar beta)
  {
    auto [M, N, K, L] = problem_shape_MNKL;

    bool passed = collective_mma_inputs.compare_reference(problem_shape_MNKL);
    passed &= collective_epilogue.compare_reference(problem_shape_MNKL, alpha, beta);
    EXPECT_TRUE(passed);
    if (!passed) {
      std::stringstream fname;
      fname << "error_Gemm_device_"
        << M << "x" << N << "x" << K << "x" << L << "_"
        << cute::get<0>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << cute::get<1>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << cute::get<2>(typename Gemm::GemmKernel::TileShape{}) << ".txt";

      std::ofstream file(fname.str());
      file
        << "problem: " << ' ' << M << "x" << N << "x" << K << ", Batch count = " << L
        << ", alpha: " << alpha << ", beta: " << beta << "\n\n";
      
      collective_mma_inputs.print_tensors(file);
      collective_epilogue.print_tensors(file);
    }

    return passed;
  }

  /// Verifies the result is a GEMM
  bool verify(
      ProblemShapeType problem_size,
      ElementScalar alpha,
      ElementScalar beta)
  {
    using namespace cute;
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto mainloop_params = collective_mma_inputs.to_host_args(problem_size);
    auto epilogue_params = collective_epilogue.to_host_args(problem_size);
    
    cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

    bool passed = compare_reference(problem_shape_MNKL, alpha, beta);
    return passed;
  }

	/// Determine if the CUDA device is sufficient to run the kernel
  bool sufficient() {
    //
    // Determine SMEM requirements and waive if not satisfied
    //

    size_t smem_size = static_cast<size_t>(Gemm::GemmKernel::SharedStorageSize);

    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDevice() API call failed.");
    }

    cudaDeviceProp properties;
    result = cudaGetDeviceProperties(&properties, device_idx);
    this->sm_count = properties.multiProcessorCount;

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProperties() failed");
    }

    if (properties.sharedMemPerBlockOptin < smem_size) {
      printf("failed due to smem_size\n");
      printf("hardware smem_size: %d, required smem_size: %d\n\n", int(properties.sharedMemPerBlockOptin), int(smem_size));
      return false;
    }

    return true;
  }

  bool profile(
    ProblemShapeType problem_size,
    int iterations,
    Gemm& gemm_op,
    typename Gemm::Arguments& arguments,
    cutlass::device_memory::allocation<uint8_t>& workspace) {
    int M = cute::size<0>(problem_size);
    int N = cute::size<1>(problem_size);
    int K = cute::size<2>(problem_size);
    int L = 1;
    if constexpr(cute::rank(ProblemShapeType{}) == 4) {
      L = cute::size<3>(problem_size);
    }


    cutlass::Status status;
    //
    // Run the GEMM
    //
    cudaError_t result;

    for (int iter = 0; iter < iterations; ++iter) {
      status = gemm_op(arguments, workspace.get());
      if (status != cutlass::Status::kSuccess) {
        EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);
        return false;
      }
    }

    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      EXPECT_EQ(result, cudaSuccess) << "Error at Kernel Sync.";
      return false;
    }

    return true;
  }

  /// Executes one test
  bool run(
    ProblemShapeType problem_size,
    ElementScalar alpha = ElementScalar(1),
    ElementScalar beta = ElementScalar(0),
    bool profiling = false,
    detail::Iterations iterations = detail::Iterations{},
    RasterOrderOptions raster_order = RasterOrderOptions::Heuristic,
    detail::MaxSwizzleSize max_swizzle = detail::MaxSwizzleSize{},
    detail::Splits splits = detail::Splits{},
    DecompositionMode decomposition_mode = DecompositionMode::Heuristic
    )
  {

    // Fail test if insufficient CUDA device
    if (!sufficient()) {
      std::cout << "Test failed due to insufficient CUDA device." << std::endl;
      return false;
    }

    if (!this->initialize(problem_size, alpha, beta)) {
      std::cerr << "Initialization failed \n";
      return false;
    }

    //
    // Initialize the GEMM operator
    //

    typename Gemm::Arguments arguments;
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    if (not profiling) {
      this->sm_count = std::min(MaxSmCount, cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id));
      hw_info.sm_count = this->sm_count;
    }
    else {
      this->sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
      hw_info.sm_count = this->sm_count;
    }

    typename Gemm::GemmKernel::TileScheduler::Arguments scheduler_args;
    if constexpr (cute::is_same_v<typename Gemm::GemmKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>) {
      scheduler_args = { static_cast<int>(splits), static_cast<int>(max_swizzle), raster_order, decomposition_mode };
    }
    else {
      scheduler_args = { static_cast<int>(max_swizzle), raster_order };
    }
    typename HostCollectiveMainloopType::Arguments mainloop_args;

    mainloop_args = collective_mma_inputs.to_args();

    arguments =
    {
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      mainloop_args,
      collective_epilogue.to_args(problem_size),
      hw_info,
      scheduler_args
    };


    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.can_implement(arguments);

    if (status != cutlass::Status::kSuccess) {
      cudaError_t error = cudaGetLastError();
      std::cerr << "This test is not supported: " << cudaGetErrorString(error) << "\n";
      return true;
    }

    //
    // Run the GEMM
    //

    if (profiling) {
      return profile(problem_size, static_cast<int>(iterations), gemm_op, arguments, workspace);
    }
    else {
      cudaError_t result;
      status = gemm_op.initialize(arguments, workspace.get());
      status = gemm_op.run();
      result = cudaDeviceSynchronize();
      if (result != cudaSuccess) {
        EXPECT_EQ(result, cudaSuccess) << "Error at Kernel Sync.";
        return false;
      }

      EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

      //
      // Verify
      //
      bool passed = this->verify(problem_size, alpha, beta);
      if (!passed) {
        std::cout << "Error : Failed : with alpha: " << alpha << ", beta: " << beta
                  << "\n";
      }

      return passed;
    }
  }
};

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Gemm,
  template <class T> class ActivationFunctor = cutlass::epilogue::thread::Identity,
  bool force_legacy_epilogue = false,
  typename ElementA = typename Gemm::GemmKernel::ElementA,
  typename ElementB = typename Gemm::GemmKernel::ElementB
>
struct Testbed3x {

  using TestBedImpl = typename detail::TestbedImpl<
                        Gemm, 
                        ActivationFunctor, 
                        force_legacy_epilogue, 
                        ElementA, 
                        ElementB
                        >;
  using Kernel      = typename Gemm::GemmKernel;
  using Epilogue    = typename Gemm::GemmKernel::CollectiveEpilogue;

  using ElementAccumulator   = typename TestBedImpl::ElementAccumulator;
  using ElementCompute       = typename TestBedImpl::ElementCompute;
  using ElementScalar        = typename TestBedImpl::ElementScalar;

  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::RasterOrderOptions;
  using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;

  // Detail Implementation
  TestBedImpl impl_;

  //
  // Methods
  //
  Testbed3x(
      CheckEquality check_relative_equality_ = CheckEquality::EXACT,
      ScalarLoc use_device_scalars_ = ScalarLoc::ON_DEVICE,
      VectorBeta disable_vector_beta_ = VectorBeta::DISABLED,
      cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_scale_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_bias_ = cutlass::Distribution::Uniform,
      uint64_t seed_ = TestBedImpl::kDefaultSeed)
      : impl_(check_relative_equality_, use_device_scalars_, disable_vector_beta_, init_A_, init_B_, init_C_, init_scale_, init_bias_, seed_) {}

  /// Executes one test
  bool run(
   typename TestBedImpl::ProblemShapeType problem_size,
    ElementScalar alpha = ElementScalar(1),
    ElementScalar beta = ElementScalar(0),
    RasterOrderOptions raster_order = RasterOrderOptions::Heuristic,
    detail::MaxSwizzleSize max_swizzle = detail::MaxSwizzleSize{},
    detail::Splits splits = detail::Splits{},
    DecompositionMode decomposition_mode = DecompositionMode::Heuristic,
    bool profiling = false,
    detail::Iterations iterations = detail::Iterations{}
    )
  {
    return impl_.run(
        problem_size, alpha, beta, profiling, iterations, raster_order, max_swizzle, splits, decomposition_mode
        );
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
bool TestGemmPerf3x(int iterations = 20) {
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
  using ElementAccumulator = typename Gemm::GemmKernel::ElementAccumulator;
  using ElementScalar = ElementAccumulator;
  bool passed = true;
  using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::RasterOrderOptions;

  std::vector<int> problem_size_m = { 4608 };
  std::vector<int> problem_size_n = { 4608 };
  std::vector<int> problem_size_k = { 8192 };

  Testbed3x<Gemm> testbed;

  for (int m : problem_size_m) {
    for (int n : problem_size_n) {
      for (int k : problem_size_k) {
        ProblemShapeType problem_size;
        if constexpr (cute::rank(ProblemShapeType{}) == 4) {
          problem_size = ProblemShapeType{m, n, k, /* l */ 1};
        }
        else {
          problem_size = ProblemShapeType{m, n, k};
        }

        passed = testbed.run(
          problem_size,
          cutlass::from_real<ElementScalar>(1),
          cutlass::from_real<ElementScalar>(0),
          RasterOrderOptions{}, detail::MaxSwizzleSize(1), detail::Splits{1}, DecompositionMode{},
          true, // profiling
          detail::Iterations{iterations});

        if (!passed) {
          return false;
        }
      }
    }
  }

  return true;
}

template <
  typename Gemm,
  template <class T> class ActivationFunctor = cutlass::epilogue::thread::Identity
>
bool TestAll(double alpha = 1.0, double beta = 0.0, CheckEquality check_relative_equality = CheckEquality::RELATIVE) {
  using ElementScalar = typename Gemm::EpilogueOutputOp::ElementScalar;
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  Testbed3x<Gemm, ActivationFunctor> testbed(check_relative_equality, ScalarLoc::ON_HOST, VectorBeta::DISABLED);

  int max_alignment = std::max(Gemm::kAlignmentA, Gemm::kAlignmentB);
  std::vector<int> problem_size_m = {max_alignment, 512 - 3 * max_alignment};
  std::vector<int> problem_size_n = {max_alignment, 512 - 2 * max_alignment};

  if constexpr (cute::is_same_v<typename Gemm::GemmKernel::DispatchPolicy::Schedule,
                cutlass::gemm::KernelTmaWarpSpecializedPingpong>) {
    problem_size_m.push_back(768);
    problem_size_n.push_back(768);
  }

  constexpr int Stages = Gemm::GemmKernel::DispatchPolicy::Stages;
  constexpr int TileShapeK = cute::size<2>(typename Gemm::GemmKernel::TileShape{});

  std::vector<int> problem_size_k = {max_alignment, TileShapeK * (Stages + 1) - max_alignment};

  using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
  std::vector<DecompositionMode> decomposition_modes = {DecompositionMode::Heuristic};
  std::vector problem_splits = {detail::Splits{1}};
  static constexpr bool UsesStreamKScheduler = cute::is_same_v<typename Gemm::GemmKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>;
  if constexpr (UsesStreamKScheduler) {
    problem_splits.push_back(detail::Splits{2});
    problem_splits.push_back(detail::Splits{3});

    decomposition_modes.push_back(DecompositionMode::DataParallel);
    decomposition_modes.push_back(DecompositionMode::SplitK);
    decomposition_modes.push_back(DecompositionMode::StreamK);

    // Use larger K sizes for stream-K tests
    static constexpr int min_tiles_per_sk_unit = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::min_iters_per_sk_unit_;
    problem_size_k = {TileShapeK * min_tiles_per_sk_unit, TileShapeK * 3 * min_tiles_per_sk_unit - max_alignment};
  }

  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::RasterOrderOptions;
  std::vector<RasterOrderOptions> raster_orders = {RasterOrderOptions::AlongM, RasterOrderOptions::AlongN};
  std::vector max_swizzle_sizes{detail::MaxSwizzleSize{1}, detail::MaxSwizzleSize{4}};

  bool passed = true;

  for (int m : problem_size_m) {
    for (int n : problem_size_n) {
      for (int k : problem_size_k) {
        for (auto raster_order : raster_orders) {
          for (auto max_swizzle_size : max_swizzle_sizes) {
            for (DecompositionMode decomp_mode : decomposition_modes) {

              std::vector problem_splits = {detail::Splits{1}};
              if (decomp_mode == DecompositionMode::Heuristic || decomp_mode == DecompositionMode::SplitK) {
                auto max_splits = (k + TileShapeK - 1) / TileShapeK;
                if (max_splits > 2) {
                  problem_splits.push_back(detail::Splits{2});
                }
                if (max_splits > 3) {
                  problem_splits.push_back(detail::Splits{3});
                }

                problem_splits.push_back(detail::Splits{max_splits});

                // Test the case in which we ask for more splits than there are K tiles in the GEMM. In this
                // case, split-K will fall back to a splitting factor of `max_splits`.
                problem_splits.push_back(detail::Splits{max_splits + 1});
              }
              for (auto splits : problem_splits) {
                ProblemShapeType problem_size;
                if constexpr (cute::rank(ProblemShapeType{}) == 4) {
                  problem_size = ProblemShapeType{m, n, k, /* l */ 1};
                }
                else {
                  problem_size = ProblemShapeType{m, n, k};
                }

                passed = testbed.run(
                  problem_size,
                  cutlass::from_real<ElementScalar>(alpha),
                  cutlass::from_real<ElementScalar>(beta),
                  raster_order,
                  max_swizzle_size,
                  splits,
                  decomp_mode
                );

                if (!passed) {
                  std::cout << __FILE__ << ':' << __LINE__ << " : GEMM MNK " << m << " " << n << " " << k << " FAILED.\n";
                  return false;
                }
              } // splits
            } // decomposition_mode
          } // max_swizzle_size
        } // raster_order
      } // k
    } // n
  } // m

  // if we do support batched GEMM, just run one test on it to save on test time
  if constexpr (cute::rank(ProblemShapeType{}) == 4) {
    auto problem_size = ProblemShapeType{256 + max_alignment, 256 + max_alignment, 160 + max_alignment, /* l */ 3};
    passed = testbed.run(
      problem_size,
      cutlass::from_real<ElementScalar>(alpha),
      cutlass::from_real<ElementScalar>(beta)
    );

    if (!passed) {
      return false;
    }
  }

  return passed;
}

template <typename Gemm>
bool TestAllBiasElementwise(double alpha = 1.0, double beta = 0.0, CheckEquality check_relative_equality = CheckEquality::EXACT) {
  return TestAll<Gemm>(alpha, beta, check_relative_equality);
}

} // namespace device
} // namespace gemm
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////
