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
/*! \file
    \brief Testbed and host reference for EVT unittest
*/


#pragma once
#include "gemm_testbed_3x.hpp" 

namespace test {
namespace gemm {
namespace device {

/// Host-side tapply, tapply in cute is HOST_DEVICE
template <class T, class F, class G, int... I>
constexpr auto
tapply(T&& t, F&& f, G&& g, cute::seq<I...>)
{
  return g(f(std::get<I>(static_cast<T&&>(t)))...);
}


/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT: Base class for EVT Node

template <
  typename Gemm_
>
class HostEVTNodeBase {
public:
  using Gemm = Gemm_;
  using TestBedImpl = typename detail::TestbedImpl<Gemm, cutlass::epilogue::thread::Identity, true>;
  using Kernel = typename Gemm::GemmKernel;
  using Epilogue = typename Kernel::CollectiveEpilogue;
  using ElementCompute = typename TestBedImpl::ElementCompute;
  using ElementScalar = typename TestBedImpl::ElementScalar;
  using ElementAccumulator = typename Kernel::ElementAccumulator;
  using ElementC = typename Kernel::ElementC;
  using ElementD = typename Kernel::ElementD;

  using LayoutTagC = typename TestBedImpl::LayoutTagC;
  using LayoutTagD = typename TestBedImpl::LayoutTagD;
private:
  bool _check_relative_equality;
  // Factors used for calculating relative equality. These default
  // values are borrowed from those used by default in the CUTLASS
  // profiler for performing relative equality checks.
  float _epsilon = 0.05f;
  float _nonzero_floor = 1.0f / 256.0f;

public:
  HostEVTNodeBase(){}
  HostEVTNodeBase(bool check_relative_equality):
    _check_relative_equality(check_relative_equality) { }


  template <
    class Element,
    class Layout
  >
  bool equality_check(
    cutlass::TensorView<Element, Layout> const& lhs,
    cutlass::TensorView<Element, Layout> const& rhs) const {
    if (_check_relative_equality) {
      return cutlass::reference::host::TensorRelativelyEquals(
        lhs, rhs, Element(_epsilon), Element(_nonzero_floor)
      );
    }
    else {
      return cutlass::reference::host::TensorEquals(lhs, rhs);
    }
  }

  void* get_tensor_C_ptr() {
    return nullptr;
  }

  void* get_tensor_D_ptr() {
    return nullptr;
  }

  bool compare_reference(std::stringstream& error_ss) {
    return true;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT - Accumulator

template <
  typename Gemm
>
class HostAccumulator: public HostEVTNodeBase<Gemm> {
public:
  using Base = HostEVTNodeBase<Gemm>;
  using TestBedImpl = typename Base::TestBedImpl;
  using ElementAccumulator = typename Base::ElementAccumulator;
  using ElementCompute = typename Base::ElementCompute;

  struct Arguments { };

private:
  cutlass::NumericConverter<ElementCompute, ElementAccumulator> accumulator_converter;
public:
  HostAccumulator(){}
  template<typename ProblemShapeType>
  HostAccumulator(ProblemShapeType problem_size, TestBedImpl impl, bool check_relative_equality=false)
    :Base(check_relative_equality) {}

  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc) {
    
    return accumulator_converter(acc);
  }

  Arguments get_arguments() {
    return Arguments{};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT - Scalar Broadcast

template <
  typename Gemm,
  int Value,
  int BroadcastCount = 1,
  template <class> class ReductionFn = cutlass::multiplies
>
class HostScalarBroadcast : public HostEVTNodeBase<Gemm> {
public:
  using Base = HostEVTNodeBase<Gemm>;
  using ElementCompute = typename Base::ElementCompute;

  struct Arguments {
    ElementCompute scalar[BroadcastCount] = {0};
    ElementCompute const* scalar_ptrs[BroadcastCount] = { nullptr };
    cute::Stride<cute::_0,cute::_0,cute::_0> dScalar{};
  };
private:
  ElementCompute _scalar{};
public:
  HostScalarBroadcast(){}

  template<typename ProblemShapeType, typename TestBedImpl>
  HostScalarBroadcast(ProblemShapeType problem_size, TestBedImpl impl, bool check_relative_equality=false)
    : Base(check_relative_equality), _scalar(ElementCompute(Value)) {}
  
  template <class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc) {
    
    return _scalar;
  }

  bool compare_reference(std::stringstream& error_ss) {
    error_ss << "Scalar: " << float(_scalar) << "\n\n";
    return true;
  }

  Arguments get_arguments() {
    if constexpr (BroadcastCount == 1)
      return Arguments{{_scalar}, {nullptr}};
    else if constexpr (BroadcastCount == 2)
      return Arguments{{_scalar, _scalar}, {nullptr, nullptr}};
    else if constexpr (BroadcastCount == 3)
      return Arguments{{_scalar, _scalar, _scalar}, {nullptr, nullptr, nullptr}};
    else
      return Arguments{{_scalar}, {nullptr}};
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT - Row Broadcast
template <
  typename Gemm,
  typename ElementBias_=void
>
class HostRowBroadcast: public HostEVTNodeBase<Gemm> {
public:
  using Base = HostEVTNodeBase<Gemm>;
  using ElementBias = std::conditional_t<std::is_void_v<ElementBias_>,
    typename Base::ElementC,
    ElementBias_>;
  
  using TestBedImpl = typename Base::TestBedImpl;
  using ElementCompute = typename Base::ElementCompute;
  using LayoutTagVector = cutlass::layout::PackedVectorLayout;
  
  struct Arguments {
    ElementBias const* ptr_row = nullptr;
    ElementBias null_default = ElementBias(0);
    cute::Stride<cute::_0,cute::_1,cute::_0> dRow = {};
  };
private:
  cutlass::NumericConverter<ElementCompute, ElementBias> _bias_converter;
  cutlass::HostTensor<ElementBias, LayoutTagVector> _bias;
  int _N;
  TestBedImpl impl_;
public:
  HostRowBroadcast(){}
  template<typename ProblemShapeType>
  HostRowBroadcast(ProblemShapeType problem_size, TestBedImpl impl, bool check_relative_equality=false)
    : Base(check_relative_equality), impl_(impl) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    _N = cute::get<1>(problem_shape_MNKL);
    _bias.resize(cutlass::Coord<1>(_N));
    
    EXPECT_TRUE(
      detail::initialize_tensor(
        _bias.host_view(), cutlass::Distribution::Uniform, 
        impl_.collective_mma_inputs.seed + 2023
      )
    );
    _bias.sync_device();
  }

  template <class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc) {
    auto TensorBias = cute::make_tensor(_bias.host_data(),
      cute::make_layout(cute::make_shape(cute::_1{}, _N)));
    
    return _bias_converter(TensorBias(1, n + n_b));
  }

  bool compare_reference(std::stringstream& error_ss) {
    error_ss
      << "PerColumnBias = \n" << _bias.host_view() << "\n\n";
    return true;
  }

  Arguments get_arguments() {
    return {_bias.device_data()};
  }

};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT - Column Broadcast
template <
  typename Gemm,
  typename ElementBias_=void
>
class HostColBroadcast: public HostEVTNodeBase<Gemm> {
public:
  using Base = HostEVTNodeBase<Gemm>;
  using ElementBias = std::conditional_t<std::is_void_v<ElementBias_>,
    typename Base::ElementC,
    ElementBias_>;
  
  using TestBedImpl = typename Base::TestBedImpl;
  using ElementCompute = typename Base::ElementCompute;
  using LayoutTagVector = cutlass::layout::PackedVectorLayout;
  
  struct Arguments {
    ElementBias const* ptr_row = nullptr;
    ElementBias null_default = ElementBias(0);
    cute::Stride<cute::_1,cute::_0,cute::_0> dRow = {};
  };
private:
  cutlass::NumericConverter<ElementCompute, ElementBias> _bias_converter;
  cutlass::HostTensor<ElementBias, LayoutTagVector> _bias;
  int _M;
  TestBedImpl impl_;
public:
  HostColBroadcast(){}
  template<typename ProblemShapeType>
  HostColBroadcast(ProblemShapeType problem_size, TestBedImpl impl, bool check_relative_equality=false)
    : Base(check_relative_equality), impl_(impl) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    _M = cute::get<0>(problem_shape_MNKL);
    _bias.resize(cutlass::Coord<1>(_M));
    
    EXPECT_TRUE(
      detail::initialize_tensor(
        _bias.host_view(), cutlass::Distribution::Uniform, 
        impl_.collective_mma_inputs.seed + 2023
      )
    );
    _bias.sync_device();
  }

  template <class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc) {
    auto TensorBias = cute::make_tensor(_bias.host_data(),
      cute::make_layout(cute::make_shape(_M, cute::_1{})));
    
    return _bias_converter(TensorBias(m + m_b, 1));
  }

  bool compare_reference(std::stringstream& error_ss) {
    error_ss
      << "PerRowBias = \n" << _bias.host_view() << "\n\n";
    return true;
  }

  Arguments get_arguments() {
    return {_bias.device_data()};
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT - Aux Load

template <
  typename Gemm,
  bool isC=false,
  typename ElementAuxLoad_=void,
  typename LayoutTagAux_=void
>
class HostAuxLoad: public HostEVTNodeBase<Gemm> {
public:
  using ElementAuxLoad = std::conditional_t<std::is_void_v<ElementAuxLoad_>,
    typename HostEVTNodeBase<Gemm>::ElementC,
    ElementAuxLoad_>;
  using LayoutTagAux = std::conditional_t<std::is_void_v<LayoutTagAux_>,
    typename HostEVTNodeBase<Gemm>::LayoutTagC,
    LayoutTagAux_>;

  using Base = HostEVTNodeBase<Gemm>;
  using TestBedImpl = typename Base::TestBedImpl;
  using ElementCompute = typename Base::ElementCompute;

  using StrideAux = cutlass::gemm::TagToStrideC_t<LayoutTagAux>;
  struct Arguments_Aux {
    ElementAuxLoad const *ptr_aux = nullptr;
    ElementAuxLoad null_default = ElementAuxLoad(0);
    StrideAux dAux = {};
  };

  struct Arguments_C {};

  using Arguments = cute::conditional_t<isC, Arguments_C, Arguments_Aux>;

private:
  cutlass::NumericConverter<ElementCompute, ElementAuxLoad> _aux_load_converter;
  cutlass::HostTensor<ElementAuxLoad, LayoutTagAux> _tensor_aux_load;

  int _M, _N, _L;

  TestBedImpl impl_;

  StrideAux _stride_aux;
public:
  HostAuxLoad(){}
  template<typename ProblemShapeType>
  HostAuxLoad(ProblemShapeType problem_size, TestBedImpl impl, bool check_relative_equality=false)
    : Base(check_relative_equality), impl_(impl) {
    auto problem_shape_NMKL = cute::append<4>(problem_size, 1);
    auto [_M, _N, K, _L] = problem_shape_NMKL;
    auto aux_coord = cutlass::make_Coord(_M * _L, _N);
    _tensor_aux_load.resize(
      aux_coord, 
      cutlass::layout::Affine2Layout_Factory<LayoutTagAux>::layout_factory(
        aux_coord, typename LayoutTagAux::Stride()
      )
    );
    EXPECT_TRUE(
      detail::initialize_tensor(
        _tensor_aux_load.host_view(), 
        cutlass::Distribution::Uniform, 
        impl_.collective_mma_inputs.seed + 2023
      )
    );
    _tensor_aux_load.sync_device();
    _stride_aux = cutlass::make_cute_packed_stride(StrideAux{}, cute::make_shape(_M, _N, _L));
  }

  template <class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc) {

    
    auto TensorAuxLoad = cute::make_tensor(_tensor_aux_load.host_data(),
      cute::make_layout(cute::make_shape(_M, _N, _L), _stride_aux));
    return _aux_load_converter(TensorAuxLoad(m + m_b, n + n_b, l));
  }

  bool compare_reference(std::stringstream& error_ss) {
    if constexpr (!isC) {
      error_ss
        << "AuxLoad = \n" << _tensor_aux_load.host_view()<< "\n\n";
    }
    return true;
  }

  void* get_tensor_C_ptr() {
    if constexpr (isC) {
      return static_cast<void*>(_tensor_aux_load.device_data());
    } else {
      return nullptr;
    }
  }

  Arguments get_arguments() {
    if constexpr (isC)
      return {};
    else
      return {_tensor_aux_load.device_data(), ElementAuxLoad(0), _stride_aux};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT - Compute

template<typename T>
T* findNonNullPtr(T* first_ptr) {
  return first_ptr;
}

template <typename T, typename... Args>
T* findNonNullPtr(T* first_ptr, Args... args) {
  if (first_ptr) {
    return first_ptr;
  }
  return findNonNullPtr(args...);
}

template <
  typename Gemm,
  template <class> class ComputeOp_
>
class HostCompute: public HostEVTNodeBase<Gemm> {
public:
  using Base = HostEVTNodeBase<Gemm>;
  using ElementCompute = typename Base::ElementCompute;
  using ComputeOp = ComputeOp_<ElementCompute>;

  struct Arguments {
    struct OpArgs {} op;
  };
private:
  ComputeOp _op;
public:
  HostCompute(){}
  template <typename ProblemShapeType, typename TestBedImpl>
  HostCompute(ProblemShapeType problem_size, TestBedImpl impl, bool check_relative_equality=false):
    Base(check_relative_equality) { }

  template <class ElementAccumulator, typename... Args>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc, Args... frg_inputs) {
    return _op(frg_inputs...);
  }

  Arguments get_arguments(){
    return {};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT - Unary Compute

template <
  typename Gemm,
  template <class> class ComputeOp_,
  typename Child0
>
class HostUnaryCompute: public HostEVTNodeBase<Gemm> {
public:

  using Base = HostEVTNodeBase<Gemm>;
  using ElementCompute = typename Base::ElementCompute;
  using ComputeOp = ComputeOp_<ElementCompute>;

  struct Arguments {
    typename Child0::Arguments child_0_args; 
    struct OpArgs {} op;
  };
private:
  ComputeOp _op;
  Child0 _child_0;
public:
  HostUnaryCompute(){}
  template <typename ProblemShapeType, typename TestBedImpl>
  HostUnaryCompute(ProblemShapeType problem_size, TestBedImpl impl, bool check_relative_equality=false):
    Base(check_relative_equality),
    _child_0(problem_size, impl, check_relative_equality) { }

  template <class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc) {
    ElementCompute child_0_result = _child_0.visit(m, n, l, m_b, n_b, acc);
    return _op(child_0_result);
  }

  Arguments get_arguments(){
    return {
      _child_0.get_arguments(),
      {},
    };
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT - Aux Store

template <
  typename Gemm,
  bool isD=false,
  class ElementAuxStore_=void,
  typename LayoutTagAux_=void
>
class HostAuxStore: public HostEVTNodeBase<Gemm> {
public:
  using ElementAuxStore = std::conditional_t<std::is_void_v<ElementAuxStore_>,
    typename HostEVTNodeBase<Gemm>::ElementD,
    ElementAuxStore_>;
  using LayoutTagAux = std::conditional_t<std::is_void_v<LayoutTagAux_>,
    typename HostEVTNodeBase<Gemm>::LayoutTagD,
    LayoutTagAux_>;

  using Base = HostEVTNodeBase<Gemm>;
  using TestBedImpl = typename Base::TestBedImpl;
  using ElementCompute = typename Base::ElementCompute;

  using StrideAux = cutlass::gemm::TagToStrideC_t<LayoutTagAux>;
  struct Arguments_Aux {
    struct OpArgs {
      ElementAuxStore* ptr_aux = nullptr;
      StrideAux dAux = {};
    } op;
  };

  struct Arguments_D {};

  using Arguments = cute::conditional_t<isD, Arguments_D, Arguments_Aux>;


private:
  cutlass::NumericConverter<ElementAuxStore, ElementCompute> destination_converter;
  cutlass::HostTensor<ElementAuxStore, LayoutTagAux> _tensor_aux_store;
  cutlass::HostTensor<ElementAuxStore, LayoutTagAux> _reference_aux_store;
  int _M, _N, _L;
  TestBedImpl impl_;
  StrideAux _stride_aux;
public:
  HostAuxStore(){}
  template <typename ProblemShapeType>
  HostAuxStore(ProblemShapeType problem_size, TestBedImpl impl, bool check_relative_equality=false):
    Base(check_relative_equality),
    impl_(impl) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [_M, _N, K, _L] = problem_shape_MNKL;
    auto aux_coord = cutlass::make_Coord(_M * _L, _N);
    _tensor_aux_store.resize(
      aux_coord, 
      cutlass::layout::Affine2Layout_Factory<LayoutTagAux>::layout_factory(
        aux_coord, typename LayoutTagAux::Stride()
      )
    );

    _reference_aux_store.resize(
      aux_coord,
      cutlass::layout::Affine2Layout_Factory<LayoutTagAux>::layout_factory(
        aux_coord, typename LayoutTagAux::Stride()
      )
    );
    _tensor_aux_store.sync_device();
    _stride_aux = cutlass::make_cute_packed_stride(StrideAux{}, cute::make_shape(_M, _N, _L));
  }

  template <class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc, ElementCompute child_0_result) {

    auto TensorAuxStore = cute::make_tensor(static_cast<ElementAuxStore*>(_reference_aux_store.host_data()),
      cute::make_layout(cute::make_shape(_M, _N, _L), _stride_aux));
    TensorAuxStore(m + m_b, n + n_b, l) = destination_converter(child_0_result);
    return child_0_result;
  }

  bool compare_reference(std::stringstream& error_ss) {
    // Verify the store node
    _tensor_aux_store.sync_host();

    bool equal = this->equality_check(_reference_aux_store.host_view(), _tensor_aux_store.host_view());
    if (!equal) {
      error_ss 
        << "\n\nReference =\n" << _reference_aux_store.host_view()
        << "\n\nComputed =\n" << _tensor_aux_store.host_view() << "\n\n";
    }
    return equal;
  }

  void* get_tensor_D_ptr() {
    if constexpr (isD) 
      return static_cast<void*>(_tensor_aux_store.device_data());
    else
      return nullptr;
  }

  Arguments get_arguments() {
    if constexpr (isD) {
      return {};
    } else {
      return {_tensor_aux_store.device_data(), _stride_aux};
    }
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT - Row Reduce

template <
  typename Gemm,
  template <class> class ReduceFn,
  typename ElementReduce
>
class HostRowReduce: public HostEVTNodeBase<Gemm> {
public:
  using Base = HostEVTNodeBase<Gemm>;
  using TestBedImpl = typename Base::TestBedImpl;
  using ElementCompute = typename Base::ElementCompute;
  using ElementOutput = typename Base::ElementD;
  using LayoutTagVector = cutlass::layout::PackedVectorLayout;

  struct Arguments {
    struct OpArgs {
      ElementReduce* ptr_row = nullptr;
      ElementCompute reduce_identity = 0;
      cute::Stride<cute::_0, cute::_1, cute::_0> dRow = {};
    } op;
  };

private:
  cutlass::NumericConverter<ElementReduce, ElementCompute> destination_converter;
  cutlass::HostTensor<ElementReduce, LayoutTagVector> _tensor_row_reduce;
  cutlass::HostTensor<ElementCompute, LayoutTagVector> _reduce_buffer;
  cutlass::HostTensor<ElementReduce, LayoutTagVector> _reference_row_reduce;
  int _N;
  TestBedImpl impl_;
  ReduceFn<ElementCompute> reduce_fn;
public:
  HostRowReduce(){}
  template <typename ProblemShapeType>
  HostRowReduce(ProblemShapeType problem_size, TestBedImpl impl, bool check_relative_equality=false):
    Base(check_relative_equality),
    impl_(impl) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    _N = cute::get<1>(problem_shape_MNKL);
    _tensor_row_reduce.resize(cutlass::Coord<1>(_N));
    _reference_row_reduce.resize(cutlass::Coord<1>(_N));
    _reduce_buffer.resize(cutlass::Coord<1>(_N));

    _tensor_row_reduce.sync_device();
  }

  template <class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc, ElementCompute child_0_result) {
    auto TensorRowReduce = cute::make_tensor(_reduce_buffer.host_data(),
      cute::make_layout(cute::make_shape(cute::_1{}, _N)));
    TensorRowReduce(1, n + n_b) = reduce_fn(TensorRowReduce(1, n + n_b), child_0_result);
    return child_0_result;
  }

  bool compare_reference(std::stringstream& error_ss) {
    // Verify the store node
    _tensor_row_reduce.sync_host();

    auto TensorRowReduce = cute::make_tensor(_reference_row_reduce.host_data(),
      cute::make_layout(cute::make_shape(cute::_1{}, _N)));
    
    auto TensorReduceBuffer = cute::make_tensor(_reduce_buffer.host_data(),
      cute::make_layout(cute::make_shape(cute::_1{}, _N)));

    // Filling the reference tensor with the reduce buffer
    for (int n = 0; n < _N; n ++) {
      TensorRowReduce(1, n) = destination_converter(TensorReduceBuffer(1, n));
    }

    bool equal = this->equality_check(_reference_row_reduce.host_view(), _tensor_row_reduce.host_view());
    if (!equal) {
      error_ss 
        << "\n\nRow Reduce Reference =\n" << _reference_row_reduce.host_view()
        << "\n\nRow Reduce Computed =\n" << _tensor_row_reduce.host_view() << "\n\n";
    }
    return equal;
  }

  Arguments get_arguments() {
    return {_tensor_row_reduce.device_data()};
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT - Column Reduce

template <
  typename Gemm,
  template <class> class ReduceFn,
  typename ElementReduce
>
class HostColumnReduce: public HostEVTNodeBase<Gemm> {
public:
  using Base = HostEVTNodeBase<Gemm>;
  using TestBedImpl = typename Base::TestBedImpl;
  using ElementCompute = typename Base::ElementCompute;
  using ElementOutput = typename Base::ElementD;
  using LayoutTagVector = cutlass::layout::PackedVectorLayout;

  struct Arguments {
    struct OpArgs {
      ElementReduce* ptr_col = nullptr;
      ElementCompute reduce_identity = 0;
      cute::Stride<cute::_1, cute::_0, cute::_0> dRow = {};
    } op;
  };

private:
  cutlass::NumericConverter<ElementReduce, ElementCompute> destination_converter;
  cutlass::HostTensor<ElementReduce, LayoutTagVector> _tensor_column_reduce;
  cutlass::HostTensor<ElementCompute, LayoutTagVector> _reduce_buffer;
  cutlass::HostTensor<ElementReduce, LayoutTagVector> _reference_column_reduce;
  int _M;
  TestBedImpl impl_;
  ReduceFn<ElementCompute> reduce_fn;
public:
  HostColumnReduce(){}
  template <typename ProblemShapeType>
  HostColumnReduce(ProblemShapeType problem_size, TestBedImpl impl, bool check_relative_equality=false):
    Base(check_relative_equality),
    impl_(impl) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    _M = cute::get<0>(problem_shape_MNKL);
    _tensor_column_reduce.resize(cutlass::Coord<1>(_M));
    _reference_column_reduce.resize(cutlass::Coord<1>(_M));
    _reduce_buffer.resize(cutlass::Coord<1>(_M));

    _tensor_column_reduce.sync_device();
  }

  template <class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc, ElementCompute child_0_result) {
    auto TensorColReduce = cute::make_tensor(_reduce_buffer.host_data(),
      cute::make_layout(cute::make_shape(_M, cute::_1{})));
    TensorColReduce(m + m_b, 1) = reduce_fn(TensorColReduce(m + m_b, 1), child_0_result);
    return child_0_result;
  }

  bool compare_reference(std::stringstream& error_ss) {
    // Verify the store node
    _tensor_column_reduce.sync_host();

    auto TensorColReduce = cute::make_tensor(_reference_column_reduce.host_data(),
      cute::make_layout(cute::make_shape(_M, cute::_1{})));
    
    auto TensorReduceBuffer = cute::make_tensor(_reduce_buffer.host_data(),
      cute::make_layout(cute::make_shape(_M, cute::_1{})));

    // Filling the reference tensor with the reduce buffer
    for (int m = 0; m < _M; m ++) {
      TensorColReduce(m, 1) = destination_converter(TensorReduceBuffer(m, 1));
    }

    bool equal = this->equality_check(_reference_column_reduce.host_view(), _tensor_column_reduce.host_view());
    if (!equal) {
      error_ss 
        << "\n\nColumn Reduce Reference =\n" << _reference_column_reduce.host_view()
        << "\n\nColumn Reduce Computed =\n" << _tensor_column_reduce.host_view() << "\n\n";
    }
    return equal;
  }

  Arguments get_arguments() {
    return {_tensor_column_reduce.device_data()};
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// EVT - Scalar Reduce

template <
  typename Gemm,
  template <class> class ReduceFn,
  typename ElementReduce
>
class HostScalarReduce: public HostEVTNodeBase<Gemm> {
public:
  using Base = HostEVTNodeBase<Gemm>;
  using TestBedImpl = typename Base::TestBedImpl;
  using ElementCompute = typename Base::ElementCompute;
  using ElementOutput = typename Base::ElementD;
  using LayoutTagVector = cutlass::layout::PackedVectorLayout;

  struct Arguments {
    struct OpArgs {
      ElementReduce* ptr_scalar = nullptr;
      ElementCompute reduce_identity = 0;
      cute::Stride<cute::_0, cute::_0, cute::_0> dScalar = {};
    } op;
  };

private:
  cutlass::NumericConverter<ElementReduce, ElementCompute> destination_converter;
  cutlass::HostTensor<ElementReduce, LayoutTagVector> _tensor_scalar_reduce;
  cutlass::HostTensor<ElementCompute, LayoutTagVector> _reduce_buffer;
  cutlass::HostTensor<ElementReduce, LayoutTagVector> _reference_scalar_reduce;
  ReduceFn<ElementCompute> reduce_fn;
  TestBedImpl impl_;
public:
  HostScalarReduce(){}
  template <typename ProblemShapeType>
  HostScalarReduce(ProblemShapeType problem_size, TestBedImpl impl, bool check_relative_equality=false):
    Base(check_relative_equality),
    impl_(impl) {
    _tensor_scalar_reduce.resize(cutlass::Coord<1>(1));
    _reference_scalar_reduce.resize(cutlass::Coord<1>(1));
    _reduce_buffer.resize(cutlass::Coord<1>(1));

    _tensor_scalar_reduce.sync_device();
  }

  template <class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc, ElementCompute child_0_result) {
    auto TensorRowReduce = cute::make_tensor(_reduce_buffer.host_data(),
      cute::make_layout(cute::make_shape(cute::_1{})));
    TensorRowReduce(0) = reduce_fn(TensorRowReduce(0), child_0_result);
    return child_0_result;
  }

  bool compare_reference(std::stringstream& error_ss) {
    // Verify the store node
    _tensor_scalar_reduce.sync_host();

    auto TensorRowReduce = cute::make_tensor(_reference_scalar_reduce.host_data(),
      cute::make_layout(cute::make_shape(cute::_1{})));
    
    auto TensorReduceBuffer = cute::make_tensor(_reduce_buffer.host_data(),
      cute::make_layout(cute::make_shape(cute::_1{})));

    // Filling the reference tensor with the reduce buffer
    TensorRowReduce(0) = destination_converter(TensorReduceBuffer(0));

    bool equal = this->equality_check(_reference_scalar_reduce.host_view(), _tensor_scalar_reduce.host_view());
    if (!equal) {
      error_ss 
        << "\n\nScalar Reduce Reference =\n" << _reference_scalar_reduce.host_view()
        << "\n\nScalar Reduce Computed =\n" << _tensor_scalar_reduce.host_view() << "\n\n";
    }
    return equal;
  }

  Arguments get_arguments() {
    return {_tensor_scalar_reduce.device_data()};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Host EVT wrapper

/// The ArgumentPack is used to model the alignment when num ops <= 4
template <typename... Ops>
struct ArgumentPack;

template <typename T>
struct ArgumentPack<T> {
  T arg;
  ArgumentPack(T first):
    arg(first) {}
};

template <typename First, typename... Rest>
struct ArgumentPack<First, Rest...> {
  First arg;
  ArgumentPack<Rest...> rest_args;

  ArgumentPack(First first, Rest... rest) :
    arg(first), rest_args(rest...) {}
};


/// Base class for Host Visitor
template <typename Gemm, class... Ops>
struct HostVisitorBase: public HostEVTNodeBase<Gemm> {
public:
  using Base = HostEVTNodeBase<Gemm>;
  using ElementCompute = typename Base::ElementCompute;

  using Arguments_struct = ArgumentPack<typename Ops::Arguments...>;
  using Arguments_tuple = cute::tuple<typename Ops::Arguments...>;

  constexpr static int Rm1 = sizeof...(Ops);
  constexpr static bool cond = Rm1 > 4;
  using Arguments = cute::conditional_t<cond, Arguments_tuple, Arguments_struct>;

  std::tuple<Ops...> ops;

  HostVisitorBase(){}
  template<typename ProblemShapeType, typename TestBedImpl>
  HostVisitorBase(ProblemShapeType problem_size, TestBedImpl impl, bool check_relative_equality=false)
    :Base(check_relative_equality),
    ops(test::gemm::device::tapply(std::tuple<Ops...>{}, 
      [&] (auto&& op) {
        using Op = cute::remove_cvref_t<decltype(op)>;
        return Op(problem_size, impl, check_relative_equality);
      },
      [] (auto&&... _ops) { 
        return std::make_tuple(_ops...); 
      },
      cute::make_seq<Rm1>{}
    )){ }

  bool compare_reference(std::stringstream& error_ss) {
    return cute::detail::tapply(ops,
      [&](auto& op) {
        return op.compare_reference(error_ss);
      },
      [&] (auto&&... inputs) {
        return arrayAnd(inputs...);
      },
      cute::make_seq<Rm1>{}
    );
  }

  void* get_tensor_C_ptr() {
    return cute::detail::tapply(ops,
      [&](auto& op) {
        return op.get_tensor_C_ptr();
      },
      [&] (auto&&... inputs) {
        return findNonNullPtr(inputs...);
      },
      cute::make_seq<Rm1>{}
    );
  }

  void* get_tensor_D_ptr() {
    return cute::detail::tapply(ops,
      [&](auto& op) {
        return op.get_tensor_D_ptr();
      },
      [&] (auto&&... inputs) {
        return findNonNullPtr(inputs...);
      },
      cute::make_seq<Rm1>{}
    );
  }

  Arguments get_arguments() {
    return test::gemm::device::tapply(ops,
      [&](auto& op) {
        return op.get_arguments();
      },
      [&] (auto&&... args) {
        if constexpr (Rm1 > 4) {
          return cute::make_tuple(args...);
        } else {
          return Arguments(args...);
        }  
      },
      cute::make_seq<Rm1>{}
    );
  }

  bool arrayAnd(bool passed) {
    return passed;
  }

  template <typename... Args>
  bool arrayAnd(bool first_passed, Args... passed) {
    if (first_passed) {
      return arrayAnd(passed...);
    }
    return first_passed;
  }

};


/// Tree-struct visitor
template <class NodeOp, class... ChildOps>
struct HostTreeVisitor: public HostVisitorBase<typename NodeOp::Base::Gemm, ChildOps..., NodeOp> {
public:
  using Gemm = typename NodeOp::Base::Gemm;
  using Base = HostVisitorBase<Gemm, ChildOps..., NodeOp>;
  using ElementCompute = typename Base::ElementCompute;
  using Arguments = typename Base::Arguments;
  
  constexpr static int Rm1 = sizeof...(ChildOps);

  HostTreeVisitor(){}
  template<typename ProblemShapeType, typename TestBedImpl>
  HostTreeVisitor(ProblemShapeType problem_size, TestBedImpl impl, bool check_relative_equality=false)
    :Base(problem_size, impl, check_relative_equality){ }

  template <class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc) {
    return cute::detail::tapply(this->ops,
      [&] (auto& op) {
        return op.visit(m, n, l, m_b, n_b, acc);
      },
      [&] (auto&&... frg_inputs) {
        return std::get<Rm1>(this->ops).visit(m, n, l, m_b, n_b, acc, frg_inputs...);
      },
      cute::make_seq<Rm1>{}
    );
  }
};


/// General Graph visitor
template <class Gemm, class EdgeTuple, class... Ops>
struct HostTopoVisitor: public HostVisitorBase<Gemm, Ops...> {
public:
  using Base = HostVisitorBase<Gemm, Ops...>;
  using ElementCompute = typename Base::ElementCompute;
  constexpr static int Rm1 = Base::Rm1;
  using Arguments = typename Base::Arguments;
  
private:
  ElementCompute frg_outputs[Rm1];
public:
  HostTopoVisitor(){}
  template<typename ProblemShapeType, typename TestBedImpl>
  HostTopoVisitor(ProblemShapeType problem_size, TestBedImpl impl, bool check_relative_equality=false)
    :Base(problem_size, impl, check_relative_equality) { }

  template<class ElementAccumulator, int I>
  ElementCompute visit_(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc) {
      frg_outputs[I] = cute::transform_apply(cute::get<I>(EdgeTuple{}),
        [&] (auto&& _E) {
          constexpr int e = cute::remove_cvref_t<decltype(_E)>::value;
          return frg_outputs[e];
        },
        [&] (auto const&... frg_inputs) {
          ElementCompute res = std::get<I>(this->ops).visit(m, n, l, m_b, n_b, acc, frg_inputs...);
          return res;
        }
      );

      if constexpr (I < Rm1 - 1) {
        return visit_<ElementAccumulator, I+1>(m, n, l, m_b, n_b, acc);
      } else {
        return frg_outputs[I];
      }
  }

  template <class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc) {

    return visit_<ElementAccumulator, 0>(m, n, l, m_b, n_b, acc);
  }

};


/// SplitTree visitor
template <class Gemm, class InputTree, class OutputTree, class... AuxOutTrees>
struct HostSplitTreeVisitor: public HostVisitorBase<Gemm, InputTree, AuxOutTrees..., OutputTree> {
public:
  using Base = HostVisitorBase<Gemm, InputTree, AuxOutTrees..., OutputTree>;
  using ElementCompute = typename Base::ElementCompute;
  using Arguments = typename Base::Arguments;

  constexpr static int Rm2 = sizeof...(AuxOutTrees);

private:
  ElementCompute frg_input;
public:
  HostSplitTreeVisitor(){}
  template<typename ProblemShapeType, typename TestBedImpl>
  HostSplitTreeVisitor(ProblemShapeType problem_size, TestBedImpl impl, bool check_relative_equality=false)
    :Base(problem_size, impl, check_relative_equality) { }

  template<class ElementAccumulator, int I>
  void visitAux(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator frag) {
    std::get<I+1>(this->ops).visit(m, n, l, m_b, n_b, frag);

    if constexpr (I < Rm2 - 1) {
      return visitAux<ElementAccumulator, I+1>(m, n, l, m_b, n_b, frag);
    } else {
      return;
    }
  }

  template<class ElementAccumulator>
  ElementCompute visit(
    int64_t m, int64_t n, int64_t l, int m_b, int n_b,
    ElementAccumulator acc) {
    
    /// Compute the input tree
    frg_input = std::get<0>(this->ops).visit(m, n, l, m_b, n_b, acc);

    /// Compute the aux out tree
    visitAux<ElementAccumulator, 0>(m, n, l, m_b, n_b, frg_input);
    /// Visit the output tree
    return std::get<Rm2+1>(this->ops).visit(m, n, l, m_b, n_b, frg_input);
  }
};

/// Universal testbed for EVT
template <class Gemm, typename EVT>
class Testbed3xEVT {
public:
  // The EVT Module to test
  using EVTModule = typename EVT::EVTModule;

  using TestBedImpl = typename detail::TestbedImpl<Gemm, cutlass::epilogue::thread::Identity, true>;
  using Kernel = typename Gemm::GemmKernel;
  using Epilogue = typename Gemm::GemmKernel::CollectiveEpilogue;
  using ElementAccumulator = typename Kernel::ElementAccumulator;
  using ElementC = typename Kernel::ElementC;
  using ElementD = typename Kernel::ElementD;

  using ProblemShapeType = typename Kernel::ProblemShape;

  using LayoutTagA = typename TestBedImpl::LayoutTagA;
  using LayoutTagB = typename TestBedImpl::LayoutTagB;
  using LayoutTagC = typename TestBedImpl::LayoutTagC;
  using LayoutTagD = typename TestBedImpl::LayoutTagD;

  //
  // Methods
  //
  Testbed3xEVT(
    bool check_relative_equality_,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = TestBedImpl::kDefaultSeed
  ) :
     impl_((check_relative_equality_ ? CheckEquality::RELATIVE : CheckEquality::EXACT), ScalarLoc::ON_DEVICE, VectorBeta::ENABLED,
           init_A_, init_B_, init_C_, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform, seed_),
           check_relative_equality(check_relative_equality_) { }

  Testbed3xEVT(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = TestBedImpl::kDefaultSeed
  ) :
     impl_(CheckEquality::EXACT, ScalarLoc::ON_DEVICE, VectorBeta::ENABLED,
           init_A_, init_B_, init_C_, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform, seed_),
           check_relative_equality(false)  { }

  Testbed3xEVT(
    typename LayoutTagA::Stride stride_factor_A_,
    typename LayoutTagB::Stride stride_factor_B_,
    typename LayoutTagC::Stride stride_factor_C_,
    typename LayoutTagD::Stride stride_factor_D_,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = TestBedImpl::kDefaultSeed
  ) :
    impl_(stride_factor_A_, stride_factor_B_, stride_factor_C_, stride_factor_D_,
          CheckEquality::EXACT, ScalarLoc::ON_DEVICE, VectorBeta::ENABLED,
          init_A_, init_B_, init_C_, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform, seed_),
          check_relative_equality(false)  { }
  
  /// Initializes data structures
  void initialize(ProblemShapeType problem_size) {
    //
    // Allocate the GEMM workspace for A/B tensor
    //
    impl_.initialize(problem_size);
  }
  // Detail Implementation
  TestBedImpl impl_;

  // Whether to use relative equality checks
  bool check_relative_equality;

  bool verify(ProblemShapeType problem_size, EVTModule& host_reference) {
    
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::get<0>(problem_shape_MNKL);
    auto N = cute::get<1>(problem_shape_MNKL);
    auto K = cute::get<2>(problem_shape_MNKL);
    auto L = cute::get<3>(problem_shape_MNKL);

    auto A = cute::make_tensor(impl_.collective_mma_inputs.tensor_A.host_data(),
      cute::make_layout(cute::make_shape(M, K, L), impl_.collective_mma_inputs.stride_a));
    auto B = cute::make_tensor(impl_.collective_mma_inputs.tensor_B.host_data(),
      cute::make_layout(cute::make_shape(N, K, L), impl_.collective_mma_inputs.stride_b));
    auto LayoutD = cute::make_layout(cute::make_shape(M, N, L), impl_.collective_epilogue.stride_d);

    cutlass::reference::host::GettMainloopParams<ElementAccumulator, decltype(A), decltype(B)> mainloop_params{A, B};

    /// Reference Kernel
    static int constexpr kBlockM = 64;
    static int constexpr kBlockN = 64;

#if defined(_OPENMP)
    #pragma omp parallel for collapse(3)
#endif
    for (int64_t l = 0; l < cute::size<2>(mainloop_params.A.layout()); ++l) {
      for (int64_t m = 0; m < cute::size<0>(mainloop_params.A.layout()); m += kBlockM) {
        for (int64_t n = 0; n < cute::size<0>(mainloop_params.B.layout()); n += kBlockN) {
          ElementAccumulator acc[kBlockM][kBlockN];
          gett_mainloop(mainloop_params, m, n, l, acc);
          /// Epilogue EVT
          for (int n_b = 0; n_b < kBlockN; ++n_b) {
            for (int m_b = 0; m_b < kBlockM; ++m_b) {
              if (m + m_b < cute::size<0>(LayoutD) && n + n_b < cute::size<1>(LayoutD)) {
                host_reference.visit(m, n, l, m_b, n_b, acc[m_b][n_b]);
              }
            }
          }
        }
      }
    }

    std::stringstream error_ss;
    bool passed = host_reference.compare_reference(error_ss);
    if (!passed) {
      std::stringstream fname;
      fname << "error_Gemm_device_"
        << M << "x" << N << "x" << K << "x" << L << "_"
        << cute::get<0>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << cute::get<1>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << cute::get<2>(typename Gemm::GemmKernel::TileShape{}) << ".txt";
      
      std::ofstream file(fname.str());
      file
        << "problem: " << ' ' << M << "x" << N << "x" << K
        << ", Batch count = " << L << "\n\n";
      
      file
        << "A =\n" << impl_.collective_mma_inputs.tensor_A.host_view()
        << "\nB =\n" << impl_.collective_mma_inputs.tensor_B.host_view()
        << "\nC =\n" << impl_.collective_epilogue.tensor_C.host_view() << "\n\n";
      
      file << error_ss.str();
    }

    return passed;
  }

  bool run(
    ProblemShapeType problem_size,
    bool profiling = false,
    int iterations = 20,
    int splits = 1) {   
    // Fail test if insufficient CUDA device
    if (!impl_.sufficient()) {
      std::cout << "Test failed due to insufficient CUDA device." << std::endl;
      return false;
    }
    //
    // Initialize the Gemm operator
    //

    typename Gemm::Arguments arguments;
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    if (not profiling) {
      impl_.sm_count = std::min(impl_.MaxSmCount, cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id));
      hw_info.sm_count = impl_.sm_count;
    }
    else {
      impl_.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
      hw_info.sm_count = impl_.sm_count;
    }

    typename Gemm::GemmKernel::TileScheduler::Arguments scheduler_args;
    if constexpr (cute::is_same_v<typename Gemm::GemmKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>) {
      scheduler_args = { splits };
    }

    /// Initializes data structures
    /// A/B/C/D Tensor
    initialize(problem_size);

    /// Initialize the epilogue arguments
    EVTModule host_reference(problem_size, impl_, check_relative_equality);

    arguments = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {
        impl_.collective_mma_inputs.tensor_A.device_data(), impl_.collective_mma_inputs.stride_a,
        impl_.collective_mma_inputs.tensor_B.device_data(), impl_.collective_mma_inputs.stride_b
      },
      {   // Epilogue arguments
        {}, // thread
        static_cast<ElementC*>(host_reference.get_tensor_C_ptr()),
        impl_.collective_epilogue.stride_c,
        static_cast<ElementD*>(host_reference.get_tensor_D_ptr()),
        impl_.collective_epilogue.stride_d
      },  // Epilogue arguments end
      hw_info,
      scheduler_args
    };

    // Filling in the thread arguments
    typename EVTModule::Arguments epilogue_args = host_reference.get_arguments();
    std::memcpy(&arguments.epilogue.thread, &epilogue_args.arg, sizeof(epilogue_args.arg));

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
      return impl_.profile(problem_size, iterations, gemm_op, arguments, workspace);
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
    }

    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    //
    // Verify
    //
    bool passed = this->verify(problem_size, host_reference);
    if (!passed) {
      std::cout << "Error : Failed \n";
    }

    return passed;
  }
};


template <typename Gemm, typename EVT>
bool TestAllEVT(bool check_relative_equality=false) {
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

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

  Testbed3xEVT<Gemm, EVT> testbed(check_relative_equality);
  bool passed = true;

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

    passed = testbed.run(problem_size);

    if (!passed) {
      return false;
    }
    }
  }
  }

  // if we do support batched GEMM, just run one test on it to save on test time
  if constexpr (cute::rank(ProblemShapeType{}) == 4) {
  auto problem_size = ProblemShapeType{256 + max_alignment, 256 + max_alignment, 160 + max_alignment, /* l */ 3};
  passed = testbed.run(
    problem_size
  );

  if (!passed) {
    return false;
  }
  }

  return passed;
}

} // namespace device
} // namespace gemm
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////
