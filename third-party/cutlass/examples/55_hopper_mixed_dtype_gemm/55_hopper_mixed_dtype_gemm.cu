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
    \brief Hopper GEMM example with different data types using CUTLASS 3.0 APIs for NVIDIA Hopper architecture

    This example shows how to perform GEMM where the input tensors A and B have different element types. CUTLASS currently supports upcasting
    from a narrower (fewer bits) to a wider (more bits) type and utilizing the tensor core instruction for the wider type. For instance, when doing
    INT8 x FP16, CUTLASS will convert INT8 -> FP16 and do math using FP16 tensor cores. Similarly, for INT4 x INT8, it will upcast to INT8 and issue math 
    using INT8 tensor cores.

    The narrower type always passes through the register file. Therefore, in cases where the narrower type is operand B, the collective will implicitly swap 
    A and B in the main loop. However, implicit swaps do not support TMA epilogues. Consequently, it is essential to consider this when constructing the epilogue, 
    as illustrated in this example.

    Note that in this example, we explicitly swap A and B in order to use TMA epilogues. We do this since TMA epilogues are more performant on problem sizes of interest.

    It is expected that the scale's K dimension be scale_k = ceil_div(problem_k, group_size). 
    
    Scales are always expected to be MN major. This means the fastest changing dimension must be M if A is scaled or N if B is scaled.
    
    If A is being scaled, the scales should have shape [M, scale_k],  while if B is scaled, it must have shape [N, scale_k].

    The implementation only supports "group-wise" scales. However, we can make it work for per-column scales by setting the groups size
    equal to the gemm problem K.

    Limitations:
      1) Only supported combinations are 16-bit x {8-bit, 4-bit, 2-bit} and {8-bit} x {4-bit, 2-bit}.
      2) The narrow type must always be in K-major format.
      3) The scales and zeros must be MN major. That means if A is scaled, it must be column major, but if B is scaled it must be row major.
      4) The scales and the zeros must have the same layout and groupsize.
      5) When dealing with 8-bit x {4-bit, 2-bit}, both inputs must be in K-major format.
      6) Currently, TMA epilogues cannot be used when the narrow type is the B operand. This limitation arises because the implementation always swaps the 
         operands to ensure that the narrow type passes through the register file, and TMA epilogues do not currently support implicit swap + transpose operations. 
         We plan to address this limitation in the future. However, we address this in the example by explicitly swapping and transposing the operands.

    Examples:
      
      Runs the mixed input batched gemm (with batch size 2), converting B to the type of A (mode 0)
      $ ./examples/55_hopper_mixed_dtype_gemm/55_hopper_mixed_dtype_gemm --m=2048 --n=2048 --k=2048 --l=2 --mode=0

      Runs the mixed input gemm, and applies a scaling factor to B before mma (mode 1). Applies a vector of scales to the entire
      matrix (group size is the same as the gemm k dimension).
      $ ./examples/55_hopper_mixed_dtype_gemm/55_hopper_mixed_dtype_gemm --m=4096 --n=5120 --k=8192 --g=8192 --mode=1

      Runs the mixed input gemm, and applies a scaling factor and adds a zero-point to B before mma (mode 2). Uses a group size of 128.
      $ ./examples/55_hopper_mixed_dtype_gemm/55_hopper_mixed_dtype_gemm --m=2048 --n=5120 --k=8192 --g=128 --mode=2
*/

#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gett.hpp"

#include "helper.h"
#include "unfused_weight_dequantize.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

// This is just an example, so we use a regular enum so we can compare directly to the command-line int.
enum GemmMode {
  ConvertOnly,
  ScaleOnly,
  ScaleWithZeroPoint
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////
using MmaType = cutlass::float_e4m3_t;
using QuantType = cutlass::int4b_t;
constexpr int TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value;

// A matrix configuration
using         ElementA    = MmaType;                                        // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = QuantType;                                      // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// This example manually swaps and transposes, so keep transpose of input layouts
using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

using ElementZero = cutlass::half_t;
using ElementScale = cutlass::half_t;
using LayoutScale = cutlass::layout::RowMajor;

// C/D matrix configuration
using         ElementC    = cutlass::half_t;                                // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// D matrix configuration
using         ElementD    = ElementC;
using         LayoutD     = LayoutC;
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementCompute      = float;                                          // Element type for epilogue computation
using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
using TileShape           = Shape<_128,_256,cute::Int<TileShapeK>>;         // Threadblock-level tile size
using ClusterShape        = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster
using KernelSchedule      = cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput;  // Kernel to launch based on the default setting in the Collective Builder 
using EpilogueSchedule    = cutlass::epilogue::TmaWarpSpecializedCooperative;
using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementAccumulator,
    // Transpose layout of D here since we use explicit swap + transpose
    // the void type for C tells the builder to allocate 0 smem for the C matrix.
    // We can enable this if beta == 0 by changing ElementC to void below.
    ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type, AlignmentC,
    ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type, AlignmentD,
    EpilogueSchedule // This is the only epi supporting the required swap + transpose.
  >::CollectiveOp;

// ============================================================ MIXED INPUT NO SCALES ============================================================================
// The collective will infer that the narrow type should be upcasted to the wide type.
// We swap A and B operands to the builder here
using CollectiveMainloopConvertOnly = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementB, LayoutB_Transpose, AlignmentB,
    ElementA, LayoutA_Transpose, AlignmentA,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
  >::CollectiveOp;

using GemmKernelConvertOnly = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, // Indicates ProblemShape
    CollectiveMainloopConvertOnly,
    CollectiveEpilogue
>;

using GemmConvertOnly = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelConvertOnly>;

// =========================================================== MIXED INPUT WITH SCALES ===========================================================================
// The Scale information must get paired with the operand that will be scaled. In this example, B is scaled so we make a tuple of B's information and the scale information.
using CollectiveMainloopScaleOnly = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    cute::tuple<ElementB, ElementScale>, LayoutB_Transpose, AlignmentB,
    ElementA, LayoutA_Transpose, AlignmentA,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
  >::CollectiveOp;

using GemmKernelScaleOnly = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, // Indicates ProblemShape
    CollectiveMainloopScaleOnly,
    CollectiveEpilogue
>;

using GemmScaleOnly = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleOnly>;

// =========================================================== MIXED INPUT WITH SCALES AND ZEROS ==================================================================
// We specify scale + zero elements to indicate that we require both. Scales and biases have the same format.
using CollectiveMainloopScaleWithZeroPoint = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    cute::tuple<ElementB, ElementScale, ElementZero>, LayoutB_Transpose, AlignmentB,
    ElementA, LayoutA_Transpose, AlignmentA,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
  >::CollectiveOp;

using GemmKernelScaleWithZeroPoint = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, // Indicates ProblemShape
    CollectiveMainloopScaleWithZeroPoint,
    CollectiveEpilogue
>;

using GemmScaleWithZeroPoint = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleWithZeroPoint>;
// =================================================================================================================================================================

using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
using StrideC = typename GemmKernelScaleWithZeroPoint::StrideC;
using StrideD = typename GemmKernelScaleWithZeroPoint::StrideD;

using StrideC_ref = cutlass::detail::TagToStrideC_t<LayoutC>;
using StrideD_ref = cutlass::detail::TagToStrideC_t<LayoutD>;

//
// Data members
//

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideC_ref stride_C_ref;
StrideD stride_D;
StrideD_ref stride_D_ref;
uint64_t seed;

// Scale and Zero share a stride since the layout and shapes must be the same.
using StrideS = typename CollectiveMainloopScaleWithZeroPoint::StrideScale;
using StrideS_ref = cutlass::detail::TagToStrideB_t<LayoutScale>;
StrideS stride_S;
StrideS_ref stride_S_ref;

cutlass::HostTensor<MmaType, LayoutA> tensor_A;
cutlass::HostTensor<QuantType, LayoutB> tensor_B;
cutlass::HostTensor<MmaType, LayoutB> tensor_B_dq;
cutlass::HostTensor<ElementScale, LayoutScale> tensor_scale;
cutlass::HostTensor<ElementZero, LayoutScale> tensor_zero;
cutlass::HostTensor<ElementC, LayoutC> tensor_C;
cutlass::HostTensor<typename GemmScaleWithZeroPoint::EpilogueOutputOp::ElementOutput, LayoutD> tensor_D;
cutlass::HostTensor<typename GemmScaleWithZeroPoint::EpilogueOutputOp::ElementOutput, LayoutD> tensor_ref_D;

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help = false;

  float alpha = 1.0f;
  float beta = 0.0f;
  int iterations = 1000;
  int mode = 2;
  int m = 5120, n = 4096, k = 4096;
  int g = 128;
  int l = 1;

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("l", l);
    cmd.get_cmd_line_argument("g", g);
    cmd.get_cmd_line_argument("mode", mode);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "55_hopper_warp_specialized_gemm\n\n"
      << "  Hopper FP32 GEMM using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   The number of independent gemm problems with mnk shape\n"
      << "  --g=<int>                   The size of each group for the scales and zeros. To broadcast a vector of scales or zeros, set the group size to K.\n"
      << "  --mode=<int>                The mode to run the gemm. 0 does (A @ B), 1 means A @ (scale * B), 2 means A @ (scale * B + zero-point).\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "55_hopper_warp_specialized_gemm" << " --m=1024 --n=512 --k=1024 -g 0 --l=10 --alpha=2 --mode=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k * l;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms = 0.0;
  double gflops = 0.0;
  cutlass::Status status = cutlass::Status::kSuccess;
  cudaError_t error = cudaSuccess;
  bool passed = false;

};

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element, class Layout>
bool initialize_tensor(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed=2023) {

  double scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;
  int bits_output = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  }
  else if (bits_input <= 8) {
    scope_max = 2;
    scope_min = -2;
  }
  else if (bits_output == 16) {
    scope_max = 5;
    scope_min = -5;
  }
  else {
    scope_max = 8;
    scope_min = -8;
  }
  cutlass::reference::host::TensorFillRandomUniform(
    view, seed, scope_max, scope_min);

  return true;
}

template <typename Element, typename Layout>
bool initialize_quant_tensor(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed=2023) {
  
  float scope_min = float(cutlass::platform::numeric_limits<Element>::lowest());
  float scope_max = float(cutlass::platform::numeric_limits<Element>::max());

  cutlass::reference::host::TensorFillRandomUniform(
    view, seed, scope_max, scope_min);

  return true;
}

template <class Element, class Layout>
bool initialize_scale(
  cutlass::TensorView<Element, Layout> view, 
  const Options &options) {
  
  if (options.mode == GemmMode::ConvertOnly) {
    // No scales, so just initialize with 1 so we can use the same kernel to dequantize the data.
    cutlass::reference::host::TensorFill(view, Element(1.0f));
  } 
  else {
    float elt_max_f = float(cutlass::platform::numeric_limits<QuantType>::max());
    const float max_dequant_val = 4.f;
    const float min_dequant_val = 0.5f;

    float scope_max(max_dequant_val / elt_max_f);
    float scope_min(min_dequant_val / elt_max_f);

    cutlass::reference::host::TensorFillRandomUniform(
      view, seed, scope_max, scope_min);
  }
  return true;
}

template <class Element, class Layout>
bool initialize_zero(
  cutlass::TensorView<Element, Layout> view,
  const Options &options) {
  
  if (options.mode == GemmMode::ScaleWithZeroPoint) {
    cutlass::reference::host::TensorFillRandomUniform(
      view, seed, 2.0f, -2.0f);
  } else {
    // No bias, so just initialize with 1 so we can use the same kernel to dequantize the data.
    cutlass::reference::host::TensorFill(view, Element(0.0f));
  }
  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {

  auto shape_b = cute::make_shape(options.n, options.k, options.l);
  const int scale_k = (options.k + options.g - 1) / options.g;
  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, options.l));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_b);
  // Reverse stride here due to swap and transpose
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(options.n, options.m, options.l));
  stride_C_ref = cutlass::make_cute_packed_stride(StrideC_ref{}, cute::make_shape(options.m, options.n, options.l));
  // Reverse stride here due to swap and transpose
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.n, options.m, options.l));
  stride_D_ref = cutlass::make_cute_packed_stride(StrideD_ref{}, cute::make_shape(options.m, options.n, options.l));

  auto a_coord = cutlass::make_Coord(options.m * options.l, options.k);
  auto b_coord = cutlass::make_Coord(options.k, options.n * options.l);
  auto c_coord = cutlass::make_Coord(options.m * options.l, options.n);

  tensor_A.resize(a_coord);
  tensor_B.resize(b_coord);
  tensor_B_dq.resize(b_coord);
  tensor_C.resize(c_coord);
  tensor_D.resize(c_coord);
  tensor_ref_D.resize(c_coord);

  tensor_scale.resize({scale_k * options.l, options.n});
  tensor_zero.resize({scale_k * options.l, options.n});

  initialize_tensor(tensor_A.host_view(), seed + 2022);
  initialize_quant_tensor(tensor_B.host_view(), seed + 2021);
  initialize_tensor(tensor_C.host_view(), seed + 2020);
  initialize_scale(tensor_scale.host_view(), options);
  initialize_zero(tensor_zero.host_view(), options);

  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_C.sync_device();
  tensor_scale.sync_device();
  tensor_zero.sync_device();

  auto layout_B = make_layout(shape_b, stride_B);

  auto shape_scale_zero = cute::make_shape(options.n, scale_k, options.l);
  stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(options.n, scale_k, options.l));
  stride_S_ref = cutlass::make_cute_packed_stride(StrideS_ref{}, cute::make_shape(options.n, scale_k, options.l));
  auto layout_scale_zero = make_layout(shape_scale_zero, stride_S_ref);

  dequantize_weight(tensor_B_dq.device_data(), tensor_B.device_data(), layout_B, tensor_scale.device_data(), tensor_zero.device_data(), layout_scale_zero, options.g);
  tensor_B_dq.sync_host();
}

/// Populates a Gemm::Arguments structure from the given commandline options
template <typename Args>
Args args_from_options(const Options &options)
{
// Swap the A and B tensors, as well as problem shapes here.
  if (options.mode == GemmMode::ConvertOnly) {
    return Args {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.n, options.m, options.k, options.l},
      {tensor_B.device_data(), stride_B, tensor_A.device_data(), stride_A},
      {{options.alpha, options.beta}, tensor_C.device_data(), stride_C, tensor_D.device_data(), stride_D}
    };
  } 
  else if (options.mode == GemmMode::ScaleOnly) {
    return Args {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.n, options.m, options.k, options.l},
      {tensor_B.device_data(), stride_B, tensor_A.device_data(), stride_A, tensor_scale.device_data(), stride_S, options.g},
      {{options.alpha, options.beta}, tensor_C.device_data(), stride_C, tensor_D.device_data(), stride_D}
    };
  } 
  else if (options.mode == GemmMode::ScaleWithZeroPoint) {
    return Args {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.n, options.m, options.k, options.l},
      {tensor_B.device_data(), stride_B, tensor_A.device_data(), stride_A, tensor_scale.device_data(), stride_S, options.g, tensor_zero.device_data()},
      {{options.alpha, options.beta}, tensor_C.device_data(), stride_C, tensor_D.device_data(), stride_D}
    };
  } else {
    std::cerr << "Invalid mode " << options.mode << ". Must be 0, 1 or 2." << std::endl;
    exit(-1);
  }
}

bool verify(const Options &options) {
  //
  // Compute reference output
  //

  // In this example, we use the GPU default kernels as a reference (unfused scale)
  // This is to avoid numerical differences from different accumulation order.

  // Again, due to numerical differences, we must use fast acc here when the mma type is
  // FP8 as the fused implementation only supports fast acc at the moment.
  constexpr bool IsFP8Input = cute::is_same_v<MmaType, cutlass::float_e4m3_t> || cute::is_same_v<MmaType, cutlass::float_e5m2_t>;
  using FP8Sched = cute::conditional_t<size<0>(TileShape{}) == 64, cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum, cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum>;
  using ScheduleRef = cute::conditional_t<IsFP8Input, FP8Sched, cutlass::gemm::collective::KernelScheduleAuto>;

  using CollectiveMainloopRef = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      MmaType, LayoutA, AlignmentA,
      MmaType, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAuto,
      ScheduleRef
    >::CollectiveOp;

  using CollectiveEpilogueRef = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using GemmKernelRef = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>, // Indicates ProblemShape
      CollectiveMainloopRef,
      CollectiveEpilogueRef
  >;

  using GemmRef = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelRef>;

  typename GemmRef::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, options.l},
    {tensor_A.device_data(), stride_A, tensor_B_dq.device_data(), stride_B},
    {{options.alpha, options.beta}, tensor_C.device_data(), stride_C_ref, tensor_ref_D.device_data(), stride_D_ref}
  };

  // Run the gemm where the scaling is performed outside of the kernel.
  GemmRef gemm_ref;
  size_t workspace_size = GemmRef::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  CUTLASS_CHECK(gemm_ref.can_implement(arguments));
  CUTLASS_CHECK(gemm_ref.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_ref.run());

  // compare_reference
  tensor_D.sync_host();
  tensor_ref_D.sync_host();
  const ElementD epsilon(1e-2f);
  const ElementD non_zero_floor(1e-4f);
  bool passed = cutlass::reference::host::TensorRelativelyEquals(tensor_ref_D.host_view(), tensor_D.host_view(), epsilon, non_zero_floor);
  return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
  initialize(options);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options<typename Gemm::Arguments>(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = verify(options);

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed) {
    exit(-1);
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least 90.
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA 12 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major < 9) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Hopper Architecture or "
      << "later (compute capability 90 or greater).\n";
    return 0;
  }

  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate CUTLASS kernels
  //

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  if (options.mode == GemmMode::ConvertOnly) {
    std::cout << "Running in no scale mode." << std::endl;
    run<GemmConvertOnly>(options);
  }
  else if (options.mode == GemmMode::ScaleOnly) {
    if (options.g == options.k) {
      std::cout << "Running in per-column scale mode." << std::endl;
    } else {
      std::cout << "Running in group scale mode." << std::endl;
    }
    run<GemmScaleOnly>(options);
  }
  else if (options.mode == GemmMode::ScaleWithZeroPoint) {
    if (options.g == options.k) {
      std::cout << "Running in per-column scale and zero mode." << std::endl;
    } else {
      std::cout << "Running in group scale and zero mode." << std::endl;
    }
    run<GemmScaleWithZeroPoint>(options);
  }
#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
