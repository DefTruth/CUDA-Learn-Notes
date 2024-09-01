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
/* \file
   \brief Defines operations for all GEMM operation kinds in CUTLASS Library.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "library_internal.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include <unordered_map>

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::library {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class GemmOperation3xBase : public Operation {
public:
  using Operator = Operator_;
  using OperatorArguments = typename Operator::Arguments;
  using ElementA = typename Operator::ElementA;
  using LayoutA = typename Operator::LayoutA;
  using ElementB = typename Operator::ElementB;
  using LayoutB = typename Operator::LayoutB;
  using ElementC = typename Operator::ElementC;
  using LayoutC = typename Operator::LayoutC;
  using ElementD = typename Operator::ElementD;
  using LayoutD = typename Operator::LayoutD;
  // assuming all tensors use same type for StrideIndex
  using StrideIndex = typename Operator::LayoutA::Index;
  using ElementAccumulator = typename Operator::ElementAccumulator;
  using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;

private:
  GemmDescription description_;

public:

  /// Constructor
  GemmOperation3xBase(char const *name = "unknown_gemm", GemmKind gemm_kind_ = GemmKind::kGemm) {

    description_.name = name;
    description_.provider = Provider::kCUTLASS;
    description_.kind = OperationKind::kGemm;
    description_.gemm_kind = gemm_kind_;

    description_.tile_description.threadblock_shape = make_Coord(
      Operator::ThreadblockShape::kM,
      Operator::ThreadblockShape::kN,
      Operator::ThreadblockShape::kK);

    if constexpr (Operator::ArchTag::kMinComputeCapability >= 90) {
      description_.tile_description.cluster_shape = make_Coord(
        Operator::ClusterShape::kM,
        Operator::ClusterShape::kN,
        Operator::ClusterShape::kK);
    }

    description_.tile_description.threadblock_stages = Operator::kStages;

    description_.tile_description.warp_count = make_Coord(
      Operator::WarpCount::kM,
      Operator::WarpCount::kN,
      Operator::WarpCount::kK);

    description_.tile_description.math_instruction.instruction_shape = make_Coord(
      Operator::InstructionShape::kM,
      Operator::InstructionShape::kN,
      Operator::InstructionShape::kK);

    description_.tile_description.math_instruction.element_accumulator =
      NumericTypeMap<ElementAccumulator>::kId;

    description_.tile_description.math_instruction.opcode_class =
      OpcodeClassMap<typename Operator::OperatorClass>::kId;

    description_.tile_description.math_instruction.math_operation =
      MathOperationMap<typename Operator::MathOperator>::kId;

    description_.tile_description.minimum_compute_capability =
      ArchMap<typename Operator::ArchTag, typename Operator::OperatorClass>::kMin;

    description_.tile_description.maximum_compute_capability =
      ArchMap<typename Operator::ArchTag, typename Operator::OperatorClass>::kMax;

    description_.A = make_TensorDescription<ElementA, LayoutA>(Operator::kAlignmentA);
    description_.B = make_TensorDescription<ElementB, LayoutB>(Operator::kAlignmentB);
    description_.C = make_TensorDescription<ElementC, LayoutC>(Operator::kAlignmentC);
    description_.D = make_TensorDescription<ElementD, LayoutD>(Operator::kAlignmentD);
    description_.element_epilogue = NumericTypeMap<ElementCompute>::kId;

    description_.split_k_mode = SplitKMode::kNone;
    description_.transform_A = ComplexTransformMap<Operator::kTransformA>::kId;
    description_.transform_B = ComplexTransformMap<Operator::kTransformB>::kId;
  }

  /// Returns the description of the GEMM operation
  virtual OperationDescription const & description() const {
    return description_;
  }

  /// Returns the description of the GEMM operation
  GemmDescription const& get_gemm_description() const {
    return description_;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class GemmUniversal3xOperation : public GemmOperation3xBase<Operator_> {
public:

  using Operator = Operator_;
  using OperatorArguments = typename Operator::Arguments;
  using ElementA = typename Operator::ElementA;
  using LayoutA = typename Operator::LayoutA;
  using ElementB = typename Operator::ElementB;
  using LayoutB = typename Operator::LayoutB;
  using ElementC = typename Operator::ElementC;
  using LayoutC = typename Operator::LayoutC;
  using ElementD = typename Operator::ElementD;
  using LayoutD = typename Operator::LayoutD;
  using ElementAccumulator = typename Operator::ElementAccumulator;
  using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;

  using CollectiveMainloop = typename Operator::CollectiveMainloop;
  using CollectiveEpilogue = typename Operator::CollectiveEpilogue;
  using ThreadEpilogueOp = typename CollectiveEpilogue::ThreadEpilogueOp;

public:

  /// Constructor
  GemmUniversal3xOperation(char const *name = "unknown_gemm"):
    GemmOperation3xBase<Operator_>(name, GemmKind::kUniversal) {}

protected:

  /// Constructs the arguments structure given the configuration and arguments
  static Status construct_arguments_(
      OperatorArguments &operator_args, GemmUniversalConfiguration const *configuration) {
    // NOTE: GemmUniversalConfiguration does not contain problem shapes or batch strides
    // Do nothing here and construct kernel arguments in update_arguments_ instead
    // We also cannot construct TMA descriptors without all the arguments available

    operator_args.mode = configuration->mode;
    return Status::kSuccess;
  }

  template<class FusionArgs, class = void>
  struct UpdateFusionArgs {
    static Status update_(FusionArgs const& fusion_args, GemmUniversalArguments const &arguments) {
      // If a custom EVT is instantiated then it is the users's responsibility
      // to ensure alpha and beta are updated appropriately
      return Status::kSuccess;
    }
  };

  template<class FusionArgs>
  struct UpdateFusionArgs<FusionArgs, cute::void_t<decltype(FusionArgs{}.alpha)>> {
    static Status update_(FusionArgs& fusion_args, GemmUniversalArguments const &arguments) {
      if (arguments.pointer_mode == ScalarPointerMode::kHost) {
        fusion_args.alpha = *static_cast<ElementCompute const *>(arguments.alpha);
        fusion_args.beta = *static_cast<ElementCompute const *>(arguments.beta);
        fusion_args.alpha_ptr = nullptr;
        fusion_args.beta_ptr = nullptr;

        return Status::kSuccess;
      }
      else if (arguments.pointer_mode == ScalarPointerMode::kDevice) {
        fusion_args.alpha = 0;
        fusion_args.beta = 0;
        fusion_args.alpha_ptr = static_cast<ElementCompute const *>(arguments.alpha);
        fusion_args.beta_ptr = static_cast<ElementCompute const *>(arguments.beta);

        return Status::kSuccess;
      }
      else {
        return Status::kErrorInvalidProblem;
      }
    }
  };

  /// Constructs the arguments structure given the configuration and arguments
  static Status update_arguments_(
      OperatorArguments &operator_args,
      GemmUniversalArguments const *arguments) {
    Status status = Status::kSuccess;

    status = UpdateFusionArgs<decltype(operator_args.epilogue.thread)>::update_(
      operator_args.epilogue.thread, *arguments);
    if (status != Status::kSuccess) {
      return status;
    }

    // TODO: type erase Arguments structure in 3.0 GEMM
    operator_args.problem_shape = cute::make_shape(
      arguments->problem_size.m(),
      arguments->problem_size.n(),
      arguments->problem_size.k(),
      arguments->batch_count);

    // update arguments
    operator_args.mainloop.ptr_A = static_cast<ElementA const *>(arguments->A);
    operator_args.mainloop.ptr_B = static_cast<ElementB const *>(arguments->B);
    operator_args.epilogue.ptr_C = static_cast<ElementC const *>(arguments->C);
    operator_args.epilogue.ptr_D = static_cast<ElementD       *>(arguments->D);

    operator_args.mainloop.dA = cute::make_int_tuple_from<typename Operator::GemmKernel::StrideA>(
        arguments->lda, arguments->batch_stride_A);
    operator_args.mainloop.dB = cute::make_int_tuple_from<typename Operator::GemmKernel::StrideB>(
        arguments->ldb, arguments->batch_stride_B);
    operator_args.epilogue.dC = cute::make_int_tuple_from<typename Operator::GemmKernel::StrideC>(
        arguments->ldc, arguments->batch_stride_C);
    operator_args.epilogue.dD = operator_args.epilogue.dC;

    /* Query device SM count to pass onto the kernel as an argument, where needed */
    operator_args.hw_info.sm_count = arguments->sm_count;

    if constexpr (!std::is_const_v<decltype(operator_args.scheduler.raster_order)>) {
      using Enum_t = decltype(operator_args.scheduler.raster_order);
      switch (arguments->raster_order) {
        case RasterOrder::kAlongN:
          operator_args.scheduler.raster_order = Enum_t::AlongN;
          break;
        case RasterOrder::kAlongM:
          operator_args.scheduler.raster_order = Enum_t::AlongM;
          break;
        default:
          operator_args.scheduler.raster_order = Enum_t::Heuristic;
      }
    }

    return status;
  }

public:

  /// Returns success if the operation can proceed
  Status can_implement(
      void const *configuration_ptr, void const *arguments_ptr) const override {
    GemmUniversalConfiguration const *configuration =
      static_cast<GemmUniversalConfiguration const *>(configuration_ptr);
    GemmUniversalArguments const *arguments =
      static_cast<GemmUniversalArguments const *>(arguments_ptr);

    OperatorArguments args;
    auto status = update_arguments_(args, arguments);
    if (status != Status::kSuccess) {
      return status;
    }

    // can_implement rules may need access to problem shape
    args.problem_shape = cute::make_shape(
      configuration->problem_size.m(),
      configuration->problem_size.n(),
      configuration->problem_size.k(),
      configuration->batch_count);
    return Operator::can_implement(args);
  }

  /// Gets the host-side workspace
  uint64_t get_host_workspace_size(void const *configuration) const override {
    return sizeof(Operator);
  }

  /// Gets the device-side workspace
  uint64_t get_device_workspace_size(
      void const *configuration_ptr,void const *arguments_ptr) const override {

    OperatorArguments args;
    auto status = update_arguments_(
      args, static_cast<GemmUniversalArguments const *>(arguments_ptr));
    if (status != Status::kSuccess) {
      return 0;
    }

    uint64_t size = Operator::get_workspace_size(args);
    return size;
  }

  /// Initializes the workspace
  Status initialize(
      void const *configuration_ptr,
      void *host_workspace,
      void *device_workspace,
      cudaStream_t stream = nullptr) const override {
    Operator *op = new (host_workspace) Operator;
    return Status::kSuccess;
  }

  /// Runs the kernel
  Status run(
      void const *arguments_ptr,
      void *host_workspace,
      void *device_workspace = nullptr,
      cudaStream_t stream = nullptr) const override {

    OperatorArguments args;
    Status status = update_arguments_(args, static_cast<GemmUniversalArguments const *>(arguments_ptr));
    if (status != Status::kSuccess) {
      return status;
    }

    Operator *op = static_cast<Operator *>(host_workspace);
    // We need to call initialize() since we have to rebuild TMA desc for every new set of args
    status = op->run(args, device_workspace, stream);
    return status;
  }
};
///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::library

///////////////////////////////////////////////////////////////////////////////////////////////////
