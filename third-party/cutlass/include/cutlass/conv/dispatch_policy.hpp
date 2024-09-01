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

#include "cutlass/conv/convolution.h"
#include "cutlass/arch/arch.h"

#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"

//////////////////////////////////////////////////////////////////////////////

namespace cutlass::conv {

//////////////////////////////////////////////////////////////////////////////

//
// Policies for categorical dispatch of mainloop against kernel grid schedules
//
struct KernelImplicitTmaWarpSpecializedSm90 { };
struct KernelImplicitTmaWarpSpecializedSm90Cooperative { };
struct KernelImplicitTmaWarpSpecializedSm90Pingpong { };

//
// Collective Mainloop Policies
//

// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, static schedule between TMA and GMMA
// for fprop
template<
  conv::Operator ConvOp_,
  int Stages_,
  int NumSpatialDimensions_,
  class ClusterShape_ = cute::Shape<cute::C<1>,cute::C<1>,cute::C<1>>,
  class KernelSchedule = KernelImplicitTmaWarpSpecializedSm90,
  int PipelineAsyncMmaStages_ = 1
>
struct MainloopSm90TmaGmmaWarpSpecializedImplicitGemm {
  static constexpr int Stages = Stages_;
  static constexpr int NumSpatialDimensions = NumSpatialDimensions_;
  static constexpr Operator ConvOp = ConvOp_;
  static constexpr int PipelineAsyncMmaStages = PipelineAsyncMmaStages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;

  static_assert(NumSpatialDimensions >= 1);
  static_assert(! (cute::is_same_v<KernelSchedule,KernelImplicitTmaWarpSpecializedSm90Cooperative> ||
                   cute::is_same_v<KernelSchedule,KernelImplicitTmaWarpSpecializedSm90Pingpong>),
    "Persistent schedules not support for conv yet.");
};

//////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::conv 
