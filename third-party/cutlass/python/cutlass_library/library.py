#################################################################################################
#
# Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Data types and tags used for emitting CUTLASS C++ kernels
"""

import enum
import re

# The following block implements enum.auto() for Python 3.5 variants that don't include it such
# as the default 3.5.2 on Ubuntu 16.04.
#
# https://codereview.stackexchange.com/questions/177309/reimplementing-pythons-enum-auto-for-compatibility

try:
  from enum import auto as enum_auto
except ImportError:
  __cutlass_library_auto_enum = 0
  def enum_auto() -> int:
    global __cutlass_library_auto_enum
    i = __cutlass_library_auto_enum
    __cutlass_library_auto_enum += 1
    return i

###################################################################################################

#
class GeneratorTarget(enum.Enum):
  Library = enum_auto()
#
GeneratorTargetNames = {
  GeneratorTarget.Library: 'library'
}
#

###################################################################################################

#
class DataType(enum.Enum):
  void = enum_auto()  # primarily used to disable C tensor for epilogues
  b1 = enum_auto()
  u4 = enum_auto()
  u8 = enum_auto()
  u16 = enum_auto()
  u32 = enum_auto()
  u64 = enum_auto()
  s4 = enum_auto()
  s8 = enum_auto()
  s16 = enum_auto()
  s32 = enum_auto()
  s64 = enum_auto()
  e4m3 = enum_auto()
  e5m2 = enum_auto()
  f16 = enum_auto()
  bf16 = enum_auto()
  f32 = enum_auto()
  tf32 = enum_auto()
  f64 = enum_auto()
  cf16 = enum_auto()
  cbf16 = enum_auto()
  cf32 = enum_auto()
  ctf32 = enum_auto()
  cf64 = enum_auto()
  cs4 = enum_auto()
  cs8 = enum_auto()
  cs16 = enum_auto()
  cs32 = enum_auto()
  cs64 = enum_auto()
  cu4 = enum_auto()
  cu8 = enum_auto()
  cu16 = enum_auto()
  cu32 = enum_auto()
  cu64 = enum_auto()
  invalid = enum_auto()

#
ShortDataTypeNames = {
  DataType.s32: 'i',
  DataType.e4m3: 'e4m3',
  DataType.e5m2: 'e5m2',
  DataType.f16: 'h',
  DataType.f32: 's',
  DataType.f64: 'd',
  DataType.cf32: 'c',
  DataType.cf64: 'z',
}

#
DataTypeNames = {
  DataType.void: "void",
  DataType.b1: "b1",
  DataType.u4: "u4",
  DataType.u8: "u8",
  DataType.u16: "u16",
  DataType.u32: "u32",
  DataType.u64: "u64",
  DataType.s4: "s4",
  DataType.s8: "s8",
  DataType.s16: "s16",
  DataType.s32: "s32",
  DataType.s64: "s64",
  DataType.e4m3: 'e4m3',
  DataType.e5m2: 'e5m2',
  DataType.f16: "f16",
  DataType.bf16: "bf16",
  DataType.f32: "f32",
  DataType.tf32: "tf32",
  DataType.f64: "f64",
  DataType.cf16: "cf16",
  DataType.cbf16: "cbf16",
  DataType.cf32: "cf32",
  DataType.ctf32: "ctf32",
  DataType.cf64: "cf64",
  DataType.cu4: "cu4",
  DataType.cu8: "cu8",
  DataType.cu16: "cu16",
  DataType.cu32: "cu32",
  DataType.cu64: "cu64",
  DataType.cs4: "cs4",
  DataType.cs8: "cs8",
  DataType.cs16: "cs16",
  DataType.cs32: "cs32",
  DataType.cs64: "cs64",
}

DataTypeTag = {
  DataType.void: "void",
  DataType.b1: "cutlass::uint1b_t",
  DataType.u4: "cutlass::uint4b_t",
  DataType.u8: "uint8_t",
  DataType.u16: "uint16_t",
  DataType.u32: "uint32_t",
  DataType.u64: "uint64_t",
  DataType.s4: "cutlass::int4b_t",
  DataType.s8: "int8_t",
  DataType.s16: "int16_t",
  DataType.s32: "int32_t",
  DataType.s64: "int64_t",
  DataType.e4m3: 'cutlass::float_e4m3_t',
  DataType.e5m2: 'cutlass::float_e5m2_t',
  DataType.f16: "cutlass::half_t",
  DataType.bf16: "cutlass::bfloat16_t",
  DataType.f32: "float",
  DataType.tf32: "cutlass::tfloat32_t",
  DataType.f64: "double",
  DataType.cf16: "cutlass::complex<cutlass::half_t>",
  DataType.cbf16: "cutlass::complex<cutlass::bfloat16_t>",
  DataType.cf32: "cutlass::complex<float>",
  DataType.ctf32: "cutlass::complex<cutlass::tfloat32_t>",
  DataType.cf64: "cutlass::complex<double>",
  DataType.cu4: "cutlass::complex<cutlass::uint4b_t>",
  DataType.cu8: "cutlass::complex<cutlass::uint8_t>",
  DataType.cu16: "cutlass::complex<cutlass::uint16_t>",
  DataType.cu32: "cutlass::complex<cutlass::uint32_t>",
  DataType.cu64: "cutlass::complex<cutlass::uint64_t>",
  DataType.cs4: "cutlass::complex<cutlass::int4b_t>",
  DataType.cs8: "cutlass::complex<cutlass::int8_t>",
  DataType.cs16: "cutlass::complex<cutlass::int16_t>",
  DataType.cs32: "cutlass::complex<cutlass::int32_t>",
  DataType.cs64: "cutlass::complex<cutlass::int64_t>",
}

DataTypeSize = {
  DataType.void: 0,
  DataType.b1: 1,
  DataType.u4: 4,
  DataType.u8: 8,
  DataType.u16: 16,
  DataType.u32: 32,
  DataType.u64: 64,
  DataType.s4: 4,
  DataType.s8: 8,
  DataType.s16: 16,
  DataType.s32: 32,
  DataType.s64: 64,
  DataType.e4m3: 8,
  DataType.e5m2: 8,
  DataType.f16: 16,
  DataType.bf16: 16,
  DataType.f32: 32,
  DataType.tf32: 32,
  DataType.f64: 64,
  DataType.cf16: 32,
  DataType.cbf16: 32,
  DataType.cf32: 64,
  DataType.ctf32: 32,
  DataType.cf64: 128,
  DataType.cu4: 8,
  DataType.cu8: 16,
  DataType.cu16: 32,
  DataType.cu32: 64,
  DataType.cu64: 128,
  DataType.cs4: 8,
  DataType.cs8: 16,
  DataType.cs16: 32,
  DataType.cs32: 64,
  DataType.cs64: 128,
}

###################################################################################################
#
class BlasMode(enum.Enum):
  symmetric = enum_auto()
  hermitian = enum_auto()

#
BlasModeTag = {
  BlasMode.symmetric: 'cutlass::BlasMode::kSymmetric',
  BlasMode.hermitian: 'cutlass::BlasMode::kHermitian',
}

#
class ComplexTransform(enum.Enum):
  none = enum_auto()
  conj = enum_auto()

#
ComplexTransformTag = {
  ComplexTransform.none: 'cutlass::ComplexTransform::kNone',
  ComplexTransform.conj: 'cutlass::ComplexTransform::kConjugate',
}

# Used for cutlass3x complex kernel collective mainloop builder instantiation
ComplexTransformTag3x = {
  ComplexTransform.none: 'cute::identity',
  ComplexTransform.conj: 'cute::conjugate',
}

#
RealComplexBijection = [
  (DataType.f16, DataType.cf16),
  (DataType.f32, DataType.cf32),
  (DataType.f64, DataType.cf64),
]

#
def is_complex(data_type):
  for r, c in RealComplexBijection:
    if data_type == c:
      return True
  return False

#
def get_complex_from_real(real_type):
  for r, c in RealComplexBijection:
    if real_type == r:
      return c
  return DataType.invalid

#
def get_real_from_complex(complex_type):
  for r, c in RealComplexBijection:
    if complex_type == c:
      return r
  return DataType.invalid

#
class ComplexMultiplyOp(enum.Enum):
  multiply_add = enum_auto()
  gaussian = enum_auto()

###################################################################################################

#
class MathOperation(enum.Enum):
  multiply_add = enum_auto()
  multiply_add_saturate = enum_auto()
  multiply_add_mixed_input_upcast = enum_auto()
  xor_popc = enum_auto()
  and_popc = enum_auto()
  multiply_add_fast_bf16 = enum_auto()
  multiply_add_fast_f16 = enum_auto()
  multiply_add_fast_f32 = enum_auto()
  multiply_add_complex_fast_f32 = enum_auto()
  multiply_add_complex = enum_auto()
  multiply_add_complex_gaussian = enum_auto()
  multiply_add_fast_accum = enum_auto()

#
MathOperationTag = {
  MathOperation.multiply_add: 'cutlass::arch::OpMultiplyAdd',
  MathOperation.multiply_add_saturate: 'cutlass::arch::OpMultiplyAddSaturate',
  MathOperation.multiply_add_mixed_input_upcast: 'cutlass::arch::OpMultiplyAddMixedInputUpcast',
  MathOperation.xor_popc: 'cutlass::arch::OpXorPopc',
  MathOperation.and_popc: 'cutlass::arch::OpAndPopc',
  MathOperation.multiply_add_fast_bf16: 'cutlass::arch::OpMultiplyAddFastBF16',
  MathOperation.multiply_add_fast_f16: 'cutlass::arch::OpMultiplyAddFastF16',
  MathOperation.multiply_add_fast_f32: 'cutlass::arch::OpMultiplyAddFastF32',
  MathOperation.multiply_add_complex_fast_f32: 'cutlass::arch::OpMultiplyAddComplexFastF32',
  MathOperation.multiply_add_complex: 'cutlass::arch::OpMultiplyAddComplex',
  MathOperation.multiply_add_complex_gaussian: 'cutlass::arch::OpMultiplyAddGaussianComplex',
  MathOperation.multiply_add_fast_accum: 'cutlass::arch::OpMultiplyAddFastAccum',
}

###################################################################################################

#
class LayoutType(enum.Enum):
  ColumnMajor = enum_auto()
  RowMajor = enum_auto()
  ColumnMajorInterleaved2 = enum_auto()
  RowMajorInterleaved2 = enum_auto()
  ColumnMajorInterleaved32 = enum_auto()
  RowMajorInterleaved32 = enum_auto()
  ColumnMajorInterleaved64 = enum_auto()
  RowMajorInterleaved64 = enum_auto()
  TensorNWC = enum_auto()
  TensorNHWC = enum_auto()
  TensorNDHWC = enum_auto()
  TensorNCHW = enum_auto()
  TensorNGHWC = enum_auto()
  TensorNC32HW32 = enum_auto()
  TensorNC64HW64 = enum_auto()
  TensorC32RSK32 = enum_auto()
  TensorC64RSK64 = enum_auto()
  TensorKCS = enum_auto()
  TensorKCSR = enum_auto()
  TensorKCSRT = enum_auto()

#
LayoutTag = {
  LayoutType.ColumnMajor: 'cutlass::layout::ColumnMajor',
  LayoutType.RowMajor: 'cutlass::layout::RowMajor',
  LayoutType.ColumnMajorInterleaved2: 'cutlass::layout::ColumnMajorInterleaved<2>',
  LayoutType.RowMajorInterleaved2: 'cutlass::layout::RowMajorInterleaved<2>',
  LayoutType.ColumnMajorInterleaved32: 'cutlass::layout::ColumnMajorInterleaved<32>',
  LayoutType.RowMajorInterleaved32: 'cutlass::layout::RowMajorInterleaved<32>',
  LayoutType.ColumnMajorInterleaved64: 'cutlass::layout::ColumnMajorInterleaved<64>',
  LayoutType.RowMajorInterleaved64: 'cutlass::layout::RowMajorInterleaved<64>',
  LayoutType.TensorNWC: 'cutlass::layout::TensorNWC',
  LayoutType.TensorNHWC: 'cutlass::layout::TensorNHWC',
  LayoutType.TensorNDHWC: 'cutlass::layout::TensorNDHWC',
  LayoutType.TensorNCHW: 'cutlass::layout::TensorNCHW',
  LayoutType.TensorNGHWC: 'cutlass::layout::TensorNGHWC',
  LayoutType.TensorNC32HW32: 'cutlass::layout::TensorNCxHWx<32>',
  LayoutType.TensorC32RSK32: 'cutlass::layout::TensorCxRSKx<32>',
  LayoutType.TensorNC64HW64: 'cutlass::layout::TensorNCxHWx<64>',
  LayoutType.TensorC64RSK64: 'cutlass::layout::TensorCxRSKx<64>',
  LayoutType.TensorKCS: 'cutlass::layout::TensorKCS',
  LayoutType.TensorKCSR: 'cutlass::layout::TensorKCSR',
  LayoutType.TensorKCSRT: 'cutlass::layout::TensorKCSRT'
}

#
TransposedLayout = {
  LayoutType.ColumnMajor: LayoutType.RowMajor,
  LayoutType.RowMajor: LayoutType.ColumnMajor,
  LayoutType.ColumnMajorInterleaved2: LayoutType.RowMajorInterleaved2,
  LayoutType.RowMajorInterleaved2: LayoutType.ColumnMajorInterleaved2,
  LayoutType.ColumnMajorInterleaved32: LayoutType.RowMajorInterleaved32,
  LayoutType.RowMajorInterleaved32: LayoutType.ColumnMajorInterleaved32,
  LayoutType.ColumnMajorInterleaved64: LayoutType.RowMajorInterleaved64,
  LayoutType.RowMajorInterleaved64: LayoutType.ColumnMajorInterleaved64,
  LayoutType.TensorNHWC: LayoutType.TensorNHWC
}

#
ShortLayoutTypeNames = {
  LayoutType.ColumnMajor: 'n',
  LayoutType.ColumnMajorInterleaved2: 'n2',
  LayoutType.ColumnMajorInterleaved32: 'n32',
  LayoutType.ColumnMajorInterleaved64: 'n64',
  LayoutType.RowMajor: 't',
  LayoutType.RowMajorInterleaved2: 't2',
  LayoutType.RowMajorInterleaved32: 't32',
  LayoutType.RowMajorInterleaved64: 't64',
  LayoutType.TensorNWC: 'nwc',
  LayoutType.TensorNHWC: 'nhwc',
  LayoutType.TensorNDHWC: 'ndhwc',
  LayoutType.TensorNCHW: 'nchw',
  LayoutType.TensorNGHWC: 'nghwc',
  LayoutType.TensorNC32HW32: 'nc32hw32',
  LayoutType.TensorNC64HW64: 'nc64hw64',
  LayoutType.TensorC32RSK32: 'c32rsk32',
  LayoutType.TensorC64RSK64: 'c64rsk64',
  LayoutType.TensorKCS: 'kcs',
  LayoutType.TensorKCSR: 'kcsr',
  LayoutType.TensorKCSRT: 'kcsrt'
}

#
ShortComplexLayoutNames = {
  (LayoutType.ColumnMajor, ComplexTransform.none): 'n',
  (LayoutType.ColumnMajor, ComplexTransform.conj): 'c',
  (LayoutType.RowMajor, ComplexTransform.none): 't',
  (LayoutType.RowMajor, ComplexTransform.conj): 'h'
}

###################################################################################################
class KernelScheduleType(enum.Enum):
  ScheduleAuto = enum_auto()
  Multistage = enum_auto()
  CpAsyncWarpSpecialized = enum_auto()
  CpAsyncWarpSpecializedPingpong = enum_auto()
  CpAsyncWarpSpecializedCooperative = enum_auto()
  Tma = enum_auto()
  TmaWarpSpecialized = enum_auto()
  TmaWarpSpecializedPingpong = enum_auto()
  TmaWarpSpecializedCooperative = enum_auto()
  TmaWarpSpecializedFP8FastAccum = enum_auto()
  TmaWarpSpecializedCooperativeFP8FastAccum = enum_auto()
  TmaWarpSpecializedPingpongFP8FastAccum = enum_auto()
  ImplicitTmaWarpSpecializedSm90 = enum_auto()
#
KernelScheduleTag = {
  KernelScheduleType.ScheduleAuto: 'cutlass::gemm::collective::KernelScheduleAuto',
  KernelScheduleType.Multistage: 'cutlass::gemm::KernelMultistage',
  KernelScheduleType.CpAsyncWarpSpecialized: 'cutlass::gemm::KernelCpAsyncWarpSpecialized',
  KernelScheduleType.CpAsyncWarpSpecializedPingpong: 'cutlass::gemm::KernelCpAsyncWarpSpecializedPingpong',
  KernelScheduleType.CpAsyncWarpSpecializedCooperative: 'cutlass::gemm::KernelCpAsyncWarpSpecializedCooperative',
  KernelScheduleType.Tma: 'cutlass::gemm::KernelTma',
  KernelScheduleType.TmaWarpSpecialized: 'cutlass::gemm::KernelTmaWarpSpecialized',
  KernelScheduleType.TmaWarpSpecializedPingpong: 'cutlass::gemm::KernelTmaWarpSpecializedPingpong',
  KernelScheduleType.TmaWarpSpecializedCooperative: 'cutlass::gemm::KernelTmaWarpSpecializedCooperative',
  KernelScheduleType.TmaWarpSpecializedFP8FastAccum: 'cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum',
  KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum: 'cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum',
  KernelScheduleType.TmaWarpSpecializedPingpongFP8FastAccum: 'cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum',
  KernelScheduleType.ImplicitTmaWarpSpecializedSm90: 'cutlass::conv::KernelImplicitTmaWarpSpecializedSm90',
}

#
KernelScheduleSuffixes = {
  KernelScheduleType.ScheduleAuto: '',
  KernelScheduleType.Multistage: '_cpasync',
  KernelScheduleType.CpAsyncWarpSpecialized: '_cpasync_warpspecialized',
  KernelScheduleType.CpAsyncWarpSpecializedPingpong: '_cpasync_warpspecialized_pingpong',
  KernelScheduleType.CpAsyncWarpSpecializedCooperative: '_cpasync_warpspecialized_cooperative',
  KernelScheduleType.Tma: '_unspecialized',
  KernelScheduleType.TmaWarpSpecialized: '_warpspecialized',
  KernelScheduleType.TmaWarpSpecializedPingpong: '_warpspecialized_pingpong',
  KernelScheduleType.TmaWarpSpecializedCooperative: '_warpspecialized_cooperative',
  KernelScheduleType.TmaWarpSpecializedFP8FastAccum: '_warpspecialized_fp8_fastaccum',
  KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum: '_warpspecialized_cooperative_fp8_fastaccum',
  KernelScheduleType.TmaWarpSpecializedPingpongFP8FastAccum: '_warpspecialized_pingpong_fp8_fastaccum',
  KernelScheduleType.ImplicitTmaWarpSpecializedSm90: '_warpspecialized',
}

class EpilogueScheduleType(enum.Enum):
  ScheduleAuto = enum_auto()
  EpilogueTransposed = enum_auto()
  NoSmemWarpSpecialized = enum_auto()
  TmaWarpSpecialized = enum_auto()
  TmaWarpSpecializedCooperative = enum_auto()
#
EpilogueScheduleTag = {
  EpilogueScheduleType.ScheduleAuto: 'cutlass::epilogue::collective::EpilogueScheduleAuto',
  EpilogueScheduleType.EpilogueTransposed: 'cutlass::gemm::EpilogueTransposed',
  EpilogueScheduleType.NoSmemWarpSpecialized: 'cutlass::epilogue::NoSmemWarpSpecialized',
  EpilogueScheduleType.TmaWarpSpecialized: 'cutlass::epilogue::TmaWarpSpecialized',
  EpilogueScheduleType.TmaWarpSpecializedCooperative: 'cutlass::epilogue::TmaWarpSpecializedCooperative',
}

#
EpilogueScheduleSuffixes = {
  EpilogueScheduleType.ScheduleAuto: '',
  EpilogueScheduleType.EpilogueTransposed: '',
  EpilogueScheduleType.NoSmemWarpSpecialized: '_epi_nosmem',
  EpilogueScheduleType.TmaWarpSpecialized: '_epi_tma',
  EpilogueScheduleType.TmaWarpSpecializedCooperative: '_epi_tma',
}

class EpilogueFunctor3x(enum.Enum):
  LinearCombination = enum_auto()
#
EpilogueFunctor3xTag = {
  EpilogueFunctor3x.LinearCombination: 'cutlass::epilogue::fusion::LinearCombination',
}

class TileSchedulerType(enum.Enum):
  Default = enum_auto()
  Persistent = enum_auto()
  StreamK = enum_auto()
#
TileSchedulerTag = {
  TileSchedulerType.Default: 'void',
  TileSchedulerType.Persistent: 'cutlass::gemm::PersistentScheduler',
  TileSchedulerType.StreamK: 'cutlass::gemm::StreamKScheduler',
}

#
TileSchedulerSuffixes = {
  TileSchedulerType.Default: '',
  TileSchedulerType.Persistent: '',
  TileSchedulerType.StreamK: '_stream_k',
}

###################################################################################################

#
class SideMode(enum.Enum):
  Left = enum_auto()
  Right = enum_auto()

#
SideModeTag = {
  SideMode.Left: 'cutlass::SideMode::kLeft',
  SideMode.Right: 'cutlass::SideMode::kRight'
}

#
ShortSideModeNames = {
  SideMode.Left: 'ls',
  SideMode.Right: 'rs'
}

###################################################################################################

#
class FillMode(enum.Enum):
  Lower = enum_auto()
  Upper = enum_auto()

#
FillModeTag = {
  FillMode.Lower: 'cutlass::FillMode::kLower',
  FillMode.Upper: 'cutlass::FillMode::kUpper'
}

#
ShortFillModeNames = {
  FillMode.Lower: 'l',
  FillMode.Upper: 'u'
}

###################################################################################################

#
class DiagType(enum.Enum):
  NonUnit = enum_auto()
  Unit = enum_auto()

#
DiagTypeTag = {
  DiagType.NonUnit: 'cutlass::DiagType::kNonUnit',
  DiagType.Unit: 'cutlass::DiagType::kUnit'
}

#
ShortDiagTypeNames = {
  DiagType.NonUnit: 'nu',
  DiagType.Unit: 'un'
}

###################################################################################################

#
class OpcodeClass(enum.Enum):
  Simt = enum_auto()
  TensorOp = enum_auto()
  WmmaTensorOp = enum_auto()
  SparseTensorOp = enum_auto()

OpcodeClassNames = {
  OpcodeClass.Simt: 'simt',
  OpcodeClass.TensorOp: 'tensorop',
  OpcodeClass.WmmaTensorOp: 'wmma_tensorop',
}

OpcodeClassTag = {
  OpcodeClass.Simt: 'cutlass::arch::OpClassSimt',
  OpcodeClass.TensorOp: 'cutlass::arch::OpClassTensorOp',
  OpcodeClass.WmmaTensorOp: 'cutlass::arch::OpClassWmmaTensorOp',
}

###################################################################################################

#
class OperationKind(enum.Enum):
  Gemm = enum_auto()
  RankK = enum_auto()
  Rank2K = enum_auto()
  Trmm = enum_auto()
  Symm = enum_auto()
  Conv2d = enum_auto()
  Conv3d = enum_auto()

#
OperationKindNames = {
  OperationKind.Gemm: 'gemm'
  , OperationKind.RankK: 'rank_k'
  , OperationKind.Rank2K: 'rank_2k'
  , OperationKind.Trmm: 'trmm'
  , OperationKind.Symm: 'symm'
  , OperationKind.Conv2d: 'conv2d'
  , OperationKind.Conv3d: 'conv3d'
}

#
class Target(enum.Enum):
  library = enum_auto()
#
ArchitectureNames = {
  50: 'maxwell',
  60: 'pascal',
  61: 'pascal',
  70: 'volta',
  75: 'turing',
  80: 'ampere',
  89: 'ada',
  90: 'hopper'
}

#
SharedMemPerCC = {
  70:  96, #  96KB of SMEM
  72:  96, #  96KB of SMEM
  75:  64, #  64KB of SMEM
  80: 163, # 163KB of SMEM - 1KB reserved for the driver
  86:  99, #  99KB of SMEM - 1KB reserved for the driver
  87: 163, # 163KB of SMEM - 1KB reserved for the driver
  89:  99, #  99KB of SMEM - 1KB reserved for the driver
  90: 227, # 227KB of SMEM - 1KB reserved for the driver
}

###################################################################################################

#
def SubstituteTemplate(template, values):
  text = template
  changed = True
  while changed:
    changed = False
    for key, value in values.items():
      regex = "\\$\\{%s\\}" % key
      newtext = re.sub(regex, value, text)
      if newtext != text:
        changed = True
      text = newtext
  return text

###################################################################################################

#
class GemmKind(enum.Enum):
  Gemm = enum_auto()
  Sparse = enum_auto()
  Universal = enum_auto()
  Universal3x = enum_auto()
  SparseUniversal3x = enum_auto()
  PlanarComplex = enum_auto()
  PlanarComplexArray = enum_auto()
  Grouped = enum_auto()
#
GemmKindNames = {
  GemmKind.Gemm: "gemm",
  GemmKind.Sparse: "spgemm",
  GemmKind.Universal: "gemm",
  GemmKind.Universal3x: "gemm",
  GemmKind.SparseUniversal3x: "spgemm",
  GemmKind.PlanarComplex: "gemm_planar_complex",
  GemmKind.PlanarComplexArray: "gemm_planar_complex_array",
  GemmKind.Grouped: "gemm_grouped",
}

#
class RankKKind(enum.Enum):
  Universal = enum_auto()

#
RankKKindNames = {
  RankKKind.Universal: "rank_k"
}

#
class TrmmKind(enum.Enum):
  Universal = enum_auto()

#
TrmmKindNames = {
  TrmmKind.Universal: "trmm"
}

#
class SymmKind(enum.Enum):
  Universal = enum_auto()

#
SymmKindNames = {
  SymmKind.Universal: "symm"
}

#
class EpilogueFunctor(enum.Enum):
  LinearCombination = enum_auto()
  LinearCombinationClamp = enum_auto()

#
EpilogueFunctorTag = {
  EpilogueFunctor.LinearCombination: 'cutlass::epilogue::thread::LinearCombination',
  EpilogueFunctor.LinearCombinationClamp: 'cutlass::epilogue::thread::LinearCombinationClamp',
}

#
class SwizzlingFunctor(enum.Enum):
  Identity1 = enum_auto()
  Identity2 = enum_auto()
  Identity4 = enum_auto()
  Identity8 = enum_auto()
  Horizontal = enum_auto()
  StridedDgradIdentity1 = enum_auto()
  StridedDgradIdentity4 = enum_auto()
  StridedDgradHorizontal = enum_auto()
  StreamK = enum_auto()

#
SwizzlingFunctorTag = {
  SwizzlingFunctor.Identity1: 'cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>',
  SwizzlingFunctor.Identity2: 'cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>',
  SwizzlingFunctor.Identity4: 'cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>',
  SwizzlingFunctor.Identity8: 'cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>',
  SwizzlingFunctor.Horizontal: 'cutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle',
  SwizzlingFunctor.StridedDgradIdentity1: 'cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<1>',
  SwizzlingFunctor.StridedDgradIdentity4: 'cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<4>',
  SwizzlingFunctor.StridedDgradHorizontal: 'cutlass::conv::threadblock::StridedDgradHorizontalThreadblockSwizzle',
  SwizzlingFunctor.StreamK: 'cutlass::gemm::threadblock::ThreadblockSwizzleStreamK',
}

#
class GroupScheduleMode(enum.Enum):
  Device = enum_auto(),
  Host = enum_auto()

#
GroupScheduleModeTag = {
  GroupScheduleMode.Device: 'cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly',
  GroupScheduleMode.Host: 'cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute'
}

#
ShortGroupScheduleModeNames = {
  GroupScheduleMode.Device: 'Device',
  GroupScheduleMode.Host: 'Host'
}

###################################################################################################

#
class ConvKind(enum.IntEnum):
  Fprop = 0
  Dgrad = 1
  Wgrad = 2

#
ConvKindTag = {
  ConvKind.Fprop: 'cutlass::conv::Operator::kFprop',
  ConvKind.Dgrad: 'cutlass::conv::Operator::kDgrad',
  ConvKind.Wgrad: 'cutlass::conv::Operator::kWgrad'
}

ConvKindNames = {
  ConvKind.Fprop: 'fprop',
  ConvKind.Dgrad: 'dgrad',
  ConvKind.Wgrad: 'wgrad',
}

class ConvMode(enum.IntEnum):
  CrossCorrelation = 0
  Convolution = 1

#
class IteratorAlgorithm(enum.Enum):
  Analytic = 0
  Optimized = 1
  FixedChannels = 2
  FewChannels = 3
  FixedStrideDilation = 4

#
IteratorAlgorithmTag = {
  IteratorAlgorithm.Analytic: 'cutlass::conv::IteratorAlgorithm::kAnalytic',
  IteratorAlgorithm.Optimized: 'cutlass::conv::IteratorAlgorithm::kOptimized',
  IteratorAlgorithm.FixedChannels: 'cutlass::conv::IteratorAlgorithm::kFixedChannels',
  IteratorAlgorithm.FewChannels: 'cutlass::conv::IteratorAlgorithm::kFewChannels',
  IteratorAlgorithm.FixedStrideDilation: 'cutlass::conv::IteratorAlgorithm::kFixedStrideDilation'
}

IteratorAlgorithmNames = {
  IteratorAlgorithm.Analytic: 'analytic',
  IteratorAlgorithm.Optimized: 'optimized',
  IteratorAlgorithm.FixedChannels: 'fixed_channels',
  IteratorAlgorithm.FewChannels: 'few_channels',
  IteratorAlgorithm.FixedStrideDilation: 'fixed_stride_dilation'
}

#
class StrideSupport(enum.Enum):
  Strided = 0
  Unity = 1
  Fixed = 2

#
StrideSupportTag = {
  StrideSupport.Strided: 'cutlass::conv::StrideSupport::kStrided',
  StrideSupport.Unity: 'cutlass::conv::StrideSupport::kUnity',
  StrideSupport.Fixed: 'cutlass::conv::StrideSupport::kFixed'
}

StrideSupportNames = {
  StrideSupport.Strided: '',
  StrideSupport.Unity: 'unity_stride',
  StrideSupport.Fixed: 'fixed_stride'
}

#
class GroupMode(enum.Enum):
  NoneGroup = enum_auto()         # dense conv (G=1)
  SingleGroup = enum_auto()       # grouped convolution (single group per CTA)
  MultipleGroup = enum_auto()     # grouped convolution ( multiple groups per CTA)
  Depthwise = enum_auto()         # Depthwise convolution ( C=K=G )

#
GroupModeTag = {
  GroupMode.NoneGroup: 'cutlass::conv::GroupMode::kNone',
  GroupMode.SingleGroup: 'cutlass::conv::GroupMode::kSingleGroup',
  GroupMode.MultipleGroup: 'cutlass::conv::GroupMode::kMultipleGroup',
  GroupMode.Depthwise: 'cutlass::conv::GroupMode::kDepthwise',
}

GroupModeNames = {
  GroupMode.NoneGroup: '',
  GroupMode.SingleGroup: 'single_group',
  GroupMode.MultipleGroup: 'multiple_group',
  GroupMode.Depthwise: 'depthwise',
}

###################################################################################################

#
class MathInstruction:
  def __init__(self,
      instruction_shape,                                            \
      element_a, element_b, element_accumulator,                    \
      opcode_class, math_operation = MathOperation.multiply_add     \
    ):

    self.instruction_shape = instruction_shape
    self.element_a = element_a
    self.element_b = element_b
    self.element_accumulator = element_accumulator
    self.opcode_class = opcode_class
    self.math_operation = math_operation
#
class TileDescription:

  def __init__(self, threadblock_shape, stages, warp_count, math_instruction, min_compute, max_compute, cluster_shape = [1,1,1]):
    self.threadblock_shape = threadblock_shape
    self.tile_shape = threadblock_shape
    self.stages = stages
    self.warp_count = warp_count
    self.math_instruction = math_instruction
    self.minimum_compute_capability = min_compute
    self.maximum_compute_capability = max_compute
    self.cluster_shape = cluster_shape

  def procedural_name(self):
    if self.minimum_compute_capability >= 90:
      return "{tbm}x{tbn}x{tbk}_{cm}x{cn}x{ck}_{s}".format(
        tbm = self.threadblock_shape[0],
        tbn = self.threadblock_shape[1],
        tbk = self.threadblock_shape[2],
        cm = self.cluster_shape[0],
        cn = self.cluster_shape[1],
        ck = self.cluster_shape[2],
        s = self.stages)
    else:
      return "%dx%d_%dx%d" % (self.threadblock_shape[0], self.threadblock_shape[1], self.threadblock_shape[2], self.stages)

#
class Direct2dConvFixedStrideDilationTileDescription:
  def __init__(self, threadblock_output_shape, filter_shape, stages, stride, dilation, warp_count, math_instruction, min_compute, max_compute):
    self.threadblock_shape = [threadblock_output_shape[0]*threadblock_output_shape[1]*threadblock_output_shape[2], threadblock_output_shape[3], filter_shape[0]*filter_shape[1]]
    self.threadblock_output_shape = threadblock_output_shape
    self.filter_shape = filter_shape
    self.stages = stages
    self.warp_count = warp_count
    self.stride = stride
    self.dilation =  dilation
    self.math_instruction = math_instruction
    self.minimum_compute_capability = min_compute
    self.maximum_compute_capability = max_compute

  def procedural_name(self):
    str_name = "%dx%dx%d_%dx%dx%dx%d_%d_filter%dx%d" % (self.threadblock_shape[0],
                                      self.threadblock_shape[1],
                                      self.threadblock_shape[2],
                                      self.threadblock_output_shape[0],
                                      self.threadblock_output_shape[1],
                                      self.threadblock_output_shape[2],
                                      self.threadblock_output_shape[3],
                                      self.stages,
                                      self.filter_shape[0],
                                      self.filter_shape[1])
    # Fixed Strided and dilation
    if self.stride != [-1, -1] and self.dilation != [-1, -1]:
      str_name += "_stride%dx%d_dilation%dx%d" % (self.stride[0],
                                                  self.stride[1],
                                                  self.dilation[0],
                                                  self.dilation[1])
    return str_name

#
class Direct2dConvFixedStrideDilationTileDescription:
  def __init__(self, threadblock_output_shape, filter_shape, stages, stride, dilation, warp_count, math_instruction, min_compute, max_compute):
    self.threadblock_shape = [threadblock_output_shape[0]*threadblock_output_shape[1]*threadblock_output_shape[2], threadblock_output_shape[3], filter_shape[0]*filter_shape[1]]
    self.threadblock_output_shape = threadblock_output_shape
    self.filter_shape = filter_shape
    self.stages = stages
    self.warp_count = warp_count
    self.stride = stride
    self.dilation =  dilation
    self.math_instruction = math_instruction
    self.minimum_compute_capability = min_compute
    self.maximum_compute_capability = max_compute

  def procedural_name(self):
    str_name = "%dx%dx%d_%dx%dx%dx%d_%d_filter%dx%d" % (self.threadblock_shape[0],
                                      self.threadblock_shape[1],
                                      self.threadblock_shape[2],
                                      self.threadblock_output_shape[0],
                                      self.threadblock_output_shape[1],
                                      self.threadblock_output_shape[2],
                                      self.threadblock_output_shape[3],
                                      self.stages,
                                      self.filter_shape[0],
                                      self.filter_shape[1])
    # Fixed Strided and dilation
    if self.stride != [-1, -1] and self.dilation != [-1, -1]:
      str_name += "_stride%dx%d_dilation%dx%d" % (self.stride[0],
                                                  self.stride[1],
                                                  self.dilation[0],
                                                  self.dilation[1])
    return str_name

#
class TensorDescription:
  def __init__(self, element, layout, alignment = 1, complex_transform = ComplexTransform.none):
    self.element = element
    self.layout = layout
    self.alignment = alignment
    self.complex_transform = complex_transform

#
class SymmetricTensorDescription:
  def __init__(self, element, layout, fill_mode, alignment = 1, complex_transform = ComplexTransform.none, side_mode = SideMode.Left):
    self.element = element
    self.layout = layout
    self.fill_mode = fill_mode
    self.alignment = alignment
    self.complex_transform = complex_transform
    self.side_mode = side_mode

#
class TriangularTensorDescription:
  def __init__(self, element, layout, side_mode, fill_mode, diag_type, alignment = 1, complex_transform = ComplexTransform.none):
    self.element = element
    self.layout = layout
    self.side_mode = side_mode
    self.fill_mode = fill_mode
    self.diag_type = diag_type
    self.alignment = alignment
    self.complex_transform = complex_transform

#
def CalculateSmemUsage(operation):
  cta_shape = operation.tile_description.threadblock_shape
  stages = operation.tile_description.stages

  if operation.operation_kind == OperationKind.Gemm and operation.gemm_kind == GemmKind.Sparse:
    # Elements represented by 8 bits of metadata (based on 4:8, 2:4 or 1:2 sparsity)
    if DataTypeSize[operation.A.element] == 32:
      elements_per_8b_md = 2
    elif DataTypeSize[operation.A.element] == 4:
      elements_per_8b_md = 8
    else:
      elements_per_8b_md = 4

    smem_per_stage = DataTypeSize[operation.A.element] * cta_shape[0] * (cta_shape[2] // 2) // 8 + \
                     DataTypeSize[operation.B.element] * cta_shape[1] * cta_shape[2] // 8 + \
                     cta_shape[0] * (cta_shape[2] // 2) // elements_per_8b_md
  else:
    # Few BLAS3 operations only have A tensor
    data_type_size_a = DataTypeSize[operation.A.element]
    data_type_size_b = DataTypeSize[operation.A.element]
    if operation.is_mixed_input():
      data_type_size_b = DataTypeSize[operation.B.element]

    smem_per_stage = data_type_size_a * cta_shape[0] * cta_shape[2] // 8 + \
                     data_type_size_b * cta_shape[1] * cta_shape[2] // 8

  smem_usage = smem_per_stage * stages
  return (smem_usage >> 10)


class GemmUniversalMode(enum.IntEnum):
  """
  Types corresponding to GemmUniversalMode
  """
  Gemm = 0
  GemmSplitKParallel = 1
  Batched = 2
  Array = 3


class SplitKMode(enum.IntEnum):
  """
  Types corresponding to SplitKMode
  """
  NoneSplitK = 0
  Serial = 1
  Parallel = 2
