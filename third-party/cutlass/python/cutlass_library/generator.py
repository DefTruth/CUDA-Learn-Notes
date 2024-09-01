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
Utilities for enumerating CUTLASS library kernels
"""

import argparse
import enum
from itertools import product
import logging
import os.path
import shutil
import sys
import copy
from typing import Any, Optional, Sequence, Tuple

_LOGGER = logging.getLogger(__name__)

def logging_prefix(indent_level: int = 0) -> str:
  """String prefix for start of each debug log entry"""
  prefix = '*** '
  indent = '  '
  return f"{prefix}{indent_level * indent}"

def log_debug_line(line: str, indent_level: int = 0) -> None:
  """Log one line of debug output"""
  prefix = logging_prefix(indent_level)
  _LOGGER.debug(prefix + line)

# Certain usecases of cutlass_library nearly always prefer to run as scripts with
# relative imports, rather than via an installed Python package. An example of this
# is using CUTLASS's CMake system to generate a library of kernels to be profiled.
# To make it easy to use these use cases when an existing installation of cutlass_library
# exists, this global flag can be set to true (via command-line arguments) to ensure
# that package-based installations are not used.

# Create a temporary argument parser to check only for the availability of the
# --disable-cutlass-package-imports argument, which controls whether package-based
# imports are disabled.
def _add_package_disablement_flag(argparser):
  argparser.add_argument("--disable-cutlass-package-imports", action='store_true', required=False,
                     help="Disable use of cutlass_library from Python package")

_parser = argparse.ArgumentParser()
_add_package_disablement_flag(_parser)
_args, _ = _parser.parse_known_args()

# Add `CUTLASS_IGNORE_PACKAGE` to `builtins` so that it is visible for gating future
# imports without requiring importing another module. Ideally, we would just place this
# as a global variable in a module to that could be imported and checked (e.g.,
# utils.CUTLASS_IGNORE_PACKAGE). However, this raises the issue of determining
# where this module should be sourced (from the cutlass_library package or from
# a relative import), which is the problem this variable is being used to solve in the
# first place.
import builtins
builtins.CUTLASS_IGNORE_PACKAGE = _args.disable_cutlass_package_imports

try:
  if CUTLASS_IGNORE_PACKAGE:
    raise ImportError("Disabling attempt to import cutlass_library")
  from cutlass_library.library import *
  from cutlass_library.manifest import *
except ImportError:
  from library import *
  from manifest import *
###################################################################################################

#
def CudaToolkitVersionSatisfies(semantic_ver_string, major, minor, patch = 0):

  # by default, use the latest CUDA Toolkit version
  cuda_version = [11, 0, 132]

  # Update cuda_version based on parsed string
  if semantic_ver_string != '':
    for i, x in enumerate([int(x) for x in semantic_ver_string.split('.')]):
      if i < len(cuda_version):
        cuda_version[i] = x
      else:
        cuda_version.append(x)
  return cuda_version >= [major, minor, patch]


###################################################################################################
###################################################################################################

#
def EpilogueAlignment(max_alignment, tile, epilogue_steps = 8):
  ''' Helper to compute the maximum alignment of the epilogue '''

  def product(X, identity = 1):
    result = identity
    for item in X:
      result *= item
    return result

  elements_per_thread = product(tile.threadblock_shape[:-1]) // product(tile.warp_count) // 32 // epilogue_steps
  return min(max_alignment, elements_per_thread)

def DefaultSwizzlingFunctor():
    return SwizzlingFunctor.Identity8
    # To use StreamK decomposition for basic GEMMs, set `swizzling_functor = SwizzlingFunctor.StreamK`

#
def CreateGemmOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms = None, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = DefaultSwizzlingFunctor()):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for tile_description in tile_descriptions:
      for alignment in alignment_constraints:
        for complex_transform in complex_transforms:

            # If alignment is a tuple or a list, then we have different alignments for A and B
            alignment_a = alignment if isinstance(alignment, int) else alignment[0]
            alignment_b = alignment if isinstance(alignment, int) else alignment[1]
            alignment_c = min(8, alignment_a) if isinstance(alignment, int) else alignment[2]

            A = TensorDescription(element_a, layout[0], alignment_a, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment_b, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            new_operation = GemmOperation(GemmKind.Universal, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)

            manifest.append(new_operation)
            operations.append(new_operation)

  return operations

# Generates 3.0 API based GemmUniversal API kernels. Alignment constraints are folded in with layouts
def CreateGemmUniversal3xOperator(
    manifest, layouts, tile_descriptions, data_types,
    schedules = [[KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto]],
    complex_transforms=None,
    epilogue_functor=EpilogueFunctor.LinearCombination,
    swizzling_functor=SwizzlingFunctor.Identity1,
    tile_schedulers=[TileSchedulerType.Persistent]):

  if type(data_types) is dict:
    data_types = [data_types]

  for s in schedules:
    assert(len(s) == 2)

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none), ]

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0]]

  combinations = product(layouts, tile_descriptions, data_types, complex_transforms, schedules, tile_schedulers)
  for layout, tile_description, data_type, complex_transform, schedules, tile_scheduler in combinations:
    kernel_schedule, epilogue_schedule = schedules
    A = TensorDescription(
        data_type["a_type"], layout[0][0], layout[0][1], complex_transform[0])
    B = TensorDescription(
        data_type["b_type"], layout[1][0], layout[1][1], complex_transform[1])

    C = TensorDescription(data_type["c_type"], layout[2][0], layout[2][1])
    D = TensorDescription(data_type["d_type"], layout[2][0], layout[2][1])

    gemm_op_extra_args = {}
    gemm_kind = GemmKind.Universal3x
    element_compute = data_type.get("epi_type", data_type["acc_type"])

    operation = GemmOperation(
        gemm_kind, tile_description.minimum_compute_capability,
        tile_description, A, B, C, element_compute, epilogue_functor, swizzling_functor, D,
        kernel_schedule, epilogue_schedule, tile_scheduler, **gemm_op_extra_args)

    manifest.append(operation)
    operations.append(operation)

  return operations

#
def CreateSparseGemmOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms = None, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = SwizzlingFunctor.Identity8):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  gemm_kinds = [GemmKind.Sparse]

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for tile_description in tile_descriptions:
      for alignment in alignment_constraints:
        for complex_transform in complex_transforms:

            alignment_c = min(8, alignment)

            A = TensorDescription(element_a, layout[0], alignment, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            new_operation = GemmOperation(GemmKind.Sparse, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)

            manifest.append(new_operation)
            operations.append(new_operation)

  return operations

#
def CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  gemm_kinds = [GemmKind.PlanarComplex, GemmKind.PlanarComplexArray]

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for gemm_kind in gemm_kinds:
    for layout in layouts:
      for tile_description in tile_descriptions:
        for alignment in alignment_constraints:
          for complex_transform in complex_transforms:

            alignment_c = min(8, alignment)

            A = TensorDescription(element_a, layout[0], alignment, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            manifest.append(GemmOperation(gemm_kind, \
              tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue))
  return

#
def CreateGemmGroupedOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms = None, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = SwizzlingFunctor.Identity8):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for tile_description in tile_descriptions:
      for alignment in alignment_constraints:
        for complex_transform in complex_transforms:

            alignment_c = min(8, alignment)

            A = TensorDescription(element_a, layout[0], alignment, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            new_operation = GroupedGemmOperation(GemmKind.Grouped, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)

            manifest.append(new_operation)
            operations.append(new_operation)

  return operations

#
def CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, data_type, \
  alignment_constraints, blas_mode, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = SwizzlingFunctor.Identity8):

  element_a, element_c, element_epilogue = data_type

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for fill_mode in fill_modes:
      for tile_description in tile_descriptions:
        for alignment in alignment_constraints:

          # SERK supported layouts (RowMajor, ColumnMajor) with no conjugation
          complex_transform = ComplexTransform.none

          # HERK supported layouts (RowMajor + conj, ColumnMajor)
          if blas_mode == BlasMode.hermitian and layout[0] == LayoutType.RowMajor:
            complex_transform = ComplexTransform.conj

          alignment_c = 1 # Alignment only applies to A in SYRK

          A = TensorDescription(element_a, layout[0], alignment, complex_transform)
          C = SymmetricTensorDescription(element_c, layout[1], fill_mode, alignment_c)

          # Rank-K update
          new_operation = RankKOperation(RankKKind.Universal, tile_description.minimum_compute_capability, \
            tile_description, A, C, element_epilogue, epilogue_functor, swizzling_functor, blas_mode)

          manifest.append(new_operation)
          operations.append(new_operation)

          # Rank-2K update
          new_operation = Rank2KOperation(RankKKind.Universal, tile_description.minimum_compute_capability, \
            tile_description, A, C, element_epilogue, epilogue_functor, swizzling_functor, blas_mode)

          manifest.append(new_operation)
          operations.append(new_operation)

  return operations

#
def CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms = None, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = SwizzlingFunctor.Identity8):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for side_mode in side_modes:
      for fill_mode in fill_modes:
        for diag_type in diag_types:
          for tile_description in tile_descriptions:
            for alignment in alignment_constraints:
              for complex_transform in complex_transforms:

                  alignment_c = min(8, alignment)

                  A = TriangularTensorDescription(element_a, layout[0], side_mode, fill_mode, diag_type,
                                                  alignment, complex_transform)
                  B = TensorDescription(element_b, layout[1], alignment)
                  C = TensorDescription(element_c, layout[2], alignment_c)

                  new_operation = TrmmOperation(TrmmKind.Universal, tile_description.minimum_compute_capability, \
                    tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)

                  manifest.append(new_operation)
                  operations.append(new_operation)

  return operations

#
def CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, data_type, \
  alignment_constraints, blas_mode, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = SwizzlingFunctor.Identity8):

  element_a, element_b, element_c, element_epilogue = data_type

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for side_mode in side_modes:
      for fill_mode in fill_modes:
        for tile_description in tile_descriptions:
          for alignment in alignment_constraints:

            # SYMM supported layouts (RowMajor, ColumnMajor) with no conjugation
            complex_transform = ComplexTransform.none

            alignment_a = 1 # No vectorized access for the triangular matrix
            alignment_c = min(8, alignment)

            A = SymmetricTensorDescription(element_a, layout[0], fill_mode, alignment_a, complex_transform, side_mode)
            # tensor A and B have same data type and layout
            B = TensorDescription(element_b, layout[0], alignment)
            C = TensorDescription(element_c, layout[1], alignment_c)

            # SYMM/HEMM update
            new_operation = SymmOperation(SymmKind.Universal, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor, blas_mode)

            manifest.append(new_operation)
            operations.append(new_operation)

            # SYMM/HEMM update
            new_operation = SymmOperation(SymmKind.Universal, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor, blas_mode)

            manifest.append(new_operation)
            operations.append(new_operation)

  return operations

###########################################################################################################
#   ConvolutionOperator support variations
#        ____________________________________________________________________
#         ConvolutionalOperator |      Analytic          |    Optimized
#        ____________________________________________________________________
#        |       Fprop          |     (strided)          |    (strided)
#        |       Dgrad          |     (strided, unity*)  |    (strided, unity)
#        |       Wgrad          |     (strided)          |    (strided)
#        ____________________________________________________________________
#
# Note :  Operator marked (*) are supported but not generated to keep the instantiated kernel count low
###########################################################################################################
# Convolution for 2D operations
def CreateConv2dOperator(manifest, layout, tile_descriptions, data_type, alignment_constraints, \
  conv_kinds = [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad], \
  epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity4):

  element_a, element_b, element_c, element_epilogue = data_type

  # one exceptional case

  # iterator algorithm (analytic and optimized)
  iterator_algorithms = [IteratorAlgorithm.Analytic, IteratorAlgorithm.Optimized]

  # by default, only generate the largest tile size, largest alignment, and optimized iterator
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]
    iterator_algorithms = [IteratorAlgorithm.Optimized]

  operations = []

  for tile in tile_descriptions:
    for alignment in alignment_constraints:

      alignment_c = min(8, alignment)

      A = TensorDescription(element_a, layout[0], alignment)
      B = TensorDescription(element_b, layout[1], alignment)
      C = TensorDescription(element_c, layout[2], alignment_c)

      swizzling_functor_ = swizzling_functor

      #
      # Conv2d Fprop
      #
      if ConvKind.Fprop in conv_kinds:

        # Strided support for Analytic and Optimized Fprop
        for iterator_algorithm in iterator_algorithms:
          new_operations = [
            # None grouped kernel
            Conv2dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
              A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_),
          ]

          # Instance group conv kernel
          if tile.math_instruction.opcode_class == OpcodeClass.TensorOp and A.layout == LayoutType.TensorNHWC and \
            tile.minimum_compute_capability >= 80:
            # SingleGroup kernel
            new_operations.append(Conv2dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
              A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_, group_mode=GroupMode.SingleGroup))

            # Analytic iterator supports MultipleGroup mode
            if iterator_algorithm == IteratorAlgorithm.Analytic:
              new_operations.append(Conv2dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
                A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_, group_mode=GroupMode.MultipleGroup))

          for new_operation in new_operations:
            manifest.append(new_operation)
            operations.append(new_operation)

      #
      # Conv2d Dgrad
      #
      if ConvKind.Dgrad in conv_kinds:

        # Unity stride for Analytic and Optimized Dgrad
        for iterator_algorithm in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Dgrad, iterator_algorithm, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Unity, epilogue_functor, swizzling_functor_)

          manifest.append(new_operation)
          operations.append(new_operation)

        # Strided support for Analytic Dgrad
        # strided dgrad uses a special threadblock swizzle
        # note that SwizzlingFunctor.StridedDgradHorizontal might be
        # better for problem sizes with large activation channel count
        swizzling_functor_strided_dgrad_ = SwizzlingFunctor.StridedDgradIdentity1

        if IteratorAlgorithm.Analytic in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Dgrad, IteratorAlgorithm.Analytic, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_strided_dgrad_)

          manifest.append(new_operation)
          operations.append(new_operation)

        # Strided support for Optimized Dgrad
        if IteratorAlgorithm.Optimized in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Dgrad, IteratorAlgorithm.Optimized, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_strided_dgrad_)

          manifest.append(new_operation)
          operations.append(new_operation)

      #
      # Conv2d Wgrad
      #
      if ConvKind.Wgrad in conv_kinds:

        # Strided support for Analytic and Optimized Wgrad
        for iterator_algorithm in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Wgrad, iterator_algorithm, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_)

          manifest.append(new_operation)
          operations.append(new_operation)

  return operations

# Convolution for 2D operations specialized for few channels
def CreateConv2dFixedChannelsOperator(manifest, layout, tile_descriptions, data_type, channel_counts, \
  conv_kinds = [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad], \
  epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity4):

  element_a, element_b, element_c, element_epilogue = data_type

  # one exceptional case

  # iterator algorithm (analytic and optimized)
  iterator_algorithms = [IteratorAlgorithm.FixedChannels,]

  # by default, only generate the largest tile size, largest alignment, and optimized iterator
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    channel_counts = [channel_counts[0],]

  operations = []



  for tile in tile_descriptions:
    for channel_count in channel_counts:

      alignment_c = EpilogueAlignment(channel_count, tile)

      A = TensorDescription(element_a, layout[0], channel_count)
      B = TensorDescription(element_b, layout[1], channel_count)
      C = TensorDescription(element_c, layout[2], alignment_c)

      swizzling_functor_ = swizzling_functor

      #
      # Conv2d Fprop
      #
      if ConvKind.Fprop in conv_kinds:

        # Strided support for Analytic and Optimized Fprop
        for iterator_algorithm in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_)

          manifest.append(new_operation)
          operations.append(new_operation)

  return operations

# Convolution for 2D operations specialized for few channels
def CreateConv2dFewChannelsOperator(manifest, layout, tile_descriptions, data_type, channel_counts, \
  conv_kinds = [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad], \
  epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity4):

  element_a, element_b, element_c, element_epilogue = data_type

  # one exceptional case

  # iterator algorithm (analytic and optimized)
  iterator_algorithms = [IteratorAlgorithm.FewChannels,]

  # by default, only generate the largest tile size, largest alignment, and optimized iterator
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    channel_counts = [channel_counts[0],]

  operations = []

  for tile in tile_descriptions:
    for channel_count in channel_counts:

      alignment_c = EpilogueAlignment(channel_count, tile)

      A = TensorDescription(element_a, layout[0], channel_count)
      B = TensorDescription(element_b, layout[1], channel_count)
      C = TensorDescription(element_c, layout[2], alignment_c)

      swizzling_functor_ = swizzling_functor

      #
      # Conv2d Fprop
      #
      if ConvKind.Fprop in conv_kinds:

        # Strided support for Analytic and Optimized Fprop
        for iterator_algorithm in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_)

          manifest.append(new_operation)
          operations.append(new_operation)

  return operations

# Convolution for 3D operations
def CreateConv3dOperator(manifest, layout, tile_descriptions, data_type, alignment, \
  conv_kinds = [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad], epilogue_functor = EpilogueFunctor.LinearCombination):

  element_a, element_b, element_c, element_epilogue = data_type

  # one exceptional case
  alignment_c = min(8, alignment)

  # iterator algorithm (analytic and optimized)
  iterator_algorithms = [IteratorAlgorithm.Analytic, IteratorAlgorithm.Optimized]

  # by default, only generate the largest tile size and optimized iterators
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    iterator_algorithms = [IteratorAlgorithm.Optimized]

  operations = []

  # All tile sizes for Conv3dFprop and Conv3dWgrad
  for tile in tile_descriptions:
    A = TensorDescription(element_a, layout, alignment)
    B = TensorDescription(element_b, layout, alignment)
    C = TensorDescription(element_c, layout, alignment_c)

    #
    # Conv3d Fprop
    #
    if ConvKind.Fprop in conv_kinds:
      # Strided support for Analytic and Optimized Fprop
      for iterator_algorithm in iterator_algorithms:
        new_operation = Conv3dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
                                        A, B, C, element_epilogue, StrideSupport.Strided)
        manifest.append(new_operation)
        operations.append(new_operation)
    #
    # Conv3d Wgrad
    #
    if ConvKind.Wgrad in conv_kinds:

      # Strided support for Analytic and Optimized Wgrad
      for iterator_algorithm in iterator_algorithms:
        new_operation = Conv3dOperation(ConvKind.Wgrad, iterator_algorithm, tile.minimum_compute_capability, tile,\
          A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor)
        manifest.append(new_operation)
        operations.append(new_operation)

  # All tile sizes for Conv3dDgrad
  for tile in tile_descriptions:

    A = TensorDescription(element_a, layout, alignment)
    B = TensorDescription(element_b, layout, alignment)
    C = TensorDescription(element_c, layout, alignment_c)

    #
    # Conv3d Dgrad
    #
    if ConvKind.Dgrad in conv_kinds:
      # Unity stride for Optimized Dgrad
      new_operation = Conv3dOperation(ConvKind.Dgrad, IteratorAlgorithm.Optimized, tile.minimum_compute_capability, tile,\
        A, B, C, element_epilogue, StrideSupport.Unity, epilogue_functor)

      manifest.append(new_operation)
      operations.append(new_operation)

      # Strided support for Analytic Dgrad
      # Conv3dDgrad has a naive strided support which does not cut down redundant MMAs
      new_operation = Conv3dOperation(ConvKind.Dgrad, IteratorAlgorithm.Analytic, tile.minimum_compute_capability, tile,\
        A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor)

      manifest.append(new_operation)
      operations.append(new_operation)

  return operations

# Convolution for Depthwise 2d conv
def CreateDepthwiseConv2dOperator(manifest, layout, tile_descriptions, data_type, alignment_constraints, \
  conv_kinds = [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad], \
  epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity4):

  element_a, element_b, element_c, element_epilogue = data_type

  # iterator algorithm (FixedStrideDilation, Optimized)
  iterator_algorithms = [IteratorAlgorithm.FixedStrideDilation, IteratorAlgorithm.Optimized]

  # by default, only generate the largest tile size, largest alignment, and optimized iterator
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  operations = []

  for tile in tile_descriptions:
    for alignment in alignment_constraints:

      alignment_c = min(8, alignment)

      A = TensorDescription(element_a, layout[0], alignment)
      B = TensorDescription(element_b, layout[1], alignment)
      C = TensorDescription(element_c, layout[2], alignment_c)

      swizzling_functor_ = swizzling_functor

      if ConvKind.Fprop in conv_kinds:

        # Strided support for Optimized and FixedStridedDilation Depthwise Conv
        for iterator_algorithm in iterator_algorithms:
          stride_support = StrideSupport.Strided
          if iterator_algorithm == IteratorAlgorithm.FixedStrideDilation:
              if tile.stride == [-1, -1] or tile.dilation == [-1,-1]:
                continue
              stride_support = StrideSupport.Fixed

          if iterator_algorithm == IteratorAlgorithm.Optimized:
              if tile.stride != [-1, -1] or tile.dilation != [-1,-1]:
                continue
          new_operation = Conv2dOperation(ConvKind.Fprop,
                                          iterator_algorithm,
                                          tile.minimum_compute_capability,
                                          tile,
                                          A, B, C,
                                          element_epilogue,
                                          stride_support,
                                          epilogue_functor,
                                          swizzling_functor_,
                                          group_mode=GroupMode.Depthwise)

          manifest.append(new_operation)
          operations.append(new_operation)

  return operations

class ConvOperation3x:
  """All parameters of a CUTLASS 3 convolution operation.

  Unlike CUTLASS 2 convolutions, CUTLASS 3 convolutions do not
  distinguish between 2-D and 3-D convolutions by kernel class name.
  Instead, for CUTLASS 3 convolutions, the tensor layouts encode
  whether the convolution is 2-D or 3-D.  Thus, this class deduces
  the OperationKind (either Conv2d or Conv3d) from the layouts,
  rather than taking it as a constructor parameter.
  """
  def __init__(self,
               conv_kind: ConvKind,
               tile_description: TileDescription,
               A: TensorDescription,
               B: TensorDescription,
               C: TensorDescription,
               element_compute: Optional[DataType] = None,
               D: Optional[TensorDescription] = None,
               kernel_schedule: KernelScheduleType = KernelScheduleType.ScheduleAuto,
               epilogue_schedule: EpilogueScheduleType = EpilogueScheduleType.ScheduleAuto,
               tile_scheduler: TileSchedulerType = TileSchedulerType.Default,
               log_indent_level: int = 1):
    log_debug_line(f'ConvOperation3x::init: conv_kind: {conv_kind}', log_indent_level)
    log_indent_level = log_indent_level + 1

    self.conv_kind = conv_kind
    self.tile_description = tile_description
    self.A = A
    self.B = B
    self.C = C
    self.element_compute = C.element if element_compute is None else element_compute
    self.kernel_schedule = kernel_schedule
    self.epilogue_schedule = epilogue_schedule

    self.arch = tile_description.minimum_compute_capability
    self.tile_scheduler = tile_scheduler
    if D == None:
      self.D = C
    else:
      self.D = D

    self.is_3x = True
    self.group_mode = GroupMode.NoneGroup # CUTLASS 3 convolutions currently aren't grouped

    operation_kind = None
    for layout in (A.layout, B.layout, C.layout):
      assert(isinstance(layout, LayoutType))
      new_operation_kind = convolution_tensor_layout_type_to_operation_kind(layout)
      if operation_kind is None:
        operation_kind = new_operation_kind
      else: # CUTLASS 3 convolutions don't permit mixing 2-D and 3-D layouts.
        assert(operation_kind == new_operation_kind)
    assert(operation_kind is not None)
    self.operation_kind = operation_kind

  def __str__(self):
    return f"ConvOperation3x: operation_kind={self.operation_kind}, conv_kind={self.conv_kind}, tile_description={self.tile_description}"

  def is_complex(self):
    complex_operators = [
      MathOperation.multiply_add_complex,
      MathOperation.multiply_add_complex_gaussian,
      MathOperation.multiply_add_complex_fast_f32
    ]
    return self.tile_description.math_instruction.math_operation in complex_operators

  def is_mixed_input(self):
    return self.A.element != self.B.element

  def accumulator_type(self):
    accum = self.tile_description.math_instruction.element_accumulator
    if self.is_complex():
      return get_complex_from_real(accum)
    return accum

  def short_math_name(self):
    prefix = ''
    if self.tile_description.math_instruction.math_operation == MathOperation.multiply_add_complex_gaussian:
      prefix = 'g'
    return prefix + ShortDataTypeNames[self.accumulator_type()]

  def is_tensor_op(self):
    tensor_ops = [
      OpcodeClass.TensorOp,
      OpcodeClass.WmmaTensorOp
    ]
    return self.tile_description.math_instruction.opcode_class in tensor_ops

  def instruction_shape_string(self):
    math_operations_map = {
      MathOperation.xor_popc: 'xor',
      MathOperation.and_popc: 'and'
    }
    if self.is_tensor_op():
      is0, is1, is2 = self.tile_description.math_instruction.instruction_shape
      math_op = self.tile_description.math_instruction.math_operation
      math_op_string = math_operations_map[math_op] if math_op in math_operations_map.keys() else ''
      return f"{is0}x{is1}x{is2}{math_op_string}"
    else:
      return ''

  def intermediate_type_string(self):
    '''
    Name of the distinct intermediate type used by the tensor operation,
    or the empty string if none.

    Tensor ops (opcode_clas *TensorOp) may use an intermediate data type
    that differs from the element type of A or the accumulator type.
    '''
    if not self.is_tensor_op():
      return ''
    elif self.tile_description.math_instruction.element_a == self.A.element:
      return ''
    elif self.tile_description.math_instruction.element_a == self.tile_description.math_instruction.element_accumulator:
      return ''
    else:
      return DataTypeNames[self.tile_description.math_instruction.element_a]

  def core_name(self):
    inst_shape = self.instruction_shape_string()
    intermediate_type = self.intermediate_type_string()
    conv_kind_name = ConvKindNames[self.conv_kind]
    return f"{self.short_math_name()}{inst_shape}{intermediate_type}{conv_kind_name}"

  def extended_name(self):
    core_name = self.core_name()
    element_a = DataTypeNames[self.A.element]
    element_b = DataTypeNames[self.B.element]
    element_acc = DataTypeNames[self.tile_description.math_instruction.element_accumulator]
    element_c = DataTypeNames[self.C.element]
    element_d = DataTypeNames[self.D.element]
    return f"{core_name}_{element_a}_{element_b}_{element_acc}_{element_c}_{element_d}"

  def is_complex(self):
    complex_operators = [
      MathOperation.multiply_add_complex,
      MathOperation.multiply_add_complex_gaussian,
      MathOperation.multiply_add_complex_fast_f32
    ]
    return self.tile_description.math_instruction.math_operation in complex_operators

  def layout_names(self):
    '''Layout strings for A and B, respectively'''
    if self.is_complex():
      return (ShortComplexLayoutNames[(self.A.layout, self.A.complex_transform)],
              ShortComplexLayoutNames[(self.B.layout, self.B.complex_transform)])
    else:
      return (ShortLayoutTypeNames[self.A.layout],
              ShortLayoutTypeNames[self.B.layout])

  def extended_name(self):
    core_name = self.core_name()
    element_a = DataTypeNames[self.A.element]
    element_b = DataTypeNames[self.B.element]
    element_acc = DataTypeNames[self.tile_description.math_instruction.element_accumulator]
    element_c = DataTypeNames[self.C.element]
    element_d = DataTypeNames[self.D.element]
    layout_a, layout_b = self.layout_names()
    return f"{core_name}_{element_a}{layout_a}_{element_b}{layout_b}_{element_acc}_{element_c}_{element_d}"

  def configuration_name(self):
    prefix = 'cutlass3x'
    arch = self.arch
    opcode_class_name = OpcodeClassNames[self.tile_description.math_instruction.opcode_class]
    tbm = self.tile_description.tile_shape[0]
    tbn = self.tile_description.tile_shape[1]
    tbk = self.tile_description.tile_shape[2]
    cm = self.tile_description.cluster_shape[0]
    cn = self.tile_description.cluster_shape[1]
    ck = self.tile_description.cluster_shape[2]
    alignment = max(self.A.alignment, self.B.alignment)
    tile_scheduler = TileSchedulerSuffixes[self.tile_scheduler]
    kernel_schedule = KernelScheduleSuffixes[self.kernel_schedule]
    epilogue_schedule = EpilogueScheduleSuffixes[self.epilogue_schedule]

    return f"{prefix}_sm{arch}_{opcode_class_name}_{self.extended_name()}_{tbm}x{tbn}x{tbk}_{cm}x{cn}x{ck}_{self.tile_description.stages}_align{alignment}{tile_scheduler}{kernel_schedule}{epilogue_schedule}"

  def procedural_name(self):
    return self.configuration_name()

def convolution_tensor_layout_type_to_operation_kind(layout: LayoutType) -> OperationKind:
  if layout == LayoutType.TensorNHWC or layout == LayoutType.TensorKCSR:
    return OperationKind.Conv2d
  elif layout == LayoutType.TensorNDHWC or layout == LayoutType.TensorKCSRT:
    return OperationKind.Conv3d
  else:
    raise RuntimeError(f'LayoutType {layout} does not have a corresponding OperationKind')

def CreateConvOperator3x(manifest: Manifest,
                         dims_and_alignments: Sequence[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]],
                         tile_descriptions: Sequence[Sequence[TileDescription]],
                         data_types,
                         schedule_pairs: Sequence[Tuple[KernelScheduleType, KernelScheduleType]] = \
                           [(KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto)],
                         complex_transforms: Optional[Sequence[ComplexTransform]] = None,
                         tile_schedulers: Sequence[TileSchedulerType] = [TileSchedulerType.Persistent],
                         conv_kind: ConvKind = ConvKind.Fprop,
                         log_indent_level: int = 1):
  """
  Create zero or more CUTLASS 3 two-dimensional convolution operators.

  Create a CUTLASS 3 two-dimensional convolution operator
  for all feasible combinations of the input parameters.
  Add the operators to the manifest.

  dims_and_alignments: 3-level list.  Each outer list term is a list [A, B, C].
    Each inner list (A, B, or C) has the form [num_spatial_dimensions, alignment].
    Both are integers; the first is the number of spatial dimensions
    (currently, only 2 or 3 are supported), and the second is the byte alignment.
    We deduce the operation_kind (either OperationKind.Conv2d or OperationKind.Conv3d)
    from num_spatial_dimensions.

  This function doesn't take layouts, unlike the GEMM functions.
  CUTLASS 3 convolutions currently support three input layouts:

  * TensorNWC for 1-D convolutions,
  * TensorNHWC for 2-D convolutions, and
  * TensorNDHWC for 3-D convolutions.

  Output (C and D) layouts are the same as input layouts,
  except for Wgrad convolutions, where the layouts are

  * TensorKCS for 1-D convolutions,
  * TensorKCSR for 2-D convolutions, and
  * TensorKCSRT for 3-D convolutions.

  The output layouts are completely constrained by the input layouts
  and the convolution kind.

  tile_descriptions: 2-level list.
    Outer level has one list per math instruction.
    Inner level has one TileDescription for each cluster shape.

  data_types: Either a single data_type dictionary, or a list of them.
    Keys: 'a_type', 'b_type', 'c_type', 'd_type', 'acc_type', 'epi_type'

  complex_transforms: Optional list of pairs.
    First element of each pair is the complex transform for A, and
    second element of each pair is the complex transform for B.

  schedule_pairs: [(kernel_schedule, epilogue_schedule), ...]

  conv_kind: Convolution kind (Fprop, Dgrad, or Wgrad).
  """
  log_debug_line('CreateConvOperator3x', log_indent_level)
  log_indent_level = log_indent_level + 1
  log_debug_line(f'conv_kind: {conv_kind}', log_indent_level)

  for triple in dims_and_alignments:
    spatial_dimensionality = None # to be determined by loop below
    assert(len(triple) == 3)
    for entry in triple: # [A, B, C]
      assert(len(entry) == 2)
      [dim, alignment] = entry
      assert(type(dim) is int)
      assert(dim == 2 or dim == 3)
      assert(type(alignment) is int)
      assert(alignment > 0)
      if spatial_dimensionality is None:
        spatial_dimensionality = dim
      else:
        # A, B, and C need to have the same spatial dimensionality
        assert(spatial_dimensionality == dim)

  def input_and_output_layouts(spatial_dim: int, kind: ConvKind) -> Tuple[LayoutType, LayoutType]:
    if spatial_dim == 1:
      input_layout = LayoutType.TensorNWC
      if kind == ConvKind.Wgrad:
        output_layout = LayoutType.TensorKCS
      else:
        output_layout = input_layout
    elif spatial_dim == 2:
      input_layout = LayoutType.TensorNHWC
      if kind == ConvKind.Wgrad:
        output_layout = LayoutType.TensorKCSR
      else:
        output_layout = input_layout
    elif spatial_dim == 3:
      input_layout = LayoutType.TensorNDHWC
      if kind == ConvKind.Wgrad:
        output_layout = LayoutType.TensorKCSRT
      else:
        output_layout = input_layout
    else:
      assert(False)
    return (input_layout, output_layout)

  def dims_to_layouts(A_B_C: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> \
      Tuple[Tuple[LayoutType, int], Tuple[LayoutType, int], Tuple[LayoutType, int]]:
    [A, B, C] = A_B_C
    [spatial_dim, alignment] = A
    [input_layout, output_layout] = input_and_output_layouts(spatial_dim, conv_kind)
    return ((input_layout, A[1]),
            (input_layout, B[1]),
            (output_layout, C[1]))

  # layouts: list of triples (A, B, C).
  # Each of A, B, and C has the form [layout, alignment].
  layouts = [dims_to_layouts(A_B_C) for A_B_C in dims_and_alignments]

  if type(data_types) is dict:
    data_types = [data_types]

  for s in schedule_pairs:
    assert(len(s) == 2)

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none)]

  # product produces a one-pass generator, so the loop must call it anew each time.
  def make_combinations():
    return product(
      layouts,
      tile_descriptions,
      data_types,
      complex_transforms,
      schedule_pairs,
      tile_schedulers
    )

  operations = []
  for layout_triple, tile_description, data_type, complex_transform_pair, schedule_pair, tile_scheduler in make_combinations():
    A_layout, A_alignment = layout_triple[0]
    A_xform = complex_transform_pair[0]
    B_layout, B_alignment = layout_triple[1]
    B_xform = complex_transform_pair[1]
    C_layout, C_alignment = layout_triple[2]
    D_layout = C_layout
    D_alignment = C_alignment

    A = TensorDescription(data_type["a_type"], A_layout, A_alignment, A_xform)
    B = TensorDescription(data_type["b_type"], B_layout, B_alignment, B_xform)
    C = TensorDescription(data_type["c_type"], C_layout, C_alignment)
    D = TensorDescription(data_type["d_type"], D_layout, D_alignment)
    element_compute = data_type.get("epi_type", data_type["acc_type"])
    kernel_schedule, epilogue_schedule = schedule_pair

    operation = ConvOperation3x(conv_kind=conv_kind,
                                tile_description=tile_description,
                                A=A,
                                B=B,
                                C=C,
                                element_compute=element_compute,
                                D=D,
                                kernel_schedule=kernel_schedule,
                                epilogue_schedule=epilogue_schedule,
                                tile_scheduler=tile_scheduler,
                                log_indent_level=log_indent_level)
    log_debug_line(f'Created ConvOperation3x: {str(operation)}', log_indent_level)
    manifest.append(operation)
    operations.append(operation)

  return operations

###################################################################################################
###################################################################################################

#
def GenerateSM50_Simt(manifest, cuda_version):
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f32, DataType.f32, DataType.f32,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f64, DataType.f64, DataType.f64,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]

  min_cc = 50
  max_cc = 1024

  alignment_constraints = [1,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 128, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 8], 2, [1, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    if math_inst.element_a == DataType.f32:
      conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
      CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#

#
def GenerateSM50_Simt_complex(manifest, cuda_version):
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f32, DataType.f32, DataType.f32,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add_complex),
  ]

  min_cc = 50
  max_cc = 1024

  alignment_constraints = [1,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128,  64, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 8], 2, [1, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      DataType.cf32,
      DataType.cf32,
      DataType.cf32,
      DataType.cf32,
    ]


    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#

#
def GenerateSM50(manifest, cuda_version):
  GenerateSM50_Simt(manifest, cuda_version)
  GenerateSM50_Simt_complex(manifest, cuda_version)

###################################################################################################
###################################################################################################

#
def GenerateSM60_Simt(manifest, cuda_version):
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]

  min_cc = 60
  max_cc = 1024

  alignment_constraints = [1,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 8], 2, [1, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)
#
def GenerateSM60_Simt_DepthwiseConv2d(manifest, cuda_version):

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]

  min_cc = 60
  max_cc = 1024

  alignment_constraints = [8,]

  filter_3x3 = [3, 3]
  filter_5x5 = [5, 5]

  # [stride_h, stride_w]
  # [-1, -1] means all stride size.
  strides = [[-1,-1], [1, 1], [2, 2]]
  # [dilation_h, dilation_w]
  # [-1, -1] means all dilation size.
  dilations = [[-1,-1], [1, 1], [2, 2]]

  #groups per thread block
  g16 = 16
  g32 = 32
  g64 = 64

  #output shape per thread block
  npq_1x4x4 = [1, 4, 4]
  npq_1x8x8 = [1, 8, 8]
  npq_1x10x10 = [1, 10, 10]

  tile_descriptions = []
  for math_inst in math_instructions:
    for stride, dilation in product(strides, dilations):
      tile_descriptions.extend([
        # filter3x3               ThreadBlock_output, filter, stage, warp
        Direct2dConvFixedStrideDilationTileDescription(npq_1x8x8+[g32], filter_3x3, 3, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x8x8+[g64], filter_3x3, 3, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x8x8+[g16], filter_3x3, 3, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),

        Direct2dConvFixedStrideDilationTileDescription(npq_1x10x10+[g64], filter_3x3, 2, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),

        Direct2dConvFixedStrideDilationTileDescription(npq_1x4x4+[g32], filter_3x3, 4, stride, dilation, [4, 1, 1],  math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x4x4+[g64], filter_3x3, 4,  stride, dilation,[4, 1, 1], math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x4x4+[g16], filter_3x3, 4, stride, dilation, [4, 1, 1],  math_inst, min_cc, max_cc),

        # filter5x5               ThreadBlock_output, filter, stage, warp
        Direct2dConvFixedStrideDilationTileDescription(npq_1x8x8+[g32], filter_5x5, 3, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x8x8+[g64], filter_5x5, 3, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x8x8+[g16], filter_5x5, 3, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),

        Direct2dConvFixedStrideDilationTileDescription(npq_1x10x10+[g64], filter_5x5, 2, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),

        Direct2dConvFixedStrideDilationTileDescription(npq_1x4x4+[g32], filter_5x5, 4, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x4x4+[g64], filter_5x5, 4, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x4x4+[g16], filter_5x5, 4, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc)
      ])

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateDepthwiseConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#

#
def GenerateSM60(manifest, cuda_version):
  GenerateSM60_Simt(manifest, cuda_version)
  GenerateSM60_Simt_DepthwiseConv2d(manifest, cuda_version)

###################################################################################################
###################################################################################################

#
def GenerateSM61_Simt(manifest, cuda_version):
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 4],                                      \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]

  min_cc = 61
  max_cc = 1024

  alignment_constraints = [1,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 32], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 32], 2, [1, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]
    data_type_mixed = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_a,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)
#

#
def GenerateSM61(manifest, cuda_version):
  GenerateSM61_Simt(manifest, cuda_version)

###################################################################################################
###################################################################################################

#
def GenerateSM70_TensorOp_884(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 1):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 4],                                      \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [8, 8, 4],                                      \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 70
  max_cc = 75

  alignment_constraints = [8, 4, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints)

      CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, alignment_constraints)

#
def GenerateSM70_PlanarComplexTensorOp_884(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 1):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 4],                                      \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [8, 8, 4],                                      \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 70
  max_cc = 75

  alignment_constraints = [8, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, complex_transforms)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, complex_transforms)


#
def GenerateSM70_WmmaTensorOp_161616(manifest, cuda_version):

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 16, 16],                                   \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.WmmaTensorOp,                       \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 16, 16],                                   \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.WmmaTensorOp,                       \
      MathOperation.multiply_add),
  ]

  min_cc = 70
  max_cc = 1024

  alignment_constraints = [8,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints)

#
##################################################################################################
#

def GenerateSM70(manifest, cuda_version):
  GenerateSM70_TensorOp_884(manifest, cuda_version)
  GenerateSM70_PlanarComplexTensorOp_884(manifest, cuda_version)

  # To limit build size, WMMA GEMMs are disabled for now.
  #
  #GenerateSM70_WmmaTensorOp_161616(manifest, cuda_version)

###################################################################################################
###################################################################################################

#
def GenerateSM75_TensorOp_1688_FewChannels(manifest, cuda_version, math_inst):

  min_cc = 75
  max_cc = 1024

  tile_descriptions = [
    TileDescription([128,  64, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([256,  64, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 128, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64,  64, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 128, 64], 2, [2, 2, 2], math_inst, min_cc, max_cc),
  ]

  data_type = [
    math_inst.element_a,
    math_inst.element_b,
    math_inst.element_accumulator,
    math_inst.element_accumulator,
  ]

  conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)

  CreateConv2dFixedChannelsOperator(manifest, conv_layout, tile_descriptions, data_type, [4, 8])
  CreateConv2dFewChannelsOperator(manifest, conv_layout, tile_descriptions, data_type, [1, 2, 4])

  # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
  if math_inst.element_a != math_inst.element_accumulator:

    data_type_mixed = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_a,
      math_inst.element_accumulator,
    ]

    CreateConv2dFixedChannelsOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, [4, 8])
    CreateConv2dFewChannelsOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, [1, 2, 4])

#
def GenerateSM75_TensorOp_1688(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 2):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 8],                                     \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 8],                                     \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 75
  max_cc = 1024

  alignment_constraints = [8, 4, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 64], 2, [1, 2, 2], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)

    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints)

      CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, alignment_constraints)

    # Separate generator for 'few channels' specializations
    GenerateSM75_TensorOp_1688_FewChannels(manifest, cuda_version, math_inst)

#

#
def GenerateSM75_PlanarComplexTensorOp_1688(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 2):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 8],                                     \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 8],                                     \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 75
  max_cc = 1024

  alignment_constraints = [8, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([ 64, 128, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, complex_transforms)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, complex_transforms)

#
def GenerateSM75_TensorOp_8816_TN(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 2):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 16],                                     \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [8, 8, 16],                                     \
      DataType.u8, DataType.u8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 75
  max_cc = 90

  alignment_constraints = [16,]
  alignment_constraints_small_channels = [16, 8, 4]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 64], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 64], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 64], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 64], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  32, 64], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 256, 64], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 64], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  32, 64], 2, [2, 1, 1], math_inst, min_cc, max_cc),

      TileDescription([256, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 32], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  32, 32], 2, [2, 1, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      DataType.s32,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombination)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        DataType.f32,
      ]

      operations = []

      operations += CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

      operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
        data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

      operations += CreateConv2dFixedChannelsOperator(manifest, conv_layout, tile_descriptions,
        data_type_mixed, alignment_constraints_small_channels, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

      operations += CreateConv2dFewChannelsOperator(manifest, conv_layout, tile_descriptions,
        data_type_mixed, alignment_constraints_small_channels, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

      for op in operations:
        if op.tile_description.threadblock_shape[1] >= 128:
          op.C.alignment = 16
        else:
          op.C.alignment = 8

#

#
def GenerateSM75_TensorOp_8816_Interleaved(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 2):
    return

  layouts = [
    (LayoutType.ColumnMajorInterleaved32, LayoutType.RowMajorInterleaved32, LayoutType.ColumnMajorInterleaved32),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 16],                                     \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [8, 8, 16],                                     \
      DataType.u8, DataType.u8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 75
  max_cc = 90

  alignment_constraints = [16,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 64], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 64], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 64], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 64], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type_mixed = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_a,
      DataType.f32,
    ]

    operations = CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

    conv_layout = (LayoutType.TensorNC32HW32, LayoutType.TensorC32RSK32, LayoutType.TensorNC32HW32)

    operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

    for op in operations:
      op.C.alignment = 8
#

#
def GenerateSM75_TensorOp_8832_TN(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 2):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 32],                                     \
      DataType.s4, DataType.s4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [8, 8, 32],                                     \
      DataType.u4, DataType.u4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 75
  max_cc = 89

  alignment_constraints = [32,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 128], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 128], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 128], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 128], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 128], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      DataType.s32,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombination)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        DataType.f32,
      ]

      operations = []

      operations += CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

      operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
        data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

      for op in operations:
        if op.tile_description.threadblock_shape[1] >= 128:
          op.C.alignment = 16
        elif op.tile_description.threadblock_shape[1] == 64:
          op.C.alignment = 8
        else:
          op.C.alignment = 8

#

#
def GenerateSM75_TensorOp_8832_Interleaved(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 2):
    return

  layouts = [
    (LayoutType.ColumnMajorInterleaved64, LayoutType.RowMajorInterleaved64, LayoutType.ColumnMajorInterleaved64),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 32],                                     \
      DataType.s4, DataType.s4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [8, 8, 32],                                     \
      DataType.u4, DataType.u4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 75
  max_cc = 89

  alignment_constraints = [32,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 128], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 128], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 128], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        DataType.f32,
      ]

      operations = CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

      conv_layout = (LayoutType.TensorNC64HW64, LayoutType.TensorC64RSK64, LayoutType.TensorNC64HW64)

      operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
        data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

      for op in operations:
        op.C.alignment = 16
#

#
def GenerateSM75_TensorOp_88128(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 128],                                   \
      DataType.b1, DataType.b1, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.xor_popc),
  ]

  min_cc = 75
  max_cc = {
    MathOperation.xor_popc: 89,
    MathOperation.and_popc: 90
  }

  alignment_constraints = [128,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 512], 2, [4, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128, 256, 512], 2, [2, 4, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128, 128, 512], 2, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64, 256, 512], 2, [1, 4, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([256,  64, 512], 2, [4, 1, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64, 128, 512], 2, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128,  64, 512], 2, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64,  64, 512], 2, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
    ]

    data_type = [DataType.b1, DataType.b1, DataType.s32, DataType.s32]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

#

#
def GenerateSM75_WmmaTensorOp_161616(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 16, 16],                                   \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.WmmaTensorOp,                       \
      MathOperation.multiply_add),
  ]

  min_cc = 75
  max_cc = 1024

  alignment_constraints = [16,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      DataType.f32,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        DataType.f32,
      ]

      CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints)
#

#
def GenerateSM75_Simt_complex(manifest, cuda_version):
  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f32, DataType.f32, DataType.f32,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add_complex),
  ]

  min_cc = 75
  max_cc = 1024

  alignment_constraints = [1,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 128, 8], 5, [4, 2, 1], math_inst, min_cc, max_cc)
    ]
    data_type = [
      DataType.cf32,
      DataType.cf32,
      DataType.cf32,
      DataType.cf32
    ]

    complex_transforms = [
      (ComplexTransform.none, ComplexTransform.none),
      (ComplexTransform.conj, ComplexTransform.none),
      (ComplexTransform.none, ComplexTransform.conj),
      (ComplexTransform.conj, ComplexTransform.conj)
    ]

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#

def GenerateSM75(manifest, cuda_version):
  GenerateSM75_TensorOp_1688(manifest, cuda_version)
  GenerateSM75_PlanarComplexTensorOp_1688(manifest, cuda_version)
  GenerateSM75_TensorOp_8816_TN(manifest, cuda_version)
  GenerateSM75_TensorOp_8816_Interleaved(manifest, cuda_version)
  GenerateSM75_TensorOp_8832_TN(manifest, cuda_version)
  GenerateSM75_TensorOp_8832_Interleaved(manifest, cuda_version)
  GenerateSM75_TensorOp_88128(manifest, cuda_version)
  #GenerateSM75_WmmaTensorOp_161616(manifest, cuda_version)
  GenerateSM75_Simt_complex(manifest, cuda_version)


###################################################################################################
###################################################################################################

#
def GenerateSM80_TensorOp_16816(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.bf16, DataType.bf16, DataType.f32,     \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [8, 4, 2]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 64],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 64],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 64],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 64],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 64],  3, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    CreateGemmGroupedOperator(manifest, layouts, tile_descriptions, data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
    CreateConv2dFixedChannelsOperator(manifest, conv_layout, tile_descriptions, data_type, [4, 8])
    CreateConv3dOperator(manifest, LayoutType.TensorNDHWC, tile_descriptions, data_type, 8)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints)

      CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, alignment_constraints)
      CreateConv2dFixedChannelsOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, [4, 8])
      CreateConv3dOperator(manifest, LayoutType.TensorNDHWC, tile_descriptions, data_type_mixed, 8)
#

#
def GenerateSM80_SparseTensorOp_16832(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 1):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.RowMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.RowMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.bf16, DataType.bf16, DataType.f32,     \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [8]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([ 64, 128,  64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128,  64],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256,  64],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128,  64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64,  64],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256,  64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64,  64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64,  64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 128],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateSparseGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateSparseGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints)

#

#
def GenerateSM80_PlanarComplexTensorOp_16816(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.bf16, DataType.bf16, DataType.f32,     \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [8, ]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([ 64, 128, 32], 3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, complex_transforms)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, complex_transforms)

#
def GenerateSM80_TensorOp_16816_mixed_input_upcast_a(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  # Upcast on Operand A
  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.s8, DataType.f16, DataType.f16,        \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.s8, DataType.f16, DataType.f32,        \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.u8, DataType.f16, DataType.f32,        \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.u8, DataType.bf16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.s8, DataType.bf16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
  ]

  min_cc = 80
  max_cc = 1024

  # For mixed-input alignment constraints are a list of lists, where the
  # inner list contains the alignment constraints for operands/matrices
  # [[alignA, alignB, alignC],..]
  alignment_constraints = [[16, 8, 8],]

  for math_inst in math_instructions:
    tile_descriptions = [
      # 128x128
      TileDescription([128, 128, 64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      # 128x64
      TileDescription([128, 64, 64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 64, 64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 64, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      # 128x32
      TileDescription([128, 32, 64],  9, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 32, 64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      # 128x16
      TileDescription([128, 16, 64],  5, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 16, 64],  3, [2, 1, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    # streamk uses more regs which can cause spill for the biggest warp tile size when the accumulators are 32bit.
    operations = CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination, SwizzlingFunctor.Identity8)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_b,
        math_inst.element_accumulator,
      ]

      operations += CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombination, SwizzlingFunctor.Identity8)

      for op in operations:
        if (DataTypeSize[op.C.element] == 16) and \
           (op.tile_description.threadblock_shape[1] <= 32):
          op.C.alignment = 4

#
def GenerateSM80_TensorOp_16816_mixed_input_upcast_b(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.s8, DataType.f32,        \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.bf16, DataType.s8, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.u8, DataType.f32,        \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.bf16, DataType.u8, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_mixed_input_upcast),
  ]

  min_cc = 80
  max_cc = 1024

  # For mixed-input alignment constraints are a list of lists, where the
  # inner list contains the alignment constraints for operands/matrices
  # [[alignA, alignB, alignC],..]
  alignment_constraints = [[8, 16, 8],]

  for math_inst in math_instructions:
    tile_descriptions = [
      # 128x128
      TileDescription([128, 128, 64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      # 128x64
      TileDescription([128, 64, 64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 64, 64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 64, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      # 128x32
      TileDescription([128, 32, 64],  9, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 32, 64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 32, 32],  9, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 32, 32],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      # 128x16
      TileDescription([128, 16, 64],  5, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 16, 64],  3, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 16, 32],  9, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 16, 32],  5, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 16, 32],  3, [2, 1, 1], math_inst, min_cc, max_cc),
      # 256x16
      TileDescription([256, 16, 32],  5, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 16, 32],  3, [2, 1, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    # streamk uses more regs which can cause spill for the biggest warp tile size when the accumulators are 32bit.
    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination, SwizzlingFunctor.Identity8)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      operations = CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombination, SwizzlingFunctor.Identity8)

      for op in operations:
        if op.tile_description.threadblock_shape[1] <= 32:
          op.C.alignment = 4

#
def GenerateSM80_TensorOp_16832_TN(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.u8, DataType.u8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 80
  max_cc = 1024
  smem_usage = 164

  alignment_constraints = [16,]
  alignment_constraints_small_channels = [16, 8, 4]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128,  64],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256,  64],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64,  64],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256,  64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  32,  64],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 256,  64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128,  64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64,  64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128,  64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32,  64],  6, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128,  64],  6, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64,  64], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 128],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 128],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  32, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 128],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [math_inst.element_a, math_inst.element_b, math_inst.element_accumulator, DataType.s32]
    data_type_mixed = [math_inst.element_a, math_inst.element_b, math_inst.element_a, DataType.f32]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombination)

    operations = []

    operations += CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

    operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

    operations += CreateConv2dFixedChannelsOperator(manifest, conv_layout, tile_descriptions,
      data_type_mixed, alignment_constraints_small_channels, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

    operations += CreateConv2dFewChannelsOperator(manifest, conv_layout, tile_descriptions,
      data_type_mixed, alignment_constraints_small_channels, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

    for op in operations:
      if op.tile_description.threadblock_shape[1] >= 128:
        if op.tile_description.threadblock_shape[0] == 32:
          op.C.alignment = 8
        else:
          op.C.alignment = 16
      else:
        op.C.alignment = 8

#

#
def GenerateSM80_SparseTensorOp_16864_TN(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 1):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
  ]

  math_inst =                                         \
    MathInstruction(                                  \
      [16, 8, 64],                                    \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [16,]

  tile_descriptions = [
    TileDescription([128,  64, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([256, 128, 128],  3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 256, 128],  3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 128, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([256,  64, 128],  3, [4, 1, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 128, 128],  6, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64,  64, 128],  4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 128, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128,  64, 256],  4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 128, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64,  64, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.s8, DataType.s8, DataType.s32, DataType.s32]
  data_type_mixed = [DataType.s8, DataType.s8, DataType.s8, DataType.f32]

  CreateSparseGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination)

  operations = []

  operations += CreateSparseGemmOperator(manifest, layouts, tile_descriptions, \
    data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

  for op in operations:
    if op.tile_description.threadblock_shape[1] >= 128:
      op.C.alignment = 16
    else:
      op.C.alignment = 8
#

#
def GenerateSM80_TensorOp_16832_Interleaved(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajorInterleaved32, LayoutType.RowMajorInterleaved32, LayoutType.ColumnMajorInterleaved32),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.u8, DataType.u8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [16,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 64],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 64],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 64],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 64], 10, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type_mixed = [math_inst.element_a, math_inst.element_b, math_inst.element_a, DataType.f32]

    operations = CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

    conv_layout = (LayoutType.TensorNC32HW32, LayoutType.TensorC32RSK32, LayoutType.TensorNC32HW32)

    operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

    for op in operations:
      op.C.alignment = 8
#

#
def GenerateSM80_TensorOp_16864_TN(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 64],                                    \
      DataType.s4, DataType.s4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [16, 8, 64],                                    \
      DataType.u4, DataType.u4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 80
  max_cc = 1024
  alignment_constraints = [32,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 128],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 128],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 128],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 128], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 256],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 256],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 256],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 256],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 256],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 256],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [math_inst.element_a, math_inst.element_b, math_inst.element_accumulator, DataType.s32]
    data_type_mixed = [math_inst.element_a, math_inst.element_b, math_inst.element_a, DataType.f32]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination)

    operations = []

    operations += CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombination)

    operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

    for op in operations:
      if op.tile_description.threadblock_shape[1] >= 128:
        op.C.alignment = 16
      elif op.tile_description.threadblock_shape[1] == 64:
        op.C.alignment = 8
      else:
        op.C.alignment = 8
#

#
def GenerateSM80_SparseTensorOp_168128_TN(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 1):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
  ]

  math_inst =                                         \
    MathInstruction(                                  \
      [16, 8, 128],                                    \
      DataType.s4, DataType.s4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate)

  min_cc = 80
  max_cc = 1024
  alignment_constraints = [32,]

  tile_descriptions = [
    TileDescription([ 64,  64, 256],  4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([256,  64, 256],  3, [4, 1, 1], math_inst, min_cc, max_cc),
    TileDescription([256, 128, 256],  3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 256, 256],  3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 128, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 256, 256],  4, [1, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([128,  64, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 128, 256],  6, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 128, 512],  3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([128,  64, 512],  4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 128, 512],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64,  64, 512],  3, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.s4, DataType.s4, DataType.s32, DataType.s32]
  data_type_mixed = [DataType.s4, DataType.s4, DataType.s4, DataType.f32]

  CreateSparseGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination)

  operations = []

  operations += CreateSparseGemmOperator(manifest, layouts, tile_descriptions, \
    data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

  for op in operations:
    if op.tile_description.threadblock_shape[1] > 128:
      op.C.alignment = 16
    else:
      op.C.alignment = 8
#

#
def GenerateSM80_TensorOp_16864_Interleaved(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
      (LayoutType.ColumnMajorInterleaved64, LayoutType.RowMajorInterleaved64, LayoutType.ColumnMajorInterleaved64),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 64],                                    \
      DataType.s4, DataType.s4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [16, 8, 64],                                    \
      DataType.u4, DataType.u4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 80
  max_cc = 1024
  alignment_constraints = [32,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 128],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 128],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128],  6, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type_mixed = [math_inst.element_a, math_inst.element_b, math_inst.element_a, DataType.f32]

    operations = []

    operations += CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

    conv_layout = (LayoutType.TensorNC64HW64, LayoutType.TensorC64RSK64, LayoutType.TensorNC64HW64)

    operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

    for op in operations:
      op.C.alignment = 16
#

#
def GenerateSM80_TensorOp_168256(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 256],                                   \
      DataType.b1, DataType.b1, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.xor_popc),
    MathInstruction(                                  \
      [16, 8, 256],                                   \
      DataType.b1, DataType.b1, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.and_popc),
  ]

  min_cc = 80
  max_cc = {
    MathOperation.xor_popc: 89,
    MathOperation.and_popc: 90
  }

  alignment_constraints = [128,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128,  512],  3, [4, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128, 256,  512],  3, [2, 4, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([256,  64,  512],  4, [4, 1, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64, 256,  512],  4, [1, 4, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128, 128,  512],  5, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128,  64,  512],  6, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64, 128,  512],  6, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64,  64,  512], 10, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([256, 128, 1024],  3, [4, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128, 256, 1024],  3, [2, 4, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([256,  64, 1024],  4, [4, 1, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64, 256, 1024],  4, [1, 4, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128, 128, 1024],  4, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128,  64, 1024],  3, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64, 128, 1024],  3, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64,  64, 1024],  5, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
    ]

    data_type = [DataType.b1, DataType.b1, DataType.s32, DataType.s32]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

#

#
def GenerateSM80_TensorOp_1688(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                      \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,     \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [4, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 16],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 16],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 16],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 16],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 16], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([64,  128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    data_type_mixed = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_a,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type_mixed, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)

    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, alignment_constraints)
#

#
def GenerateSM80_TensorOp_1688_fast_math(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                      \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,     \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add),
    MathInstruction(                                      \
      [16, 8, 8],                                         \
      DataType.f16, DataType.f16, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_fast_f16),
    MathInstruction(                                      \
      [16, 8, 8],                                         \
      DataType.bf16, DataType.bf16, DataType.f32,       \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_fast_bf16),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [4, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 16],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 16],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 16],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 16],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 16], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [DataType.f32, DataType.f32, DataType.f32, DataType.f32]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#

#
def GenerateSM80_TensorOp_1688_fast_fp32_math(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                      \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,       \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_fast_f32),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [4, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 128, 16],  4, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 16],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 16],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 16],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 16],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 16],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [DataType.f32, DataType.f32, DataType.f32, DataType.f32]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#

def GenerateSM80_TensorOp_1688_fast_fp32_math_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst = MathInstruction(                            \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_fast_f32)

  min_cc = 80
  max_cc = 1024

  tile_descriptions = [
    TileDescription([128, 64, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [
    DataType.cf32, DataType.cf32, DataType.cf32, DataType.cf32
  ]

  alignment_constraints = [1,]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  CreateGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)


#
def GenerateSM80_SparseTensorOp_16816_fast_math(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 1):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.RowMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.RowMajor),
  ]

  math_instructions = [
    MathInstruction(                                      \
      [16, 8, 16],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,     \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [4]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 64],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [DataType.f32, DataType.f32, DataType.f32, DataType.f32]

    CreateSparseGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)
#

#
def GenerateSM80_TensorOp_1688_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst = MathInstruction(                  \
    [16, 8, 8],                                 \
    DataType.tf32, DataType.tf32, DataType.f32,   \
    OpcodeClass.TensorOp,                       \
    MathOperation.multiply_add_complex)

  min_cc = 80
  max_cc = 1024

  tile_descriptions = [
    TileDescription([128, 128, 16], 4, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 4, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 4, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 1, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [
    DataType.cf32, DataType.cf32, DataType.cf32, DataType.cf32
  ]

  alignment_constraints = [1,]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  CreateGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#

#
def GenerateSM80_TensorOp_1688_rank_k(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_instructions = [
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,         \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add),
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_fast_f32),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1, 2, 4]  # Alignment only applies to A in SYRK

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 16],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 16],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      #TileDescription([256,  64, 16],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 256, 16],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([128,  64, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 128, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64,  64, 16], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      #TileDescription([256,  64, 32],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([128,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64,  64, 32],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [DataType.f32, DataType.f32, DataType.f32]

    CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
      data_type, alignment_constraints, BlasMode.symmetric)
#

#
def GenerateSM80_TensorOp_1688_rank_k_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_instructions = [
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,         \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex),
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_fast_f32),
  ]

  min_cc = 80
  max_cc = 1024

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 64, 16], 4, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 128, 16], 4, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([64, 32, 16], 4, [2, 1, 1], math_inst, min_cc, max_cc),
      #TileDescription([32, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      DataType.cf32, DataType.cf32, DataType.cf32
    ]

    alignment_constraints = [1,]

    # SYRK
    CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
      data_type, alignment_constraints, BlasMode.symmetric)

    # HERK
    CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
      data_type, alignment_constraints, BlasMode.hermitian)
#

#
def GenerateSM80_TensorOp_1688_trmm(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_instructions = [
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,         \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add),
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_fast_f32),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1, 2, 4]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 16],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 16],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 16],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 16],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 128, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 16], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      #TileDescription([256,  64, 32],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([128,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64,  64, 32],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [DataType.f32, DataType.f32, DataType.f32, DataType.f32]

    CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
      data_type, alignment_constraints)
#

#
def GenerateSM80_TensorOp_1688_trmm_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_instructions = [
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,         \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex),
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_fast_f32),
  ]

  min_cc = 80
  max_cc = 1024

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 64, 16], 4, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 128, 16], 4, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 32, 16], 4, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([32, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      DataType.cf32, DataType.cf32, DataType.cf32, DataType.cf32
    ]

    alignment_constraints = [1,]

    complex_transforms = [
      ComplexTransform.none, ComplexTransform.conj,
    ]

    CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
      data_type, alignment_constraints, complex_transforms)
#

#
def GenerateSM80_TensorOp_1688_symm(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  # A and B have same layouts
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_instructions = [
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,         \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add),
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_fast_f32),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [
    1, 2, 4
  ]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 16],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 16],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      #TileDescription([256,  64, 16],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 256, 16],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([128,  64, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 128, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64,  64, 16], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      #TileDescription([256,  64, 32],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([128,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64,  64, 32],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [DataType.f32, DataType.f32, DataType.f32, DataType.f32]

    CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
      data_type, alignment_constraints, BlasMode.symmetric)
#

#
def GenerateSM80_TensorOp_1688_symm_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_instructions = [
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,         \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex),
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_fast_f32),
  ]

  min_cc = 80
  max_cc = 1024

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 64, 16], 4, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 128, 16], 4, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([64, 32, 16], 4, [2, 1, 1], math_inst, min_cc, max_cc),
      #TileDescription([32, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      DataType.cf32, DataType.cf32, DataType.cf32, DataType.cf32
    ]

    alignment_constraints = [1,]

    # SYMM
    CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
      data_type, alignment_constraints, BlasMode.symmetric)

    # HEMM
    CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
      data_type, alignment_constraints, BlasMode.hermitian)
#

#
def GenerateSM80_TensorOp_884(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([256, 64, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 256, 16], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([256, 32, 16], 3, [4, 1, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 256, 16], 3, [1, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 16], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 16], 5, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64, DataType.f64]

  CreateGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints)
#

#
def GenerateSM80_TensorOp_884_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64,  8 ], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  128, 8 ], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  64,  8 ], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  32,  8 ], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  64,  8 ], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  32,  8 ], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16,  32,  8 ], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  16,  8 ], 4, [2, 1, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64,  16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  128, 16], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  64,  16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  32,  16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  64,  16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  32,  16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16,  32,  16], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  16,  16], 3, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  CreateGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)

#
def GenerateSM80_TensorOp_884_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  CreateGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#

#
def GenerateSM80_TensorOp_884_rank_k(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 16], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 16], 5, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64]

  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)
#

#
def GenerateSM80_TensorOp_884_rank_k_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 8], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 8], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64]

  # SYRK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HERK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)

#

#
def GenerateSM80_TensorOp_884_rank_k_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [ComplexTransform.none,]

  # SYRK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HERK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)
#

#
def GenerateSM80_TensorOp_884_trmm(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64, DataType.f64]

  CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
    data_type, alignment_constraints)
#

#
def GenerateSM80_TensorOp_884_trmm_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 8], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 8], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    ComplexTransform.none, ComplexTransform.conj,
  ]

  CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#


#
def GenerateSM80_TensorOp_884_trmm_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    ComplexTransform.none, ComplexTransform.conj,
  ]

  CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#

#
def GenerateSM80_TensorOp_884_symm(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 16], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 16], 5, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64, DataType.f64]

  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)
#

#
def GenerateSM80_TensorOp_884_symm_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 8], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 8], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  # SYMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HEMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)
#

#
def GenerateSM80_TensorOp_884_symm_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [ComplexTransform.none,]

  # SYMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HEMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)
#

###################################################################################################

#
def GenerateSM80_Simt_f32(manifest, cuda_version):
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f32, DataType.f32, DataType.f32,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 8], 5, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 8], 5, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 8], 5, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 8], 4, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 8], 4, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 8], 4, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 8], 5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 8], 5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 8], 5, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 8], 5, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 8], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#


#
def GenerateSM80_Simt_f64(manifest, cuda_version):
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f64, DataType.f64, DataType.f64,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 128, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 8], 5, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 8], 5, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 8], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)
#


##################################################################################################
#
def GenerateSM80_Simt_complex(manifest, cuda_version):
  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f32, DataType.f32, DataType.f32,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add_complex),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  data_type = [
    DataType.cf32,
    DataType.cf32,
    DataType.cf32,
    DataType.cf32
  ]

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  for math_inst in math_instructions:

    tile_descriptions = [
      TileDescription([128, 128, 8], 5, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 8], 4, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([32, 32, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, data_type, alignment_constraints, complex_transforms)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#

###################################################################################################

#
def GenerateSM80(manifest, cuda_version):
  GenerateSM80_TensorOp_16816(manifest, cuda_version)
  GenerateSM80_SparseTensorOp_16832(manifest, cuda_version)
  GenerateSM80_PlanarComplexTensorOp_16816(manifest, cuda_version)
  GenerateSM80_TensorOp_1688(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_fast_math(manifest, cuda_version)
  GenerateSM80_SparseTensorOp_16816_fast_math(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_complex(manifest, cuda_version)
  # 3xTF32
  GenerateSM80_TensorOp_1688_fast_fp32_math(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_fast_fp32_math_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_rank_k(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_rank_k_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_trmm(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_trmm_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_symm(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_symm_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_884(manifest, cuda_version)
  GenerateSM80_TensorOp_884_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_884_complex_gaussian(manifest, cuda_version)
  GenerateSM80_TensorOp_884_rank_k(manifest, cuda_version)
  GenerateSM80_TensorOp_884_rank_k_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_884_rank_k_complex_gaussian(manifest, cuda_version)
  GenerateSM80_TensorOp_884_trmm(manifest, cuda_version)
  GenerateSM80_TensorOp_884_trmm_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_884_trmm_complex_gaussian(manifest, cuda_version)
  GenerateSM80_TensorOp_884_symm(manifest, cuda_version)
  GenerateSM80_TensorOp_884_symm_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_884_symm_complex_gaussian(manifest, cuda_version)
  GenerateSM80_TensorOp_16816_mixed_input_upcast_a(manifest, cuda_version)
  GenerateSM80_TensorOp_16816_mixed_input_upcast_b(manifest, cuda_version)
  GenerateSM80_TensorOp_16832_TN(manifest, cuda_version)
  GenerateSM80_SparseTensorOp_16864_TN(manifest, cuda_version)
  GenerateSM80_TensorOp_16832_Interleaved(manifest, cuda_version)
  GenerateSM80_TensorOp_16864_TN(manifest, cuda_version)
  GenerateSM80_SparseTensorOp_168128_TN(manifest, cuda_version)
  GenerateSM80_TensorOp_16864_Interleaved(manifest, cuda_version)
  GenerateSM80_TensorOp_168256(manifest, cuda_version)
  GenerateSM80_Simt_f32(manifest, cuda_version)
  GenerateSM80_Simt_f64(manifest, cuda_version)
  GenerateSM80_Simt_complex(manifest, cuda_version)

###################################################################################################

def GenerateSM89_TensorOp_16832_fp8(manifest, cuda_version):
  if (
    not CudaToolkitVersionSatisfies(cuda_version, 12, 4)
  ):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor)
  ]

  math_instructions = [
    MathInstruction(
      [16, 8, 32],
      DataType.e4m3, DataType.e4m3, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [16, 8, 32],
      DataType.e4m3, DataType.e5m2, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [16, 8, 32],
      DataType.e5m2, DataType.e4m3, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [16, 8, 32],
      DataType.e5m2, DataType.e5m2, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [16, 8, 32],
      DataType.e4m3, DataType.e4m3, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add_fast_accum),
    MathInstruction(
      [16, 8, 32],
      DataType.e4m3, DataType.e5m2, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add_fast_accum),
    MathInstruction(
      [16, 8, 32],
      DataType.e5m2, DataType.e4m3, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add_fast_accum),
    MathInstruction(
      [16, 8, 32],
      DataType.e5m2, DataType.e5m2, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add_fast_accum),
  ]

  min_cc = 89
  max_cc = 89

  alignment_constraints = [16,]
  alignment_constraints_small_channels = [16, 8, 4]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128,  64],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256,  64],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128,  64],  6, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256,  64],  6, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64,  64],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256,  64],  3, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64,  64],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256,  64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  32,  64],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 256,  64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128,  64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128,  64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128,  64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128,  64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64,  64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128,  64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32,  64],  6, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128,  64],  6, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64,  64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64,  64], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 128],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 128],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  32, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 128],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 128],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 128],  6, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_types = [
      [
        math_inst.element_a,
        math_inst.element_b,
        DataType.f32,
        math_inst.element_accumulator
      ],
    ]

    operations = []
    for data_type in data_types:
      operations += CreateGemmOperator(manifest, layouts, tile_descriptions, data_type,
        alignment_constraints, None, EpilogueFunctor.LinearCombination)

      conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
      operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
        data_type, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombination)

      operations += CreateConv2dFixedChannelsOperator(manifest, conv_layout, tile_descriptions,
        data_type, alignment_constraints_small_channels, [ConvKind.Fprop], EpilogueFunctor.LinearCombination)

    for op in operations:
      if op.tile_description.threadblock_shape[1] >= 128:
        if op.tile_description.threadblock_shape[0] == 32:
          op.C.alignment = 8
        else:
          op.C.alignment = 16
      else:
        op.C.alignment = 8

#
def GenerateSM89_SparseTensorOp_16864_fp8(manifest, cuda_version):

  if (
    not CudaToolkitVersionSatisfies(cuda_version, 12, 4)
  ):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor)
  ]

  math_instructions = [
    MathInstruction(
      [16, 8, 64],
      DataType.e4m3, DataType.e4m3, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [16, 8, 64],
      DataType.e4m3, DataType.e5m2, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [16, 8, 64],
      DataType.e5m2, DataType.e4m3, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [16, 8, 64],
      DataType.e5m2, DataType.e5m2, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [16, 8, 64],
      DataType.e4m3, DataType.e4m3, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add_fast_accum),
    MathInstruction(
      [16, 8, 64],
      DataType.e4m3, DataType.e5m2, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add_fast_accum),
    MathInstruction(
      [16, 8, 64],
      DataType.e5m2, DataType.e4m3, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add_fast_accum),
    MathInstruction(
      [16, 8, 64],
      DataType.e5m2, DataType.e5m2, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add_fast_accum),
  ]

  min_cc = 89
  max_cc = 89

  alignment_constraints = [16,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128,  64, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 128],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 128],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 128],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 256],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_types = [
      [
        math_inst.element_a,
        math_inst.element_b,
        DataType.f32,
        math_inst.element_accumulator
      ],
    ]

    operations = []
    for data_type in data_types:
      operations += CreateSparseGemmOperator(manifest, layouts, tile_descriptions, data_type,
        alignment_constraints, None, EpilogueFunctor.LinearCombination)

    for op in operations:
      if op.tile_description.threadblock_shape[1] >= 128:
        op.C.alignment = 16
      else:
        op.C.alignment = 8

###################################################################################################

#
def GenerateSM89(manifest, cuda_version):
  GenerateSM89_TensorOp_16832_fp8(manifest, cuda_version)
  GenerateSM89_SparseTensorOp_16864_fp8(manifest, cuda_version)

###################################################################################################

#
def GenerateSM90_TensorOp_16b_WGMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  # layouts for ABC and their alignments.
  layouts = [
    [[LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 8], [LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    8], [LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 1]],
  ]

  math_instructions = [
    MathInstruction(
      [64, 128, 16],
      DataType.f16, DataType.f16, DataType.f16,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 128, 16],
      DataType.f16, DataType.f16, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 128, 16],
      DataType.bf16, DataType.bf16, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
  ]

  min_cc = 90
  max_cc = 90

  for math_inst in math_instructions:
    tile_descriptions_small = [
      # Not compatible with TmaWarpSpecializedCooperative
      TileDescription([math_inst.instruction_shape[0], math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
       0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
      TileDescription([math_inst.instruction_shape[0], math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
       0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
    ]
    tile_descriptions_medium = [
      TileDescription([math_inst.instruction_shape[0]*2, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
      TileDescription([math_inst.instruction_shape[0]*2, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
    ]
    tile_descriptions_large = [
      TileDescription([math_inst.instruction_shape[0]*4, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
      TileDescription([math_inst.instruction_shape[0]*4, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
      TileDescription([math_inst.instruction_shape[0]*2, math_inst.instruction_shape[1]*2, math_inst.instruction_shape[2]*4],
        0, [4, 2, 1], math_inst, min_cc, max_cc, [2,1,1]),
      TileDescription([math_inst.instruction_shape[0]*2, math_inst.instruction_shape[1]*2, math_inst.instruction_shape[2]*4],
        0, [4, 2, 1], math_inst, min_cc, max_cc, [1,2,1]),
    ]
    tile_descriptions = tile_descriptions_medium + tile_descriptions_large

    data_type = {
      "a_type"   : math_inst.element_a,
      "b_type"   : math_inst.element_b,
      "c_type"   : math_inst.element_accumulator,
      "d_type"   : math_inst.element_accumulator,
      "acc_type" : math_inst.element_accumulator,
      "epi_type" : math_inst.element_accumulator
    }

    # Set alignment c based on Destination format.
    for layout in layouts:
      if data_type["c_type"] in [DataType.s32, DataType.f32]:
        layout[2][1] = 4
      elif data_type["c_type"] in [DataType.f16, DataType.bf16]:
        layout[2][1] = 8

    if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
      schedules = [
        [KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto],
        [KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.NoSmemWarpSpecialized],
        [KernelScheduleType.TmaWarpSpecializedPingpong, EpilogueScheduleType.NoSmemWarpSpecialized],
        [KernelScheduleType.TmaWarpSpecialized, EpilogueScheduleType.NoSmemWarpSpecialized]
      ]
      stream_k_schedules = [[KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.NoSmemWarpSpecialized]]
    else:
      schedules = [
        [KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto],
        [KernelScheduleType.TmaWarpSpecialized, EpilogueScheduleType.NoSmemWarpSpecialized]
        # TmaWarpSpecializedCooperative and TmaWarpSpecializedPingpong require CUDA version >= 12.1 for optimal performance.
      ]
      stream_k_schedules = []

    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, schedules)

    if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
      # Add stream-K variants
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, stream_k_schedules, tile_schedulers=[TileSchedulerType.StreamK])

    # persistent kernels with TMA epilogues
    if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
      # not enough smem for 256x128 f32 out with C allocation
      if data_type["d_type"] == DataType.f32:
        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions_medium, data_type,
          [[KernelScheduleType.TmaWarpSpecializedPingpong,    EpilogueScheduleType.TmaWarpSpecialized],
           [KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]])

        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions_medium, data_type,
          [[KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]],
          tile_schedulers=[TileSchedulerType.StreamK])
      else:
        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
          [[KernelScheduleType.TmaWarpSpecializedPingpong,    EpilogueScheduleType.TmaWarpSpecialized],
           [KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]])

        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
          [[KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]],
          tile_schedulers=[TileSchedulerType.StreamK])

      # Emit instance without C allocation + load
      data_type["c_type"] = DataType.void
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
        [[KernelScheduleType.TmaWarpSpecializedPingpong,    EpilogueScheduleType.TmaWarpSpecialized],
         [KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]])

      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
        [[KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]],
        tile_schedulers=[TileSchedulerType.StreamK])

    # for mixed precision kernels, also generate kernels that write output matrix in the A/B format
    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:
      data_type_mixed = {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : math_inst.element_a,
        "d_type"   : math_inst.element_a,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      }

      # Set alignment c based on Destination format.
      for layout in layouts:
        if data_type_mixed["c_type"] in [DataType.s32, DataType.f32]:
          layout[2][1] = 4
        elif data_type_mixed["c_type"] in [DataType.f16, DataType.bf16]:
          layout[2][1] = 8

      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed, schedules)
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed, stream_k_schedules, tile_schedulers=[TileSchedulerType.StreamK])

      # persistent kernels with TMA epilogues
      if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed,
          [[KernelScheduleType.TmaWarpSpecializedPingpong,    EpilogueScheduleType.TmaWarpSpecialized],
           [KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]])

        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed,
          [[KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]],
          tile_schedulers=[TileSchedulerType.StreamK])

        # Emit instance without C allocation+load
        data_type_mixed["c_type"] = DataType.void
        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed,
          [[KernelScheduleType.TmaWarpSpecializedPingpong,    EpilogueScheduleType.TmaWarpSpecialized],
           [KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]])

        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed,
          [[KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]],
          tile_schedulers=[TileSchedulerType.StreamK])

#
def GenerateSM90_TensorOp_16b_WGMMA_alignx_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  # layouts for ABC and their alignments.
  layouts = [
    [[LayoutType.RowMajor,    4], [LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    4], [LayoutType.RowMajor,    4], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 4], [LayoutType.RowMajor,    4], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    2], [LayoutType.ColumnMajor, 2], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    2], [LayoutType.RowMajor,    2], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 2], [LayoutType.ColumnMajor, 2], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 2], [LayoutType.RowMajor,    2], [LayoutType.ColumnMajor, 1]],
  ]

  math_instructions = [
    MathInstruction(
      [64, 128, 16],
      DataType.f16, DataType.f16, DataType.f16,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 128, 16],
      DataType.f16, DataType.f16, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 128, 16],
      DataType.bf16, DataType.bf16, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
  ]

  min_cc = 90
  max_cc = 90

  for math_inst in math_instructions:
    tile_descriptions_small = [
      # TileDescription([math_inst.instruction_shape[0], math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
      #   0, [4, 1, 1], math_inst, min_cc, max_cc, [1,1,1]),
    ]
    tile_descriptions_medium = [
      TileDescription([math_inst.instruction_shape[0]*2, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [1,1,1]),
      # TileDescription([math_inst.instruction_shape[0], math_inst.instruction_shape[1]*2, math_inst.instruction_shape[2]*4],
      #   0, [4, 1, 1], math_inst, min_cc, max_cc, [1,1,1]),
    ]
    tile_descriptions = tile_descriptions_small + tile_descriptions_medium

    data_type = {
      "a_type"   : math_inst.element_a,
      "b_type"   : math_inst.element_b,
      "c_type"   : math_inst.element_accumulator,
      "d_type"   : math_inst.element_accumulator,
      "acc_type" : math_inst.element_accumulator,
      "epi_type" : math_inst.element_accumulator
    }

    # Set alignment c based on Destination format.
    for layout in layouts:
      if data_type["c_type"] in [DataType.s32, DataType.f32]:
        layout[2][1] = 4
      elif data_type["c_type"] in [DataType.f16, DataType.bf16]:
        layout[2][1] = 8

    schedules = [
      # [KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto],
      [KernelScheduleType.CpAsyncWarpSpecialized, EpilogueScheduleType.NoSmemWarpSpecialized]
    ]
    stream_k_schedules = []

    if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
      schedules += [
        [KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.NoSmemWarpSpecialized],
        # [KernelScheduleType.CpAsyncWarpSpecializedPingpong, EpilogueScheduleType.NoSmemWarpSpecialized]
      ]
      stream_k_schedules += [[KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.NoSmemWarpSpecialized]]

    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, schedules)

    if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
      # Add stream-K variants
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, stream_k_schedules, tile_schedulers=[TileSchedulerType.StreamK])

    # persistent kernels with TMA epilogues
    # if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
    #   CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
    #     [[KernelScheduleType.CpAsyncWarpSpecializedPingpong,    EpilogueScheduleType.TmaWarpSpecialized],
    #      [KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]])

    #   CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
    #     [[KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]],
    #     tile_schedulers=[TileSchedulerType.StreamK])

    #   # Emit instance without C allocation + load
    #   data_type["c_type"] = DataType.void
    #   CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
    #     [[KernelScheduleType.CpAsyncWarpSpecializedPingpong,    EpilogueScheduleType.TmaWarpSpecialized],
    #      [KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]])

    #   CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
    #     [[KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]],
    #     tile_schedulers=[TileSchedulerType.StreamK])

    # for mixed precision kernels, also generate kernels that write output matrix in the A/B format
    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:
      data_type_mixed = {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : math_inst.element_a,
        "d_type"   : math_inst.element_a,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      }

      # Set alignment c based on Destination format.
      for layout in layouts:
        if data_type_mixed["c_type"] in [DataType.s32, DataType.f32]:
          layout[2][1] = 4
        elif data_type_mixed["c_type"] in [DataType.f16, DataType.bf16]:
          layout[2][1] = 8

      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed, schedules)
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed, stream_k_schedules, tile_schedulers=[TileSchedulerType.StreamK])

      # persistent kernels with TMA epilogues
      # if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
      #   CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed,
      #     [[KernelScheduleType.CpAsyncWarpSpecializedPingpong,    EpilogueScheduleType.TmaWarpSpecialized],
      #      [KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]])

      #   CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed,
      #     [[KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]],
      #     tile_schedulers=[TileSchedulerType.StreamK])

      #   # Emit instance without C allocation+load
      #   data_type_mixed["c_type"] = DataType.void
      #   CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed,
      #     [[KernelScheduleType.CpAsyncWarpSpecializedPingpong,    EpilogueScheduleType.TmaWarpSpecialized],
      #      [KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]])

      #   CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed,
      #     [[KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]],
      #     tile_schedulers=[TileSchedulerType.StreamK])

#
def GenerateSM90_TensorOp_tf32_WGMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  # layouts for ABC and their alignments
  layouts_tf32 = [
    [[LayoutType.RowMajor,    4], [LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 4]],
    [[LayoutType.RowMajor,    4], [LayoutType.RowMajor,    4], [LayoutType.ColumnMajor, 4]],
    [[LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 4]],
    [[LayoutType.ColumnMajor, 4], [LayoutType.RowMajor,    4], [LayoutType.ColumnMajor, 4]],
  ]

  math_inst = MathInstruction(
      [64, 128, 8],
      DataType.tf32, DataType.tf32, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add)

  math_inst_largeN = MathInstruction(
      [64, 256, 8],
      DataType.tf32, DataType.tf32, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add)

  min_cc = 90
  max_cc = 90

  tile_descriptions_large = [
    TileDescription([math_inst.instruction_shape[0]*4, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
      0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
    TileDescription([math_inst.instruction_shape[0]*4, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
      0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
    TileDescription([math_inst_largeN.instruction_shape[0]*2, math_inst_largeN.instruction_shape[1], math_inst_largeN.instruction_shape[2]*4],
      0, [4, 1, 1], math_inst_largeN, min_cc, max_cc, [2,1,1]),
    TileDescription([math_inst_largeN.instruction_shape[0]*2, math_inst_largeN.instruction_shape[1], math_inst_largeN.instruction_shape[2]*4],
      0, [4, 1, 1], math_inst_largeN, min_cc, max_cc, [1,2,1]),
  ]

  tile_descriptions_medium = [
    TileDescription([math_inst.instruction_shape[0]*2, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
      0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
    TileDescription([math_inst.instruction_shape[0]*2, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
      0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
  ]

  tile_descriptions_small = [
    TileDescription([math_inst.instruction_shape[0], math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
      0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
    TileDescription([math_inst.instruction_shape[0], math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
      0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
  ]
  tile_descriptions = tile_descriptions_medium + tile_descriptions_small

  data_types = [
    {
      "a_type"   : math_inst.element_a,
      "b_type"   : math_inst.element_b,
      "c_type"   : math_inst.element_accumulator,
      "d_type"   : math_inst.element_accumulator,
      "acc_type" : math_inst.element_accumulator,
      "epi_type" : math_inst.element_accumulator
    },
    {
      "a_type"   : DataType.f32,
      "b_type"   : DataType.f32,
      "c_type"   : math_inst.element_accumulator,
      "d_type"   : math_inst.element_accumulator,
      "acc_type" : math_inst.element_accumulator,
      "epi_type" : DataType.f32
    }
  ]

  schedules_default = [
    [KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto],
    [KernelScheduleType.TmaWarpSpecialized, EpilogueScheduleType.NoSmemWarpSpecialized],
  ]

  # TMA kernels with TT layout use EpilogueTransposed (NoSmemWarpSpecialized with swapped strides),
  # because they use NN kernels underneath and transposing its epilogue will get the correct output
  schedules_transposed_epilogue = [
    [KernelScheduleType.ScheduleAuto, EpilogueScheduleType.EpilogueTransposed],
    [KernelScheduleType.TmaWarpSpecialized, EpilogueScheduleType.EpilogueTransposed]
  ]

  # TMA kernels with TN, NN, or NT layout
  layouts_tf32_tn_nn_nt = [layouts_tf32[0], layouts_tf32[2], layouts_tf32[3]]
  # TMA kernels with TT layout
  layouts_tf32_tt = [layouts_tf32[1]]

  if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
    CreateGemmUniversal3xOperator(manifest, layouts_tf32_tn_nn_nt, tile_descriptions_small, data_types, [
      [KernelScheduleType.TmaWarpSpecializedPingpong, EpilogueScheduleType.TmaWarpSpecialized],
      [KernelScheduleType.TmaWarpSpecializedPingpong, EpilogueScheduleType.NoSmemWarpSpecialized]
    ])

    CreateGemmUniversal3xOperator(manifest, layouts_tf32_tn_nn_nt, tile_descriptions_medium, data_types, [
      [KernelScheduleType.TmaWarpSpecializedPingpong, EpilogueScheduleType.TmaWarpSpecialized],
      [KernelScheduleType.TmaWarpSpecializedPingpong, EpilogueScheduleType.NoSmemWarpSpecialized],
      [KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative],
      [KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.NoSmemWarpSpecialized]
    ])

    CreateGemmUniversal3xOperator(manifest, layouts_tf32_tn_nn_nt, tile_descriptions_large, data_types, [
      [KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.NoSmemWarpSpecialized],
    ])

    CreateGemmUniversal3xOperator(manifest, layouts_tf32_tt, tile_descriptions_small, data_types, [
      [KernelScheduleType.TmaWarpSpecializedPingpong, EpilogueScheduleType.EpilogueTransposed]
    ])

    CreateGemmUniversal3xOperator(manifest, layouts_tf32_tt, tile_descriptions_medium, data_types, [
      [KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.EpilogueTransposed],
      [KernelScheduleType.TmaWarpSpecializedPingpong, EpilogueScheduleType.EpilogueTransposed]
    ])

    CreateGemmUniversal3xOperator(manifest, layouts_tf32_tt, tile_descriptions_large, data_types, [
      [KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.EpilogueTransposed],
    ])

  else:
    CreateGemmUniversal3xOperator(manifest, layouts_tf32_tn_nn_nt, tile_descriptions, data_types, schedules_default)
    CreateGemmUniversal3xOperator(manifest, layouts_tf32_tt, tile_descriptions, data_types, schedules_transposed_epilogue)

#
def GenerateSM90_TensorOp_tf32_WGMMA_alignx_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  # layouts for ABC and their alignments.
  layouts = [
    [[LayoutType.RowMajor,    2], [LayoutType.ColumnMajor, 2], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    2], [LayoutType.RowMajor,    2], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 2], [LayoutType.ColumnMajor, 2], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 2], [LayoutType.RowMajor,    2], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    1], [LayoutType.ColumnMajor, 1], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    1], [LayoutType.RowMajor,    1], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 1], [LayoutType.ColumnMajor, 1], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 1], [LayoutType.RowMajor,    1], [LayoutType.ColumnMajor, 1]],
  ]

  math_inst = MathInstruction(
      [64, 128, 8],
      DataType.tf32, DataType.tf32, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add)

  min_cc = 90
  max_cc = 90

  tile_descriptions_medium = [
    TileDescription([math_inst.instruction_shape[0]*2, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
      0, [4, 1, 1], math_inst, min_cc, max_cc, [1,1,1])
  ]

  tile_descriptions_small = [
    # TileDescription([math_inst.instruction_shape[0], math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
    #   0, [4, 1, 1], math_inst, min_cc, max_cc, [1,1,1])
  ]

  tile_descriptions = tile_descriptions_medium + tile_descriptions_small

  data_types = [
    {
      "a_type"   : math_inst.element_a,
      "b_type"   : math_inst.element_b,
      "c_type"   : math_inst.element_accumulator,
      "d_type"   : math_inst.element_accumulator,
      "acc_type" : math_inst.element_accumulator,
      "epi_type" : math_inst.element_accumulator
    },
    {
      "a_type"   : DataType.f32,
      "b_type"   : DataType.f32,
      "c_type"   : math_inst.element_accumulator,
      "d_type"   : math_inst.element_accumulator,
      "acc_type" : math_inst.element_accumulator,
      "epi_type" : DataType.f32
    }
  ]

  is_tt_layout = lambda v: v[0][0] == LayoutType.RowMajor and v[1][0] == LayoutType.RowMajor
  # Split kernels into TN/NT, NN or TT layouts
  layouts_tn_nn_nt = filter(lambda v: not is_tt_layout(v), layouts)
  layouts_tt = filter(is_tt_layout, layouts)

  CreateGemmUniversal3xOperator(manifest, layouts_tn_nn_nt, tile_descriptions, data_types, [
    # [KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto],
    [KernelScheduleType.CpAsyncWarpSpecialized, EpilogueScheduleType.NoSmemWarpSpecialized],
  ])

  # Kernels with TT layout use EpilogueTransposed (NoSmemWarpSpecialized with swapped strides),
  # because they use NN kernels underneath and transposing its epilogue will get the correct output
  CreateGemmUniversal3xOperator(manifest, layouts_tt, tile_descriptions, data_types, [
    # [KernelScheduleType.ScheduleAuto, EpilogueScheduleType.EpilogueTransposed],
    [KernelScheduleType.CpAsyncWarpSpecialized, EpilogueScheduleType.EpilogueTransposed]
  ])

  if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
    CreateGemmUniversal3xOperator(manifest, layouts_tn_nn_nt, tile_descriptions, data_types, [
      # [KernelScheduleType.CpAsyncWarpSpecializedPingpong, EpilogueScheduleType.NoSmemWarpSpecialized],
      [KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.NoSmemWarpSpecialized]
    ])

    # Stream-K schedules
    CreateGemmUniversal3xOperator(manifest, layouts_tn_nn_nt, tile_descriptions, data_types, [
      [KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.NoSmemWarpSpecialized]
    ], tile_schedulers=[TileSchedulerType.StreamK])

#
def GenerateSM90_TensorOp_int8_WGMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  # layouts for ABC and their alignments
  layouts = [
    [[LayoutType.RowMajor, 16], [LayoutType.ColumnMajor, 16], [LayoutType.ColumnMajor, 16]],
  ]

  math_instructions = [
    MathInstruction(
      [64, 128, 32],
      DataType.s8, DataType.s8, DataType.s32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 128, 32],
      DataType.u8, DataType.u8, DataType.s32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
  ]

  min_cc = 90
  max_cc = 90

  for math_inst in math_instructions:
    # 64x128x128
    tile_descriptions_small = [
      TileDescription([math_inst.instruction_shape[0], math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
      TileDescription([math_inst.instruction_shape[0], math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
    ]
    # 128x128x128
    tile_descriptions_medium = [
      TileDescription([math_inst.instruction_shape[0]*2, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
      TileDescription([math_inst.instruction_shape[0]*2, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
    ]
    tile_descriptions = tile_descriptions_medium + tile_descriptions_small

    data_types = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : math_inst.element_accumulator,
        "d_type"   : math_inst.element_accumulator,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.s8,
        "d_type"   : math_inst.element_a,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : DataType.f32
      }
    ]

    for data_type in data_types:
      for layout in layouts:
        layout[2][1] = 128 // DataTypeSize[data_type["d_type"]]
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type)

    # persistent kernels with TMA epilogues
    if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
      # Emit instance without C allocation+load
      data_types += [
        {
          "a_type"   : math_inst.element_a,
          "b_type"   : math_inst.element_b,
          "c_type"   : DataType.void,
          "d_type"   : math_inst.element_accumulator,
          "acc_type" : math_inst.element_accumulator,
          "epi_type" : math_inst.element_accumulator
        }
      ]
      for data_type in data_types:
        # Set output alignment based on destination format first
        for layout in layouts:
          layout[2][1] = 128 // DataTypeSize[data_type["d_type"]]
        # Pingpong persistent
        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
          [[KernelScheduleType.TmaWarpSpecializedPingpong,    EpilogueScheduleType.TmaWarpSpecialized],
           [KernelScheduleType.TmaWarpSpecializedPingpong,    EpilogueScheduleType.NoSmemWarpSpecialized]])
        # Cooperative persistent
        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions_medium, data_type,
          [[KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative],
           [KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.NoSmemWarpSpecialized]],
           tile_schedulers=[TileSchedulerType.Persistent, TileSchedulerType.StreamK]
           )

#
def GenerateSM90_TensorOp_int8_WGMMA_alignx_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  # layouts for ABC and their alignments
  layouts = [
    [[LayoutType.RowMajor,  8], [LayoutType.ColumnMajor,  8], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,  4], [LayoutType.ColumnMajor,  4], [LayoutType.ColumnMajor, 1]],
  ]

  math_instructions = [
    MathInstruction(
      [64, 128, 32],
      DataType.s8, DataType.s8, DataType.s32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 128, 32],
      DataType.u8, DataType.u8, DataType.s32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
  ]

  min_cc = 90
  max_cc = 90

  for math_inst in math_instructions:
    tile_descriptions_small = [
      # TileDescription([math_inst.instruction_shape[0], math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
      #   0, [4, 1, 1], math_inst, min_cc, max_cc, [1,1,1]),
    ]
    tile_descriptions_medium = [
      TileDescription([math_inst.instruction_shape[0]*2, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [1,1,1]),
    ]
    tile_descriptions = tile_descriptions_medium + tile_descriptions_small

    data_types = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : math_inst.element_accumulator,
        "d_type"   : math_inst.element_accumulator,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.s8,
        "d_type"   : math_inst.element_a,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : DataType.f32
      }
    ]

    for data_type in data_types:
      for layout in layouts:
        layout[2][1] = 128 // DataTypeSize[data_type["d_type"]]
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, [
        # [KernelScheduleType.ScheduleAuto, EpilogueScheduleType.NoSmemWarpSpecialized],
        [KernelScheduleType.CpAsyncWarpSpecialized, EpilogueScheduleType.NoSmemWarpSpecialized]
      ])

      if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, [
          # [KernelScheduleType.CpAsyncWarpSpecializedPingpong,    EpilogueScheduleType.NoSmemWarpSpecialized],
          [KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.NoSmemWarpSpecialized]
        ])
        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
          [[KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.NoSmemWarpSpecialized]],
          tile_schedulers=[TileSchedulerType.StreamK])

#
def GenerateSM90_TensorOp_fp8_WGMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  # layouts for ABC and their alignments
  layouts = [
    [[LayoutType.RowMajor, 16], [LayoutType.ColumnMajor, 16], [LayoutType.ColumnMajor, 1]],  # TN Layout
  ]

  math_instructions = [
    # inst 64x128x32
    MathInstruction(
      [64, 128, 32],
      DataType.e4m3, DataType.e4m3, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 128, 32],
      DataType.e4m3, DataType.e5m2, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 128, 32],
      DataType.e5m2, DataType.e4m3, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 128, 32],
      DataType.e5m2, DataType.e5m2, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
  ]

  min_cc = 90
  max_cc = 90

  for math_inst in math_instructions:
    data_types = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f32,
        "d_type"   : DataType.f32,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f32,
        "d_type"   : DataType.e4m3,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f32,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.bf16,
        "d_type"   : DataType.bf16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.bf16,
        "d_type"   : DataType.e4m3,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.bf16,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.f16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e4m3,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      },
    ]

    data_types_large_tile = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.void,
        "d_type"   : DataType.e4m3,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      }
    ]

    if math_inst.instruction_shape[1] == 128:
      tile_descriptions_small = [
        # 64x128x128
        TileDescription([math_inst.instruction_shape[0], math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
          0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
        TileDescription([math_inst.instruction_shape[0], math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
          0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
      ]
      tile_descriptions_large = [
        # 256x128x128
        TileDescription([math_inst.instruction_shape[0]*4, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
          0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
        TileDescription([math_inst.instruction_shape[0]*4, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
          0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
      ]
      tile_descriptions = [
        # 128x128x128
        TileDescription([math_inst.instruction_shape[0]*2, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
          0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
        TileDescription([math_inst.instruction_shape[0]*2, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
          0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
      ]

    else:
      assert False, "math inst is not supported"

    # some schedules disabled to save on library size
    if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
      schedules = [
        #[KernelScheduleType.ScheduleAuto,                             EpilogueScheduleType.ScheduleAuto],
        [KernelScheduleType.TmaWarpSpecializedCooperative,             EpilogueScheduleType.NoSmemWarpSpecialized],
        [KernelScheduleType.TmaWarpSpecialized,                        EpilogueScheduleType.NoSmemWarpSpecialized],
        [KernelScheduleType.TmaWarpSpecializedPingpongFP8FastAccum,    EpilogueScheduleType.NoSmemWarpSpecialized],
        [KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum, EpilogueScheduleType.NoSmemWarpSpecialized],
        [KernelScheduleType.TmaWarpSpecializedFP8FastAccum,            EpilogueScheduleType.NoSmemWarpSpecialized]
      ]
      stream_k_schedules = [[KernelScheduleType.TmaWarpSpecializedCooperative,             EpilogueScheduleType.NoSmemWarpSpecialized],
                            [KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum, EpilogueScheduleType.NoSmemWarpSpecialized]]
    else:
      schedules = [
        # [KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto],
        [KernelScheduleType.TmaWarpSpecialized, EpilogueScheduleType.NoSmemWarpSpecialized]
        # TmaWarpSpecializedCooperative require CUDA version >= 12.1 for optimal performance.
      ]
      stream_k_schedules = []

    for data_type in data_types:
      # With No-SMEM epilogues
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, schedules)

      if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
        # Persistent kernels with TMA epilogues
        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
          [[KernelScheduleType.TmaWarpSpecializedCooperative,             EpilogueScheduleType.TmaWarpSpecializedCooperative],
           [KernelScheduleType.TmaWarpSpecializedPingpongFP8FastAccum,    EpilogueScheduleType.TmaWarpSpecialized],
           [KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum, EpilogueScheduleType.TmaWarpSpecializedCooperative]])

        # Small tiles
        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions_small, data_type,
           [[KernelScheduleType.TmaWarpSpecializedPingpongFP8FastAccum, EpilogueScheduleType.TmaWarpSpecialized],
            [KernelScheduleType.TmaWarpSpecializedPingpongFP8FastAccum, EpilogueScheduleType.NoSmemWarpSpecialized]])

        # Large tiles
        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions_large, data_types_large_tile,
          [[KernelScheduleType.TmaWarpSpecializedCooperative,             EpilogueScheduleType.TmaWarpSpecializedCooperative],
           [KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum, EpilogueScheduleType.TmaWarpSpecializedCooperative]])

        # Add stream-K variants (with and without TMA epilogues)
        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, stream_k_schedules, tile_schedulers=[TileSchedulerType.StreamK])
        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
          [[KernelScheduleType.TmaWarpSpecializedCooperative,             EpilogueScheduleType.TmaWarpSpecializedCooperative],
           [KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum, EpilogueScheduleType.TmaWarpSpecializedCooperative]],
          tile_schedulers=[TileSchedulerType.StreamK])

#
def GenerateSM90_TensorOp_fp8_WGMMA_alignx_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  # layouts for ABC and their alignments
  layouts = [
    [[LayoutType.RowMajor, 8], [LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 1]],  # TN Layout
    [[LayoutType.RowMajor, 4], [LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 1]],  # TN Layout
  ]

  math_instructions = [
    # inst 64x128x32
    MathInstruction(
      [64, 128, 32],
      DataType.e4m3, DataType.e4m3, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 128, 32],
      DataType.e4m3, DataType.e5m2, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 128, 32],
      DataType.e5m2, DataType.e4m3, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 128, 32],
      DataType.e5m2, DataType.e5m2, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    # inst 64x64x32
    # MathInstruction(
    #   [64, 64, 32],
    #   DataType.e4m3, DataType.e4m3, DataType.f32,
    #   OpcodeClass.TensorOp,
    #   MathOperation.multiply_add),
    # MathInstruction(
    #   [64, 64, 32],
    #   DataType.e4m3, DataType.e5m2, DataType.f32,
    #   OpcodeClass.TensorOp,
    #   MathOperation.multiply_add),
    # MathInstruction(
    #   [64, 64, 32],
    #   DataType.e5m2, DataType.e4m3, DataType.f32,
    #   OpcodeClass.TensorOp,
    #   MathOperation.multiply_add),
    # MathInstruction(
    #   [64, 64, 32],
    #   DataType.e5m2, DataType.e5m2, DataType.f32,
    #   OpcodeClass.TensorOp,
    #   MathOperation.multiply_add),
  ]

  min_cc = 90
  max_cc = 90

  for math_inst in math_instructions:
    data_types = [
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f32,
        "d_type"   : DataType.f32,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f32,
        "d_type"   : DataType.e4m3,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f32,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.bf16,
        "d_type"   : DataType.bf16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.bf16,
        "d_type"   : DataType.e4m3,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.bf16,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.f16,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e4m3,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      },
      {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : DataType.f16,
        "d_type"   : DataType.e5m2,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      },
    ]

    if math_inst.instruction_shape[1] == 128:
      tile_descriptions = [
        # 128x128x128
        TileDescription([math_inst.instruction_shape[0]*2, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
          0, [4, 1, 1], math_inst, min_cc, max_cc, [1,1,1]),
      ]

    # elif math_inst.instruction_shape[1] == 64:
    #   tile_descriptions = [
    #     # 256x64x128
    #     TileDescription([math_inst.instruction_shape[0]*4, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
    #       0, [4, 1, 1], math_inst, min_cc, max_cc, [1,1,1]),
    #   ]

    else:
      assert False, "math inst is not supported"

    if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
      schedules = [
        # [KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto],
        [KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.NoSmemWarpSpecialized],
        # [KernelScheduleType.CpAsyncWarpSpecializedPingpong, EpilogueScheduleType.NoSmemWarpSpecialized],
        [KernelScheduleType.CpAsyncWarpSpecialized, EpilogueScheduleType.NoSmemWarpSpecialized],
      ]
      stream_k_schedules = [[KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.NoSmemWarpSpecialized]]
    else:
      schedules = [
        # [KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto],
        [KernelScheduleType.CpAsyncWarpSpecialized, EpilogueScheduleType.NoSmemWarpSpecialized]
      ]
      stream_k_schedules = []


    for data_type in data_types:
      # With No-SMEM epilogues
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, schedules)

      if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
        # Persistent kernels with TMA epilogues
        # CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
        #   [[KernelScheduleType.CpAsyncWarpSpecializedCooperative,    EpilogueScheduleType.TmaWarpSpecializedCooperative]])

        # Add stream-K variants (with and without TMA epilogues)
        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, stream_k_schedules, tile_schedulers=[TileSchedulerType.StreamK])
        # CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
        #   [[KernelScheduleType.CpAsyncWarpSpecializedCooperative,    EpilogueScheduleType.TmaWarpSpecializedCooperative]],
        #   tile_schedulers=[TileSchedulerType.StreamK])

#
def GenerateSM90_TensorOp_1684(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst = MathInstruction(
      [16, 8, 4],
      DataType.f64, DataType.f64, DataType.f64,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([256, 64, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 256, 16], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([256, 32, 16], 3, [4, 1, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 256, 16], 3, [1, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 16], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 16], 5, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64, DataType.f64]

  CreateGemmOperator(manifest, layouts, tile_descriptions,
    data_type, alignment_constraints)

#

#
def GenerateSM90_TensorOp_1684_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64,  8 ], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  128, 8 ], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  64,  8 ], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  32,  8 ], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  64,  8 ], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  32,  8 ], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16,  32,  8 ], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  16,  8 ], 4, [2, 1, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64,  16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  128, 16], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  64,  16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  32,  16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  64,  16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  32,  16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16,  32,  16], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  16,  16], 3, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  CreateGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#

#
def GenerateSM90_TensorOp_1684_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  CreateGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#

#
def GenerateSM90_TensorOp_1684_rank_k(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 16], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 16], 5, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64]

  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)
#

#
def GenerateSM90_TensorOp_1684_rank_k_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 8], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 8], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64]

  # SYRK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HERK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)

#

#
def GenerateSM90_TensorOp_1684_rank_k_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [ComplexTransform.none,]

  # SYRK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HERK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)
#

#
def GenerateSM90_TensorOp_1684_trmm(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64, DataType.f64]

  CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
    data_type, alignment_constraints)
#

#
def GenerateSM90_TensorOp_1684_trmm_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 8], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 8], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    ComplexTransform.none, ComplexTransform.conj,
  ]

  CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#


#
def GenerateSM90_TensorOp_1684_trmm_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    ComplexTransform.none, ComplexTransform.conj,
  ]

  CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#

#
def GenerateSM90_TensorOp_1684_symm(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 16], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 16], 5, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64, DataType.f64]

  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)
#

#
def GenerateSM90_TensorOp_1684_symm_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 8], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 8], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  # SYMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HEMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)
#

#
def GenerateSM90_TensorOp_1684_symm_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 90
  max_cc = 90

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [ComplexTransform.none,]

  # SYMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HEMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)
#

###################################################################################################

def GenerateSM90_Conv3x(manifest, cuda_version,
                        log_indent_level: int = 0):
  """
  Generate CUTLASS 3 convolution kernel(s) for SM90.

  This is meant to be called from GenerateSM90.
  """
  log_debug_line('GenerateSM90_Conv3x', log_indent_level)
  log_indent_level = log_indent_level + 1

  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  minimum_compute_capability = 90
  maximum_compute_capability = 90

  spatial_dims = [2, 3]

  def make_dims_and_alignments_triple(dim: int):
    byte_alignment_required_by_tma = 16
    return ((dim, byte_alignment_required_by_tma), # A
            (dim, byte_alignment_required_by_tma), # B
            (dim, byte_alignment_required_by_tma)) # C
  dims_and_alignments = [make_dims_and_alignments_triple(dim) for dim in spatial_dims]

  def make_math_instruction(data_types: Tuple[DataType, DataType, DataType],
                            instruction_shape: Tuple[int, int, int]) -> MathInstruction:
    default_opcode = OpcodeClass.TensorOp
    default_math_op = MathOperation.multiply_add
    [A_data_type, B_data_type, C_data_type] = data_types
    return MathInstruction(
      instruction_shape,
      A_data_type, B_data_type, C_data_type,
      default_opcode,
      default_math_op
    )
  data_types_and_instruction_shapes = [
    ((DataType.f16, DataType.f16, DataType.f16),   (64, 64, 16)),
    ((DataType.f16, DataType.f16, DataType.f32),   (64, 64, 16)),
    ((DataType.bf16, DataType.bf16, DataType.f32), (64, 64, 16)),
  ]
  math_instructions = map(lambda x: make_math_instruction(*x),
                          data_types_and_instruction_shapes)

  cluster_shapes = [
    [2, 1, 1],
    [1, 1, 1],
  ]
  conv_kinds = [
    ConvKind.Fprop,
    ConvKind.Dgrad
  ]
  mainloop_schedule = KernelScheduleType.ImplicitTmaWarpSpecializedSm90
  stages = 0 # zero means "deduce the number of stages automatically"

  # tile_descriptions is a 2-level list.
  # Each inner list is for each cluster shape.
  for math_inst in math_instructions:
    tile_descriptions = []
    for cluster_shape in cluster_shapes:
      tile_shape = [
        math_inst.instruction_shape[0],
        math_inst.instruction_shape[1],
        math_inst.instruction_shape[2] * 4
      ]
      warp_count = [4, 1, 1]
      tile_description = TileDescription(
        tile_shape, stages, warp_count, math_inst,
        minimum_compute_capability, maximum_compute_capability,
        cluster_shape)
      tile_descriptions.append(tile_description)

    # It's typical to get the data types from the math instruction.
    data_type = {
      "a_type"   : math_inst.element_a,
      "b_type"   : math_inst.element_b,
      "c_type"   : math_inst.element_accumulator,
      "d_type"   : math_inst.element_accumulator,
      "acc_type" : math_inst.element_accumulator,
      "epi_type" : math_inst.element_accumulator
    }

    for conv_kind in conv_kinds:
      epilogue_schedule = EpilogueScheduleType.TmaWarpSpecialized
      schedule_pairs = [
        (mainloop_schedule, epilogue_schedule)
      ]
      CreateConvOperator3x(manifest,
                           dims_and_alignments = dims_and_alignments,
                           tile_descriptions = tile_descriptions,
                           data_types = data_type,
                           schedule_pairs = schedule_pairs,
                           tile_schedulers = [TileSchedulerType.Default], # -> void
                           conv_kind = conv_kind,
                           log_indent_level = log_indent_level)

def GenerateSM90(manifest, cuda_version):
  GenerateSM90_TensorOp_16b_WGMMA_gemm(manifest, cuda_version)
  GenerateSM90_TensorOp_16b_WGMMA_alignx_gemm(manifest, cuda_version)
  GenerateSM90_TensorOp_tf32_WGMMA_gemm(manifest, cuda_version)
  GenerateSM90_TensorOp_tf32_WGMMA_alignx_gemm(manifest, cuda_version)
  GenerateSM90_TensorOp_int8_WGMMA_gemm(manifest, cuda_version)
  GenerateSM90_TensorOp_int8_WGMMA_alignx_gemm(manifest, cuda_version)
  GenerateSM90_TensorOp_fp8_WGMMA_gemm(manifest, cuda_version)
  GenerateSM90_TensorOp_fp8_WGMMA_alignx_gemm(manifest, cuda_version)
  GenerateSM90_TensorOp_1684(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_complex(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_complex_gaussian(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_rank_k(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_rank_k_complex(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_rank_k_complex_gaussian(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_trmm(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_trmm_complex(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_trmm_complex_gaussian(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_symm(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_symm_complex(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_symm_complex_gaussian(manifest, cuda_version)
  GenerateSM90_Conv3x(manifest, cuda_version)

###################################################################################################

def numeric_log_level(log_level: str) -> int:
  """
  Converts the string identifier of the log level into the numeric identifier used
  in setting the log level

  :param x: string representation of log level (e.g., 'INFO', 'DEBUG')
  :type x: str

  :return: numeric representation of log level
  :rtype: int
  """
  numeric_level = getattr(logging, log_level.upper(), None)
  if not isinstance(numeric_level, int):
    raise ValueError(f'Invalid log level: {log_level}')
  return numeric_level


# This function for defining the ArgumentParser is used to make it easy for the CUTLASS Python interface
# to leverage the functionality in this file without running this script via a shell prompt.
def define_parser():
  parser = argparse.ArgumentParser(description="Generates device kernel registration code for CUTLASS Kernels")
  parser.add_argument("--operations", default="all", help="Specifies the operation to generate (gemm, all)")
  parser.add_argument("--build-dir", default=".", required=False, help="CUTLASS top-level build directory")
  parser.add_argument("--curr-build-dir", default=".", help="CUTLASS current build directory. cmake files will be emitted in this directory")
  parser.add_argument("--generator-target", default='library', help="Target of CUTLASS Library Generator.")
  parser.add_argument("--architectures", default='53;60;61;70;75;80;90', help="Target compute architectures")
  parser.add_argument("--kernels", default='', help='Comma delimited list to filter kernels by name.')
  parser.add_argument("--ignore-kernels", default='', help='Comma delimited list of kernels to exclude from build.')
  parser.add_argument("--filter-by-cc", default='True', type=str, help='If enabled, kernels whose compute capability range is not satisfied by the build target are excluded.')
  parser.add_argument("--cuda-version", default="11.0.0", help="Semantic version string of CUDA Toolkit")
  parser.add_argument('--kernel-filter-file',   type=str, default=None, required=False, help='Full path of filter file')
  parser.add_argument('--selected-kernel-list',   type=str, default=None, required=False,
                        help='Specify the output log file containing all enabled kernels in this build')
  parser.add_argument("--interface-dir", default=None, required=False, help="Interface header to kernels")
  parser.add_argument("--disable-full-archs-compilation", action="store_true", required=False, help="Disable compilation for every archs in --architectures")
  parser.add_argument("--log-level", default='info', type=numeric_log_level, required=False,
                      help='Logging level to be used by the generator script')
  _add_package_disablement_flag(parser)
  return parser


if __name__ == "__main__":
  parser = define_parser()
  args = parser.parse_args()

  # Set the logging level based on the user-provided `--log-level` command-line option
  logging.basicConfig(level=args.log_level)

  manifest = Manifest(args)

  GenerateSM50(manifest, args.cuda_version)
  GenerateSM60(manifest, args.cuda_version)
  GenerateSM61(manifest, args.cuda_version)
  GenerateSM70(manifest, args.cuda_version)
  GenerateSM75(manifest, args.cuda_version)
  GenerateSM80(manifest, args.cuda_version)
  GenerateSM89(manifest, args.cuda_version)
  GenerateSM90(manifest, args.cuda_version)
  if 'library' in args.generator_target.split(','):
    manifest.emit(GeneratorTarget.Library)

  if args.selected_kernel_list is not None:
    if len(manifest.selected_kernels) > 0:
      with open(args.selected_kernel_list, 'w') as file_writer:
        for line in manifest.selected_kernels:
          file_writer.write("%s\n" % line)

###################################################################################################
