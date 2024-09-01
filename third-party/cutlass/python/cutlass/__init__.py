#################################################################################################
#
# Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import os
import sys

import cutlass_library


def _cuda_install_path_from_nvcc() -> str:
    import subprocess
    # Attempt to detect CUDA_INSTALL_PATH based on location of NVCC
    result = subprocess.run(['/usr/bin/which', 'nvcc'], capture_output=True)
    if result.returncode != 0:
        raise Exception(f'Unable to find nvcc via `which` utility.')

    cuda_install_path = result.stdout.decode('utf-8').split('/bin/nvcc')[0]
    if not os.path.isdir(cuda_install_path):
        raise Exception(f'Environment variable "CUDA_INSTALL_PATH" is not defined, '
                        f'and default path of {cuda_install_path} does not exist.')

    return cuda_install_path


CUTLASS_PATH = os.getenv("CUTLASS_PATH", cutlass_library.source_path)

# Alias CUTLASS_PATH as source_path
source_path = CUTLASS_PATH

_CUDA_INSTALL_PATH = None
def cuda_install_path():
    """
    Helper method for on-demand fetching of the CUDA installation path. This allows
    the import of CUTLASS to proceed even if NVCC is not available, preferring to
    raise this error only when an operation that needs NVCC is being performed.
    """
    global _CUDA_INSTALL_PATH
    if _CUDA_INSTALL_PATH is None:
        _CUDA_INSTALL_PATH = os.getenv("CUDA_INSTALL_PATH", _cuda_install_path_from_nvcc())
    return _CUDA_INSTALL_PATH

CACHE_FILE = "compiled_cache.db"

from cutlass_library import (
    DataType,
    EpilogueScheduleType,
    KernelScheduleType,
    MathOperation,
    LayoutType,
    OpcodeClass,
    TileDescription,
    TileSchedulerType,
)

this = sys.modules[__name__]
this.logger = logging.getLogger(__name__)

# RMM is only supported for Python 3.9+
if (sys.version_info.major == 3 and sys.version_info.major > 8) or sys.version_info.major > 3:
    try:
        import rmm
        this.use_rmm = True
    except ImportError:
        this.use_rmm = False
else:
    this.use_rmm = False


def set_log_level(level: int):
    """
    Sets the log level

    :param log_level: severity of logging level to use. See https://docs.python.org/3/library/logging.html#logging-levels for options
    :type log_level: int
    """
    this.logger.setLevel(level)

set_log_level(logging.ERROR)

from cutlass.library_defaults import OptionRegistry
from cutlass.backend.utils.device import device_cc

this._option_registry = None
def get_option_registry():
    """
    Helper method for on-demand initialization of the options registry. This avoids building
    the registry when CUTLASS is imported.
    """
    if this._option_registry is None:
        this.logger.info("Initializing option registry")
        this._option_registry = OptionRegistry(device_cc())
    return this._option_registry

this.__version__ = '3.5.0'

from cutlass.backend import create_memory_pool
from cutlass.emit.pytorch import pytorch
from cutlass.op.gemm import Gemm
from cutlass.op.conv import Conv2d, Conv2dFprop, Conv2dDgrad, Conv2dWgrad
from cutlass.op.gemm_grouped import GroupedGemm
from cutlass.op.op import OperationBase
from cutlass.backend.evt.ir.tensor import Tensor


this.memory_pool = None
def get_memory_pool():
    """"
    Helper method for on-demand memory pool. This avoids allocating the memory pool unnecessarily
    whe CUTLASS is imported.
    """
    if this.use_rmm and this.memory_pool is None:
        this.memory_pool = create_memory_pool(init_pool_size=2 ** 30, max_pool_size=2 ** 32)
    return this.memory_pool


from cuda import cuda, cudart

this._device_id = None
def initialize_cuda_context():
    if this._device_id is not None:
        return

    if this.use_rmm:
        # This also covers initializing the CUDA context
        get_memory_pool()

    device_id = os.getenv("CUTLASS_CUDA_DEVICE_ID")
    if device_id is None:
        if not this.use_rmm:
            # Manually call cuInit() and create context by making a runtime API call
            err, = cudart.cudaFree(0)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaFree failed with error {err}")

        err, device_count = cuda.cuDeviceGetCount()
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise Exception(f"cuDeviceGetCount failed with error {err}")
        if device_count <= 0:
            raise Exception("No CUDA devices found")
        device_id = 0

    this._device_id = int(device_id)


def device_id() -> int:
    initialize_cuda_context()
    return this._device_id
