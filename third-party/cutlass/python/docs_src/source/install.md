# Installation

## Installing a stable release

Stable releases of the CUTLASS Python interface are available via the `nvidia-cutlass` PyPI package. Any other packages with the name `cutlass` are not affiliated with NVIDIA CUTLASS.
```bash
pip install nvidia-cutlass
```

## Installing from source

Installing from source requires the latest CUDA Toolkit that matches the major.minor of CUDA Python installed.

Prior to installing the CUTLASS Python interface, one may optionally set the following environment variables:
* `CUTLASS_PATH`: the path to the cloned CUTLASS repository
* `CUDA_INSTALL_PATH`: the path to the installation of CUDA

If these environment variables are not set, the installation process will infer them to be the following:
* `CUTLASS_PATH`: either one directory level above the current directory (i.e., `$(pwd)/..`) if installed locally or in the `source` directory of the location in which `cutlass_library` was installed
* `CUDA_INSTALL_PATH`: the directory holding `/bin/nvcc` for the first version of `nvcc` on `$PATH` (i.e., `which nvcc | awk -F'/bin/nvcc' '{print $1}'`)

**NOTE:** The version of `cuda-python` installed must match the CUDA version in `CUDA_INSTALL_PATH`.

### Installing a developer-mode package
The CUTLASS Python interface can currently be installed by navigating to the root of the CUTLASS directory and performing
```bash
pip install .
```

If you would like to be able to make changes to CULASS Python interface and have them reflected when using the interface, perform:
```bash
pip install -e .
```

## Docker
We recommend using the CUTLASS Python interface via an [NGC PyTorch Docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch):

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:23.08-py3
```
