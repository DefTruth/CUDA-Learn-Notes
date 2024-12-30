import os
import torch
from torch.utils.cpp_extension import load


def get_device_name():
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    # since we will run GPU on WSL2, so add WSL2 tag.
    if "Laptop" in device_name:
        device_name += " WSL2"
    return device_name


def get_device_capability():
    return torch.cuda.get_device_capability(torch.cuda.current_device())


def get_build_sources():
    build_sources = []
    build_sources.append('naive/hgemm.cu')
    build_sources.append('naive/hgemm_async.cu')
    build_sources.append('cublas/hgemm_cublas.cu')
    build_sources.append('wmma/hgemm_wmma.cu')
    build_sources.append('wmma/hgemm_wmma_stage.cu')
    build_sources.append('mma/basic/hgemm_mma.cu')
    build_sources.append('mma/basic/hgemm_mma_stage.cu')
    build_sources.append('mma/basic/hgemm_mma_stage_tn.cu')
    build_sources.append('mma/swizzle/hgemm_mma_stage_swizzle.cu')
    build_sources.append('mma/swizzle/hgemm_mma_stage_tn_swizzle_x4.cu')
    build_sources.append('cutlass/hgemm_mma_stage_tn_cute.cu')
    build_sources.append('pybind/hgemm.cc')
    return build_sources


def get_project_dir():
    return os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def get_build_cuda_cflags(build_pkg: bool = False):
    # -Xptxas -v:
    # registers, smem, cmem, stack, gmem usage
    # registers: 寄存器，访问速度最快。Ada Lovelace架构每个SM的寄存器文件大小
    # 为256KB，这相当于65536个32位寄存器，65536/256=256。一个SM可以同时执行多
    # 个block，对一个Kernel，同时存在于一个SM中的Block和Warp数量取决于SM中可用
    # 且所需的寄存器和共享内存数量。每个Thread需要的寄存器越多，那么SM中的Warp就
    # 越少。即减少Thread所需寄存器数量，即可增加SM中的Warp数。每个Block需要的共
    # 享内存越多，那么SM中可以被同时处理的Block就会变少。即减少每个Block所需的共
    # 享内存，即可同时处理更多Block。SM内的资源没办法处理一个完整Block，Kernel
    # 将无法启动。
    # cmem: 常量内存，被缓存，访问速度快。
    # stack frame: 由于寄存器的数量有限，当需要使用的变量数量超过可用寄存器数量时，
    # 编译器会将某些变量从寄存器“溢出”到栈上，这个过程称为spill。访问栈上的数据比
    # 访问寄存器慢得多。
    # spill stores: 指的是在执行过程中，数据因为寄存器不足而被存储到了栈上。
    # spill loads: 则是指将之前溢出到栈上的数据重新加载回寄存器。
    # diag 177: variable was declared but never referenced
    extra_cuda_cflags = []
    extra_cuda_cflags.append("-O3")
    extra_cuda_cflags.append("-std=c++17")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF_OPERATORS__")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF_CONVERSIONS__")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF2_OPERATORS__")
    extra_cuda_cflags.append("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
    extra_cuda_cflags.append("--expt-relaxed-constexpr")
    extra_cuda_cflags.append("--expt-extended-lambda")
    extra_cuda_cflags.append("--use_fast_math")
    if not build_pkg:
      extra_cuda_cflags.append("-diag-suppress 177")
      extra_cuda_cflags.append("-Xptxas -v")
    else:
      extra_cuda_cflags.append("--ptxas-options=-v")
      extra_cuda_cflags.append("--ptxas-options=-O3")
    # extra cuda flags for cute hgemm
    project_dir = get_project_dir()
    extra_cuda_cflags.append('-DNO_MMA_HGEMM_BIN')
    extra_cuda_cflags.append('-DNO_WMMA_HGEMM_BIN')
    extra_cuda_cflags.append('-DNO_CUTE_HGEMM_BIN')
    extra_cuda_cflags.append('-DNO_CUBLAS_HGEMM_BIN')
    # add cutlass headers and link cublas.
    extra_cuda_cflags.append(f'-I {project_dir}/kernels/hgemm')
    extra_cuda_cflags.append(f'-I {project_dir}/kernels/hgemm/utils')
    extra_cuda_cflags.append(f'-I {project_dir}/kernels/hgemm/naive')
    extra_cuda_cflags.append(f'-I {project_dir}/kernels/hgemm/wmma')
    extra_cuda_cflags.append(f'-I {project_dir}/kernels/hgemm/mma/basic')
    extra_cuda_cflags.append(f'-I {project_dir}/kernels/hgemm/mma/swizzle')
    extra_cuda_cflags.append(f'-I {project_dir}/kernels/hgemm/cutlass')
    extra_cuda_cflags.append(f'-I {project_dir}/kernels/hgemm/cublas')
    extra_cuda_cflags.append(f'-I {project_dir}/kernels/hgemm/pybind')
    extra_cuda_cflags.append(f'-I {project_dir}/third-party/cutlass/include')
    extra_cuda_cflags.append(f'-I {project_dir}/third-party/cutlass/tools/util/include')
    extra_cuda_cflags.append('-lcublas')
    return extra_cuda_cflags


def pretty_print_line(m: str = "", sep: str = "-", width: int = 150):
    res_len = width - len(m)
    left_len = int(res_len / 2)
    right_len = res_len - left_len
    pretty_line = sep * left_len + m + sep * right_len
    print(pretty_line)


def build_from_sources(verbose: bool = False):
    torch_arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST", None)         
    # Load the CUDA kernel as a python module
    pretty_print_line(f"Loading hgemm lib on device: {get_device_name()}, "
                      f"capability: {get_device_capability()}, "
                      f"Arch ENV: {torch_arch_list_env}")
    return load(name='hgemm_lib', sources=get_build_sources(),
                extra_cuda_cflags=get_build_cuda_cflags(), 
                extra_cflags=['-std=c++17'], 
                verbose=verbose)


def try_load_hgemm_library(force_build: bool = False, verbose: bool = False):
    if not force_build:
        # check if can import toy_hgemm
        try:
            import toy_hgemm as hgemm
            pretty_print_line(f"Import toy-hgemm library done, use it!")
        except Exception:
            pretty_print_line(f"Can't import toy-hgemm, force build "
                              f"from source or run <bash tools/install.sh>")
            pretty_print_line(f"Also may need export LD_LIBRARY_PATH="
                              f"PATH-TO/torch/lib:$LD_LIBRARY_PATH")
            hgemm = build_from_sources(verbose=verbose)
    else:
        pretty_print_line("Force hgemm lib build from sources")
        hgemm = build_from_sources(verbose=verbose)

    return hgemm


@torch.no_grad
def as_col_major(x: torch.Tensor):
    # convert a row major tensor -> col major with contiguous storage
    x_trans = x.t()
    x_col_major = x_trans.reshape(x.shape)
    return x_col_major.contiguous() # must be a contiguous tensor
