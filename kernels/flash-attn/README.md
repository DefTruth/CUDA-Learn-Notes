## ⚡️⚡️FlashAttention-2 MMA: Write FlashAttention using Tensor Cores with pure MMA PTX 

![flash-attn-mma](https://github.com/user-attachments/assets/6f66796d-44d5-4ec1-b224-af997bd152b2)

|Tensor Cores|Loop over Seqlen/HeadDim |Tile Block (Br, Bc)|MMA (m16n8k16)|
|:---:|:---:|:---:|:---:|
|✔️|✔️|✔️|✔️|
|Pack LDST (pack 128 bits)|SMEM **Swizzle**/Padding |Copy Async (cp.async.cg/ca)|Tile MMA (More Threads)
|✔️|✔️|✔️|✔️|
|Tile Warp (More Values)|Multi Stages (1/2)|Collective Store (Warp Shfl & Reg Reuse)|**Split KV/Q**|
|✔️|✔️|✔️|✔️|
|**Shared QKV/KV** SMEM|**Prefetch Q** s2r|**Prefetch K/V** g2s|**QK Fine-grained Tiling**|
|✔️|✔️|✔️|✔️|

This repository's implementation of FlashAttention is intended solely for learning CUDA programming. For optimal performance, please use the official [flash-attention](https://github.com/Dao-AILab/flash-attention). Currently, for small-scale attention `(B<=4, H <=48, SeqLen <= 8192, D <= 64)` it can run faster than offical FA2/SDPA on some Devices. However, for large-scale attention, there remains a performance gap. Performance is continuously being optimized. Stay tuned for updates ~  (MMA Acc F16/F32, softmax Acc F32 vs FA2 MMA/softmax Acc F32, 👇Benchmark)

|Algorithm| (B,H,N,D) | NVIDIA RTX 3080 Laptop | NVIDIA L20 | NVIDIA GeForce RTX 4090 |   
|:---:|:---:|:---:|:---:|:---:|  
|FlashAttention-2|(1,8,8192,64)|37 TFLOPS|100 TFLOPS|145 TFLOPS|  
|share-qkv+stage2|(1,8,8192,64)|**55 TFLOPS**|99 TFLOPS|**221 TFLOPS**|  
|FlashAttention-2|(1,48,8192,64)|37 TFLOPS|109 TFLOPS|163 TFLOPS|
|share-qkv+stage2|(1,48,8192,64)|**48 TFLOPS**|107 TFLOPS|**224 TFLOPS**|
|SDPA(EFFICIENT ATTENTION)|(1,48,8192,512)|16 TFLOPS|58 TFLOPS|85 TFLOPS|
|🤖[ffpa-attn-mma](https://github.com/DefTruth/ffpa-attn-mma)|(1,48,8192,512)|**39 TFLOPS**|**104 TFLOPS**|**200 TFLOPS**|
|Precision Errors vs FA2/SDPA| / | max: < ~1e-3 | min: ~0.0 | mean: < ~1e-5 |

For example, on NVIDIA RTX 3080 Laptop, [📚 Split Q + Fully Shared QKV SMEM](#mma-share-qkv) method can achieve **55 TFLOPS (D=64)** that almost **~1.5x** 🎉 faster than FA2. On NVIDIA L20, 🤖ffpa-attn-mma method can achieve 104 TFLOPS (D=512) that almost ~1.8x 🎉 faster than SDPA (EFFICIENT ATTENTION). However, for large-scale attention, there remains a performance gap. Stay tuned for updates ~ 

## 📖 Contents

- [📖 FlashAttetion MMA Kernels](#mma)
  - [📚 Split KV](#mma-split-kv)
  - [📚 Split Q ](#mma-split-q)
  - [📚 Shared KV SMEM](#mma-share-kv)
  - [📚 Fully Shared QKV SMEM](#mma-share-qkv)
  - [📚 QK Fine-grained Tiling](#mma-tiling-qk)
  - [📚 Fully QKV Fine-grained Tiling](#mma-tiling-qkv)
- [📖 Prerequisites](#prerequisites)
- [📖 Installation](#install)
- [📖 Performance](#perf)
- [📖 Python Testing](#test)
  
## 📖 FlashAttetion MMA Kernels
<div id="mma"></div>  

The `Split KV` and `Split Q` implementations have been carried out in [flash-attention-mma⚡️⚡️](.) for performance comparison. The `Split KV` method, which involves splitting all QKV across MMA (Warps) using a naive matmul (MMA) and Warp tiling policy, is slower compared to the `Split Q` policy, which splitting Q across MMA(Warps) and keep access KV for all MMA(Warps).

- 📚 Split Q (Faster, FlashAttention-2)
<div id="mma-split-q"></div>  

```C++
// Split Q across MMA(Warps) and keep access KV for all MMA(Warps),
// in order to reduce the comm between warps via smem and warp shuffle.
// case: MMA = m16n8k16, Br=16x4=64, Bc=8x8=64, layout: 4 warps
// |   64x64   |      warp_KV 0       |
// | warp_QP 0 | MMA 0 ... MMA 0 (x8) |
// | warp_QP 1 | MMA 1 ... MMA 1 (x8) |
// | warp_QP 2 | MMA 2 ... MMA 2 (x8) |
// | warp_QP 3 | MMA 3 ... MMA 3 (x8) |
__global__ void // Q, K, V, O -> [B, H, N, D]
flash_attn_mma_stages_split_q_kernel(half* Q, half* K, half* V, half* O, ...);
```

- 📚 Split Q + Shared KV SMEM (**1/2 SRAM** vs FA2)
<div id="mma-share-kv"></div>  

```C++
// K, V shared the same shared memory, improve block occupancy.
__global__ void // Q, K, V, O -> [B, H, N, D]
flash_attn_mma_stages_split_q_shared_kv_kernel(half* Q, half* K, half* V, half* O, ...);
```
- 📚 Split Q + Fully Shared QKV SMEM (**1/4 SRAM** vs FA2)

<div id="mma-share-qkv"></div>  

```C++
// Q, K, V fully shared the same shared memory and prefetch Q s2r, improve block occupancy
// and reduce Q SMEM IO-Access.
__global__ void // Q, K, V, O -> [B, H, N, D]
flash_attn_mma_stages_split_q_shared_qkv_kernel(half* Q, half* K, half* V, half* O, ...);
```  
- 📚 Split Q + QK Fine-grained Tiling (**O(16xd) SRAM** vs FA2 **O(4xBrxd) SRAM**, `Headdim -> 1024`)

<div id="mma-tiling-qk"></div>  

```C++
// Fine-grained tiling at the MMA level for Q and K results in a constant SRAM usage of
// 64 * kMmaAtomK for Q and K. For V, the SRAM complexity is O(kMmaAtomK * d), leading to
// an overall SRAM complexity of O(kMmaAtomK * d). Consequently, this approach allows us to
// extend D (head dimension) up to 1024. Stay tuned for updates ~
__global__ void // Q, K, V, O -> [B, H, N, D]
flash_attn_mma_stages_split_q_tiling_qk_kernel(half* Q, half* K, half* V, half* O, ...);
```

- 📚 Split Q + Fully QKV Fine-grained Tiling (**O(Brx16)~O(1) SRAM** vs FA2 **O(4xBrxd) SRAM**)

<div id="mma-tiling-qkv"></div>  

```C++
// Fine-grained tiling at the MMA level for all Q@K^T and P@V results in a constant SRAM usage of
// Br * 16 or Bc * 16 for Q, K, V, leading to an overall SRAM complexity of O(Br * 16). Consequently,
// this approach allows us to run faster than SDPA w or w/o MMA Acc F32, e.g d>=512. 
__global__ void // Q, K, V, O -> [B, H, N, D]
flash_attn_mma_stages_split_q_tiling_qkv_kernel(half* Q, half* K, half* V, half* O, ...);
```

## 📖 Prerequisites
<div id="prerequisites"></div>  

- flash-attention >= 2.6
- PyTorch >= 2.0, CUDA >= 12.0
- Recommended: PyTorch 2.5.1, CUDA 12.5

## 📖 Installation  
<div id="install"></div>    

```bash
pip install flash-attn --no-build-isolation # need offical flash-attention for comparison
```

## 📖 Performance
<div id="perf"></div>  

Currently, for small-scale attention (B<=4, H <=48, SeqLen <= 8192), the flash-attention-mma implemented in this repository matches the performance of the official FA version. However, for large-scale attention computations, there remains a performance gap. Performance optimizations are ongoing; stay tuned for updates.

## 📖 Python Testing  
<div id="test"></div>  

```bash
cd kernels/flash-attn
# Volta, Ampere, Ada, Hopper, ...
python3 -m pip install flash-attn --no-build-isolation
export TORCH_CUDA_ARCH_LIST=Ada # for Ada only
export TORCH_CUDA_ARCH_LIST=Ampere # for Ampere only 
python3 flash_attn_mma.py --D 64 # test all default settings for D=64
```

- Example: B=1, H=8, N=8192, `D=64` (NVIDIA RTX 3080 Laptop), Faster than FA2~🎉🎉
```bash
python3 flash_attn_mma.py --B 1 --H 8 --D 64 --N 8192 --iters 10 --torch # NVIDIA RTX 3080 Laptop
-------------------------------------------B=1, H=8, N=8192, D=64, Warmup: 1, Iters: 10-------------------------------------------
                  torch(unfused): ['-0.00514603 ', '0.05783081  ', '-0.00026727 '], time:20.999861ms, TFLOPS:6.67 (+0.00%)
            mma(split-kv+stage1): ['-0.00511169 ', '0.05795288  ', '-0.00029612 '], time:5.120730ms, TFLOPS:27.36 (+310.10%)
            mma(split-kv+stage2): ['-0.00511169 ', '0.05795288  ', '-0.00029612 '], time:5.004287ms, TFLOPS:28.00 (+2.33%)
             mma(split-q+stage1): ['-0.00511169 ', '0.05795288  ', '-0.00029612 '], time:3.462291ms, TFLOPS:40.47 (+44.54%)
             mma(split-q+stage2): ['-0.00511169 ', '0.05795288  ', '-0.00029612 '], time:3.658915ms, TFLOPS:38.30
   mma(split-q+share-qkv+stage1): ['-0.00511169 ', '0.05795288  ', '-0.00029612 '], time:2.551699ms, TFLOPS:54.91 (+35.69%)
   mma(split-q+share-qkv+stage2): ['-0.00511169 ', '0.05795288  ', '-0.00029612 '], time:2.532172ms, TFLOPS:55.34 (+0.77%)
    mma(split-q+share-kv+stage1): ['-0.00511169 ', '0.05795288  ', '-0.00029612 '], time:2.776575ms, TFLOPS:50.46
    mma(split-q+share-kv+stage2): ['-0.00511169 ', '0.05795288  ', '-0.00029612 '], time:2.596927ms, TFLOPS:53.96
                         (flash): ['-0.00516129 ', '0.05783081  ', '-0.00027728 '], time:3.776550ms, TFLOPS:37.10
----------------------------------------------------------------------------------------------------------------------------------
```

- Example: B=1, H=48, N=8192, `D=64` (NVIDIA RTX 3080 Laptop), Faster than FA2~🎉🎉
```bash
python3 flash_attn_mma.py --B 1 --H 48 --D 64 --N 8192 --iters 10 --torch  # NVIDIA RTX 3080 Laptop
------------------------------------------B=1, H=48, N=8192, D=64, Warmup: 1, Iters: 10-------------------------------------------
                  torch(unfused): ['-0.00043964 ', '0.03292847  ', '0.01331329  '], time:1708.712411ms, TFLOPS:0.49  (+0.00%)
            mma(split-kv+stage1): ['-0.00042009 ', '0.03286743  ', '0.01330566  '], time:32.308507ms, TFLOPS:26.02 (+5188.74%)
            mma(split-kv+stage2): ['-0.00042009 ', '0.03286743  ', '0.01330566  '], time:31.260324ms, TFLOPS:26.89 (+3.35%)
             mma(split-q+stage1): ['-0.00042009 ', '0.03286743  ', '0.01330566  '], time:23.505139ms, TFLOPS:35.77 (+32.99%)
             mma(split-q+stage2): ['-0.00042009 ', '0.03286743  ', '0.01330566  '], time:24.225831ms, TFLOPS:34.70
   mma(split-q+share-qkv+stage1): ['-0.00042009 ', '0.03286743  ', '0.01330566  '], time:17.338157ms, TFLOPS:48.49 (+35.57%)
   mma(split-q+share-qkv+stage2): ['-0.00042009 ', '0.03286743  ', '0.01330566  '], time:17.652464ms, TFLOPS:47.63
    mma(split-q+share-kv+stage1): ['-0.00042009 ', '0.03286743  ', '0.01330566  '], time:18.073559ms, TFLOPS:46.52
    mma(split-q+share-kv+stage2): ['-0.00042009 ', '0.03286743  ', '0.01330566  '], time:17.378855ms, TFLOPS:48.38
                         (flash): ['-0.00041986 ', '0.03292847  ', '0.01330566  '], time:22.468138ms, TFLOPS:37.42
----------------------------------------------------------------------------------------------------------------------------------
```
- Example: B=1, H=48, N=8192, `D=512` (NVIDIA RTX 3080 Laptop), FA2 not supported, `QK Tiling` Faster than SDPA~🎉🎉
```bash
python3 flash_attn_mma.py --B 1 --H 8 --N 8192 --iters 10 --show-all --sdpa --D 512 # NVIDIA RTX 3080 Laptop, Faster than SDPA
------------------------------------------B=1, H=8, N=8192, D=512, Warmup: 1, Iters: 10-------------------------------------------
   mma(split-q+tiling-qk+stage1): ['-0.00433731 ', '0.02165222  ', '-0.01544189 '], time:48.775554ms, TFLOPS:22.60 (+0.00%)
   mma(split-q+tiling-qk+stage2): ['-0.00433731 ', '0.02165222  ', '-0.01544189 '], time:47.503424ms, TFLOPS:23.20 (+2.68%)
                          (sdpa): ['-0.00438309 ', '0.02174377  ', '-0.01551056 '], time:66.486573ms, TFLOPS:16.58
----------------------------------------------------------------------------------------------------------------------------------
```

- Example: B=1, H=48, N=8192, `D=16384` (NVIDIA L20), FA2 not supported, `QKV Tiling` Faster than SDPA~🎉🎉
```bash
---------------------------------------------------B=1, H=48, N=16384, D=512, Warmup: 1, Iters: 10----------------------------------------------------
                     mma(split-q+tiling-qk+stage1): ['-0.00386429 ', '0.00828552  ', '0.01831055  '], time:374.5436ms, TFLOPS:70.63 (+0.00%)
                     mma(split-q+tiling-qk+stage2): ['-0.00386429 ', '0.00828552  ', '0.01831055  '], time:320.5431ms, TFLOPS:82.52 (+16.85%)
           mma(split-q+tiling-qk+swizzle-q+stage1): ['-0.00386429 ', '0.00828552  ', '0.01831055  '], time:370.0427ms, TFLOPS:71.48
           mma(split-q+tiling-qk+swizzle-q+stage2): ['-0.00386429 ', '0.00828552  ', '0.01831055  '], time:318.7205ms, TFLOPS:83.00 (+0.57%)
          mma(split-q+tiling-qk+swizzle-qk+stage1): ['-0.00386429 ', '0.00828552  ', '0.01831055  '], time:374.6879ms, TFLOPS:70.60
          mma(split-q+tiling-qk+swizzle-qk+stage2): ['-0.00386429 ', '0.00828552  ', '0.01831055  '], time:321.8044ms, TFLOPS:82.20
                    mma(split-q+tiling-qkv+stage1): ['-0.00386429 ', '0.00828552  ', '0.01831055  '], time:383.5075ms, TFLOPS:68.97
                    mma(split-q+tiling-qkv+stage2): ['-0.00386429 ', '0.00828552  ', '0.01831055  '], time:290.3107ms, TFLOPS:91.12 (+9.79%)
                                            (sdpa): ['-0.00387764 ', '0.00831604  ', '0.01831055  '], time:452.0751ms, TFLOPS:58.51
------------------------------------------------------------------------------------------------------------------------------------------------------
```
