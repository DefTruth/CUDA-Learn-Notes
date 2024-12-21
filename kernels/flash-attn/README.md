## ⚡️⚡️FlashAttention-2 MMA: Write FlashAttention using Tensor Cores with pure MMA PTX 

![flash-attn-mma](https://github.com/user-attachments/assets/6f66796d-44d5-4ec1-b224-af997bd152b2)

|Tensor Cores|Loop over Seqlen/HeadDim |Tile Block (Br, Bc)|MMA (m16n8k16)|
|:---:|:---:|:---:|:---:|
|✔️|✔️|✔️|✔️|
|Pack LDST (pack 128 bits)|SMEM Padding|Copy Async (cp.async.cg/ca)|Tile MMA (More Threads)
|✔️|✔️|✔️|✔️|
|Tile Warp (More Values)|Multi Stages (1/2)|Collective Store (Warp Shuffle & Reg Reuse)|**Split KV/Q**|
|✔️|✔️|✔️|✔️|
|**Shared QKV/KV** SMEM|**Prefetch Q** s2r|**Prefetch K/V** g2s|SMEM/Block Swizzle|
|✔️|✔️|✔️|?|

This repository's implementation of FlashAttention is intended solely for learning CUDA programming. For optimal performance, please use the official [flash-attention](https://github.com/Dao-AILab/flash-attention). Currently, for small-scale attention `(B<=4, H <=48, SeqLen <= 8192)` can run faster than offical FA2 on some Devices, for example, NVIDIA RTX 3080 Laptop. However, for large-scale attention computations, there remains a performance gap. Performance optimizations are ongoing; stay tuned for updates.

- Example: B=1, H=8, N=8192, D=64 (NVIDIA RTX 3080 Laptop)
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

- Example: B=1, H=48, N=8192, D=64 (NVIDIA RTX 3080 Laptop)
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
- Example: B=1, H=48, N=8192, D=512 (NVIDIA RTX 3080 Laptop), FA2 not supported.
```bash
 python3 flash_attn_mma.py --B 1 --H 8 --N 8192 --iters 10 --show-all --sdpa --D 512 # NVIDIA RTX 3080 Laptop, Faster than SDPA
-----------------------B: batch_size, H: n_head, N: seq_len, D: head_dim, seed: 1041, Warmup: 1, Iters: 10------------------------
----------------------------------------------------------------------------------------------------------------------------------
------------------------------------------B=1, H=8, N=8192, D=512, Warmup: 1, Iters: 10-------------------------------------------
   mma(split-q+tiling-qk+stage1): ['-0.00433731 ', '0.02165222  ', '-0.01544189 '], time:48.775554ms, TFLOPS:22.60 (+0.00%)
   mma(split-q+tiling-qk+stage2): ['-0.00433731 ', '0.02165222  ', '-0.01544189 '], time:47.503424ms, TFLOPS:23.20 (+2.68%)
                          (sdpa): ['-0.00438309 ', '0.02174377  ', '-0.01551056 '], time:66.486573ms, TFLOPS:16.58
----------------------------------------------------------------------------------------------------------------------------------
```

## 📖 Contents

- [📖 FlashAttetion MMA Kernels](#mma)
  - [📚 Split KV](#mma-split-kv)
  - [📚 Split Q ](#mma-split-q)
  - [📚 Shared KV SMEM](#mma-share-kv)
  - [📚 Fully Shared QKV SMEM](#mma-share-qkv)
  - [📚 QK Fine-grained Tiling](#mma-tiling-qk)
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
// Fine-grained tiling (MMA level) for Q/K, it cause constant SRAM size 64*kMmaAtomK for Q/K, 
// and O(kMmaAtomK*d) SRAM complexity for V, thus, the SRAM complexity is O(kMmaAtomK*d).
// Thus, we can extend D(headdim) to 1024. Performance is stay tuned for updates ~
__global__ void // Q, K, V, O -> [B, H, N, D]
flash_attn_mma_stages_split_q_tiling_qk_kernel(half* Q, half* K, half* V, half* O, ...);
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

- B=2, H=2, N=4096, D=64
  
```bash
python3 flash_attn_mma.py --B 2 --H 2 --D 64 --N 4096 --iters 10 # NVIDIA RTX 3080 Laptop
------------------------------------------------------------------------------------------------------------------------
                    B: batch_size, H: n_head, N: seq_len, D: head_dim, seed: 9655, Warmup: 1, Iters: 10
------------------------------------------------------------------------------------------------------------------------
                              B=2, H=2, N=4096, D=64, Warmup: 1, Iters: 10
          mma(split-kv+stage1): ['0.01901245  ', '-0.02037048 ', '-0.01722717 '], time:0.765753ms, TFLOPS:22.87
          mma(split-kv+stage2): ['0.01901245  ', '-0.02037048 ', '-0.01722717 '], time:0.731516ms, TFLOPS:23.94
           mma(split-q+stage1): ['0.01901245  ', '-0.02037048 ', '-0.01722717 '], time:0.526834ms, TFLOPS:33.24
           mma(split-q+stage2): ['0.01901245  ', '-0.02037048 ', '-0.01722717 '], time:0.660753ms, TFLOPS:26.51
  mma(split-q+share-kv+stage1): ['0.01901245  ', '-0.02037048 ', '-0.01722717 '], time:0.460815ms, TFLOPS:38.01
 mma(split-q+share-qkv+stage1): ['0.01901245  ', '-0.02037048 ', '-0.01722717 '], time:0.465345ms, TFLOPS:37.64
 mma(split-q+share-qkv+stage2): ['0.01901245  ', '-0.02037048 ', '-0.01722717 '], time:0.474334ms, TFLOPS:36.92
                       (flash): ['0.01904297  ', '-0.02037048 ', '-0.01724243 '], time:0.596189ms, TFLOPS:29.38
------------------------------------------------------------------------------------------------------------------------
```


- B=2, H=2, N=8192, D=64
```bash
 python3 flash_attn_mma.py --B 1 --H 8 --D 64 --N 8192 --iters 10 # NVIDIA RTX 3080 Laptop
------------------------------------------------------------------------------------------------------------------------
                    B: batch_size, H: n_head, N: seq_len, D: head_dim, seed: 5669, Warmup: 1, Iters: 10
------------------------------------------------------------------------------------------------------------------------
                              B=1, H=8, N=8192, D=64, Warmup: 1, Iters: 10
          mma(split-kv+stage1): ['-0.0087738  ', '0.012146    ', '-0.01319122 '], time:5.572367ms, TFLOPS:25.15
          mma(split-kv+stage2): ['-0.0087738  ', '0.012146    ', '-0.01319122 '], time:5.295920ms, TFLOPS:26.46
           mma(split-q+stage1): ['-0.0087738  ', '0.012146    ', '-0.01319122 '], time:3.607082ms, TFLOPS:38.85
           mma(split-q+stage2): ['-0.0087738  ', '0.012146    ', '-0.01319122 '], time:4.600883ms, TFLOPS:30.45
  mma(split-q+share-kv+stage1): ['-0.0087738  ', '0.012146    ', '-0.01319122 '], time:2.744508ms, TFLOPS:51.05
 mma(split-q+share-qkv+stage1): ['-0.0087738  ', '0.012146    ', '-0.01319122 '], time:2.700114ms, TFLOPS:51.89
 mma(split-q+share-qkv+stage2): ['-0.0087738  ', '0.012146    ', '-0.01319122 '], time:2.692103ms, TFLOPS:52.05
                       (flash): ['-0.00882721 ', '0.01213074  ', '-0.01314545 '], time:3.778219ms, TFLOPS:37.09
------------------------------------------------------------------------------------------------------------------------
```

- B=1, H=8, N=8192, D=64
```bash
python3 flash_attn_mma.py --B 1 --H 8 --D 64 --N 8192 --iters 10 # NVIDIA RTX 3080 Laptop
------------------------------------------------------------------------------------------------------------------------
                    B: batch_size, H: n_head, N: seq_len, D: head_dim, seed: 1617, Warmup: 1, Iters: 10
------------------------------------------------------------------------------------------------------------------------
                              B=1, H=8, N=8192, D=64, Warmup: 1, Iters: 10
          mma(split-kv+stage1): ['0.01960754  ', '0.01452637  ', '-0.02592468 '], time:5.586338ms, TFLOPS:25.08
          mma(split-kv+stage2): ['0.01960754  ', '0.01452637  ', '-0.02592468 '], time:5.326223ms, TFLOPS:26.31
           mma(split-q+stage1): ['0.01960754  ', '0.01452637  ', '-0.02592468 '], time:3.834152ms, TFLOPS:36.54
           mma(split-q+stage2): ['0.01960754  ', '0.01452637  ', '-0.02592468 '], time:4.328346ms, TFLOPS:32.37
  mma(split-q+share-kv+stage1): ['0.01960754  ', '0.01452637  ', '-0.02592468 '], time:2.636528ms, TFLOPS:53.15
 mma(split-q+share-qkv+stage1): ['0.01960754  ', '0.01452637  ', '-0.02592468 '], time:2.594471ms, TFLOPS:54.01
 mma(split-q+share-qkv+stage2): ['0.01960754  ', '0.01452637  ', '-0.02592468 '], time:2.574611ms, TFLOPS:54.42
                       (flash): ['0.01963806  ', '0.0145874   ', '-0.02593994 '], time:3.764462ms, TFLOPS:37.22
-----------------------------------------------------------------------------------------------------------------------
```

- B=1, H=48, N=8192, D=64  
```bash
python3 flash_attn_mma.py --B 1 --H 48 --D 64 --N 8192 --iters 10 # NVIDIA RTX 3080 Laptop
------------------------------------------------------------------------------------------------------------------------
                    B: batch_size, H: n_head, N: seq_len, D: head_dim, seed: 4669, Warmup: 1, Iters: 10
------------------------------------------------------------------------------------------------------------------------
                              B=1, H=48, N=8192, D=64, Warmup: 1, Iters: 10
          mma(split-kv+stage1): ['-0.01280212 ', '-0.02825928 ', '0.0146637   '], time:42.534423ms, TFLOPS:19.77
          mma(split-kv+stage2): ['-0.01280212 ', '-0.02825928 ', '0.0146637   '], time:42.349815ms, TFLOPS:19.85
           mma(split-q+stage1): ['-0.01280212 ', '-0.02825928 ', '0.0146637   '], time:35.657477ms, TFLOPS:23.58
           mma(split-q+stage2): ['-0.01280212 ', '-0.02825928 ', '0.0146637   '], time:36.065412ms, TFLOPS:23.31
  mma(split-q+share-kv+stage1): ['-0.01280212 ', '-0.02825928 ', '0.0146637   '], time:23.619652ms, TFLOPS:35.59
 mma(split-q+share-qkv+stage1): ['-0.01280212 ', '-0.02825928 ', '0.0146637   '], time:23.893070ms, TFLOPS:35.19
 mma(split-q+share-qkv+stage2): ['-0.01280212 ', '-0.02825928 ', '0.0146637   '], time:23.590446ms, TFLOPS:35.64
                       (flash): ['-0.01280212 ', '-0.02825928 ', '0.0146637   '], time:22.385812ms, TFLOPS:37.56
------------------------------------------------------------------------------------------------------------------------
```

- B=1, H=8, N=8192, D=32  
```bash
python3 flash_attn_mma.py --B 1 --H 8 --D 32 --N 8192 --iters 10 # NVIDIA RTX 3080 Laptop
------------------------------------------------------------------------------------------------------------------------
                    B: batch_size, H: n_head, N: seq_len, D: head_dim, seed: 2322, Warmup: 1, Iters: 10
------------------------------------------------------------------------------------------------------------------------
                              B=1, H=8, N=8192, D=32, Warmup: 1, Iters: 10
          mma(split-kv+stage1): ['-0.00616074 ', '-0.00230789 ', '0.02029419  '], time:3.930807ms, TFLOPS:18.16
          mma(split-kv+stage2): ['-0.00616074 ', '-0.00230789 ', '0.02029419  '], time:3.901839ms, TFLOPS:18.30
           mma(split-q+stage1): ['-0.00616074 ', '-0.00230789 ', '0.02029419  '], time:1.839685ms, TFLOPS:38.81
           mma(split-q+stage2): ['-0.00607681 ', '-0.00229454 ', '0.02029419  '], time:1.511669ms, TFLOPS:47.23
  mma(split-q+share-kv+stage1): ['-0.00616074 ', '-0.00230789 ', '0.02029419  '], time:1.400948ms, TFLOPS:50.97
 mma(split-q+share-qkv+stage1): ['-0.00616074 ', '-0.00230789 ', '0.02029419  '], time:1.393318ms, TFLOPS:51.25
 mma(split-q+share-qkv+stage2): ['-0.00616074 ', '-0.00230789 ', '0.02029419  '], time:1.322961ms, TFLOPS:53.97
                       (flash): ['-0.00617599 ', '-0.00231934 ', '0.02029419  '], time:1.810646ms, TFLOPS:39.43
------------------------------------------------------------------------------------------------------------------------
```
