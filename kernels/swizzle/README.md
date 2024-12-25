# 📖 Learn how to apply SMEM Swizzle for bank conflicts free

## 📚 build bin

```bash
make
```

## 📚 ncu profile

Achieve 0 bank conflicts for LDSM via smem swizzle.

```bash
ncu --metrics l1tex__data_bank_reads ./mat_trans_swizzle.bin
ncu --metrics l1tex__data_bank_writes ./mat_trans_swizzle.bin
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld ./mat_trans_swizzle.bin
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st ./mat_trans_swizzle.bin

ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld ./hgemm_mma_swizzle.bin 1024 1024 1024 0 1
ncu --metrics sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm ./hgemm_mma_swizzle.bin 1024 1024 1024 0 1
```

log: (achieve 0 bank conflicts for LDSM via smem swizzle)

```bash
ncu --metrics sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm ./hgemm_mma_swizzle.bin 1024 1024 1024 0 1
[1542675] hgemm_mma_swizzle.bin@127.0.0.1
  void hgemm_mma_m16n8k16_naive_kernel<16, 8, 16>(__half *, __half *, __half *, int, int, int) (128, 64, 1)x(32, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    ------------------------------------------------------------------ ----------- ------------
    Metric Name                                                        Metric Unit Metric Value
    ------------------------------------------------------------------ ----------- ------------
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.avg                 22795.13
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.max                    24576
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.min                    18432
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.sum                  2097152
    ------------------------------------------------------------------ ----------- ------------

  void hgemm_mma_m16n8k16_naive_smem_swizzle_kernel<16, 8, 16>(__half *, __half *, __half *, int, int, int) (128, 64, 1)x(32, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    ------------------------------------------------------------------ ----------- ------------
    Metric Name                                                        Metric Unit Metric Value
    ------------------------------------------------------------------ ----------- ------------
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.avg                        0
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.max                        0
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.min                        0
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.sum                        0
    ------------------------------------------------------------------ ----------- ------------

  void hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel<16, 8, 16, 2, 4, 4, 4, 0, 0>(__half *, __half *, __half *, int, int, int) (8, 8, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    ------------------------------------------------------------------ ----------- ------------
    Metric Name                                                        Metric Unit Metric Value
    ------------------------------------------------------------------ ----------- ------------
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.avg                 25644.52
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.max                    36864
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.min                        0
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.sum                  2359296
    ------------------------------------------------------------------ ----------- ------------

  void hgemm_mma_m16n8k16_mma2x4_warp4x4_smem_swizzle_kernel<16, 8, 16, 2, 4, 4, 4, 0, 8>(__half *, __half *, __half *, int, int, int) (8, 8, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    ------------------------------------------------------------------ ----------- ------------
    Metric Name                                                        Metric Unit Metric Value
    ------------------------------------------------------------------ ----------- ------------
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.avg                        0
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.max                        0
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.min                        0
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.sum                        0
    ------------------------------------------------------------------ ----------- ------------
```
## 📚 print swizzle layout  
```bash
python3 print_swizzle_layout.py --col 64
-------------------------------------------
--------------swizzle layout---------------
-------------col 0~64, step 8--------------
-------------------------------------------
| row 0  | (0, 8, 16, 24, 32, 40, 48, 56) |
| row 1  | (0, 8, 16, 24, 32, 40, 48, 56) |
| row 2  | (0, 8, 16, 24, 32, 40, 48, 56) |
| row 3  | (0, 8, 16, 24, 32, 40, 48, 56) |
-------------------------------------------
| row 4  | (8, 0, 24, 16, 40, 32, 56, 48) |
| row 5  | (8, 0, 24, 16, 40, 32, 56, 48) |
| row 6  | (8, 0, 24, 16, 40, 32, 56, 48) |
| row 7  | (8, 0, 24, 16, 40, 32, 56, 48) |
-------------------------------------------
| row 8  | (16, 24, 0, 8, 48, 56, 32, 40) |
| row 9  | (16, 24, 0, 8, 48, 56, 32, 40) |
| row 10 | (16, 24, 0, 8, 48, 56, 32, 40) |
| row 11 | (16, 24, 0, 8, 48, 56, 32, 40) |
-------------------------------------------
| row 12 | (24, 16, 8, 0, 56, 48, 40, 32) |
| row 13 | (24, 16, 8, 0, 56, 48, 40, 32) |
| row 14 | (24, 16, 8, 0, 56, 48, 40, 32) |
| row 15 | (24, 16, 8, 0, 56, 48, 40, 32) |
-------------------------------------------

python3 print_swizzle_layout.py --col 16
-------------------
--swizzle layout---
-col 0~16, step 8--
-------------------
| row 0  | (0, 8) |
| row 1  | (0, 8) |
| row 2  | (0, 8) |
| row 3  | (0, 8) |
-------------------
| row 4  | (8, 0) |
| row 5  | (8, 0) |
| row 6  | (8, 0) |
| row 7  | (8, 0) |
-------------------
| row 8  | (0, 8) |
| row 9  | (0, 8) |
| row 10 | (0, 8) |
| row 11 | (0, 8) |
-------------------
| row 12 | (8, 0) |
| row 13 | (8, 0) |
| row 14 | (8, 0) |
| row 15 | (8, 0) |
-------------------
```
