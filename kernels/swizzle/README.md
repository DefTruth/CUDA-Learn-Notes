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

## 📚 performance  

- NVIDIA TRX 3080 Laptop
```bash
 ./hgemm_mma_swizzle.bin 4096 4096 4096 1 10

ALGO = HGEMM MMA NAIVE
M N K =   4096   4096   4096, W = 1, R = 10, Time =   0.02986609 s, AVG Performance =     4.6018 Tflops

ALGO = HGEMM MMA NAIVE + SMEM SWIZZLE
M N K =   4096   4096   4096, W = 1, R = 10, Time =   0.02860964 s, AVG Performance =     4.8039 Tflops

ALGO = HGEMM mma2x4_warp4x4
M N K =   4096   4096   4096, W = 1, R = 10, Time =   0.00392888 s, AVG Performance =    34.9817 Tflops

ALGO = HGEMM mma2x4_warp4x4 + SMEM SWIZZLE
M N K =   4096   4096   4096, W = 1, R = 10, Time =   0.00234496 s, AVG Performance =    58.6104 Tflops
```


## 📚 print swizzle layout  

- M16K16  

```bash
python3 print_swizzle_layout.py --logical-col 64 --show-logical-col
----------------------------------------------------------------
[INFO] Assert smem store layout col_stride <= 16, prefer 16.   |
[INFO] For logical_col_stride > 16, we have to permute the     |
[INFO] smem store layout using col major ZigZag method:        |
[INFO] e.g, --> Q smem logical layout [Br][64].                |
[INFO]      --> col major ZigZag permuted -->                  |
[INFO]      --> Q smem store layout [4][Br][16].               |
----------------------------------------------------------------
----------------------
----swizzle layout----
logical col 0~16, step 8
smem col 0~16, step 8-
----------------------
|bank  |b 0~3 |b 4~7 |
|row 0 | 0:0  | 8:8  |
|bank  |b 8~11|b12~15|
|row 1 | 0:0  | 8:8  |
|bank  |b16~19|b20~23|
|row 2 | 0:0  | 8:8  |
|bank  |b24~27|b28~31|
|row 3 | 0:0  | 8:8  |
----------------------
|bank  |b 0~3 |b 4~7 |
|row 4 | 0:8  | 8:0  |
|bank  |b 8~11|b12~15|
|row 5 | 0:8  | 8:0  |
|bank  |b16~19|b20~23|
|row 6 | 0:8  | 8:0  |
|bank  |b24~27|b28~31|
|row 7 | 0:8  | 8:0  |
----------------------
|bank  |b 0~3 |b 4~7 |
|row 8 | 0:0  | 8:8  |
|bank  |b 8~11|b12~15|
|row 9 | 0:0  | 8:8  |
|bank  |b16~19|b20~23|
|row 10| 0:0  | 8:8  |
|bank  |b24~27|b28~31|
|row 11| 0:0  | 8:8  |
----------------------
|bank  |b 0~3 |b 4~7 |
|row 12| 0:8  | 8:0  |
|bank  |b 8~11|b12~15|
|row 13| 0:8  | 8:0  |
|bank  |b16~19|b20~23|
|row 14| 0:8  | 8:0  |
|bank  |b24~27|b28~31|
|row 15| 0:8  | 8:0  |
----------------------
```

- M16K64  

```bash
python3 print_swizzle_layout.py --logical-col 64 --show-logical-col
----------------------------------------------------------------
[INFO] Assert smem store layout col_stride <= 16, prefer 16.   |
[INFO] For logical_col_stride > 16, we have to permute the     |
[INFO] smem store layout using col major ZigZag method:        |
[INFO] e.g, --> Q smem logical layout [Br][64].                |
[INFO]      --> col major ZigZag permuted -->                  |
[INFO]      --> Q smem store layout [4][Br][16].               |
----------------------------------------------------------------
----------------------------------------------------------------
-------------------------swizzle layout-------------------------
--------------------logical col 0~64, step 8--------------------
---------------------smem col 0~16, step 8----------------------
----------------------------------------------------------------
|bank  |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |
|row 0 | 0:0  | 8:8  |16:0  |24:8  |32:0  |40:8  |48:0  |56:8  |
|bank  |b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|
|row 1 | 0:0  | 8:8  |16:0  |24:8  |32:0  |40:8  |48:0  |56:8  |
|bank  |b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|
|row 2 | 0:0  | 8:8  |16:0  |24:8  |32:0  |40:8  |48:0  |56:8  |
|bank  |b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|
|row 3 | 0:0  | 8:8  |16:0  |24:8  |32:0  |40:8  |48:0  |56:8  |
----------------------------------------------------------------
|bank  |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |
|row 4 | 0:8  | 8:0  |16:8  |24:0  |32:8  |40:0  |48:8  |56:0  |
|bank  |b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|
|row 5 | 0:8  | 8:0  |16:8  |24:0  |32:8  |40:0  |48:8  |56:0  |
|bank  |b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|
|row 6 | 0:8  | 8:0  |16:8  |24:0  |32:8  |40:0  |48:8  |56:0  |
|bank  |b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|
|row 7 | 0:8  | 8:0  |16:8  |24:0  |32:8  |40:0  |48:8  |56:0  |
----------------------------------------------------------------
|bank  |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |
|row 8 | 0:0  | 8:8  |16:0  |24:8  |32:0  |40:8  |48:0  |56:8  |
|bank  |b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|
|row 9 | 0:0  | 8:8  |16:0  |24:8  |32:0  |40:8  |48:0  |56:8  |
|bank  |b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|
|row 10| 0:0  | 8:8  |16:0  |24:8  |32:0  |40:8  |48:0  |56:8  |
|bank  |b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|
|row 11| 0:0  | 8:8  |16:0  |24:8  |32:0  |40:8  |48:0  |56:8  |
----------------------------------------------------------------
|bank  |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |
|row 12| 0:8  | 8:0  |16:8  |24:0  |32:8  |40:0  |48:8  |56:0  |
|bank  |b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|
|row 13| 0:8  | 8:0  |16:8  |24:0  |32:8  |40:0  |48:8  |56:0  |
|bank  |b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|
|row 14| 0:8  | 8:0  |16:8  |24:0  |32:8  |40:0  |48:8  |56:0  |
|bank  |b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|
|row 15| 0:8  | 8:0  |16:8  |24:0  |32:8  |40:0  |48:8  |56:0  |
----------------------------------------------------------------
```

