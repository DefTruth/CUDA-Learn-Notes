# Mat Transpose

## 0x00 说明

包含以下内容：

- [X] mat_transpose_f32_col2row_kernel
- [X] mat_transpose_f32_row2col_kernel
- [X] mat_transpose_f32x4_col2row_kernel(float4向量化版本)
- [X] mat_transpose_f32x4_row2col_kernel(float4向量化版本)
- [X] mat_transpose_f32_diagnonal(对角轴应用于S=K)
- [X] mat_transpose_f32x4_shared_col2row_kernel(float4向量化版本，共享内存)
- [X] mat_transpose_f32x4_shared_row2col_kernel(float4向量化版本，共享内存)
- [X] mat_transpose_f32x4_shared_bcf_col2row_kernel(float4向量化版本，共享内存，去bank conflict)
- [X] mat_transpose_f32x4_shared_bcf_row2col_kernel(float4向量化版本，共享内存，去bank conflict)
- [X] PyTorch bindings

虽然是基础操作但是很适合练手，比矩阵乘法难度低一点但是可以其中可以用到的优化技巧都可以想办法用到这里来。

## 测试

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长: Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada 
python3 mat_transpose.py
```

输出:

```bash
------------------------------------------------------------------------------------------------------------------------
                                                  S=1024, K=1024
                  out_original: [0.2706067, 1.89055979, 0.62714416], validate False, time:0.00007796ms
               out_f32_col2row: [0.2706067, 0.62714416, 1.89055979], validate True , time:0.03732634ms
               out_f32_row2col: [0.2706067, 0.62714416, 1.89055979], validate True , time:0.03055906ms
           out_f32_col2row(2d): [0.2706067, 0.62714416, 1.89055979], validate True , time:0.02096868ms
           out_f32_row2col(2d): [0.2706067, 0.62714416, 1.89055979], validate True , time:0.03112197ms
             out_f32_diagnonal: [0.2706067, 0.62714416, 1.89055979], validate True , time:0.02037907ms
             out_f32x4_col2row: [0.2706067, 0.62714416, 1.89055979], validate True , time:0.06107259ms
             out_f32x4_row2col: [0.2706067, 0.62714416, 1.89055979], validate True , time:0.02692676ms
         out_f32x4_col2row(2d): [0.2706067, 0.62714416, 1.89055979], validate True , time:0.03207874ms
         out_f32x4_row2col(2d): [0.2706067, 0.62714416, 1.89055979], validate True , time:0.01719213ms
      out_f32x4_shared_col2row: [0.2706067, 0.62714416, 1.89055979], validate True , time:0.01326251ms
      out_f32x4_shared_row2col: [0.2706067, 0.62714416, 1.89055979], validate True , time:0.02352262ms
  out_f32x4_shared_bcf_col2row: [0.2706067, 0.62714416, 1.89055979], validate True , time:0.01917195ms
  out_f32x4_shared_bcf_row2col: [0.2706067, 0.62714416, 1.89055979], validate True , time:0.01389265ms
                    out_f32_th: [0.2706067, 0.62714416, 1.89055979], validate True , time:0.05057526ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                  S=1024, K=2048
                  out_original: [0.1013972, 0.10635406, 0.45091254], validate False, time:0.00007367ms
               out_f32_col2row: [0.1013972, 0.45091254, 0.10635406], validate True , time:0.11233115ms
               out_f32_row2col: [0.1013972, 0.45091254, 0.10635406], validate True , time:0.05733228ms
           out_f32_col2row(2d): [0.1013972, 0.45091254, 0.10635406], validate True , time:0.04851723ms
           out_f32_row2col(2d): [0.1013972, 0.45091254, 0.10635406], validate True , time:0.05224919ms
             out_f32x4_col2row: [0.1013972, 0.45091254, 0.10635406], validate True , time:0.10379744ms
             out_f32x4_row2col: [0.1013972, 0.45091254, 0.10635406], validate True , time:0.05431175ms
         out_f32x4_col2row(2d): [0.1013972, 0.45091254, 0.10635406], validate True , time:0.05774999ms
         out_f32x4_row2col(2d): [0.1013972, 0.45091254, 0.10635406], validate True , time:0.03115702ms
      out_f32x4_shared_col2row: [0.1013972, 0.45091254, 0.10635406], validate True , time:0.03814983ms
      out_f32x4_shared_row2col: [0.1013972, 0.45091254, 0.10635406], validate True , time:0.03473568ms
  out_f32x4_shared_bcf_col2row: [0.1013972, 0.45091254, 0.10635406], validate True , time:0.03495407ms
  out_f32x4_shared_bcf_row2col: [0.1013972, 0.45091254, 0.10635406], validate True , time:0.03433728ms
                    out_f32_th: [0.1013972, 0.45091254, 0.10635406], validate True , time:0.08867288ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                  S=1024, K=4096
                  out_original: [1.78550363, -1.60489535, -0.16560346], validate False, time:0.00007296ms
               out_f32_col2row: [1.78550363, -0.16560346, -1.60489535], validate True , time:0.19823909ms
               out_f32_row2col: [1.78550363, -0.16560346, -1.60489535], validate True , time:0.11195445ms
           out_f32_col2row(2d): [1.78550363, -0.16560346, -1.60489535], validate True , time:0.09996772ms
           out_f32_row2col(2d): [1.78550363, -0.16560346, -1.60489535], validate True , time:0.09864736ms
             out_f32x4_col2row: [1.78550363, -0.16560346, -1.60489535], validate True , time:0.19718719ms
             out_f32x4_row2col: [1.78550363, -0.16560346, -1.60489535], validate True , time:0.11092091ms
         out_f32x4_col2row(2d): [1.78550363, -0.16560346, -1.60489535], validate True , time:0.10105634ms
         out_f32x4_row2col(2d): [1.78550363, -0.16560346, -1.60489535], validate True , time:0.06530714ms
      out_f32x4_shared_col2row: [1.78550363, -0.16560346, -1.60489535], validate True , time:0.06287837ms
      out_f32x4_shared_row2col: [1.78550363, -0.16560346, -1.60489535], validate True , time:0.07055283ms
  out_f32x4_shared_bcf_col2row: [1.78550363, -0.16560346, -1.60489535], validate True , time:0.06612253ms
  out_f32x4_shared_bcf_row2col: [1.78550363, -0.16560346, -1.60489535], validate True , time:0.06411195ms
                    out_f32_th: [1.78550363, -0.16560346, -1.60489535], validate True , time:0.17973542ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                  S=2048, K=1024
                  out_original: [-0.96589017, -0.53940338, 1.51841831], validate False, time:0.00007153ms
               out_f32_col2row: [-0.96589017, 1.51841831, -0.53940338], validate True , time:0.10408664ms
               out_f32_row2col: [-0.96589017, 1.51841831, -0.53940338], validate True , time:0.05784106ms
           out_f32_col2row(2d): [-0.96589017, 1.51841831, -0.53940338], validate True , time:0.04911971ms
           out_f32_row2col(2d): [-0.96589017, 1.51841831, -0.53940338], validate True , time:0.04792857ms
             out_f32x4_col2row: [-0.96589017, 1.51841831, -0.53940338], validate True , time:0.15571523ms
             out_f32x4_row2col: [-0.96589017, 1.51841831, -0.53940338], validate True , time:0.07688594ms
         out_f32x4_col2row(2d): [-0.96589017, 1.51841831, -0.53940338], validate True , time:0.05413485ms
         out_f32x4_row2col(2d): [-0.96589017, 1.51841831, -0.53940338], validate True , time:0.03497577ms
      out_f32x4_shared_col2row: [-0.96589017, 1.51841831, -0.53940338], validate True , time:0.04818010ms
      out_f32x4_shared_row2col: [-0.96589017, 1.51841831, -0.53940338], validate True , time:0.05148292ms
  out_f32x4_shared_bcf_col2row: [-0.96589017, 1.51841831, -0.53940338], validate True , time:0.04849076ms
  out_f32x4_shared_bcf_row2col: [-0.96589017, 1.51841831, -0.53940338], validate True , time:0.03030324ms
                    out_f32_th: [-0.96589017, 1.51841831, -0.53940338], validate True , time:0.09853792ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                  S=2048, K=2048
                  out_original: [0.66138971, 0.43854904, -1.19618118], validate False, time:0.00007439ms
               out_f32_col2row: [0.66138971, -1.19618118, 0.43854904], validate True , time:0.24223709ms
               out_f32_row2col: [0.66138971, -1.19618118, 0.43854904], validate True , time:0.15707016ms
           out_f32_col2row(2d): [0.66138971, -1.19618118, 0.43854904], validate True , time:0.09814286ms
           out_f32_row2col(2d): [0.66138971, -1.19618118, 0.43854904], validate True , time:0.13747311ms
             out_f32_diagnonal: [0.66138971, -1.19618118, 0.43854904], validate True , time:0.08852434ms
             out_f32x4_col2row: [0.66138971, -1.19618118, 0.43854904], validate True , time:0.26274681ms
             out_f32x4_row2col: [0.66138971, -1.19618118, 0.43854904], validate True , time:0.12002778ms
         out_f32x4_col2row(2d): [0.66138971, -1.19618118, 0.43854904], validate True , time:0.15025878ms
         out_f32x4_row2col(2d): [0.66138971, -1.19618118, 0.43854904], validate True , time:0.07008457ms
      out_f32x4_shared_col2row: [0.66138971, -1.19618118, 0.43854904], validate True , time:0.07605863ms
      out_f32x4_shared_row2col: [0.66138971, -1.19618118, 0.43854904], validate True , time:0.09375811ms
  out_f32x4_shared_bcf_col2row: [0.66138971, -1.19618118, 0.43854904], validate True , time:0.07940960ms
  out_f32x4_shared_bcf_row2col: [0.66138971, -1.19618118, 0.43854904], validate True , time:0.07159257ms
                    out_f32_th: [0.66138971, -1.19618118, 0.43854904], validate True , time:0.25392270ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                  S=2048, K=4096
                  out_original: [0.21140628, 0.86610204, -0.61084032], validate False, time:0.00007534ms
               out_f32_col2row: [0.21140628, -0.61084032, 0.86610204], validate True , time:0.51111245ms
               out_f32_row2col: [0.21140628, -0.61084032, 0.86610204], validate True , time:0.29512668ms
           out_f32_col2row(2d): [0.21140628, -0.61084032, 0.86610204], validate True , time:0.25763965ms
           out_f32_row2col(2d): [0.21140628, -0.61084032, 0.86610204], validate True , time:0.25509524ms
             out_f32x4_col2row: [0.21140628, -0.61084032, 0.86610204], validate True , time:0.47753954ms
             out_f32x4_row2col: [0.21140628, -0.61084032, 0.86610204], validate True , time:0.27053690ms
         out_f32x4_col2row(2d): [0.21140628, -0.61084032, 0.86610204], validate True , time:0.26033616ms
         out_f32x4_row2col(2d): [0.21140628, -0.61084032, 0.86610204], validate True , time:0.16601658ms
      out_f32x4_shared_col2row: [0.21140628, -0.61084032, 0.86610204], validate True , time:0.14935517ms
      out_f32x4_shared_row2col: [0.21140628, -0.61084032, 0.86610204], validate True , time:0.17617536ms
  out_f32x4_shared_bcf_col2row: [0.21140628, -0.61084032, 0.86610204], validate True , time:0.14183927ms
  out_f32x4_shared_bcf_row2col: [0.21140628, -0.61084032, 0.86610204], validate True , time:0.17589092ms
                    out_f32_th: [0.21140628, -0.61084032, 0.86610204], validate True , time:0.43119144ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                  S=4096, K=1024
                  out_original: [-0.33594334, -0.13206008, 0.8452214], validate False, time:0.00007868ms
               out_f32_col2row: [-0.33594334, 0.8452214, -0.13206008], validate True , time:0.26727128ms
               out_f32_row2col: [-0.33594334, 0.8452214, -0.13206008], validate True , time:0.17777562ms
           out_f32_col2row(2d): [-0.33594334, 0.8452214, -0.13206008], validate True , time:0.09764647ms
           out_f32_row2col(2d): [-0.33594334, 0.8452214, -0.13206008], validate True , time:0.13735604ms
             out_f32x4_col2row: [-0.33594334, 0.8452214, -0.13206008], validate True , time:0.25628328ms
             out_f32x4_row2col: [-0.33594334, 0.8452214, -0.13206008], validate True , time:0.15057874ms
         out_f32x4_col2row(2d): [-0.33594334, 0.8452214, -0.13206008], validate True , time:0.12607431ms
         out_f32x4_row2col(2d): [-0.33594334, 0.8452214, -0.13206008], validate True , time:0.09281611ms
      out_f32x4_shared_col2row: [-0.33594334, 0.8452214, -0.13206008], validate True , time:0.07143378ms
      out_f32x4_shared_row2col: [-0.33594334, 0.8452214, -0.13206008], validate True , time:0.08804989ms
  out_f32x4_shared_bcf_col2row: [-0.33594334, 0.8452214, -0.13206008], validate True , time:0.09320903ms
  out_f32x4_shared_bcf_row2col: [-0.33594334, 0.8452214, -0.13206008], validate True , time:0.07376838ms
                    out_f32_th: [-0.33594334, 0.8452214, -0.13206008], validate True , time:0.25272131ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                  S=4096, K=2048
                  out_original: [1.44601941, 1.46612203, -2.00953078], validate False, time:0.00007796ms
               out_f32_col2row: [1.44601941, -2.00953078, 1.46612203], validate True , time:0.51826644ms
               out_f32_row2col: [1.44601941, -2.00953078, 1.46612203], validate True , time:0.31751609ms
           out_f32_col2row(2d): [1.44601941, -2.00953078, 1.46612203], validate True , time:0.26685858ms
           out_f32_row2col(2d): [1.44601941, -2.00953078, 1.46612203], validate True , time:0.18520737ms
             out_f32x4_col2row: [1.44601941, -2.00953078, 1.46612203], validate True , time:0.29121876ms
             out_f32x4_row2col: [1.44601941, -2.00953078, 1.46612203], validate True , time:0.16650081ms
         out_f32x4_col2row(2d): [1.44601941, -2.00953078, 1.46612203], validate True , time:0.14630580ms
         out_f32x4_row2col(2d): [1.44601941, -2.00953078, 1.46612203], validate True , time:0.09408069ms
      out_f32x4_shared_col2row: [1.44601941, -2.00953078, 1.46612203], validate True , time:0.09475493ms
      out_f32x4_shared_row2col: [1.44601941, -2.00953078, 1.46612203], validate True , time:0.09508491ms
  out_f32x4_shared_bcf_col2row: [1.44601941, -2.00953078, 1.46612203], validate True , time:0.09532118ms
  out_f32x4_shared_bcf_row2col: [1.44601941, -2.00953078, 1.46612203], validate True , time:0.09467864ms
                    out_f32_th: [1.44601941, -2.00953078, 1.46612203], validate True , time:0.26716113ms
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
                                                  S=4096, K=4096
                  out_original: [-1.07092094, -1.13755226, 0.99070781], validate False, time:0.00007606ms
               out_f32_col2row: [-1.07092094, 0.99070781, -1.13755226], validate True , time:0.75331712ms
               out_f32_row2col: [-1.07092094, 0.99070781, -1.13755226], validate True , time:0.52119255ms
           out_f32_col2row(2d): [-1.07092094, 0.99070781, -1.13755226], validate True , time:0.36621094ms
           out_f32_row2col(2d): [-1.07092094, 0.99070781, -1.13755226], validate True , time:0.36603284ms
             out_f32_diagnonal: [-1.07092094, 0.99070781, -1.13755226], validate True , time:0.37416911ms
             out_f32x4_col2row: [-1.07092094, 0.99070781, -1.13755226], validate True , time:0.96249247ms
             out_f32x4_row2col: [-1.07092094, 0.99070781, -1.13755226], validate True , time:0.56916833ms
         out_f32x4_col2row(2d): [-1.07092094, 0.99070781, -1.13755226], validate True , time:0.48158646ms
         out_f32x4_row2col(2d): [-1.07092094, 0.99070781, -1.13755226], validate True , time:0.30216074ms
      out_f32x4_shared_col2row: [-1.07092094, 0.99070781, -1.13755226], validate True , time:0.32637930ms
      out_f32x4_shared_row2col: [-1.07092094, 0.99070781, -1.13755226], validate True , time:0.32455182ms
  out_f32x4_shared_bcf_col2row: [-1.07092094, 0.99070781, -1.13755226], validate True , time:0.30707669ms
  out_f32x4_shared_bcf_row2col: [-1.07092094, 0.99070781, -1.13755226], validate True , time:0.31853962ms
                    out_f32_th: [-1.07092094, 0.99070781, -1.13755226], validate True , time:0.91187215ms
------------------------------------------------------------------------------------------------------------------------
```
