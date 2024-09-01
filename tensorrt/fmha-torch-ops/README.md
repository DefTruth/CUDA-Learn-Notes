# FMHA Torch Ops

- commands

```bash
# export onnx
python3 export_fmha.py
# build engine
trtexec --onnx=fmha.onnx --saveEngine=fmha.fp16.engine --fp16
# nsys profile
nsys profile --stats=true -t cuda,osrt,nvtx -o fmha.onnx --force-overwrite true trtexec --loadEngine=fmha.fp16.engine
```

- logs

hit `_gemm_mha_v2_0x7d506f90cde376753e5ca2590e8e79f6` kernel
```bash
Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                                                  Name
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     63.7      315,204,989      1,589  198,366.9  198,368.0   198,208   199,745         77.2  _gemm_mha_v2_0x7d506f90cde376753e5ca2590e8e79f6
     23.6      116,811,147      4,767   24,504.1   29,217.0     9,920    37,568     10,516.0  void genericReformat::copyVectorizedKernel<double, float, __half, (bool)1, (bool)0, (int)1>(unsigne…
      5.7       28,183,704      3,178    8,868.4    9,120.0     4,575    12,384      1,563.9  __myl_TraRes_0x78d7a9345f6dcc42cd792b32565e643a
      3.1       15,570,993      1,589    9,799.2    9,792.0     9,759    10,208         19.9  void genericReformat::copyVectorizedKernel<double, __half, float, (bool)1, (bool)0, (int)1>(unsigne…
      2.4       11,950,588      1,589    7,520.8    7,264.0     4,928    11,936        955.8  __myl_TraRes_0x197ff798c934e7e07cf4292e3634b96c
      1.5        7,408,653      1,589    4,662.5    4,672.0     4,512     4,864         45.8  __myl_ResTra_0x804e2fd0410c92cd55b88f92db33aa74
```