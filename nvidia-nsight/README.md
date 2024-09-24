# NVIDIA Nsight System

## Docs
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [How to see PTX/CU/source code?](https://forums.developer.nvidia.com/t/how-to-see-ptx-cu-source-code/220043)

## Case

- compile `relu.cu` with debug info

```bash
nvcc -arch=sm_89 -o relu.bin --generate-line-info -g relu.cu
# nvcc -arch=sm_89 -o elementwise.bin --generate-line-info -g elementwise.cu
```

- use `nsys` cli to export timeline profile

```bash
nsys profile --stats=true -t cuda,osrt,nvtx -o relu.prof -f true relu.bin
# nsys profile --stats=true -t cuda,osrt,nvtx -o elementwise.prof -f true elementwise.bin
```

- use `ncu` cli to export kernel profile (include SASS/PTX)

```bash
ncu -o relu.prof -f relu.bin
# ncu -o elementwise.prof -f elementwise.bin
```

- run bin

```bash
./relu.bin
S=4096, K=4096, R=10
naive  relu: 0.058982 ms
f16x2  relu: 0.023962 ms
unpack relu: 0.037683 ms # f16x8
pack   relu: 0.015872 ms # f16x8_pack
```

## PTX/SASS Source  

### relu_f16_kernel

- PTX

```C++
mov.u32 	%r3, %ctaid.x;
mov.u32 	%r4, %ntid.x;
mov.u32 	%r5, %tid.x;
mad.lo.s32 	%r1, %r3, %r4, %r5;
setp.ge.s32 	%p1, %r1, %r2;
@%p1 bra 	$L__BB2_2;
cvta.to.global.u64 	%rd3, %rd1;
mov.f32 	%f1, 0f00000000;
{cvt.rn.f16.f32 %rs1, %f1;}
mul.wide.s32 	%rd4, %r1, 2;
add.s64 	%rd5, %rd3, %rd4;
ld.global.u16 	%rs4, [%rd5];
{max.f16 %rs2,%rs1,%rs4;}
cvta.to.global.u64 	%rd6, %rd2;
add.s64 	%rd7, %rd6, %rd4;
st.global.u16 	[%rd7], %rs2;
```

- SASS

```C++
      MOV R1, c[0x0][0x28] 
      S2R R4, SR_CTAID.X 
      S2R R3, SR_TID.X 
      IMAD R4, R4, c[0x0][0x0], R3 
      ISETP.GE.AND P0, PT, R4, c[0x0][0x170], PT 
@P0   EXIT 
      MOV R5, 0x2 
      ULDC.64 UR4, c[0x0][0x118] 
      IMAD.WIDE R2, R4, R5, c[0x0][0x160] 
      LDG.E.U16 R2, [R2.64] 
      IMAD.WIDE R4, R4, R5, c[0x0][0x168] 
      HMNMX2 R0, R2.H0_H0, RZ.H0_H0, !PT 
      STG.E.U16 [R4.64], R0 
```
### relu_f16x8_kernel (un-pack)

- PTX 

```C++
cvta.to.global.u64 	%rd4, %rd3;
mov.u32 	%r2, %ntid.x;
mov.u32 	%r3, %ctaid.x;
mov.u32 	%r4, %tid.x;
mad.lo.s32 	%r5, %r3, %r2, %r4;
shl.b32 	%r6, %r5, 3;
mul.wide.s32 	%rd5, %r6, 2;
add.s64 	%rd6, %rd4, %rd5;
// ... 
ld.global.v2.u16 	{%rs41, %rs42}, [%rd6];   // 非合并读
ld.global.v2.u16 	{%rs43, %rs44}, [%rd6+4];
ld.global.v2.u16 	{%rs45, %rs46}, [%rd6+8];
ld.global.v2.u16 	{%rs47, %rs48}, [%rd6+12];
// ...
st.global.v2.u16 	[%rd9], {%rs10, %rs14};  // 非合并写
st.global.v2.u16 	[%rd1], {%rs18, %rs22};
st.global.v2.u16 	[%rd1+4], {%rs26, %rs30};
st.global.v2.u16 	[%rd1+8], {%rs34, %rs38};
```

- SASS

```C++
      IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] 
      S2R R0, SR_CTAID.X 
      IMAD.MOV.U32 R13, RZ, RZ, 0x2 
      ULDC.64 UR4, c[0x0][0x118] 
      S2R R3, SR_TID.X 
      IMAD R0, R0, c[0x0][0x0], R3 
      SHF.L.U32 R0, R0, 0x3, RZ 
      IMAD.WIDE R2, R0, R13, c[0x0][0x160] 
      LDG.E R6, [R2.64]  // 非合并读
      LDG.E R7, [R2.64+0x4] 
      LDG.E R10, [R2.64+0x8] 
      LDG.E R11, [R2.64+0xc] 
      ISETP.GE.AND P0, PT, R0, c[0x0][0x170], PT 
      IADD3 R8, R0, 0x2, RZ 
      IADD3 R4, R0, 0x4, RZ 
      ISETP.GE.AND P1, PT, R8, c[0x0][0x170], PT 
      IADD3 R9, R0, 0x6, RZ 
      IMAD.WIDE R2, R8, R13, c[0x0][0x168] 
      ISETP.GE.AND P2, PT, R4, c[0x0][0x170], PT 
      ISETP.GE.AND P3, PT, R9, c[0x0][0x170], PT 
@!P0  IMAD.WIDE R4, R0, R13, c[0x0][0x168] 
      HMNMX2 R6, R6, RZ.H0_H0, !PT 
      HMNMX2 R0, R7, RZ.H0_H0, !PT 
@!P0  STG.E [R4.64], R6  // 非合并写
      HMNMX2 R7, R10, RZ.H0_H0, !PT 
@!P2  STG.E [R2.64+0x4], R7 
@!P1  IMAD.MOV.U32 R5, RZ, RZ, R0 
      HMNMX2 R0, R11, RZ.H0_H0, !PT 
@!P1  STG.E [R2.64], R5 
@P3   EXIT 
      STG.E [R2.64+0x8], R0 
```

### relu_f16x8_pack_kernel (pack)

- PTX  

```C++
mov.u32 	%r18, %ntid.x;
mov.u32 	%r19, %ctaid.x;
mov.u32 	%r20, %tid.x;
// ...
ld.global.v4.u32 	{%r23, %r24, %r25, %r26}, [%rd6]; // 读合并
{  cvt.rn.f16.f32 %rs2, %f2;}
{  cvt.rn.f16.f32 %rs1, %f2;}
mov.b32 	%r16, {%rs1, %rs2};
{max.f16x2 %r5,%r23,%r16;}
{max.f16x2 %r8,%r24,%r16;}
{max.f16x2 %r11,%r25,%r16;}
{max.f16x2 %r14,%r26,%r16;}
// ...
st.global.v4.u32 	[%rd9], {%r5, %r8, %r11, %r14}; // 写合并
```

- SASS  

```C++
      IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] 
      S2R R0, SR_CTAID.X 
      ULDC.64 UR4, c[0x0][0x118] 
      S2R R3, SR_TID.X 
      IMAD R0, R0, c[0x0][0x0], R3 
      IMAD.MOV.U32 R3, RZ, RZ, 0x2 
      IMAD.SHL.U32 R0, R0, 0x8, RZ 
      IMAD.WIDE R2, R0, R3, c[0x0][0x160] 
      LDG.E.128 R4, [R2.64] // 读合并
      IADD3 R8, R0, 0x7, RZ 
      ISETP.GE.AND P0, PT, R8, c[0x0][0x170], PT 
@P0   EXIT 
      SHF.R.S32.HI R3, RZ, 0x1f, R0 
      LEA R2, P0, R0, c[0x0][0x168], 0x1 
      HMNMX2 R4, R4, RZ.H0_H0, !PT 
      HMNMX2 R5, R5, RZ.H0_H0, !PT 
      LEA.HI.X R3, R0, c[0x0][0x16c], R3, 0x1, P0 
      HMNMX2 R6, R6, RZ.H0_H0, !PT 
      HMNMX2 R7, R7, RZ.H0_H0, !PT 
      STG.E.128 [R2.64], R4 // 写合并
```
