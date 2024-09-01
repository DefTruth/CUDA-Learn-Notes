This example shows how to do mixed types GEMMs in CUTLASS.

## High level overview 
This example shows how to perform GEMMs on Hopper when A and B have different types. This implementation always passes the type with fewer bits through the register file and upcasts to the type with the higher bit count.

When relying on `KernelScheduleAuto`, the main loop supporting different A and B types will be selected whenever the bit count of A is not equal to the bit count of B. Users can manually select the mixed type main loop and explicitly choose the scheduling policy by specifying one of the following schedules to the `CollectiveBuilder`:  `KernelTmaWarpSpecializedMixedInput`, `KernelTmaWarpSpecializedPingpongMixedInput` or `KernelTmaWarpSpecializedCooperativeMixedInput`.

This first version only supports mixed type GEMMs using TMA.

## Performance

While the example offers a harness for straightforward benchmarking, this initial implementation isn't optimized for performance in the majority of scenarios. We expect this implementation to be performant for `{fp16, bf16} x {int8, int4}` and `{fp8} x {int4}` for problems that are compute bound. Additionally, we expect good performance for `fp16, bf16` or `fp32` scales and zero-points. For best performance, it is ideal to have the scales and zero-points be the same type.

We are currently optimizing the following cases:
1. Memory bound cases for all types

## Limitations

* The type that needs to be converted must go through the register file. This means that the collective will swap and transpose whenever the type with fewer bits is the B operand. The user must be aware of when these swaps happen. Note that TMA epilogues currently do not support *implicit* swap + transpose, so non-tma epilogues must be used in this case. We plan to relax this limitation in a future release.

* The layout of the narrow type must be K-major. This means the following:
  * Narrow type is the A operand: Must be Row-Major
  * Narrow type is the B operand: Must be Column-Major

* For 8-bit x 4-bit or 2-bit, both inputs must be K-major.

* TMA requires an alignment of 128 bits. As a result, for a type with `B` bits, `B x TILE_K` must be a multiple of 128 bits.

* The type of the scale and zero-point type must be two bytes or more.

* The group size must be equal to gemm-k size (indicating a broadcast), or it must be a multiple of the threadblock-k size.

## Upcoming features

* Optimizations for memory bound cases.

* Optimizations for scale and zero-point loading when the group size is not equal to the threadblock-k size.
