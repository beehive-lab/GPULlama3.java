# `qwen35` — where the remaining gap to llama.cpp is

Written against the [matched baseline](qwen35-llama-bench-baseline.md) and two Nsight Systems
profiles: one of `pp1024 b32` and one of `tg256 b32`, both on the CUDA backend, RTX 5090 Laptop.

## 1. The gap, stated as a multiplier

Best configuration on each side, medians of five:

| case | llama.cpp | GPULlama | ratio | slower by |
| --- | ---: | ---: | ---: | ---: |
| pp381 | 1511.18 | 60.54 | 0.040x | 25.0x |
| pp1024 | 1509.11 | 58.80 | 0.039x | 25.7x |
| tg128 | 42.71 | 13.89 | 0.325x | 3.1x |

At matched width the prompt gap is much smaller and tells you why: 0.239x at a width of 8 (4.2x),
0.126x at 16, 0.085x at 32. Width 8 is the last width at which llama.cpp is *also* running a
matrix-vector kernel. Above it, it runs MMQ.

## 2. Efficiency against the machine

The device is roughly 896 GB/s and 25-30 TFLOP/s FP32.

| | GPULlama | llama.cpp |
| --- | ---: | ---: |
| decode, effective weight bandwidth | 207 GB/s (23%) | 638 GB/s (71%) |
| prefill, effective arithmetic rate | 3.3 TFLOP/s (12% of FP32) | 82.5 TFLOP/s-equivalent |

llama.cpp's prefill figure is not FP32. It is int8 MMA, which is why it exceeds the FP32 roof.

**The single most useful number in this document** comes from taking the largest prefill kernel
apart. `fusedFFNGateUpSiLUTiledBatchQ4_0` does 11.4 GFLOP per invocation in 3.22 ms:

- 3.55 TFLOP/s — 12% of the FP32 roof
- 125 GB/s of weights — 14% of the DRAM roof
- ~1.8 TB/s of activations — well inside the L2 roof

It is near **none** of the three. That is what an instruction-bound kernel looks like, and the
instruction mix says why: per element a lane runs about two weight decodes (two byte loads, an
fp16 convert, a shift and a mask each), eight activation loads and sixteen scalar FMAs — on the
order of **2.5 instructions per useful multiply-add**. One int8 MMA instruction retires 512 MACs.

This is also why quantizing the activations bought 1.8% (§6 of the profile record): bytes were
never the binding constraint.

## 3. Prefill — the three that matter

Shares from the `pp1024 b32` profile.

1. **No integer or tensor-core matmul path.** 38.6% (fused gate/up) plus 35.7% (the other Q4_0
   projections) is scalar FP32 with dequantization inside the loop. Structural, and the largest
   single term in the 25x.
2. **The weight decode runs once per element.** Lanes stride the row by the workgroup size, so a
   lane's consecutive elements land in different 32-element blocks and the fp16 block scale is
   unpacked per element rather than once per block.
3. **The fused gate/up is 38.6% by itself**, and its 8x2 tile has been swept — 4 output rows and a
   16-row tile both measured worse. Further gain has to come from 1 and 2.

What to do, cheapest first:

- **Amortize the block scale.** Re-map so a lane covers two or more elements of the *same* block —
  sixteen lanes per block rather than thirty-two — halving the fp16 unpacks while keeping the
  activation reads contiguous across the warp. Low risk, directly measurable, and the same trick
  that made the Q5_K tile pay.
- **Stage decoded weight tiles in shared memory**, the analogue of llama.cpp's `mmq-load-tiles`:
  decode a tile once per workgroup and reuse it across the row tile, turning a per-element decode
  into a per-tile one.
- **Use the tensor cores. TornadoVM already exposes them and this repository already uses them —
  just not for this family.** `KernelContext` carries `mma`/`mmaBF16` for FP16, and
  `mmaInt8(byte[], byte[], int[], MMAShape)` with `MMAShape.M16N8K32` for int8, alongside
  `mmaLoadAInt8`/`mmaLoadBInt8`, `mmaFragmentInt`, `mmaStoreInt` and `swizzleLoadInt8` for the
  shared-memory staging. `TransformerBatchPrefillKernels.gemmMMA`, `gemmMMAQKV` and
  `gemmMMAGateUp` are working FP16 tensor-core GEMMs in this tree, gated by
  `TensorCoreSupport.isTensorCoreCapableBackend()` and used by the FP16 batch-prefill paths for
  Llama and Qwen3. **`mmaInt8` is used nowhere.**

  So there are two routes for `qwen35`, and neither is an upstream request:

  1. *Reuse what exists.* Dequantize a Q4_0 weight tile into an FP16 shared-memory tile once per
     workgroup and hand it to the existing `gemmMMA`. The decode is then per tile rather than per
     element, and the multiply runs on tensor cores. This is the shorter path and it reuses a
     proven kernel.
  2. *Match MMQ.* Quantize the activations to int8 (the quantizer written for the rejected
     experiment does exactly this), unpack Q4_0 nibbles into int8 fragments, accumulate in int32
     through `mmaInt8` at `M16N8K32`, and apply the two block scales at store. This is what
     llama.cpp does above `MMVQ_MAX_BATCH_SIZE`, and it is why its curve keeps climbing to a
     microbatch of 512.

## 4. Decode — the three that matter

Shares from the `tg256 b32` profile. 627,077 launches over 513 tokens is about 1,222 per token;
`cuLaunchKernel` is 4.6% of kernel time, against 0.3% in prefill.

1. **Attention launches one workgroup per head — twenty-four of them.**
   `createAttentionWorker(numberOfHeads, headDim)` gives 24 workgroups on a device with about 82
   SMs, for 15.7% of decode and 680 microseconds per layer per token. The blocker is already
   recorded in the kernel's own comment: the split-KV (flash-decoding) kernel fixes its local
   arrays at 128 floats per head and this family's head is 256 wide, so it cannot be used.
   **Sizing those arrays from a parameter** spreads the same work over roughly eight times the
   workgroups, and is the largest identified decode win.
2. **The Q6_K vocabulary projection is 10.5%** — 7.3 ms per token, 1.04 GB read, 143 GB/s. A sixth
   of peak for a single matrix-vector. The fix is the one that already worked for Q5_K in prefill:
   a warp per sub-block, a lane per element, so the K-quant header is shared across the warp and
   the reads coalesce.
3. **Every matvec is one workgroup per output row with a 128-lane tree reduction** — low
   instruction-level parallelism, seven barriers per output row. Carrying several output rows per
   workgroup is worth 26% in prefill and has not been applied to the single-token kernels. And
   **CUDA graphs should be on by default for decode**: measured +4.7%, with no prefill cost.

## 5. What this adds up to

Decode items 1 and 2 are ordinary shape defects, plausibly 15-20% together, and item 3 is a known
transplant from the prefill work. None of them is structural.

Prefill item 1 is the large one, and it is **not** a TornadoVM limitation. The runtime exposes both
FP16 and int8 tensor-core MMA, this repository already ships FP16 MMA GEMMs and uses them for the
FP16 families, and `qwen35` — the family whose whole point is native quantized storage — reaches
none of it. Closing that is a kernel to write here, not a feature to request upstream.

An earlier revision of this document claimed the opposite, on an assumption rather than on a look
at the API. The claim was wrong and the correction is the most actionable line in the file.

## Packed-integer decode: what won, what lost

Measured on the RTX 5090 Laptop, `Qwen3.8-27B-Q4_0`, weights resident before timing, 100
iterations after 20 untimed.

| projection | shape (n -> d) | kernel family | floating point | packed | |
| --- | --- | --- | ---: | ---: | --- |
| branch projections | 5120 -> 6144 | plain | 0.1133 ms | 0.0415 ms | **2.7x** |
| `ffn_down` | 17408 -> 5120 | residual | 0.1567 ms | 0.1735 ms | 0.90x |
| `attn_output` | 6144 -> 5120 | residual | 0.0796 ms | 0.1207 ms | 0.66x |

Times are the matrix-vector alone; the activation quantization adds 0.026-0.042 ms and is
charged to the packed side in the decisions below. It does **not** scale with the activation's
length across these measurements — 17408 elements cost no more than 6144 — but that comparison
is across separate experiments and is not by itself evidence that a longer quantizer is
intrinsically cheaper.

The two residual candidates were discarded: `ffn_down` at 0.79x and `attn_output` at 0.54x of
the current kernel, both including preparation. Correctness was not the problem — each agreed
with a reference quantizing identically to about 1.2e-7 relative, with zero blocks, mixed signs
and finiteness checked.

**Why the packed form wins at one shape and loses at the other two is not established.** Two
things differ between them at once: the kernel family, plain against residual, and the shape,
whose reduction length and row count are swapped. Neither has been isolated. Worth noting for
anyone who returns to this: the floating-point *residual* kernel at 6144 -> 5120 costs 0.0796 ms
against the floating-point *plain* kernel's 0.1133 ms at 5120 -> 6144, which is comparable work,
so the two floating-point kernels appear to differ in more than the residual add. That
observation is a starting point, not a conclusion.
