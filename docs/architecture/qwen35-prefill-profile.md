# `qwen35` batched prefill — where the time goes

Nsight Systems over the [llama-bench baseline](qwen35-llama-bench-baseline.md)'s `pp381 b32`
case, three passes (one warm-up plus two measured), CUDA backend, RTX 5090 Laptop.

```bash
nsys profile --trace=cuda --output prefill-b32 \
  ./llama-tornado --gpu --cuda --gpu-memory 22GB --model Qwen3.8-27B-Q4_0.gguf --bench \
    --bench-args="-p 381 -b 32 -r 2 -o md --expect qwen35/Q4_0/BATCH_PREFILL_DECODE"
nsys stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum prefill-b32.nsys-rep
```

## 1. It is kernel-bound, and by how much

Kernel time totals 48.74 s across a 54.56 s span: **the GPU is busy 89.3% of the profile**, and
the remainder is model load and JIT at the front. `cuLaunchKernel` costs 156 ms over 46,249
launches — **0.3% of kernel time** — so the ~1,000 launches per chunk this design costs are not
what to fix. Host-to-device copies total 587 ms, nearly all of it the one-time weight upload;
the 47,817 per-step copies have a 384 ns median, which is the batch-info and position holders and
nothing else.

Reconciled against wall time: three prefill passes of 381 tokens at the measured 22.9 t/s is 50 s
of prompt processing, against 48.7 s of kernel time. The kernel summary accounts for the run.

## 2. Attribution

| Kernel | Share | Instances | Avg | Reads its weights |
| --- | ---: | ---: | ---: | --- |
| `matrixVectorBatchWithResidualQ5_K` — `ssm_out` | 25.6% | 1776 | 7.03 ms | **32x per chunk (untiled)** |
| `fusedFFNGateUpSiLUTiledBatchQ4_0` | 24.9% | 2368 | 5.12 ms | 8x (tile of 4) |
| `matrixVectorTiledBatchWithResidualQ4_0` — `ffn_down`, `attn_output` | 18.6% | 2664 | 3.39 ms | 4x (tile of 8) |
| `matrixVectorTiledBatchQ4_0` — `attn_qkv`, `attn_gate`, projections | 17.0% | 5328 | 1.55 ms | 4x (tile of 8) |
| `attentionBatchPaged` | 7.7% | 592 | 6.34 ms | — |
| `matrixVectorBatchWithResidualQ4_1` — `ffn_down`, blocks 0-7 | 3.3% | 296 | 5.43 ms | **32x per chunk (untiled)** |
| `deltaRuleScan` | 1.6% | 1776 | 0.45 ms | — |
| `batchedRmsReduce` | 0.8% | 4736 | 0.08 ms | — |
| `batchedMatVecF32` — `ssm_alpha`, `ssm_beta` | 0.1% | 3552 | 0.02 ms | — |
| `gatedNormPerHeadBatch` | 0.1% | 1776 | 0.03 ms | — |
| `causalConv1dScan` | 0.05% | 1776 | 0.01 ms | — |
| everything else (norms, SiLU, splits, RoPE, KV append) | <0.2% total | | | |

**Quantized projections are 89.4% of prompt processing.** Attention is 7.7%. Normalization and
elementwise work together are under 1%.

## 3. The recurrence is not the bottleneck

The Gated Delta Net scan is **1.6%** and the convolution scan is **0.05%**. The design record's
expectation — that a three-quarters-recurrent stack has a low batching ceiling because the scans
are serial in the chunk — is measurably not what limits this implementation today. At a chunk of
32 the scans cost 0.45 ms against 7.03 ms for the single Q5_K projection in the same layer.

That reorders the optimization plan. Warp-cooperative delta-net execution, register-resident state
shards and a transposed state layout — all of which llama.cpp's `gated_delta_net.cu` does and this
implementation does not — are worth at most 1.6% of prefill, and are only worth doing for their
effect on **decode**, where the same kernel runs once per token.

## 4. What limits the projections

Two different limits, and they need different fixes.

**The untiled kernels re-read their weights per prompt row.** `ssm_out` (Q5_K) and the early
`ffn_down` (Q4_1) run one workgroup per (prompt row, output row), so a chunk of 32 decodes each
weight 32 times. Q5_K is the expensive representation to decode — a K-quant matvec runs at roughly
a third of Q4_0's rate — so paying for that decode 32 times instead of 4 is why one projection in
a recurrent layer costs more than the layer's entire feed-forward.

**The tiled kernels are limited by activation traffic, not weight traffic.** A tiled workgroup
reads one output row of weights (2,560 B at `n`=5120, 8,704 B at `n`=17408) and `ROW_TILE` whole
rows of activations (160 KB and 557 KB respectively). Counting both, `ffn_down` moves about
11.6 GB per layer per chunk and does it in 5.2 ms — roughly 2.3 TB/s, which is L2 rate, not DRAM
rate. The untiled Q4_1 kernel moves 13.3 GB in 5.43 ms: **the same aggregate rate**. Both kernels
are running at the cache's ceiling and differ only in how many bytes they ask for.

So for the Q4_0 kernels the lever is not a bigger row tile — it is giving one workgroup several
**output** rows, so that the activation tile it stages is reused instead of re-read by every
output row's workgroup. That is the shape llama.cpp's MMQ has for a different reason.

## 5. Order of work this implies

1. Row-tile `ssm_out` (Q5_K) — 25.6%, decode-bound, and the tiling pattern already exists.
2. Row-tile the early `ffn_down` (Q4_1) — 3.3%, same change.
3. Share the activation tile across output rows in the Q4_0 tiled kernels — 60.5% combined, and
   the larger structural change.
4. Attention — 7.7%, only after the above.
5. Delta-net cooperative execution — 1.6% of prefill; judge it on decode instead.
6. Launch overhead and elementwise fusion — 0.3% and <1%. Not worth doing.

## 6. Rejected: quantized activations, both ways

llama.cpp does not dequantize inside its matmul. Above `MMVQ_MAX_BATCH_SIZE` it quantizes the
*activations* to `Q8_1` and runs an integer dot product, applying the two block scales once per
block, with Q4_0's `-8` recentring folded into the activation block's stored sum. Since the profile
says activation traffic is what limits our tiled kernels, and `Q8_1` is 1.125 bytes an element
against four, this looked like the right thing to copy. It was measured twice and kept neither
time.

**Per-block `Q8_1`, warp-per-block integer dot — 52.28 -> 41.46 t/s, a 21% regression.** A lane that
owns a 32-element block reads 32 contiguous bytes no other lane in its warp reads, which is a
32-byte transaction where the float kernel gets a 128-byte one; and every lane of the warp re-reads
the block's scales for all eight tiled rows, 64 bytes of duplicated parameter traffic against one
byte of quant. Charging the recentring to lane zero only, so the other 31 lanes stop reading the
block sums, recovered part of it — 42.81 — and no more. The compression is undone by the access
pattern it forces.

**Per-row byte activations, float kernel shape — 52.11 -> 53.06 t/s, +1.8%.** One scale per row
instead of one per block keeps the lane striding, and therefore the coalescing, exactly as it was:
the only change is that the activation read is a byte. That is a genuine 4x cut in the dominant
read and it bought under two percent, which says the activation tile was already being served from
cache rather than from DRAM.

Under two percent does not pay for what it costs: a second kernel pair, a per-row scale buffer, a
quantization pass after every norm, and a real accuracy change — eight-bit activations, per row,
through 64 layers. Neither variant is in the tree. What both measurements establish is that the
remaining limit is not activation *bytes*, so the next thing to try is not a smaller activation.

## 7. Not obtained

Nsight Compute counters are unavailable on this host: `ERR_NVGPUCTRPERM`. Every bandwidth figure
above is bytes-moved over measured kernel time from the Nsight Systems trace and the tensor
shapes, not a hardware counter. Occupancy, stall reasons and L1/L2 hit rates were not measured.

## 6. The wide-tile experiment, and what it did and did not settle

Tried after the profile above put `projectionMMAQ4_0GateUp` at 44.9% of prefill: one workgroup
staging the activation tile once for a 32x128 output tile — four warps, `warpM` picking the 16-row
band and `warpN` the 64-column half, eight accumulators per projection — against the shipped
kernel's one warp per 16x8 tile, which re-stages its 16-row activation tile for every one of the
2176 column tiles of a 17408-wide projection.

It lost. Interleaved at matched temperature (70 C, 1710 MHz), `pp381 b32` was 70.95 t/s against
75.72 and 75.48 for the shipped kernel, and the kernel itself cost 3.26 ms per call against 2.76.
Registers were not the limit: `ptxas` reports 128 registers, 0 bytes spilled, 18432 bytes of shared
memory for the 18 KiB variant.

**What that settles is this implementation, not the principle.** Three things changed together and
the experiment cannot separate them: the tile geometry, the grid — 136 workgroups over ~82 SMs,
where the shipped kernel launches 4352 — and the staging cadence, one MMA step per round rather
than a whole Q4_0 block. Activation reuse may still be worth having at a wider chunk, where the
grid argument reverses. The patch is kept locally rather than committed; it is correct, and it is
slower here.

Getting it correct first cost a long bisection, and the cause was not geometry at all: see finding
4 in `tornadovm-issues`, an unsigned nibble recentring that wrapped to 2^32 - 8 and overflowed
fp16 to infinity. Every simplified probe passed because the wrap only affects the high nibble
below eight.

---

# Re-profiled 2026-09-12 on `51de832d`

The sections above were taken before the MMA projections shipped and are kept for their reasoning;
**their attribution is superseded by this section**. `ssm_out` is no longer an untiled per-row
matvec at 25.6%, and the projections no longer decode weights on the fly in a float kernel.

## Configuration, and which executions are counted

Throughput configuration, stated because two different runs are quoted below: `--gpu --cuda
--gpu-memory 22GB --tensor-cores --fp16-kv-cache`, model `Qwen3.8-27B-Q4_0.gguf`, bench
`-p 381 -n 0 -b 32`. `-b 32` is the **physical prefill chunk**. 381 tokens is 12 chunks: eleven of
32 rows and one of 29.

Two captures, deliberately not mixed:

| capture | graphs | instrumentation | reported | what it is for |
| --- | --- | --- | ---: | --- |
| TornadoVM profiler | off | `--profiler`, per-task `TASK_KERNEL_TIME` | 71.33 t/s | per-task attribution |
| Nsight Systems | **on** | `--trace=cuda --cuda-graph-trace=node` | 80.44 t/s | timeline, transfers, launch API |
| neither | on | none | 75.8-76.2 t/s | the throughput number |

Those three throughputs are three different runs, and **none of them may be subtracted from another
to price the instrumentation**: they differ in more than one thing at a time. Prefill throughput on
this build has been observed between 71 and 81 t/s across configurations; see the handoff's note
that the prefill figure itself is unexplained.

**Isolation.** `-r 2` is one untimed warm-up pass plus two measured passes. The profiler stream
carries 37 chunk executions: one compilation/JIT chunk, then 36 = 3 passes x 12 chunks. Every
number in the table below comes from the **last 24 chunk executions** — the two measured passes
only, 762 tokens, 1,536 `batchLayer` graph executions over 64 trunk layers, warm-up and the JIT
chunk excluded. Totals for that window: **9,511 ms of kernel time**, 10,583 ms of task-graph time,
123 ms of copy-in.

## Attribution, measured passes only

| task | kernel | calls | ms | share | µs/call |
| --- | --- | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | `projectionMMAQ4_0GateUp` | 1536 | 4385.3 | **46.11%** | 2855.0 |
| `ffn_down_proj` | `projectionMMAQ4_0` (blocks 8+), `projectionMMAQ4_1` (blocks 0-7) | 1536 | 1653.3 | 17.38% | 1076.4 |
| `ssm_qkv_proj` | `projectionMMAQ4_0` | 1152 | 770.8 | 8.10% | 669.1 |
| `ssm_out_proj` | `projectionMMAQ5_K` | 1152 | 548.6 | 5.77% | 476.2 |
| `ssm_delta_rule` | `deltaRuleScan` | 1152 | 435.5 | 4.58% | 378.0 |
| `ssm_gate_proj` | `projectionMMAQ4_0` | 1152 | 392.1 | 4.12% | 340.3 |
| `attn_q_proj` (query+gate, 12288 wide) | `projectionMMAQ4_0` | 384 | 260.4 | 2.74% | 678.1 |
| `attention` | `attentionBatchFP16Paged` | 384 | 237.3 | 2.50% | 618.0 |
| `attn_output_proj` | `matrixVectorTiledBatchWithResidualQ4_0` — **not MMA** | 384 | 204.4 | 2.15% | 532.3 |
| `attn_k_proj`, `attn_v_proj` | `projectionMMAQ4_0` | 384 each | 105.9 each | 1.11% each | 275.9 |
| `attn_rms_reduce`, `ffn_rms_reduce` | `batchedRmsReduce` | 1536 each | ~90 each | 0.95% each | 58.7 |
| everything else | norms, SiLU, RoPE, KV append, splits | | <60 | <0.7% | |

**Activation preparation is not a cost here**: `convertToFP16` and `convertNormedToFP16` together are
10.9 ms of 9,671 ms in the Nsight capture, **0.11%**. Neither is data movement: host-to-device copies
total 2.0 s but are dominated by the one-time weight upload (largest single operation 36.6 ms,
median operation 448 ns); per-step copies are the batch-info and position holders.
`cuGraphLaunch` was 36 ms over 1,691 calls, 0.4% of kernel time. In the Nsight capture, summed
kernel time (9.671 s) is close to the wall time of the two passes it covers (~9.5 s at the
80.44 t/s it reported), which is consistent with the GPU being busy nearly continuously **in that
capture** — it is not a measurement of the uninstrumented run.

## Which projections use the tensor cores, and how they are launched

Everything except `attn_output_proj`, which folds a residual and is excluded by `mmaEligible`
(`Qwen35BatchPrefillLayers:109`). Q4_0, Q4_1 and Q5_K each have their own MMA kernel.

Geometry is one `m16n8k16` tile per workgroup of **one warp** (`BM=16`, `BN=8`, `BK=16`,
`LOCAL=32`): the grid is `(m/16) * (n/8)` workgroups, and each walks the whole reduction dimension
in Q4_0 blocks of 32. For `ffn_gate_up` at `m=32, n=17408, k=5120` that is **4,352 workgroups of 32
threads, each iterating 160 blocks**; for `ffn_down` at `m=32, n=5120, k=17408` it is 1,280
workgroups of 544 blocks. Both therefore run **696,320 block iterations** per call.

**The partial chunk costs a full chunk.** `mmaEligible` tests the configured batch size, not the
live chunk, so the 29-row chunk launches the same 32-row grid; `convertToFP16` zeroes the padding
rows (`row < batchInfo.get(1)`) and they are computed, stored and never read. Measured: chunks 11
and 23 — the 29-row ones — cost 402.9 ms and 403.3 ms against 385-406 ms for the full chunks. Three
wasted rows in 381 is ~0.8% of prefill.

## What the numbers say the limit is

Per **block iteration**, which is the unit of work both kernels repeat:

| kernel | panels staged per iteration | block iterations per call | ns per iteration |
| --- | ---: | ---: | ---: |
| `projectionMMAQ4_0GateUp` (`ffn_gate_up`) | 2 | 696,320 | **4.10** |
| `projectionMMAQ4_0` (`ffn_down`, k=17408) | 1 | 696,320 | 1.55 |
| `projectionMMAQ4_0` (`ssm_qkv`, n=10240) | 1 | 409,600 | 1.63 |
| `projectionMMAQ4_0` (`ssm_gate`, n=6144) | 1 | 245,760 | 1.38 |
| `projectionMMAQ4_0` (`attn_q`, n=12288) | 1 | 491,520 | 1.38 |

Single-panel iterations cost 1.38-1.63 ns **regardless of shape, grid size or how much activation
traffic the shape implies**. The fused two-panel iteration costs 4.10 ns — 2.6x a single panel, not
2x. `ffn_gate_up` and `ffn_down` perform the identical number of block iterations and stage the same
16x32 activation tile per iteration, yet differ by 2.65x per call.

**Measured**: the cost tracks panels decoded per iteration, not activation bytes moved.
**Hypothesis, not measured**: the extra ~40% over two single panels comes from the fused kernel's
doubled shared-memory footprint (four half arrays plus two int arrays) and its two MMA accumulator
fragments, either of which can cost occupancy or registers for one warp per workgroup. Nsight
Compute cannot arbitrate this on this host (`ERR_NVGPUCTRPERM`, unchanged), and no counter evidence
exists.

Two things this does **not** say. It does not say activation reuse is worthless — the wide-tile
experiment in §6 changed geometry, grid and staging cadence together and settled only that
implementation. And a share is not a bound on what a change is worth.

## The split gate/up experiment — kept

Acting on the observation above: the fused two-panel task was replaced with **two
`projectionMMAQ4_0` calls at the same M, N and K**, writing the same two buffers, followed by the
same SwiGLU task. No new kernel, no option, no precision or layout change; the scalar fallback for
an ineligible shape is untouched.

Equivalence was checked before anything was measured: at the real 32x17408x5120, full chunk and a
29-row chunk with zeroed padding rows, gate and up came back **bit-identical** to the fused kernel,
557,056 of 557,056 values each, all finite. The whole-model cross-width capture then produced the
same logits as before the change (SHA-256 `e17b0f731220525c`, 15,644,160 values, 63 token ids).

| | fused | split |
| --- | ---: | ---: |
| `pp381 b32`, interleaved, graphs on | 76.21, 75.92, 75.89 | **85.65, 85.56, 92.00** |
| `tg128 b32` | 28.42, 28.33 | 28.41, 28.35 |
| that projection, per layer-chunk, profiler | 2855.0 µs | 985.1 + 1047.6 = **2032.7 µs** |
| measured-window kernel total, profiler | 9511.5 ms | 8184.4 ms |

**+12.8% on prefill medians**, decode unchanged, and the projection's kernel time down 28.8% at the
same shape. The 92.00 t/s reading is real but out of line with the other two split runs; the
conservative pairing (85.56 against 76.21) is +12.3%.

**What this establishes and what it does not.** It establishes that the two-panel form cost more
than two single-panel forms of the same shape on this device, at this geometry, for this model —
and that doing the activation staging twice is cheaper here than whatever the fused body was paying
for. It does **not** identify the cause: registers, shared memory, occupancy, scheduling and
intra-kernel synchronization were not measured, and Nsight Compute is still unavailable on this host
(`ERR_NVGPUCTRPERM`). The panel-count hypothesis in the section above is consistent with the result
and remains unproven.

The fused kernel had no other production caller and is deleted. Its host-parity case is covered by
the single-panel case; the unsigned-nibble guard it hosted
(`everyNibbleDecodesWithTheSignItsScaleGivesIt`, finding 4 in `tornadovm-issues`) now runs against
two `projectionMMAQ4_0` tasks and asserts the same values. The fused-against-split equivalence
harness is kept out of the tree at `~/qwen35-gateup-split-equivalence.java`, since it needs a kernel
that no longer exists.

## Ranking after the split, from the same capture

No recapture: this is the post-split profiler run already taken for the A/B, attributed the same way
— the **last 24 chunk executions only**, two measured passes, 762 tokens, 1,536 `batchLayer` graph
executions, compilation chunk and warm-up pass excluded. Window totals: **8,184.4 ms kernel**,
8,903.0 ms task-graph, 131.8 ms copy-in, 718.6 ms of graph time that is not kernel time (down from
9,511.5 / 10,582.7 / 123.3 / 1,071.2 before the split).

Every MMA projection is one warp per `m16n8k16` tile: **local size 32**, grid `(M/16) * (N/8)`
workgroups.

| task | kernel | dtype | M x N x K | workgroups | calls | ms | share | µs/call |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ffn_down_proj` | `projectionMMAQ4_0` / `…Q4_1` | Q4_0, Q4_1 on blocks 0-7 | 32 x 5120 x 17408 | 1280 | 1536 | 1637.5 | 20.01% | 1066.1 |
| `ffn_up_proj` | `projectionMMAQ4_0` | Q4_0 | 32 x 17408 x 5120 | 4352 | 1536 | 1609.1 | **19.66%** | 1047.6 |
| `ffn_gate_proj` | `projectionMMAQ4_0` | Q4_0 | 32 x 17408 x 5120 | 4352 | 1536 | 1513.1 | **18.49%** | 985.1 |
| `ssm_qkv_proj` | `projectionMMAQ4_0` | Q4_0 | 32 x 10240 x 5120 | 2560 | 1152 | 759.8 | 9.28% | 659.6 |
| `ssm_out_proj` | `projectionMMAQ5_K` | Q5_K | 32 x 5120 x 6144 | 1280 | 1152 | 539.4 | 6.59% | 468.2 |
| `ssm_delta_rule` | `deltaRuleScan` | — | — | — | 1152 | 428.7 | 5.24% | 372.1 |
| `ssm_gate_proj` | `projectionMMAQ4_0` | Q4_0 | 32 x 6144 x 5120 | 1536 | 1152 | 387.2 | 4.73% | 336.1 |
| `attn_q_proj` (query+gate) | `projectionMMAQ4_0` | Q4_0 | 32 x 12288 x 5120 | 3072 | 384 | 256.8 | 3.14% | 668.8 |
| `attention` | `attentionBatchFP16Paged` | — | — | — | 384 | 232.7 | 2.84% | 606.1 |
| `attn_output_proj` | `matrixVectorTiledBatchWithResidualQ4_0` — not MMA, local 128 | Q4_0 | 32 x 5120 x 6144 | — | 384 | 200.7 | 2.45% | 522.7 |
| `attn_k_proj`, `attn_v_proj` | `projectionMMAQ4_0` | Q4_0 | 32 x 1024 x 5120 | 256 each | 384 each | 104.2 each | 1.27% each | 271 |

Gate and up separately are **18.49% and 19.66%, 38.15% together** — the same two matrices that were
46.11% as one fused task. `projectionMMAQ4_0` alone is now **77.86%** of prefill kernel time across
six task names; all MMA projections together are 84.45%.

## Inspection of `projectionMMAQ4_0`, and one candidate

Generated CUDA obtained through `TornadoExecutionPlan.withPrintKernel()` at a small shape; the
kernel body does not depend on the shape.

**The block scale is decoded in software, per lane, per block.** Source
(`Qwen35MMAKernels.projectionMMAQ4_0`):

```java
int base = ((blockCol + stageCol) * blocksPerRow + blockIndex) * BLOCK_BYTES;
float scale = halfFromBytes(w, base);
```

`halfFromBytes` reconstructs the half by hand — the file records why it was written that way for a
different backend — and the generated CUDA is a ten-branch expansion per call, beginning:

```c
ui_282  =  ui_281 & 1023;          // mantissa
f_283   =  (float) ui_282;
ui_284  =  ui_281 >> 10;
b_285   =  (ui_284 & 31) == 0;     // subnormal?
if(b_285) { f_286 = f_283 * 5.9604645E-8F; ... }
else { /* exponent rebuilt by four conditional multiplies */ }
```

The same dump contains `matrixVectorGenericQ4_0DP4A`, which reads its block scale through
`w.getHalfFloat(base).getFloat32()` instead, and there the conversion is one instruction:

```c
f_146  =  __half2float(half_19);
```

Counted over the two kernels in that one dump: `projectionMMAQ4_0` has **zero** `__half2float` and
one software expansion; the DP4A kernel has one `__half2float` and no expansion.

Two further observations about the same site, recorded but **not** proposed as the experiment:
every lane decodes a scale although only eight distinct column scales exist per block
(`stageCol = lane >> 2`, so four lanes duplicate each), and the loop-invariant nibble-half and
destination-tile decisions are re-tested per element in the emitted C. Deduplicating across lanes is
a restructure, not a small change; and the per-element tests are in pre-`ptxas` source, which may
well hoist them.

**None of this is a measured bottleneck.** Instruction counts in generated C say what work exists,
not what the hardware stalls on, and Nsight Compute remains unavailable here (`ERR_NVGPUCTRPERM`).

## The block-scale conversion — kept

The candidate above was taken: in `projectionMMAQ4_0` only, `halfFromBytes(w, base)` became
`w.getHalfFloat(base).getFloat32()`. Geometry, staging order, synchronization, the A-tile path, the
Q4_1 and Q5_K MMA kernels and `halfFromBytes` itself are untouched.

The offset and interpretation are the same: `getHalfFloat` takes a **byte** index, requires two-byte
alignment, and reads a native-endian short — Q4_0's block stride is 18 bytes and the scale sits at
offset 0, so every scale is two-byte aligned, and `halfFromBytes` assembles the same little-endian
pair. Emission changed as intended: the kernel now contains one `__half2float` and none of the
ten-branch software expansion, where before it had the expansion and no hardware conversion.

Checks before timing:

- **Every finite half encoding** read from a Q4_0 block-scale position agrees bit for bit with
  `Float.float16ToFloat` — 63,488 encodings, 31,744 of them negative, 2,046 subnormal, both zeros.
  That case lives in `HalfFloatConversionAccelTest`.
- **The whole projection output** at 32 x 17408 x 5120, captured from both builds and compared byte
  for byte: **identical**, on a full chunk and on a 29-row chunk with zeroed padding rows, 1,114,112
  values, all finite, weights chosen with small scales of both signs so nothing overflows.

Interleaved, warmed, graphs on, tensor cores on, FP16 KV, two repetitions each:

| | before | after |
| --- | ---: | ---: |
| `pp381 b32` | 85.54, 85.31, 85.40 | **88.46, 88.39, 88.27** |
| medians | 85.40 | **88.39 (+3.50%)** |

The ranges do not overlap. **The speedup is not inferred from the instruction count**: the counts
say the work exists, the A/B says what removing it was worth, and nothing here identifies what the
hardware was doing. 28 focused MMA, topology, parity and lifecycle tests pass with the same parity
numbers as before, which is what bit-identical outputs require.

## What survives final compilation

The generated CUDA is not what runs. To get close to what does, the emitted source for
`projectionMMAQ4_0` was **recompiled with `nvcc`** — not read out of the runtime NVRTC cubin, which
this inspection never touched — using the same architecture TornadoVM compiles for — `CUDACompiler.build`
passes `--gpu-architecture=sm_120` and nothing else on this machine (cc 12.0, cubin path, no PTX
JIT) — and disassembled:

```bash
nvcc -arch=sm_120 -cubin --resource-usage -o mma.cubin mma.cu   # ptxas resource report
cuobjdump -sass mma.cubin                                        # final SASS
```

**This is an NVCC reconstruction, supporting evidence rather than an inspection of the runtime
image**: the same source and the same `ptxas`, but `nvcc`'s front end instead of NVRTC's, and a
cubin this process compiled rather than the one the JIT loaded. No hardware counters are involved
and none are available (`ERR_NVGPUCTRPERM`).

**Resources.** 61 registers, **0 bytes spill stores, 0 bytes spill loads**, 1,536 bytes of shared
memory, 1 barrier. Shared matches the four declared tiles exactly (512 + 512 + 256 + 256). *No
occupancy claim follows from this*: what limits residency was not measured.

**The repeated decisions are half optimized away.** Only 2 `BRA` and 3 `ISETP` survive in the whole
kernel — the loop back-edge and its test. The per-element comparisons visible in the generated C are
gone. What remains is **predication, not branching**, and in one place it costs a duplicated store:

- the nibble half is one predicated instruction per element, `@P1 SHF.R.U32.HI R22, RZ, 0x4, R58`
  against the unpredicated `LOP3.LUT R22, R58, 0xf, …` for the low nibble;
- the destination tile is a **pair** of predicated shared stores per element,
  `@!P2 STS.U16 [R13+UR8]` / `@P2 STS.U16 [R13+UR7]`. These are **static instructions, not memory
  writes**: each thread executes exactly one of the pair and the other is predicated off, so the
  tile is written once per value and the cost is issue slots, not store traffic. Sixteen `STS.U16`
  appear in the listing for 8 values, and 12 `STS` for the A tile's 8 ints. `ptxas` predicates both
  rather than selecting one address, because the two tiles are distinct shared allocations.

**Loads are individual, not combined.** Per lane per block iteration: **8 × `LD.E.U8`** at
consecutive offsets (`[R20.64]`, `+0x3`, `+0x4`, `+0x5`, `+0x6`, …) for the eight contiguous packed
weight bytes; **16 × `LD.E.U16`** for the A tile, two per adjacent half pair rather than one 32-bit
load; and **1 × `LD.E.U16`** for the block scale. No `LDG.64` or `LDG.128` anywhere. (Coalescing
across the warp is a different question and is not visible here.)

**The scale is loaded and converted per lane.** One `LD.E.U16` and one `HADD2.F32 R22, -RZ,
R36.H0_H0` per thread per iteration. `stageCol = lane >> 2`, so four lanes carry the same `base` and
do that work on the same bytes four times; no compiler can remove that, since it is duplication
across threads rather than within one.

**Retrospective on `5e68ffca`.** Compiling the pre-change source the same way settles that the
hand-written half decode was not being optimized away: 64 registers, 472 instructions, **9 `BRA`, 9
`ISETP`, 5 `FSEL`** against 61 registers, 368 instructions, **2 `BRA`, 3 `ISETP`, 0 `FSEL`** after.
The ten-branch expansion reached the hardware, and removing it removed about a fifth of the kernel's
instructions. That is a *retrospective check of the premise*, not a re-derivation of the speedup —
the +3.50% came from the whole-model A/B.

## Correction: the offset-aware tile API exists

An earlier note in this file said merging the two B tiles had no small implementation because
`swizzleStoreFp16Stride32` and `mmaLoadBSwizzled` take an array and not an array plus offset. **That
was wrong.** The installed SDK (`tornado-api-6.0.1-jdk21-dev.jar`, verified with `javap`) carries
byte-offset overloads and a matching offset-aware swizzled store:

```java
HalfFloat[] mmaLoadA(int[] aTile, int wmmaK, int byteOffset);
HalfFloat[] mmaLoadB(int[] bTile, int wmmaK, int byteOffset);
HalfFloat[] mmaLoadBSwizzled(HalfFloat[] bTile, int wmmaK, int byteOffset);
void        mmaStoreBSwizzled(HalfFloat[] arr, int row, int col, int stride,
                              HalfFloat value, int byteOffset);
```

**Where the offset is applied, verified in the emitters rather than assumed** (`CUDALIRStmt`): the
store computes `__lin = row * stride + col`, `__bo = __lin << 1`, applies the swizzle
`__bo ^= (((__bo >> 7) & 7u) << 4)`, and only then adds `__bo += byteOffset`. The swizzled load
computes its per-lane `__bo`, applies the same XOR, and then adds `byteOffset`. **The offset is
applied after the swizzle on both sides**, so a sub-tile at a multiple of the panel size holds
exactly the layout a separate allocation would, and with `byteOffset = 0` the emitted address
arithmetic is what `swizzleStoreFp16Stride32` emits today.

`tornado-examples`' `MatrixMultiplicationMMA` uses this shape: one `bTile`, stores with
`mmaStoreBSwizzled(bTile, k_row, j, 8, val, subTileId * B_SUBTILE_BYTES)` and loads with
`mmaLoadBSwizzled(bTile, BK, (bBase + i) * B_SUBTILE_BYTES)`. This repository's own
`Qwen35MMAKernels` already declares `B_SUBTILE_BYTES = 256` — eight columns, `BK` deep, two bytes —
and uses it nowhere, a leftover from the wide-tile work.

So one shared allocation can represent today's two B panels: `HalfFloat[2 * PANEL * BK]`, low half at
`byteOffset = 0` and high half at `byteOffset = B_SUBTILE_BYTES`, with the per-lane choice becoming
a value rather than a branch. The A tiles need no API at all — they are plain `int[]` indexing.

## Wider reads of the native weights: not available, and exactly why

`ByteArray`'s device-side accessors are `get(int index)` (one byte) and
`getHalfFloat(int byteIndex)` (two bytes, two-byte aligned). There is no four- or eight-byte read
and no typed view; `slice`, `getSegment` and the constructors are host-side. Nothing in the API
widens a global load of Q4_0 weight bytes.

The one wider path that exists is `asyncCopyToLocal(int[] dstTile, int dstIndex, ByteArray src, int
srcIndex)`, which lowers to `cp.async.ca.shared.global [dst], [src], 4` — a **four-byte** global to
shared copy whose source address is `srcArray + 16 + srcIndex * elementSize`. PTX requires that
address to be naturally aligned to the four-byte access, and **Q4_0's 18-byte block stride breaks
it**: the quantized bytes of block `B` start at `16 + 18B + 2`, and `18B + 2 ≡ 2B + 2 (mod 4)`, which
is 4-byte aligned only for odd `B`. Half the blocks would be misaligned, so this route is closed for
the weights without changing their layout — which would mean materializing or duplicating them.

What is precisely missing is a **four-byte read at two-byte alignment** on `ByteArray` (or a
`cp.async` variant with the same relaxation). Neither exists, and inventing one is a TornadoVM
change, not a kernel change.

Adjacent, and genuinely available: the same `cp.async` applies to the **activation** tile, whose
source is a `HalfFloatArray` at an even element index, so its byte address is always four-byte
aligned; `asyncCopyToLocal(int[], int, HalfFloatArray, int)` packs `src[i] | src[i+1] << 16`, which
is exactly what the A-tile staging assembles by hand today. It is not as small a change as the tile
merge — it brings `asyncCopyCommit`/`asyncCopyWaitGroup` and their ordering contract with it.

## The merged B tile — kept

Acting on the corrected capability note: `projectionMMAQ4_0`'s two B panels are now **one shared
allocation** of `2 * PANEL * BK` halves — 512 bytes — with the first panel at byte offset 0 and the
second at `B_SUBTILE_BYTES` (256), staged through `mmaStoreBSwizzled(..., stageOffset)` and read
through `mmaLoadBSwizzled(bTile, BK, 0)` and `mmaLoadBSwizzled(bTile, BK, B_SUBTILE_BYTES)`. A
tiles, geometry, grid, reduction order, activation staging, barriers, weight layout and precision
are untouched, and no other kernel changed.

Capacity and non-overlap: a panel's in-panel address is `((row * PANEL) + col) * 2` with
`row < BK`, `col < PANEL`, so at most 254 before the swizzle; the swizzle is an XOR of bit 4 driven
by bit 7, which permutes within the same 256-byte window; the offset is added afterwards. Each panel
therefore stays inside its own 256 bytes, and both offsets are multiples of 16 for `ldmatrix`.

**Bit-identical**, which is the check that matters for addressing: the whole projection output at
32 x 17408 x 5120, captured from both builds into NaN-poisoned buffers and compared byte for byte —
identical on a full chunk and on a 29-row padded chunk, 1,114,112 values, all finite. The
whole-model cross-width capture returns the same logits as before (SHA-256 `e17b0f731220525c`).

**The generated code did what was intended; the compiled code is not uniformly smaller.** The
emitted CUDA now declares one `__shared__ half half_5[256]` instead of two of 128, and the staging
store is a single `((half *) half_5)[__bo >> 1] = …` with `__bo += <offset>` after the swizzle, with
no `if`/`else`. In the **NVCC reconstruction** (same source and `ptxas`, not the runtime NVRTC
image):

| | accepted | merged |
| --- | ---: | ---: |
| predicated `STS` | 24 | **8** (the 8 that remain are the A tile's, untouched) |
| `STS.U16` | 16 | **8** |
| registers | 61 | **64** |
| shared memory | 1,536 B | 1,536 B |
| spill stores / loads | 0 / 0 | 0 / 0 |
| instructions | 368 | **392** |
| `IADD` | 59 | 78 |

So the predicated destination selection for B did disappear, and the kernel got **larger**, not
smaller: the offset arithmetic costs more instructions than the predication saved, and three more
registers. Whether that is better is not something the listing can answer.

Interleaved, warmed, graphs on, tensor cores on, FP16 KV, two repetitions each:

| | accepted | merged |
| --- | ---: | ---: |
| `pp381 b32` | 88.72, 88.93, 88.66 | **92.91, 93.19, 93.07** |
| medians | 88.72 | **93.07 (+4.90%)** |

Ranges disjoint. A static instruction count that rose while the measured time fell is the reason
this file keeps saying that counts are not time.

26 focused MMA, topology, batched-parity, sequential and STANDARD parity and lifecycle tests pass
with the parity numbers unchanged, and the cross-width comparison was driven explicitly.

## The merged A tile — kept

The same representation change on the other operand, and the last one of this kind: the two A panels
become one `int[2 * BM * BK / 2]` — 256 ints, 1,024 bytes — the first at byte offset 0 and the
second at `A_SUBTILE_BYTES` (512). Staging writes `aTile[half * (BM * BK / 2) + j]`, a plain index
where it used to be `if (half == 0) aTileLo[j] else aTileHi[j]`; the reads become
`mmaLoadA(aTile, BK, 0)` and `mmaLoadA(aTile, BK, A_SUBTILE_BYTES)`. Merged B, geometry, grid,
activation reads and packing, arithmetic, barriers, staging order, weight representation and bounds
are all unchanged, and no other kernel is touched.

**Why each load sees the same panel layout.** The A load is `LdmatrixStmt.Variant.X4`, which is
`trans=false, swizzle=false` — there is no permutation to preserve. Its per-lane address is
`__bo = (__row << 5) + __col` with `__row = ((__grp & 1) << 3) + __rit` in [0,15] and
`__col = (__grp >> 1) << 4` in {0,16}, so a panel reaches at most byte 496 and fits inside 512. The
offset is added after that address is formed, which the emitted code shows as `__bo += 0` for the
first load and `__bo += 512` for the second. Element `aTile[128 + j]` is therefore exactly what
`aTileHi[j]` was, and the panels cannot overlap.

**Bit-identical** to the accepted default at 32 x 17408 x 5120, full chunk and 29-row padded chunk,
NaN-poisoned buffers, 1,114,112 values, all finite; the whole-model cross-width capture returns the
same logits again (SHA-256 `e17b0f731220525c`).

In the emission the two `__shared__ int adi_*[128]` become one `adi_3[256]` and the staging stores
are plain `adi_3[i] = …` with no predicate. **NVCC reconstruction**, against the accepted
merged-B kernel:

| | merged B (accepted) | merged A+B |
| --- | ---: | ---: |
| predicated `STS` | 8 | **0** |
| `STS` total | 20 | 16 |
| registers | 64 | 64 |
| shared memory | 1,536 B | 1,536 B |
| spill stores / loads | 0 / 0 | 0 / 0 |
| instructions | 392 | 344 |
| `IADD` | 78 | 67 |

Unlike the B merge, this one is smaller as well as simpler — but that is a description of the
listing, not a cost model.

| | accepted | merged A |
| --- | ---: | ---: |
| `pp381 b32` | 92.80, 92.69, 92.55 | **117.63, 117.46, 117.16** |
| medians | 92.69 | **117.46 (+26.7%)** |

Interleaved, warmed, graphs on, tensor cores on, FP16 KV, two repetitions each; ranges disjoint and
the within-run spread is under 0.3%. 26 focused MMA, topology, batched-parity, sequential and
STANDARD parity and lifecycle tests pass with the parity numbers unchanged, and cross-width was
driven explicitly.

## Re-profiled at `ad44666e`

Same method as every capture in this file: TornadoVM profiler, graphs off, `-p 381 -n 0 -b 32 -r 2`,
tensor cores and FP16 KV on, and the **last 24 chunk executions only** — two measured passes, 762
tokens, 1,536 `batchLayer` graph executions, with the compilation chunk, the warm-up pass and the
one-time uploads outside the window. That run reported 105.26 t/s, which is an instrumented figure
and not the 117.2-117.6 t/s the uninstrumented build measures.

Window totals: **6,047.9 ms kernel**, 7,130.1 ms task-graph, 121.5 ms copy-in, 1,082.2 ms of graph
time that is not kernel time (was 8,184.4 / 8,903.0 / 131.8 / 718.6 at the previous capture).

| task | kernel | calls | ms | share | µs/call |
| --- | --- | ---: | ---: | ---: | ---: |
| `ffn_up_proj` | `projectionMMAQ4_0` | 1536 | 1047.7 | **17.32%** | 682.1 |
| `ffn_gate_proj` | `projectionMMAQ4_0` | 1536 | 1030.9 | **17.05%** | 671.1 |
| `ffn_down_proj` (Q4_0, blocks 8+) | `projectionMMAQ4_0` | 1344 | 878.7 | 14.53% | 653.8 |
| `ssm_out_proj` | `projectionMMAQ5_K` | 1152 | 548.2 | 9.06% | 475.8 |
| `ssm_qkv_proj` | `projectionMMAQ4_0` | 1152 | 484.6 | 8.01% | 420.7 |
| `ssm_delta_rule` | `deltaRuleScan` | 1152 | 434.2 | 7.18% | 376.9 |
| `ssm_gate_proj` | `projectionMMAQ4_0` | 1152 | 274.2 | 4.53% | 238.0 |
| `attention` | `attentionBatchFP16Paged` | 384 | 236.1 | 3.90% | 614.9 |
| `ffn_down_proj` (Q4_1, blocks 0-7) | `projectionMMAQ4_1` | 192 | 224.5 | 3.71% | 1169.1 |
| `attn_output_proj` | `matrixVectorTiledBatchWithResidualQ4_0` | 384 | 203.8 | 3.37% | 530.8 |
| `attn_q_proj` | `projectionMMAQ4_0` | 384 | 180.7 | 2.99% | 470.5 |
| `attn_k_proj`, `attn_v_proj` | `projectionMMAQ4_0` | 384 each | 47.8 each | 0.79% each | 124.4 |

Gate and up separately: **17.05%** and **17.32%**, **34.37% combined**. `projectionMMAQ4_0` across
its eight task names: **3,992.1 ms, 66.01%**.

### Per-call against the previous capture, with its own controls

Same shapes, same workload, same instrumentation; only the kernel changed.

| task | previous | now | |
| --- | ---: | ---: | --- |
| `ffn_up_proj` | 1047.6 | 682.1 | **-34.9%** |
| `ffn_gate_proj` | 985.1 | 671.1 | **-31.9%** |
| `ffn_down_proj` (Q4_0) | 1062.6 | 653.8 | **-38.5%** |
| `ssm_qkv_proj` | 659.6 | 420.7 | -36.2% |
| `ssm_gate_proj` | 336.1 | 238.0 | -29.2% |
| `attn_q_proj` | 668.8 | 470.5 | -29.6% |
| `ssm_out_proj` (Q5_K, **unchanged kernel**) | 468.2 | 475.8 | +1.6% |
| `ffn_down_proj` (Q4_1, **unchanged kernel**) | 1172.9 | 1169.1 | -0.3% |
| `attention` (**unchanged**) | 606.0 | 614.9 | +1.5% |
| `attn_output_proj` (**unchanged**) | 522.7 | 530.8 | +1.5% |
| `ssm_delta_rule` (**unchanged**) | 378.0 | 376.9 | -0.3% |

The five untouched kernels move by at most 1.6%, which is what makes the 30-38% on the touched ones
readable. **These are task times and nothing more**: no occupancy, bandwidth or hardware cause is
inferred, and none was measured — no counters are available (`ERR_NVGPUCTRPERM`).

## The same tile merge for Q5_K and Q4_1

`projectionMMAQ5_K` and `projectionMMAQ4_1` carried the structure `projectionMMAQ4_0` had before
`3d350942`/`ad44666e` — four allocations and a branch per element on both operands. They now use one
allocation per operand with the second panel at a byte offset, exactly as Q4_0 does. Decoding,
scale and minimum arithmetic, geometry, activation reads, staging order, barriers and bounds are
unchanged, and Q4_0 itself was not touched.

Bounds were checked per kernel rather than inherited: both use the same `BM`, `BK`, `PANEL` and the
same store arguments, so a B panel's in-panel address is at most 254 before the swizzle (which
permutes within the same 256 bytes) and an A panel's per-lane address reaches at most 496 of its
512. Emission for both shows one `__shared__ int adi_3[256]` and one `__shared__ half half_4[256]`,
loads at `__bo += 0`, `+= 256` (B) and `+= 512` (A), and unpredicated stores.

**Bit-identical** to each accepted implementation at its production shape — Q5_K `ssm_out`
32x5120x6144 and Q4_1 `ffn_down` 32x5120x17408 — full chunk and 29-row padded chunk, NaN-poisoned
destinations, 327,680 values each, all finite.

Isolated kernel screens, warmed, resident weights, medians of nine rounds, two alternating rounds
per build:

| kernel | accepted | merged |
| --- | ---: | ---: |
| Q5_K `ssm_out` | 0.5086, 0.5199 ms | **0.4587, 0.4510 ms** |
| Q4_1 `ffn_down` | 1.3646, 1.2967 ms | **0.8666, 0.8700 ms** |

Whole model, `pp381 b32`, interleaved, four runs per build, all preserved:

| | runs | median | mean |
| --- | --- | ---: | ---: |
| accepted | 120.08, 121.03, 117.74, 117.98 | 119.03 | 119.21 |
| both merges | 123.58, 122.73, 124.34, 120.68 | **123.16** | **122.83** |

**An observed improvement of about 3%** — +3.47% on medians, +3.04% on means — **with run-to-run
variation of a few percent in both builds, and overlapping ranges. It is not a proven minimum
gain**, and the isolated screens are the cleaner evidence that each kernel improved. The whole-model
figure was not repeated further; no cross-pairing of best and worst runs is used as an uncertainty
estimate, because pairing runs from different points in a drifting session measures the drift.

## Asynchronous A staging in `projectionMMAQ4_0`

The manual A-tile sequence — two 2-byte global loads, a shift-and-or, a shared store, per slot — is
replaced by `asyncCopyToLocal(aTile, half * (BM * BK / 2) + j, aFP16, base)`, which lowers to
`cp.async.ca.shared.global [dst], [src], 4`. The merged A/B allocations, geometry, weight decoding,
arithmetic, MMA order and barriers are unchanged; B staging is untouched; Q5_K and Q4_1 keep their
manual staging.

**Architecture floor, read from the compiler rather than from a successful run.** `cp.async`
requires compute capability 8.0. TornadoVM's `CUDATensorCoreSupportPhase` enforces
`MMA_MAJOR_MIN = 8, MMA_MINOR_MIN = 0` and puts `CUDACpAsyncCopyNode`,
`CUDACpAsyncCommitGroupNode` and `CUDACpAsyncWaitGroupNode` in the **same** node set as the MMA
nodes — its own comment says "cp.async shares the sm_80 floor with mma.sync, so its nodes are gated
by this same phase". The kernel already contains `CUDAMMALoadANode`, `CUDAMMALoadBSwizzledNode`,
`CUDAMMAFragmentNode`, `CUDAMMAComputeNode` and `CUDAMMAStoreNode`, so **any device that can run
this kernel at all has already cleared sm_80**; a device below it is refused by that phase whether
or not cp.async is present. **No device selection changes, and no new capability, gate or option is
introduced.** Devices and shapes that are not MMA-eligible keep the scalar tiled kernels, as before.

**Alignment, against the kernel's actual reduction dimension.** The kernel's parameters are
`(m, n, k)` and `k` is the reduction dimension. At every call site the first argument of
`mmaEligible`, which is the one checked `% 32 == 0`, is the value passed as `k`:

| call site | guard | task arguments `(m, n, k)` |
| --- | --- | --- |
| generic `matVec` | `mmaEligible(n, d)` | `batchSize, d, n` |
| `ffn_gate_proj`, `ffn_up_proj` | `mmaEligible(config.dim(), config.hiddenDim())` | `batchSize, hiddenDim, dim` |
| `ffn_down_proj` | `mmaEligible(config.hiddenDim(), config.dim())` | `batchSize, dim, hiddenDim` |

So `32 | k`, hence `k` is even, and `base = (blockRow + row) * k + kBase + half * BK + kk` is even
for every issued copy — `kBase = blockIndex * 32`, `half * BK` and `kk = (j & 7) << 1` are all even.
The source byte address is `header(16) + 2 * base`, four-byte aligned, which is what the instruction
requires; the emission shows `(const char *) ul_0 + 16u + ((long long) i) * 2`.

**Coverage.** Each lane issues eight copies **to its own destinations**, not the same eight: with
`i = lane + slot * WARP_SIZE` over `lane < 32` and `slot < 8`, `i` takes every value in `[0, 256)`
exactly once, and the destination `half * 128 + j` equals `i`. The 256-int tile is therefore written
completely, once per slot, with no overlap between lanes.

**Synchronization.** Every lane commits and waits (`cp.async.commit_group`, `cp.async.wait_group 0`)
before the `localBarrier` that publishes both tiles, so no MMA reads a slot whose copy is in flight.
The trailing barrier after the two MMAs still separates a round's reads from the next round's
writes, including the next round's copies.

**Bit-identical** to the accepted kernel at 32 x 17408 x 5120, full chunk and 29-row padded chunk,
NaN-poisoned destinations, 1,114,112 values, all finite.

| | accepted | async staging |
| --- | ---: | ---: |
| `pp381 b32` | 121.21, 120.80, 120.57 | **139.94, 139.69, 139.51** |
| medians | 120.80 | **139.69 (+15.6%)** |

Interleaved, warmed, identical settings, two repetitions each; ranges disjoint, within-build spread
0.53% and 0.31%. **What caused it is not established.** The change replaces a load-pack-store
sequence with one copy instruction per slot; whether any transfer overlapped the B staging that
follows was not measured, and there is no cross-iteration pipelining here. No SASS was inspected for
this variant and no claim rests on instruction counts.

## The same cp.async staging for Q5_K and Q4_1 — **does not compile**

Attempted, reverted, recorded so nobody repeats it. The A-staging replacement that works in
`projectionMMAQ4_0` was applied unchanged to `projectionMMAQ5_K` and `projectionMMAQ4_1`. The
prerequisites all hold for both: their reduction dimension is checked by the same `mmaEligible`
first argument (`mmaEligible(valueDim, dim)` for `ssm_out`, `mmaEligible(hiddenDim, dim)` for the
early `ffn_down`), so `32 | k` and every `base` is even and four-byte aligned; the staging loop is
character-for-character the one in Q4_0, so `i = lane + slot * 32` covers `[0, 256)` exactly once
and each lane copies to its own destinations; the commit/wait sits before the publishing barrier and
the trailing barrier still separates a round's reads from the next round's writes; and the sm_80
floor is the same gate the MMA nodes in those kernels already pass.

**NVRTC rejects the result**: `tornado_kernel.cu(17): error: identifier "half" is undefined` on
`__shared__ half half_4[256];`.

The cause is the include gate, not the kernel. `CUDACompilationResultBuilder#finish` prepends
`#include <cuda_fp16.h>` only when the emitted source contains `__half`, `half2` or `2half`. Once
the A staging becomes `cp.async`, these two kernels emit **no** such spelling: their A staging no
longer produces `__half_as_ushort`, their B store emits the bare `((half *) half_4)[...]`, and their
block scales go through `halfFromBytes`, which is byte arithmetic. `projectionMMAQ4_0` survives the
same change only because `5e68ffca` made its scale read emit `__half2float`, which the scan matches.

Both kernels therefore keep their manual A staging. Making them compile would mean either a
TornadoVM-side change to that scan — the backend emits bare `half` itself, so the gate does not
cover its own output — or re-introducing an fp16 spelling in the kernel purely to satisfy a text
match. The first is out of scope here and the second is a workaround, so neither was pursued and no
timing was taken.

## Native half reads, then async staging, for Q5_K and Q4_1

Two stages, kept separate so their correctness and their timings are not conflated.

**Stage one — native half reads (`6f98a77f`).** Both kernels assembled their fp16 block fields from
two byte loads; they now read them with `ByteArray.getHalfFloat`. The inventory was checked field by
field rather than inherited from Q4_0:

| kernel | field | offset | block stride | address | two-byte aligned |
| --- | --- | ---: | ---: | --- | --- |
| Q4_1 | scale | 0 | 20 | `20k` | yes |
| Q4_1 | minimum | 2 | 20 | `20k + 2` | yes |
| Q5_K | `d` | 0 | 176 | `176k` | yes |
| Q5_K | `dmin` | 2 | 176 | `176k + 2` | yes |
| Q5_K | six-bit sub-block scales | 4 | 176 | — | **not halves**, still byte reads |

`getHalfFloat` reads a native-endian short where `halfFromBytes` composed the byte pair
little-endian. The two agree **on this little-endian CUDA target**, which is where it was verified;
no claim is made for a big-endian one. `HalfFloatConversionAccelTest` now reads every finite half
encoding at each layout — strides 18, 20 and 176, offsets 0 and 2 — against `Float.float16ToFloat`:
63,488 encodings per layout, bit for bit.

**Stage two — async A staging (`54bc0db3`).** The manual
A sequence becomes `asyncCopyToLocal` plus a commit and a wait before the publishing barrier, as in
`projectionMMAQ4_0`. It compiles here **only because stage one restored an fp16 spelling**: see
finding 5 in `tornadovm-issues`, which this repository avoids rather than fixes.

Both stages are bit-identical to the accepted kernels at the production shapes (Q5_K `ssm_out`
32x5120x6144, Q4_1 `ffn_down` 32x5120x17408), full chunk and 29-row padded chunk, NaN-poisoned
destinations, 327,680 values each, all finite.

Isolated kernel screens, medians of nine rounds, warmed, resident weights, two alternating rounds
per build:

| kernel | accepted | stage one | stage one + two |
| --- | ---: | ---: | ---: |
| Q5_K `ssm_out` | 0.4331, 0.4330 ms | 0.4224, 0.4142 | **0.3670, 0.3653** |
| Q4_1 `ffn_down` | 0.8315, 0.8373 ms | 0.7538, 0.7550 | **0.6315, 0.6325** |

Whole model, `pp381 b32`, interleaved, identical settings, all runs preserved:

| | runs | median |
| --- | --- | ---: |
| accepted | 147.15, 146.51, 146.36 | 146.51 |
| both stages | 149.93, 150.03, 149.60 | **149.93** |

**+2.33% observed within this session**, ranges disjoint — not a guaranteed minimum. The absolute
figures are much higher than the previous session's on the same code path; the GPU held 62-63 °C and
1590 MHz throughout this one. Only the interleaved within-session comparison means anything, and
these two kernels were 12.8% of prefill kernel time in the last profile.
