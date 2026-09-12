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
