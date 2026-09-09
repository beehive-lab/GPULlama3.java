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
