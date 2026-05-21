# Attention Kernel Findings — Qwen3

**Date:** 2026-05-21  
**Branch:** fix/qwen3-performance  
**Backends tested so far:** NVIDIA PTX  
**Models:** Qwen3-0.6B, 1.7B, 4B, 8B × FP16 + Q8_0  

---

## Background

TornadoVM profiler logs showed attention is the dominant bottleneck across all 8 Qwen3 model/quant combinations. Three flash attention kernel variants exist in `TransformerComputeKernelsLayered.java`. Qwen3 currently uses `processHeadsFlashAttention` (V1). This document records the analysis and benchmark results across all three.

---

## Kernel Variants

All three variants share the same signature and are 1D-grid kernels: one workgroup per Q-head, `findOptimalLocalSize(headDim)` threads per workgroup (= 64 for headDim=128).

### V1 — `processHeadsFlashAttention` (current Qwen3 default)
- `BLOCK_SIZE_C = 16`
- K/V loading: stride-based — only threads `0..tileSize-1` work; **75% of threads idle** at LWS=64
- Max reduction: all 64 threads redundantly scan s_tile sequentially
- **V-accumulation: every thread computes all 128 output dims** → 64× redundant arithmetic; 63/64 of all multiply-adds are discarded
- Output write: strided across threads

### Opt — `processHeadsFlashAttentionOpt` (line 996)
- `BLOCK_SIZE_C = 32` (larger tile)
- K/V loading: cooperative element-indexed — all threads participate
- Max reduction: parallel tree reduction ✓
- **V-accumulation: still 64× redundant** — `output[headSize]` per thread, full loop unchanged
- Output write: dimension-partitioned + 4-element unroll ✓

### OptV2 — `processHeadsFlashAttentionOptV2` (line 808)
- `BLOCK_SIZE_C = 32`
- K/V loading: cooperative element-indexed ✓
- Max reduction: parallel tree reduction ✓
- **V-accumulation: dimension-partitioned** — `output[dimsPerThread]` per thread (2 dims at LWS=64, headDim=128); zero redundancy ✓
- Output write: dimension-partitioned ✓
- Note: uses static `MAX_HEAD_SIZE=256`, `MAX_LOCAL_SIZE=256` allocations — written as an OpenCL compatibility workaround; PTX handles dynamic sizes fine but static sizing costs nothing

**The dominant bottleneck is V-accumulation redundancy.** Opt fixes K/V loading and max reduction but not V-accumulation, which is why it barely improves on V1. OptV2 fixes all three.

---

## Benchmark Results — PTX (5-run average, profiler OFF)

**Setup:** `--max-tokens 2048 --prompt "Explain the concept of entropy in one paragraph. /no_think"`

| Model          | Quant | v1 (tok/s) | opt (tok/s) | optv2 (tok/s) | opt/v1  | optv2/v1 |
|----------------|-------|-----------:|------------:|--------------:|--------:|---------:|
| Qwen3-0.6B     | FP16  | 24.24      | 24.18       | **45.22**     | −0.2%   | +86.6%   |
| Qwen3-0.6B     | Q8_0  | 25.77      | 25.63       | **53.99**     | −0.5%   | +109.5%  |
| Qwen3-1.7B     | FP16  | 16.82      | 17.36       | **31.74**     | +3.2%   | +88.7%   |
| Qwen3-1.7B     | Q8_0  | 20.63      | 21.12       | **46.87**     | +2.4%   | +127.2%  |
| Qwen3-4B       | FP16  | 11.86      | 12.12       | **21.25**     | +2.2%   | +79.2%   |
| Qwen3-4B       | Q8_0  | 14.76      | 14.87       | **32.91**     | +0.7%   | +123.0%  |
| Qwen3-8B       | FP16  | 8.78       | 8.95        | **13.98**     | +1.9%   | +59.2%   |
| Qwen3-8B       | Q8_0  | 11.92      | 12.26       | **25.40**     | +2.9%   | +113.1%  |

**Key conclusions (PTX):**
- `opt` is statistically indistinguishable from `v1` (±3%, within run-to-run noise) — fixing K/V load and max reduction alone does not move the needle.
- `optv2` is **~2× faster** across all 8 combinations. Gains range from +59% (8B FP16) to +127% (1.7B Q8_0).
- The speedup is consistent across model sizes and quantizations — this is a structural improvement, not noise.
- `optv2` is the clear winner **on PTX**. Cross-backend validation is pending.

---

## Switching the Kernel

The attention kernel is selectable at runtime via `-Dllama.attentionKernel=<variant>` (default: `v1`).  
Valid values: `v1`, `opt`, `optv2`.

This property is read in `AbstractFFNLayers.ATTENTION_KERNEL` and applied in:
- `Qwen3FP16FFNLayers.createFFNLayerTaskGraph`
- `Qwen3Q8_0FFNLayers.createFFNLayerTaskGraph`

Llama/Mistral paths are unaffected.

---

## How to Repeat on a Different Backend

### Prerequisites

1. Clone or pull the branch `fix/qwen3-performance`.
2. Build the JAR:
   ```bash
   sdk use java 25.0.2-open
   sdk use tornadovm <your-tornadovm-build>   # e.g. 4.0.0-jdk25-opencl or 4.0.0-jdk25-metal
   mvn package -DskipTests
   ```
3. Ensure model files are present at `~/LLMModels/` (same names as below).

### Run performance benchmark (5-run average, profiler OFF)

```bash
cd ~/GPULlama3.java
./scripts/benchmark_qwen3_attn_perf.sh 5 opencl   # or: metal
```

Output goes to `/tmp/qwen3_attn_perf/<backend>/`. The summary table is printed at the end.

### Run profiler collection (1 run per combination, profiler ON)

```bash
cd ~/GPULlama3.java
./scripts/benchmark_qwen3_attn_profiler.sh opencl   # or: metal
```

Profiler JSON dumps go to `/tmp/qwen3_attn_profiler/<backend>/`. One JSON per model+variant combination.

### What to report back

From the performance run, copy the summary table printed at the end (the `============` block).  
From the profiler run, share the JSON dumps or note the relative time share of the `attention` task graph.

---

## Open Questions

1. **Does `optv2` hold on OpenCL?** OptV2's static allocations were written specifically to fix OpenCL issues — so it may already be the right choice there. Or a different variant may win.
2. **Does `optv2` hold on Metal?** Metal has different shared memory and warp-equivalent semantics.
3. **Is a cleaned-up kernel needed?** `optv2` has cosmetic issues: `MAX_OUTPUT_DIMS = MAX_HEAD_SIZE / 8 = 32` wastes register space (actual use is 2), and `reduction_shared[MAX_LOCAL_SIZE=256]` wastes shared memory (actual use is 64). These do not affect correctness but reduce SM occupancy. A `processHeadsFlashAttentionQwen3` could address this once backend data is in.
4. **Should the default change?** Only after cross-backend results confirm `optv2` is safe everywhere — or a backend-conditional default is implemented.
