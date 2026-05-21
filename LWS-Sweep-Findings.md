# LWS Sweep Findings — Qwen3 on TornadoVM PTX

**Date:** 2026-05-21  
**Branch:** fix/qwen3-performance  
**GPU backend:** NVIDIA PTX  
**Sweep:** `llama.localWorkGroupSize` ∈ {32, 64, 128} × 8 Qwen3 models  
**Baseline:** LWS=32 (original hardcoded default)

---

## Results (tok/s)

| Model    | Quant | LWS=32 | LWS=64 | LWS=128 | Winner        |
|----------|-------|-------:|-------:|--------:|---------------|
| Qwen3-0.6B | FP16 | 23.73 | **25.13** | 15.70 | 64 (+5.9%)  |
| Qwen3-0.6B | Q8_0 | **24.22** | 22.82 | 15.85 | 32 (64 −5.8%) |
| Qwen3-1.7B | FP16 | 16.53 | **18.64** | 17.22 | 64 (+12.8%) |
| Qwen3-1.7B | Q8_0 | 19.49 | **21.04** | 11.89 | 64 (+8.0%)  |
| Qwen3-4B   | FP16 | 11.77 | **13.62** | 23.81† | 64 (+15.7%) |
| Qwen3-4B   | Q8_0 | 14.73 | **14.91** | 4.82‡  | 64 (+1.2%)  |
| Qwen3-8B   | FP16 | 8.75  | **10.83** | 6.92  | 64 (+23.8%) |
| Qwen3-8B   | Q8_0 | 12.31 | 12.78  | **14.20** | 128 (+15.3%) |

† 4B FP16 LWS=128: only 33 tokens generated — short response cuts off after warmup; not representative.  
‡ 4B Q8_0 LWS=128: 678 tokens in 140 s — catastrophic regression; likely GPU serialization or occupancy collapse.

---

## Key Observations

**LWS=64 is the clear winner for FP16.**  
Gains are consistent across all model sizes and grow with model size: +5.9% (0.6B) → +23.8% (8B). This matches the occupancy theory: LWS=32 leaves ~50% Ampere SM occupancy; LWS=64 restores 100%.

**LWS=64 for Q8_0 is neutral-to-positive, with one exception.**  
Gains on 1.7B (+8.0%), 4B (+1.2%), 8B (+3.8%). The 0.6B Q8_0 regresses by −5.8%, which may be noise on a single run — the model is small enough that kernel launch overhead and JIT variance matter more.

**LWS=128 is unreliable in single-run measurements.**  
Results are erratic: some Q8_0 models collapse catastrophically (4B: −67%), others improve (8B Q8_0: +15.3%). Token counts vary widely across runs, making comparisons unreliable. LWS=128 requires multi-run validation before drawing conclusions.

---

## Kernel Dimensions Reference

| Model    | dim  | hiddenDim | layers | rms_ffn_gate_up WGs | attn_rms_qkv WGs |
|----------|------|-----------|--------|--------------------:|-----------------:|
| Qwen3-0.6B | 1024 | 3072   | 28     | 3,072               | 4,096            |
| Qwen3-1.7B | 2048 | 6144   | 28     | 6,144               | 4,096            |
| Qwen3-4B   | 2560 | 9728   | 36     | 9,728               | 6,144            |
| Qwen3-8B   | 4096 | 12288  | 36     | 12,288              | 6,144            |

All models: headDim=128, attention LWS=64 (fixed by `findOptimalLocalSize`), RoPE local=(8,1).  
Valid GEMV LWS candidates: 32, 64, 128, 256 (all kernel-safe).  
Valid RMS LWS candidates: 32, 64, 128, 256, 512 (all divide dim for every model size).

---

## Current Status

- Default remains LWS=32 (no change to `AbstractLayer.java` default).
- `llama.localWorkGroupSize` system property is live and configurable.
- Next: deeper bottleneck analysis via TornadoVM profiler logs.

---

## Open Questions

1. Is 0.6B Q8_0 LWS=64 regression real or single-run noise?
2. Why does 4B Q8_0 at LWS=128 collapse so severely? Investigate occupancy vs serialization.
3. Does 8B Q8_0 LWS=128 improvement hold over multiple runs?
4. Does `state.localSize` (RMS norm LWS=256) have measurable impact — worth sweeping separately?
