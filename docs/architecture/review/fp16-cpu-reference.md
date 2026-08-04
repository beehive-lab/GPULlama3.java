# The FP16 CPU reference was the inaccurate side

**Status: fixed.** 2026-08-04. Follows from the Granite FP16 question raised by
[`model-family-sweep.md`](model-family-sweep.md).

## The question

granite-4.0-1b F16 sat at relative L2 **1.4e-02** against the CPU, roughly 3× the other FP16 models
and 2000× its own Q8_0 variant (6.7e-06). Since every parity gate treats the CPU as the reference,
that read as "the Granite FP16 GPU path is worse than everyone else's".

## What it actually was

`FP16FloatTensor.vectorDot` — the **CPU** dot product for FP16 weights — converts half to float with
a bit-shift shortcut that cannot represent denormals, and its comment says so:

```java
// Does not support infinities nor NaNs, preserves sign, emulate DAZ (denormals-are-zero).
// Expects well-formed float16 values only (e.g. model weights).
```

Model weights are not free of denormals. Measured fraction of weights below the smallest normal FP16
(6.1e-05):

| model | wq[0] | w1[0] | embeddings |
| --- | --- | --- | --- |
| granite-4.0-1b F16 | 0.359% | **0.655%** | 0.122% |
| Llama-3.2-1B F16 | 0.226% | 0.260% | 0.244% |
| Qwen3-0.6B F16 | 0.183% | 0.152% | 0.160% |

The CPU dropped all of them; the GPU converted them properly. The ordering of that table is the
ordering of the CPU↔GPU gaps, and Granite has the most.

Confirmed by running the same comparison with the vectorized path disabled
(`-Dllama.VectorBitSize=0`, which falls back to the scalar `Float.float16ToFloat`):

| | vectorized CPU dot (DAZ) | scalar CPU dot |
| --- | --- | --- |
| Llama-3.2-1B F16 | 4.5e-03 | **6.3e-04** |
| granite-4.0-1b F16 | 1.4e-02 | **5.2e-04** |

Granite improves 26×, Llama 7×, and Granite ends up *better* than Llama. The remaining ~5e-04 is
the GPU's FP16 activation storage, which `golden/Fp16SimParity` predicts independently: simulating
that rounding on the CPU moves the CPU by 5e-04–9e-04, the size of what is left.

So: not a Granite defect, not a GPU defect. The gate's reference was wrong, most on the model with
the most denormal weights.

## The fix

`vectorDot` now converts denormals: their value is `mantissa · 2^-24`, computed by converting the
mantissa as an integer and scaling, blended in under a mask. The mask is false for almost every
lane, so the cost is nil — CPU decode measured 22.8/23.1/22.9 tok/s after versus 22.6/22.5/22.8
before, on Llama-3.2-1B F16.

The vectorized result now matches the scalar reference (6.34e-04 vs 6.31e-04 on Llama, 5.15e-04 vs
5.16e-04 on Granite).

All FP16 models now sit in one band instead of spread across an order of magnitude:

| model | before | after |
| --- | --- | --- |
| Llama-3.2-1B F16 | 4.5e-03 | 6.3e-04 |
| granite-4.0-1b F16 | 1.4e-02 | 5.2e-04 |
| Qwen3-0.6B F16 | 2.3e-03 | 7.8e-04 |

`CpuGpuParityAccelTest`'s FP16 bounds tighten accordingly: max-error ceiling 0.1 → 0.02, relative L2
1e-2 → 2e-3, cosine 0.9999 → 0.99999, `atol` 0.06 → 0.015. Most of what that gate used to tolerate
was error on the reference side.

## Worth noting

This also means CPU inference itself was slightly wrong for every FP16 model, independently of the
GPU — the denormal weights simply vanished from every dot product. Q8_0 was never affected (a
different tensor type, no denormals).

The general lesson matches the rest of this investigation: "the reference disagrees with the
implementation" says nothing about *which* one is wrong until something measures both. The Q8_0
configuration was the control that made this visible — same GPU kernels, same CPU harness, 2000×
closer — and it was sitting in the results for weeks.

## Tools

| Class | Purpose |
| --- | --- |
| `golden/Fp16Sensitivity` | FP16 round-trip cost per model, activation magnitude range, denormal weight counts |
| `golden/Fp16SimParity` | CPU forward with the GPU's FP16 activation rounding, to attribute what remains |
