# CPU↔GPU parity (T1.5) — root cause and gate design

**Status: the parity defect is fixed and the gate passes.** Closed 2026-08-04.
Task ID **T1.5** in [`../execution-backlog.md`](../execution-backlog.md).

## The defect

`TransformerComputeKernelsLayered.ropeRotationWithCacheCopy` computed RoPE frequencies from a
constant compiled into the kernel:

```java
float freq = 1.0f / TornadoMath.pow(50000.0f, head_dim / (float) headSize);
```

Llama-3.2's `rope_theta` is **500000** — the kernel's base was missing a zero. The CPU path builds
its `freq_cis` tables from the model metadata (`RoPE.precomputeFreqsCis`, via the loader), so the
two paths applied different rotation angles at every position, and any Llama 3.1 frequency scaling
was ignored on the GPU entirely.

The output still read as fluent English, which is why this survived: a consistently wrong
positional encoding degrades quality without producing obvious garbage. What it did produce was a
CPU↔GPU logit gap that **grew with position** — mean absolute error 0.11 at row 0 rising to 0.84 by
row 63 — and that was read as accumulated FP16 noise.

## How it was localized

`golden/LayerParity` compares each layer-0 intermediate the GPU exposes under
`-Dgpullama3.diag.transfers` against the same quantity recomputed on the CPU from the standard
weights:

| quantity | result |
| --- | --- |
| RMS scale (`temp[0]`) | matches to 7.9e-08 relative |
| normalized activation (`wrapXbFP16`) | **bit-identical** to the CPU FP32 value rounded to FP16, 2048/2048 |
| Q after RoPE | relL2 **0.386** — the only real disagreement |
| Q pair magnitudes (rotation-invariant) | match to relL2 5.0e-05 |

The last two lines are the whole diagnosis: the QKV projection is faithful (its output magnitudes
are right), so the two paths disagreed about the *rotation*, not the arithmetic before it. Feeding
the CPU matmul the GPU's own FP16 activation changed nothing (0.386 either way), which ruled the
input out as well.

Note the readback is taken at the end of the layer graph, so `wrapQ` is already rotated; comparing
it against a pre-RoPE CPU value is meaningless. `LayerParity` rotates the CPU side first.

## The fix

Llama and Mistral (FP16 and Q8_0) now use `ropeRotationWithCacheCopyPrecomputed` with
`weights.freq_cis_realFlat` / `freq_cis_imagFlat` — the model's own tables, exactly as Devstral
already did. The tables are added to the layer-0 `FIRST_EXECUTION` transfer and to the
`consumeFromDevice` list of later layers.

### Measured effect

CPU vs GPU, 64 teacher-forced rows, Llama-3.2-1B:

| comparison | before | after |
| --- | --- | --- |
| CPU-F16 vs GPU-F16, worst | 3.58 | **0.032** |
| CPU-F16 vs GPU-F16, mean | 0.277 | **0.0039** |
| CPU-Q8_0 vs GPU-Q8_0, worst | 3.54 | **4.3e-05** |
| free-running decode, Q8_0 | differed | **identical token ids to the CPU** |
| free-running decode, F16 | differed | differs at one near-tie (CPU gap 0.0075) |

For scale, switching *weights* between FP16 and Q8_0 moves the logits by 0.35 worst / 0.043 mean —
so the residual GPU-vs-CPU difference is now an order of magnitude below the quantization choice,
where before it was six times larger.

Decode also got **faster**, since a table lookup replaces `pow`/`cos`/`sin` per element:
CUDA ~135 → ~153 tok/s, OpenCL ~100 → ~105 tok/s. Determinism is unaffected (0/300 both backends,
see [`fp16-determinism-investigation.md`](fp16-determinism-investigation.md)).

## The gate

`CpuGpuParityAccelTest` now asserts four complementary things instead of one row-wide tolerance.

1. **Elementwise** `|gpu − cpu| ≤ atol + rtol·|cpu|` — the conventional mixed bound, the same shape
   as `torch.testing.assert_close` — with a violation budget of 0.01% of elements, because the
   maximum over 8.2 million logits is decided by one tail value.
2. **Hard ceiling** on the largest single absolute error, so the budget can never hide an excursion.
3. **Whole-vector** relative L2 and cosine similarity, which describe the row rather than its most
   extreme element.
4. **Decision-level** argmax agreement and top-5/top-10 overlap. A reversal fails only when the CPU
   decision was not close — measured as the gap between the two tokens that actually competed,
   `ref[cpuPick] − ref[gpuPick]`, not each path's own top1−top2 margin, which can involve a third
   token.

### Why the old bound was wrong

It computed one tolerance per row from the row maximum, `1e-2·max|ref| + 1e-3` ≈ 0.26, and applied
it to every element. A logit of 0.002 could differ by 0.2 and pass. The javadoc justified this via
`Σ|wᵢaᵢ| ≥ |Σ wᵢaᵢ|`, but that inequality relates a *single* output element to *its own* absolute
product sum; it says nothing about using the row maximum as a surrogate for every element. And
cancellation in dot products is normal, not exceptional, so `Σ|wᵢaᵢ|` can far exceed the logit it
produces.

The test also reported `worst |cpu-gpu|=0 ... row -1` whenever everything passed, because it only
recorded a worst value when the tolerance was exceeded, and it never asserted that the two rows had
the same length.

### Thresholds, and how to re-derive them

Measured with `golden/ParityProfile` on the pinned tuple, then set roughly 2× above the observed
worst case. They are **per quantization**: the FP16 path stores normalized activations as FP16 while
the Q8_0 path keeps FP32, and the two are three orders of magnitude apart.

| | FP16 observed | FP16 bound | Q8_0 observed | Q8_0 bound |
| --- | --- | --- | --- | --- |
| max abs error | 0.032 | 0.1 ceiling, `atol` 0.06 | 4.3e-05 | 1e-3 ceiling, `atol` 5e-4 |
| relative L2 | 4.5e-03 | 1e-2 | 6.4e-06 | 1e-4 |
| cosine | 0.99999173 | 0.9999 | 1.00000000 | 0.999999 |
| elementwise violations | 0 at `atol` 0.05 | budget 0.01% | 0 at `atol` 1e-4 | budget 0.01% |

`atol` dominates because most logits sit near zero; `rtol` (1e-2) is what keeps the bound honest on
the few large ones. Re-derive with `ParityProfile` if the tuple, prompt or compared-token count
changes. **Do not widen a bound to make a failing run pass** — that is what the row-max surrogate
effectively did.

Teacher forcing is retained and is what makes any of this meaningful: greedy decoding is
autoregressive, so without it the first tipped near-tie sends the paths into different contexts and
later rows compare unrelated states.

## Follow-ups

- **The same hardcoded base still exists elsewhere.** `TransformerBatchPrefillKernels` (two sites,
  `50000.0f`), the dead `TransformerComputeKernelsLayered.ropeRotation` (`50000.0f`), `Phi3Kernels`
  (`10000.0f`), and `Qwen3Kernels` (`1000000.0f`, four sites). Qwen3 and Phi3 happen to match their
  models' current `rope_theta`, so they are latent rather than active bugs — but the value belongs
  to the model, not the kernel. `GraniteKernels` already takes `ropeTheta` as a parameter.
- **Batch prefill (`--with-prefill-decode --batch-prefill-size N`) produces garbage** — output is
  `. 0 0 0 0 …`. Verified pre-existing (reproduces before the RoPE fix), and it is more broken than
  a wrong rotation would explain. Plain `--with-prefill-decode` (no batching) is fine.
- Golden logits must be regenerated: the committed goldens recorded the wrong-RoPE GPU output, so
  `GoldenLogitsAccelTest` now fails on token ids. That is T1.4's remaining step.
- A free-running (non-teacher-forced) behavioural check that records *where* the paths first
  diverge would complement the numerical gate; currently only the teacher-forced comparison exists.

## Tools

| Class | Purpose |
| --- | --- |
| `golden/LayerParity` | layer-0 CPU↔GPU comparison of RMS scale, activation, Q pre/post RoPE, pair magnitudes |
| `golden/ParityProfile` | error distributions and `atol` sweep the thresholds are derived from |
| `golden/ParityCross` | 4-way CPU/GPU × F16/Q8_0 comparison; separates path effects from weight-precision effects |
