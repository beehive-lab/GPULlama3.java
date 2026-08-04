# Model-family sweep and batch prefill (2026-08-04)

Checks that the RMS-reduction and RoPE fixes hold beyond Llama-3.2-1B, and a root-cause pass over
batch prefill. Models from `~/LLMModels`, CUDA backend, `Llama-3.2-1B` fixtures from
`~/.gpullama3/test-models`.

Two measurements per model:

- **determinism** — `golden/DivergenceRate`, 200 identical executions of one fixed plan;
- **parity** — `golden/ParityProfile`, CPU vs GPU over 64 teacher-forced rows, plus whether
  free-running greedy decode emits the CPU's token ids.

## Results

| Model | Arch | Quant | Diverged | worst relL2 | tokens == CPU |
| --- | --- | --- | --- | --- | --- |
| Llama-3.2-1B | Llama | F16 | 0/300 | 4.5e-03 | one near-tie differs |
| Llama-3.2-1B | Llama | Q8_0 | 0/300 | 6.4e-06 | yes |
| Llama-3.2-3B | Llama | Q8_0 | 0/200 | 6.2e-06 | yes |
| Mistral-7B-v0.3 | Mistral | Q8_0 | 0/200 | 7.6e-06 | yes |
| Qwen3-0.6B | Qwen3 | F16 | 0/200 | 2.3e-03 | yes |
| Qwen3-0.6B | Qwen3 | Q8_0 | 0/200 | 5.9e-06 | yes |
| Qwen3-1.7B | Qwen3 | F16 | 0/200 | 2.4e-03 | yes |
| Qwen2.5-0.5B | Qwen2 | F16 | 0/200 | 1.2e-03 | yes |
| Qwen2.5-0.5B | Qwen2 | Q8_0 | 0/200 | 8.1e-06 | yes |
| Phi-3-mini-4k | Phi3 | F16 | 0/200 | 7.7e-04 | yes |
| Phi-3-mini-4k | Phi3 | Q8_0 | 0/200 | 4.1e-06 | yes |
| granite-4.0-1b | Granite | F16 | 0/200 | 1.4e-02 | yes |
| granite-4.0-1b | Granite | Q8_0 | 0/200 | 6.7e-06 | yes |
| granite-3.2-2b | Granite | Q8_0 | 0/200 | 4.6e-06 | yes |
| DeepSeek-R1-Distill-Qwen-1.5B | Qwen2 | Q8_0 | 0/200 | 7.3e-06 (was 0.80) | yes (was no) |

Determinism holds everywhere — the RMS-reduction fix is confirmed across all families, not just the
one it was found on. The FP16 rows sit around 1e-3–5e-3 relative L2 and the Q8_0 rows around 1e-5,
which is the expected split: the FP16 GPU path stores its normalized activations as FP16 while the
Q8_0 path keeps FP32.

Two defects surfaced. Both were the same shape as the original RoPE bug: a constant in a kernel
standing in for a value that belongs to the model.

### DeepSeek-R1-Distill-Qwen (fixed)

`Qwen3Kernels` hardcoded `theta = 1000000`, which is right for Qwen2.5 and Qwen3 checkpoints. A
Qwen2-architecture distill need not agree: DeepSeek-R1-Distill-Qwen-1.5B uses **10000**, so its GPU
output was rotated wrong at every position — relative L2 0.80, cosine 0.70, different tokens. The
decode kernel and both Qwen3 batch-prefill kernels now take `ropeTheta` from the configuration.

### Phi-3 on the CPU path (fixed)

`Phi3ModelLoader` declared `modelContextLength` and never assigned it, so
`precomputeRopeFrequencies()` built zero-length `freq_cis` tables and the **CPU** path threw
`ArrayIndexOutOfBoundsException` on the first token. The GPU path never noticed, because the Phi3
RoPE kernel computes its frequencies inline. Phi-3 had presumably never been run on CPU since that
loader was written.

### Granite F16 — flagged, not diagnosed

granite-4.0-1b F16 has relative L2 1.4e-02, about 3× the other FP16 models, with absolute errors up
to 21.9 (its logits are large, so the *relative* picture is less alarming than the absolute one).
Tokens match the CPU and the Q8_0 variant is at 6.7e-06, so this is FP16-path-specific. Granite has
extra scaling factors (embedding, residual, attention, logit) that the FP16 path may be applying at
lower precision. Not investigated.

## Batch prefill

`--with-prefill-decode --batch-prefill-size N` produced garbage or subtly wrong output. Two
independent defects, both fixed.

**1. The batch RoPE kernels carried the same hardcoded base** (50000) as the single-token kernel.
Once decode was corrected to use the model's tables, prefill and decode also disagreed *with each
other*, so the KV cache written during prefill no longer matched what decode expected. They now use
`freq_cis` too.

**2. Padding rows wrote KV outside their layer's slice.** The kernels launch a fixed `batchSize`
rows regardless of how many tokens the current chunk holds, and those padding rows wrote K/V at
`startPos + rowIndex` with no bound. `contextLength` is the CLI's `--max-tokens`, so a short
generation with a large batch wrote past the layer's KV region and corrupted the next layer's keys
and values. `batchStartPosHolder` now carries the chunk's real token count and the RoPE and
attention kernels return early for padding rows.

The second defect is why the symptom looked so erratic:

| prompt | batch | context (`--max-tokens`) | before |
| --- | --- | --- | --- |
| 21 tok | 8, 32, 64, 127 | 40 | text plausible, logits wrong |
| 21 tok | 128 | 40 | garbage (`. The . The …`) |
| 21 tok | 128 | 200 or 600 | correct |
| 128 tok (full chunk) | 128 | 170 | correct |
| 15 tok | 8 | 40 | different text from the non-batch path |
| 16 tok (exact multiple) | 8, 16 | 40 | identical to the non-batch path |

After both fixes, batch sizes 8/32/64/127/128 reproduce the non-batch path's output exactly at
`--temperature 0` on Llama-3.2-1B F16 and Q8_0 and Qwen3-0.6B, including the previously destroyed
`--max-tokens 40` + batch 128 case.

Note `contextLength = options.maxTokens()` (`ModelLoader:93`): the KV cache is sized by the token
budget, so "how many tokens am I generating" silently sets "how much history can exist". That is
what turned a padding-row bug into layer-to-layer corruption.

## Follow-ups

- **Granite F16** relative L2 3× the other FP16 models (above).
- `Qwen2 + Q8_0` with `--with-prefill-decode` throws `UnsupportedOperationException` by design
  (`ForwardPlanFactory:174`) — worth stating in user-facing docs, since the flag combination looks
  supported from the CLI.
- Remaining hardcoded RoPE bases: `Phi3Kernels` (10000) and the dead
  `TransformerComputeKernelsLayered.ropeRotation` (50000). Phi3's matches its models today; the
  dead one should be deleted.
- Q4_K / Q5_K fixtures exist in `~/LLMModels` but were not swept; only F16 and Q8_0 were.

## Tools

`golden/DivergenceRate`, `golden/ParityProfile` (accepts `-Dparity.model`, `-Dparity.perRow`),
`golden/LayerParity`, `golden/ParityCross`, `golden/TokenCount` (prompt token counts and the model
configuration, for building prompts of an exact length).
