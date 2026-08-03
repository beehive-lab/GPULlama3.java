# Local refactoring baseline — RTX 5090 Laptop, TornadoVM 5.2.0

Replaces `docs/perf-history.jsonl` as the reference for the architecture refactor.
Those entries came from different (CI) hardware and are not comparable to this machine.

## Pinned tuple

| Field | Value |
| --- | --- |
| Machine | local laptop |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU, 23.4 GB |
| Driver | 580.142 |
| TornadoVM | 5.2.0-jdk21 (`cuda` and `opencl` SDKs) |
| JDK | OpenJDK 21.0.2 |
| Repo commit | 7de4d5b + Phase 0 pom bump (`tornadovm.base.version` 5.2.0) |
| Date | 2026-08-03 |

## Method

Fixed prompt `"Write a detailed 400-word essay about the history of computing."`,
`-n 256`, `--gpu --verbose-init`, device memory sized per model. Each configuration
ran **3 times in separate processes**; the recorded figure is the **median** of
`achieved tok/s` (decode rate). Every run is a cold process, so each includes its own
model load, code generation and warmup.

**Deviation from `verification-gates.md` §Benchmark gate:** that procedure specifies 3
discarded warm-ups plus 5 measured runs. This baseline uses 3 measured runs and no
discarded warm-ups, to keep a 28-configuration sweep tractable. Worst observed spread
`(max-min)/median` was **6.24%**, and all 24 successful configurations came in under the
gate's 10% "unstable environment" threshold, so the medians are usable as baselines.
Re-measure with the full 3+5 procedure when the M1.7 gate script lands.

## Results — decode tok/s (median of 3)

| Family | Model | Quant | CUDA | OpenCL | OpenCL/CUDA |
| --- | --- | --- | ---: | ---: | ---: |
| LLAMA_3 | Llama-3.2-1B-Instruct | F16 | 85.85 | 73.54 | 0.86x |
| LLAMA_3 | Llama-3.2-1B-Instruct | Q8_0 | 91.87 | 79.22 | 0.86x |
| QWEN_3 | Qwen3-0.6B | F16 | 66.14 | 49.74 | 0.75x |
| QWEN_3 | Qwen3-0.6B | Q8_0 | 68.69 | 51.52 | 0.75x |
| QWEN_2 | Qwen2.5-0.5B-Instruct | F16 | 61.36 | 57.16 | 0.93x |
| QWEN_2 | Qwen2.5-0.5B-Instruct | Q8_0 | 63.85 | 59.66 | 0.93x |
| GRANITE | granite-4.0-1b | F16 | 33.16 | 30.30 | 0.91x |
| GRANITE | granite-4.0-1b | Q8_0 | 34.20 | 32.26 | 0.94x |
| PHI_3 | Phi-3-mini-4k-instruct | F16 | 40.37 | 32.69 | 0.81x |
| PHI_3 | Phi-3-mini-4k-instruct | Q8_0 | 44.45 | 36.22 | 0.81x |
| MISTRAL | Mistral-7B-Instruct-v0.3 | F16 | 13.64 | 10.63 | 0.78x |
| MISTRAL | Mistral-7B-Instruct-v0.3 | Q8_0 | 14.41 | 11.38 | 0.79x |
| DEEPSEEK_R1_DISTILL_QWEN | DeepSeek-R1-Distill-Qwen-1.5B | F16 | **FAIL** | **FAIL** | — |
| DEEPSEEK_R1_DISTILL_QWEN | DeepSeek-R1-Distill-Qwen-1.5B | Q8_0 | **FAIL** | **FAIL** | — |

CUDA beats OpenCL on every family, by 6%–25%. Q8_0 is faster than F16 everywhere.

## Coverage gaps

- **`DEVSTRAL_2`** — no local GGUF for this family; not measured.
- **`DEEPSEEK_R1_DISTILL_QWEN`** — **broken before any GPU work**, both backends, both
  quantizations, and independent of prompt (fails on `"Hi"`):

  ```
  java.util.NoSuchElementException: No value present
    at java.util.OptionalInt.orElseThrow(OptionalInt.java:247)
    at org.beehive.gpullama3.tokenizer.Qwen3Tokenizer.encodeChunk(Qwen3Tokenizer.java:222)
  ```

  `encodeChunk` resolves each raw character with
  `vocabulary.getIndex(String.valueOf((char) b)).orElseThrow()`; DeepSeek's byte-level BPE
  vocabulary has no single-character entries for plain ASCII, so the lookup is empty.
  Pre-existing and unrelated to the TornadoVM 5.2.0 bump — it is a tokenizer defect, not a
  backend one. Needs a fix before this family can be baselined or covered by goldens.

## Files

- `results.jsonl` — one record per configuration: median, all runs, and the
  `--verbose-init` breakdown (load / codegen / warmup / weight copy-in) in ms.
- `logs/` — raw stdout for every individual run.
