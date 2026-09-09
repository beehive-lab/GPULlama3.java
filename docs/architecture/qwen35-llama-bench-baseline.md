# `qwen35` — matched llama-bench baseline

Both engines measured with the llama-bench protocol and its output schema, so the numbers are
comparable row for row. GPULlama CLI metrics (`prompt eval`, `tok/s` from a chat run) are *not*
comparable with llama.cpp `llama-bench` results and are not used here.

## 1. What was measured

| | |
| --- | --- |
| Model | `Qwen3.8-27B-Q4_0.gguf` |
| sha256 | `ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d` |
| Reported by both | qwen35, `Q4_0` (`general.file_type = 2`), 27.32 B params, 14.94 GiB |
| Device | NVIDIA GeForce RTX 5090 Laptop GPU (24 GB), driver 580.142, CUDA 13.1 |
| GPULlama3.java | `feat/qwen3-8`, TornadoVM 6.0.0-jdk21-cuda, JDK 21, full GPU offload |
| llama.cpp | worktree at `e2d2c0d6a`, `-DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120`, `-ngl 99` |
| Cases | `pp64`, `pp381`, `pp512`, `pp1024`, `tg128` |
| Repetitions | 1 untimed warm-up + 5 measured, per case, per width |
| Sampling | none — both harnesses time the forward pass, greedy/non-stochastic |

Commands, verbatim:

```bash
# llama.cpp, once per microbatch width w in 8 16 32 64 512
./build/bin/llama-bench -m Qwen3.8-27B-Q4_0.gguf \
  -p 64,381,512,1024 -n 128 -b $w -ub $w -ngl 99 -r 5 -o json

# GPULlama3.java, once per batch width w in 1 8 16 32 64
./llama-tornado --gpu --cuda --gpu-memory 22GB --model Qwen3.8-27B-Q4_0.gguf --bench \
  --bench-args="-p 64,381,512,1024 -n 128 -b $w -r 5 -o json \
                --expect qwen35/Q4_0/BATCH_PREFILL_DECODE"
```

`--expect` is an assertion, not a label: the benchmark reads the architecture and
`general.file_type` out of the GGUF header and the execution mode off the plan object it built,
and refuses to report numbers under a selection it did not make. Every batched row below carries
`qwen35 / Q4_0 / BATCH_PREFILL_DECODE`; the `b1` rows carry `qwen35 / Q4_0 / STANDARD`.

Raw machine-readable rows, five samples each, are in
[`perf-results/qwen35-llamabench/`](../../perf-results/qwen35-llamabench).

## 2. Results, median of five (tok/s)

### GPULlama3.java

| batch | pp64 | pp381 | pp512 | pp1024 | tg128 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (`STANDARD`) | 14.21 | 12.26 | 11.56 | 9.31 | 13.05 |
| 8 | 24.01 | 21.35 | 20.73 | 18.02 | 13.16 |
| 16 | 24.94 | 22.85 | 22.18 | 20.40 | 13.03 |
| 32 | 25.11 | 22.92 | 22.22 | 20.61 | 13.04 |
| 64 | 25.00 | 22.96 | 22.54 | 20.71 | 13.20 |

Spread is small: the largest standard deviation in the whole matrix is 0.28 t/s.

### llama.cpp

| ubatch | pp64 | pp381 | pp512 | pp1024 | tg128 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 201.37 | 201.58 | 202.12 | 199.01 | 42.71 |
| 16 | 425.08 | 442.02 | 448.05 | 447.51 | 42.61 |
| 32 | 677.15 | 696.66 | 707.14 | 706.69 | 42.50 |
| 64 | 1059.33 | 1012.62 | 1034.33 | 1039.29 | 42.50 |
| 512 | 1061.20 | 1511.18 | 1542.88 | 1509.11 | 42.54 |

### Ratio, best validated configuration on each side

| case | GPULlama3.java | llama.cpp | ratio |
| --- | ---: | ---: | ---: |
| pp64 | 25.11 (b32) | 1061.20 (ub512) | 0.024x |
| pp381 | 22.96 (b64) | 1511.18 (ub512) | 0.015x |
| pp512 | 22.54 (b64) | 1542.88 (ub512) | 0.015x |
| pp1024 | 20.71 (b64) | 1509.11 (ub512) | 0.014x |
| tg128 | 13.20 (b64) | 42.71 (ub8) | 0.309x |

At the *same* width the gap is smaller but the same shape: at 32, pp381 is 22.92 against 696.66
(0.033x).

## 3. What the widths say

**GPULlama saturates at 16.** 1 → 8 is the batching win (1.74x at pp381); 8 → 16 adds 7%; 16 → 64
adds nothing outside noise. Nothing about memory, correctness or throughput forbids a wider chunk —
the curve is simply flat, so 64 is the boundary worth reporting rather than a limit that was hit.
Generation is unaffected by the batch width, as it should be: `tg` is single-token decode in every
mode.

**llama.cpp does not saturate until far higher, and for a reason that is visible in its source.**
`MMVQ_MAX_BATCH_SIZE` is 8: at or below eight columns a quantized matmul runs the matrix-*vector*
kernel, and above it llama.cpp switches to MMQ, which quantizes the activations to Q8_1 and runs
int8 tensor-core MMA over a tile of rows. That is the 201 → 442 step between ub 8 and 16, and it
keeps paying to 512.

GPULlama has no such second regime. Every batched projection in `BATCH_PREFILL_DECODE` is a
matrix-vector kernel with a row tile bolted on, so its `pp` curve flattens where the tile stops
reducing weight traffic. **llama.cpp's *matrix-vector* configuration — ub 8, 201 t/s — is still
8x GPULlama's best.** The gap is therefore not "we lack tensor cores"; it is present before tensor
cores enter the picture.

## 4. Phases kept separate

`llama-bench` and this harness both time the forward pass only. Model load, plan construction, JIT
compilation and weight copy-in are outside every number above: the harness builds the plan, runs an
untimed warm-up repetition through the same path, and only then starts the clock. Prompt processing
and token generation are separate cases (`pp`/`tg`) and are never averaged together.

## 5. After the optimization work

Same protocol, same commands, after the tiling, output-row and attention commits. Medians of five.

| batch | pp64 | pp381 | pp512 | pp1024 | tg128 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (`STANDARD`) | 14.68 | 12.56 | 11.87 | 9.84 | 13.89 |
| 8 | 49.83 | 48.22 | 47.97 | 46.62 | 13.82 |
| 16 | 57.68 | 55.83 | 55.62 | 54.23 | 13.83 |
| 32 | 60.98 | 59.15 | 58.98 | 57.35 | 13.83 |
| 64 | 62.21 | 60.54 | 60.35 | 58.80 | 13.84 |

### Ratio, GPULlama3.java / llama.cpp

At matched width — GPULlama's `-b` against llama.cpp's `-ub`:

| width | pp64 | pp381 | pp512 | pp1024 | tg128 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.247 | 0.239 | 0.237 | 0.234 | 0.324 |
| 16 | 0.136 | 0.126 | 0.124 | 0.121 | 0.325 |
| 32 | 0.090 | 0.085 | 0.083 | 0.081 | 0.325 |
| 64 | 0.059 | 0.060 | 0.058 | 0.057 | 0.326 |

At each side's best configuration:

| case | llama.cpp | before | after | ratio | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| pp64 | 1061.20 (ub512) | 25.11 | 62.21 (b64) | 0.024 -> 0.059 | 2.48x |
| pp381 | 1511.18 (ub512) | 22.96 | 60.54 (b64) | 0.015 -> 0.040 | 2.64x |
| pp512 | 1542.88 (ub512) | 22.54 | 60.35 (b64) | 0.015 -> 0.039 | 2.68x |
| pp1024 | 1509.11 (ub512) | 20.71 | 58.80 (b64) | 0.014 -> 0.039 | 2.84x |
| tg128 | 42.71 (ub8) | 13.20 | 13.89 (b1) | 0.309 -> 0.325 | 1.05x |

Two things are worth reading off this. Prompt throughput now barely degrades with prompt length —
62.21 to 58.80 between 64 and 1024 tokens, where it used to fall from 25.11 to 20.71 — because the
attention work that grew with the attended range was mostly redundant. And the best matched-width
ratio is at **width 8**, 0.24, which is the one width where llama.cpp is also still running a
matrix-vector kernel. Every wider comparison is our matrix-vector against its int8 tensor cores.

## 6. CUDA graphs

Already implemented for this mode. Measured at a chunk of 32, medians of three:

| case | graphs off | graphs on |
| --- | ---: | ---: |
| pp381 | 59.93 | 59.65 |
| pp1024 | 57.71 | 57.78 |
| tg128 | 13.88 | **14.53** |
| tg256, with FP16 key/value as well | 13.14 | **14.34** |

**Prompt processing does not move; generation gains 4.7%, and 9.1% with FP16 key/value alongside
it.** The profile says why: `cuLaunchKernel` is 0.3% of prefill kernel time, because a chunk of 32
tokens amortizes about a thousand launches across 32 tokens. In decode the same thousand launches
serve one token, so the overhead is some thirty times more significant, and that is the part a
captured graph removes.

## 7. Not run

- OpenCL and Metal. CUDA only; no claim is made about either.
- `-d` context-depth cases.
- MTP on either side.
