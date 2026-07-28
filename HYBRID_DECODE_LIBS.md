# Hybrid CUDA libraries for single-token decode — findings

Can the TornadoVM hybrid library API (cuBLAS / cuBLASLt / cuDNN / cuFFT library tasks
riding the task-graph pipeline) speed up **single-request, single-token** decode?
Measured on the CUDA backend, RTX 4090, TornadoVM 5.0.1-jdk21-dev, JDK 21;
Llama-3.2-1B FP16 and Mistral-7B FP16 (both verified coherent on every change).

**TL;DR: no.** The n=1 decode matvecs are memory-bound and the JIT matvec already runs
at ~95% of peak DRAM bandwidth — cuBLAS and cuBLASLt land on the *same* bandwidth wall
(within 0.6% per-kernel). Worse, decode is **launch-bound** (GPU busy only ~33–38% of
wall), and library tasks de-fuse TornadoVM's fused kernels, *adding* launches. The
single-token levers are CUDA graphs (+8–12%, `--cuda-graphs`) and kernel fusion — both
already in the code. Hybrid GEMM libraries pay off from n≳16: batch prefill (PR #127)
and batched decode (PR #129), where the same projections become compute-bound
tensor-core GEMMs.

## What was tried

`-Dllama.logitsLib={jit,gemmEx,lt}` switches the biggest single matvec — the logits
vocabulary projection (`Wcls(vocab×dim)·x`, 525 MB FP16 read/token on Llama-1B = ~23%
of all decode bytes) — between:

- `jit` — stock `matrixVectorGeneric` (TornadoVM JIT).
- `gemmEx` — `CuBlas::cublasGemmExFP16FP32` library task (FP16 in, FP32 out;
  `gemm(OP_T, OP_N, m=vocab, n=1, k=dim)` — cuBLAS is column-major, row-major `Wcls`
  is a `dim×vocab` column-major matrix, `OP_T` recovers it).
- `lt` — `CuBlasLt::ltMatmulFP16` (heuristic kernel selection) + a 1.2 µs
  `halfToFloat` copy task (Lt FP16 writes half; the sampler reads FP32).

All three produce coherent output; library tasks run in the same task graph, on the
same stream, and are CUDA-graph-capturable.

## Per-kernel truth (nsys, 101 logits calls, Llama-1B)

| logits GEMV 128256×2048 FP16 | avg / call | effective BW |
|------------------------------|-----------:|-------------:|
| JIT `matrixVectorGeneric` | 553.8 µs | ~948 GB/s (94% peak) |
| cuBLAS `gemvx::kernel` (gemmEx) | 550.6 µs | ~953 GB/s |
| cuBLASLt `gemvx::kernel` (+halfToFloat 1.2 µs) | 550.9 µs | ~953 GB/s |

Both libraries dispatch a GEMV-specialized `internal::gemvx::kernel` — 0.6% faster than
the JIT kernel, i.e. **parity**. There is nothing to win: the kernel is a
DRAM-bandwidth measurement device.

## End-to-end (RTX 4090, 3 runs each, tok/s)

| Llama-3.2-1B FP16 | graphs off | graphs on |
|--------------------|-----------|-----------|
| jit | 88.7–91.3 | 98.1–102.4 |
| lt | 88.3–90.5 | 99.9–102.2 |
| gemmEx | 92.1 | 105.3 |

| Mistral-7B FP16 (graphs on) | tok/s |
|------------------------------|------:|
| jit | 23.5 |
| lt | 23.7 |

Statistical ties everywhere (run-to-run spread ±2% exceeds any config delta).
CUDA graphs, by contrast, are worth +8–12% on every config.

## Why libraries can't win at n=1 (and what does)

Decode kernel profile (nsys, Llama-1B, % of GPU time): `fusedRmsNormFFNGateUp` 36.6%
(805 GB/s), `matrixVectorGenericWithResidual` (wo, w2+residual) 21.8%,
`processHeadsFlashAttention` 16.4%, logits GEMV 15.2% (948 GB/s), `fusedQKVMatmulX`
6.7% (823 GB/s). Every projection is a bandwidth-bound GEMV at 80–95% of peak.

1. **Bandwidth wall.** At n=1 each projection reads its full weight matrix for one
   output vector (arithmetic intensity ~1). Tensor cores never engage; the best any
   kernel can do is stream weights at DRAM speed — which JIT, cuBLAS, and cuBLASLt all
   already do.
2. **Launch-bound regime.** GPU busy is ~3.6 ms/token vs ~11.2 ms wall (no graphs) —
   ~65 µs of host overhead per kernel launch. TornadoVM's fused JIT kernels
   (rms+gate+up in one, qkv in one, matvec+residual in one) exist to *remove*
   launches. Replacing `fusedRmsNormFFNGateUp` with library GEMVs would de-fuse it
   into ≥3 launches per layer (48/step on Llama-1B) to save ~13 µs/layer of GPU time —
   a net loss even under graph replay.
3. **Attention** (16%) is already a fused flash kernel; cuDNN `sdpaForward` was
   assessed earlier — no single-request win (pays off for batched/ragged attention).

**When hybrid libraries do win:** n≥16 turns the same projections into compute-bound
GEMMs — batch prefill via tensor-core MMA (PR #127), batched/continuous decode at
200–4200 tok/s aggregate (PR #129), and cuBLAS/cuBLASLt GEMMs generally. The library
integration itself is sound: correct results, same stream, graph-capturable — it is
the n=1 arithmetic intensity that leaves no room.

## Reproduce

```bash
# build (JDK 21, TornadoVM 5.0.1-jdk21-dev CUDA backend with tornado-cublas)
mvn -Pjdk21 -Dtornadovm.base.version=5.0.1 -Djdk.version.suffix=-jdk21-dev clean package -DskipTests

# stock vs library logits (verify text is coherent every run)
JAVA_TOOL_OPTIONS="-Dllama.logitsLib=jit"    llama-tornado --gpu --cuda [--cuda-graphs] --model <fp16.gguf> --prompt "..." -n 200 --instruct
JAVA_TOOL_OPTIONS="-Dllama.logitsLib=lt"     ...
JAVA_TOOL_OPTIONS="-Dllama.logitsLib=gemmEx" ...

# per-kernel comparison
nsys profile -o out --trace=cuda python3 llama-tornado --gpu --cuda ... 
nsys stats --report cuda_gpu_kern_sum out.nsys-rep   # matrixVectorGeneric vs gemvx::kernel
```
