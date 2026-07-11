# Static batched decode (LLaMA FP16)

Decode **B independent sequences at once**, one token per step, each attending its
**own** KV cache. Turns the bandwidth-bound single-token matvecs of decode into
compute-bound tensor-core GEMMs (one weight read amortized across B tokens), so
aggregate throughput scales ~linearly with B until the 128-row MMA tile fills.

Measured **41× aggregate throughput** vs single-stream decode on an RTX 4090
(Llama-3.2-1B FP16), output verified coherent + bit-exact against the single-stream
greedy reference.

---

## Why batching wins decode

Single-token decode is **memory-bound**: every projection reads the full weight
matrix to produce one output vector (arithmetic intensity ~1). The GPU stalls on
HBM bandwidth; tensor cores sit idle.

Batch B tokens that share the same weights and each weight row is read **once** and
applied to **all B** tokens — arithmetic intensity ~B, i.e. a GEMM. Past B≈16 the
projections become compute-bound and the tensor cores engage. Attention does *not*
batch this way (each sequence has its own KV), but it is a small fraction of the
decode FLOPs, so the projection win dominates.

```
             single-token decode              static batched decode (B)
             ────────────────────             ─────────────────────────
  weight W    read once  ─┐                     read once ─┐
  token x0    ───────────┼─> y0                 ──────────┼─> y0
  token x1    (next step: read W again)         ──────────┼─> y1     one GEMM,
  token x2    (read W again) ...                ──────────┼─> y2     W read once
   ...                                          ──────────┘  ...
  intensity ~1  (HBM-bound)                     intensity ~B (compute-bound)
```

---

## Design

### Engine loop (`bench/BatchedDecodeEngine.java`)

One step routine serves **both** phases. Prompt tokens are fed with logits
discarded — this fills every slot's KV region with the *same* RoPE the decode step
uses, so there is no prefill/decode cache mismatch. Then generated tokens feed back.

```
  prompt "…paint."   ┌──────────────────────────────────────────────┐
        │            │  for each step:                              │
        ▼            │    host: write B token embeddings → embXBatch │
  ┌───────────┐      │          write positions → seqPositions[B]    │
  │  PREFILL  │      │    GPU:  activation → L0 → L1 → … → L15 → logit│
  │ P tokens  │──────│    host: sample B rows of logitsBatch         │
  │(no logits)│      └──────────────────────────────────────────────┘
  └───────────┘                        │
        │                              ▼
        ▼                      ┌────────────────┐
  ┌───────────┐   per step     │  B next tokens │  greedy → all B identical
  │  DECODE   │───────────────>│  (one / slot)  │  temp>0 → B divergent streams
  │ N steps   │                └────────────────┘
  └───────────┘
```

### Per-step task-graph pipeline (2 + N graphs)

```
 embXBatch(FP16 B×dim)
      │  [0] prefillActivation      FP16 → FP32  wrapXBatch
      ▼
 ┌─────────────────  [1..N] batchDecodeLayer_i  (12 tasks, all GEMMs on MMA) ─────────────────┐
 │  batch_attn_rms → batch_attn_rms_apply → qkvProj(MMA) → batch_rope_kv* → batch_attention*  │
 │  → woProj(MMA) → batch_ffn_rms → batch_ffn_rms_apply → gateUpProj(MMA) → swiglu            │
 │  → w2Proj(MMA) → w2Resid                                                                    │
 │      (*) the only decode-specific kernels: per-slot position + per-slot KV base            │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
      │  wrapXBatch (B×dim, persists to next layer)
      ▼
 [N+1] batchLogits    final RMS → gemmMMA over vocab(128256) → logitsBatch (B×vocab, FP32)
```

The 12-task layer pipeline is **identical** to the existing batched-prefill MMA
layer (`LlamaFP16LayersBatchPrefillMMA`). Only two kernels change, and only in KV
addressing:

| kernel | prefill (one sequence) | decode (B sequences) |
|--------|------------------------|----------------------|
| RoPE + KV write | `batchedRopeWithKVCachePacked` — `pos = start + b`, one shared cache | `batchedDecodeRopeWithKVCachePacked` — `pos = seqPositions[b]`, per-slot base |
| flash attention | `batchedFlashAttentionFP16Out` — shared causal KV | `batchedDecodeAttentionFP16Out` — own KV region per slot |

The math (RoPE rotation, online-softmax flash attention, register-partitioned P·V)
is untouched; both forks only change the index into the KV cache.

### KV cache layout (B-sized)

Each slot owns a contiguous region; `batchIdx` stride = `numLayers·ctx·kvDim`.

```
 keyCacheBatch  [ slot0 | slot1 | … | slot B-1 ]      size = B · numLayers · ctx · kvDim
                    │
                    └─ [ layer0 | layer1 | … ]         stride numLayers·ctx·kvDim
                          │
                          └─ [ pos0 | pos1 | … ]       row = pos·kvDim, written at seqPositions[slot]
```

Because the engine (not `State`) owns this buffer, `ctx` is capped independently
(default 512) to keep `B·L·ctx·kvDim` in VRAM.

### Static batching

All slots advance one position per step, so `seqPositions` is the same scalar for
every slot at every step. The load-bearing part is the per-slot KV **base** address,
not the position — the `IntArray seqPositions` keeps the door open for ragged /
continuous batching later.

---

## Correctness

Greedy sampling with B copies of the same prompt → **all B streams are bit-exact
identical AND equal to the single-stream greedy reference**. Each slot independently
reproduces the reference, so the per-slot forward (including per-slot KV) is correct.

```
[verify] mode=greedy: all 128 streams identical (== single-stream greedy ref): true
```

Temperature sampling with a per-slot RNG → **B distinct, individually coherent**
continuations, proving the KV regions are genuinely independent (real concurrent
requests, not replication):

```
[verify] mode=sample temp=0.70: 128/128 streams distinct (independent per-slot KV)
  slot 0: … a robot named Zeta stood at a workbench …
  slot 1: … Zeta sat in the studio, its mechanical arms splayed …
  slot 2: … a robot named Zeta whirred to life …
```

---

## Performance

RTX 4090, Llama-3.2-1B FP16, `ctx=512`, 64 decode steps, CUDA graphs on,
steady-state (capture steps excluded).

| config | aggregate tok/s | vs single-stream | per-step latency |
|-------:|----------------:|-----------------:|-----------------:|
| single-stream (stock decode) | 101 | 1.0× | — |
| batched B=32  | 1085 | 10.7× | 29.5 ms |
| batched B=64  | 2167 | 21.5× | 29.5 ms |
| batched B=128 | **4175** | **41.3×** | 30.7 ms |

CUDA graphs (`-Dbatch.decode.cudaGraphs`, default on) cut a uniform ~6% off per-step
(dispatch overhead across the 2+N graphs).

**Operating point.** Per-step latency is ~flat (29–31 ms) from B=8 to B=128: every
step runs at `paddedBatch = 128` (the MMA tile is 128 rows) and the logits GEMM spans
the full 128256 vocab regardless of B. So B<128 does the same GPU work as B=128
(wasted pad); aggregate scales ~linearly up to B=128, then per-step grows.
**B=128 is the efficient operating point.** The residual per-step cost is real GEMM
compute — full-vocab logits (~33 GFLOP) + 16 layer projections at 128 rows — not
launch overhead, which is why CUDA graphs help only marginally.

---

## Reproduce

### 1. TornadoVM (CUDA backend + tensor-core MMA)

Requires a TornadoVM with the **CUDA/PTX backend** and the **MMA `KernelContext`**
API (`mmaFragment` / `mmaLoadA` / `mmaMultiply`, `HalfFloatArray`) — the same
TornadoVM the `feat/mma_cuda` GPULlama3 branch already needs. Tested against
**TornadoVM 5.0.1-jdk21-dev** built with the PTX backend on **JDK 21**.

```bash
git clone https://github.com/beehive-lab/TornadoVM
cd TornadoVM
# JDK 21; build the CUDA/PTX backend (installs artifacts into ~/.m2)
make BACKEND=ptx
export TORNADOVM_HOME=$PWD/dist/tornadovm-*-cuda
```

### 2. GPULlama3 (this branch)

```bash
# JDK 21
mvn -Pjdk21 -Dtornadovm.base.version=5.0.1 -Djdk.version.suffix=-jdk21-dev \
    clean package -DskipTests
```

### 3. Run the engine

`-Dllama.prefillBatchSize` MUST equal `-Dbatch.decode.B` (it sizes the batch
activation buffers). Launch `org.beehive.gpullama3.bench.BatchedDecodeEngine` on the
standard TornadoVM module path (easiest: take `llama-tornado --show-command …`,
swap the main class to the engine, and prepend the `-D` flags):

```
-Dllama.prefillBatchSize=128     # batch buffers = B
-Dbatch.decode.B=128             # concurrent sequences
-Dbatch.decode.ctx=512           # per-slot KV context cap (VRAM)
-Dbatch.decode.n=64              # decode steps
-Dbatch.decode.temp=0.0          # 0 = greedy (bit-exact verify); >0 = divergent streams
-Dbatch.decode.cudaGraphs=true   # CUDA-graph capture/replay
org.beehive.gpullama3.bench.BatchedDecodeEngine -m <llama-3.2-1b-fp16.gguf> -p "…" --instruct
```

Two supporting micro-benchmarks (synthetic dims, no model load) isolate the two
regimes: `bench/BatchedProjectionBench` (compute-bound projection crossover, ~20×)
and `bench/BatchedDecodeAttentionBench` (memory-bound per-slot attention, ~2×,
bit-exact vs a CPU reference).

---

## Limitations / next

- **FP16 LLaMA only.** Q8_0 and other architectures reuse the same pattern but need
  their own decode layer graph. Qwen3 is blocked upstream (empty output on this
  TornadoVM build) independent of this work.
- **Same prompt length across slots** (static batching). Ragged prompts / iteration-
  level (continuous) batching are the follow-up — the per-slot `seqPositions` +
  per-slot KV base already support per-slot positions.
- **Logits GEMM over the full vocab every step** is ~half the per-step cost; only
  needed for slots actually sampling.
