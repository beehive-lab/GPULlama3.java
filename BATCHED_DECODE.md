# Static batched decode (LLaMA / Qwen3 FP16)

Decode **B independent sequences at once**, one token per step, each attending its
**own** KV cache. Turns the bandwidth-bound single-token matvecs of decode into
compute-bound tensor-core GEMMs (one weight read amortized across B tokens), so
aggregate throughput scales ~linearly with B until the 128-row MMA tile fills.

Measured up to **41× aggregate throughput** vs single-stream decode on an RTX 4090
(Llama-3.2-1B FP16), output verified coherent + bit-exact against the single-stream
greedy reference. Runs on the TornadoVM **CUDA backend** (tensor-core MMA).
Both **LLaMA** and **Qwen3** are supported (Qwen3 adds per-head Q/K RMS norm + split-
half RoPE).

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

**Qwen3** reuses the same pattern: `Qwen3FP16LayersBatchDecodeMMA` keeps the per-head
Q/K RMS-norm task (`batchedFusedQKRmsNormPacked`, per-token — reused as-is) and swaps in
`batchedDecodeRopeWithKVCacheQwen3Packed` (split-half RoPE, per-slot) +
`batchedDecodeAttentionFP16Out` (the same per-slot flash kernel, parameterized by
`qDim`/`nEmbdHead`/`gqa`). The engine dispatches on model type; embeddings and the
batched-logits GEMM are model-agnostic.

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

RTX 4090, TornadoVM **CUDA backend**, FP16, `ctx=512`, CUDA graphs on, steady-state
(capture steps excluded).

**Llama-3.2-1B** (16 layers, vocab 128256):

| config | aggregate tok/s | vs single-stream | per-step latency |
|-------:|----------------:|-----------------:|-----------------:|
| single-stream (stock decode) | 101 | 1.0× | — |
| batched B=32  | 1085 | 10.7× | 29.5 ms |
| batched B=64  | 2167 | 21.5× | 29.5 ms |
| batched B=128 | **4175** | **41.3×** | 30.7 ms |

CUDA graphs (`-Dbatch.decode.cudaGraphs`, default on) cut a uniform ~6% off per-step
(dispatch overhead across the 2+N graphs).

**Across models** (largest B that fits 24 GB; deeper/wider → higher per-step, so B is
capped lower):

| model | layers · dim · vocab | single-stream | batched (B) | aggregate | speedup |
|-------|----------------------|--------------:|:-----------:|----------:|--------:|
| Llama-3.2-1B | 16 · 2048 · 128256 | 101 tok/s | 128 | 4175 tok/s | 41× |
| Qwen3-1.7B   | 28 · 2048 · 151936 |  48 tok/s |  64 | 1433 tok/s | 30× |
| Qwen3-4B     | 36 · 2560 · 151936 |  39 tok/s |  32 |  405 tok/s | 10× |

All bit-exact vs the single-stream greedy reference (`all B streams identical: true`)
and coherent.

**Operating point.** Per-step latency is ~flat (29–31 ms) from B=8 to B=128: every
step runs at `paddedBatch = 128` (the MMA tile is 128 rows) and the logits GEMM spans
the full 128256 vocab regardless of B. So B<128 does the same GPU work as B=128
(wasted pad); aggregate scales ~linearly up to B=128, then per-step grows.
**B=128 is the efficient operating point.** The residual per-step cost is real GEMM
compute — full-vocab logits (~33 GFLOP) + 16 layer projections at 128 rows — not
launch overhead, which is why CUDA graphs help only marginally.

---

## Continuous batching (iteration-level scheduling)

`-Dbatch.decode.continuous=true` runs an Orca-style scheduler over the same B slots:
each slot is independently either **prefilling** its prompt (token-by-token, logits
ignored) or **decoding**; a slot that hits a stop token / its max-gen is evicted and
**immediately refilled** from a pending queue — new requests join a partly-decoded
batch mid-flight, so no slot waits for the slowest in a wave. This works with zero
kernel changes because the per-step forward already feeds one token per slot at its own
`seqPositions[b]` with its own KV region — prefill and decode are the same op.

Same 512-request workload (Llama-1B, B=128, prompt 22 tok, max-gen ∈ [8,64], greedy —
so identical token counts and all outputs mutually prefix-consistent):

| scheduling | steps | gen tok/s | requests/s | slot utilization |
|------------|------:|----------:|-----------:|-----------------:|
| static wave (`refill=false`) | 336 | 1645 | 46.9 | 66.4% |
| **continuous** (`refill=true`) | **272** | **1972** | **56.2** | **82.2%** |

**+20% throughput / +24% relative utilization** by refilling freed slots instead of
draining each wave to its longest request. The gap widens with more length variance.
Correctness under scheduling = all completed outputs are mutually prefix-consistent.

## PagedAttention (`-Dbatch.decode.paged=true`, LLaMA)

The contiguous KV cache reserves the full `ctx` per slot — `B·ctx` token-slots even
when most sequences are short. Paging stores KV in a **shared pool of fixed-size blocks**
(`blockSize` positions each) addressed through a per-slot **block table**; the two decode
KV kernels index `blockTable[b][pos/blockSize]` instead of `pos·kvDim`. Blocks are
allocated on demand as a sequence grows and returned to the pool on eviction, so the pool
is sized to **actual concurrent demand**, not the worst case.

Same 512-request continuous workload (Llama-1B, B=128, ctx=512, blockSize=16):

| KV cache | pool | peak used | throughput | correct |
|----------|-----:|----------:|-----------:|:-------:|
| contiguous | 4096 MB (=B·ctx reservation) | — | 1972 tok/s | ✓ |
| paged, 768-block pool | 768 MB | 372 blk (48%) | 1971 tok/s | ✓ |
| paged, 384-block pool | 384 MB | 372 blk (97%) | 1939 tok/s | ✓ |

**~10.7× less KV memory** (384 MB vs 4096 MB) at ~1% throughput overhead, output still
bit-exact. The pool floor is the peak concurrent block demand (~372 here); undersizing it
throws a clear `KV block pool exhausted` (the point where admission control / backpressure
takes over — the vLLM behavior). One reserved *scratch* block absorbs the KV writes of
inactive slots (they still execute the kernels each step) so they never corrupt a live
block. Paging is the prerequisite for prefix caching (share a prompt's blocks across
requests via the block table + refcounting).

## Prefix caching (`-Dbatch.decode.prefixCache=true`, needs paging)

When requests share a common prompt prefix (a system prompt, a few-shot preamble), its KV
is identical across all of them. Prefix caching prefills the block-aligned shared prefix
**once** into pinned blocks; every request points its block-table prefix rows at those
shared blocks and starts decoding at `pos = sharedPrefixLen`, skipping the prefix's prefill
entirely. The attention kernel walks the block table, so it reads shared-then-private blocks
transparently — **no kernel change**, just scheduling + block-table setup.

Same 512-request continuous+paged workload, 48-token shared prompt (Llama-1B, B=128):

| | steps | gen tok/s | requests/s | prefill tokens |
|--|------:|----------:|-----------:|---------------:|
| no prefix cache   | 419 | 1307 | 37.2 | 28672 |
| **prefix cache**  | **211** | **2422** | **69.0** | 4096 (**85.7% saved**) |

**~2× fewer steps, +85% throughput** — the 48-token prefix is prefilled once (3 pinned
blocks) instead of 512×, and its KV blocks are shared. Output still bit-exact (all completed
requests mutually prefix-consistent). The win scales with prefix length / request count. The
prefix is block-aligned down (`sharedPrefixLen = ⌊(promptLen-1)/blockSize⌋·blockSize`) so a
request only ever writes into its own private blocks; shared blocks stay read-only.

## Reproduce

### 1. TornadoVM (CUDA backend + tensor-core MMA)

Requires a TornadoVM with the **CUDA backend** and the **MMA `KernelContext`** API
(`mmaFragment` / `mmaLoadA` / `mmaMultiply`, `HalfFloatArray`) — the same TornadoVM the
`feat/mma_cuda` GPULlama3 branch already needs. Tested against **TornadoVM
5.0.1-jdk21-dev**, CUDA backend, **JDK 21**. (All numbers here are from the CUDA
backend.)

```bash
git clone https://github.com/beehive-lab/TornadoVM
cd TornadoVM
# JDK 21; build the CUDA backend (installs artifacts into ~/.m2)
make BACKEND=cuda
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

- **FP16 LLaMA + Qwen3.** Q8_0 and other architectures reuse the same pattern but need
  their own decode layer graph. Qwen3-8B produces garbage in the **stock** decode path
  on this build (independent of batched decode — the engine faithfully reproduces the
  stock output); Qwen3-1.7B/4B are fine.
- **Same prompt length across slots** (static batching). Ragged prompts / iteration-
  level (continuous) batching are the follow-up — the per-slot `seqPositions` +
  per-slot KV base already support per-slot positions.
- **Logits GEMM over the full vocab every step** is ~half the per-step cost; only
  needed for slots actually sampling.
