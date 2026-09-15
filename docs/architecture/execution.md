# Execution

What happens between `LocalModels.load(...)` and a token appearing: loading, sessions,
programs, backend lowering, caching, execution modes and sampling.

## Loading a model

```
GGUF file ──> GgufRecognition ──> ArchitectureId
                                       │
                                       v
                             ModelProvider (ServiceLoader)
                                       │
                    ┌──────────────────┼──────────────────┐
                    v                  v                  v
              Configuration        Weights            Tokenizer
                                                    + ChatFormat
                                       │
                                       v
                                  LocalModel
```

Recognition reads the file's declared `general.architecture`, and its `general.name` only
where the format leaves no other signal — several families ship as `llama`, and a
DeepSeek distill ships as `qwen2`. An architecture no provider claims is reported as
unrecognized. It is never mapped to the nearest thing, because a near-miss family loads
and then produces wrong numbers.

Weights are materialized in the representation the chosen backend executes, not the file
representation. Which representations a backend accepts, and where a K-quant is decoded,
is a backend concern — see [`models-and-backends.md`](models-and-backends.md).

A load is preceded by a capacity prediction. `LocalModels.preflight(...)` returns a
`MemoryPlan` before anything is allocated, and a load known to exceed capacity fails with
`GPUL-MEM-001` before the first device buffer, rather than dying part-allocated.

## Sessions

A session is one in-progress sequence. `model.newSession(...)` creates it; many sessions
may share one loaded model. A session owns its position, its KV lease, its sampler and its
conversation; it owns no weights and no device workspace.

`reset()` starts the conversation over and keeps the session's storage. `close()` is
idempotent, and using a closed session — or a session whose model has been closed — throws
with `GPUL-LIFE-001`. Ownership and close ordering are in
[`memory-and-concurrency.md`](memory-and-concurrency.md).

## Logical programs

An inference program is a backend-neutral description of one forward pass: an ordered list
of program components built from a shared operation vocabulary, over named weights,
producing named results. It is data. It holds no device handles and no task graphs, and it
is not a graph IR — components are ordered, and there is no optimizer between the
description and the backend.

The vocabulary is fixed and small: embedding lookup, RMSNorm, RoPE, matrix–vector and
matrix–matrix multiply, attention, KV append, residual add, SwiGLU, GeGLU, split of a
fused QKV or gate/up projection, MoE routing and expert feed-forward, logit soft-cap, bias
add, scale, softmax, vocabulary projection, argmax and sampling. Adding a family means
composing these; it does not mean adding a kernel body. A family that genuinely needs new
arithmetic gets a new operation, defined once and implemented by every backend that claims
it — a numerical change disguised as a refactor is the failure mode this exists to prevent.

A program carries a **signature**: what it consumes, what it produces, and what mutable
state it reads and writes. Bindings fall into three kinds, and the distinction is
load-bearing:

| Binding | Lifetime | Example |
| --- | --- | --- |
| program-fixed | the compiled program's whole life | weights, KV pool, block table, activation and attention workspace |
| invocation value | one call | token id, position, block-table slot, active-request count, sampling parameters |
| host-visible result | one call | logits row, sampled token |

A program declares **phases**. `PREFILL` ingests prompt tokens and skips the vocabulary
projection, since logits are not needed for prompt positions except the last. `DECODE`
runs every component including projection and sampling. A single-token program simply has
one phase.

## Backend lowering

A backend turns a program into a compiled program. For the TornadoVM backend, **TornadoVM
is the compiler**: this project translates its program description into task graphs and
grid schedulers and hands them over. It does not build an IR and does not generate device
code.

Fusion, kernel selection and worker-grid sizing are the backend's decisions, made from the
program plus the device's capabilities. Nothing above the backend SPI expresses them, and
nothing above it can observe them except through metrics.

Lowering is gated by a qualification table rather than switched on wholesale, because a
lowered path that is merely *implemented* has not been shown to agree with the legacy one
on that family, dtype and mode. `jllm.lowering` selects between three answers:

| Mode | Behaviour |
| --- | --- |
| `auto` (default) | lower the combinations the qualification table names; select legacy for everything else, deliberately |
| `on` | lower an implemented combination even if it is not yet qualified — the evidence-gathering setting. An **unimplemented** combination throws rather than quietly running legacy |
| `off` | always legacy |

Under `auto` the qualified set today is **Llama / F16 / STANDARD**. Everything else selects
legacy. `on` throwing on an unimplemented combination is the point: a user who asked for
lowering and silently got the old path would measure the old path and record it as the new
one.

**No silent fallback**, generally. A device selector this build cannot honour throws
`UnsupportedOperationException`; an unimplemented lowering under `jllm.lowering=on` throws
`UnsupportedLoweringException`. A configuration is never quietly substituted for a
different one.

## Compiled programs and their cache

A compiled program is built once and invoked many times. It is never rebuilt per token,
and an invocation binds existing buffers rather than allocating.

The cache lives on the `LocalModel` and is keyed by everything that can change the
generated code or the buffers it addresses:

```
ProgramCacheKey = signature
                + backend
                + device
                + compile options
                + device capability fingerprint
                + binding domain
```

The **binding domain** is the physical KV pool, block table and captured workspace taken
together. Two sessions in the same domain may share a compiled program; an engine has its
own pool and therefore its own domain, so an engine and a standalone session never share
one, and neither do two engines. Equal shapes are not the same domain.

The cache is internally synchronized. Concurrent identical misses compile once; a failed
compilation removes the pending entry and is not cached. There is no public compile entry
point and no eviction.

`CompiledProgramIdentityAccelTest` asserts what "compile once" means in observable terms:
after warm-up, the task-graph count, the ordered task names, the grid-scheduler entries and
a SHA-256 over each task's generated kernel source are all unchanged, and no further
compilation occurs.

## Execution modes

`ExecutionMode` says how a turn is executed. It is policy, not phase structure.

| Mode | Prompt ingestion | Notes |
| --- | --- | --- |
| `STANDARD` | one token at a time | the default |
| `PREFILL_DECODE` | sequential prefill, then single-token decode | token-identical to `STANDARD` |
| `BATCH_PREFILL_DECODE` | a chunk of prompt tokens per invocation, then decode | **default-off**; see below |

Policy is resolved **once per generation**, never per token, and reaches the plan as an
`ExecutionPolicy` value rather than as process-global system properties. The properties
remain as defaults for the CLI; they are not read from inside the execution path.

**Batched prefill stays default-off on every backend**, and what keeps it off is
correctness, not cost. Its per-backend availability is in
[`models-and-backends.md`](models-and-backends.md).

### What is settled

Memory and speed no longer argue against it. Batched prefill used to hold the model's
weights twice on the device — a prefill graph and a decode graph per layer each bound the
same weight arrays — which put Llama-3.2-3B-F16 at 12463 MiB against 5531 MiB for
`STANDARD`, and left 8B FP16 models unable to allocate on a 24 GB card in a mode they ran
fine without. The decode graphs now consume the copy the prefill graph uploaded, and the
mode costs about 3% more device memory than `STANDARD` (7081 MiB at batch 128, 7153 MiB at
batch 512, against 6909 MiB). Decode throughput is unchanged (203.4 ± 2.0 against
203.9 ± 2.5 tok/s on Llama-3.2-1B Q8_0), and prefill is several times faster.

### What is not

`CpuGpuParity` now runs each family in both modes. Against the same CPU reference and the
same bounds, at batch 128:

| Fixture | `STANDARD` | `BATCH_PREFILL_DECODE` |
| --- | --- | --- |
| Llama-3.2-1B F16 | passes | passes |
| Llama-3.2-1B Q8_0 | passes | **0.76% of elements violate** (budget 0.01%), worst 8.25x tolerance |
| Qwen3-0.6B F16 | passes | **64 logits rows against the reference's 63** |
| Qwen3-0.6B Q8_0 | passes | **64 logits rows against the reference's 63** |

Two independent problems.

**Q8_0 batched prefill computes something different.** The batched Q8_0 projections
dequantize to FP16 and accumulate through tensor-core MMA; the single-token path
accumulates in FP32. The extra error is arithmetic rather than a defect, but it is far
outside the bounds the single-token path meets, so promoting the mode would change the
numbers every Q8_0 model produces.

**Qwen3 generates one more token in batched mode.** The two loops drifted before they were
merged — the batched one carries a generated-token budget the sequential one does not — and
that difference is now measurable as a row-count mismatch. A prompt should not produce a
different number of tokens because of how its prefill was scheduled, whatever the default
is.

### The options, and what each costs

For the Q8_0 accuracy gap:

1. **Accumulate the batched Q8_0 GEMM in FP32.** Keeps one set of bounds and one numerical
   story; gives back some of the tensor-core speed that motivated the mode.
2. **Give the batched Q8_0 path its own documented bounds**, the way
   `DeviceCapability.PACKED_HALF2_MATH` documents a narrower guarantee. Honest only if the
   looser bound is written down with its reason and measured, never widened to make a red
   test green.
3. **Keep batched prefill off for Q8_0 and on for FP16.** Precise, and the per-family
   support matrix already makes availability conditional, so the machinery exists — but it
   makes "which arithmetic ran" depend on the weight format, which is a thing callers then
   have to know.
4. **Leave it default-off for both.** No new evidence needed, and no gain.

For the Qwen3 row count, only one of these is a fix; the rest are ways of not fixing it:

1. **Align the two loops on one budget.** The drift is documented in
   `TokenGenerationLoop`; making the batched loop count what the sequential one counts
   removes the discrepancy at the source.
2. **Align the sequential loop to the batched one instead**, if the batched count is the
   intended one — same work, opposite direction, and it changes existing behaviour.
3. **Compare only the overlapping rows in the harness.** Makes the test pass and leaves the
   inconsistency in the engine; recorded here so nobody reaches for it by accident.

A default flip needs the Qwen3 drift fixed and a deliberate choice from the Q8_0 list, in
that order — the row count is wrong independently of what the default is.

## The generation loop

One loop, above the model and below the API. It ingests the prompt using the ingestion
strategy the session's policy selects, then repeats: forward, sample, decode incrementally
to UTF-8, emit an event, test stop conditions.

Generation is **not** part of the forward pass and is separable from it. Embedding,
classification and reranking need a program, a backend and a device; they must not be
forced to acquire a sampler, a KV cache or a token loop. That separation is enforced, not
merely intended (rules 8a and 14).

Streaming emits one ordered `GenerationEvent` per emitted token, with incremental UTF-8
decoding so a multi-byte character is never split across events. The callback runs
**outside** the invocation lock: the boundary copies results out, releases, and only then
runs caller code.

## Sampling

Sampling is an operation, so it may execute on the device. `SamplingResidency` chooses:

- `HOST` — the logits row comes back and the host samples;
- `DEVICE` — the device chooses, and the full logits row never leaves it.

Device residency is a capability, not a default: a backend that does not offer it is not
silently downgraded — the policy is resolved against the device's declared capabilities
before the program is compiled, and the resolved value is what the plan is built from.

Greedy decoding compares argmax; temperature and top-p sampling do not, which is why the
numerical gates check top-k membership rather than only the emitted token. See
[`verification.md`](verification.md).
