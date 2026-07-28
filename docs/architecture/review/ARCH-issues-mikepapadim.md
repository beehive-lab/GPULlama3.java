# ARCH issues — review of the architecture baseline (`7ee6f86`)

Review of `docs/architecture/` as of commit `7ee6f86` (branch `refactor/framework-abstractions`).

One topic per issue. Each is Open until we decide Accept / Reject / Postpone together.

Context for the reviewer: the baseline is strong on the *library* axis — layering direction, the 15
dependency rules, ADR-001/002, and the P1–P10 pressure points are accurate and better specified than
anything we had. Most issues below are about the *product* axis: the serving path (batching, paged KV,
prefix cache), quantization, and the numbers that decide whether a phase regressed anything. Two issues
(ARCH-01, ARCH-03) are conflicts between a stated rule and code that already exists on `main` or in an
open PR, and those are the ones worth settling first.

| # | Title | Severity |
|---|---|---|
| ARCH-01 | KV cache ownership as stated prevents paged and shared-prefix KV | Critical |
| ARCH-02 | No engine layer: continuous batching has no home in the layering | Critical |
| ARCH-03 | Rule 8 forbids the on-device sampler that already exists on `main` | High |
| ARCH-04 | Roadmap has no phase for promoting the batched-decode engine | High |
| ARCH-05 | Quantization scheme excluded from scope although P6 identifies it | High |
| ARCH-06 | Phase 1 has no numerical regression net | High |
| ARCH-07 | `GenerationSession` cannot express concurrent serving | High |
| ARCH-08 | Prefix / prompt cache has no owner in the layering | Medium |
| ARCH-09 | Backend SPI has no device-memory allocation seam | Medium |
| ARCH-10 | Backend SPI does not express hybrid native library dispatch | Medium |
| ARCH-11 | Rule 13 has no benchmark gate wired to it | Medium |
| ARCH-12 | Model provider SPI lands at phase 8, after the pain | Medium |
| ARCH-13 | In-flight PRs are not sequenced against the phases | Medium |
| ARCH-14 | Multi-device execution absent from the target architecture | Low |
| ARCH-15 | Public API does not expose quantization of a loaded model | Low |
| ARCH-16 | No metrics seam, and device-level metrics cannot flow to their consumer | High |
| ARCH-17 | Observability is scheduled last, but earlier phases and the server need it now | Medium |
| ARCH-18 | No logging policy for an embeddable library | Medium |
| ARCH-19 | Serving-level metrics are undefined | Medium |

---

## ARCH-01

**Title:**
KV cache ownership as stated prevents paged and shared-prefix KV

**Section:**
`dependency-rules.md` § Rule 7 — The KV cache is never global model state; `public-api.md` § `GenerationSession`

**Severity:**
Critical

**Observation:**
Rule 7 states: *"A KV cache belongs to one sequence… KV cache types are reachable only from session
state."* `public-api.md` reinforces it in the type sketch: `/** One sequence. NOT thread-safe. Owns KV
cache and activation buffers. */`. Taken together, the KV cache is owned by the session and scoped to
one sequence.

**Suggestion:**
Keep the prohibition (never on the model), but move ownership up rather than down: an engine-scoped
`KvCacheManager` owns a block pool; a session holds a *lease* — a block table referencing blocks it does
not own. Reword Rule 7 as "the KV cache is never global model state; KV storage is owned by a
cache manager scoped to the engine and leased to sessions", and change the `GenerationSession` javadoc
from "owns KV cache" to "holds a KV lease".

**Rationale:**
Paged attention and prefix sharing are defined by blocks outliving and being shared *between* sequences.
Under the current wording, a block pool is a rule violation, and shared prefix blocks are impossible
because two sessions cannot reference the same cache memory. The rule was written against a real
problem (cache on the model, ordering-dependent correctness); lease semantics preserve that property
while allowing sharing.

**Impact:**
`State.keyCache` / `valueCache` and their device mirrors; the block pool and per-slot `blockTable` in
`bench/BatchedDecodeEngine` (PR #129); `InferenceService`, which today holds a `State` behind a lock;
phase 3 (loaded-model/session separation) would need to split lease from storage rather than move the
cache wholesale into the session.

**Status:**
Open

---

## ARCH-02

**Title:**
No engine layer: continuous batching has no home in the layering

**Section:**
`target-architecture.md` § Layering

**Severity:**
Critical

**Observation:**
The layer stack is Integrations → Public API and generation → Models and sessions → Inference programs
and operations → Runtime, tensors and state → Backend SPI → TornadoVM backend. There is no tier that
owns *scheduling across sequences*. The API layer is described as "sampling, stop conditions,
streaming"; the model/session layer is per-sequence.

**Suggestion:**
Insert an engine tier between the public API and models/sessions, owning admission and batch
composition: `LLMEngine` (`addRequest`, `step`), `Scheduler`, `KvCacheManager`, `BlockPool`. Sessions
become handles the engine schedules, not the unit of execution.

**Rationale:**
Continuous batching is not a feature of a session or of a model — it is a policy over many sequences,
and it must sit above sessions and below the API to be usable by both the CLI and the server. Without
this tier the architecture can only express "one request occupies one session occupies the device",
which is the throughput ceiling the project is trying to remove. vLLM's v1 split (engine core /
scheduler / KV manager) exists for the same reason.

**Impact:**
`bench/BatchedDecodeEngine` (PR #129) becomes the seed of this tier instead of a benchmark; the
OpenAI server (`server/OpenAIServer`, `InferenceService`) drives the engine rather than serialising on
one `State`; ARCH-01 (lease semantics) is a precondition.

**Status:**
Open

---

## ARCH-03

**Title:**
Rule 8 forbids the on-device sampler that already exists on `main`

**Section:**
`dependency-rules.md` § Rule 8 — Generation and sampling are separate from forward execution

**Severity:**
High

**Observation:**
Rule 8's enforceable form is:

```java
noClasses().that().resideInAnyPackage("..model..", "..program..", "..runtime..", "..backend..")
    .should().dependOnClassesThat().resideInAnyPackage("..generation..", "..api..", "..integration..");
```

On `main`, sampling already happens inside the backend: `tornadovm/layers/type/fp16/LogitsFP16Layer`
performs device-side sampling, `InferenceEngine` consumes it, and `LlamaApp.guardDeviceSample` exists
precisely to manage it. The layering diagram likewise lists sampling in the API tier.

**Suggestion:**
Separate the two things the rule conflates: *generation policy* (loop, stop conditions, transport) must
stay out of the backend — keep that. *Sampling* is an operation that may legitimately execute on the
device. Model it as an operation in the program/operations vocabulary with a backend implementation,
and restrict Rule 8 to the generation loop and transport concerns.

**Rationale:**
Device-side sampling removes a logits round-trip per token, which is why it was implemented. A rule
that makes the current, faster code a violation will either be granted a permanent allowlist entry —
weakening every other rule — or drive a revert of a performance feature.

**Impact:**
`LogitsFP16Layer` and the equivalent Q8_0 path; `InferenceEngine`; `LlamaApp.guardDeviceSample`; the
sampler selection in `Sampler.createSampler`; the allowlist that phase 1 would otherwise have to
enumerate.

**Status:**
Open

---

## ARCH-04

**Title:**
Roadmap has no phase for promoting the batched-decode engine

**Section:**
`migration-roadmap.md` § Phase 7 — Consolidate engine variants behind execution policies; § What is
deliberately not scheduled

**Severity:**
High

**Observation:**
Phase 7 consolidates the *existing* engine variants (single-token, prefill+decode) behind execution
policies. Nothing in the ten phases promotes the batched-decode path, and no phase mentions batching,
scheduling or a KV manager.

**Suggestion:**
Add a phase, after the session/state split and before the server phase, that promotes
`bench/BatchedDecodeEngine` (PR #129) into the engine tier from ARCH-02, with its four capabilities
becoming named components: continuous batching → `Scheduler`, block pool + per-slot block table →
`KvCacheManager`, prefix reuse → `PrefixCache` (ARCH-08), on-device sampling → an operation (ARCH-03).

**Rationale:**
The most valuable serving code in the repository currently lives in a benchmark `main()`, driven by
`-Dbatch.decode.*` properties and hard-cast to LLaMA/Qwen3 + FP16/CUDA. If the refactor completes
without a phase for it, the code either stays trapped or is re-implemented; either way the architecture
is defined without its most demanding consumer, and phases 3 and 7 will make decisions (session shape,
execution policy) that it then has to fight.

**Impact:**
PR #129; `InferenceService`; phase 3 (`State` split) acquires a second consumer with different
requirements — per-slot state — which is better known before that phase than after.

**Status:**
Open

---

## ARCH-05

**Title:**
Quantization scheme excluded from scope although P6 identifies it

**Section:**
`migration-roadmap.md` § What is deliberately not scheduled; § Phase 5 — Reusable transformer operation
extraction; `current-architecture.md` § P6

**Severity:**
High

**Observation:**
"What is deliberately not scheduled" excludes *"new quantization formats, new kernels, new performance
work. Those continue independently on their own branches; this roadmap must not block them."* Yet P6
records that adding an architecture means touching `ForwardPlanFactory` with "two switch branches per
quantization", and phase 5 extracts reusable operations without saying anything about dtype.

**Suggestion:**
Distinguish *new formats* (correctly out of scope) from *the dtype seam* (in scope, and load-bearing).
Make phase 5's operation vocabulary dtype-parameterized — an operation is `matmul` over a
`QuantScheme`, not `MatmulQ8_0` — so that `layers/type/{fp16,q8_0}` collapses to M operation templates ×
D schemes. Add an invariant test: introducing a scheme adds at most k classes.

**Rationale:**
Today the layer classes are written per (model × dtype × mode × MMA) combination. Extracting "reusable
operations" without parameterizing dtype preserves that multiplication inside the new vocabulary, and
the next format (Q4_K, MXFP4/NVFP4 — the flagship performance goal) multiplies it again. The seam is
what makes the excluded work cheap; scheduling the seam does not schedule the formats.

**Impact:**
`tornadovm/layers/type/**` (the ~32 classes), `ForwardPlanFactory` switches on `GGMLType`,
`TransformerComputeKernelsLayered` Q8_0 kernels; unblocks int8/FP4 tensor-core matmul work later
without another refactor.

**Status:**
Open

---

## ARCH-06

**Title:**
Phase 1 has no numerical regression net

**Section:**
`migration-roadmap.md` § Recommended first milestone; § Phase 1

**Severity:**
High

**Observation:**
Milestone 1 is "ArchUnit test module with rules 1, 2, 5, 7, 11, 13 and enumerated allowlists", three
fields made final, and a read-only façade. The repository currently contains one test file
(`ToolCallParserUtilsTest`). Principle 4 says performance is a correctness criterion, but no phase
defines how *numerical* correctness is checked.

**Suggestion:**
Add to phase 1, before any code motion: a golden-reference dump (per-layer and final logits for the
supported model families × {FP16, Q8_0} at a fixed prompt and seed) and a test that re-runs each
configuration and asserts bit-identical logits; plus a CPU↔GPU parity check with a stated tolerance
(`|got − ref| ≤ 1e-2·Σ|wᵢaᵢ| + 1e-3`).

**Rationale:**
ArchUnit proves the dependency direction, not that the model still produces the same tokens. Every
subsequent phase moves state, splits objects and re-routes execution; without a bit-exact baseline a
silent numerical regression is indistinguishable from a sampling difference, and the "every phase leaves
it working" principle cannot be demonstrated.

**Impact:**
New test sources and golden binaries; CI time; models must be available to CI or the goldens committed
for a small model. It also gives phases 3, 5, 6 and 7 an objective exit criterion.

**Status:**
Open

---

## ARCH-07

**Title:**
`GenerationSession` cannot express concurrent serving

**Section:**
`public-api.md` § `GenerationSession`; § Simple example

**Severity:**
High

**Observation:**
`GenerationSession` is specified as "One sequence. NOT thread-safe." with a blocking
`GenerationResult generate(GenerationRequest request)`. There is no non-blocking submission, no
streaming callback in the interface sketch, and no type through which several in-flight requests can
share a device.

**Suggestion:**
Keep `GenerationSession` as the simple, single-sequence path, and add an explicit concurrent entry
point on the engine tier (ARCH-02): submit a request, receive a handle or reactive stream, with the
engine batching across handles. Decide explicitly whether the server uses the session API or the engine
API — it should be the latter.

**Rationale:**
The public API shape decides the throughput ceiling for every consumer. If the only API is a blocking
per-sequence call, a server can only serialise or spawn a session per connection, each with its own KV
allocation — which is what `InferenceService` does today behind a lock. The API needs to admit
"many requests, one device" without forcing every user into it.

**Impact:**
`server/OpenAIServer`, `InferenceService`; the façade in milestone 1 (better to land the session API
knowing the engine API is coming than to retrofit); LangChain4j/Quarkus adapters.

**Status:**
Open

---

## ARCH-08

**Title:**
Prefix / prompt cache has no owner in the layering

**Section:**
`target-architecture.md` § Layering; § Session lifecycle

**Severity:**
Medium

**Observation:**
The session lifecycle covers create → prefill → decode → reset/close for one sequence. No layer owns
state that is *keyed by token prefix and shared across sessions*.

**Suggestion:**
Name a `PrefixCache` in the engine tier, keyed by token-prefix hash and mapping to KV blocks held by the
`KvCacheManager` (ARCH-01), with an explicit statement about eviction and correctness (prefix identity
must include model, dtype and position offset).

**Rationale:**
Prefix reuse is the largest single win for chat and agent workloads — repeated system prompts are re-
prefilled today. It is architecturally distinctive because it deliberately crosses the session
boundary, so it must be designed in rather than added later; retrofitting shared, refcounted blocks
into a design whose invariant is "one sequence owns its cache" is the expensive path.

**Impact:**
Prefix caching in PR #129; `KvCacheManager` (ARCH-01/02); session `reset()` semantics in `public-api.md`.

**Status:**
Open

---

## ARCH-09

**Title:**
Backend SPI has no device-memory allocation seam

**Section:**
`target-architecture.md` § Backend providers; `public-api.md` § `Backend` / `DeviceSelector`

**Severity:**
Medium

**Observation:**
The backend SPI is described as `Backend`, `DeviceSelector`, `CompiledProgram`, `Invocation`. Device
memory appears only implicitly, through compiled programs and invocations.

**Suggestion:**
Add an allocation/lifetime concept to the SPI — device buffers with an owner and a lifetime longer than
one invocation (weights: model lifetime; KV blocks: engine lifetime; activations: invocation lifetime)
— and state which layer may allocate.

**Rationale:**
Three planned things need allocation lifetime to be explicit: KV blocks owned by a manager (ARCH-01),
CUDA-graph capture which requires stable device addresses across replays, and memory planning
(phase 10). Without a seam, allocation stays inside the Tornado backend and the engine cannot reason
about capacity — which is exactly what an admission scheduler must do to decide how many sequences fit.

**Impact:**
`TornadoVMMasterPlan` and the device-mirror fields on `State`; phase 9 (backend SPI) and phase 10
(memory planning); block pool sizing in PR #129.

**Status:**
Open

---

## ARCH-10

**Title:**
Backend SPI does not express hybrid native library dispatch

**Section:**
`decisions/ADR-003-tornado-backend-boundary.md`; `target-architecture.md` § Backend providers

**Severity:**
Medium

**Observation:**
ADR-003 places TornadoVM as the compiler and heterogeneous runtime, with the backend SPI exposing
compiled programs and invocations. Some operations, however, are dispatched to vendor libraries rather
than compiled from Java — TornadoVM library tasks calling cuBLAS/cuDNN, as explored in PR #131.

**Suggestion:**
State whether an operation implementation may be a native library call, and if so how it appears in the
SPI — e.g. an operation may be backed either by a compiled program or by a library dispatch, chosen by
the backend from device capabilities.

**Rationale:**
A GEMM served by cuBLAS and one served by a generated kernel are the same node in a backend-neutral
program but very different below the SPI. If the SPI admits only compiled programs, hybrid dispatch
becomes a special case inside the Tornado backend and cannot be selected per operation or per device.

**Impact:**
PR #131; any future cuBLAS/CUTLASS path for quantized matmul; the operations vocabulary in phase 5.

**Status:**
Open

---

## ARCH-11

**Title:**
Rule 13 has no benchmark gate wired to it

**Section:**
`dependency-rules.md` § Rule 13; `migration-roadmap.md` § Principles (4)

**Severity:**
Medium

**Observation:**
Rule 13 proposes three enforcement mechanisms, one of which is "benchmark regression checks on
tokens/second". No phase names a benchmark, a threshold or a place to record results. The repository
already has `--bench` and `docs/perf-history.jsonl` with recorded runs.

**Suggestion:**
Bind the rule to the existing machinery: each phase that touches the execution path records a `--bench`
run to `perf-history.jsonl` and states a tolerance (for example, decode tok/s within 3% of the previous
recorded run on the same machine and model). Keep the compiled-program-identity test as the structural
half of the rule.

**Rationale:**
Compile-once/execute-many is called non-negotiable, but an unenforced rule is documentation. The
tooling already exists; the roadmap just needs to reference it. Performance regressions from refactors
are usually gradual, and only a recorded series catches them.

**Impact:**
CI or a documented manual step per phase; `perf-history.jsonl` becomes a gate rather than a log.

**Status:**
Open

---

## ARCH-12

**Title:**
Model provider SPI lands at phase 8, after the pain

**Section:**
`migration-roadmap.md` § Phase 8 — Model provider SPI; `current-architecture.md` § P6

**Severity:**
Medium

**Observation:**
P6 records that adding a family today touches `ModelType`, `ModelLoader.detectModelType`,
`ForwardPlanFactory` (two branches per quantization), `InferenceCore`, `Sampler.createSampler` and the
state/weights hierarchies. The provider SPI that fixes this is phase 8 of 10.

**Suggestion:**
Either move the provider SPI earlier (it depends mainly on the model/session split, not on the backend
SPI), or state explicitly that families added before phase 8 will pay the switch-editing cost and will
be migrated by phase 8.

**Rationale:**
Model support is the most frequent change in this repository — Gemma 4 is in flight in PR #120. Ten
phases is a long time to keep paying a cost the design has already diagnosed, and each family added in
the meantime enlarges the phase-8 migration.

**Impact:**
PR #120 (Gemma 4) and any family after it; phase 8's size; `ForwardPlanFactory`, `ModelLoader`.

**Status:**
Open

---

## ARCH-13

**Title:**
In-flight PRs are not sequenced against the phases

**Section:**
`migration-roadmap.md` § Principles; § Recommended first milestone

**Severity:**
Medium

**Observation:**
The roadmap does not reference the open PRs. At the time of review these are #129 (static batched
decode), #120 (Gemma 4), #138 (FP16 KV cache with packed half2 split-KV attention) and #131 (hybrid CUDA
library tasks). All four touch code that phases 3, 5 and 7 restructure.

**Suggestion:**
State a land order and a freeze rule: land the feature PRs that the architecture depends on (at minimum
#129, since it is the source for the engine tier) before the phases that restructure their code, and
declare which files are "in refactor" so feature work rebases rather than collides.

**Rationale:**
#138 changes the KV cache layout while phase 3 moves cache ownership; #129 adds a second execution path
while phase 7 consolidates execution paths. Sequencing avoids resolving the same conflict twice, and it
is cheaper to promote #129 into the engine tier than to refactor around its absence and re-land it.

**Impact:**
Merge order and review effort on #129/#120/#131/#138; phase 3 and phase 7 scope.

**Status:**
Open

---

## ARCH-14

**Title:**
Multi-device execution absent from the target architecture

**Section:**
`target-architecture.md` § Backend providers; § Open questions

**Severity:**
Low

**Observation:**
`DeviceSelector` selects *a* device. Nothing in the target architecture describes work split across
devices — tensor, pipeline or data parallel.

**Suggestion:**
Add seams only, no implementation: a shard plan describing how a program's weights and work map to
devices, and an executor that runs the plan across backends. Mark it explicitly as design-only so it
constrains the SPI shape without scheduling the work.

**Rationale:**
Model size drives this eventually, and multi-device changes the shape of the backend SPI (an invocation
targets a device set, not a device) and of the KV manager (blocks per device). Recording the seam now is
cheap; discovering it after the SPI is frozen is not.

**Impact:**
Backend SPI signatures in phase 9; nothing before that.

**Status:**
Open

---

## ARCH-15

**Title:**
Public API does not expose quantization of a loaded model

**Section:**
`public-api.md` § `LocalModel`, `ModelOptions`

**Severity:**
Low

**Observation:**
`ModelInfo` and `ModelOptions` describe the loaded model, device and execution policy. The
quantization of the weights — the property that decides memory footprint, achievable speed and, today,
which execution paths exist — is not part of the sketch.

**Suggestion:**
Expose the weight quantization (and the compute dtype, if they differ) on `ModelInfo` as a runtime
enum, not a GGUF type, consistent with ADR-004.

**Rationale:**
A user choosing between two GGUF files, or deciding whether a model fits a device, needs this
information, and it is the first thing every consumer will otherwise obtain by reaching into the format
layer — the exact leak ADR-004 prohibits.

**Impact:**
`ModelInfo`; the runtime `DataType` introduced in phase 4; no behavioural change.

**Status:**
Open

---

## ARCH-16

**Title:**
No metrics seam, and device-level metrics cannot flow to their consumer

**Section:**
`target-architecture.md` § Layering; `migration-roadmap.md` § Phase 10 — Memory planning, diagnostics
and developer experience

**Severity:**
High

**Observation:**
Observability appears once, in phase 10, whose affected packages include `auxiliary/metrics/` and whose
scope is "diagnostics: which backend and device were chosen and why… where time goes (load / prefill /
decode)". No layer in the architecture owns metric emission, and no type carries it. Meanwhile Rule 8's
enforceable form forbids `..backend..` from depending on `..api..` or `..generation..`, so a metrics
facility that lives in the API layer cannot be called from where device timings originate. TornadoVM
already exposes per-execution device data (kernel time, copy-in/copy-out, device memory); no file in
`main` references `ProfilerMode`, `getProfilerResult` or `TornadoProfiler`, so none of it is surfaced
today.

**Suggestion:**
Define a metrics sink interface low in the stack (runtime layer, below the backend SPI's consumers) that
the backend writes into and the API/engine read from, and state that the backend contributes device
timings through it. Sink implementations (in-memory, bench recorder, exporter) live above; the interface
does not.

**Rationale:**
Metrics have the opposite dependency direction to everything else in the design: the data is produced at
the bottom and consumed at the top. If the seam is not placed explicitly below the producers, either the
backend acquires an upward dependency — the exact thing the rules forbid — or device timings are simply
never available, which is what the current state (profiler unused) shows happens by default. "Where time
goes" cannot be answered for the GPU path without it.

**Impact:**
`auxiliary/RunMetrics` (today a static holder) and `auxiliary/metrics/`; the Tornado backend, which
would gain a reporting call at execution boundaries; phase 10 scope; `TornadoProfilerResult` becomes
reachable.

**Status:**
Open

---

## ARCH-17

**Title:**
Observability is scheduled last, but earlier phases and the server need it now

**Section:**
`migration-roadmap.md` § Phase dependency summary; § Phase 10

**Severity:**
Medium

**Observation:**
Observability work sits in phase 10 of 10. Principle 4 states that performance is a correctness
criterion, and phases 3, 5, 6 and 7 all move execution-path code. The OpenAI-compatible server already
exists and has no metrics surface.

**Suggestion:**
Split the phase: land the minimal metrics seam and a small set of counters/timers (load, prefill,
decode, tokens/s) in an early phase alongside the guardrails, and keep memory planning, error-message
work and exporters in phase 10.

**Rationale:**
Every phase that claims "no behaviour change" needs evidence, and per-phase evidence is exactly what a
timer produces. Deferring the seam to the end means the phases that most need measurement run without
it, and the server ships unobservable in the meantime. Related to ARCH-06 (numerical gate) and ARCH-11
(benchmark gate), but distinct: those are CI gates for the refactor, this is runtime telemetry for
users and operators.

**Impact:**
Phase ordering only; the seam itself is ARCH-16.

**Status:**
Open

---

## ARCH-18

**Title:**
No logging policy for an embeddable library

**Section:**
`vision.md` § Target users; `public-api.md` § What must never leak

**Severity:**
Medium

**Observation:**
The vision targets embedding in JVM applications (LangChain4j, Quarkus, user services). The documents
say nothing about logging: no logging facade is named, and `pom.xml` declares no logging dependency. On
`main`, `System.out` / `System.err` appear 65 times across 20 main-source files, including the
generation loop.

**Suggestion:**
State a logging policy in the public-API document: the library emits through a facade (or a
no-op-by-default interface it owns), never to `System.out`/`System.err`; console output belongs to the
CLI integration only. Add it as a dependency rule with the current occurrences as the allowlist, the
same way the TornadoVM-import rule works.

**Rationale:**
An embedded library that prints to stdout is unusable in a server: it corrupts structured logs and
cannot be silenced or routed. This is the same class of concern as P2 (Model owns the application loop)
and will otherwise be discovered by the first Quarkus user rather than by a rule.

**Impact:**
The 20 files containing console I/O, principally `Model.runInteractive` / `runInstructOnce`; the CLI,
which keeps its printing; one new dependency rule and allowlist.

**Status:**
Open

---

## ARCH-19

**Title:**
Serving-level metrics are undefined

**Section:**
`migration-roadmap.md` § Phase 10; `public-api.md` § `GenerationResult`

**Severity:**
Medium

**Observation:**
The diagnostics scope is per-run and single-sequence: load, prefill, decode, chosen device and policy.
There is no notion of the quantities that describe a serving system — time to first token, queue wait,
batch occupancy, KV block utilisation, preemptions, admitted versus rejected requests.

**Suggestion:**
Define these as part of the engine tier proposed in ARCH-02, so the scheduler and KV manager emit them
through the ARCH-16 seam, and expose the per-request subset (TTFT, queue wait, tokens generated) on the
result type returned to callers.

**Rationale:**
These are the numbers that decide whether a batching scheduler and a block pool are working; without
them, tuning batch size, block count or admission policy is guesswork, and a regression in occupancy is
invisible while tokens/s for a single request looks unchanged. They also have to be designed with the
engine — retrofitting queue-wait accounting after the scheduler exists means threading timestamps
through code that was not built to carry them.

**Impact:**
The engine tier (ARCH-02) and the promotion of `bench/BatchedDecodeEngine` (ARCH-04); `GenerationResult`
/ `GenerationMetrics`; the server's ability to report usage.

**Status:**
Open
