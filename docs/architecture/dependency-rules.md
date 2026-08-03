# Dependency Rules

**Status: normative — agreed 2026-07-30.** The rules are binding for new code. The
concrete package names in the snippets match
[`target-architecture.md`](target-architecture.md) and do not exist in code yet: until a
package exists its rule is a review criterion, and it becomes an ArchUnit test when the
milestone that creates the package lands.

Terms: [`terminology.md`](terminology.md). Milestones:
[`migration-roadmap.md`](migration-roadmap.md).

## Allowed dependency direction

```
integration  →  api  →  generation  →  model  →  program  →  runtime  →  backend (SPI)
                                                                            ↑
                                                            backend.tornado ┘
                                                            backend.cpu     ┘

format  ←  model            (loading only; format must not be depended on by program/runtime)
tokenizer                   (depended on by generation and model; depends on nothing else here)
```

Downward only. No layer depends on a layer above it. Backend implementations depend on
the backend SPI and on the runtime/program layers they execute; nothing depends on a
backend implementation except a service-loader registration.

## Rule format

Each rule states its **intent**, the **allowed direction**, **violation examples**
(drawn from today's code where possible), and **migration considerations**.

---

## Rule 1 — TornadoVM stays in the Tornado backend

**Intent.** TornadoVM is an implementation detail of one backend. Everything above the
backend SPI must compile and be reasoned about without it. This is the single most
important rule in this document.

**Allowed direction.** `backend.tornado.**` → `uk.ac.manchester.tornado.**`.
Nothing else.

**Proposed ArchUnit rule:**

```java
noClasses()
    .that().resideOutsideOfPackage("..backend.tornado..")
    .should().dependOnClassesThat()
    .resideInAnyPackage("uk.ac.manchester.tornado..");
```

**Violations today.** 26 files outside `tornadovm/**` import
`uk.ac.manchester.tornado.*`:

| Package | Files | Nature |
| --- | --- | --- |
| `model/loader` | 8 | loaders build `TornadoTensor` / native arrays directly |
| `inference/state` | 7 | `State` and subclasses hold `FloatArray`/`HalfFloatArray`/`IntArray` fields |
| `tensor/tornado` | 4 | `TornadoTensor` wraps TornadoVM native arrays |
| `inference` | 3 | `InferenceCore*`, `InferenceEngine` return/accept `FloatArray` |
| `inference/sampler` | 3 | `Sampler` is typed against `FloatArray` |
| `tensor` | 1 | `GGUF` |

Additionally, `model.Model` and `model.AbstractModel` depend on
`tornadovm.TornadoVMMasterPlan`, which is Tornado-specific even though the import is
not `uk.ac.manchester.tornado.*`. See Rule 2.

**Migration considerations.** This rule cannot be turned on at once. The first
implementation **must ship with a documented allowlist** of the existing violating
classes (see [Allowlist policy](#allowlist-policy)). The natural order of removal:
`sampler` (smallest, and a clear abstraction leak), then `inference`, then
`model/loader`, then `state`, then `tensor/tornado` (which becomes backend-owned
storage).

---

## Rule 2 — Model architecture packages do not import TornadoVM

**Intent.** A model architecture describes a transformer. It must be readable,
testable and extendable without a GPU runtime on the classpath.

**Allowed direction.** `model.**` → `program.**`, `runtime.**`, `tokenizer.**`.
Never → `backend.tornado.**` or `uk.ac.manchester.tornado.**`.

```java
noClasses()
    .that().resideInAPackage("..model..")
    .should().dependOnClassesThat()
    .resideInAnyPackage("uk.ac.manchester.tornado..", "..backend.tornado..");
```

**Violations today.**

- `model.Model` declares `TornadoVMMasterPlan tornadoVMPlan()` and
  `void setTornadoVMPlan(TornadoVMMasterPlan)`.
- `model.AbstractModel` holds a `TornadoVMMasterPlan plan` field.
- `model.llama.Llama` (and every sibling) imports `TornadoVMMasterPlan` for the
  `generateTokensGPU` signature and for `PREFILL_BATCH_SIZE`.
- `model.loader.*` builds TornadoVM tensors directly.

**Migration considerations.** Removing the plan from `Model` is
[ADR-001](decisions/ADR-001-model-session-separation.md) and milestone M6; it is
blocked on having somewhere else for the plan to live (a session). `model/loader` is
separable earlier: it can produce format-neutral tensor descriptors and let the
backend materialize device storage (M4).

---

## Rule 3 — Backend-neutral program interfaces do not import TornadoVM

**Intent.** An inference program is a description. If it references `TaskGraph`,
`GridScheduler` or `ImmutableTaskGraph`, it is not a description — it is TornadoVM
code, and no second backend can ever implement it.

**Allowed direction.** `program.**` → `runtime.**` only.

```java
noClasses()
    .that().resideInAnyPackage("..program..", "..backend")
    .should().dependOnClassesThat()
    .resideInAnyPackage("uk.ac.manchester.tornado..", "..backend.tornado..", "..backend.cpu..");
```

Note this also forbids the backend **SPI package** (`..backend`, not its
subpackages) from depending on any backend implementation.

**Violations today.** No `program` package exists. The nearest existing types —
`tornadovm.plan.ForwardPlan`, `ExecutionMode`, `*ForwardPlanComponents` — are all
Tornado-typed, which is why they are classified as compiled-program internals rather
than program descriptions ([ADR-002](decisions/ADR-002-program-and-compiled-program.md)).

**Migration considerations.** The risk here is introducing the program layer as a thin
rename of `ForwardPlanComponents`, which would import the Tornado types by accident.
When M9 starts, write this ArchUnit rule **first**, then add the package.

---

## Rule 4 — GGUF is a format concern, not a tensor or operation concern

**Intent.** GGUF/GGML describe how weights are stored in a file. Runtime tensors and
operations describe how data is laid out for execution. Conflating them means a second
file format cannot be added, and it puts a file-format enum in operation signatures.

**Allowed direction.** `format.gguf.**` → `runtime.**` (to produce runtime
descriptors). Never `runtime.**` → `format.**`, and never `program.**` → `format.**`.

```java
noClasses()
    .that().resideInAnyPackage("..runtime..", "..program..", "..backend..")
    .should().dependOnClassesThat()
    .resideInAPackage("..format..");
```

**Violations today.**

- `inference.weights.Weights.getWeightType()` returns `GGMLType`.
- `tensor.standard.FloatTensor` and `tensor.tornado.TornadoTensor` both expose
  `GGMLType`.
- `tornadovm.plan.ForwardPlanFactory.create(...)` takes `GGMLType` as its first
  dispatch axis.
- `tensor/` holds `GGUF`, `GGMLTensorEntry`, `MetadataValueType` and `GGMLType`
  alongside the runtime tensor hierarchies.

**Migration considerations.** Requires a runtime `DataType` first
([ADR-004](decisions/ADR-004-tensor-and-format-separation.md), M4). The mapping
`GGMLType → DataType` is not one-to-one: `AbstractModelLoader.effectiveGpuWeightType`
already collapses `Q4_K`/`Q5_K`/`Q6_K` to `Q8_0` for GPU execution, which is exactly
the format→runtime mapping this rule wants to make explicit.

---

## Rule 5 — Models own immutable configuration and weights

**Intent.** A loaded model is a shareable, thread-safe value. Mutable fields on it make
it unshareable and make concurrency undefined.

**Allowed direction.** N/A (structural rule).

```java
classes()
    .that().resideInAPackage("..model..")
    .and().areNotInterfaces()
    .should().haveOnlyFinalFields();
```

(Applied to the loaded-model types, not to loaders or builders.)

**Violations today.** `AbstractModel` has four non-final fields (`tokenizer`,
`weights`, `chatFormat`, `plan`); `plan` is additionally mutated at runtime via
`setTornadoVMPlan`.

**Migration considerations.** `tokenizer`/`weights`/`chatFormat` are assigned once in
the constructor and can be made final immediately. `plan` cannot, until sessions exist.

---

## Rule 6 — Sessions own mutable inference state

**Intent.** All mutation belonging to one sequence lives in one place with a clear
owner and a clear lifetime.

**Allowed direction.** `session` → `runtime` state types; nothing else holds a
reference to a session's state.

```java
noClasses()
    .that().resideInAPackage("..model..")
    .should().dependOnClassesThat()
    .haveSimpleNameEndingWith("SessionState");
```

**Violations today.** No session type exists. `server.InferenceService` holds one
`State` and one plan for the whole service and serializes access with a lock;
`Model.runInteractive` creates a `State` and drives it inline.

**Migration considerations.** M6. The first version can keep serialized access
(matching today's behaviour) while still moving ownership.

---

## Rule 7 — The KV cache is never global model state; storage is managed and leased

**Intent.** A KV cache must never hang off a loaded model — that prevents concurrent
sessions and makes correctness depend on call ordering. But it must also not be *owned*
by a single session, because paged attention and prefix sharing are defined by blocks
outliving and being shared between sequences.

So: **KV storage is owned by a cache manager scoped to the engine, and leased to
sessions.** A session holds a block table referencing blocks it does not own.

**Allowed direction.** `engine.KvCacheManager` owns block storage. Sessions hold leases.
Models reference neither.

```java
noClasses()
    .that().resideInAnyPackage("..model..", "..program..")
    .should().dependOnClassesThat()
    .haveSimpleNameContaining("KvCache")
    .orShould().dependOnClassesThat().haveSimpleNameContaining("BlockPool");
```

**Violations today.** `State.keyCache` / `valueCache` and their device mirrors live on
`State`, created by `Model.createNewState()`. The cache is not *on* the model, which is
correct; the risk is that the "one `State` per service" pattern in `InferenceService`
hardens into a per-model cache.

**Migration considerations.** M6 splits lease from storage rather than moving the
cache wholesale into the session. See
[ADR-005](decisions/ADR-005-kv-cache-ownership-and-leases.md), which also records the
CUDA-graph invariant: the pool is one persistent array with in-kernel indexing, and
leased blocks are pinned against eviction
([capability C1](tornadovm-capabilities.md#c1--cuda-graph-capture-fixes-device-addresses)).

**History.** This rule originally read "KV cache types are reachable only from session
state", which made a shared block pool a violation and shared prefix blocks
unrepresentable. Corrected during the ARCH-01 review.

---

## Rule 8a — Generation policy is separate from forward execution

**Intent.** Embedding, classification and reranking models have no generation loop.
If forward execution depends on generation, those use cases cannot exist. It also keeps
the model interface free of console and transport concerns.

Generation *policy* means: the token loop, stop conditions, streaming, transport,
console I/O, prompt construction.

**Allowed direction.** `generation.**` → `model.**` → `program.**`.
Never `model.**` → `generation.**`.

```java
noClasses()
    .that().resideInAnyPackage("..model..", "..program..", "..runtime..", "..backend..")
    .should().dependOnClassesThat()
    .resideInAnyPackage("..generation..", "..api..", "..integration..");
```

**Violations today.**

- `Model.runInteractive` / `runInstructOnce` / `runInstructOnceLangChain4J` implement
  the generation loop, read `System.in`, write `System.out`, and take
  `org.beehive.gpullama3.Options` (a CLI type).
- `Model.generateTokens` / `generateTokensGPU` put the generation loop on the model
  interface and take a `Sampler`.
- `inference.sampler.Sampler.createSampler(Model, Options)` depends on both `Model`
  and the CLI options record.

**Migration considerations.** The API façade *calls* the existing default methods; the
execution-policy phase moves the loops out of `Model`. Deprecate before removal —
`runInstructOnceLangChain4J` is an external integration point.

---

## Rule 8b — Sampling is an operation, and may execute on the device

**Intent.** Sampling is *not* generation policy. It is an operation over logits, and a
backend may legitimately implement it — on-device argmax removes a full logits-row
transfer per token.

This rule exists to stop Rule 8a from being read as "nothing in the backend may sample".

**Allowed direction.** `Sample` / `ArgMax` are entries in the operation vocabulary
(`program.op`), with backend implementations like any other operation. What a backend
must **not** do is own the loop that decides *whether to keep sampling*.

**Grounding.** On-device sampling already ships on `main`:
`tornadovm.layers.type.fp16.LogitsFP16Layer.DEVICE_SAMPLE` runs argmax on device and
writes `State.sampledToken`; `InferenceEngine.sampleTokenGpu` reads it instead of
transferring the vocabulary row; `LlamaApp.guardDeviceSample` polices the preconditions.
It is an ordinary TornadoVM task writing one `IntArray` element.

**Non-violation note.** The device sampler is explicitly **not** a Rule 8a violation and
must never be added to its allowlist. A rule that makes shipped, faster, correct code a
violation either gets a permanent exemption — weakening every other rule — or drives a
revert.

**Relation to Rule 14.** Unchanged: core abstractions must not *require* a sampler.
Sampling may exist; it may not be mandatory.

**History.** Rules 8a and 8b were one rule whose enforceable form forbade
`..backend..` → `..generation..`, making the existing device sampler a violation. Split
during the ARCH-03 review.

---

## Rule 9 — Logical programs describe backend-neutral work

**Intent.** A program says what to compute; it does not know where. This is what makes
one architecture implementation run on several backends.

**Allowed direction.** `program.**` → `runtime.**` (descriptors, `DataType`,
configuration). No device handles, no buffers, no backend types.

```java
noClasses()
    .that().resideInAPackage("..program..")
    .should().dependOnClassesThat()
    .haveSimpleNameEndingWith("CompiledProgram")
    .orShould().dependOnClassesThat().resideInAPackage("..backend.tornado..");
```

**Violations today.** N/A — the layer does not exist.

**Migration considerations.** See Rule 3.

---

## Rule 10 — Compiled programs are backend-specific and reusable

**Intent.** Compilation is expensive; invocation is not. Keeping them as distinct types
makes it structurally hard to accidentally compile inside a loop.

**Allowed direction.** `backend.tornado.**` may implement `CompiledProgram`. Callers
hold the SPI type, never the implementation.

```java
classes()
    .that().implement("..backend.CompiledProgram")
    .should().resideInAPackage("..backend..");
```

**Violations today.** N/A as a type, but the concept exists correctly:
`TornadoVMMasterPlan` implementations are built once and executed per token.

**Migration considerations.** The existing behaviour is the thing to preserve, not to
change. See Rule 13.

---

## Rule 11 — `TaskGraph` and `TornadoExecutionPlan` live inside the Tornado backend

**Intent.** A stricter, type-specific restatement of Rule 1 for the types most likely
to leak, because they appear in constructor and method signatures across the current
`tornadovm/**` tree.

**Allowed direction.** `backend.tornado.**` only.

```java
noClasses()
    .that().resideOutsideOfPackage("..backend.tornado..")
    .should().dependOnClassesThat()
    .haveFullyQualifiedName("uk.ac.manchester.tornado.api.TaskGraph")
    .orShould().dependOnClassesThat()
    .haveFullyQualifiedName("uk.ac.manchester.tornado.api.ImmutableTaskGraph")
    .orShould().dependOnClassesThat()
    .haveFullyQualifiedName("uk.ac.manchester.tornado.api.TornadoExecutionPlan")
    .orShould().dependOnClassesThat()
    .haveFullyQualifiedName("uk.ac.manchester.tornado.api.GridScheduler");
```

**Violations today.** These types are confined to `tornadovm/**` already — which is
the good news, and the reason Rule 11 is close to satisfiable once `tornadovm/**` is
relocated to `backend/tornado/**`.

**Migration considerations.** This is the rule most likely to pass early. Worth
enabling as soon as the package move happens, ahead of the broader Rule 1.

---

## Rule 12 — Forward plans are transitional Tornado compiled-program structures

**Intent.** Prevent `ForwardPlan` from being generalized upward into the neutral
layers. It is typed in `ImmutableTaskGraph` and `GridScheduler`; generalizing it would
drag TornadoVM into the program layer.

**Allowed direction.** `ForwardPlan` and subclasses reachable only from within the
Tornado backend.

```java
noClasses()
    .that().resideOutsideOfPackage("..backend.tornado..")
    .should().dependOnClassesThat()
    .haveSimpleNameEndingWith("ForwardPlan")
    .orShould().dependOnClassesThat()
    .haveSimpleNameEndingWith("ForwardPlanComponents");
```

**Violations today.** `ForwardPlanFactory` is called from
`TornadoVMMasterPlan*` (inside `tornadovm/`), so the plan types are already
well-contained. The factory does, however, depend upward on `model.Model`,
`model.ModelType` and the family-specific `State` subclasses.

**Migration considerations.** Do not rename `ForwardPlan` to something neutral-sounding
during migration; that invites exactly the confusion this rule prevents. See
[`terminology.md`](terminology.md#forward-plan).

---

## Rule 13 — No compilation or task-graph construction per token

**Intent.** The compile-once/execute-many property is the reason the GPU path performs
at all. It is easy to break accidentally during refactoring and hard to notice without
a benchmark.

**Allowed direction.** N/A — a behavioural rule.

This one cannot be fully expressed in ArchUnit. Enforce it by:

- keeping construction of task graphs confined to constructors and explicit
  `compile(...)` entry points (checkable: no `new TaskGraph(...)` outside
  `backend.tornado` compile paths);
- a test that compiles once and asserts the compiled-program identity is unchanged
  across N generated tokens;
- benchmark regression checks on tokens/second.

```java
noClasses()
    .that().haveSimpleNameEndingWith("Session")
    .or().haveSimpleNameEndingWith("Invocation")
    .should().callConstructorWhere(target ->
            target.getOwner().getName().startsWith("uk.ac.manchester.tornado"));
```

**Violations today.** None. `TornadoVMMasterPlan.initializeTornadoVMPlan` runs once per
run and `withPreCompilation()` is called at construction.

---

## Rule 14 — Core abstractions do not assume generation

**Intent.** Keep the door open for embeddings, classification, reranking, encoder-only
and multimodal models without a second framework.

Core types (`LoadedModel`, `InferenceProgram`, `CompiledProgram`, `Backend`, session
state, tensor descriptors) must **not require** any of:

- a tokenizer;
- a chat format;
- a KV cache;
- a prefill/decode split;
- a token-generation loop;
- a sampler.

These are capabilities that a text-generation model *adds*.

```java
noClasses()
    .that().resideInAnyPackage("..program..", "..runtime..", "..backend..")
    .should().dependOnClassesThat()
    .resideInAnyPackage("..tokenizer..", "..generation..")
    .orShould().dependOnClassesThat().haveSimpleNameContaining("ChatFormat");
```

**Violations today.** `Model` requires `tokenizer()`, `chatFormat()`,
`generateTokens(...)` and `generateTokensGPU(...)` from every implementation. `State`
always allocates a KV cache regardless of whether the model needs one.

**Migration considerations.** M6 and M10. A capability-interface split
(`TextGenerationModel extends LoadedModel`) is the likely mechanism, but the exact
shape is an open question.

---

## Rule 15 — No central model-type switches for new architectures

**Intent.** Adding an architecture should mean adding a provider, not editing five
switch statements in four packages.

**Allowed direction.** Architecture-specific code is discovered, not enumerated.

```java
noClasses()
    .that().resideOutsideOfPackage("..model.provider..")
    .should().dependOnClassesThat()
    .haveSimpleName("ModelType");
```

**Violations today.** Adding a family currently touches at least:

- `ModelType` — new enum constant with a `loadModel` override;
- `ModelLoader.detectModelType` — string matching on `general.name`;
- `ForwardPlanFactory` — a branch in `createFP16Plan` and in `createQ8_0Plan`, plus
  two `create<Family><Quant>Plan` helpers;
- `InferenceCore` — a new `forwardJava*` method;
- `Model.forward` implementation in the new family class;
- `Sampler.createSampler`;
- new `State` and `Weights` subclasses.

**Migration considerations.** M5 and M11. `ModelType` is likely to survive as an internal
identifier long after dispatch moves to providers; the rule targets *dispatch on* it,
not its existence. Note that `ForwardPlanFactory` also casts `State` to family-specific
subtypes, so removing the switch depends on the state work in M6.

---

## Rule 16 — No console I/O outside the CLI integration

**Intent.** The stated audience is developers embedding this in JVM applications
(LangChain4j, Quarkus, servers). A library that prints to stdout corrupts structured
logs and cannot be silenced or routed. This is the same class of problem as
[P2](current-architecture.md#p2--model-also-owns-the-application-loop) — the model
owning the application loop — and it will otherwise be discovered by the first embedder
rather than by a rule.

**Allowed direction.** Library code emits through a project-owned logging sink, no-op by
default. Only the CLI integration writes to `System.out` / `System.err`.

```java
noClasses()
    .that().resideOutsideOfPackage("..integration.cli..")
    .should().callMethodWhere(target ->
            target.getOwner().getName().equals("java.io.PrintStream")
            && (target.getName().equals("println") || target.getName().equals("print")));
```

**No external logging facade.** The library owns a small sink interface rather than
depending on SLF4J or similar. `pom.xml` declares no logging dependency today, and the
project ships a shaded jar with native-image considerations; importing a facade's
dependency surface into a self-contained inference library buys little. An optional
SLF4J bridge belongs in an integration module. The sink composes with the metrics sink
of [Rule 17](#rule-17--metrics-flow-bottom-to-top-by-design) — same shape, same
direction, plausibly the same lifecycle.

**Violations today.** 65 `System.out` / `System.err` occurrences across 20 main-source
files, including the generation loop — principally `Model.runInteractive` and
`Model.runInstructOnce`.

**Migration considerations.** Ships with the 20 files as an enumerated allowlist under
the standard shrink-only policy. The CLI keeps printing; that is its job.

---

## Rule 17 — Metrics flow bottom-to-top, by design

**Intent.** Metrics are the one thing in this architecture with the *opposite*
dependency direction to everything else: produced at the bottom (backend, device),
consumed at the top (API, engine, operator). Rule 8a forbids `..backend..` → `..api..`,
so a metrics facility placed in the API layer is un-callable from where the data
originates. Without an explicit seam the outcome is either a forbidden upward dependency
or — the actual default — device timings that are simply never surfaced.

**Allowed direction.** A metrics **sink interface** lives in the runtime layer, below
the backend SPI's consumers. Backends write to it; API and engine read from it. Sink
implementations (in-memory, bench recorder, exporter) live above the interface.

```java
// Permitted, and deliberately so — the one upward-looking seam in the design.
classes()
    .that().resideInAPackage("..backend..")
    .may().dependOnClassesThat().resideInAPackage("..runtime.metrics..");

// Still forbidden: backends must not reach the sink implementations.
noClasses()
    .that().resideInAPackage("..backend..")
    .should().dependOnClassesThat()
    .resideInAnyPackage("..api..", "..generation..", "..integration..");
```

**This is a designed permission, not an allowlist entry.** The distinction is the whole
point: an allowlist entry says "this violation is tolerated for now", which invites
removal. A permitted edge says "this direction is correct here", which invites use.

**Grounding.** TornadoVM already produces everything needed and we discard all of it —
`withProfiler(ProfilerMode)` plus `TornadoExecutionResult.getProfilerResult()` yields
device kernel time, host↔device transfer time and bytes, device memory usage and compile
time. Zero references exist in main sources. See
[the capability ledger](tornadovm-capabilities.md#available-capabilities).

**Cost caveat, to be designed in.** `withProfiler(...)` is not free. The sink must be off
by default on the decode path, with profiling opt-in per execution — otherwise telemetry
is paid for per token.

**Violations today.** No seam exists. `auxiliary/RunMetrics` is a static holder that
prints; it becomes one sink implementation.

---

## Rule 18 — Engine tier direction

**Intent.** The engine owns scheduling across sequences. It must sit above sessions and
below the public API, so that both the CLI and the server can drive it.

**Allowed direction.** `engine.**` → `model.**`, `session`, `program.**`, `runtime.**`.
Never `model.**` or session types → `engine.**`.

```java
noClasses()
    .that().resideInAnyPackage("..model..", "..program..", "..runtime..", "..backend..")
    .should().dependOnClassesThat().resideInAPackage("..engine..");
```

**Rationale.** A session must remain usable without an engine — that is the simple
single-sequence path. If sessions depend on the engine, the simple path acquires a
scheduler it does not need.

**Violations today.** No engine tier exists. `bench/BatchedDecodeEngine` (PR #129) is
its seed and currently lives in a benchmark package.

**Migration considerations.** See
[ADR-006](decisions/ADR-006-engine-tier.md) and the engine milestone in the roadmap.

---

## Allowlist policy

The first ArchUnit implementation will not pass on existing code. That is expected.

**Policy:**

1. Every rule that cannot pass ships with an explicit, enumerated allowlist of the
   currently-violating classes — by fully qualified name, never by wildcard package.
2. Each allowlist entry carries a comment naming the milestone that removes it.
3. The allowlist may **shrink** in any pull request. It may **not grow** without an
   ADR or an explicit maintainer decision recorded in the pull request.
4. A rule with an empty allowlist has its allowlist deleted, not left as an empty list.
5. CI fails if a class not on the allowlist violates a rule. CI should also fail if an
   allowlist entry no longer violates anything (stale entries hide progress).

Wildcards are banned because `..tornadovm..` as an allowlist entry would permanently
exempt the largest part of the codebase and make the rule meaningless.

## Rules not yet enforceable

Rules 3, 6, 9, 10, 12, 14, 15, 17 and 18 reference packages or types that do not exist.
They are review criteria until the corresponding milestone creates the package, at which
point the ArchUnit test is added **before** the implementation.

Rules 1, 2, 5, 7, 8a, 11, 13 and 16 can be written against today's code — with
allowlists — as part of
[milestone M1](migration-roadmap.md#m1--guardrails).

Rule 8b is a non-violation clarification rather than a check: it exists to keep the
device sampler off Rule 8a's allowlist.

## Rule index

| Rule | Subject | Enforceable now? |
| --- | --- | --- |
| 1 | TornadoVM confined to the Tornado backend | Yes, with allowlist |
| 2 | Model packages free of TornadoVM | Yes, with allowlist |
| 3 | Program interfaces free of TornadoVM | On package creation |
| 4 | GGUF is a format concern | After `DataType` exists |
| 5 | Models immutable | Yes, partially |
| 6 | Sessions own mutable state | On session type |
| 7 | KV storage managed and leased | Yes, as regression guard |
| 8a | Generation policy out of the backend | Yes, with allowlist |
| 8b | Sampling is an operation | Clarification, not a check |
| 9 | Programs are backend-neutral | On package creation |
| 10 | Compiled programs are backend-specific | On SPI |
| 11 | `TaskGraph` / `TornadoExecutionPlan` confined | Yes |
| 12 | Forward plans stay transitional | Yes |
| 13 | No per-token compilation | Behavioural + test |
| 14 | Core does not assume generation | On package creation |
| 15 | No central model-type switches | On provider SPI |
| 16 | No console I/O outside the CLI | Yes, with allowlist |
| 17 | Metrics seam direction | On metrics package |
| 18 | Engine tier direction | On engine package |
