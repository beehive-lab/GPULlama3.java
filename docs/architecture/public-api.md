# Public API (Proposed)

**Status: proposal. None of these types exist in the repository.**

Every name below is a placeholder. The purpose of this document is to fix the *shape*
of the developer-facing surface — what a user holds, what owns what, what is closeable,
what is thread-safe — not to commit to identifiers. Names will be settled when the
types are implemented.

Terms: [`terminology.md`](terminology.md). Layering:
[`target-architecture.md`](target-architecture.md).

## Design constraints

1. A Java developer running a local model should need one import block and no GPU
   concepts.
2. Ownership must be visible in the types: what holds device memory is `AutoCloseable`.
3. Thread-safety must be stated per type, not left to be discovered.
4. Nothing backend-specific, format-specific or CLI-specific appears in any signature.
5. The advanced API is the same API with more of it visible — not a parallel stack.

## Type sketch

### `LocalModels` — entry point

```java
public final class LocalModels {

    /** Load with defaults; auto-select backend and device. */
    public static LocalModel load(Path modelFile);

    public static LocalModel load(Path modelFile, ModelOptions options);

    /** Builder form for backend/device/policy control. */
    public static LocalModelLoader loader(Path modelFile);
}
```

### `LocalModel` — the loaded model

```java
/** Immutable and thread-safe. Owns weights and cached compiled programs. */
public interface LocalModel extends AutoCloseable {

    ModelInfo info();                  // architecture, name, dtypes, context length
    ModelConfiguration configuration(); // read-only hyperparameters

    GenerationSession newSession();
    GenerationSession newSession(SessionOptions options);

    @Override void close();            // releases weights and compiled programs
}
```

Note what is *not* here: no `forward(...)`, no `generateTokens(...)`, no sampler, no
plan setter. Generation lives on the session; execution lives below the model.

Text generation is a *capability*. A model that only produces embeddings would
implement `LocalModel` without `newSession()` — the exact mechanism (capability
interfaces such as `TextGenerationModel extends LocalModel`) is an open question, but
[dependency rule 14](dependency-rules.md#rule-14--core-abstractions-do-not-assume-generation)
requires that it be possible.

### `GenerationSession` — one sequence

```java
/** One sequence. NOT thread-safe. Holds a KV lease and its own invocation buffers. */
public interface GenerationSession extends AutoCloseable {

    GenerationResult generate(GenerationRequest request);

    /** Current sequence position; how much context is consumed. */
    int position();

    /**
     * Discard sequence state and start fresh.
     * Releases the current lease's private blocks; shared prefix blocks are
     * refcounted and survive if another lease still references them.
     */
    void reset();

    @Override void close();            // releases the lease and invocation buffers
}
```

The session **holds** a KV lease rather than owning cache storage — storage belongs to
the engine's cache manager, so blocks can be shared between sequences (prefix reuse) and
survive an individual session
([ADR-005](decisions/ADR-005-kv-cache-ownership-and-leases.md)).

### `LLMEngine` — concurrent serving

The session API is the simple, single-sequence path. Serving many requests at once uses
the engine tier instead:

```java
public interface LLMEngine extends AutoCloseable {

    /** Non-blocking admission. Returns immediately; work happens in step(). */
    RequestHandle addRequest(GenerationRequest request);

    /** One batched iteration across all admitted sequences. */
    StepResult step();

    EngineMetrics metrics();     // TTFT, queue wait, occupancy, block utilization
}
```

**The server uses the engine API, not the session API.** A blocking per-sequence call as
the only entry point forces a server to serialize or to spawn a session per connection —
which is what `InferenceService` does today behind a lock.

**Device concurrency comes from batching, not from parallel plans.** Several
`TornadoExecutionPlan`s can run concurrently — that is supported and tested — but device
buffers are per task graph, so a plan per session duplicates the weights on device
(~3.4 GB per concurrent session for a 3B-Q8 model). The engine therefore batches many
sequences into one invocation of one compiled program. See
[capability C2](tornadovm-capabilities.md#c2--device-buffers-are-per-task-graph).

### `ModelInfo` — dtypes are public

```java
public interface ModelInfo {
    String name();
    String architecture();
    int contextLength();

    /** How the weights are stored for execution. Runtime DataType, never GGMLType. */
    DataType weightType();

    /** What the kernels compute in. Differs from weightType for K-quants. */
    DataType computeType();
}
```

Two dtypes rather than one, because they already differ:
`AbstractModelLoader.effectiveGpuWeightType` collapses `Q4_K`/`Q5_K`/`Q6_K` to `Q8_0`,
and `getModelQuantization` maps GGUF file types 14–18 to `"Q8_0"`. A "Q6_K model"
executes as Q8_0. One field would mislead exactly the user asking in order to size a
device.

### `GenerationRequest` / `GenerationResult`

```java
public final class GenerationRequest {
    public static Builder builder();

    public interface Builder {
        Builder prompt(String prompt);
        Builder systemPrompt(String systemPrompt);
        Builder messages(List<ChatMessage> messages);
        Builder maxNewTokens(int maxNewTokens);
        Builder temperature(float temperature);
        Builder topP(float topP);
        Builder seed(long seed);
        Builder stopSequences(List<String> stopSequences);
        Builder onToken(Consumer<String> tokenCallback);   // streaming
        GenerationRequest build();
    }
}

public interface GenerationResult {
    String text();
    int promptTokens();
    int generatedTokens();
    FinishReason finishReason();       // STOP_TOKEN, MAX_TOKENS, STOP_SEQUENCE, CONTEXT_FULL
    GenerationTimings timings();       // load / prefill / decode, tokens per second
}
```

`GenerationTimings` is the public replacement for today's `RunMetrics` static
printing.

### `ModelOptions` / `SessionOptions`

```java
public final class ModelOptions {
    public static Builder builder();

    public interface Builder {
        Builder contextLength(int contextLength);
        Builder backend(Backend backend);
        Builder device(DeviceSelector selector);
        Builder executionPolicy(ExecutionPolicy policy);
        ModelOptions build();
    }
}

public final class SessionOptions {
    public static Builder builder();

    public interface Builder {
        Builder contextLength(int contextLength);     // ≤ model context length
        Builder executionPolicy(ExecutionPolicy policy);
        SessionOptions build();
    }
}
```

`ExecutionPolicy` replaces the current `llama.withPrefillDecode`,
`llama.prefillBatchSize` and `llama.deviceSample` system properties with explicit
values:

```java
ExecutionPolicy.singleToken();
ExecutionPolicy.prefillDecode();
ExecutionPolicy.prefillDecode(int prefillBatchSize);
```

**Open question:** whether execution policy is per-model, per-session, or both. Today
it is process-global and read at class-initialization time, which is the behaviour this
is meant to replace.

### `Backend` / `DeviceSelector` — advanced

```java
public interface Backend extends AutoCloseable {

    BackendId id();                    // e.g. "tornado", "cpu"
    List<Device> devices();
    Device defaultDevice();

    CompiledProgram compile(InferenceProgram program, CompileOptions options);
}

public interface Device {
    String name();
    DeviceKind kind();                 // GPU, CPU, ACCELERATOR
    long globalMemoryBytes();
}

public interface DeviceSelector {
    static DeviceSelector preferGpu();
    static DeviceSelector cpuOnly();
    static DeviceSelector byIndex(int backendIndex, int deviceIndex);
    static DeviceSelector matching(Predicate<Device> predicate);
}
```

`Device` exposes descriptive information only. It does not expose a TornadoVM device
handle. Inside the TornadoVM backend, PTX/CUDA, OpenCL and SPIR-V are device backends
of TornadoVM, surfaced here as devices — not as separate GPULlama backends. See
[ADR-003](decisions/ADR-003-tornado-backend-boundary.md).

### `InferenceProgram` / `CompiledProgram` — advanced

```java
/** Backend-neutral description of one forward pass. Immutable. No device handles. */
public interface InferenceProgram {
    ProgramSignature signature();
    List<ProgramComponent> components();
}

/** Backend-specific, reusable executable. Compiled once, invoked many times. */
public interface CompiledProgram extends AutoCloseable {
    Backend backend();
    Device device();
    ProgramSignature signature();

    void invoke(Invocation invocation);

    @Override void close();            // releases device resources
}
```

`Invocation` binds inputs, outputs and mutable session state for one call. It performs
no compilation. See [ADR-002](decisions/ADR-002-program-and-compiled-program.md).

## Simple example

```java
try (LocalModel model = LocalModels.load(modelPath)) {
    try (GenerationSession session = model.newSession()) {
        GenerationResult result = session.generate(
                GenerationRequest.builder()
                        .prompt("Explain heterogeneous computing.")
                        .maxNewTokens(128)
                        .build());

        System.out.println(result.text());
    }
}
```

## Advanced conceptual example

```java
InferenceProgram program = architecture.createProgram(...);
CompiledProgram compiled = backend.compile(program, options);
```

Expanded, with ownership visible:

```java
Backend backend = Backends.select(DeviceSelector.preferGpu());

try (LocalModel model = LocalModels.loader(modelPath)
        .backend(backend)
        .options(ModelOptions.builder()
                .contextLength(4096)
                .executionPolicy(ExecutionPolicy.prefillDecode(256))
                .build())
        .load()) {

    // Compiled once; shared by every session of this model on this device.
    // Two sessions below reuse it — no recompilation, no per-token graph building.
    try (GenerationSession a = model.newSession();
         GenerationSession b = model.newSession()) {

        a.generate(GenerationRequest.builder().prompt("First conversation.").build());
        b.generate(GenerationRequest.builder().prompt("Second conversation.").build());
    }
}
```

Whether `a` and `b` may run **concurrently** is unresolved — see
[ADR-001](decisions/ADR-001-model-session-separation.md). The API shape above does not
prevent concurrency, but the first implementation may serialize invocations.

## What must never leak through the public API

If any of these appears in a public signature, the boundary has been broken:

| Must not leak | Why |
| --- | --- |
| `uk.ac.manchester.tornado.api.TaskGraph` | Backend implementation detail; makes TornadoVM a compile dependency of every user. |
| `uk.ac.manchester.tornado.api.ImmutableTaskGraph` | Same. |
| `uk.ac.manchester.tornado.api.TornadoExecutionPlan` | Same, plus it owns device memory with its own lifecycle. |
| `uk.ac.manchester.tornado.api.GridScheduler` | Scheduling detail; not meaningful to an API user. |
| TornadoVM array types (`FloatArray`, `HalfFloatArray`, `IntArray`) | Backend storage. Currently leaks through `Sampler` and `InferenceCore` return types. |
| `GGUF`, `GGMLTensorEntry`, `MetadataValueType` | File-format objects; would pin the API to one format. |
| `GGMLType` | A format type used today as a runtime type tag — see [ADR-004](decisions/ADR-004-tensor-and-format-separation.md). |
| Backend-specific tensor handles (`TornadoTensor` and subclasses) | Device storage, not user data. |
| Internal state objects (`State`, `LlamaState`, `Qwen3State`, …) | Mutable inference internals with public fields; exposing them makes every field a compatibility commitment. |
| `Weights`, `StandardWeights`, `TornadoWeights` | Internal weight layout. |
| `org.beehive.gpullama3.Options` | A CLI record. It also mutates global system properties in its constructor. |
| `ModelType` | Internal dispatch identifier; see [rule 15](dependency-rules.md#rule-15--no-central-model-type-switches-for-new-architectures). |
| `TornadoVMMasterPlan`, `ForwardPlan` | Compiled-program internals of one backend. |

The exception is deliberate opt-in: a clearly separate, clearly named
Tornado-specific extension module may expose TornadoVM configuration to users who
explicitly want it. Depending on that module is a choice the user makes, not something
the generic API imposes.

## Logging policy

**The library never writes to `System.out` or `System.err`.** It emits through a
project-owned sink, no-op by default. Console output belongs to the CLI integration.

```java
public interface LogSink {
    void log(Level level, String message, Object... args);

    LogSink NOOP = (level, message, args) -> { };
}
```

**No external logging facade dependency.** `pom.xml` declares no logging dependency and
the project ships a shaded jar with native-image considerations; pulling SLF4J's
dependency surface into a self-contained inference library buys little. An SLF4J bridge
belongs in an integration module, for users who want it.

Enforced by [Rule 16](dependency-rules.md#rule-16--no-console-io-outside-the-cli-integration),
which ships with today's 20 offending files as a shrink-only allowlist.

## Compatibility note

`ModelLoader.loadModel(Path, int, boolean, boolean)` and
`Model.runInstructOnceLangChain4J(...)` are documented in the code as integration
points for LangChain4j and Quarkus. They are existing public surface. The façade in
[roadmap phase 2](migration-roadmap.md#m3--public-api-façade)
must be added alongside them, and they should be deprecated with a documented
replacement before any removal.

## Open questions

*Resolved during the ARCH review:* concurrency comes from engine batching, not parallel
plans (ARCH-07); the session holds a lease rather than owning the cache (ARCH-01);
`ModelInfo` exposes both dtypes (ARCH-15).

1. Is `ExecutionPolicy` a model-level or session-level choice?
3. How are chat templates exposed — is `ChatMessage` public API, or is prompt
   formatting an internal concern driven by model metadata?
4. Should `GenerationSession` expose a lower-level `forward(token, position)` for
   advanced users, or only `generate(...)`?
5. Do compiled programs have a user-visible cache with a user-visible key?
6. What is the module/artifact name for the Tornado-specific extension API?
