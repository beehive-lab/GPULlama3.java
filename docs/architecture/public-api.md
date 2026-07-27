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

    ModelInfo info();                  // architecture, name, quantization, context length
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
/** One sequence. NOT thread-safe. Owns KV cache and activation buffers. */
public interface GenerationSession extends AutoCloseable {

    GenerationResult generate(GenerationRequest request);

    /** Current sequence position; how much context is consumed. */
    int position();

    /** Discard sequence state and start fresh without reallocating. */
    void reset();

    @Override void close();            // releases KV cache and buffers
}
```

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

## Compatibility note

`ModelLoader.loadModel(Path, int, boolean, boolean)` and
`Model.runInstructOnceLangChain4J(...)` are documented in the code as integration
points for LangChain4j and Quarkus. They are existing public surface. The façade in
[roadmap phase 2](migration-roadmap.md#phase-2--public-api-façade-over-current-implementation)
must be added alongside them, and they should be deprecated with a documented
replacement before any removal.

## Open questions

1. Do sessions support concurrent invocation on one device, or is invocation
   serialized?
2. Is `ExecutionPolicy` a model-level or session-level choice?
3. How are chat templates exposed — is `ChatMessage` public API, or is prompt
   formatting an internal concern driven by model metadata?
4. Should `GenerationSession` expose a lower-level `forward(token, position)` for
   advanced users, or only `generate(...)`?
5. Do compiled programs have a user-visible cache with a user-visible key?
6. What is the module/artifact name for the Tornado-specific extension API?
