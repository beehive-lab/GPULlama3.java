# Vision

**Status:** normative for scope. API examples are **proposals** — see
[`public-api.md`](public-api.md).

## Mission

> A Java-native transformer inference framework that compiles reusable Java inference
> components through TornadoVM into heterogeneous execution plans for local
> accelerators.

GPULlama3.java today is a working GPU-accelerated LLM engine. The intent is to keep
that engine and grow a framework around it: a set of building blocks Java developers
can use to run local inference on GPUs without writing GPU code and without leaving
Java.

## Target users

1. **Java application developers** who want local model inference in a JVM
   application. They want to load a model and generate text. They should not need to
   know what a task graph is.
2. **Java developers integrating inference into a stack** — Quarkus, Spring,
   LangChain4j, Micronaut, plain server processes. They need session lifecycle,
   concurrency behaviour and resource ownership to be predictable.
3. **Advanced Java developers and performance engineers** who want to control device
   selection, execution policy, batching and memory behaviour, and to add new model
   architectures.
4. **Researchers and contributors** extending the framework with new architectures,
   operations or backends.

Users 1 and 2 use the high-level API. Users 3 and 4 use the lower-level API. Both are
Java; neither writes CUDA or OpenCL.

## Primary use cases

Current focus:

- local generative language model inference (chat, instruct, completion);
- streaming token generation;
- prompt prefill and autoregressive decode;
- reuse of a loaded model across multiple concurrent sessions;
- embedding inference into existing JVM services.

The design should remain extensible to, without being built for them now:

- embeddings;
- text classification;
- reranking;
- encoder-only transformers;
- encoder–decoder models;
- vision transformers;
- multimodal pipelines.

"Extensible to" means the core abstractions must not *assume* token generation, a
tokenizer, a KV cache, or a prefill/decode split. See
[`dependency-rules.md`](dependency-rules.md) rule 14.

## Non-goals

The framework is **not**:

- a training framework;
- an autograd system;
- a general-purpose tensor compiler;
- a replacement for TornadoVM's compiler or runtime;
- a distributed / multi-node serving system;
- a model conversion or quantization toolchain;
- a universal ONNX-style operator zoo.

Explicitly out of scope: building a tensor IR, a loop IR, or backend code generation
inside GPULlama3. That is TornadoVM's job. See
[ADR-003](decisions/ADR-003-tornado-backend-boundary.md).

## Why TornadoVM is the differentiator

Java has no shortage of ways to call a native inference runtime. What Java does not
have is a way to write inference logic *in Java* and have it run on a GPU.

TornadoVM provides exactly that: it JIT-compiles annotated Java methods to PTX, OpenCL
and SPIR-V, manages device memory, and executes task graphs across heterogeneous
devices. The transformer kernels in
`src/main/java/org/beehive/gpullama3/tornadovm/kernels/` are ordinary Java.

That gives this project a position other Java inference stacks do not have:

- inference logic stays in Java, debuggable and modifiable in Java;
- one Java implementation targets NVIDIA (PTX), OpenCL and SPIR-V devices, rather
  than one hand-written kernel set per backend;
- no JNI/FFI boundary around the hot path;
- the JIT can specialize to the device actually present.

The framework's value is therefore *the layers above TornadoVM* — model, session,
program, state, operation and developer API — not a second compiler. Duplicating
TornadoVM would remove the only thing that makes this stack distinct.

This document makes no claim about how GPULlama3.java performs relative to any other
runtime. Performance claims belong in benchmark reports, not in architecture docs.

## Product-level example (proposed)

The names below are **proposed** and do not exist in the repository today.

```java
try (LocalModel model = LocalModels.load(Path.of("Llama-3.2-1B-Instruct-FP16.gguf"))) {

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

Properties this example is meant to establish:

- one loaded model, many sessions;
- the session, not the model, owns conversation position and KV cache;
- both are `AutoCloseable`, because both own device memory;
- nothing TornadoVM-specific is visible.

## Advanced example (proposed)

The lower-level API exposes programs, backends and devices — still without exposing
TornadoVM types.

```java
Backend backend = Backends.select(DeviceSelector.preferGpu());

try (LocalModel model = LocalModels.loader(modelPath)
        .backend(backend)
        .options(ModelOptions.builder()
                .contextLength(4096)
                .executionPolicy(ExecutionPolicy.prefillDecode(256))
                .build())
        .load()) {

    // A backend-neutral description of the inference work.
    InferenceProgram program = model.architecture().createDecodeProgram(model.configuration());

    // Compiled once, reused for every token of every session.
    CompiledProgram compiled = backend.compile(program, model.compileOptions());

    try (GenerationSession session = model.newSession(compiled)) {
        session.generate(GenerationRequest.builder().prompt("...").build());
    }
}
```

`Backend`, `InferenceProgram` and `CompiledProgram` are backend-neutral types.
`TaskGraph`, `ImmutableTaskGraph`, `GridScheduler` and `TornadoExecutionPlan` appear
only inside the TornadoVM backend implementation. Tornado-specific tuning, if needed,
is reached through an explicitly optional backend-specific API — not through the
generic one.

## What this vision does not claim

- It does not claim the API above exists. It does not.
- It does not claim the current code is already layered this way. It is not; see
  [`current-architecture.md`](current-architecture.md).
- It does not claim portability beyond what TornadoVM supports on the devices the
  project actually tests.
