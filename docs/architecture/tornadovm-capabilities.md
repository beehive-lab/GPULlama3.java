# TornadoVM Capability Ledger

**Status: normative reference.** Every architectural position in this directory is grounded in a
capability listed here. A proposal that depends on a capability absent from this table is not grounded
and must either be re-grounded or raised as an upstream TornadoVM proposal.

**Why this document exists.** The baseline was originally written without checking which capabilities
were reachable from the version the project builds against. Several positions turned out to assume
capabilities that exist in a newer TornadoVM than the one pinned in `pom.xml`. The **Minimum version**
column is the fix: it makes "we can do this" and "we can do this *today*" different claims.

## Version situation

| | Version |
| --- | --- |
| Pinned in `pom.xml` (`tornadovm.base.version`) | **5.0.0** |
| Verified local development tree | 5.2.1-jdk21-dev |
| Required by this architecture | **≥ 5.2.x** — see [migration-roadmap.md § Phase 0](migration-roadmap.md#phase-0--tornadovm-version-floor) |

Adopting ≥ 5.2.x is Phase 0 of the roadmap, ahead of all other work.

## Available capabilities

Verified against the local tree. "Min version" verified by checking the type's presence in the release
tags `v5.0.0-jdk21`, `v5.1.0-jdk21`, `v5.2.0-jdk21`.

| Capability | API | Min version | Serves | Used today? |
| --- | --- | --- | --- | --- |
| Execution profiling | `TornadoExecutionPlan.withProfiler(ProfilerMode)`; `TornadoExecutionResult.getProfilerResult()` → `getDeviceKernelTime()`, `getDeviceWriteTime()`, `getDeviceReadTime()`, `getDataTransfersTime()`, `getTotalBytesCopyIn/Out()`, `getTotalDeviceMemoryUsage()`, `getCompileTime()`, `getKernelDispatchTime()` | 5.0.0 | ADR-006, metrics seam, benchmark gate | **No — zero references** |
| Device enumeration | `TornadoDeviceMap` — `getNumBackends()`, `getAllBackends()`, `getDevicesByType()`, `getBackendsWithDevicePredicate()` | 5.0.0 | `DeviceSelector`, multi-device seam | No |
| Device memory query | `TornadoTargetDevice.getDeviceGlobalMemorySize()`, `getDeviceMaxAllocationSize()`, `getDeviceLocalMemorySize()` | 5.0.0 | Memory planning, admission | No |
| Plan memory limit | `withMemoryLimit(String)` / `withoutMemoryLimit()`; `memory/` — `TornadoMemoryProvider`, `XPUBuffer`, `DeviceBufferState` | 5.0.0 | Backend SPI allocation seam | No |
| Explicit device placement | `withDevice(TornadoDevice)`; **per-task** `withDevice(String taskName, TornadoDevice)`; `withConcurrentDevices()` | 5.0.0 | Multi-device seam | No |
| Concurrent execution plans | Two `TornadoExecutionPlan` instances in separate threads over the same immutable graph. Exercised by `tornado-unittests/.../multithreaded/TestMultiThreadedExecutionPlans` (4 tests, 0 failures, CUDA/RTX 4090) | 5.0.0 | ADR-006 concurrency | No |
| Native library tasks | `TaskGraph.task(...)` with a library binding factory (`CuBlas::cublasSgemv` style); modules `tornado-cublas`, `tornado-cudnn` | 5.0.0 | ADR-003 library dispatch | PR #131 only |
| Tensor-core MMA | `enums/MMAShape` — `M16N8K16` (fp16, bf16), `M16N8K32` (int8, FP8 E4M3/E5M2) | 5.0.0 | Batched prefill/decode | Yes |
| `Int8Array` | `types/arrays/Int8Array` | 5.0.0 | Q8_0 storage | Yes |
| `HalfFloatArray` | `types/arrays/HalfFloatArray` | 5.0.0 | FP16 paths | Yes |
| **`FP8Array`** | `types/arrays/FP8Array` | **5.1.0** | FP8 execution paths | No |
| **`BFloat16Array`** | `types/arrays/BFloat16Array` | **5.2.0** | BF16 models (PR #120) | No |
| CUDA graph capture | `withCUDAGraph()` | 5.0.0 | Decode replay | Yes (`llama.cudaGraphs`) |
| Selective graph execution | `withGraph(int)`, `withAllGraphs()` | 5.0.0 | Prefill/decode phase skipping | Yes |
| Pre-compilation | `withPreCompilation()` | 5.0.0 | Compile-once property | Yes |
| Intra-plan concurrency | `withIntraPlanConcurrency()` | 5.0.0 | Within one plan only | No |

## Runtime behaviours that constrain the design

Capabilities that exist but whose behaviour dictates a design choice. Each names the document it
constrains. TornadoVM issue numbers are given so the constraint can be re-checked when the version moves.

### C1 — CUDA graph capture fixes device addresses

`withCUDAGraph()` bakes device addresses into the captured graph. Re-pointing a captured buffer between
replays fails at replay with `CUresult=700`, and because `tornado.recover.bailout` defaults to `TRUE`
(`TornadoOptions.RECOVER_BAILOUT`), the first symptom is **wrong output rather than an error**.
(TornadoVM #1006.)

**Constrains:** [ADR-005](decisions/ADR-005-kv-cache-ownership-and-leases.md). A KV block pool must be
one persistent pooled array with in-kernel `blockTable` indexing. Handing a slot a different device
buffer per step is incompatible with graph capture. Leased blocks must be pinned against eviction.

### C2 — Device buffers are per task graph

Each `TaskGraph` owns its own device buffer state, so the same Java object referenced by two graphs gets
**two device buffers**. (This is why TornadoVM #996 had to make cross-graph aliasing explicit for
`consumeFromDevice`.)

**Constrains:** [ADR-006](decisions/ADR-006-engine-tier.md) and `public-api.md`. Concurrent plans are
*possible* (see the ledger row above) but duplicate the weights per plan — roughly 3.4 GB per concurrent
session for a 3B-Q8 model. Device concurrency therefore comes from **batching inside one compiled
program**, as an economic choice, not because the API is missing.

### C3 — Generated kernel source was non-deterministic before 5.2

`emitVariableDefs` used hash containers keyed by identity, so generated source declaration order varied
between JVM runs. Fixed and verified byte-identical (TornadoVM #999).

**Constrains:** [ADR-002](decisions/ADR-002-program-and-compiled-program.md) and the compiled-program
identity test. A stable hash of generated source only holds from that fix onward — another reason the
version floor is Phase 0.

### C4 — Interpreter bytecode buffer overflow was silent

The interpreter's bytecode buffer was a fixed 4096 bytes; a graph of roughly 50 tasks overflowed it and
the failure was swallowed by `recover.bailout`, producing truncated bytecode and silently wrong results
(TornadoVM #1004, fixed — now sized from task and object counts).

**Constrains:** the golden-test suite must run with **`-Dtornado.recover.bailout=False`**, because the
default swallows exactly the failure class that state-motion refactoring risks. Also relevant to the
engine tier, which composes B slots × N layers per step — the shape that overflowed.

### C5 — Performance history has version-sized discontinuities

Two merged runtime changes moved GPULlama3 numbers by more than any refactor would:

- **#1002** — the interpreter eagerly allocated and cleared an `int[dependencies][32768]` wait-event
  matrix on every execution. Fixed: **53 → 103 tok/s**, identical output.
- **#1008** — on-disk cubin cache: start-up **11.5 s → 5.2 s**.

**Constrains:** the benchmark gate. The comparison tuple must include **TornadoVM version** and
**cubin-cache warm/cold**, not only (machine, model, quantization, backend, configuration). A 1.94× step
change already sits inside the existing `docs/perf-history.jsonl` entries, so records from before the
floor are not a baseline.

### C6 — CUDA compiler flags were not reaching NVRTC

`withCompilerFlags(TornadoVMBackendType.PTX, ...)` and `-Dtornado.cuda.compiler.flags` were accepted and
then dropped in the CUDA JNI layer (TornadoVM #1010 fixes this and adds `default|fast|debug|repro`
profiles; `debug` = `-lineinfo`, which is what makes `nsys` attribute samples to generated CUDA C).

**Constrains:** any diagnostics work assuming NVRTC flags are reachable. Treat as unavailable until the
fix is in the adopted version.

## Missing — genuine upstream proposals

| Gap | Needed for | Status |
| --- | --- | --- |
| **FP4 / MXFP4 / NVFP4** — no native array type, no `MMAShape` entry, no PTX codegen. `types/arrays/` stops at `FP8Array`; `MMAShape` stops at `M16N8K32`. | The FP4 quantization goal named in the review | **The one genuine upstream ask.** Raise as a TornadoVM proposal. Explicitly **off** this roadmap's critical path |
| Per-task profiler attribution | A sharper benchmark gate than plan-level device time | Nice-to-have. Not blocking — plan-level timings plus `perf-history.jsonl` suffice |

Previously listed here and **removed**: *concurrent independent execution plans on one device*. They
exist and are tested (see the ledger row and C2). The reason to batch is buffer duplication, not a
missing API.

## Maintaining this document

- Adding a position that depends on TornadoVM behaviour means adding or citing a row here.
- When the pinned version changes, re-verify the **Min version** column against the release tags.
- Constraint entries (C1–C6) name an upstream issue so they can be re-tested rather than assumed.
