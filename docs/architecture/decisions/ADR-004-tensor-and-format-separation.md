# ADR-004: Tensor and format separation

## Status

**Accepted** — 2026-07-30, following the ARCH-05 and ARCH-15 review on PR #140.

Amended on acceptance:
- `ModelInfo` exposes **both** weight dtype and compute dtype, since
  `effectiveGpuWeightType` already collapses K-quants to Q8_0 for execution.
- Which `DataType` values are reachable depends on the TornadoVM version: `FP8Array`
  requires 5.1.0 and `BFloat16Array` requires 5.2.0, neither of which exists in the
  pinned 5.0.0. See [the capability ledger](../tornadovm-capabilities.md).

## Context

`GGMLType` is a GGUF/GGML **file format** enum. It enumerates how tensors are stored in
a `.gguf` file: `F32`, `F16`, `Q4_0`, `Q8_0`, `Q4_K`, `Q6_K`, `IQ2_XXS`, `BF16` and so
on, with per-type block sizes and byte sizes.

In this repository it is also the runtime type tag:

- `inference.weights.Weights.getWeightType()` returns `GGMLType`;
- `tensor.standard.FloatTensor` exposes `GGMLType`, and its subclasses are named after
  GGML types (`Q4_KFloatTensor`, `Q6_KFloatTensor`, …);
- `tensor.tornado.TornadoTensor.type()` returns `GGMLType`;
- `tornadovm.plan.ForwardPlanFactory.create(...)` takes `GGMLType` as its **first
  dispatch axis**, before model family and execution mode.

The `tensor` package holds both concerns together: `GGUF`, `GGMLTensorEntry`,
`MetadataValueType` and `GGMLType` (file format) live alongside `tensor/standard/**`
and `tensor/tornado/**` (runtime storage).

The mapping between the two is already not one-to-one, and the code already knows it.
`AbstractModelLoader.effectiveGpuWeightType` collapses `Q4_K`, `Q5_K` and `Q6_K` to
`Q8_0` for GPU execution, and `getModelQuantization` maps GGUF file-type integers 14–18
to the string `"Q8_0"`. So there are already file types that are not execution types —
the distinction exists in behaviour but not in the type system.

Consequences:

- adding a second file format would mean either extending `GGMLType` with non-GGML
  entries or converting everything to GGML types at load;
- a format enum appears in operation and plan signatures, where it means "execution
  precision" rather than "file encoding";
- the runtime cannot express an execution representation that has no GGUF equivalent;
- `tensor` cannot be understood without understanding GGUF.

## Decision

Separate file representation from execution representation.

1. **Runtime tensor abstractions are independent of GGUF.** No type in the runtime,
   program, operation or backend layers references `GGMLType`, `GGUF` or
   `GGMLTensorEntry`.
   ([Rule 4](../dependency-rules.md#rule-4--gguf-is-a-format-concern-not-a-tensor-or-operation-concern).)

2. **GGUF and GGML types describe file representation.** They live in a format package
   and are used by the loading path only. `GGUF`, `GGMLTensorEntry`,
   `MetadataValueType` and `GGMLType` move together.

3. **A runtime `DataType` describes execution representation.** Owned by the runtime
   layer. It enumerates what the engine actually computes with — at minimum today:
   `F32`, `F16`, `Q8_0` (with its block structure and scales). It is deliberately
   smaller than `GGMLType`, because most GGML types are never executed directly.

4. **Tensor descriptors are separate from tensor storage.** A descriptor says
   *what* — dtype, element count / shape, layout, role. Storage says *where* — a host
   `MemorySegment`, a TornadoVM native array, a device buffer. Descriptors are neutral;
   storage is backend-owned.

5. **Loading is an explicit mapping.** The format layer maps format tensors to runtime
   descriptors and asks the backend to materialize storage:

   ```
   GGMLTensorEntry  ──map──▶  TensorDescriptor  ──materialize──▶  backend storage
   (file: GGMLType,           (runtime: DataType,                 (TornadoVM native
    shape, MemorySegment)      shape, layout)                      array, or host
                                                                   MemorySegment)
   ```

   The existing `Q4_K`/`Q5_K`/`Q6_K` → `Q8_0` collapse in
   `AbstractModelLoader.effectiveGpuWeightType` becomes a documented, testable part of
   this mapping instead of an implicit rule.

6. **Generic operation APIs accept neither GGUF entries nor backend-specific storage
   handles.** An operation is parameterized by `DataType` and descriptors. A backend
   resolves those to its own storage internally.

## Consequences

Positive:

- A second file format (safetensors, a project-specific format, a memory-resident
  source) becomes addable without touching runtime or operation code.
- `ForwardPlanFactory`'s first dispatch axis becomes "what precision do we execute at",
  which is what it actually means.
- The runtime can express execution representations with no file-format equivalent —
  packed half2 layouts, for example, are an execution concern.
- `tensor/` stops meaning two things.
- Removes the 8 `model/loader` files and the `tensor` GGUF file from the
  [rule 1](../dependency-rules.md#rule-1--tornadovm-stays-in-the-tornado-backend)
  allowlist, since loaders stop constructing TornadoVM arrays directly.

Negative / costs:

- `Weights.getWeightType()`, the `FloatTensor` hierarchy and `TornadoTensor` are public
  and change shape.
- The `Q4_KFloatTensor` / `Q6_KFloatTensor` CPU classes decode GGML block formats
  directly; they are inherently format-aware. They either stay format-coupled behind a
  `DataType`-typed interface, or the dequantization moves into loading. Both have
  costs — see open questions.
- One more indirection at load time. Load time is already tracked
  (`RunMetrics.setLoadDuration`) and must not regress; the descriptor layer must not
  add a copy.
- Two enums where there was one, and reviewers must know which is which. The naming has
  to be unambiguous (`DataType` vs `GGMLType` is probably enough).

## Alternatives considered

**Keep `GGMLType` as the single type system.** Zero cost today. Rejected: it makes GGUF
a permanent dependency of the runtime and every operation signature, and it cannot
express execution-only layouts.

**Runtime `DataType` as a strict subset of `GGMLType`, sharing names and ordinals.**
Tempting, and the mapping would be mostly trivial. Rejected: the existing K-quant
collapse already breaks the subset relation, and sharing names invites treating them as
interchangeable — which is the current problem with extra steps.

**Convert everything to a single execution type at load (e.g. always dequantize to
F16).** Simple runtime, and it is already partially what happens for K-quants.
Rejected: Q8_0 execution is a supported and measured path; forcing dequantization would
increase memory use and change performance characteristics.

**Full shaped tensors with strides, views and broadcasting.** Rejected as scope. The
CPU `FloatTensor` is explicitly documented as "over-simplified, shapeless" and the
kernels work with flat offsets. A descriptor needs enough shape information to size
buffers and validate bindings, not a full shaped-tensor algebra. Adding one would be a
step toward the general tensor library that
[`vision.md`](../vision.md#non-goals) rules out.

## Migration notes

Corresponds to [roadmap phase 4](../migration-roadmap.md#m4--datatype-and-gguf-isolation).
Independent of [ADR-001](ADR-001-model-session-separation.md)'s state work, so the two
can run in parallel.

Suggested order:

1. Introduce `DataType` in the runtime layer alongside `GGMLType`. Additive.
2. Add the explicit `GGMLType → DataType` mapping, seeded from
   `AbstractModelLoader.effectiveGpuWeightType` and `getModelQuantization`. Test it
   directly — this is the first time that logic becomes visible.
3. Add `TensorDescriptor`. Have loaders produce descriptors, then materialize storage.
4. Change `Weights`, `FloatTensor` and `TornadoTensor` to expose `DataType`. Deprecate
   the `GGMLType` accessors rather than deleting them.
5. Switch `ForwardPlanFactory`'s first dispatch axis to `DataType`.
6. Move `GGUF`, `GGMLTensorEntry`, `MetadataValueType`, `GGMLType` to the format
   package.
7. Enable rule 4 and shrink its allowlist to empty.

Verification at every step: identical output for a fixed seed, unchanged model load
time, unchanged resident memory. Load time and memory are the metrics at risk here, not
throughput.

## Open questions

1. What exactly is in `DataType`? Just `{F32, F16, Q8_0}` (what executes today), or
   also the K-quant types as *load-time-only* values?
2. Where does dequantization live — in loading (descriptor materialization) or in
   operations? Today `Q4_KFloatTensor` and friends dequantize during CPU compute, while
   the GPU path dequantizes K-quants at load.
3. Does `TensorDescriptor` carry a shape, or an element count plus a layout tag?
   `FloatTensor` is shapeless today and the kernels index flat.
4. How are quantization block parameters (block size, scale layout) expressed —
   part of `DataType`, or a separate layout descriptor? Packed half2 and Q8_0 scales
   both need this.
5. Do the CPU `*FloatTensor` classes stay format-named, or get renamed to execution
   terms? They decode GGML block layouts directly, so their names are honest — but they
   then sit oddly in a format-free runtime.
6. Should `Weights` become a map of descriptors keyed by role, rather than a fixed set
   of named fields per family? That would decouple weight layout from architecture, but
   it is a larger change than this ADR requires.
