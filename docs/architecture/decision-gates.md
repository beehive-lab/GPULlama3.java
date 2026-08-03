# Decision Gates

**Status: normative index, living document.** Every open architectural question that a
milestone needs answered is listed here with the point by which it must be decided. A
milestone task whose gate is open must not start. Deciding a gate earlier than its
"required before" point is optional; deciding it later blocks the milestone.

A gate is closed by a maintainer decision recorded in an ADR, an ADR amendment, or an
explicit note in the pull request that implements it — never silently by code. When a
gate closes, update its Status here and the document that carries the question.

Recommendations shown in the Status column are proposals from the
[roadmap-hardening pass](review/roadmap-hardening-2026-08-03.md); they are not decisions.

## Gate table

| ID | Decision | Required before | Owner | Status | Document / ADR |
| --- | --- | --- | --- | --- | --- |
| D-01 | Is `LocalModel` generation-neutral, with generation exposed via a capability interface (`TextGenerationModel extends LocalModel` carrying `newSession()`)? | M3.1 | maintainers | Open — recommended: yes, capability interface | [`public-api.md`](public-api.md#localmodel--the-loaded-model), Rule 14, ADR-001 |
| D-02 | Is `ExecutionPolicy` model-level, session-level, or model defaults + session overrides? | M10.2 (façade v1 omits policy per ADR-007 D3) | maintainers | Open — recommended: model default + session override | [`public-api.md`](public-api.md#modeloptions--sessionoptions), target-arch OQ 3 |
| D-03 | How do `prompt`, `systemPrompt`, `messages` and chat templates interact — is `ChatMessage` public API or is formatting internal, driven by model metadata? | M3.1 | maintainers | Open — recommended: v1 exposes `prompt` + `systemPrompt` only; `messages(...)` deferred until the template decision | [`public-api.md`](public-api.md#generationrequest--generationresult) |
| D-04 | Is backend/device selection part of the initial façade? | M3.1 | maintainers | Open — recommended: no; added in M12.1 (ADR-007 D3) | [`public-api.md`](public-api.md), ADR-007 |
| D-05 | Is the compiled-program cache visible to ordinary users (user-visible cache, user-visible key)? | M9.2 (hidden in façade v1) | maintainers | Open — recommended: internal in v1; key = (architecture, configuration shape, policy, backend, device) | ADR-002 OQ 4/5, [`public-api.md`](public-api.md#open-questions) |
| D-06 | Is a lower-level `forward(token, position)` public on the session? | M3.1 | maintainers | Open — recommended: not in v1; revisit with the embeddings use case | [`public-api.md`](public-api.md#open-questions) |
| D-07 | Experimental API compatibility policy: what may break between releases while the M3.4 marker is on, and how is breakage communicated? | M3.1 | maintainers | Open — recommended: annotated `@Experimental`; breaking changes allowed until M13 with a CHANGELOG entry and a deprecation shim where cheap | [`public-api.md`](public-api.md), roadmap M3.4/M13 |
| D-08 | `TensorDescriptor` shape: full shape, or element count + layout tag? How are quantization block parameters (block size, scale layout) expressed? | M4.3 | maintainers | Open | ADR-004 OQ 3/4 |
| D-09 | `DataType` value set: only `{F32, F16, Q8_0}` (what executes), or K-quants as load-time-only values? | M4.1 | maintainers | Open — recommended: execution types only; K-quants stay `GGMLType`, collapsed by the M4.2 mapping | ADR-004 OQ 1 |
| D-10 | Who owns the `KvCacheManager` when `newSession()` is used without an explicit engine: does the simple path run through a hidden single-slot engine, or bypass the engine with a model-scoped manager? | M6.1 | maintainers | Open — recommended: model-scoped single-lease manager in M6 (no engine dependency), superseded by the engine's manager when an engine is attached in M7 | ADR-006 OQ 5, [`ownership-and-lifecycle.md`](ownership-and-lifecycle.md) |
| D-11 | Does `LocalModel.close()` with live sessions throw or force-close? | M3.1 (user-visible close semantics) | maintainers | Open — ADR-001 leans towards throwing with a clear message | ADR-001 OQ 3, [`ownership-and-lifecycle.md`](ownership-and-lifecycle.md) |
| D-12 | May one `CompiledProgram` be invoked concurrently, or is `invoke(...)` single-threaded with the engine as the only multiplexer? | M6.4 (buffer binding design), enforced by M7.1 | maintainers | Open — recommended: `invoke(...)` not thread-safe; sessions sharing a program serialize; the engine's `step()` is the single caller in batched mode | ADR-001 OQ 1, [`ownership-and-lifecycle.md`](ownership-and-lifecycle.md) |
| D-13 | KV block size (fixed vs per-model) and pool-exhaustion policy (refuse, preempt, swap to host) | M7.1 (block size), M7.2 (exhaustion) | maintainers | Open — recommended: fixed global block size v1; exhaustion = queue at admission, no preemption v1 | ADR-005 OQ 1/2 |
| D-14 | Block table: device array walked in-kernel (as #129) or host-side indirection resolved before launch? | M6.2 | maintainers | Open — recommended: in-kernel device array; the only shape compatible with CUDA-graph capture (C1) already proven by #129 | ADR-005 OQ 5 |
| D-15 | Does `step()` block? Does the engine own a background thread, and who owns its lifecycle? | M7.4 | maintainers | Open — recommended: `step()` blocks for one batched iteration; no engine-owned thread in v1; callers drive the loop | ADR-006 OQ 2, [`engine-contract.md`](engine-contract.md) |
| D-16 | Scheduling baseline and preemption policy | M7.2 | maintainers | Open — recommended: FCFS admission, no mid-sequence preemption in the minimum viable engine | ADR-006 OQ 1, [`engine-contract.md`](engine-contract.md) |
| D-17 | Batch sizing: fixed B or adaptive to free blocks? | M7.2 | maintainers | Open — recommended: fixed B v1 (matches #129's static mode), adaptive as an M7-ext follow-up | ADR-006 OQ 3 |
| D-18 | Ragged batching: supported shape for mixed-length prompts/decodes in one step | M7.2 | maintainers | Open — #129 notes same-length prompts for the static case | ADR-006 OQ 6, [`engine-contract.md`](engine-contract.md) |
| D-19 | One engine per model, or one engine over several models on one device? | M7.4 | maintainers | Open — recommended: one engine per (model, device) in v1 | ADR-006 OQ 4 |
| D-20 | Prefill and decode: two `InferenceProgram`s, or one program with phase-selectable components? | M9.2 | maintainers | Open — today's `PrefillDecodeForwardPlan` shares N+2 graphs, which argues for one program | ADR-002 OQ 1 |
| D-21 | Do the loader SPI and the architecture-description SPI share one interface or two? | M5.1 | maintainers | Open — programs are needed by the backend, loading by the format layer; they may want to be separate | target-arch OQ 5, ADR-002 |
| D-22 | Where does dequantization live — loading (descriptor materialization) or operations? | M8.2 | maintainers | Open | ADR-004 OQ 2 |
| D-23 | `ProgramSignature` contents and validation point (compile time, invocation time, both) | M9.2 | maintainers | Open | ADR-002 OQ 2 |
| D-24 | Callback threading and callback-failure policy for `onToken` and engine delivery | M7.4 (engine), M3.2 for the session's `onToken` | maintainers | Open — recommended: callbacks run on the calling thread of `generate()`/`step()`; a throwing callback cancels the request | [`engine-contract.md`](engine-contract.md) |

## Closed gates

Recorded so they are not reopened.

| Decision | Answer | Where |
| --- | --- | --- |
| Concurrency mechanism on one device | Batching many sequences into one invocation of one compiled program; not one plan per session (weight duplication, C2) | ADR-001, ADR-006 |
| KV storage ownership | Engine-scoped manager; sessions hold leases | ADR-005 |
| Block pool shape | One persistent device array, in-kernel indexing (C1 invariant) | ADR-005 |
| Ordered components vs graph IR | Ordered components | ADR-002 |
| Sampling in the backend | Allowed — it is an operation (Rule 8b) | ADR-002, Rule 8b |
| `ModelInfo` dtype exposure | Both weight and compute dtype, as runtime `DataType` | ADR-004 |
| Logging | Project-owned sink, no external facade | Rule 16, public-api.md |
