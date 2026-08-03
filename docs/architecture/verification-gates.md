# Verification Gates

**Status: proposed-normative.** Defines the three gate classes the roadmap's
"Definition of done" refers to, and makes the golden, parity, identity and benchmark
gates executable — inputs, environments, comparison rules, and what happens when
hardware is absent. M1 implements the machinery described here.

## Gate classes

| Class | Name | Runs | Needs | Trigger |
| --- | --- | --- | --- | --- |
| **A** | Ordinary CI gates | ArchUnit rules + allowlist staleness check, unit tests (GGML→DataType mapping, tokenizer, parsers, scheduler logic with a fake backend), link check on `docs/architecture` | JVM only — no model file, no accelerator, no TornadoVM device | every PR, `mvn test` |
| **B** | Accelerator qualification gates | golden logits, CPU↔GPU parity, compiled-program identity, CUDA-graph/lease eviction tests | pinned model fixture + pinned device tuple + TornadoVM installation | `mvn verify -Paccel-tests` on the pinned runner; required before merging any PR that touches an execution path |
| **C** | Release / milestone gates | benchmark gate against `perf-history.jsonl`, full model-matrix goldens, milestone acceptance checklist | Class B environment + performance history | milestone close and releases |

**`mvn test` never requires an accelerator or a model.** Class B is opt-in via the
`accel-tests` Maven profile; on a machine without the pinned tuple those tests **skip
with an explicit "environment absent" marker** — a skip is recorded, never reported as
a pass, and the milestone checklist requires a real Class B run, not a skipped one.

## Golden logits (M1.4)

- **Fixture:** `Llama-3.2-1B-Instruct` GGUF, FP16 and Q8_0 variants. Provenance: the
  file's SHA-256 is pinned in the test resources; the fixture itself is **not**
  committed (too large) — it is fetched to a local cache directory
  (`~/.gpullama3/test-models/` or `GPULLAMA_TEST_MODELS`), and the test fails with a
  download instruction if absent.
- **Goldens are committed.** Small binary logits + a JSON metadata sidecar per
  configuration.
- **What is captured:** fixed prompt (stated verbatim in the metadata), greedy
  sampling, `seed` irrelevant under greedy, 64 generated tokens. Compared tensors: the
  final logits row at the last prompt position and at each generated position 1..64,
  plus the emitted token ids.
- **Serialization:** raw little-endian float32 logits (`.f32le` blob) + metadata JSON:
  `{model_sha256, quantization, prompt, tokens_compared, backend, device_name, driver,
  tornadovm_version, build_commit, recover_bailout:false, created_by_commit}`.
- **Tuple pinning:** bit-exactness is asserted **only** on the pinned tuple (device,
  driver, TornadoVM version, backend, build flags). On any other tuple the golden test
  downgrades to the parity tolerance below and says so in its output.
- **Comparison:** bit-identical (`Float.floatToRawIntBits` equality) on the pinned
  tuple, **for configurations demonstrated reproducible on that tuple** (see below).
  **Any NaN/Inf in produced logits fails immediately**, before comparison —
  goldens must never contain NaN/Inf, and a NaN-vs-NaN "match" must not pass.

### Reproducibility is demonstrated, not assumed

Bit-exactness was originally written here as a property of the pinned tuple. That is
**not universally true on real hardware**: on the reference tuple Q8_0 reproduces
exactly while FP16 does not
([the FP16 determinism defect](HANDOFF.md#open-defect-fp16-logits-are-not-reproducible-run-to-run)).

Therefore the golden generator **measures** reproducibility — it captures each
configuration twice and compares — and records the outcome as `bit_exact` in the
golden's metadata. The gate then applies:

| `bit_exact` | Assertion |
| --- | --- |
| `true` | full bit-identical comparison of every compared row, as above |
| `false` | the **reproducibility-envelope gate** below, which is *provisional* |

A configuration may only carry `bit_exact: false` while a corresponding open defect is
recorded. This is a **temporary accommodation of a known defect, not a relaxed
standard**, and it must not be extended to new configurations silently.

### Reproducibility-envelope gate (provisional, FP16 only)

Applies where `bit_exact: false`. It bounds how far a non-reproducible configuration may
drift and asserts the properties that actually affect output. Over repeated captures on
the pinned tuple, all of the following must hold:

| Property | Bound |
| --- | --- |
| NaN/Inf | none, ever — checked before anything else |
| Max absolute drift | ≤ `1.0` per element |
| Max relative drift | ≤ `0.05` on elements with \|reference\| ≥ 1.0 (small logits are dominated by absolute noise) |
| Changed elements | recorded, not bounded — it is ~100% when this defect fires |
| **Argmax** | **must be identical** — greedy decoding must not change |
| **Top-k membership** | recorded for k=5 and k=10; **k=5 must be identical** |
| Token sequence | must be identical |

**Token equality alone is explicitly not sufficient.** On the reference tuple argmax and
top-5 survive but **top-10 membership already changes**, so greedy decoding hides a
defect that top-k/top-p sampling would expose. The envelope exists to make that visible
rather than to bless it.

**This gate is provisional.** Resolving the FP16 defect — or explicitly accepting the
behaviour with a recorded rationale — is a **blocker before M6**, which is where session
and KV-storage restructuring begins and where an unexplained numerical drift would become
impossible to distinguish from a refactor regression.
- **Execution flags:** always `-Dtornado.recover.bailout=False`
  ([C4](tornadovm-capabilities.md#c4--interpreter-bytecode-buffer-overflow-was-silent)).
- **Regeneration:** only via `scripts/regenerate-goldens.sh`, which refuses to run with
  a dirty working tree and writes the generating commit into the metadata. The commit
  that regenerates goldens must change nothing else and must say why in its message.
  Never regenerated automatically on failure.

## CPU↔GPU parity (M1.5)

- Same fixture, same prompt. Tolerance per element:
  `|got − ref| ≤ 1e-2 · Σ|wᵢaᵢ| + 1e-3` (the ARCH-06 bound); the reference is the CPU
  path. NaN/Inf on either side fails.
- Runs on both backends of the pinned tuple; cross-device tuples compare with the same
  tolerance, never bit-exact.

## Compiled-program identity (M1.6)

Defines exactly what "identity" observes, in one process:

1. compile once; record: number of task graphs, ordered task names, grid-scheduler
   entry set, and a SHA-256 over the generated kernel source of every task
   (deterministic from the Phase 0 floor,
   [C3](tornadovm-capabilities.md#c3--generated-kernel-source-was-non-deterministic-before-52));
2. decode ≥ 100 tokens;
3. assert all recorded values unchanged and that no additional compilation occurred
   (compile-time counters via the profiler result are zero after step 1).

This is the structural half of Rule 13; the benchmark gate is the behavioural half.

**Compilation identity is independent of numerical determinism.** The two must not be
conflated: a configuration can compile to a byte-identical program and still produce
drifting logits, which is exactly the FP16 situation. So:

- steps 1–3 above assert **compilation identity** directly — task-graph count, ordered
  task names, grid-scheduler entries, per-task kernel-source SHA-256, and zero
  recompilation after warm-up. These are asserted for **every** configuration, FP16
  included, because they do not depend on the numerics being reproducible.
- the **bit-exact numerical** half of the identity claim is carried by **Q8_0**, which is
  demonstrated reproducible on the pinned tuple.
- FP16 additionally runs the reproducibility-envelope gate.

An FP16 identity failure therefore means the *program* changed, which is a real
regression, and cannot be dismissed as the known logits drift.

## Benchmark gate (M1.7)

- **Record schema:** existing `perf-history.jsonl` fields **plus** `machine`, `gpu`,
  `tornadovm_version`, `cache_warm` (the current schema has none of these — verified
  2026-08-03).
- **Tuple:** (machine, gpu, model, quantization, backend, configuration,
  tornadovm_version). Comparisons only ever happen within one tuple.
- **Procedure:** 3 warm-up generations discarded, then 5 measured runs; metric is
  decode `eval_rate` (tok/s); aggregate is the **median** of the 5.
- **Baseline:** the most recent green (gate-passing) entry of the same tuple.
- **Tolerance:** stated per tuple in `scripts/perf-gate-tolerances.json`; default 3% on
  the pinned self-hosted runner. Shared-CI tuples are record-only (no gate) — a 3% gate
  over heterogeneous runners is noise.
- **Missing baseline** (new tuple, or first run after a TornadoVM version change):
  record-only pass, entry becomes the new baseline. C5 makes cross-version comparison
  meaningless by design.
- **Noisy baseline:** if the 5-run spread (max−min)/median exceeds 10%, the gate
  reports "unstable environment" and neither passes nor records a baseline.
- **Cache warm/cold:** `cache_warm` records whether the on-disk cubin cache was warm.
  Throughput comparisons use warm runs; TTFT/start-up comparisons record both and
  compare warm-to-warm and cold-to-cold only.

## Milestone acceptance mapping

The roadmap's Definition of done maps onto the classes as:

1. goldens bit-identical → Class B, pinned tuple;
2. benchmark gate → Class C on the milestone-closing commit;
3. ArchUnit allowlists shrank or equal → Class A (CI also fails on stale entries);
4. deprecations-not-deletions → review checklist item, Class A javadoc check where
   expressible.
