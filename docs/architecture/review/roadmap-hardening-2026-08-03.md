# Roadmap-hardening review — 2026-08-03

Consistency and decision inventory over `docs/architecture/` (baseline v1.0 +
ARCH-01..19 fold-in, commit `0759e5c`), followed by the resolutions applied. Companion
outputs: [ADR-007 (Proposed)](../decisions/ADR-007-roadmap-ordering-and-transitional-contracts.md),
[`decision-gates.md`](../decision-gates.md), [`execution-backlog.md`](../execution-backlog.md),
[`verification-gates.md`](../verification-gates.md),
[`ownership-and-lifecycle.md`](../ownership-and-lifecycle.md),
[`engine-contract.md`](../engine-contract.md).

Nothing here changes an accepted ADR's text or status. Resolutions that amend milestone
objectives/acceptance are collected in ADR-007, status **Proposed**, per the roadmap's
own rule that such changes need an ADR.

## Findings

| # | Finding | Class | Resolution |
| --- | --- | --- | --- |
| F-01 | Phase 0 acceptance requires goldens; golden infrastructure is M1.4 (backward edge) | backward dep | ADR-007 D1: goldens out of Phase 0 acceptance; M1.4 generates on the new floor |
| F-02 | M1 "tests only, no production code" vs M1.8 changing `AbstractModel` | contradiction | ADR-007 D2: M1.8 declared exception, ordered after M1.4 |
| F-03 | M3 façade exposes `DataType` (M4), `ExecutionPolicy` (M10), `Backend`/`DeviceSelector` (M12) while depending only on M1 | backward dep | ADR-007 D3: staged façade; v1 omits those surfaces; added at M4.7 / M10.2 / M12.1 |
| F-04 | `ModelProvider.load(..., Backend)` in target-architecture, but backend SPI is M12 and M5 depends on M4 only | backward dep | ADR-007 D4: `LoadTarget` transitional internal adapter, removed by T12.6 |
| F-05 | M7.2 admission needs a capacity query introduced in M12.2 | backward dep | ADR-007 D5: internal capacity contract in M6.2; M12.2 formalizes on the SPI |
| F-06 | "M7 … depends only on M6" contradicts M7.5 (needs M2), M7.6 (designed with M10), and the #129 land prerequisite | contradiction | ADR-007 D6: dependencies stated by kind; M7 split into core/ext; T7.7 after T10.2 |
| F-07 | PR land order: roadmap #129→#138; HANDOFF "Then #138 → #129" | contradiction | ADR-007 D9: roadmap order stands (it carries the rationale; HANDOFF carried none); HANDOFF corrected |
| F-08 | `public-api.md` says session concurrency "unresolved — see ADR-001" although ADR-001/006 resolved the strategy (engine batching) | stale vs ADR | public-api reworded: strategy resolved; only v1 *serialization latitude* remains, which is an implementation note, not an open design question |
| F-09 | M4.3 fixes descriptor shape ("dtype + element count/shape + layout") while ADR-004 Q3/Q4 are open | task w/ undecided input | Gate D-08 required before T4.3; roadmap M4.3 references it |
| F-10 | M5.1 acceptance is "—" | missing acceptance | T5.1: ServiceLoader-discovery test, supports-dispatch test, `LoadTarget` visibility test, no accelerator needed |
| F-11 | M8.2 acceptance "adds ≤ k dispatch classes" with k undefined | subjective acceptance | T8.2: k = 2, enforced by a dispatch-enumeration test |
| F-12 | M12.4 acceptance "Design-only, no implementation" | missing acceptance | T12.4: merged design section + recorded maintainer sign-off + no-new-packages check |
| F-13 | M13 is one paragraph with no tasks or acceptance | missing acceptance | Expanded into T13.1–T13.5 with executable acceptance |
| F-14 | ADR-001 migration notes put M6 "after the M3 façade exists"; roadmap M6 lists only M1 + #138 | contradiction | ADR-007 D8: M3 added as design prerequisite of M6 |
| F-15 | M11 lists only M9, but removing `ForwardPlanFactory` branches also needs the M5 providers | missing dep | ADR-007 D7: M11 depends on M5 and M9 |
| F-16 | "M7 is the product win and depends only on M6" in the dependency summary | contradiction (same as F-06) | Summary reworded |
| F-17 | Session-without-engine path: target-architecture lifecycle says `newSession()` acquires a lease "from the engine's `KvCacheManager`", but sessions must work without an engine | unresolved ownership | Gate D-10 (blocking for T6.1); options + recommendation in `ownership-and-lifecycle.md` |
| F-18 | Model close with live sessions: ADR-001 lifecycle says "an error, not a silent free" while its open question 3 still weighs throw vs force-close | unresolved, user-visible | Gate D-11, required before M3.1 (close semantics are façade contract) |
| F-19 | Concurrent invocation of one shared `CompiledProgram` unspecified anywhere | unresolved ownership | Gate D-12; recommendation: `invoke()` not thread-safe, engine is the sole batched invoker |
| F-20 | No engine request state machine, cancellation, callback threading, backpressure, shutdown semantics before M7 | missing contract | [`engine-contract.md`](../engine-contract.md); gates D-15…D-19, D-24 |
| F-21 | "Goldens bit-identical" required of every milestone, but nothing separates CI from accelerator-requiring gates; `mvn test` would need a GPU + model | missing verification model | [`verification-gates.md`](../verification-gates.md): Class A/B/C; `accel-tests` profile; skip ≠ pass |
| F-22 | Golden/benchmark procedures undefined (fixture provenance, serialization, NaN/Inf, regeneration, warm/cold, missing baseline) | missing verification detail | Specified in `verification-gates.md` |
| F-23 | `public-api.md` open-questions list numbering skips 2 | cosmetic | Renumbered |
| F-24 | Accepted ADRs still use pre-baseline "roadmap phase N" labels (ADR-001: phases 2/3/10; ADR-002: 6/7; ADR-003: 9; ADR-004: 4) while the roadmap uses M-numbers. Links resolve to the right anchors; only the prose labels are stale | stale, in accepted ADRs | **Not edited** — accepted ADRs are historical records. Proposed as a one-line editorial amendment for maintainer approval |
| F-25 | HANDOFF stale: says 5 commits, none pushed (now 6, branch pushed to `upstream/refactor/framework-abstractions`); says PR #140 open (now closed); lists `review/` leftover files that no longer exist; #129 retarget still pending (true — verified still targeting `feat/mma_cuda`) | stale HANDOFF | HANDOFF rewritten with verified state |
| F-26 | M12.2 acceptance "engine admission consumes the capacity query" unmeasurable if M7 has not landed (M12 does not depend on M7) | dependent acceptance | T12.2 gains M7-core as an explicit prerequisite |
| F-27 | `LocalModel` sketch carries `newSession()` on the base type while the same section says a non-generative model would not have it | internal tension | Sketch restructured around `TextGenerationModel` (pending gate D-01); base `LocalModel` generation-free |
| F-28 | Doc claims "65 `System.out/err` occurrences"; current count is 61 across the same 20 files | count drift | Noted only — descriptive doc; file count (the allowlist unit) is unchanged, no edit needed |

## Claims checked against the repository

Verified true: `tornadovm.base.version` still 5.0.0 (Phase 0 not done); 26 files
outside `tornadovm/**` import TornadoVM; 20 files with console I/O; 36 classes under
`tornadovm/layers/type/**`; one test source file; `perf-history.jsonl` schema lacks
`machine`/`gpu`/`tornadovm_version`/`cache_warm`; open PRs #120/#129/#131/#138/#142
with #129 still based on `feat/mma_cuda`; PR #140 no longer open.

Not verifiable from this repository (taken on the review record's authority): TornadoVM
issue numbers and behaviours (#996, #999, #1002, #1004, #1006, #1008, #1010),
`TestMultiThreadedExecutionPlans` results, the 53→103 tok/s and 11.5→5.2 s numbers, PR
#129's benchmark figures (419→211 steps etc.). Each is already cited with its source in
`tornadovm-capabilities.md`; re-verify against `~/TornadoVM` when the pinned version moves.

## Decisions that require maintainer approval

1. **ADR-007** as a whole (D1–D9): ordering fixes, staged façade, `LoadTarget`,
   capacity staging, M7 split, land-order confirmation.
2. **The M3 gate set** (D-01, D-03, D-04, D-06, D-07, D-11) — M3 is not implementable
   until these close.
3. **The M6/M7 gate set** (D-10, D-12…D-19, D-24) — recommendations are on the table in
   `ownership-and-lifecycle.md` and `engine-contract.md`.
4. **Editorial amendment** to ADR-001..004 replacing stale "phase N" labels with
   M-numbers (F-24) — cosmetic, but touches accepted ADR text, so it is the
   maintainers' call.
5. New documents `ownership-and-lifecycle.md`, `engine-contract.md`,
   `verification-gates.md`, `execution-backlog.md`, `decision-gates.md` are marked
   proposed; promoting them to normative is a maintainer decision.
