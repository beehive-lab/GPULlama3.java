# Session handoff

**Transient working state, not architecture.** Delete when stale. Nothing normative
lives here — the roadmap, backlog and gates own the plan; this file only records git
state, PR mechanics and immediate next actions. Last refreshed **2026-08-03** (roadmap-
hardening pass).

Read [`README.md`](README.md) first — it indexes the architecture documents.

## Where things stand

**Phase 0 landed 2026-08-03** — `pom.xml` pins TornadoVM **5.2.0**, and a local
RTX 5090 performance baseline replaces `docs/perf-history.jsonl` as the refactor
reference. M1 is the current milestone.

Architecture baseline v1.0, ADR-001..007 `Accepted` (**ADR-007 accepted 2026-08-03**,
same day the hardening pass produced it). The pass added five companion documents
([`decision-gates.md`](decision-gates.md), [`execution-backlog.md`](execution-backlog.md),
[`verification-gates.md`](verification-gates.md),
[`ownership-and-lifecycle.md`](ownership-and-lifecycle.md),
[`engine-contract.md`](engine-contract.md)), the
[hardening report](review/roadmap-hardening-2026-08-03.md) and
[`START-HERE.md`](START-HERE.md). Roadmap + backlog are now binding. The **gate
recommendations** (D-01..D-24) remain open — the M3 set is the next maintainer batch.

## Git state (verified 2026-08-03)

- Branch: **`refactor/framework-abstractions`**, now 8+ commits ahead of `upstream/main`.
  The Phase 0 commits are **local only — not pushed**.
- Owner's decision: **not merging to `main`** for now. Do not merge or push without asking.
- PR #140 is **closed** on GitHub (previous handoff said open — resolved).
- `pom.xml` pins TornadoVM **5.2.0** — Phase 0 done.

## Local performance baseline

`perf-results/baseline-rtx5090-tvm520-20260803/` is the reference for the refactor.
`docs/perf-history.jsonl` was recorded on CI hardware and is **not** comparable to the
development laptop — do not gate against it. The local baseline covers 7 of 8 families
across CUDA and OpenCL, F16 and Q8_0, median of 3 cold runs, worst spread 6.24%.
CUDA leads OpenCL by 6–25% everywhere; Q8_0 leads F16 everywhere.

## Open defect: GPU logits are not reproducible run-to-run (FP16 **and** Q8_0)

**Q8_0 is affected too — corrected 2026-08-03.** An earlier note here claimed Q8_0
reproduced exactly. That was **under-sampled**: the golden generator compares only two
captures, and the probe compared only the *final* logits row. Running the committed Q8_0
golden twice settles it — one run passes, the next fails at row 19; an earlier run failed
at row 48. The drift is **intermittent and can land on any row**.

Consequences, beyond the FP16 detail below:

- **No configuration is currently demonstrated reproducible on the pinned tuple.** The
  committed goldens' `bit_exact: true` for Q8_0 is wrong and must not be trusted.
- **T1.6 cannot lean on Q8_0 for the bit-exact numerical assertion**, which was the plan.
  Compilation identity is still assertable for every configuration, since it does not
  depend on the numerics — that separation holds and is worth keeping.
- The reproducibility measurement must sample far more than twice, and must compare
  **all** rows, before any `bit_exact: true` is believed.

This is a bigger finding than the FP16-only defect and needs a decision before the golden
gate means anything.

## Open defect: FP16 logits are not reproducible run-to-run

Found while generating the T1.4 goldens on the pinned tuple (RTX 5090, CUDA,
TornadoVM 5.2.0-jdk21). Two identical back-to-back captures of Llama-3.2-1B **F16**
differ on **all 64** compared logits rows: max absolute difference **0.168** on logits
spanning −8.0 to +23.3, with 128255 of 128256 elements differing on the final row.
That is a non-deterministic reduction, not last-bit rounding. **Q8_0 on the same
harness reproduces exactly**, so it is the FP16 path, not the capture code.

Token ids are so far unaffected — the argmax margins exceed the drift — but that is
luck, not a guarantee, and it will not hold for near-ties.

Consequences:

- [`verification-gates.md`](verification-gates.md) assumes bit-exactness is a property
  of the pinned tuple. That holds for Q8_0 and **not** for F16 here. Goldens therefore
  **measure** reproducibility at generation time and record `bit_exact` in the
  metadata, rather than assuming it. The F16 golden asserts token ids and the NaN/Inf
  check; row hashes are asserted only when `bit_exact` is true.
- **M1.6 compiled-program identity** and the roadmap's "goldens bit-identical"
  definition of done cannot mean bit-identical for FP16 until this is fixed.
- Fixing it is production work (a kernel/reduction change) and outside M1's tests-only
  scope. It needs a maintainer decision on where it lands.

Once fixed, re-running `scripts/regenerate-goldens.sh` records `bit_exact=true`
automatically — there is no flag to edit.

## Two blockers for M1.4 golden coverage

1. **`DEEPSEEK_R1_DISTILL_QWEN` is broken** — `Qwen3Tokenizer.encodeChunk` throws
   `NoSuchElementException` at `vocabulary.getIndex(String.valueOf((char) b)).orElseThrow()`
   because DeepSeek's byte-level BPE vocabulary has no single-character ASCII entries.
   Fails on both backends, both quantizations, any prompt, before any GPU work.
   Pre-existing, unrelated to the version bump. Needs a fix before that family can be
   baselined or covered by goldens.
2. **`DEVSTRAL_2` has no local GGUF** — unmeasured and untestable here.

## Pending PRs (open on GitHub, verified 2026-08-03)

Land order is normative in the
[roadmap](migration-roadmap.md#pr-land-order): Phase 0 → **#129 → #138** → #120 → #131,
with #142 assessed against M1.7 first. (An earlier handoff said #138 before #129; the
roadmap order carries the rationale and stands — ADR-007 D9.)

| PR | Note |
| --- | --- |
| #129 batched decode | **Still targets `feat/mma_cuda`** (already contained in `main`) — retarget to `main`, no rebase needed. Time-critical: +2694 lines into exactly the files M6/M7 restructure |
| #138 FP16 KV cache | Only conflict is `pom.xml` (it bumps to 5.1.1, a partial Phase 0 — no `BFloat16Array`). Phase 0 is now done at 5.2.0, so #138 should **drop its pom change** and rebase clean |
| #120 Gemma 4 | Merges clean; the bump it needed (BF16 → `BFloat16Array`, 5.2.0) has landed. Note `ModelType` still has no `GEMMA` entry — it arrives with this PR |
| #131 hybrid libs | 2 conflicts (`llama-tornado`, `LogitsFP16Layer.java`); last, additive, default-off |
| #142 ci/metal-migration | Not in the roadmap; check interaction with M1.7's CI changes before either lands |

## Immediate next actions

1. ~~Phase 0~~ — **done**, pin at 5.2.0.
2. **M1 is current** — T1.1–T1.3 (ArchUnit) then T1.4–T1.7 (goldens, parity, identity,
   benchmark gate) on the 5.2.0 floor.
3. Then #129 (retarget) → #138 → #120 → #131 per the land order.
4. Before M3 opens: maintainer closes the M3 gate set
   (D-01, D-03, D-04, D-06, D-07, D-11) in [`decision-gates.md`](decision-gates.md).

## Working conventions

- **The architecture documents are human-owned.** Propose and implement; never silently
  redefine. ADR status changes are the owner's call, not yours.
- **Ground every claim in a real capability.** [`tornadovm-capabilities.md`](tornadovm-capabilities.md)
  is the ledger. Verify against the local checkout at `~/TornadoVM` (5.2.1-jdk21-dev)
  rather than trusting recall.
- **Verify assertions before writing them into a document** — the review standard here.
- Markdown links checked before each commit — all internal links and anchors must
  resolve (now 26 files under `docs/architecture/`).

## Two facts worth not re-deriving

Both were wrong in the first draft of the baseline and are now load-bearing:

1. **Concurrent independent `TornadoExecutionPlan`s on one device are supported and
   tested** (`TestMultiThreadedExecutionPlans`, 4/4 on CUDA). Batching is chosen because
   device buffers are per task graph — a plan per session duplicates the weights, ~3.4 GB
   on 3B-Q8. Not because an API is missing.
2. **CUDA-graph capture fixes device addresses.** Re-pointing a captured buffer fails at
   replay with `CUresult=700`, and `tornado.recover.bailout` defaults to `TRUE`, so the
   symptom is **wrong output, not an error**. This is why the KV block pool must be one
   persistent array with in-kernel indexing, and why the golden suite runs with
   `-Dtornado.recover.bailout=False`.
