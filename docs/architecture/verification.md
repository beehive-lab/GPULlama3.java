# Verification

What is gated, what the gates assert, and what is honestly not covered.

## Two rules that shape everything here

**A process exiting 0 is not evidence of correctness, and neither is a throughput number.**
An accelerator computing entirely wrong numbers produces both, cheerfully. A matrix that
scores rows by grepping for `achieved tok/s` once reported fourteen passing Metal rows,
three of which were emitting token salad.

**A GPU-versus-GPU comparison cannot see a defect that moves the whole GPU.** Two lowered
and legacy paths that are equally wrong agree with each other and both pass. Every
numerical claim is therefore scored against a **CPU reference**. Which families that
reference actually covers is stated below, and it is not all of them.

## Gate classes

| Class | What runs | Needs | When |
| --- | --- | --- | --- |
| **A** | architecture rules, unit tests, launcher-flag and benchmark-gate tooling tests, documentation links | a JVM. No model file, no accelerator, no TornadoVM | every push; `mvn test` and `make test-scripts` |
| **B** | golden logits, CPU↔GPU parity, compiled-program identity, lowering parity, lifecycle, multi-session, KV/workspace sharing, diagnostics, metrics, execution modes | a TornadoVM SDK, a device, and the pinned model fixtures | `mvn verify -Paccel-tests` on a pinned tuple; required before merging any change to an execution path |
| **C** | benchmark gate against `docs/perf-history.jsonl`, full model-matrix goldens | a Class B environment plus performance history | releases, and on every push to `main` |

`mvn test` never requires an accelerator or a model. Class B tests are named `*AccelTest`
and are excluded from Class A by name. On a machine without the pinned tuple they **skip
with an explicit reason** — a skip is recorded, never reported as a pass, and the two
profiles are never summed into one number.

Class B forks one JVM per test class. Device memory a closed session frees goes back to
TornadoVM's buffer provider but not to the driver, so a shared JVM exhausts the device
after a handful of classes; see [`memory-and-concurrency.md`](memory-and-concurrency.md).

`-Dtornado.recover.bailout=False` is mandatory for every Class B run. With TornadoVM's
default, a failed kernel silently falls back to sequential Java, which produces a *wrong
golden* instead of an error.

## Golden logits

Fixture: `Llama-3.2-1B-Instruct`, F16 and Q8_0. The file's SHA-256 is pinned in the test
resources; the file itself is not committed — it is resolved from `$GPULLAMA_TEST_MODELS`
or `~/.gpullama3/test-models/`, and the test fails with a fetch instruction if absent.

Captured: a fixed prompt, greedy sampling, 64 generated tokens. Compared: the final logits
row at the last prompt position and at each generated position, plus the emitted token ids.
Stored as raw little-endian float32 alongside a metadata sidecar recording the model hash,
quantization, prompt, backend, device, driver, TornadoVM version, build commit and
`recover_bailout: false`.

Bit-exactness is asserted **only on the pinned tuple** (device, driver, TornadoVM version,
backend, build flags). On any other tuple the gate says so and drops to the parity
tolerance. **Any NaN or Inf fails immediately**, before comparison — a NaN-versus-NaN match
must never pass.

Reproducibility is **measured, not assumed**: the generator captures each configuration
twice and records the outcome as `bit_exact`. That policy is what found a racy RMS
reduction which had made both representations non-reproducible. A configuration may carry
`bit_exact: false` only while a corresponding open defect is recorded, and it then runs the
envelope gate below. It is a temporary accommodation of a known defect, not a relaxed
standard.

**Goldens are regenerated only through `scripts/regenerate-goldens.sh`**, which refuses a
dirty working tree and writes the generating commit into the metadata. That commit must
change nothing else and must say why. Never regenerate to make a failure go away.

### Reproducibility envelope

Applies where `bit_exact: false`. Over repeated captures on the pinned tuple:

| Property | Bound |
| --- | --- |
| NaN/Inf | none, ever — checked first |
| Max absolute drift | ≤ 1.0 per element |
| Max relative drift | ≤ 0.05 where \|reference\| ≥ 1.0 |
| Argmax | must be identical |
| Top-5 membership | must be identical |
| Top-10 membership | recorded |
| Token sequence | must be identical |

Token equality alone is explicitly not sufficient: on the reference tuple argmax and top-5
survive while top-10 membership already changes, so greedy decoding hides a defect that
top-k or top-p sampling would expose.

## CPU↔GPU parity

Same fixture, same prompt, CPU as the reference. Per-element tolerance
`|got − ref| ≤ 1e-2 · Σ|wᵢaᵢ| + 1e-3`, with `atol=1.5e-2` and `rtol=1e-2`, and a budget on
the fraction of elements allowed to violate it. NaN or Inf on either side fails.

Coverage is every family with a fixture, in every representation it ships: Llama, Qwen3,
Qwen2.5, Granite and Phi-3 in F16 and Q8_0, and Mistral in Q8_0. One class per family, so
surefire forks a JVM per family — device memory a closed session frees goes back to
TornadoVM's buffer provider but not to the driver, so a single class covering everything
exhausts the device partway through and the failures land on whichever model ran late rather
than whichever one is wrong.

DeepSeek-R1-Distill-Qwen and Devstral have no fixture on the reference machine and are
therefore not gated here.

## Compiled-program identity

In one process: compile once and record the number of task graphs, the ordered task names,
the grid-scheduler entry set, and a SHA-256 over each task's generated kernel source; then
decode at least 100 tokens; then assert every recorded value unchanged and that no further
compilation happened.

Compilation identity is independent of numerical determinism, and the two are not
conflated. The structural assertions run for every configuration, including F16, because
they do not depend on the numerics being reproducible. The bit-exact numerical half is
carried by Q8_0.

## Benchmark gate

Tuple: (machine, gpu, model, quantization, backend, configuration, tornadovm_version).
Comparisons only ever happen within one tuple. Procedure: three warm-up generations
discarded, then five measured runs; the metric is decode `eval_rate`, aggregated as the
median.

- **Tracking a tuple over time** compares against the most recent gate-passing entry.
- **Judging a change** measures the baseline *in the same session*, interleaved with the
  candidate, and consults no history. A stored baseline ages, and the age shows up as a
  regression: one machine measured 172.5 tok/s and, two hours later, 167 tok/s from an
  unchanged build.
- **Missing baseline** — a new tuple, or the first run after a TornadoVM version change —
  is a record-only pass. Cross-version comparison is meaningless by construction.
- **Noisy baseline** — if the five-run spread exceeds 10% of the median, the gate reports
  an unstable environment and neither passes nor records. A machine too noisy to measure
  has said nothing about the code, and failing there would train people to ignore the gate.
- Tolerances and which machines are gated rather than record-only live in
  `scripts/perf-gate-tolerances.json`; the default on a pinned self-hosted runner is 3%.
  Shared-CI tuples are record-only.

Exit codes: 0 pass or record-only, 1 regression, 2 unstable environment, 3 usage error — a
usage or environment problem is never reported as a performance verdict. In CI, 2 warns and
passes.

## CI matrix

| Job | Runner | What it proves |
| --- | --- | --- |
| `code-quality` | Linux | formatting, Python tooling tests |
| `build-linux` | Linux | clean build and Class A gates on **JDK 21 and JDK 25**, on CUDA and OpenCL SDKs; artifact suffix, class-file version and service-file count; launcher builds its command from the SDK argfile |
| `build-macos` | macOS | clean build and Class A gates on the Metal SDK |
| `standalone-inference-linux` / `-macos` | Linux / macOS | end-to-end generation per family, quantization and execution mode, scored against a recorded expectations table |
| `quarkus-langchain4j-integration` | Linux | the Quarkus extension builds and serves against this build |
| `performance-gate`, `publish-performance-history` | Linux | `main` only |

The two OS families are separate jobs, not cells of one matrix, because jobs depend on
jobs: with one job an unavailable macOS runner leaves every Linux consumer queued behind
it.

### The standalone expectations table

Every row records an outcome and does not abort the job; the assertion step decides the
result after the whole matrix has run, against `.github/standalone-expectations.tsv`.

- A row **not** in the table must pass, with the resolved backend and a real execution path
  asserted, and with an expected substring present in the generated text. The matrix label
  is not taken on trust, and a CPU fallback must not look like success.
- A row **in** the table must fail, and fail with the recorded cause. An unexpected pass and
  a failure for a different reason are both deviations, so a toolchain fix and a new defect
  each turn the job red rather than blending in.
- A row that ran and produced **wrong output** is a correctness defect. It can never be
  silenced by adding a line to the table, and the assertion step enforces that.

## Known limitations

Recorded honestly rather than gated away. None of these is a passing configuration.

**Kernel capture on Metal.** `withPrintKernel()` produces no kernel source, so
`CompiledProgramIdentityAccelTest` cannot observe there. A capture-path gap, not a
numerical one.

**Phi-3-mini on Metal does not complete, in either representation.** The run reaches the
accelerator, emits a few tokens and then makes no further progress. A hang with no
diagnostic rather than a toolchain refusal, and unresolved. Both rows are in the
expectations table so that one hanging row cannot go on erasing the rest — each row now runs
under its own budget.

**Qwen2.5-1.5B Q8_0 on Metal produces wrong output.** It resolves the backend, reports a
real execution path, exits 0 at a normal throughput, and generates a stream of backticks
instead of an answer. This is a correctness defect, not a recorded limitation, and it is
deliberately *not* in the expectations table: the assertion step refuses to let a
`WRONG-OUTPUT` row be silenced, so the Metal leg stays red until it is fixed.

It was invisible until now. Every previous Metal run was cancelled at the job wall clock
before the assertion step ran, so that backend had never produced a results table at all —
which is why an output check that exists, on a backend that runs, had never once been
applied there. Qwen2.5 F16 passes on the same machine, as do both Qwen3 representations, so
the shape resembles the earlier Qwen FP16 Metal defect: capability-gated kernel selection
choosing a reduction that is wrong on that device. Confirming that needs the Mac; nothing
here claims to have reproduced it from Linux.

**Batched prefill on Metal, and Q8_0 batched prefill on CUDA.** TornadoVM toolchain gaps,
with named causes, in [`models-and-backends.md`](models-and-backends.md).

**Devstral.** The `mistral3` fixture loads and generates correct text on Metal in
`STANDARD`. `PREFILL_DECODE` and `BATCH_PREFILL_DECODE` are unsupported for the family,
each with its own accurate diagnostic. The rest of its acceptance — teacher-forced CPU/GPU
logit parity, reset and multi-turn behaviour, memory-preflight accuracy — is **unrun**, and
Devstral is not claimed as verified. It blocks nothing.

**Memory preflight on Metal** is capped at `CONSERVATIVE`; the bisection that would justify
`EXACT` there has not been run.

**Metal evidence is CI and Mac-session evidence.** Last full local run on Apple silicon:
86 accelerator tests, 2 failures, 0 errors, 6 skipped, where the two failures are the
kernel-capture gap above. Nothing here claims a Metal run performed from Linux.

## Current results

Measured on this branch, RTX 5090 Laptop, TornadoVM 6.0.0 built from the pinned tag:

| Configuration | Class A | Class B |
| --- | --- | --- |
| JDK 21, CUDA | 531 tests, 0 failures | 94 tests, 0 failures, 0 errors, 6 skipped |
| JDK 25, CUDA | 531 tests, 0 failures | 94 tests, 0 failures, 0 errors, 6 skipped |
| JDK 21, OpenCL | 531 tests, 0 failures | 94 tests, 0 failures, 0 errors, 7 skipped |
| JDK 25, OpenCL | 531 tests, 0 failures | 94 tests, 0 failures, 0 errors, 7 skipped |

CPU↔GPU parity covers all eleven fixture/representation combinations on both backends. The
skips each name a reason: an absent fixture, an absent `TENSOR_CORE_MMA`, or a Metal-only
kernel-selection check.
