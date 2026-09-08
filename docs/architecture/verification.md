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

## `qwen35` (Qwen3.5 / Qwen3.8)

Verified on `Qwen3.8-27B-Q4_0.gguf`
(`ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d`), host path only.

| Check | Result |
| --- | --- |
| Synthetic decomposition, both layer kinds, against an independently written reference | pass (`Qwen35CpuOperationEquivalenceTest`) |
| MTP draft head against an independently written reference | pass (`Qwen35MtpTest`) |
| Q4_1 block decode against hand-encoded blocks | pass (`Q4_1FloatTensorTest`) |
| Derived geometry against the real metadata block | pass (`Qwen35ConfigurationTest`) |
| Text against llama.cpp, same file, same prompt, greedy, both on CPU | agrees except at single-token near-ties |
| MTP draft agreement with the trunk, real model | 63 of 79 (80%) |
| Tool-call wire format, both directions, including round-trip | pass (`Qwen35ToolCallsTest`) |
| Full tool round-trip through the public API, real model | pass (`examples.ToolCalling`) |
| CPU/accelerator parity | n/a — no backend claims the architecture |

**No CI rows, deliberately.** `standalone-inference.yml` is an accelerator matrix: every row
asserts a resolved backend and a real `execution_path`, and this family has neither. The
smallest `qwen35` release is also far larger than the fixtures that matrix carries — the one
verified here is 16 GB. The unit gates above run in CI as ordinary tests; the fixture-backed
checks are local, and `GoldenFixture` skips them by name when the file is absent.

The tool round-trip is the check that could not be replaced by a unit test: the model has to
emit the format we prompt for, and no amount of parser testing says whether it does. It called
`get_weather({"city":"Athens"})`, the call was parsed, the assistant turn was replayed, and the
final answer used the tool's data. A model prompted for the wrong format simply answers in prose,
which is indistinguishable from one that decided a tool was unnecessary.

**Two things this port got wrong first, both of which produced fluent output.**

The delta rule pairs 48 value heads with 16 key heads, and the reference repeats the key heads
by *tiling* — value head `h` reads key head `h % 16`. Dividing instead (`h / 3`) pairs every value
head with the wrong key. Short answers stayed correct; a 220-token generation collapsed into a
repetition loop, because the error compounds through the recurrence rather than failing outright.

The equivalence test passed throughout, because its reference had been written from the same
misreading. **A reference that shares the implementation's misunderstanding is not a check** —
it is the trap in the porting skill's own list, and it cost a full generation run to notice. What
found it was reading the fused kernel in the reference implementation, which states the mapping
directly, rather than re-reading the graph builder that expresses it as a repeat.

The second is why the MTP acceptance rate is measured at all. The trunk's own token is what gets
emitted whether or not the draft head is fed correctly, so a head reading the wrong hidden state
produces output indistinguishable from a correct one. Agreement with the trunk is the only visible
signal: ~80% for a head that is fed correctly, chance for one that is not.

## Q4_0 device residency

Verified on `Llama-3.2-1B-Instruct-Q4_0.gguf`, quantized locally from the F16 fixture with
`llama-quantize`, on CUDA.

| Check | Result |
| --- | --- |
| Device decode against the host tensor, random bytes | pass (`Q4_0DecodeTest`) |
| Device decode against the specification, hand-built bytes | pass (`Q4_0DecodeTest`) |
| Real run, correct output, `execution_combination llama/Q4_0/STANDARD` | pass |
| Device peak and decode rate, same file, retention switched | 1570 MiB / 172.4 tok/s retained; 2060 MiB / 136.0 tok/s materialized |
| Q8_0 and F16 paths unchanged | pass |

The A/B is on **one file**, switched with `-Dllama.q4_0.retain`. Comparing a Q4_0 model against
a separately quantized Q8_0 one — which is the easier measurement to take — would have measured
the quantization as well as the residency, and the two answers differ.

Agreement between the device and host decoders is necessary but not sufficient, which is why the
specification check is there too: two implementations can agree and both be a different format.

## Gated Delta Net device kernels

The mixer's four kernels, against the host operations, lane by lane, at Qwen3.8-27B's own
dimensions — 10240 convolution channels, 48 value heads against 16 key heads, a 128×128 state
per head.

| Check | Result |
| --- | --- |
| Causal convolution output and its advanced window | bit-exact |
| Per-head L2 norm | bit-exact |
| Delta rule readout and the state it leaves behind | bit-exact |
| A value head reads key head `h % keyHeads`, asserted directly | pass |
| Decay and write strength | equal to float rounding |
| Gated norm | equal to float rounding |

Two of those are not bit-exact and the reason is arithmetic rather than addressing: the host
takes its reciprocal square root, its logarithm and its logistic in double and narrows once,
where `TornadoMath` works in float throughout. Everything else asserts **bit equality**, because
everywhere else the operations and their order are identical and a tolerance would hide a real
difference.

This is a host test, and it is possible at all only because each kernel body is a static method
taking its lane index, with the kernel a two-line wrapper passing `context.globalIdx`. A body
written directly against `KernelContext` cannot be called on the host, so it can only be
exercised by running a model on a device — and an indexing mistake in it surfaces as slightly
wrong text rather than as a failure. The delta rule needs no barrier and no cross-lane reduction
to begin with: a lane owning one value column of a head's state finds every quantity it needs is
its own.

**What this does not establish**: that the kernels compile and run on a device, that the layer
graph binds them correctly, or that the family runs on a GPU at all. Those are separate gates and
none of them is met yet.

## Known limitations

- **`qwen35` has no accelerator path, and no CPU/GPU parity gate.** Nothing claims the
  architecture, so the gate that matters most for every other family does not apply here and
  the CPU is verified against llama.cpp instead. The delta-net layers have no kernels. The
  memory objection is now weaker than it was: Q4_0 device residency exists, and this file is
  mostly Q4_0, so retaining it would leave roughly 17 GB rather than 28 GB against 24 GB of
  VRAM — plausible at a modest context. Retention would still have to be wired into this
  family's loader, and the kernels are the real work.
- **`qwen35` speculative decoding is not a speedup on the host path.** The draft head is
  correct and its acceptance rate is high, but an accepted draft only saves work where several
  positions are verified in one forward pass, and the host path verifies them one at a time.
  Measured cost of enabling it: 0.90 → 0.70 tok/s. Default off.
- **Consecutive `qwen35` tool results become separate user turns.** The template puts them in
  one, `ConversationEncoder` encodes one turn per result. No difference for a single result;
  merging needs a batched entry point on the shared encoder, which would change every family.
- **`qwen35` does not describe itself as an `InferenceProgram`.** It has a provider and a host
  forward pass but no `ModelArchitecture`, exactly as Gemma-4 does. A description is consumed by
  the lowered path, nothing lowers this family, and writing one would mean adding roughly a dozen
  `TensorRole` values and two `OperationKind` values that no backend reads — vocabulary ahead of a
  kernel, which is what the support tables exist to prevent. It arrives with the backend.

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
