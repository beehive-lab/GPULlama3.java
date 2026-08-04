# FP16 GPU determinism defect — investigation handoff

**Status: open, root cause not identified.** Last worked 2026-08-04.
Task ID **T1.4-FP16** in [`../execution-backlog.md`](../execution-backlog.md).
Blocker before M6.

This file exists so the investigation can resume without re-deriving anything. It is a
working record, not architecture.

## The defect in one line

Repeated **identical** GPU execution of one fixed plan produces different logits for FP16
about 11% of the time; Q8_0 and the CPU path are reproducible.

## Measured baseline (reproduce before trusting any fix)

Same plan, same token, same position, nothing advanced between iterations:

| Config | Diverged | Rate | worstAbs |
| --- | --- | --- | --- |
| FP16 | 33/300 | **11.0%** | 0.137 |
| Q8_0 | 0/300 | **0.0%** | — |

```bash
export TORNADOVM_HOME=$HOME/TornadoVM/dist/tornadovm-5.2.1-jdk21-dev-opencl-linux-amd64/tornadovm-5.2.1-jdk21-dev-opencl
export JAVA_HOME=/home/orion/.sdkman/candidates/java/21.0.2-open
./mvnw -q -B dependency:build-classpath -Dmdep.outputFile=/tmp/cp.txt -Dmdep.includeScope=test
CP="target/classes:target/test-classes:$(cat /tmp/cp.txt)"

java "@$TORNADOVM_HOME/tornado-argfile" --add-modules jdk.incubator.vector \
  -Dtornado.recover.bailout=False -Dllama.deviceSample=false \
  -Dtornado.device.memory=20GB \
  -Drate.iterations=300 \
  -Drate.model=$HOME/.gpullama3/test-models/Llama-3.2-1B-Instruct-F16.gguf \
  -cp "$CP" org.beehive.gpullama3.golden.DivergenceRate
```

**A fix must be judged against the 11% rate over ≥300 iterations, never against a handful
of clean runs.** See "the sampling trap" below.

## Established, with evidence

| Claim | Evidence |
| --- | --- |
| FP16 GPU is intermittently non-deterministic | 33/300 identical executions |
| Q8_0 GPU is not (or far rarer) | 0/300 here; but 1-in-4 seen at **12GB** device memory |
| CPU is fully reproducible, both quantizations | all 64 rows identical, 3/3 |
| Not code generation | kernel source across two plans is **byte-identical**, SHA-256 `52be5cb5…`, 0 diff lines over 33970 lines / 148 kernels |
| Not upstream-fixed | reproduces on TornadoVM `develop` `e22835059` at the same magnitude |
| Not backend-specific | reproduces on CUDA **and** OpenCL |
| Not cross-run contamination | Q8_0 after a bad FP16 run in the same JVM gives the known-good fingerprint 3/3 |
| Not `wrapXFP16`/`wrapXbFP16` being uninitialized | zeroing both at allocation changed nothing; reverted |
| Divergence present from **layer 0** | at layer 0, `state.temp`, the FP16 normalized activation and the QKV output all diverge at the same rate as the final logits |

A separate, **stable** CPU↔GPU offset of ~0.5–0.7 on large logits exists in *both*
quantizations. It is reproducible, so it is not this defect — it is accumulation-order
difference, and it is what a parity tolerance is meant to absorb.

## The next measurement (this is where to start)

`state.temp` at layer 0 is produced by `reductionOneBlockWithLayer`, which **Q8_0 also
uses at 0%**. So either its input `wrapX` is already wrong on entry to layer 0, or that
shared kernel misbehaves only on FP16-shaped input.

**Snapshot `wrapX` straight out of the `activationUpdate` graph, before layer 0.**

- diverges there ⇒ cause is `convertFP16toFP32`, or the `embeddingX` host-write /
  device-read ordering. `embeddingX` is written host-side by `MemorySegment.copy` in
  `InferenceCore.forwardTornadoVM` (~`:785`) and transferred `EVERY_EXECUTION`; an
  ordering gap there would produce exactly this intermittency.
- clean there ⇒ the shared reduction is fine on Q8_0 input but not FP16; look at the
  `HalfFloat` accumulation paths.

## Latent bug found (real, but NOT this defect)

`TransformerComputeKernels.convertFP16toFP32` has **no bounds guard**:

```java
int i = context.globalIdx;
wrapX.set(i, x.get(i).getFloat32());          // unguarded
```

Its Q8_0 counterpart returns early when `globalId >= wrapX.getSize()`. Not triggered for
Llama-3.2-1B (dim 2048 is an exact multiple of local size 128), but it writes out of
bounds for any model whose dimension is not. Worth fixing independently.

## Tools (all test-scope, diagnostic only)

| Class | Purpose |
| --- | --- |
| `golden/DivergenceRate` | rate over N identical executions; `-Drate.iterations`, `-Drate.model` |
| `golden/OnePlanTest` | separates execution from compilation (one plan vs rebuilt plan) |
| `golden/LogitDump` | raw logits, teacher-forced; `-Ddump.forced`, `-Ddump.positions`, `-Ddump.cols` |
| `golden/Fp16DeterminismProbe` | all-row comparison, process fingerprint, cross-config |

Production hooks, **default off**, shipped graph unchanged unless set:

- `-Dgpullama3.diag.transfers=true` — pull layer intermediates back to host
- `-Dgpullama3.diag.layer=N` — which layer (default 0)

Mandatory flags for any run: `-Dtornado.recover.bailout=False` (else a failed kernel
silently falls back to sequential Java), `-Dllama.deviceSample=false` (else argmax runs
on device and no logits reach the host), and `-Dtornado.device.memory=20GB` (12GB raises
the Q8_0 divergence rate and can OOM when plans are rebuilt in a loop).

## The sampling trap — read this before concluding anything

Three separate wrong conclusions in this investigation came from the same mistake:
declaring reproducibility from a small clean sample.

1. Golden generator compares **two** captures → recorded Q8_0 `bit_exact: true`. Wrong.
2. Probe compared only the **final** row → Q8_0 looked clean. Wrong; drift is sparse in rows.
3. Five clean repeats → "execution is deterministic, no race". Wrong; the rate is ~11%.

Also: a buffer comparison is meaningless unless the buffer is actually read back. An early
layer-level localization compared host copies of `wrapKeyCache` that were never transferred
— all zeros on both sides, reported "identical". Always check the non-zero fill first.

## Environment notes

- `~/TornadoVM` is on **`develop`** (`e22835059`). The owner's branch
  `fix/opencl-packed-half2-fp16` is intact at `611843029`; 9 stashes untouched.
- The CUDA dist was replaced by an OpenCL dist (`make` wipes the other backend).
  Rebuilding CUDA from develop currently **fails**: `cudnn.h: No such file or directory`
  — develop's CUDA backend needs cuDNN headers, which are not installed.
- Model fixtures are symlinked into `~/.gpullama3/test-models/`.
