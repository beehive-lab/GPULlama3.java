# FP16 GPU determinism defect — root cause and fix

**Status: root cause found and fixed.** Closed 2026-08-04.
Task ID **T1.4-FP16** in [`../execution-backlog.md`](../execution-backlog.md).

## Root cause

`reductionOneBlockWithLayer` (duplicated in `TransformerComputeKernelsLayered` and
`TransformerComputeKernels`) splits the RMS sum of squares across workgroups, each writing its
partial sum to `output[groupId + 1]`, and then has **the single thread `gid == 0` read all the
other workgroups' partials back and compute the final scale — inside the same kernel, with no
inter-workgroup synchronization.**

```java
if (lid == 0) {
    output.set(groupId + 1, localX[0]);   // every workgroup writes its partial
}
if (gid == 0) {
    for (int i = 1; i <= (size / localMemSize); i++) {
        ss += output.get(i);              // ...and workgroup 0 reads them, unsynchronized
    }
    output.set(0, 1.0f / sqrt(ss / size + ermsNorm));
}
```

Whether workgroup 0 sees a fresh or a stale partial depends on how the workgroups happen to be
scheduled, so the RMS scale — and everything downstream of it — differs between otherwise
identical executions.

The combine is only safe when the separate `reductionFinalNormalization` task runs afterwards and
recomputes `output[0]` from the partials. That task was gated on `schedulerType == NON_NVIDIA`, so
**the NVIDIA path ran the racy combine as the final word.** The kernel javadoc for
`reductionOneBlockWithLayerSingleGroup` already described this race for Qwen3; what was missed is
that every other model family was still on the racy path.

This is quantization-independent. FP16 lost the race ~11% of the time and Q8_0 almost never at
20GB device memory, which is why Q8_0 looked reproducible; at 12GB Q8_0 had already been seen
diverging ~1-in-4.

## The fix

`AbstractTransformerLayerTaskGraphs.rmsReduceKernel()` / `AbstractLogitsTaskGraph.rmsReduceKernel()`
now select the kernel, with `rmsReduceWorker()` selecting the matching grid:

- NON_NVIDIA — `reductionOneBlockWithLayer` + `reductionFinalNormalization` (unchanged)
- NVIDIA — `reductionOneBlockWithLayerSingleGroup`, one workgroup, `global == local ==
  state.localSize`, no cross-workgroup dependency at all

Applied to every FFN layer class (Llama, Mistral, Devstral, Granite, Phi3, Qwen2, Qwen3 — FP16 and
Q8_0) and to the logits layers. Qwen3, which had the fix inline, now uses the shared helper.

Also fixed independently: `TransformerComputeKernels.convertFP16toFP32` had no bounds guard, unlike
its Q8_0 counterpart. Harmless for dims that are a multiple of the local size, out-of-bounds for
any that are not.

## Verification

Same plan, same token, same position, nothing advanced between iterations, 300 identical
executions:

| Config | Before | After |
| --- | --- | --- |
| FP16 @20GB | 33/300 (11.0%), worstAbs 0.137 | **0/300** |
| Q8_0 @20GB | 0/300 | **0/300** |
| FP16 @12GB | — | **0/300** |
| Q8_0 @12GB | ~1-in-4 (earlier observation) | **0/300** |

Re-measured on both backends of a rebuilt `opencl,cuda` dist (see "Environment notes"), FP16 @20GB:

| Backend | Before | After |
| --- | --- | --- |
| CUDA | 24/300 (8.0%), worstAbs 0.142 | **0/300** |
| OpenCL | 15/300 (5.0%), worstAbs 0.137 | **0/300** |

Q8_0 is 0/300 after the fix on both backends. The "before" figures come from checking out the
pre-fix layer sources (`git checkout 7aa8e03 -- src/.../tornadovm/layers/`); note that stashing
does not work for this once the fix is committed — a `git stash push -- src` with nothing to save
silently produces a post-fix "baseline".

Layer-0 stage buffers (`temp`, `wrapXbFP16`, `wrapQ`, `wrapX`), which diverged 31/300 with the
racy kernel, are 0/300 after the fix.

Throughput, Llama-3.2-1B-Instruct-F16, 3 runs each: before 105.5 / 104.6 / 104.5 tok/s, after
108.7 / 109.7 / 104.5 tok/s. No cost.

`mvn verify -Paccel-tests`: all gates pass. `CpuGpuParityAccelTest` failed at the time of this fix
(|cpu-gpu| ≈ 3.5 at row 48) with identical values before and after it, so it was never a regression
here; that turned out to be a separate defect — the GPU's RoPE base — and is resolved in
[`cpu-gpu-parity.md`](cpu-gpu-parity.md).

### Reproducing

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

Mandatory flags: `-Dtornado.recover.bailout=False` (else a failed kernel silently falls back to
sequential Java), `-Dllama.deviceSample=false` (else argmax runs on device and no logits reach the
host), `-Dtornado.device.memory=20GB` (12GB can OOM when plans are rebuilt in a loop).

To re-create the defect, make `shouldUseFinalNormalization()` irrelevant by pointing
`rmsReduceKernel()` at `reductionOneBlockWithLayer` unconditionally.

## How it was localized

The decisive step was perturbing the graph and measuring the *rate*, not the outcome. In the
activation graph, and again at the head of the `layer_0` graph:

| perturbation | rate |
| --- | --- |
| extra kernel that **reads** `wrapX` | 0/300 |
| copy-out of `wrapX` itself | 0/300 |
| copy-out of an unrelated, untouched buffer (same blocking D2H) | 11.7% |
| extra kernel writing an unrelated buffer (same launch, same D2H) | 8.3% |

Only touching the buffer changed anything, and the TornadoVM bytecode streams of diverged and clean
iterations were **byte-identical** (same buffers, same events, same order) — so it was not buffer
binding, not launch count, and not a host-side sync. Forcing the finalize task on NVIDIA
(`shouldUseFinalNormalization()` → true) took FP16 from 11.7% to 0/300, which named the kernel.

Ruled out along the way: code generation (kernel source byte-identical across plans), an upstream
TornadoVM fix (reproduces on `develop` `e22835059`), backend specificity (CUDA and OpenCL both),
cross-run contamination, uninitialized `wrapXFP16`/`wrapXbFP16`, and the host-side
`temp`/`tempFFN`/`positionHolder` mutations interleaved between graph executions (hoisting them
ahead of every execution left the rate at 11.7%).

A separate, **stable** CPU↔GPU offset of ~0.5–0.7 on large logits exists in both quantizations. It
is reproducible, so it was never this defect. It was assumed here to be accumulation order; it was
not. It was the GPU computing RoPE from a hardcoded base of 50000 while the model's `rope_theta` is
500000 — see [`cpu-gpu-parity.md`](cpu-gpu-parity.md). After that fix the offset is 0.032 worst on
FP16 and 4e-05 on Q8_0. Worth noting as its own lesson: "this difference is just accumulation
order" is a hypothesis, not an explanation, until something measures it.

## The sampling trap — still worth reading

Four wrong conclusions in this investigation came from declaring reproducibility off a small clean
sample:

1. Golden generator compared **two** captures → recorded Q8_0 `bit_exact: true`. Wrong.
2. Probe compared only the **final** row → Q8_0 looked clean. Wrong; drift is sparse in rows.
3. Five clean repeats → "execution is deterministic, no race". Wrong; the rate was ~11%.
4. Q8_0 at 0/300 → "FP16-specific". Wrong; same racy kernel, Q8_0 just usually wins.

Judge any change here against ≥300 iterations. Also: a buffer comparison is meaningless unless the
buffer is actually read back — an early localization compared host copies of `wrapKeyCache` that
were never transferred (all zeros on both sides) and reported "identical".

## Tools

| Class | Purpose |
| --- | --- |
| `golden/DivergenceRate` | rate over N identical executions; `-Drate.iterations`, `-Drate.model` |
| `golden/OnePlanTest` | separates execution from compilation (one plan vs rebuilt plan) |
| `golden/LogitDump` | raw logits, teacher-forced; `-Ddump.forced`, `-Ddump.positions`, `-Ddump.cols` |
| `golden/Fp16DeterminismProbe` | all-row comparison, process fingerprint, cross-config |
| `golden/KvReadback` | characterises the `wrapKeyCache` diagnostic readback (see below) |

Production hook, default off: `-Dgpullama3.diag.transfers=true` pulls layer intermediates back to
the host, `-Dgpullama3.diag.layer=N` selects the layer (default 0). Note that `DivergenceRate`'s
per-stage counters are meaningless without it — the host copies are otherwise never refreshed.

## The `wrapKeyCache` readback — explained, not a defect

`DivergenceRate` used to report `kvCache=300/300` under `-Dgpullama3.diag.transfers`, i.e. the KV
cache host copy differing on every iteration, including iterations whose logits were bit-identical.

It is a snapshot-lag artifact, measured with `golden/KvReadback`:

- The diagnostic `transferToHost` sits at the end of **layer N**'s graph (`-Dgpullama3.diag.layer`,
  default 0). It copies the whole buffer, but the layers after N have not run yet in this forward,
  so their region still holds the **previous** forward's values.
- With `diag.layer=0`, exactly 7680 elements differ once and then never again — 15 layers × 512
  (`kvDim`) at the current position, i.e. layers 1–15.
- With `diag.layer=8` it is 3584 elements starting at layer 9 (7 layers × 512). With
  `diag.layer=15` nothing differs at all.

So consecutive iterations always agreed; only the *reference* snapshot was off, because it was
captured right after the first forward while the lagging region still held prompt-ingestion values.
`DivergenceRate` now discards one warm-up forward before capturing the reference, and reports
`kvCache=0/300`. No partial copy, no race, no production impact — but it did fake a defect for a
while, which is the same trap as the rest of this file: a comparison is only as good as its
reference.

## Aftermath

Goldens were regenerated on 2026-08-04 and record `bit_exact: true` for both fixtures — the first
time any GPU golden has carried the bit-exact assertion. Note what that flag was hiding: while it
was `false`, `GoldenLogitsAccelTest` compared token ids and a loose envelope only, so it passed
throughout the period when the GPU was both non-deterministic *and* computing RoPE with the wrong
base (see [`cpu-gpu-parity.md`](cpu-gpu-parity.md)).

## Follow-ups

- The `reductionOneBlockWithLayer` / `reductionFinalNormalization` pair is duplicated across
  `TransformerComputeKernels` and `TransformerComputeKernelsLayered`; the racy in-kernel combine
  should probably be deleted outright rather than left reachable on the NON_NVIDIA path.
- `CpuGpuParityAccelTest` (T1.5) fails on the pinned tuple, unrelated to this defect.

## Environment notes

- `~/TornadoVM` is on **`develop`** (`e22835059`). The owner's branch
  `fix/opencl-packed-half2-fp16` is intact at `611843029`; 9 stashes untouched.
- **Both backends are installed again** as one dist:
  `~/TornadoVM/dist/tornadovm-5.2.1-jdk21-dev-full-linux-amd64/tornadovm-5.2.1-jdk21-dev-full`
  (`etc/tornado.backend` = `opencl-backend,cuda-backend`). Build both at once with
  `make BACKEND=opencl,cuda` — a single-backend build wipes the other one's dist. The previous
  OpenCL-only dist is backed up at `~/TornadoVM/dist-backup-opencl`.
- Building the CUDA backend on develop needs **cuDNN** (develop added `tornado-cudnn` and
  `tornado-cutlass` modules). No cuDNN package exists in the configured CUDA repo, and installing
  system-wide needs root, so the redistributable was unpacked into
  `~/cudnn/cudnn-linux-x86_64-9.25.0.15_cuda13-archive`. `cudnn-jni` looks for `cudnn.h` and
  `libcudnn` under `$CUDA_PATH`, and `cutlass-jni` links against `$CUDA_PATH/targets/x86_64-linux/lib`,
  so the build needs a merged toolkit tree — `~/cuda-cudnn`, whose `include/`, `lib64/`, `bin` and
  `targets` symlink CUDA 13.1 plus the cuDNN files. Build with `CUDA_PATH=$HOME/cuda-cudnn`,
  that tree's `bin` on `PATH` and its `lib64` on `LD_LIBRARY_PATH`.
- Select the backend at runtime with `-Dtornado.cuda.priority` / `-Dtornado.opencl.priority` (higher
  wins), or `llama-tornado --gpu --cuda|--opencl`. On this box CUDA decodes ~135 tok/s vs ~100 tok/s
  on OpenCL for Llama-3.2-1B-F16.
- Model fixtures are symlinked into `~/.gpullama3/test-models/`.
