---
name: build-n-run-engine
description: Build GPULlama3.java from source with Maven and run. Use when the compiled jar/classes are missing or stale, or after pulling new changes.
license: Apache-2.0
metadata:
  author: TornadoVM Team
---

# Build GPULlama3.java

Build GPULlama3.java (this repo) with Maven, skipping tests for speed.

## When to Use

| Scenario                                                          | Use This Skill? |
|--------------------------------------------------------------------|-----------------|
| `target/classes` missing or older than latest source/pom changes  | Yes |
| Just pulled/checked out new code and need it compiled              | Yes |
| Only running an already-built jar, no source changes               | No |

## Prerequisites

- `JAVA_HOME` set to JDK 21 or 25 (`java -version`)
- `TORNADOVM_HOME` set and `tornado --devices` succeeds — if not, run the `build-tornado` skill first
- `~/TornadoVM/setvars.sh` sourced in **this** shell (env vars don't persist across shells/tool calls)

## Instructions

### Step 1: Verify Environment

```bash
echo $JAVA_HOME && echo $TORNADOVM_HOME && tornado --devices
```

If `TORNADOVM_HOME` is unset or `tornado --devices` fails, stop and run the `build-tornado` skill first — do not proceed with a missing TornadoVM SDK.

### Step 2: Locate the Codebase

`cd` to the GPULlama3.java repository. If the path is not provided by the user, ask for it. If the path doesn't exist, stop and ask — don't clone or guess.

### Step 3: (Optional) Checkout Branch

A single word/arg to this skill is most likely the intent for **which JDK/backend to build with**, not a branch. Only checkout when a branch is explicitly named:
```bash
git checkout <branch> && git pull
```

### Step 4: Build

```bash
mvn clean install -DskipTests
```

For JDK 25 instead of the default JDK 21, ensure `JAVA_HOME` points at JDK 25 before running `make` — the pom auto-activates the `jdk25` profile from the detected JDK version, there is no separate `BACKEND=`-style flag.

### Step 5: Verify

```bash
./llama-tornado --help
```

If this prints usage instead of erroring, the build succeeded and the launcher is executable.

### Step 6: (Optional) Smoke-test a run

Only if a GGUF model path is available. The backend (OpenCL/PTX/CUDA/Metal) is
auto-detected from `$TORNADOVM_HOME/etc/tornado.backend` — no backend flag needed:
```bash
./llama-tornado --gpu --verbose-init --model <path-to-model.gguf> --prompt "write a matmul in Java" --max-tokens 2048
```

## MANDATORY: Use --help When Uncertain About Flags

Before using any `llama-tornado` flag you are not 100% certain about, run `--help` first. Do not guess flag names.

## Running GPULlama3.java

`./llama-tornado --model <path> [options]`. The TornadoVM backend is auto-detected from the
installed SDK (`$TORNADOVM_HOME/etc/tornado.backend`); to run on a different backend, point
`TORNADOVM_HOME` at an SDK built for it. If the current SDK was built with more than one backend
(e.g. a `cuda-opencl` build), pass `--opencl`/`--ptx`/`--cuda`/`--metal` to force one of the
installed ones instead — these error out if the requested backend isn't part of the SDK, and
are a no-op (redundant but harmless) on a single-backend SDK.

### Core options

| Flag | When to use |
|------|-------------|
| `--model <path>` | required, GGUF path |
| `--prompt "..."` | single-shot generation |
| `-i` / `--interactive` | chat loop instead of one-shot |
| `--gpu` | required for GPU acceleration; omit to run CPU-only |
| `--opencl`/`--ptx`/`--cuda`/`--metal` | rarely needed — only to force a backend when the SDK has more than one installed |
| `--gpu-memory 15GB`/`20GB` | bump from default 14GB for 3B/8B models — avoids OOM |
| `--temperature`, `--top-p`, `--seed`, `-n` | standard sampling knobs |
| `-sp/--system-prompt` | instruct-mode framing |

### Debug/inspect (only when diagnosing a specific issue, not normal runs)

| Flag | When to use                                                                                |
|------|--------------------------------------------------------------------------------------------|
| `--verbose-init` | to inspect gguf load, copy-in, jit time                                                    |
| `--print-bytecodes` / `--print-threads` / `--print-kernel` / `--full-dump` | debugging a specific kernel/codegen problem — pair with `tornado-backend-specialist` agent |
| `--profiler` + `--profiler-dump-dir <dir>` | collect perf metrics JSON — feeds the `gpullama-benchmarking` skill                        |
| `--show-command --execute-after-show` | verify the exact java invocation before it runs                                            |

### This branch's feature: prefill/decode split (fp16-kvcache work)

| Flag | When to use |
|------|-------------|
| `--with-prefill-decode` | enable batched-prefill / single-token-decode split path |
| `--batch-prefill-size N` | only meaningful with `--with-prefill-decode`; N>1 batches prefill |
| `--cuda-graphs` | **PTX-only**; captures/replays CUDA graph to cut launch overhead — use when benchmarking decode-heavy latency on NVIDIA |

### Typical invocations

```bash
# quick single-shot GPU test (backend auto-detected from TORNADOVM_HOME)
./llama-tornado --gpu --model model.gguf --prompt "..."

# interactive chat
./llama-tornado --gpu -i --model model.gguf

# bigger model, needs more GPU mem
./llama-tornado --gpu --model llama-3.2-8b-instruct-fp16.gguf --gpu-memory 20GB --prompt "..."

# benchmarking prefill/decode split with CUDA graphs (needs a PTX-backend TORNADOVM_HOME)
./llama-tornado --gpu --model model.gguf --with-prefill-decode --cuda-graphs --profiler --profiler-dump-dir ./perf-results --prompt "..."
```
