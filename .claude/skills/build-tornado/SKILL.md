---
name: build-tornado
description: Setup TornadoVM to use with GPULlama3.java. Use this when already on a system with GPUs visible.
license: Apache-2.0
metadata:
  author: TornadoVM Team
---

# Build TornadoVM

Build TornadoVM from source. Available backends: `opencl` (default), `ptx`, `spirv`, `cuda`, `metal`. Comma-separated combos allowed (e.g. `ptx,opencl` but not suggested).

## When to Use

| Scenario                                                 | Use This Skill? |
|----------------------------------------------------------|----------------|
| `echo $TORNADOVM_HOME` does not point to a TornadoVM SDK | Yes |

## Prerequisites

- JAVA_HOME is set to jdk 21 or 25
- `nvidia-smi` succeeds (GPUs visible)

## Instructions

### Step 1: Verify Environment

Run `nvidia-smi` to confirm you are on a system with GPU access.

### Step 2: Locate the Codebase

`cd` to the TornadoVM repository. If the path is not provided by the user, ask for it. If the path doesn't exist, stop and ask — don't clone or guess.

### Step 3: (Optional) Checkout Branch

A single word/arg to this skill is the **backend** unless the user explicitly calls it a branch (e.g. "on the develop branch"). Only checkout when a branch is explicitly named:
```bash
git checkout <branch> && git pull
```

### Step 4: Build

If the backend of choice is not provided by the user, ask for it.
Run the build command:

```bash
make BACKEND=<backend>
```

### Step 5: Install

```bash
source setvars.sh
```

Note: this only sets env vars for the current shell. Each new Bash tool call is a fresh shell — re-source `setvars.sh` before any later `tornado`/GPULlama3 command in a new call.

### Step 6: Verify

```bash
tornado --devices
```