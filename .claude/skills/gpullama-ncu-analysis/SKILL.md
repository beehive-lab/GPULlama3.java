---
name: gpullama-ncu-analysis
description: >
  Nsight Compute workflow for analyzing one specific GPULlama3/TornadoVM CUDA
  kernel. Use after nsys or TornadoVM profiler output identifies the hot kernel.
tags:
  - gpullama
  - ncu
  - kernel-profiling
license: Apache-2.0
---

# GPULlama3 Nsight Compute Analysis

## When To Use

Use `ncu` only for a specific hot CUDA kernel identified by `nsys`, TornadoVM profiler output, or generated-kernel investigation.

Do not run broad `ncu` collection over an entire decode loop without a kernel target.

## Workflow

1. Identify the target kernel name or regex.
2. Use a focused GPULlama3 command that launches the kernel repeatedly.
3. Skip warmup/JIT launches.
4. Start with SpeedOfLight.
5. Add targeted sections based on the first classification.

Start with:

```bash
ncu --section SpeedOfLight --csv --kernel-name regex:"<kernel>" --launch-skip <N> --launch-count <M> -- <command>
```

Escalate:

- memory-bound: `MemoryWorkloadAnalysis`
- compute-bound: `ComputeWorkloadAnalysis`
- latency/occupancy issue: `LaunchStats`, `Occupancy`
- scheduler/warp issue: `WarpStateStats`, `SchedulerStats`

## Classification

Use measured SOL metrics:

- high memory throughput, low compute throughput: memory-bound
- high compute throughput, lower memory throughput: compute-bound
- both low with short kernels or low occupancy: latency/occupancy-bound

For the target hot kernel, pay special attention to:

- DRAM throughput
- L1/L2 behavior
- achieved occupancy
- register count
- shared/local memory usage
- global load width and instruction count

## Report

Include:

- target kernel regex
- command
- raw relevant `ncu` output or exported metrics path
- SOL classification
- occupancy/register/shared-memory observations
- recommended kernel or TornadoVM codegen change

Do not present invented or estimated metrics.
