---
name: gpullama-nsys-analysis
description: >
  Nsight Systems workflow for GPULlama3/TornadoVM CUDA runs. Use to capture
  CUDA timelines, summarize kernel/API/memcpy time, identify hot kernels,
  launch overhead, syncs, and GPU idle behavior.
tags:
  - gpullama
  - nsys
  - profiling
license: Apache-2.0
---

# GPULlama3 Nsight Systems Analysis

## When To Use

Use `nsys` when you need the system-level picture:

- which CUDA kernels dominate total time
- whether launch overhead is significant
- whether memory copies or synchronization are visible
- whether GPU is idle between kernels
- whether native CUDA libraries appear in the timeline

Do not use `nsys` for detailed per-kernel SOL/occupancy/warp-stall diagnosis. Use `gpullama-ncu-analysis` after identifying a hot kernel.

## Workflow

1. Pick a focused GPULlama3 command and a CUDA backend run.
2. Use a prompt/flag combination that reaches the phase being studied.
3. Capture a trace under a timestamped artifact directory.
4. Run targeted `nsys stats` reports.
5. Classify the bottleneck before recommending changes.

Useful reports:

```bash
nsys stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum <trace.nsys-rep>
nsys analyze -r all <trace.nsys-rep>
```

When CUDA libraries are involved and traced:

```bash
nsys stats -r cublas_api_sum,cudnn_api_sum <trace.nsys-rep>
```

## Interpretation

Classify:

- hot generated kernel dominates: use `gpullama-ncu-analysis`
- many small kernels / high launch API time: launch-overhead-bound
- high memcpy time: transfer-bound or persistence issue
- high synchronization API time: sync-bound
- GPU gaps: host/TornadoVM scheduling or graph/persistence issue
- cuBLAS/cuDNN dominates: inspect native library call shape and stream behavior

## Report

Include:

- exact profile command
- exact GPULlama3 command
- trace path
- top kernels by total GPU time
- CUDA API summary
- memory transfer summary
- bottleneck classification
- next action

Every numeric claim must cite `nsys stats`, `nsys analyze`, or a parsed trace artifact.
