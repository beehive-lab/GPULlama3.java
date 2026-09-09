# TornadoVM findings, with reproducers

> **Correction.** An earlier draft of the gap analysis listed "no `dp4a`/integer-MMA intrinsic" as
> a third finding to file. That was wrong: TornadoVM exposes `mmaInt8` at `MMAShape.M16N8K32`,
> along with `mmaLoadAInt8`, `mmaLoadBInt8`, `mmaFragmentInt`, `mmaStoreInt` and `swizzleLoadInt8`,
> and FP16 `mma`/`mmaBF16` besides. Nothing needs requesting. The two findings below stand.

Behaviour hit while optimizing the `qwen35` batched-prefill kernels, isolated far enough to hand
upstream. These files are **not** part of the build — the first is a test that fails on purpose,
and putting it under `src/test` would break the accelerator gate.

Environment for both: TornadoVM 6.0.0-jdk21, CUDA backend, JDK 21, NVIDIA RTX 5090 Laptop
(compute 12.0), driver 580.142, CUDA 13.1.

## 1. A private array inside a kernel is not zero-initialized

**Severity: silent wrong results.** Java guarantees `new float[n]` is zero-filled. In a generated
kernel it is uninitialized stack, so a kernel that allocates accumulators and adds into them
without writing them first reads garbage.

[`PrivateArrayZeroInitAccelTest.java`](PrivateArrayZeroInitAccelTest.java) — 256 threads, each
allocating `new float[8]`, adding `1.0f` to every slot and summing. Expected `8.0`; observed:

```
java.lang.AssertionError: thread 0 expected:<8.0> but was:<NaN>
```

How it was found: a batched projection kernel accumulating into `new float[ROW_TILE * COL_TILE]`
produced `NaN` logits for a 27B model. The fix in our code is an explicit zeroing loop, which is
what the comment in `TransformerComputeKernelsQ4_0` now says. That workaround is fine; the
surprise is that it is needed, because nothing in the Java source suggests it.

Either the allocation should be zeroed, or — if zeroing every private array is too expensive to do
unconditionally — the restriction belongs in the documentation for `KernelContext` kernels, loudly.

## 2. A kernel helper is rejected outright when it exceeds the inlining size

**Severity: forces hand-duplicated kernel bodies.** A `private static` helper shared by two kernels
that differ only in what they do with the result:

```
uk.ac.manchester.tornado.api.exceptions.TornadoInliningException:
Method Invoke#...TransformerComputeKernelsQ5_K.tileDot(
    KernelContext, FloatArray, ByteArray, int, int, int, int, int, float[])
cannot be inlined: node count (714) exceeds limit (600)
```

The helper is an ordinary tiled dot product: a loop over the row, eight accumulators, a
shared-memory tree reduction. The limit is Graal's `MaximumInliningSize`, read in
`TornadoPartialInliningPolicy`, and the failure is an exception rather than a fall-back to a real
call — so the only way forward is to copy the body into both kernels, which is what this repository
now does in three places.

Two things would help, in order of preference: a documented way to raise the limit for a task
graph, or a diagnostic that names the option rather than only the number.

To reproduce: any `KernelContext` kernel whose helper exceeds 600 nodes. The `tileDot` shape above
is one; the version that fails is recoverable from this repository's history at the commit that
introduced the Q5_K tile.
