# TornadoVM findings, with reproducers

> **Correction.** An earlier draft of the gap analysis listed "no `dp4a`/integer-MMA intrinsic" as
> a third finding to file. That was wrong twice over. TornadoVM exposes `mmaInt8` at
> `MMAShape.M16N8K32`, along with `mmaLoadAInt8`, `mmaLoadBInt8`, `mmaFragmentInt`, `mmaStoreInt`
> and `swizzleLoadInt8`, and FP16 `mma`/`mmaBF16` besides. It also exposes **`dp4a`**, outside
> `KernelContext`: `QuantizationUtils.dp4a(Int8Array, long, Int8Array|byte[], long, int)`,
> `QuantizationUtils.dp4a_packed(int, int, int)` and `QuantizationUtils.dequantizeFusedResult`,
> registered in `CUDAGraphBuilderPlugins` and emitted as an inline `dp4a.s32.s32`
> (`Dp4aNode` / `DP4APackedNode` -> `CUDALIRStmt`). `tornado-examples`'
> `MatrixVectorRowMajor.matrixVectorGenericDP4A` is a worked use. Nothing needs requesting.

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

## 3. An MMA accumulator fragment cannot be indexed from Java

**Severity: blocks per-block rescaling, the shape every quantized integer MMA needs.**

`ctx.mma(...)` returns a `float[]` (and `ctx.mmaInt8(...)` an `int[]`) standing for the warp's four
accumulator elements. Touching one at a **constant** index fails to lower:

```
uk.ac.manchester.tornado.api.exceptions.TornadoInternalError: unimplemented:
address origin unimplemented:
uk.ac.manchester.tornado.drivers.cuda.graal.nodes.CUDAMMAComputeNode
```

[`MmaFragmentAccessAccelTest.java`](MmaFragmentAccessAccelTest.java) — one warp, one `m16n8k16`
step, then `acc[0] * 2.0f`. Writing (`acc[0] = acc[0] * 2.0f`, then `mmaStore`) and reading
(`out.set(lane * 4, acc[0] * 2.0f)`) fail identically, so what is unsupported is addressing the
fragment value, not the direction of the access.

Why it matters: a quantized integer MMA has a weight scale per weight block and an activation
scale per activation block, so their product changes every K-block. llama.cpp's MMQ applies that
product to the integer accumulator elements between MMA steps. Without fragment access the only
route is `mmaStore`/reload per K-block — 160 round-trips per output tile at K=5120 — which is why
this repository's Q4_0 tensor-core path decodes to FP16 and accumulates in FP32 instead.

It does **not** block `QuantizationUtils.dp4a_packed`, whose result is an ordinary `int`.

Either fragment elements should be addressable at a constant index, or a scaling primitive
(`mmaScale(fragment, float)`) would cover the case it is wanted for.

Environment: TornadoVM 6.0.1-jdk21-dev built locally from develop tip `ae7152e20`, CUDA backend,
JDK 21, RTX 5090 Laptop (compute 12.0).
