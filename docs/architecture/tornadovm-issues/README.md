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

## 4. An unsigned nibble minus a constant wraps

**Severity: silent infinities in quantized decode.** Every Q4_0 decode recentres an unsigned
nibble by eight. Written straight-line, TornadoVM stamps the nibble unsigned and emits the
subtraction into an unsigned variable, so the seven values below eight wrap to about 4.29e9:

```java
int q = (packed.get(i) & 0xFF) >> 4 & 0xF;
out.set(i, q - 8);
```

```
java.lang.AssertionError: nibble 0 expected:<-8.0> but was:<4.2949673E9>
```

which is exactly 2^32 - 8. [`UnsignedNibbleRecentringAccelTest.java`](UnsignedNibbleRecentringAccelTest.java)
is the whole case. The emitted CUDA:

```c
ui_285 = ch_272 & 255U;
ui_286 = ui_285 >> 4;
ui_287 = ui_286 + -8;      // ui_287 is declared unsigned int
f_288  = (float) ui_287;
```

against the signed form the neighbouring low-nibble path gets:

```c
i_280 = (int) ch_274;
i_281 = i_280 & 15;
i_282 = i_281 + -8;        // i_282 is declared int
f_283 = (float) i_282;
```

How it was found: a tensor-core Q4_0 projection returned NaN for **every** output. 4.29e9 times a
block scale overflows fp16 to infinity as the weight is staged, and the first MMA turns the tile
into NaN. Only the high half was wrong, and only for nibbles under eight.

Every decode in this repository escapes it by accident: they all pick the nibble with a branch or
a ternary, and the phi merging the two arms is stamped signed. A kernel that computes the shift
arithmetically — `(packed >> nibbleShift) & 0xF`, which is what a loop over both halves wants —
does not. `Qwen35MMAProjectionAccelTest.everyNibbleDecodesWithTheSignItsScaleGivesIt` covers all
sixteen values in both halves against both signs of scale, so a future rewrite to the branchless
form fails there rather than in a model.

The workaround is to recentre after the conversion, and it survives the compiler:
`scale * ((float) q - 8.0f)` emits `f_215 = f_214 - 8.0F`. An `(int)` cast on the Java side does
not, because the stamp is already int-kinded; it is the *signedness* of the stamp that is wrong.

Either the stamp for `x & 0xF` should be a signed int — it is, in Java, where `&` on two ints
yields an int — or the subtraction should be emitted in the type Java gives it rather than in one
inferred from the operand's value range.

Environment: TornadoVM 6.0.1-jdk21-dev built locally from develop tip `ae7152e20`, CUDA backend,
JDK 21, RTX 5090 Laptop (compute 12.0).

## 5. `cuda_fp16.h` is omitted when the kernel's only fp16 use is a bare `half` the backend emitted

**Severity: compilation fails outright** — a kernel that stores into a half-precision shared tile is
rejected by NVRTC unless some *other* construct happens to spell fp16 in the generated text.

Environment: TornadoVM **6.0.1-jdk21-dev**, built from source at commit
**`ae7152e20797b13902590183e3f07e06fc76843b`** (branch `develop`, 2026-09-09,
`git describe` = `build-lock-v6.0.0-29-gae7152e20`), CUDA backend, JDK **21.0.2-open**, CUDA toolkit
**13.1 (nvcc V13.1.115)**, driver **580.142**, **NVIDIA GeForce RTX 5090 Laptop GPU (compute 12.0)**.
The version string alone does not identify the source: `6.0.1-jdk21-dev` is a development build, so
the commit above is what the line numbers and the quoted condition refer to.

Reproducer: [`MissingFp16IncludeRepro.java`](MissingFp16IncludeRepro.java). One kernel, four arrays,
one `m16n8k16` MMA step; no model, no GGUF, nothing from this engine.

```bash
export TORNADOVM_HOME=/path/to/tornadovm-6.0.1-jdk21-dev-cuda
export PATH="$JAVA_HOME/bin:$TORNADOVM_HOME/bin:$PATH"
mkdir -p /tmp/fp16repro
javac -proc:none --enable-preview --release 21 \
      -cp "$TORNADOVM_HOME/share/java/tornado/*" -d /tmp/fp16repro \
      docs/architecture/tornadovm-issues/MissingFp16IncludeRepro.java

# fails
tornado --jvm="-Dtornado.recover.bailout=False" \
        --classpath /tmp/fp16repro tornadovmissues.MissingFp16IncludeRepro

# same kernel plus one getHalfFloat read: compiles and prints "out[0] = 9.0"
tornado --jvm="-Dtornado.recover.bailout=False" \
        --classpath /tmp/fp16repro tornadovmissues.MissingFp16IncludeRepro trigger
```

**Expected**: both runs compile and print a finite `out[0]`.

**Actual**, for the first:

```
[TornadoVM-CUDA] NVRTC compilation failed:
tornado_kernel.cu(12): error: identifier "half" is undefined
    __shared__ half half_4[128];
               ^
tornado_kernel.cu(62): error: identifier "half" is undefined
      ((half *) half_4)[__bo >> 1] = f_28;
        ^
```

Both offending lines are **emitted by the backend itself**: the shared tile comes from
`KernelContext.allocateHalfFloatLocalArray`, and the store from
`KernelContext.mmaStoreBSwizzled` (`CUDALIRStmt`, the swizzled-store emitter). The second run emits
the *same* two lines and compiles, because the extra `getHalfFloat(...).getFloat32()` lowers to
`__half2float`:

```c
#include <cuda_fp16.h>          // present only in the second run
  __shared__ half half_5[128];
    ((half *) half_5)[__bo >> 1] = f_29;
```

**Where the condition lives.**
`uk.ac.manchester.tornado.drivers.cuda.graal.compiler.CUDACompilationResultBuilder#finish`
decides the include by scanning the generated source text:

```java
if ((source.contains("__half") || source.contains("half2") || source.contains("2half"))
        && !source.contains("cuda_fp16.h")) {
    source = CUDAPreamble.PREAMBLE + source;
}
```

The scan does not cover the bare `half` spelling that the backend's own emitters produce, so a
kernel whose only fp16 use is a half-typed shared tile misses the include. A compile-time constant
value hides the defect — `new HalfFloat(0.5f)` emits `__float2half(0.5F)`, which matches `2half` —
which is why the reproducer stores a value computed at run time.

**Suspected fix, not applied**: extend that condition to the declaration the backend emits, e.g. add
`source.contains("__shared__ half ")` and `source.contains("(half *)")` to the disjunction, or
decide the include where the half-typed local array and the swizzled store are emitted rather than
by scanning text afterwards. `CUDAPreamble`'s javadoc explains why the include is conditional — some
toolkits' `cuda_fp16.hpp` does not compile under NVRTC — so making it unconditional is not the
suggestion.

**How it was found, and what this repository did about it.** Replacing `projectionMMAQ4_0`'s manual
A-tile staging with `asyncCopyToLocal` removed its last `__half_as_ushort`; the kernel still
compiled because a separate change had already made its block-scale read emit `__half2float`. The
same replacement in `projectionMMAQ5_K` and `projectionMMAQ4_1`, whose scales were decoded from raw
bytes, produced exactly the failure above. Those two kernels then had their scale and minimum reads
moved to `ByteArray.getHalfFloat` — a change worth making on its own terms, and one that also
restores the spelling the scan looks for. **That is an avoidance, not a fix**: the defect is
unchanged, any kernel that stores into a half tile without another fp16 spelling still fails to
compile, and nothing in TornadoVM has been modified.

