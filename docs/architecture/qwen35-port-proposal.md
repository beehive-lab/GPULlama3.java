# `qwen35` architecture port — design proposal

**On the name.** The model is **Qwen3.8-27B**. The *architecture* it declares is `qwen35`,
and that string — not the model's name — is what `GgufRecognition` turns into an
`ArchitectureId`. llama.cpp does the same: one `LLM_ARCH_QWEN35` serves the Qwen3.5, 3.6 and
3.8 releases, which share this layer topology. Java types are therefore named `Qwen35*` after
the architecture, never `Qwen38*` after this one file; a Qwen3.5 GGUF declares the same
architecture and must reach the same family.


Motivating file: `Qwen3.8-27B-Q4_0.gguf` (Unsloth, 16 GB, `general.architecture = qwen35`).
Reference implementation: `llama.cpp` `src/models/qwen35.cpp` and `src/models/delta-net-base.cpp`.

This is not a Qwen3 alias. Three quarters of its layers are not attention layers at all,
and the operation vocabulary cannot express them today. Per the porting skill, that means a
proposal before code.

## 1. Inventory

### Metadata

| Key | Value |
| --- | --- |
| `general.architecture` | `qwen35` |
| `qwen35.block_count` | 65 (64 trunk + 1 MTP/NextN) |
| `qwen35.embedding_length` | 5120 |
| `qwen35.feed_forward_length` | 17408 |
| `qwen35.attention.head_count` | 24 |
| `qwen35.attention.head_count_kv` | 4 |
| `qwen35.attention.key_length` / `value_length` | 256 / 256 |
| `qwen35.attention.layer_norm_rms_epsilon` | 1e-6 |
| `qwen35.rope.freq_base` | 1e7 |
| `qwen35.rope.dimension_count` | 64 |
| `qwen35.rope.dimension_sections` | `[11, 11, 10, 0]` (MRoPE) |
| `qwen35.full_attention_interval` | 4 |
| `qwen35.ssm.conv_kernel` | 4 |
| `qwen35.ssm.state_size` | 128 |
| `qwen35.ssm.group_count` | 16 |
| `qwen35.ssm.time_step_rank` | 48 |
| `qwen35.ssm.inner_size` | 6144 |
| `qwen35.nextn_predict_layers` | 1 |
| `qwen35.context_length` | 262144 |
| `tokenizer.ggml.model` / `.pre` | `gpt2` / `qwen35` |
| vocabulary | 248320; `<|endoftext|>` at 248044 is the first special token |
| BOS / EOS / PAD | 248044 / 248046 / 248055 |

### Layer topology

`is_recurrent(il) = (il + 1) % full_attention_interval != 0`, for `il < 64`.

- **48 recurrent layers** — Gated Delta Net linear attention (il ∈ {0,1,2, 4,5,6, …}).
- **16 attention layers** — il ∈ {3, 7, …, 63}.
- **1 MTP block** — il = 64, shaped like an attention layer plus NextN tensors.

Every layer, of either kind, then runs the same dense SwiGLU FFN. The residual topology is
`x = x + branch(rms(x))` for the mixer and `x = y + ffn(rms(y))` for the FFN, where
`post_attention_norm` is the FFN's input norm (there is no separate `ffn_norm`).

### Attention-layer geometry

Head dimension is **256**, stated by `key_length`/`value_length` — *not* `5120 / 24`.

- `attn_q` is `(5120, 12288)`: per head, `[query(256) | gate(256)]` **interleaved**.
  Element stride between heads is `2 * 256`; the query view starts at 0 and the gate view at 256.
- `attn_k`, `attn_v` are `(5120, 1024)` = `256 * 4` KV heads.
- `attn_output` is `(6144, 5120)`, `6144 = 256 * 24`.
- Q and K carry per-head RMS norms of length 256.
- RoPE is **partial**: `n_rot = 64` over a 256-wide head, NEOX halves, base 1e7. MRoPE
  sections apply to multimodal position triples; with text-only positions all sections carry
  the same position and `ggml_rope_multi` degenerates to plain NEOX RoPE over the first 64 dims.
- The attention result is multiplied elementwise by `sigmoid(gate)` before `attn_output`.
- Score scale is `1 / sqrt(256)`.

### Gated Delta Net (recurrent) layer geometry

Derived, not stated: `head_k_dim = ssm.state_size = 128`, `n_k_heads = ssm.group_count = 16`,
`n_v_heads = ssm.time_step_rank = 48`, `head_v_dim = ssm.inner_size / n_v_heads = 128`,
`key_dim = 2048`, `value_dim = 6144`, `conv_dim = 2 * key_dim + value_dim = 10240`.

| Tensor | Shape | Role |
| --- | --- | --- |
| `attn_qkv` | (5120, 10240) | fused q(2048) ‖ k(2048) ‖ v(6144) |
| `attn_gate` | (5120, 6144) | the `z` gate |
| `ssm_conv1d` | (4, 10240) | depthwise causal conv over the 10240 mixed channels |
| `ssm_alpha` | (5120, 48) | per-v-head decay projection |
| `ssm_beta` | (5120, 48) | per-v-head write strength |
| `ssm_dt.bias` | (48,) | bias added before softplus |
| `ssm_a` | (48,) | `-exp(A_log)`, multiplies the softplus |
| `ssm_norm` | (128,) | gated RMS norm over `head_v_dim` |
| `ssm_out` | (6144, 5120) | output projection |

Per token, per v-head `h` (state `S[h]` is `128 × 128`, keys repeated `48 / 16 = 3` times):

```
qkv   = W_qkv · x                       # 10240
qkv   = silu(causal_conv1d(qkv))        # depthwise, kernel 4, per-channel rolling state
q,k,v = split(qkv, 2048 | 2048 | 6144)
q, k  = l2_normalize(q), l2_normalize(k)     # per 128-wide head
q    *= 1 / sqrt(128)
beta  = sigmoid(W_beta · x)                                     # 48
g     = exp(ssm_a * softplus(W_alpha · x + dt_bias))            # 48
S[h] *= g[h]
d     = (v[h] - Sᵀ[h]·k[h]) * beta[h]
S[h] += k[h] ⊗ d
o[h]  = Sᵀ[h] · q[h]
out   = W_out · ( rms_norm(o, ssm_norm) * silu(z) )
```

This is the `build_delta_net_autoregressive` recurrence from `delta-net-base.cpp`, which is
exact for one token at a time. The chunked variant is a prefill throughput optimization over
identical arithmetic and is not required for correctness.

### MTP / NextN block (il = 64)

`nextn.eh_proj (10240, 5120)`, `nextn.enorm (5120)`, `nextn.hnorm (5120)`,
`nextn.shared_head_norm (5120)`, plus a complete attention-layer weight set. `embed_tokens`
and `shared_head_head` are absent, so the block reuses `token_embd` and `output`.

```
h_e   = concat(rms(embed(next_token), enorm), rms(h_main, hnorm))   # 10240
cur   = W_eh_proj · h_e
cur   = attention_block(cur) ; cur = ffn_block(cur)
logits = output · rms(cur, shared_head_norm)
```

`h_main` is the trunk's hidden state **after** `output_norm` and **before** the LM head, so
the main forward pass must expose it.

### Quantization mix

| Type | Count | Where |
| --- | --- | --- |
| F32 | 456 | all norms, all SSM parameters, `ssm_conv1d` |
| Q4_0 | 352 | most projections, `token_embd` |
| **Q4_1** | 8 | `blk.0..7.ffn_down` |
| Q5_K | 48 | every `ssm_out` |
| Q6_K | 1 | `output` |
| Q8_0 | 1 | `blk.64.nextn.eh_proj` |

`output` is a distinct tensor from `token_embd`; the weights are **not** tied.

## 2. Extension points

| Point | Answer |
| --- | --- |
| `ModelProvider` | Yes, unchanged. `GgufRecognition`'s `default` branch already maps `qwen35` to `ArchitectureId.of("qwen35")`; only a `FamilyProvider` and a loader are new. |
| `ModelArchitecture` | **No** — see §3. Deferred: the family ships CPU-only first, and `Gemma4` is the standing precedent for a family with a provider and a CPU forward pass but no `describe(...)`. |
| `CpuForwardProvider` | Needs new arithmetic: L2 norm, softplus, causal depthwise conv1d with state, the delta rule, and a gated RMS norm. |
| `TornadoPlanProvider` | Declares nothing initially. The family is unclaimed on the GPU and fails by name. |
| `TornadoLoweringProvider` | None. |
| `KvStorageFactory` | Attention layers use the existing layout. Recurrent layers hold **no KV at all**; their state is per-session conv and delta-net state, which is session state, not KV storage. |
| `TensorRole` | New roles for the SSM tensors — but the role vocabulary is only consumed by the program description, which this port does not add yet. |
| `DataType` | **`Q4_1` is missing.** Everything else is present. |
| state / workspace | New: per-layer conv state `(3, 10240)` and delta-net state `(48, 128, 128)`; KV allocated only for the 16 attention layers. |

## 3. The new concepts

### 3.1 `DataType.Q4_1` and a `Q4_1FloatTensor`

The smallest possible change: `Q4_1` is GGML block type 3, 32 values per block, `d` and `m`
as two halves followed by 16 packed bytes; `value = d * nibble + m`. `GGMLType.Q4_1` already
carries the right block size. Like `Q4_0` it is block-encoded — decoded inside the arithmetic
that consumes it, on the host and on a device alike (see §4a). It is not converted for either.

*Alternative rejected:* dequantizing those 8 tensors to F32 at load. It costs 1.1 GB of heap
for a saving of one small class, and hides a supported quantization behind a special case.

### 3.2 Four new `OperationKind` values

Adding to a closed vocabulary is the decision this proposal exists to justify. Each is
arithmetic no existing kind performs, and each is stated in terms other families could use.

| Kind | Meaning | Why not an existing kind |
| --- | --- | --- |
| `L2_NORM` | Divide a vector by its L2 norm. | `RMS_NORM` divides by root-*mean*-square and applies a learned scale. Different scalar, no weight. |
| `CAUSAL_CONV_1D` | Depthwise causal convolution over a channel vector with a retained window. | No convolution exists in the vocabulary at all. Mamba, Mamba2, Qwen3-Next and RWKV all need this one. |
| `DELTA_RULE_UPDATE` | One step of the gated delta rule: decay the state, compute the write, accumulate the outer product, read out. | This is the recurrent mixer. Expressing it as `MAT_VEC` + `RESIDUAL_ADD` would name neither the state nor its update, and no backend could fuse it. |
| `GATED_NORM` | `rms_norm(x, w) * silu(gate)`. | `SWIGLU` is `silu(gate) * up` with no normalization; composing `RMS_NORM` then `SWIGLU` needs a scratch buffer the fused form does not. |

Deliberately **not** added:

- **Gated attention output.** `sigmoid(gate) * attention_result` is `SWIGLU`'s sibling but with
  a logistic rather than SiLU. Rather than a `qwen35`-shaped composite, the attention layer's
  gate is applied by a `SCALE`-like elementwise multiply the CPU path already performs, and the
  program description (when it lands) will express it as an elementwise product of two values.
- **A "linear attention" mega-operation.** Naming one operation after this family's mixer is
  precisely the failure the skill's §3 warns about. The mixer is `MAT_VEC` → `CAUSAL_CONV_1D` →
  `L2_NORM` → `DELTA_RULE_UPDATE` → `GATED_NORM` → `MAT_VEC`.
- **`SOFTPLUS`, `SIGMOID`, `EXP`.** Scalar activation of a projection result. Folded into
  `DELTA_RULE_UPDATE`'s inputs, which is where they are consumed and where a backend would
  fuse them anyway.
- **MRoPE.** With text-only positions it is `ROPE` with a rotation width smaller than the head.
  Partial rotation is a parameter of `ROPE`, not a second kind. If image and video positions
  are ever supported, that is when the question is reopened.

### 3.3 Recurrent session state

Conv state and delta-net state are per-sequence, mutable, and sized at allocation from
configuration. That is exactly what `State` already is, so they are fields on a
`Qwen35State`, not a new abstraction. They are **not** KV storage: they are not appended to,
not paged, not shared, and not leasable, so routing them through `KvStorage` would give the
manager something it cannot page and cannot evict.

Consequence for `reset()`: a recurrent state must be zeroed, where a KV cache only needs its
position rewound. `Qwen35State` overrides reset accordingly.

## 4. Compatibility

Nothing above changes for any existing family. `OperationKind` gains four values that no
current program description emits; `DataType` gains one value no current model file uses;
`GgufRecognition` is untouched. The one shared file that changes behaviourally is
`Qwen3Tokenizer`, which gains a constructor taking the split pattern — the `qwen35`
pre-tokenizer differs from Qwen3's only in that letter runs also consume combining marks
(`[\p{L}\p{M}]+` for `\p{L}+`, and `\p{M}` excluded from the punctuation run).

## 4a. Quantized weights are retained on every backend

**This supersedes the memory reasoning in §3.1 and §5, and the `DataType` documentation that said
the format-decoded types are CPU-only.** The engine's position is now:

- **Quantized storage is retained on the CPU and on accelerators alike.** A tensor keeps the block
  layout the file gave it, and the blocks and their scales and minima are transferred as they lie.
- **Decoding during compute is an implementation property, not a CPU-only one.** Both a host dot
  product and a device kernel decode a block inside the arithmetic that consumes it. That is what
  "block-encoded" means, and it says nothing about which backend can hold the representation.
- **Materialization to `Q8_0` is not the normal accelerator fallback.** It nearly doubles a 4-bit
  model's device footprint, which is the difference between a 27B model fitting in 24 GB and not.
  Where it still happens it is a stated, declared decision, never an unremarked one.
- **Support is declared per operation, per dtype, and where it matters per tensor role.** Not per
  backend and not per model. "The GPU cannot do Q5_K" was never true as stated; what is true is
  that a particular operation may have no Q5_K kernel yet.
- **Fusion remains the backend's choice.** A program says what is computed; whether three
  projections become one task is the backend's business.
- **Mixed quantization between tensors is legal and normal.** Qwen3.8-27B is mixed by
  construction: Q4_0 projections, eight Q4_1 `ffn_down`, Q5_K `ssm_out`, a Q6_K output projection,
  a Q8_0 MTP projection, F32 norms and SSM parameters.
- **Every operand fused into one kernel must have a layout combination that kernel explicitly
  supports.** A fused kernel written for three Q4_0 operands must *reject* a Q4_0/Q5_K/Q4_0 triple,
  at load or plan construction — not read one block layout as another. That rejection is a named
  failure, never a silent conversion of the odd tensor.

### What `DataType` says, and what it does not

`DataType` describes a representation. It answers whether values are stored in blocks with scales
(`isQuantized`), and nothing else about capability. It does **not** answer:

- whether a backend can store the representation — that is the backend's storage vocabulary;
- whether a given operation has a kernel for it — that is `OperationSupport`;
- what to convert it to — there is no general answer, and `materializedFallback` is narrowed to the
  one case that is a real narrowing rather than a capability gap (`BF16` to `F16`, because no BF16
  device arithmetic is used).

### The fixture, measured

Qwen3.8-27B-Q4_0's weight bytes, retained, total **14.944 GiB**. Materialized as `Q8_0` they are
roughly 27 GiB. Nothing about the model changed; only what the loader does with it.

## 5. Backends

| Backend | Position |
| --- | --- |
| CPU | The reference, and initially the only one. |
| CUDA / OpenCL / Metal | Unclaimed. No `TornadoPlanProvider` is registered, so a GPU request fails by name rather than silently running something else. |

**Superseded by §4a.** This originally said the GPU was blocked by memory, because the loader
materialized `Q4_0`, `Q4_1`, `Q5_K` and `Q6_K` as `Q8_0`, turning a 16 GB file into roughly 28 GB
against a 24 GB device. Native retention removes that: the file's own weight bytes are 14.944 GiB
and that is what the device holds. What remains is kernels.

## 6. Verification plan

In dependency order, each step testable before the next exists.

1. `Q4_1FloatTensor` against a hand-computed block, and against `Q8_0` on the same values.
2. Tokenizer: the `qwen35` split pattern over a fixture containing combining marks, and a
   round trip over the model's own vocabulary.
3. Configuration: `validateConfiguration` rejects a metadata block whose derived
   `head_v_dim` is not integral, or whose `attn_q` width is not `2 * key_length * head_count`,
   **before** any weight is read.
4. A `Qwen35CpuOperationEquivalenceTest` over a tiny synthetic model — no GGUF, no device —
   pinning the decomposition of both layer kinds.
5. The delta-net recurrence against values captured from `llama.cpp` for the same inputs.
6. Real fixture: `Qwen3.8-27B-Q4_0.gguf` registered in `GoldenFixture` with its SHA-256,
   skipping by name when absent.
7. Token-level smoke: a prompt whose correct answer is a substring of the generated text,
   and a logits comparison against `llama.cpp` on the same file, same prompt, greedy.
8. Lifecycle: reset (including that the recurrent state is zeroed), multi-turn recall,
   close, use-after-close.
9. MTP: the draft head's logits, then the accept/reject loop asserted **token-identical** to
   generation with speculation disabled. Speculative decoding that changes the output is a
   defect, and this is the assertion that says so.
10. `./mvnw test` for the architecture rules and the service-registration count.

No CPU/GPU parity gate applies while no backend claims the family; it becomes the primary
gate the moment one does.

## 7. Sequence

1. `Q4_1` support.
2. Recognition, configuration, validation, tokenizer, chat format.
3. Tensor loading and weights.
4. `Qwen35State`, including the recurrent state.
5. CPU forward pass: attention layers, then recurrent layers.
6. Verification against `llama.cpp` on the real file.
7. MTP block and speculative decoding.
8. Docs, CI matrix rows, `models-and-backends.md`.
