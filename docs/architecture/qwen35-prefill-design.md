# `qwen35` prompt ingestion — design record

How Qwen3.5/3.8 ingests a prompt on an accelerator, in the two modes beyond `STANDARD`:
`PREFILL_DECODE` and `BATCH_PREFILL_DECODE`. Written against
[`qwen35-port-proposal.md`](qwen35-port-proposal.md), which covers the single-token path.

## 1. What makes this family different during ingestion

Three quarters of the trunk is not attention. A Gated Delta Net layer carries two pieces of state
that make a prompt token depend on its predecessor **inside the layer**:

- a depthwise causal convolution over a rolling window of `kernel - 1` previous inputs, per channel;
- a delta-net matrix per value head, decayed and rank-one updated once per token.

Consequences, and they are the constraints everything below is checked against:

1. **Prompt tokens are not independent inside a recurrent layer.** Anything that computes token
   `t` before token `t-1` has updated the window and the state is wrong, not merely reordered.
2. **The recurrence must not be reset at the prefill/decode boundary.** It is the sequence's whole
   history; decode continues it.
3. **The result cannot depend on the chunk size.** A chunk is a scheduling unit. Sixteen tokens in
   one chunk and sixteen in sixteen chunks must leave bit-comparable state.
4. **Only 16 of 64 blocks hold key/value entries**, addressed by a dense index. A batch that
   appends KV must append exactly once per token, at that token's own position.

## 2. `PREFILL_DECODE`

Sequential ingestion: one token per invocation through the layer graphs, with the logits graph
skipped, then the existing single-token decode.

**It reuses the single-token layer computation unchanged.** The recurrent scan is already exactly
one token per invocation, and the convolution window and delta-net state already persist on the
device across invocations, so prefill is the decode graphs minus the vocabulary projection. What
differs is only the graph the first layer consumes from — `decodeActivation` rather than
`activationUpdate` — which the layer builder takes as a parameter.

- No vocabulary projection, no sampling and no logits readback for a prompt position: the
  `N+2`-graph plan runs graphs `0..N` during prefill and all `N+2` during decode.
- The decode position and first decode token are derived from the number of tokens actually
  ingested, never from the prompt size. The generic loop already does this; a regression test pins
  it, because the defect it prevents — prefilling every prompt token and then decoding from the
  last one again — puts that token in the cache twice and is invisible except as wrong logits.
- Generated-token budget semantics are the loop's, identical to `STANDARD`.

## 3. `BATCH_PREFILL_DECODE`

A chunk of `B` prompt tokens per invocation. **A host loop calling the single-token plan `B` times
is not batch prefill**, so the layer graphs below are batched: one invocation per chunk per layer.

### The recurrent layer, batched

The projections batch trivially — they are the same weights against `B` activation rows — and the
recurrence does not. It is scanned **inside the kernel**, over the chunk, in token order:

| Step | Parallelism | Sequential over |
| --- | --- | --- |
| input RMS norm | one workgroup per row | — |
| `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta` projections | one workgroup per (row, output row) | — |
| decay and beta | one lane per (row, value head) | — |
| causal convolution | one lane per **channel** | the chunk's tokens |
| SiLU, split, L2 norm, query scale | one lane per (row, element or head) | — |
| delta rule | one lane per (value head, value column) | the chunk's tokens |
| gated norm | one lane per (row, value head) | — |
| `ssm_out` projection + residual | one workgroup per (row, output row) | — |
| dense SwiGLU feed-forward | one workgroup per (row, output row) | — |

The two scanned kernels are exact rather than approximate. A convolution channel's window is
private to that channel, so a lane that walks the chunk in order performs the identical updates the
single-token kernel performs one invocation at a time. A delta-net value column's state is private
to that column for the same reason. Nothing is shared between lanes in either, so no barrier and no
cross-lane ordering is involved — the sequential dependency is entirely inside one lane's loop.

This is why the result cannot depend on `B`: the arithmetic per token is the same expression in the
same order, and the only thing `B` changes is how many iterations a lane runs before the kernel
returns.

### The attention layer, batched

Rows are independent given the key/value store, so a chunk is processed as a causal batch:

- each row's position is `startPos + row`, read from the chunk's position holder, not from a
  single scalar;
- keys and values are appended **once per row**, at that row's own position, before attention;
- a row attends over `[0, startPos + row]` — the mask is the loop bound, so a later row is not
  reachable from an earlier one;
- the query/gate split, the per-head query and key norms, the partial rotation and the output gate
  are the single-token kernels with a row index added.

The key/value store is left in exactly the state sequential ingestion would leave it in, which is
what decode then reads.

### Padding

The kernels launch a fixed `B` rows and are told how many are active. An inactive row does not
rotate, does not append key/value entries, and does not advance a recurrent lane's scan.

## 4. What is shared and what is this family's

Reused unchanged: `batchedRmsReduce` and `batchedRmsApplyFP32` (dtype-neutral, FP32 in and out),
the plan and component contracts, the master plans, the generic generation loop, the memory model.

New, and stated per operation rather than per family: batched matrix-vector kernels for the
representations this model actually holds (Q4_0, Q4_1, Q5_K, F32), and a batched fused gate/up
SwiGLU for Q4_0.

New and genuinely `qwen35`-shaped: the two scanned recurrent kernels and the batched forms of this
family's own mixer kernels. No generic recurrent framework is introduced — there is one family
with a recurrence, and a framework for it would be vocabulary ahead of a second caller.

## 5. Mixed native quantization is unchanged

Every batched task is selected by the representation of the tensor it reads, before compilation:
Q4_0 projections and embeddings, Q4_1 `ffn_down` on the early blocks, Q5_K `ssm_out`, Q6_K
vocabulary projection, F32 norms and SSM parameters. Nothing is materialized to Q8_0. A fused task
validates that its operands share a representation and refuses a mixture by name.

The batched path accumulates in **FP32**, like the single-token path. The existing Q8_0 batched
GEMM dequantizes to FP16 and accumulates through tensor cores, which is why it cannot meet the
single-token bounds; this family does not use it, and inherits no looser contract.

## 6. Verification

The CPU `STANDARD` path is the reference for both modes, teacher-forced, at the bounds the
single-token path already meets. No tolerance is widened.

- prefill/decode transition: first decode logits, subsequent rows, generated token sequence;
- final key/value state, convolution window and delta-net state after ingestion;
- batch sizes 1, 2, 7, the default, and one larger than the prompt (a partially active chunk);
- prompt lengths exactly on, one below and one above a chunk boundary;
- the same prompt under different chunk sizes produces the same state and the same decode logits;
- reset and a second independent conversation.
