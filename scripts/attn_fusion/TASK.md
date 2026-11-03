# Lowering-agent task: Qwen3.5 full-attention mixer in AveLang

Implement the computation described below using the **current** AveLang DSL on
H100 (sm_90a). The authoritative source is
`model-tunner/attn_fusion/bench.py`; `input_model.py` is its reference extracted
without any CuTe implementation. Generate the DSL yourself from this semantic
description. Do not inspect existing attention implementations in
`model-tunner/attn_fusion/ave*`, other generated candidates, or archived traces.

## Deliverable and interface

Write `candidate.py` containing one or more real `@avelang.jit` GPU kernels and:

```python
def build(x, hidden, w, tables):
    # Allocate scratch/output tensors and return an ordered dict of zero-argument
    # launch callables, plus (attended, output). Do not execute the computation
    # in build; the evaluator executes the callables in insertion order.
    return kernels, (attended, output)
```

All inputs/outputs are contiguous CUDA tensors. Host allocation, shape/view
operations and kernel launches may use torch. All runtime numerical computation,
including weight scaling and residual addition, must use AveLang kernels. No
PyTorch compute fallback, Triton, CuTe, custom CUDA, new compiler intrinsics,
precomputed outputs, or input-dependent host transfers. Kernels must read their
current inputs on every invocation and support CUDA graph replay. You may choose
the decomposition and fusion boundaries; correctness comes before optimization.
Fixed dimensions below are sufficient; do not add a generalized framework.

Also write `compile_candidate.py` to compile every kernel offline for sm_90a
with `validate_invariants=True`, save PTX under `/results`, and run ptxas to
check assembly. Use the offline compilation pattern in
`scripts/hopper/test_hopper.py`. It must not probe or initialize a GPU.

## Shapes and layout

- B=16, T=26, query heads H=8, KV heads KV=2, D=256, hidden=2048.
- M=B*T=416, packed width=5120, rotary dimension=64, epsilon=1e-6.
- x and hidden: BF16 [16,26,2048]. hidden is the residual input.
- w['qkv']: BF16 [5120,2048]; w['o']: BF16 [2048,2048].
- w['q_norm'], w['k_norm']: BF16 [256], stored as offsets from one.
- tables: BF16 [26,32,2], cos then sin; position starts at zero in each batch.
  The evaluator provides tables; kernels must use them, not regenerate them.
- attended and output: BF16 [16,26,2048].

## Computation and rounding contract

1. `packed = BF16(FP32(x) @ FP32(w['qkv']).T)` after flattening B,T.
   At each token, the first 4096 channels are eight interleaved blocks
   `[q_head(256), gate_head(256)]`, then 512 K channels, then 512 V channels.
   Q and gate are NOT stored as separate 2048-channel prefixes.
2. Independently normalize each Q and K head across all 256 channels:
   `z = BF16(FP32(t) * rsqrt(mean(FP32(t)**2) + 1e-6)
             * (1 + FP32(norm_weight)))`.
3. Apply split-half RoPE to z: for d in [0,32),
   `out[d] = BF16(FP32(z[d])*cos - FP32(z[d+32])*sin)` and
   `out[d+32] = BF16(FP32(z[d+32])*cos + FP32(z[d])*sin)`.
   Channels 64..255 are unchanged. The BF16 rounding BEFORE RoPE matters.
4. Causal grouped-query attention: query head h uses KV head h//4;
   score(q,k) = sum_d FP32(Q[q,h,d])*FP32(K[k,h//4,d]) / sqrt(256).
   Only keys k<=q in the SAME batch are visible. Stable softmax over keys;
   weighted V reduction uses FP32 accumulation, then rounds to BF16.
   Do not introduce a BF16 softmax-probability rounding point.
5. `attended = BF16(FP32(BF16(attention_result))
                     * FP32(BF16(sigmoid(FP32(gate)))))`.
6. `projected = BF16(FP32(attended) @ FP32(w['o']).T)`.
7. `output = BF16(FP32(hidden) + FP32(projected))`.

The evaluator checks both attended and final output. The original benchmark's
threshold is cosine >=0.9999. The evaluator also reports finite values, max/mean
absolute error, relative L2 error, changed inputs, structured inputs, and graph
replay. Never relax tolerances or modify the reference/evaluator to fit a kernel.

## Current compiler and knowledge sources

The compiler is `.repro/avelang-artifact` in the Lumen repository (public artifact
commit 86d4cdd plus the already-tested local Hopper patch). Do not modify it or
the sibling `avelang` checkout. Read its `docs/content/language-reference/`,
`python/avelang/language/{core,nvvm,invariant}.py`, and primitive tests for syntax.
`scripts/hopper/README.md`, `gemm.py`, and `test_hopper.py` document the new
Hopper interfaces and an unrelated syntax/reference example; they may be read.

- BF16 WGMMA supports shared/shared m64n8k16, FP32 accumulation, 128 threads.
  Its helper waits for completion before returning. A/B packing and accumulator
  lane mapping are documented in `scripts/hopper/README.md`.
- mbarrier and async-proxy fence exist. TMA and WGMMA pipeline overlap do not.
- NVIDIA scalar math, shuffles, shared storage and CTA barriers are available.
- Turn invariant validation ON. Instantiate meaningful data-flow tag assertions
  for transformations the current compiler supports, particularly coordinate
  correspondence across packed data, RoPE pairs, GQA head selection, and gates.
  You choose assertions and layouts; test negative mutations where practical.
  WGMMA arithmetic tag transfer and synchronization-protocol proofs are NOT
  implemented: do not fabricate proofs, disable validation, or silently drop
  failing invariants. Report unsupported invariants with compiler diagnostics.

## Feedback loop and file ownership

The coordinator supplies only semantics, public DSL references and compiler/test
feedback. You own `candidate.py`, `compile_candidate.py`, and `AGENT_NOTES.md` in
this directory. Record implementation choices, attempts, invariant limitations
and feedback-driven repairs in AGENT_NOTES.md. Do not edit input_model.py,
evaluate.py, run.sh or TASK.md. Do not copy an existing attention DSL solution.

Run CPU compilation via `scripts/attn_fusion/run.sh cpu` once the coordinator has
created the wrapper. The coordinator runs GPU validation and returns feedback;
you must not start GPU work independently. Preserve a passing version before
trying an optimization. Only measured improvements may replace a valid baseline.

This is an initial generation / lowering experiment following the paper's
compile–invariant–test–profile feedback pattern. It does not claim to reproduce
the full ICRL planner/selector training loop or unpublished Hopper invariant rules.
