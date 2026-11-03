# Agent feedback: fuse the attention path without reducing probability precision

User request: optimize another round after the round-3 GEMM work. The agent
writes the DSL; the coordinator owns independent checks, timing and profiling.
The round-3 baseline is frozen with SHA256
`c25bacf5405cec557f226d442e7a6cd4c4fd5bf518fb2a48650ee2e26712fd71`.

## Evidence and scope

H100 NVL, B16/T26/H8/KV2/D256, BF16 inputs/output. The round-3 single-layer
chain is 66.572 us. QKV is already 19.55 us versus CuTe 17.51 us. The unchanged
attention path costs unpack ~4.21 us, Q norm/RoPE ~4.08 us, K norm/RoPE ~2.27 us,
and SIMT attention/gate ~16.90 us measured individually. Earlier source profiles
of the identical attention kernel showed repeated K/V global loads, no HMMA,
772096 dynamic LDG.E.U16 and 46.6% long-scoreboard sampling.

Preserve all projection functions/configurations and residual. Do not change
compiler/lowering, the independent reference, or acceptance thresholds. All
runtime numerical work remains AveLang. Generic supplied DSL examples and
primitive documentation may be consulted; do not copy a sibling workload answer.

## Requested implementation

Read Q/gate/V directly from the packed QKV output; let the separate K norm/RoPE
kernel read its packed slice directly. Fuse Q norm/RoPE, attention and gate,
and reuse shared K/V across queries in one CTA. This should remove unpack and
the separate Q norm launch, reducing seven stages to five.

Start with one bounded fused/shared implementation. After actual feedback,
consider at most one Tensor Core QK or justified tiling follow-up. The native
DSL supports mma_16x8x16_bf16_f32 and ldmatrix; BF16 QK with FP32 accumulation
is allowed. **Softmax probabilities and PV accumulation must remain FP32**.
Do not add CuTe's BF16 probability rounding point. Preserve BF16 normalization
before RoPE, post-RoPE rounding, attention output rounding, sigmoid rounding,
and final gate/output rounding from the original TASK.

The agent provides `round4_candidate.py`, `PROFILE_STAGE_TAGS`,
`ATTENTION_CONFIGS` / `DEFAULT_ATTENTION_CONFIG`, and
`bind_middle(packed, qnorm_weight, knorm_weight, tables, attended, config=None)`
returning ordered zero-argument launches. `build` retains its existing public
signature/output tuple. CPU compilation uses the isolated
`.repro/attn-fusion-native-round4-results` directory. No agent GPU submissions.

## Independent acceptance

The coordinator extracts the packed-to-attended portion of the unchanged
reference AST for isolated checks. Verify changed-data CUDA graph replay,
causal prefix/batch isolation, structured KV-group/interleaved-gate indexing,
zero data, tail queries and leading/trailing output guards. Then run the
original full-layer reference tests unchanged and compare round3/new/CuTe in
one system reservation: 100 calls per graph, 20 randomized timing samples.
Measure the middle-stage bundle directly as well as the full chain; do not
replace its time with the sum of independently measured stages.

Only a measured improvement that passes the final memcheck/synccheck/racecheck
can become the default. Keep rejected drafts and exact source identities;
report static invariant validation as unavailable in this native branch.

Artifacts: `profile/attn-native-round4-h100-20261102/`.
