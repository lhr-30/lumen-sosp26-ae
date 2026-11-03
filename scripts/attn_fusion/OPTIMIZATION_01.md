# First measured-feedback round

The passing round-0 candidate is preserved at
`.repro/attn-fusion-candidate-round0.py`; numerical results and timing are in
`.repro/attn-fusion-results/round0_{test,bench}.json`.

On H100 NVL, 10 calls per CUDA graph and 10 interleaved samples gave these
median latencies (microseconds):

| Stage | Round 0 |
|---|---:|
| QKV projection | 551.418 |
| Unpack | 4.354 |
| Q norm/RoPE | 4.741 |
| K norm/RoPE | 2.811 |
| Causal GQA/gate | 24.779 |
| Output projection | 302.309 |
| Residual | 4.056 |
| Complete chain | 889.677 |

Optimize the dominant projections using the existing compiler and intrinsic
interfaces. Choose the layout and schedule yourself. Your proposal to reuse A
staging across several independent WGMMA output-column tiles is appropriate to
try. Preserve the algorithm, FP32 accumulation and BF16 output rounding, tail
masking, uniform 128-thread WGMMA participation, and shared-memory publication /
reuse ordering. Do not change other stages or their invariant checks.

Compile with invariant validation enabled and assemble before returning the
candidate for independent numerical tests, CUDA-graph timing and sanitizer
checks. WGMMA operand/result mathematical tags remain unsupported; do not claim
this round adds such a proof. Preserve the passing baseline and record feedback.
Accept the new candidate only if it passes validation and measures faster.
