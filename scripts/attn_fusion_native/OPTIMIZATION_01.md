# Native Hopper round 1: overlap WGMMA groups

Independent round-0 tests passed on H100 NVL (queue job 3551), including all
random, changed-data graph, causal/batch isolation and zero-input cases.
The source is preserved at `.repro/attn-fusion-native-round0.py`.

CUDA-graph median times, 10 calls/graph and 10 samples, in microseconds:

| Stage | Round 0 |
|---|---:|
| QKV projection | 37.922 |
| Unpack | 4.578 |
| Q norm/RoPE | 4.403 |
| K norm/RoPE | 2.568 |
| Causal GQA/gate | 17.555 |
| Output projection | 15.998 |
| Residual | 3.898 |
| Chain | 87.595 |

The lowering agent proposed keeping the current geometry, three TMA stages and
math, while permitting one WGMMA group to remain outstanding. Try that single
change. Shared-storage ownership must follow completion of the group that reads
it; the last group must complete before reading final accumulators or exiting.
Preserve all other kernels and the reference's rounding points.

Inspect generated PTX for unexpected local-memory spills or register moves
around still-pending accumulators. Then return the candidate for independent
numerical/graph tests, timing and sanitizer checks. If it fails or does not
improve latency, keep the verified round-0 source. This branch does not provide
artifact invariant validation; do not report the schedule as formally proved.
