# Agent feedback: fixed-shape GEMM scheduling

User request: continue optimizing the AveLang DSL after accepting the TMA
output epilogue. The agent writes the DSL; the coordinator owns independent
GPU validation and timing, through the system GPU queue.

## Starting point and evidence

The accepted round-2 source is frozen as the baseline in
`profile/attn-native-round3-h100-20261102/harness/sources/ave/baseline_candidate.py`.
SHA256: `b3197d3bf1880691826e59b6d99642a0fc072ac28df23bdca66b9d8ce394f1f7`.
The H100 B16/T26 single-layer chain measured 77.035 us; QKV 30.050 us,
OProj 12.532 us. Current GEMM tiles are 64 token rows x 128 output channels,
with one 128-thread consumer warpgroup and a 32-thread TMA producer.

The preceding full/source NCU reports show:

- QKV 280 CTAs, Tensor active 42.50%, achieved occupancy 16.56%.
- OProj 112 CTAs on 132 SMs, Tensor active 34.60%, occupancy 7.81%.
- QKV source samples: 45.37% long-scoreboard, 31.55% barrier; OProj 47.09%
  barrier. Major hotspots remain input-stage polling and WGMMA group waits.
- Shared-memory use and spilling did not regress with the epilogue change.
- Merely allowing one pending WGMMA group in round 1 did not improve timing.

## Requested experiment

Keep BF16 rounding, FP32 accumulation and the accepted matrix/TMA output path.
Do not change the compiler or attention math. Consider swapping GEMM operands
so the short token dimension occupies the WGMMA N dimension, using multiple
consumer warpgroups where appropriate. Select QKV and OProj geometry separately.

The existing DSL exposes WGMMA N64/96/128/160/176/192 (not N104/N144), and has
transposed stmatrix examples. A plausible bounded experiment is QKV logical
160 tokens x 128 channels (two consumer warpgroups, 120 CTAs) plus OProj
128x64 (128 CTAs) or 96x64 (160 CTAs). These are hypotheses to measure, not
assumed wins. The agent may revise the design based on source/compile evidence.

Write `round3_candidate.py`; keep the accepted default unchanged while testing.
Provide `bind_projection(A, W, C, N, ...)` for isolated validation without
coordinator assumptions about launch geometry. Optional configurations must be
explicit and few. Preserve every shared-memory lifetime, producer/consumer
count, phase transition and final async drain. Handle all valid output rows and
preserve output guards. No raw PTX/CUDA or external numerical fallback.

## Acceptance and provenance

Offline compile/assembly first, then independent full reference/CUDA-graph
checks and projection/tail/guard tests. Record bitwise equality to round 2 where
it holds, while retaining the original independent numerical thresholds.
Measure the selected candidate, round 2 and CuTe together with 100 calls per
graph and 20 randomized samples. Run all three sanitizers before promotion.
Reject regressions, preserve source identities and report unsuccessful trials.

Artifacts belong to `profile/attn-native-round3-h100-20261102/` and
`.repro/attn-fusion-native-round3-results/`. Existing reports are historical and
must not be overwritten.
