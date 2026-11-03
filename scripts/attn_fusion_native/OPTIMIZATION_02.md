# Agent feedback: GEMM output epilogue

User request: improve the generated DSL after comparing its generated code and
profiles with CuTe. Keep DSL authorship with the lowering agent. The coordinator
owns independent tests, timing and profiling; do not modify the compiler.

## Evidence supplied to the agent

The original QKV and OProj epilogues directly assign individual BF16 elements
from the WGMMA register fragment into global memory. On H100, the observed
`STG.E.U16` stores use only 8 of 32 bytes per sector per request. CuTe stages its
output in shared memory and emits TMA stores. The native AveLang branch already
exposes `stmatrix`, BF16x2 packing, proxy fences, named barriers and `tma_store`,
with worked examples in `python/avelang_kernels/nvidia_gemm.py`.

## Requested experiment

Write `round2_candidate.py`, preserving the fixed M416, K2048, N5120/N2048
workload, BF16 rounding and all non-projection kernels. Change the GEMM output
path to packed BF16, a shared-memory matrix layout and TMA stores. Keep tile
shape and the main loop unchanged to isolate this optimization. Prefer safely
reusing a finished input stage instead of increasing shared-memory demand.
Explicitly protect the input/output arena lifetime, async proxy ordering and
the final 32 valid rows. No raw PTX/CUDA or production lowering changes.

## Acceptance

- Offline compilation and assembly retain WGMMA and emit real matrix/TMA stores.
- Both GEMMs agree bitwise with the original on the comparison input, including
  all 32 tail rows; a guard after the logical output remains unchanged.
- Independent numerical and fresh-input CUDA-graph replay tests pass without
  relaxed thresholds; memory/synchronization/race sanitizers pass.
- Benchmark old/new/CuTe in one system GPU reservation, with randomized order,
  100 calls per graph and 20 timing samples. Report stages and complete chain.
- Promote only a validated improvement. Preserve frozen baseline sources and
  source hashes; report limitations and unsuccessful attempts honestly.

Artifacts: `profile/attn-native-epilogue-h100-20261102/` and
`.repro/attn-fusion-native-epilogue-results/`.
