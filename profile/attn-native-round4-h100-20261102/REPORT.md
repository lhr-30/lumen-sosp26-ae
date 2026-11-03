# Round 4: fused attention with MMA QK and asynchronous K/V staging

The selected AveLang DSL reduces **single-layer GPU latency from 66.213 to
55.404 μs (16.3%)** on H100 NVL at **B16/T26/H8/KV2/D256, hidden=2048**.
The directly captured middle bundle improves from 27.739 to 16.982 μs (38.8%).
The agent-authored source is now the default `native_candidate.py`; the
compiler/lowering, GEMM configuration and residual arithmetic are unchanged.
This measures one attention mixer, not the entire model's end-to-end latency.

## Measurement and source identity

- GPU: NVIDIA H100 NVL, 132 SM; BF16 inputs/output, FP32 QK accumulator,
  FP32 softmax probabilities and FP32 PV accumulation.
- Compiler: supplied AveLang `origin/nvidia-ave-kernels`, commit
  `5f349c594ad510ad6b90d0e1ca310c0d47aa9ed1`, LLVM23, sm_90a.
- Runtime: CUDA 13.2, Torch 2.12.0a0+5aff3928d8.nv26.05, `lumen-hopper:dev`.
- Baseline is round3, SHA256
  `c25bacf5405cec557f226d442e7a6cd4c4fd5bf518fb2a48650ee2e26712fd71`.
- Selected `round4_async_candidate.py` and default `native_candidate.py` are
  byte-identical, SHA256
  `f0bb46ed3222fd3a9ac32ff9f01f6ac6f250e9138a5ded70ae7cf3d324546cf2`.
- [Frozen manifest](harness/source_manifest.json) pins all 12 reference and
  candidate files. [AST checks](analysis/source_integrity.json) confirm unchanged
  projection functions/configs/binding, gate helper and residual.

Normal timing uses the same inputs for all five implementations, 100 calls per
CUDA graph, 20 randomized rounds, CUDA events, with allocation/compilation and
Python submission outside GPU timing. [Raw timing](analysis/timing_async.json)
contains all samples. Stage sums need not equal a captured chain's duration.

| Implementation | Chain median μs | p10–p90 μs | Middle bundle μs |
|---|---:|---:|---:|
| Round3 | 66.213 | 63.974–67.417 | 27.739 |
| Shared K/V SIMT fusion | 61.812 | 59.828–62.917 | 23.684 |
| MMA QK, scalar K/V staging | 67.855 | 66.136–70.273 | 28.129 |
| **MMA QK, 16B async K/V staging** | **55.404** | **53.582–55.998** | **16.982** |
| Original CuTe | 36.270 | 34.750–36.594 | 7.799 |

Selected individual stages: QKV 20.168, K norm 2.212, fused Q norm/RoPE +
attention/gate 14.869, OProj 12.343, residual 3.505 μs. CuTe fused attention is
5.971 μs. **CuTe rounds softmax probabilities to BF16 for Tensor Core PV;
this AveLang implementation retains FP32 probabilities/PV.** The 1.528×
remaining chain latency ratio is a workload comparison with that distinction.

## Implementation and rejected experiment

Packed Q/gate/V are read directly, and K norm/RoPE reads packed K. Fusing Q
norm/RoPE, attention and gate removes unpack and the separate Q norm launch:
**seven launches become five**. The chosen kernel has 512 CTAs, 128 threads/CTA,
eight logical queries/CTA, a padded 16×32 score tile and BF16 m16n8k16 QK MMA.
The projection kernels continue using their existing WGMMA/TMA pipeline.

Changing QK to MMA alone regressed. The [diagnostic run](../attn-native-round4-diagnostic-h100-20261102/REPORT.md)
found the first scalar `STS.U16` after a K/V global load accounted for 539 samples,
523 long-scoreboard. Each thread repeated 64 scalar load/store iterations.
The selected correction uses eight 16B copies per tensor/thread with
`cp.async.ca`, overlaps them with Q normalization, then executes
`wait_group(0)` and a CTA barrier. Invalid whole rows are explicitly zeroed;
no invalid global copies are issued. The q8 geometry and mathematical suffix
remain unchanged from the raw MMA experiment. Their full-chain outputs are
bitwise identical in the matched comparison.

## Final NCU evidence

Nsight Compute 2026.2.1, `--set full` plus PmSampling/PmSampling_WarpStates and
`--set source --section SourceCounters`, `--clock-control base`,
`--cache-control none`. The standalone frozen Python harness uses a PTX
line-info shim without altering instruction text. Source locations below are
**generated PTX lines**, not original DSL lines. Reports are parsed with
`ncu_report`, including source stalls, actual SASS, rule results and PM traces.

Baseline profiles cover unpack/Q norm/K norm/attention; selected profiles cover
K norm/fused attention. NCU replay durations are separate from normal graph
measurements and must not be substituted into the timing table above.

| NCU metric | Round3 attention/gate | Raw MMA fusion¹ | Selected async fusion |
|---|---:|---:|---:|
| Replay duration μs | 25.824 | 38.016 | 23.936 |
| CTAs / threads per CTA | 3328 /32 | 512 /128 | 512 /128 |
| Registers/thread | 43 | 55 | 56 |
| Shared bytes including system overhead | 1024 | 44544 | 44544 |
| Achieved occupancy | 25.36% | 21.91% | 21.19% |
| Long-scoreboard sample share | 50.10% | 43.77% | 43.18% |
| Executed warp SASS instructions | 4,739,072 | 7,618,304 | 4,269,184 |
| `LDG.E.U16` warp instructions | 772,096 | 348,672 | 86,528 |
| `LDGSTS.E.128` warp instructions | 0 | 0 | 18,944 |
| `HMMA.16816.F32.BF16` warp instructions | 0 | 32,768 | 32,768 |

¹ Raw MMA data is from the separate diagnostic run, with the same profiling
settings. The final normal timing compares all five implementations in one job.
Baseline attention excludes Q normalization/RoPE; both fused versions include it.

The preload correction removes the dominant scalar staging hotspot and cuts
raw-MMA instruction count by 44.0%. The whole-kernel long-scoreboard **percentage
stays similar**: the shorter kernel still waits on other loads. Its largest
remaining sample is a gate comparison (`HSETP2`, PTX 1285) immediately after a
BF16 gate load: 64 samples, 62 long-scoreboard. Q conversion at PTX 187 contributes
38 long-scoreboard samples. This supports changed hotspots, not a claim that
all memory latency is eliminated. See [source stalls](analysis/stalls_candidate_attention.json)
and [actual SASS](analysis/candidate_attention.sass).

Occupancy is limited by shared memory: 43520 user static bytes plus 1024 system
bytes allow five CTAs/SM, 31.25% theoretical occupancy. The grid occupies only
0.776 theoretical waves. Active SM cycles span 10448–21946 (mean 17829), and
causal queries have different PV loop lengths. Each warp processes two queries;
only 26 of 64 physical query rows per sequence/head are useful (40.6%). These are
remaining scheduling/padding costs, not evidence that MMA itself is slower.

[PM timeline](analysis/pm_utilization.png) shows an early tensor phase and a long
non-tensor tail: selected SM activity falls below full utilization around
11.5 μs and reaches about 12% by 20.5 μs of the sampled replay. Tensor activity
peaks near 12.7% in one sample; its whole-kernel metric is 1.52%. The baseline has
no tensor instructions despite a small 0.09% tensor activity counter, so that
counter is not evidence of baseline MMA execution. PM sampling is 1.5 μs and
cannot establish exact phase boundaries.

Warmed NCU runs report zero DRAM read bytes and selected L2 hit ratio 96.22%;
these runs do not support an HBM-bandwidth bottleneck. Selected global stores
use 32 bytes/sector, and spilling counters are zero. However, the selected kernel has **207755 shared-store bank conflicts**.
Source counters attribute most excessive shared wavefronts to explicit zero-fill:
 `STS.U16 [...], RZ` at PTX 82/90/96/101, each with 20736 excessive
wavefronts; see [shared access attribution](analysis/shared_excessive_candidate_attention.json).
Total excessive shared wavefronts are 197632. This is a new remaining cost,
even though the async implementation wins overall.

NCU rule estimates include shared-store conflicts 30.7%, uncoalesced shared
access 14.5%, achieved occupancy 32.2% and SM imbalance 13.6%. These are diagnostic
estimates for individual symptoms, not additive or measured speedups.

## Remaining opportunities, ranked by concrete evidence

1. Reassign or vectorize invalid-row zero stores while retaining the same
   valid 16B copies. Source counters directly identify the zero-fill bank
   conflicts; this is the most localized follow-up. A new measured experiment
   is needed to quantify benefit, despite the NCU 30.7% rule estimate.
2. Preload or vectorize gate/Q accesses to reduce the remaining scalar-load
   dependencies. Gate comparisons now lead source stalls. This must preserve
   the original BF16 sigmoid/gate rounding and be checked against register cost.
3. Reduce padded query work or distribute FP32 PV across more independent
   warps. Shared residency, 40.6% useful query rows and the sampled tail motivate
   a geometry study. No latency gain is claimed without testing. BF16 probability
   rounding is outside this round's precision contract.

## Correctness and limitations

- Original full-layer reference and thresholds unchanged: finite output,
  cosine≥0.9999, relative L2≤0.02. Seeds 0/17/123, graph replays with new inputs,
  weights and RoPE tables, causal/batch isolation and zero inputs pass.
- Six [independent middle checks](analysis/middle_async.json) use an AST slice
  of the unchanged reference, including structured GQA/gate indexing, final
  query, fresh graph data, and leading/trailing output guards. Maximum absolute
  error is 0.00390625; worst relative L2 is 6.093e-5.
- Every original BF16 boundary is preserved. FP32 P/PV are maintained; FP32
  reduction order changes mean the candidate is not bitwise equal to round3.
- Selected-source full-layer smoke memcheck/synccheck: 0 errors;
  racecheck: 0 hazards, 0 errors, 0 warnings. These are finite test cases, not a proof.
- Ten default offline compilation/ptxas cases pass with no spills/local memory.
  The selected kernel uses 56 registers and 43520 **static** shared bytes;
  `dynamic_shared_bytes: 0` does not mean zero shared memory. Default PTX files
  are byte-identical to frozen-candidate compilation outputs. Checks cover
  MMA/ldmatrix, 16B cp.async with full wait, and non-FTZ FP32 PV FMA.
- Native branch has no artifact-invariant module: invariant validation remains
  **unavailable**. No stubs or ignored validation flags are counted as validation.
- Two baseline derived L2 hit ratios exceed 100% in uncontrolled-cache replay;
  [warnings](analysis/measurement_warnings.json) preserve them, excluded from
  interpretation. Source sample shares are not wall-time shares.

## Queue provenance and reproduction

All GPU work used `gpu-queue-system`: 3717 (shared checks), 3719 (MMA checks),
3722 (diagnostic NCU), 3727 (final correctness and five-way timing), 3729 (final
sanitizers and NCU). All exited 0. The DSL agent performed CPU work only;
the primary agent ran independent GPU validation. The wrappers reserve until
all children exit and honor an existing `GPU_QUEUE_ACTIVE` reservation.

From the repository root, repeat into new files:

```bash
NATIVE_RESULTS="$PWD/.repro/attn-fusion-native-round4-repeat" \
  scripts/attn_fusion_native/run.sh cpu
profile/attn-native-round4-h100-20261102/harness/run.sh middle \
  --output /profile/analysis/middle_repeat.json
profile/attn-native-round4-h100-20261102/harness/run.sh timing \
  --extra-candidate round4_shared_candidate --extra-mma-module round4_mma_candidate \
  --output /profile/analysis/timing_repeat.json
scripts/attn_fusion_native/run.sh test
scripts/attn_fusion_native/run.sh memcheck
scripts/attn_fusion_native/run.sh synccheck
scripts/attn_fusion_native/run.sh racecheck
```

For a fresh NCU collection, first copy the frozen harness into a **new run
directory** with `reports/`, `logs/`, `analysis/`, preserving this run's evidence.
The harness commands are `run.sh full baseline --stage middle`,
`run.sh source baseline --stage middle`, and the two corresponding `candidate`
commands. Each uses the system queue automatically. CPU-only SASS exports use
`ncu --import reports/source_<impl>.ncu-rep --page source --print-source sass --csv`;
then run `analysis/analyze.py`. All report files and parsed metrics remain in
this directory. Main validation is [validation.md](../../scripts/attn_fusion_native/validation.md).
