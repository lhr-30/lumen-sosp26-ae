# Native Lumen harness: attention mixer generated and optimized from the reference

**All three original-harness rounds completed and passed formal evaluation.**
The starting file was the unchanged PyTorch reference, with no previous mixer
DSL supplied. The harness's formal measurements were **78.99 → 69.44 → 56.04 μs**
for generation, optimization 1, and optimization 2, respectively. These were
separate measurement batches, so they are not a controlled speedup estimate.

The final source also passed independent fresh-data graph tests, allocation
guards, memcheck, synccheck and racecheck. Its exact SHA256 is
`d7f4652bb9be417adb9a58dc481f30458a2467bf6438938cab41d1e464325cbc`.
The previous manual implementation remains unchanged.

**A precise performance ranking is unresolved:** both independent matched runs
observed a resident Triton Python service using the GPU during our system-queue
reservation. The first run put the final harness source at 55.98 μs and the manual
source at 54.29 μs; the repeat put them at 76.19 and 77.77 μs. These results are
retained as evidence of interference, not used to claim one implementation wins.

## What used the original harness, and what was adapted

The launcher [run.py](../../scripts/attn_fusion_harness/run.py) calls the unchanged
`run_optimization_sequence` from
[optimization_runtime.py](../../python/lumen/harness/datasets/lumen/optimization_runtime.py),
which invokes the unchanged [CodexRunner](../../python/lumen/harness/backend/codex/runner.py).
The native runtime creates round workspaces, invokes Codex, saves trajectories,
performs formal evaluation and passes each successful source to the next round.
Its source hashes are pinned in the [support provenance](../../runs/attn_mixer_h100_harness_20261102_01_support/provenance.json).
Neither runtime file was modified.

The built-in Table 2 attention spec is AMD FlashAttention at sequence lengths
1024–16384. This experiment instead supplies a custom `OptimizationSpec`, a
Docker/system-queue evaluator and task-specific prompts for the complete
**H100 NVL, B16/T26/H8/KV2/D256, hidden2048 attention mixer**. It is a single
layer, not whole-model end-to-end latency. It does not replay the original AMD
prompt sequence or claim to implement an unpublished ICRL planner.

Three prompts were declared before launch: generation from reference, measured
optimization, then another measured optimization. They describe semantics and
available tools, without prescribing previous tiling/fusion solutions. Generic
examples and API sources from supplied AveLang commit
`5f349c594ad510ad6b90d0e1ca310c0d47aa9ed1` were provided in a clean support tree.
The model used the existing local configuration: **gpt-6.1-sol, max reasoning,
copilot provider, Codex CLI 0.166.0**. Runtime: `lumen-hopper:dev`, CUDA 13.2,
Torch 2.12.0a0+5aff3928d8.nv26.05, LLVM 23, sm_90a; GPU has 132 SMs.

The coordinator did not edit generated DSL or feed manual optimization advice
into the sequence. One infrastructure-only revision enabled faulthandler and
printed the specialization name after the first compiler crash; this is logged
in [infrastructure_revisions.json](../../runs/attn_mixer_h100_harness_20261102_01_support/infrastructure_revisions.json).
The reference and thresholds were unchanged. The support manifest and the
revision hashes were checked. Trace tool-call scans found no references to the
previous mixer solution paths; this is an audit of supplied inputs and recorded
calls, not a claim of OS-enforced filesystem isolation.

## Completed rounds and attempts

| Harness round | Purpose | Formal median μs | Selected source SHA prefix |
|---|---|---:|---|
| round0 | Generate from reference | 78.987 | 1593ca10473f |
| round1 | First optimization | 69.439 | be62b3a93392 |
| round2 | Further optimization | 56.038 | d7f4652bb9be |

The round1 input is byte-identical to the round0 output; the round2 input is
byte-identical to the round1 output. The seed SHA is
`f78ff01334a91c75bf4ab2401ba0d4fa78d055d8bdddc5074b11b86e2ad29fb4`, matching
the authoritative original reference. [Round metadata](analysis/rounds.json),
[all attempts](analysis/attempts.json) and [frozen sources](harness/source_manifest.json)
record the exact transitions.

There were 21 CPU compile invocations and 25 GPU evaluation invocations, including
baselines and formal checks. Three GPU prototype evaluations failed. The
[original run](../../runs/attn_mixer_h100_harness_20261102_01/) retains every
round's prompt, input, output, trace, statuses and evaluation snapshots.

- Generation produced four kernels: QKV, combined Q/K norm+RoPE, attention+gate,
  and output projection+residual. It repaired compiler crashes, FP32 literal
  typing errors and an unresolved generic `__nv_fmaf` dependency.
- Optimization 1 rejected wider/three-stage linears at 81.66/82.23 μs. Its TMA
  attention prototype first failed launch, then two versions failed cosine at
  0.8018. Explicit QK chunk synchronization repaired correctness. Scalar
  attention alternatives passed but cost 92.69/79.11 μs. Larger projection
  K steps and a shared two-consumer QKV tile eventually reached 71.87 μs in the
  agent's test; the native formal check later measured 69.44 μs.
- Optimization 2 tested a 128×160 QKV tile with K128 stages, reaching 65.02 μs.
  A three-stage overlap variant was slower at 65.67 μs and was discarded.
  Paired attention loads with 16, 32 and finally 26 query-processing warps reached
  58.76,58.40 and55.55 μs in successive tests. All these passing candidates are
  retained. A longer 100-iteration comparison had broad timing spread; the agent
  recorded it and rechecked at standard settings, obtaining 68.65→54.36 μs for
  its input and selected source. The final formal check measured 56.04 μs.

The many internal attempts are not additional top-level harness rounds. The
runtime completed exactly three rounds: one generation and two optimizations.

## Final generated implementation

| Kernel | Grid CTAs | Threads/CTA | Registers/thread | Dynamic shared bytes |
|---|---:|---:|---:|---:|
| QKV | 128 | 288 | 108 | 147712 |
| Q/K norm+RoPE | 1040 | 128 | 47 | 0 |
| Attention+gate | 128 | 832 | 58 | 86272 |
| Output projection+residual | 224 | 160 | 58 | 65792 |

The QKV kernel uses two consumer warp-groups, a 128-token×160-channel tile,
two K128 TMA stages and BF16 m64n160 WGMMA with FP32 accumulators. Paired BF16
stores write 32 bits at a time. Each consumer releases a stage after a full
WGMMA wait; the empty barrier waits for both consumers.

Attention has one CTA per batch/head and **one warp per query**, 26 warps for
T=26. It stages Q/K/V with TMA, computes QK with WGMMA, then performs stable
FP32 softmax and FP32 PV. Adjacent BF16 V/gate values are loaded in pairs and
output values are stored in pairs. All 832 threads reach the shared-score
barrier; only the first 128 perform QK. Original BF16 norm/RoPE, attention,
sigmoid/gate, projection and residual rounding points are retained. There is
no BF16 probability rounding or PyTorch numerical fallback.

The generated FP32 path uses native `fast_fma`/fast-math intrinsics, including
FTZ; the previous manual round4 PV uses non-FTZ FMA. This experiment preserves
the original numerical acceptance tests and FP32 probability type, not bitwise
or full IEEE equivalence to that implementation. CuTe additionally uses BF16
probabilities for Tensor Core PV, so it has a different intermediate precision
contract.

## Independent checks

[check_final.py](harness/check_final.py) runs the unchanged full-layer validator
with leading 512-element and trailing 4096-element guards around all five
candidate allocations. The aligned guarded views also check nonzero storage
offsets. Seed 0, updated-input/weight/table graph replays at 17/123, causal/batch
isolation and zeros passed. Both outputs are finite and meet unchanged thresholds:
cosine≥0.9999 and relative L2≤0.02. All guards remain unchanged.

- memcheck: 0 errors.
- synccheck: 0 errors.
- racecheck: 0 hazards, 0 errors, 0 warnings.
- All four final specialization compiles/ptxas checks passed, with zero spills
  and no PTX local memory. [Compile output](analysis/compile_final.json) and
  [generated PTX](harness/generated/) are preserved.
- Native artifact-invariant validation is **unavailable** in this branch.
  It was not substituted with stubs or counted as a passing check.

These are finite workload checks and source review, not a proof of all possible
inputs. No lowering/compiler changes were made for this experiment.

## Independent timing and observed interference

Both independent runs used the same inputs for all implementations, 100 calls
per graph and 20 randomized rounds. Allocation, JIT compilation and Python
submission are outside the GPU timings. The middle bundle was directly captured
where interfaces allowed it. Individual stage sums need not equal chain time.

| Implementation | First matched median μs | p10–p90 μs | Repeat median μs |
|---|---:|---:|---:|
| Harness round0 | 80.056 | 78.785–81.164 | 104.785 |
| Harness round1 | 70.903 | 69.698–74.228 | 94.631 |
| Harness round2 | 55.982 | 54.875–58.498 | 76.190 |
| Prior manual round4 | 54.289 | 53.368–57.175 | 77.771 |
| Original CuTe | 35.041 | 34.624–36.684 | 44.095 |

**Both tables are affected by background GPU activity.** The first measurement
window was 08:49:22.707–08:49:24.026 UTC. `nvidia-smi pmon` recorded resident
`triton_python_backend_stub` PID 3953810 at 7% SM activity at 08:49:22 and 1% at
08:49:23. The repeated window was 08:52:22.162–08:52:24.199, with that process at
54%, 49% and 43%; preceding samples reached 67%. It was present before our CUDA
process started, and it was active while our queue reservation was held.
The service was not stopped or modified.

The 1-second process sampling is coarse; unavailable/dash metrics are not treated
as proof of zero activity. Nevertheless, positive activity overlapping both
windows is direct evidence that system-queue ownership did not provide exclusive
hardware use. The repeat's broad distributions and reversed small ranking make
it unsuitable for a precise comparison. No additional GPU tests were submitted
once persistent interference was confirmed. Coordinate a quiet interval with the
service owner before using these timings for a precise speedup claim.

The first run's middle bundle was 10.58 μs for the final harness source versus
16.87 μs for the manual source, while its QKV/O projection path was slower.
Those observations identify useful code differences to investigate later; they
are not an isolated causal profile. No new Nsight Compute collection was made
in this harness experiment.

Evidence: [first timing](analysis/timing.json), [first background activity](analysis/background_activity.json),
[repeat timing](../attn-harness-timing-recheck-h100-20261102/analysis/timing.json),
[repeat activity](../attn-harness-timing-recheck-h100-20261102/analysis/background_activity.json).

## Queue provenance and reproduction

All GPU evaluation commands, including those issued by the generation agents,
used the shared system FIFO. The harness used 23 reservations; expected prototype
failures exited 1 and were rejected. The full [queue ledger](analysis/gpu_queue.json)
records their IDs. Independent guarded/sanitizer/timing job 3773 and timing-repeat
job 3774 both exited 0. CPU compilation ran without a GPU device. No unrelated
processes were killed and the earlier accepted DSL was not replaced.

Start a new native-harness experiment:

```bash
python3 scripts/attn_fusion_harness/run.py
```

Run identifiers are fresh by default. The launcher supports `--run-dir`,
`--rounds` and `--timeout-seconds`; existing experiment directories are refused.
To repeat independent checks, first coordinate a quiet GPU interval, then copy
only the frozen harness into a new directory to preserve both existing runs:

```bash
attn_repeat="profile/attn-harness-repeat-$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$attn_repeat"/harness "$attn_repeat"/analysis "$attn_repeat"/reports "$attn_repeat"/logs
cp -a profile/attn-harness-timing-recheck-h100-20261102/harness/. "$attn_repeat/harness/"
"$attn_repeat/harness/run.sh" final
```

The wrapper automatically uses the system queue and records process activity.
The default previously optimized implementation remains
[scripts/attn_fusion_native/native_candidate.py](../../scripts/attn_fusion_native/native_candidate.py).
The independent final harness candidate is
[round2/output_model_new.py](../../runs/attn_mixer_h100_harness_20261102_01/round2/output_model_new.py).
