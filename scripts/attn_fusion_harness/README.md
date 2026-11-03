# Run the native Lumen optimization harness on the H100 attention mixer

Completed run: [attn_mixer_h100_harness_20261102_01](../../runs/attn_mixer_h100_harness_20261102_01/).
All three formal rounds passed. The harness's separate formal evaluations
reported 78.99 → 69.44 → 56.04 μs; these are different measurement batches.
The final generated source is [round2/output_model_new.py](../../runs/attn_mixer_h100_harness_20261102_01/round2/output_model_new.py),
SHA256 `d7f4652bb9be417adb9a58dc481f30458a2467bf6438938cab41d1e464325cbc`.
Independent fresh-data tests, five allocation guards and all three sanitizers
passed. Both independent matched timing runs observed another Triton process
using the GPU during our system reservation, so they cannot establish a precise
ranking against the prior manual implementation. See the [full report](../../profile/attn-harness-h100-20261102/REPORT.md).

This experiment calls the repository's **unchanged**
`lumen.harness.datasets.lumen.optimization_runtime.run_optimization_sequence`
and `CodexRunner`. The runtime prepares isolated rounds, invokes `codex exec`,
copies generated code to the next round, runs formal correctness evaluation,
and preserves trajectories and statuses. The built-in Table 2 attention spec
targets AMD FlashAttention with sequence lengths 1024–16384; this adapter adds
the existing B16/T26 complete mixer workload and H100 execution environment.

The first input is the unchanged independent PyTorch reference, with no earlier
generated mixer DSL. Three prompts are declared before launching: generation,
measured optimization, and another measured optimization. They describe semantics
and tools without prescribing tiles, fusion choices or earlier profiling findings.
Public generic examples from the supplied native AveLang branch are available.
The original Table 2 AMD optimization prompts are **not** used for this new task;
this is a workload adaptation of its execution harness, not a claim to replay
the paper's original attention experiment or an unpublished ICRL planner.

```bash
python3 scripts/attn_fusion_harness/run.py
```

The default is three rounds with a 1800-second Codex timeout per round, using
the local Codex model/provider configuration. `--rounds`, `--timeout-seconds`,
and `--run-dir` allow explicit overrides. Existing runs are never overwritten.
Do not replace a completed run's evidence with a retry; choose a new run directory.

Each run has a sibling `_support` directory containing only the task's frozen
reference/evaluator, native language references, prompts and environment adapter.
`provenance.json` records runtime hashes and the absence of prior mixer candidates.
The runtime's scripts/skills allowlist points at that clean directory, keeping
the earlier manual mixer solutions outside the generation agent's allowed inputs.

The benchmark adapter snapshots each tested source under `round*/evaluations/`.
GPU correctness and timing run through the **system-wide GPU queue**, including
agent-initiated tests and the runtime's final formal test. CPU-only compilation
uses Docker without a GPU device and checks every declared specialization with
the native compiler and ptxas. GPU invocations have a finite timeout and remove
their container before the queue reservation is released.

The reference and acceptance thresholds match the previous experiment. Softmax
probabilities and PV accumulation remain FP32; original BF16 rounding points are
preserved. Native invariant-module validation is unavailable and is never
reported as passed. The adapter emits standard domain/workload/correctness JSONL
records consumed by Lumen's existing `evaluate_candidate` function.

The primary agent independently reviews final source, compares frozen round
outputs with the manual baseline/CuTe, and checks the final candidate. It does
not edit the generated DSL or inject optimization advice during the sequence.
Formal passing rounds are not automatically promoted to the previous experiment's
default: final correctness and a measured speedup are needed for that.
