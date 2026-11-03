# Supplied Hopper backend experiment

Use the user-supplied `avelang/` repository, branch `origin/nvidia-ave-kernels`,
commit `5f349c594ad510ad6b90d0e1ca310c0d47aa9ed1`, through the isolated checkout
`.repro/avelang-native`. The user requested continuing the previous experiment
using this backend. The original checkout and previous accepted candidate stay
unchanged.

The full semantic description, shapes and BF16 rounding contract remain those
in [the original task](../attn_fusion/TASK.md), with the authoritative unchanged
[reference](../attn_fusion/input_model.py). This is B16,T26,H8,KV2,D256,hidden2048,
one attention mixer, not full-model e2e. Preserve `build(x, hidden, w, tables)`
returning ordered zero-argument launch callables and `(attended, output)`.
All runtime math must be AveLang; host allocation/view and native tensor-map ABI
setup are allowed. No torch numerical fallback or other kernel language.

The lowering agent owns `native_candidate.py`, `compile_candidate.py` and
`AGENT_NOTES.md` here. The coordinator owns the compiler build adaptation,
environment wrapper, unchanged reference and independent evaluation. The agent
may read and adapt the supplied branch's generic `nvidia_gemm.py` and
`nvidia_attention.py` DSL examples; document reused patterns. Do not inspect
other `model-tunner/attn_fusion/ave*` solutions or claim they were newly generated.

This backend supplies TMA descriptors/load/store, multiple WGMMA shapes,
explicit commit/wait, mbarrier, warp scheduling and native fast math. Design a
pipeline using the actual available interfaces instead of retaining the old
blocking m64n8 constraint. Respect M416 tails and BF16 output. Keep a passing
version and accept an optimization only after independent validation and timing.

Unlike the earlier artifact compiler, this branch has no `invariant` module or
`validate_invariants` option. This overrides the previous task's requirement to
use that API: do not fabricate stubs, pass silently ignored flags, or report
invariants as verified. Record invariant checking as unavailable. Native IR
verification, ptxas, numerical tests, fresh-data CUDA graph replay and sanitizers
must still pass. The coordinator reports this limitation separately from speed.

Compile offline with `scripts/attn_fusion_native/run.sh cpu`. The coordinator
runs GPU tests through the system GPU queue and returns actual failure/timing
feedback. Do not modify the evaluator/reference or run independent GPU jobs.
