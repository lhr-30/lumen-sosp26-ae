# Description-generated lowering notes

The implementation was generated from TASK.md and input_model.py. Public AveLang
DSL documentation, primitive invariant tests, and the unrelated Hopper GEMM
example supplied syntax and WGMMA layout information. No sibling attention
implementation or archived generated candidate was read.

The baseline has seven launches: packed BF16 projection; unpack Q/gate/K/V;
Q norm+RoPE; K norm+RoPE; causal GQA+gate; BF16 output projection; residual.
Both linear layers use the already available m64n8k16 BF16 WGMMA intrinsic,
FP32 accumulators, mbarrier phase reuse and 128-thread CTAs. The attention uses
one warp per query/head, FP32 warp reductions and stable online softmax; it
never rounds probabilities to BF16. The specified BF16 points are explicit
conversions, including normalization before RoPE and sigmoid before gating.

Invariant validation stays enabled for every offline compile and GPU launch.
Structural assertions cover exact packed source coordinates (including the
interleaved gates), split-half RoPE pair token/head/frequency agreement, batch
and KV-group correspondence in QK and PV, and residual channel correspondence.
Output tags in unpack are source-coordinate specifications consumed only by
compile-time assertions. Negative compile specializations perturb gate offset,
RoPE pair offset, or GQA group to ensure these checks reject mistakes.
These checks do not prove normalization/reduction numerical semantics, WGMMA
arithmetic, causal loop bounds, or the barrier protocol; the current compiler
has no WGMMA arithmetic tag-transfer rule. GPU numerical/graph tests are owned
by the independent coordinator.

The sections below record compile and GPU feedback, repairs, and the single
measured optimization round. All runtime numerical work remains AveLang DSL.

## Compile feedback and repairs

1. The first import needed postponed annotations for constexpr-dependent tensor
   dimensions (`from __future__ import annotations`); corrected in the DSL file.
2. WGMMA projection and packed-coordinate unpack compiled and assembled first.
   `S.sqrt` compiled to a call to `__nv_sqrtf`, which failed assembly with
   `ptxas fatal: Unresolved extern function '__nv_sqrtf'`. The fixed compiler
   does not link libdevice. Rather than change the compiler, the candidate uses
   a pure-DSL FP32 reciprocal square root: an integer-bit exponent seed followed
   by four Newton iterations. Its input is mean-square+1e-6, always positive for
   finite inputs; it does not claim correct behavior on infinities/NaNs or
   arbitrary subnormal inputs. Current finite benchmark values are in range.
3. The candidate also avoids external exp calls by expressing exp(x), x<=0,
   with base-two range reduction and a degree-seven Taylor polynomial on
   [-ln(2)/2,ln(2)/2]. Inputs below -80 return zero, whose absolute truncation
   error is below exp(-80). Softmax arguments are nonpositive by construction;
   sigmoid uses exp(-abs(g)) and a sign-dependent stable quotient. Both retain
   FP32 arithmetic and only the required BF16 rounding points. Approximation
   accuracy is assessed by the independent reference comparison, not assumed
   to be bit-identical to libdevice.
4. Floating max/abs likewise produced unresolved `__nv_fmaxf`/`__nv_fabsf`.
   Scalar comparisons implement these operations without changing finite-input
   semantics. This is another current NVIDIA math-lowering limitation.
5. GQA assertions could not prove an implicit flattened-time batch identity,
   then could not prove `h//4` in the original head layout. Inspection found the
   validator recognizes signed integer division but not unsigned division, and
   its proof domain does not carry every lowered loop bound. The representation
   was changed to explicit [B,T,KV,4,D] Q/gate/output views and [B,T,KV,D] K/V
   views. Assertions now directly compare the same batch/group/channel and pass
   even inside the reduction loops. No assertion was disabled or removed to
   obtain a successful compile. A pre-loop assertion additionally checks group
   selection independent of loop analysis.

The current `scripts/attn_fusion/run.sh cpu` passes all seven specializations,
all seven `ptxas -arch=sm_90a` assemblies, and all three intended failing
specializations (gate +1 channel, RoPE pair +1 channel, KV group +1). The exact
rejection diagnostics and PTX sizes are in `/results/compile.json`.
This successful compile was then followed by the GPU failure and repair
described below.

## First GPU feedback: local-memory lowering

The first full run (system GPU queue job 3278) reported an asynchronous illegal
memory access while loading the following output-projection kernel. The
coordinator then ran Compute Sanitizer memcheck (job 3280). It identified
`Invalid __local__ write of size 4 bytes at causal_gqa_gate+0xb70`, across many
threads (42689 reported errors). This was a real failing candidate, not a
numerical tolerance issue.

The initial attention kept eight accumulators in `S.make_local((8,), f32)`.
The emitted PTX retained a 32-byte local depot and converted its address between
local and generic address spaces inconsistently. The final indexed accumulator
loop prevented complete scalar promotion. The DSL repair explicitly names all
eight FP32 accumulators and statically expands their updates and final stores.
All existing coordinate assertions remain, including one assertion per PV
channel. No compiler or reference changes were made. The repaired attention PTX
contains no local depot, `ld.local`, `st.local`, or external device function.

After repair, all seven offline compilations/assemblies and all three negative
invariant checks pass again. Attention PTX is 18808 bytes. The coordinator subsequently reran GPU correctness and all three sanitizers
successfully, as recorded below.

## Repaired baseline numerical result

Independent full GPU validation passed (system queue job 3281): three random
seeds, graph replay after changing inputs/weights/RoPE tables, causal-prefix and
batch-isolation checks, and zero input. The minimum nonzero cosine reported by
the coordinator is 0.9999998149764577; the maximum absolute BF16 difference is
0.015625. This establishes correctness against the supplied benchmark criteria
for those tests, not bit-identical FP32 intermediate arithmetic. The passing
baseline is preserved by the coordinator at
`.repro/attn-fusion-candidate-round0.py`; independent raw report:
`.repro/attn-fusion-results/round0_test.json`.

The coordinator next measured baseline timing and ran all three sanitizers;
the results are recorded in the optimization section below.

## One measured optimization round

All three baseline sanitizers completed successfully under job 3282:
memcheck/synccheck reported zero errors; racecheck reported zero hazards,
errors and warnings. Independent baseline timing was 551.418 us for QKV,
302.309 us for output projection and 889.677 us for the complete chain. The
linear layers account for approximately 96% of chain latency.

Following OPTIMIZATION_01.md, the new projection computes a 64x32 output tile
per CTA using four separate m64n8 WGMMA operations. Four aligned shared B tiles
and four 4-register FP32 fragments reuse a single shared A tile. Each K step
loads all four B tiles, publishes all A/B stores with the existing per-thread
proxy fence and mbarrier, then executes four blocking WGMMA helpers before the
single CTA buffer-reuse barrier. Uniform participation, 128 K steps, phase
alternation and BF16 final rounding are preserved. M-tail rows remain masked;
both fixed N values are divisible by 32.

Only `linear_bf16` and its two launch grids changed; AST comparison confirmed
all attention, unpack, normalization, gating, residual and math-helper bodies
are unchanged. All seven offline compilations/assemblies and all three negative
invariant tests pass. Projection PTX has four WGMMA instructions and no retained
local memory or external calls. Attention likewise still has no local memory.
Independent correctness and timing subsequently passed; final sanitizer
validation of this optimized source is still running under the coordinator.


## Measured round-1 result and final handoff

The independent full test plus benchmark passed (system queue job 3288). All
numerical cases, including graph replay with changed data/weights/tables,
causal/batch isolation and zero input, retain the round-0 reported metrics.
The minimum nonzero cosine is 0.9999998149764577, with maximum absolute BF16
difference 0.015625. No numerical threshold or reference was changed.

Median CUDA-graph timing (10 calls/graph, 10 samples) changed as follows:

| Stage | Round 0 (us) | Round 1 (us) |
|---|---:|---:|
| QKV projection | 551.418 | 225.598 |
| Output projection | 302.309 | 130.830 |
| Complete chain | 889.677 | 398.320 |

This is a 2.23x improvement over the generated AveLang baseline. The coordinator
also measured the existing original CuTe chain separately at 35.096 us on the
same H100 (job 3289), so this candidate remains 11.35x slower than that original
implementation. It does not reproduce the original implementation's speed.
The remaining projections use synchronous WGMMA helpers without TMA or overlap;
no further optimization is attempted this round.

Final source is frozen. All seven offline compiles/assemblies and three
negative invariant mutations pass for it; the complete numerical/graph test
passes. The coordinator is running final memcheck, synccheck and racecheck for
the optimized source under job 3290 and owns the final acceptance/report. The
baseline's three sanitizer passes do not substitute for these final checks.
Reports are `.repro/attn-fusion-results/round1_bench.json` and the coordinator's
separately labeled original-baseline report. No compiler changes, other DSL
solutions, host numerical fallbacks or independent GPU jobs were used by this
lowering agent.

## Coordinator acceptance

The final optimized source passed all three sanitizer runs in queue job 3290:
memcheck and synccheck each reported zero errors; racecheck reported zero
hazards, errors or warnings. The coordinator accepted this candidate after the
full numerical checks and the measured improvement over round 0. Final source
identity, environment, original CuTe comparison and limitations are recorded in
`validation.md` and `validation.json`.
