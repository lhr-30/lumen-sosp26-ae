# Native Hopper agent-generated attention

The accepted default is `native_candidate.py`, promoted byte-for-byte from
selected `round3_candidate.py`. QKV uses two m64n160 consumer warpgroups on a
160-token x 128-channel tile; OProj uses two m64n64 consumer warpgroups on a
64-token x 128-channel tile. Both use four K64 stages, at most one pending
WGMMA group, and full-completion TMA output stores. The selected default keys
are `qkv_t160_c128_w1` and `o_t64_c128_w1`.

The matched whole-chain comparison improved from round 2's 77.179 us to
66.572 us. Complete seed/graph/causal/zero-input checks and all three sanitizers
passed; all eleven default offline compile/assembly cases passed after
promotion. The default SHA256 is
`c25bacf5405cec557f226d442e7a6cd4c4fd5bf518fb2a48650ee2e26712fd71`.

`round2_candidate.py` remains the historical TMA-epilogue baseline at SHA256
`b3197d3bf1880691826e59b6d99642a0fc072ac28df23bdca66b9d8ce394f1f7`.
The round-0 and rejected round-1 sources are also preserved. Historical results
and the bounded round-3 selection are recorded below; no compiler or attention
kernel changes were included in this promotion.

## Scope and provenance

This experiment implements the same single-layer computation and fixed
B16, T26, H8, KV2, D256, hidden2048 dimensions as
`scripts/attn_fusion/TASK.md`. It uses the supplied AveLang native Hopper branch
at commit `5f349c594ad510ad6b90d0e1ca310c0d47aa9ed1`. The previously accepted
custom-backend candidate remains unchanged.

The native implementation adapts low-level patterns from the supplied repo's
`python/avelang_kernels/nvidia_gemm.py` and TMA/WGMMA primitive tests. It does not
copy a sibling model-tunner attention solution. All runtime tensor arithmetic
is AveLang; host code allocates/views tensors and uses the native launcher's TMA
descriptor plumbing. The coordinator owns compiler build adaptation and the
independent reference, GPU tests and measurements.

## Compiler capabilities and unavailable invariant checking

The native branch supplies TMA, larger WGMMA operations, explicit asynchronous
commit/wait, producer/consumer barriers, and fast scalar math. It does not
supply the artifact `invariant` module or a `validate_invariants` option. This
experiment neither imports fake invariant helpers nor passes an ignored flag.
Native IR verification, PTX assembly, numerical/graph tests and sanitizers are
reported as such; none constitutes an artifact-style data-flow invariant proof.

## Historical round-0 implementation

Each projection CTA computes a 64x128 output tile using m64n128k16 BF16 WGMMA
and FP32 accumulators. A dedicated producer warp issues TMA into three aligned
shared staging buffers, while 128 consumer threads process 64 K elements per
iteration. Full/empty mbarriers guard stage ownership and transaction bytes.
Four K16 WGMMA instructions share one committed group; the baseline waits for
that group before releasing shared storage. TMA production overlaps computation.

Tensor maps describe exactly 416 rows, so the final tile's 32 invalid A rows
are hardware zero-filled. Stores mask invalid output rows; both fixed output
widths are divisible by 128. This adapts the supplied square-only GEMM examples
to M=416 rather than calling them unchanged.

Unpack, Q norm/RoPE, K norm/RoPE, FP32 causal GQA/gate and residual are AveLang
kernels. Native `fast_rsqrt`, `fast_exp2`, `fast_rcp` and FP32 FMA replace the
previous polynomial/Newton workaround. Attention probabilities stay FP32, and
the reference's BF16 rounding boundaries remain explicit. Eight scalar
attention accumulators avoid dynamically indexed local arrays.

## Actual CPU failures and repairs

The first offline compile exited 139. Faulthandler located the crash in
`MLIRGenerator.visit_function_def`, before lowering; GDB found it in
`mlir::detail::OperandStorage` during operation creation. Frontend statement
prefixes isolated the first `make_tma_descriptor` call. Unpack, norm/RoPE and
causal GQA compiled independently, establishing that the compiler build worked.

The offline harness had omitted `global_constexprs`, unlike the JIT runtime, so
`SWIZZLE_128B` did not reach the frontend. The native intrinsic lacked a null
operand diagnostic guard for that malformed call. The harness now mirrors
runtime capture through `fn._collect_global_constexprs(...)`. No compiler or
hardware-semantic change was needed. Its `--only`, `--gdb` and
`--frontend-until` options preserve the CPU diagnosis workflow.

A separate issue appeared after compilation succeeded: a vector initialized by
`wgmma_init_result(64)` and reassigned in the K loop was not carried out of the
loop. PTX reset accumulators each iteration and stored literal zeros afterward
(42 allocated registers). This was caught before GPU execution. The repair
uses the supplied GEMM examples' supported representation: initialize a `(1,64)`
FP32 local fragment once, load it each iteration, and write the WGMMA result
back through a statically unrolled loop. The final stores read that fragment.
It fully promotes to registers, with no retained local memory or spills.

All seven repaired offline compile/ptxas checks passed. Baseline QKV/output
projection PTX sizes are 16392/16386 bytes, using 93 registers each. Every kernel
has zero stack/spills and no retained PTX local memory. Required TMA, mbarrier,
native m64n128 WGMMA, commit and wait instructions are present.

## Historical round-0 validation and timing

Full tests passed under system queue job 3551: three seeds, changed
input/weight/table contents during CUDA-graph replay, causal-prefix and batch
isolation, and zero input. Minimum nonzero cosine was 0.9999998140494218 and
maximum absolute BF16 difference was 0.015625. The coordinator preserved this
source at `.repro/attn-fusion-native-round0.py`.

Job 3553 completed all baseline sanitizers: memcheck and synccheck reported
zero errors; racecheck reported zero hazards, errors and warnings. A fresh,
separately measured original CuTe chain took 35.891 us.

| Stage | Baseline median (us) |
|---|---:|
| QKV projection | 37.922 |
| Unpack | 4.578 |
| Q norm/RoPE | 4.403 |
| K norm/RoPE | 2.568 |
| Causal GQA/gate | 17.555 |
| Output projection | 15.998 |
| Residual | 3.898 |
| Complete chain | 87.595 |

The native baseline is about 4.55x faster than the previous accepted custom
backend at 398.320 us, and about 2.44x slower than the fresh original CuTe result.
It does not reproduce the original implementation's performance.

## Rejected round-1 overlap experiment

The coordinator authorized a separate `round1_candidate.py` while the baseline
was queued, keeping the default source byte-identical. The harness supports
`--module round1_candidate` and puts its CPU artifacts in the corresponding
`/results/round1_candidate/` subdirectory.

Only projection scheduling changed. The inner loop permits one WGMMA group to
remain outstanding, releases stage `(kt-1)%3` after its group completes, and
explicitly drains the last group before releasing its stage or reading final
accumulators. Geometry, TMA buffers, other kernels and BF16 boundaries remain
unchanged; AST comparison confirmed the scope.

All seven draft compile/assembly checks passed, with 93 projection registers
and no local memory/spills. PTX inspection found accumulator initialization
before the loop, only WGMMA accesses while a group remains outstanding, and
BF16 conversions/stores after the final wait_group(0). These structural checks
are not a synchronization proof.

Independent full numerical/CUDA-graph tests passed under job 3554. The chain
median was 87.89280057 us versus the baseline's 87.59520054 us, with overlapping
p10/p90 intervals. There was no measurable improvement, so the experiment was
rejected and the validated baseline remains the default. No additional tuning
or draft sanitizer campaign was needed to make that selection.

## Historical source identities

- Historical round-0 source SHA256:
  `028c6747cffd1ae6e32d71ad90ce4d06bf51dfa0b760cc05913d4d5feee3cdac`
- Rejected `round1_candidate.py` SHA256:
  `ce4482d56ff29376d8b7761a5ba46c59d6e4c18f5f50b7c6ee8ab3f005d0d53d`

The coordinator's `validation.md`/`validation.json` provide the final report.
No compiler changes or independent GPU jobs were performed by this lowering
agent. These were the completed round-0/round-1 results before the user requested
the separate epilogue optimization recorded below.

## Historical round 2: TMA epilogue experiment (2026-11-02)

The user requested further agent-written DSL changes after NCU identified
inefficient scalar BF16 global stores: only 8 of 32 bytes per sector were used.
`round2_candidate.py` began as a new isolated candidate; the historical baseline
and rejected round-1 source were kept unchanged during its validation. No production lowering or compiler
code changed. The KernelWiki TMA/swizzle references and the supplied native
GEMM epilogue (`nvidia_gemm.py`, around lines 604-642) informed this adaptation.

Only `linear_native` changed. The 64x128 output tile, 3-stage input TMA pipeline,
K loop, WGMMA operations and arithmetic are preserved. The epilogue packs FP32
accumulators with the existing BF16x2 conversion, then uses eight stmatrix.x4
instructions to arrange two 64x64 boxes in the 128-byte-swizzled TMA layout.

Output scratch aliases the first 8192 BF16 elements of `b_shared` (stage 0).
After all 32 input full-barrier waits and the final WGMMA wait_group(0), there
are no future producer transactions and no remaining WGMMA reads. A 128-thread
named barrier joins consumers before overwriting that stage. Every writer then
publishes its shared stores through `fence_proxy_async_shared_cta`; a second
named barrier precedes the single TMA issuer. That thread commits the two output
stores and uses full `cp_async_bulk_wait_group(0)` completion, followed by a
consumer join and the existing final CTA barrier. Output writes finish before
kernel exit. This initial variant deliberately uses the full wait, not merely
shared-source read completion.

Separate 64x64 and 32x64 output tensor-map descriptors handle full and final
row tiles. For row0=384 only the first 32 valid rows of each box are transferred;
correctness does not depend on out-of-range TMA-store behavior. There is no new
shared-memory allocation: compiled dynamic shared storage remains 73984 bytes.

`compile_candidate.py --module round2_candidate` supports the new candidate.
CPU artifacts are isolated in
`.repro/attn-fusion-native-epilogue-results/round2_candidate/`. All seven
compile/ptxas checks pass. Both projections use 90 registers (baseline 93), no
stack/spills or retained local memory, eight stmatrix instructions, two 2D TMA
output stores and zero scalar global stores. A first harness check mistook PTX
whitespace before the full-wait immediate for a missing instruction; the check
was corrected to accept whitespace and still require an unqualified full wait.
AST comparison confirms every non-projection function is unchanged.

The coordinator froze the candidate for independent bitwise projection,
final-32-row and output-guard checks, and a three-way timing comparison against
the historical baseline and original CuTe. The resulting evidence below led to
acceptance and promotion. No GPU work was performed by this lowering agent.

Round-2 independent full tests passed under system queue job 3679: all three
seeds, changed-data CUDA graphs, causal checks and zero input. The whole chain
is bitwise identical to the accepted baseline. Both projection outputs,
including the final 32 rows, matched bitwise and output guard regions remained
unchanged.

One reservation used randomized interleaved timing with 100 calls per sample
and 20 samples. Baseline/round-2 medians were 87.931/77.035 us for the chain,
38.269/30.050 us for QKV and 15.690/12.532 us for output projection. Original
CuTe measured 35.800 us. These measurements establish an improvement while
retaining the conservative full store-completion wait. No read-only wait
variant or further tuning is being attempted this round.

A post-test lifetime review of source and final PTX found no missing join or
completion step. Stage 0's final input use is kt=30; both kt=30 and kt=31 drain
WGMMA, and the producer issues exactly 32 input tiles. Thus no input transfer
or WGMMA read remains when consumers join and repurpose stage 0. The output
address range is exactly shared bytes [24576,40960), excluding other stages and
barriers. Every writer fences after its stmatrix stores and joins before the
single issuer. The issuer performs a full bulk wait; the subsequent consumer
join and final 160-thread CTA barrier precede kernel exit. PTX selects the
32-row descriptor only for block_y=6, so the final tile does not depend on OOB
store suppression. This review is specific to the fixed, nonpersistent kernel
and does not claim unavailable invariant/protocol verification.

The tested round-2 SHA256 remains
`b3197d3bf1880691826e59b6d99642a0fc072ac28df23bdca66b9d8ce394f1f7`.
All three round-2 sanitizers then passed: memcheck and synccheck reported zero
errors, and racecheck reported zero hazards, errors and warnings. The
coordinator owns the separate NCU/profile report.


## Historical round-2 promotion

After the independent correctness, bitwise, tail/guard, timing and sanitizer
checks passed, the coordinator authorized promotion. `native_candidate.py` is
now an exact byte copy of frozen `round2_candidate.py`, retaining the tested
SHA256 above. The historical round-0 source and the rejected round-1 experiment
are preserved. No new scheduling change or relaxation of the full store wait
was included in promotion.

The compile harness now checks stmatrix, TMA stores, full bulk completion and
absence of scalar global stores for both the default and round-2 modules. One
fresh default-entry offline run passed all seven kernels. Both projections use
90 registers, 73984 dynamic shared bytes, zero stack/spills and no local memory;
the other kernels remain unchanged. The default compile report is isolated at
`.repro/attn-fusion-native-epilogue-results/compile.json`.

The synchronization review and independently measured improvement are recorded
above. No further CPU/GPU experiments are required from this lowering agent;
README, validation reports and frozen profiling sources are owned by the
coordinator and were not edited here.

## Round 3: swapped projections and bounded scheduling/shape comparison

The user requested further agent-written DSL changes after the round-2 NCU
report identified remaining long-scoreboard and WGMMA completion-barrier stalls.
The accepted round-2/native source remains unchanged at SHA256 `b3197d3b...`.
The new self-contained `round3_candidate.py` swaps the WGMMA operand roles:
weights supply the 64-channel M dimension and activations supply the token N
dimension. Transposed stmatrix instructions restore the public [token,channel]
layout before full-completion TMA output stores. No scalar attention, compiler
or production-lowering changes were made.

The comparison is limited to four named configurations:

| Config | N | Token x channel tile | Consumer WGs | CTAs | Pending groups |
|---|---:|---|---:|---:|---:|
| qkv_t160_c128_w1 | 5120 | 160x128 | 2 | 120 | 1 |
| qkv_t160_c128_w0 | 5120 | 160x128 | 2 | 120 | 0 |
| o_t128_c64_w1 | 2048 | 128x64 | 1 | 128 | 1 |
| o_t96_c64_w1 | 2048 | 96x64 | 1 | 160 | 1 |

All use four K64 input stages and FP32 accumulation. The QKV tile shares each
160-token input tile across two independent consumer warpgroups, each computing
64 output channels. An empty barrier's count equals the number of consumer WGs;
each WG releases the previous stage only after wait_group(1) proves that older
group complete. The bounded wait_group(0) control releases the current stage.
Every WG drains its final group before the common consumer join.

Only after all consumer WGs join does the weight arena become output scratch.
The two QKV WGs then own disjoint 160x64 regions; each has its own output
publication/join barrier and TMA issuer. Output storage fits inside the existing
weight staging arena, so no additional shared allocation is introduced. TMA
output descriptors explicitly encode full/tail rows: 160/96, 128/32 or 96/32.
The tensor maps still describe exactly 416 input rows. The producer issues
exactly 32 K tiles and no persistent next-tile prefetch can overlap reuse.

The source exposes `PROJECTION_CONFIGS`, `OFFLINE_COMPILE_CASES` and
`bind_projection(A,W,C,N,config=None)`. The optional config is one of the string
keys above; the binder only returns a zero-argument native launch. `build`
uses this same binder and initially selects QKV w1 and OProj 128. Separate
shape-specialized JIT functions avoid relying on the frontend to prune an
unsupported WGMMA shape before operand type checking. They are explicit DSL
kernels, not a runtime source generator or open-ended autotuner.

All nine CPU compile/ptxas cases passed (four projections plus five unchanged
scalar specializations). QKV uses 106 registers and 147712 declared dynamic
shared bytes; OProj 128 uses 90 registers/98560 bytes; OProj 96 uses
74 registers/82176 bytes. All have zero stack/spills, no retained local memory,
no scalar global stores, and retain transposed stmatrix/TMA/full completion.
The QKV resource report lists 16 barrier identifiers because its per-WG barrier
ID is dynamic; only IDs 0 through 3 are used by the source. Its shared footprint
already limits residency to one CTA on H100, an intentional tradeoff for the
120-CTA, two-consumer design. Runtime resource accounting is left to NCU.

PTX inspection found no accumulator use outside WGMMA before final wait_group(0)
for the pending-group configurations. AST comparison confirms unpack, norm/RoPE,
gate math, causal GQA and residual bodies are identical to the accepted source.
CPU artifacts are isolated in
`.repro/attn-fusion-native-round3-results/round3_candidate/`.

The frozen first-round source SHA256 is
`d5d3dbaf6bfdf68468a60f2b9380158161236daecb3700435d34469bd3f37d47`.
The coordinator subsequently compared all four configurations against an
independent reference and recorded bitwise equality after swapping operands.
The resulting selection and complete-chain checks are recorded below. No GPU
job was run by this lowering agent.

Round-3 first sweep passed under job 3691: every projection/reference, finite,
guard and recorded bitwise comparison passed. QKV measured 30.085 us for the
accepted baseline, 20.600 us for the new w1 configuration (p10/p90
19.969/20.634), and 20.453 us for w0 (20.219/22.641). Both new QKV configurations
clearly improve the baseline, but their intervals overlap: this experiment does
not establish that wait_group(1) is faster than wait_group(0). The default QKV
selection remains w1 pending complete-chain validation.

The first two OProj shapes did not improve timing: baseline median 13.908 us
(p10 12.499), token128 14.365 us and token96 15.902 us. They remain recorded
experiments, not accepted defaults. The round-3 module now embeds the accepted
round-2 projection as `linear_round2` and config `o_round2`; AST comparison
confirms it differs only in function name. This is the default OProj choice.
The module stays self-contained and still uses `bind_projection` in `build`.

The coordinator authorized one final evidence-based OProj configuration,
`o_t64_c128_w1`: 64 tokens x 128 channels, two m64n64 consumer WGs, four K64
stages, 112 CTAs and the same 448-token padded work as round 2. This changes
consumer parallelism without the additional token padding of the failed
128/96-token alternatives. It is the only additional tuning candidate.

All eleven offline compile/assembly cases passed after adding the final test
and embedded baseline. The new OProj variant uses 58 registers and 98560 declared
dynamic shared bytes, with no stack/spills/local memory. Its 32 accumulator
registers have no non-WGMMA use before the final wait_group(0). The accepted
native/round-2 file remains unchanged. The updated frozen round-3 SHA256 is
`743bef998f170f9e2fdd7589c6947c3be5040955dc78ddb75972503145282c57`.
The coordinator then performed the last OProj comparison and selected
full-chain test, with results recorded below.

Round-3 final isolated sweep (job 3699) passed every bitwise and guard check.
QKV w1/w0 measured 21.219/21.275 us, again with overlapping distributions. The
final dual-WG OProj measured 13.102 us versus its same-run baseline 14.591 us,
but previous baseline observations were about 12.5 us. This variability made
whole-chain comparison necessary; isolated OProj timing alone was not used to
select it.

Job 3701 compared four complete chains within the same randomized experiment:
round 2, new QKV with old OProj, new QKV with dual-WG OProj, and original CuTe.
The chain medians were 77.179, 67.170, 66.572 and 35.874 us respectively. New QKV
was 19.55 us versus old QKV 30.44 us. The standalone dual-WG OProj was 12.913 us
versus old OProj 12.637 us, so the data does not establish an independent OProj
speedup. In the complete-chain paired samples, however, dual-WG OProj won 18 of
20 rounds: median saving 0.578 us, with bootstrap 95% interval approximately
0.521-0.675 us. Selection therefore follows the measured chain benefit.

The selected round-3 defaults are `qkv_t160_c128_w1` and `o_t64_c128_w1`. Only
the host default OProj selector changed after the last frozen source; every GPU
kernel remains the already-tested implementation. QKV w0 is retained as a
control with no demonstrated superiority/inferiority; OProj token128 and token96
were not selected after failing to improve timing; `o_round2` remains the
embedded historical baseline and a supported comparison configuration. There
are no further shape or schedule candidates this round.

The selected frozen source then passed the complete numerical/graph checks and
all three sanitizers before promotion, as recorded below. The promotion also
adapted offline instruction checks to projection metadata and fixed constexpr
injection for frontend-prefix diagnosis.


## Round-3 final validation, lifetime review and promotion

Job 3705 passed seed 0/17/123, changed-data CUDA graph replay, causal/batch
isolation and zero input. Memcheck and synccheck reported zero errors; racecheck
reported zero hazards, errors and warnings. Projection bitwise and guard checks
had already passed the independent sweeps. The measured whole-chain selection
is 66.572 us versus round 2's 77.179 us in the matched comparison. Selection of
the dual-WG OProj follows its whole-chain benefit, not a claim that its isolated
kernel is faster; w1 versus w0 also remains statistically unresolved.

The final multi-WG lifetime review found no missing release, phase transition or
completion step for these fixed nonpersistent kernels:

- Each full barrier has one TMA issuer and the exact sum of weight/token bytes.
  Each empty barrier has count two. Only lane128=0 of each consumer WG arrives;
  the producer warp is outside the consumer branch. A stage cannot be reused
  until both WGs have finished its WGMMA reads.
- Full parity is `(kt//4)%2`, and producer empty parity is its complement. For
  w1, iteration kt releases `(kt-1)%4` only after wait_group(1); final
  wait_group(0) plus the final stage-3 release retires every group. The w0
  comparison instead releases the current stage after wait_group(0).
- A 256-thread consumer join follows each WG's final drain. Only then is the
  weight arena aliased as output scratch. QKV WG regions are disjoint 20480-byte
  regions; OProj WG regions are disjoint 8192-byte regions. Both fit within the
  existing 65536-byte weight arena. All 32 input TMA tiles have completed before
  this ownership change, and no persistent next-tile prefetch exists.
- Each writer fences after its transposed stmatrix stores and joins its own WG
  before its sole TMA issuer. Both issuers perform full bulk completion waits;
  per-WG joins and the final 288-thread CTA barrier precede exit. Tail maps
  transfer only 96 valid QKV rows or 32 valid OProj rows in the last tile.

These are source/PTX review conclusions supported by the runtime checks, not
artifact-style invariant proofs, which remain unavailable in this compiler.

The coordinator authorized copying frozen `round3_candidate.py` exactly to
`native_candidate.py`; both now retain SHA256 `c25bacf...12fd71`. Historical
round-2 and round-3 experimental sources were preserved, and no frozen profile
source or coordinator-owned README/validation report was edited here.

The offline harness now decides transposed stmatrix checks from each projection
specification's `transpose`/`argument_style` metadata, independent of module
name. Frontend-prefix diagnosis selects the requested/default offline case and
passes all its constexprs, including WAIT_GROUP. One fresh default run passed
all eleven compile/ptxas cases; a full selected-QKV frontend-prefix check also
passed. Default artifacts are under
`.repro/attn-fusion-native-round3-results/`, including `compile.json`.
No further experiments are pending for this lowering agent.

## Round 4: packed-input middle-path fusion

The user requested further agent-written DSL changes focused on attention, with
all round-3 projections/default configurations and residual preserved. The first
`round4_candidate.py` implements five launches: QKV projection, packed-input K
norm/RoPE, fused Q norm/RoPE + causal attention + gate, output projection, and
residual. `PROFILE_STAGE_TAGS` is `['qkv','knorm','attention','oproj','residual']`.
Projection kernels/binder, gate rounding helper and residual were verified
AST-identical to the accepted round-3 source.

`bind_middle(packed,qnorm_weight,knorm_weight,tables,attended,config=None)` only
allocates normalized-K scratch, makes host views and returns two ordered
zero-argument launches. It performs no host tensor arithmetic. The first and
only initial attention config is `shared_q4`, exported through ATTENTION_CONFIGS
and DEFAULT_ATTENTION_CONFIG.

K norm/RoPE reads the exact packed K offsets directly. The fused attention CTA
contains four warps/queries from the same batch and query head. All threads
cooperatively stage only the required causal prefix of normalized K and packed
V in shared memory, then join once before any query work. Tail query warps branch
after that join, so they cannot skip a required CTA barrier. Query positions and
key ranges never cross batch boundaries; query head h selects KV head h//4.

Each active warp loads its eight interleaved-Q components per lane directly
from packed storage, computes the 256-channel RMS normalization, rounds to BF16,
then computes and separately rounds split-half RoPE. The two rotated channels
are owned by the same lane, so they remain in registers. FP32 SIMT QK, stable
online softmax and FP32 PV preserve the existing mathematical decomposition.
The attention result and sigmoid are independently rounded before the final
BF16 gate product. No BF16 probability rounding or compiler change is introduced.
Gate values come from h*512+256+d, not a separate gate prefix.

All nine CPU compile/ptxas cases passed: six preserved projection configurations,
packed K normalization, fused attention and residual. Fused attention uses
48 registers and 26624 bytes of static shared storage, with no stack/spills or
retained local memory. Packed K norm/RoPE uses 26 registers and 512 shared bytes.
Artifacts are isolated under
`.repro/attn-fusion-native-round4-results/round4_candidate/`.

The first frozen round-4 SHA256 is
`95ab08bda11bd64a2b65ee2be19bf14a25ce26a8e2d9e11390a26d21b71618b8`.
The accepted native/round-3 SHA remains `c25bacf...12fd71`. The coordinator owns
independent packed-to-attended reference checks, graph/prefix/batch/group/gate
checks, guards and timing. A single MMA candidate may follow only if measured
feedback justifies it; no wider search is planned.

While the first shared candidate remained queued behind another project's NCU
work, the coordinator explicitly authorized CPU implementation of the single
planned MMA alternative in a separate `round4_mma_candidate.py`. The queued
`round4_candidate.py` remained byte-identical at SHA256 `95ab08bd...`.

The MMA module keeps both `shared_q4` and `mma_q8` in ATTENTION_CONFIGS and selects
`mma_q8` by default. Its build/bind_middle/stage-tag interfaces are unchanged.
Each 128-thread CTA owns eight logical queries; all sixteen physical MMA query
rows are initialized, including padded rows and invalid final queries. K/V
storage has 32 initialized rows, zeroing positions beyond the required causal
prefix. Q and K use a 128-byte shared swizzle suitable for ldmatrix; four warps
compute the four eight-key columns of a 16x32 FP32 score tile with 16 BF16
m16n8k16 instructions per warp. A local four-element accumulator is explicitly
written back on each K step, following the supported compiler representation.

After the second CTA barrier, each warp handles two valid queries. It masks
future keys and keys >=26 before a parallel stable FP32 softmax. Each lane keeps
one FP32 probability; shuffle broadcasts probabilities to the FP32 PV loop.
PV uses the native non-FTZ `fma` intrinsic, not BF16 MMA, and final attention/gate
rounding uses the unchanged gate_store helper. No BF16 probability conversion
is present. All query-validity exits occur after both CTA barriers. The unused
MMA query rows are never read by softmax or written to output.

All ten offline compile/assembly cases passed. The MMA fused kernel uses 55
registers, 43520 shared bytes, zero stack/spills and no retained local memory.
PTX contains the expected ldmatrix x4/x2, BF16xBF16 FP32-accumulator MMA, FP32
score stores, probability shuffles and non-FTZ FP32 PV FMAs. Every pre-existing
function body, including all projections, residual, shared_q4 and bind_middle,
is AST-identical to the first shared module. No compiler/reference/tolerance
change or independent GPU workload was introduced.

The frozen MMA module SHA256 is
`90666885c85527ed1c9a0cbf1988a2e764a2f0ddb7fd70f2f61b90b40d9bfa63`.
Its CPU artifacts are under the independent round4-results
`round4_mma_candidate/` subdirectory. Numerical, structured-boundary, graph and
performance evaluation remain the coordinator's responsibility; no winner is
claimed before those measurements.

The first shared candidate passed independent middle validation under job 3717:
all six middle cases and output guards passed, as did full seed/graph/causal/
zero-input validation. Its complete chain was bitwise identical to round 3.
Matched medians were 62.216 us versus 66.587 us for the chain and 23.931 us
versus 27.821 us for the middle path. The fused kernel itself took 21.830 us,
compared with old Q norm 4.076 us plus attention/gate 17.044 us. Thus the first
version's gain mainly comes from eliminating unpack/materialization and launch
overhead; the measurements do not show that SIMT shared-K/V computation alone
is faster. They support testing the single MMA/parallel-softmax alternative.

The frozen MMA candidate was submitted by the coordinator as job 3719 for
independent middle, full-reference and four-way round3/shared/MMA/CuTe timing.
Both candidate source files remain unchanged while those checks are queued.

Job 3719 passed all independent middle and full numerical checks for MMA q8,
but timing regressed. The matched chain medians were 66.366 us for round 3,
61.508 us for shared_q4 and 67.740 us for mma_q8. Middle medians were
27.815/23.822/27.960 us respectively. The MMA fused kernel itself took 24.773 us
versus shared_q4's 21.644 us. MMA q8 is not selected or promoted.

Read-only review identified factors for NCU diagnosis, not established causes:
MMA q8 uses only 40.6% of its physical 16-row query tiles after sequence tails;
its shared footprint increases from 26624 to 43520 bytes and CTA count falls
from 896 to 512; each warp serially performs PV for two queries. A single
possible repair is to pack four query heads sharing one KV head together with
four token positions into the 16 MMA rows, using sixteen warps so each query's
FP32 PV gets its own warp. This would restore 92.9% row utilization and reuse
K/V across all four GQA heads, at the cost of more threads and idle warps during
the QK phase. The coordinator is profiling both frozen implementations before
deciding whether this one additional repair is justified; neither source has
changed since measurement.

NCU job 3722 subsequently identified the serialized K/V preload as the main
measured MMA-q8 hotspot. The coordinator reported 43.77% long-scoreboard stalls;
the first STS.U16 after a K/V global load accounted for 523 long-scoreboard
samples out of 539 total at that instruction. Shared-bank conflict totals were
similar between the two candidates (11191 versus 11226), so the evidence did
not establish ldmatrix swizzle conflicts as the regression cause. Profile data
is retained under `profile/attn-native-round4-diagnostic-h100-20261102`.

The coordinator therefore authorized one isolated 16-byte cp.async repair,
keeping the existing MMA-q8 geometry, instead of the proposed GQA/head regroup.
`round4_async_candidate.py` was copied from frozen `round4_mma_candidate.py` and
changes only the K/V preload in its new `fused_qnorm_attention_mma_q8_async`
kernel. It uses eight 16-byte copies per tensor/thread in place of sixty-four
scalar loads/stores, conditional on a complete valid key row. The swizzle
permutes whole 16-byte segments. The current native cp.async intrinsic has no
source-size/zero-fill operand, so invalid K/V rows are explicitly zeroed without
forming global loads. Copies are committed before Q normalization/RoPE and are
fully waited with wait_group(0), followed by the existing CTA barrier, before
any ldmatrix/shared read. All threads reach both CTA barriers.

The config is `mma_q8_async`: grid `(4,8,16)`, 128 threads, four warps and eight
logical queries per CTA. All existing interfaces and the five-stage chain are
preserved. Q normalization/RoPE and the entire MMA/softmax/FP32-PV/gate suffix
were checked AST-identical to frozen q8. Every other function body, including
all projection kernels/binders and residual, is also AST-identical. CPU address
enumeration verifies every K/V storage element has exactly one writer and all
valid global copy segments are aligned and in bounds. No compiler/reference,
tolerance, projection, or residual edits were made.

All ten CPU compile/ptxas cases passed. The async kernel uses 56 registers,
43520 shared bytes, zero stack/spills and no retained local memory. PTX includes
two 16-byte cp.async.ca instructions in the preload loop, commit_group before
Q normalization, wait_group(0) immediately before the first CTA barrier, the
original ldmatrix/MMA instructions and non-FTZ FP32 PV FMAs. The compile checker
now accepts the async module and explicitly checks its asynchronous copy and
completion instructions. Artifacts are isolated under
`.repro/attn-fusion-native-round4-results/round4_async_candidate/`.

The frozen async source SHA256 is
`f0bb46ed3222fd3a9ac32ff9f01f6ac6f250e9138a5ded70ae7cf3d324546cf2`.
The shared, original MMA-q8 and accepted native sources retain their previous
hashes. The coordinator owns independent GPU correctness, sanitizers and
matched timing; compilation alone does not establish a performance improvement.

## Round 4 acceptance and exact default promotion

Coordinator GPU job 3727 passed all six independent middle cases and output
guards, plus full seed 0/17/123, graph, causal and zero-input checks for the
async candidate. Its attended/output tensors were bitwise identical to the
previous MMA-q8 candidate. The final matched five-way chain measurements were
66.213 us for round 3, 61.812 us for shared-q4, 67.855 us for raw MMA-q8,
55.404 us for async MMA-q8 and 36.270 us for the CuTe baseline. The async middle
path took 16.982 us versus round 3's 27.739 us, and its fused attention took
14.869 us versus raw MMA's 24.816 us. These are coordinator-measured runtime
results, not the clock-controlled NCU profiling durations.

The coordinator reviewed actual sanitizer logs from job 3729: memcheck and
synccheck each reported `ERROR SUMMARY: 0 errors`; racecheck reported zero
hazards, errors and warnings. All three smoke runs retained normal numerical
results/hashes. Only after these results, the coordinator authorized default
promotion. No further tuning was authorized or performed.

`native_candidate.py` is now an exact byte-for-byte copy of the frozen async
candidate, SHA256
`f0bb46ed3222fd3a9ac32ff9f01f6ac6f250e9138a5ded70ae7cf3d324546cf2`.
Its attention default is `mma_q8_async`; projection defaults remain
`qkv_t160_c128_w1` and `o_t64_c128_w1`. All frozen round-3/shared/raw-MMA/async
sources retain their recorded hashes.

`compile_candidate.py` now matches attention kernels through ATTENTION_CONFIGS
metadata, with named-case fallbacks, rather than coupling attention checks to
the module name. It checks BF16 QK MMA and ldmatrix x4/x2; FP32 probability
shuffles and non-FTZ PV FMA, rejecting FTZ FMA; and 16-byte cp.async.ca with
commit and wait_group(0). Compile reports record the matched attention config
and each performed PTX check. These structural checks complement the independent
GPU numerical/sanitizer checks; they do not claim invariant-module coverage.

All ten default CPU compile/ptxas cases passed using
`NATIVE_RESULTS=.repro/attn-fusion-native-round4-results/default` (resolved to an
absolute path for Docker). Reports, PTX and cubins are in that directory. Every
generated default PTX file is byte-identical to its frozen async-candidate CPU
artifact. All cases have no stack/spills or retained PTX local memory; the
selected attention kernel remains at 56 registers and 43520 shared bytes. The
three required attention check groups are recorded in default `compile.json`.
The lowering agent performed no GPU work. Final README, validation and NCU
report updates remain the coordinator's responsibility.
