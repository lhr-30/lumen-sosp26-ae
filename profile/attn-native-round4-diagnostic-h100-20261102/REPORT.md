# Round 4 diagnostic: scalar K/V staging stalls the MMA candidate

H100 NVL, B16/T26/H8/KV2/D256, BF16 Q/K/V with FP32 softmax probabilities and
FP32 PV accumulation. This compares the two fused Q norm/RoPE + attention/gate
kernels. `baseline` means shared_q4, and `candidate` means mma_q8 in this run.
The frozen source hashes are in `harness/source_manifest.json`.

Normal graph timing from the same reservation in the parent experiment:
shared_q4 21.644 us, mma_q8 24.773 us. Both pass the unchanged independent
reference; switching QK to MMA alone regresses performance.

## Measured diagnosis

| NCU metric | shared_q4 | mma_q8 |
|---|---:|---:|
| Base-clock replay duration | 34.944 us | 38.016 us |
| CTAs / threads per CTA | 896 / 128 | 512 / 128 |
| Registers/thread | 48 | 55 |
| Shared bytes, including system overhead | 27648 | 44544 |
| Achieved occupancy | 28.86% | 21.91% |
| Shared-memory CTA residency limit | 8 | 5 |
| Long-scoreboard source samples | 25.07% | **43.77%** |
| Short-scoreboard source samples | 20.80% | 9.67% |
| Executed warp SASS instructions | 8110592 | 7618304 |
| Shared bank-conflict count | 11226 | 11191 |

Fewer instructions and real MMA execution did not compensate for the memory
waits and lower residency. DRAM read throughput is negligible in these warmed
replays; this is not evidence of HBM saturation.

The dominant mma_q8 hotspot is `STS.U16 [R8], R15`: 539 source samples, of
which 523 are long-scoreboard. It maps to PTX line 72 in
`fused_qnorm_attention_mma_q8_ad48ac4038.ptx`. The surrounding block waits on the
preceding `ld.global.b16` K load, then writes one BF16 to shared storage. A
second scalar shared store has the same dependency for V. Each thread repeats
this scalar staging loop 64 times. This is the strongest concrete diagnosis;
PV serialization and padding remain structural limitations, not the primary
hotspot found in this report.

The bank-conflict counts are similar, and a simple address review of each
8-row ldmatrix matrix finds no obvious bank aliasing. Neither observation
proves zero bank conflicts, but neither supports blaming the regression on a
large new ldmatrix bank conflict. See `analysis/layout_review.json`.

The MMA tile has 16 physical query rows but only eight logical queries, and
sequence tails reduce useful M rows to 40.6%. Together with shared usage and
per-warp serial processing of two queries, this motivates later layout work;
it does not identify a causal speedup without another experiment.

## Single targeted follow-up

Keep the q8 geometry and all Q norm/MMA/FP32 softmax/PV arithmetic unchanged.
Replace K/V scalar staging with 16-byte cp.async copies (eight BF16 per copy),
commit them, compute Q norm while the copies are in flight, then wait_group(0)
and synchronize before reading shared data. The current DSL has no optional
cp.async source-size/zero-fill operand, so invalid whole rows must be explicitly
zeroed and must not issue global copies. This needs no compiler change.

The follow-up source is recorded in the parent experiment as
`round4_async_candidate.py`; its performance must be independently measured.
The unsupported direct inference “Tensor Cores are slower” is not warranted.

## Artifacts and limitations

The primary agent collected full+PM and source reports and parsed them using
`ncu_report`: `analysis/comparison.json`, `stalls_*.json`, `opcodes_*.json`,
`*_attention.sass`, and the corresponding full/source reports. The acquisition
used system queue **3722**, exit 0; CPU parsing used no GPU reservation.

NCU uses base clocks and `--cache-control none`. It is separate from ordinary
CUDA graph timing. One derived L2 hit ratio exceeds 100% due to replay-counter
inconsistency and is excluded from interpretation (`measurement_warnings.json`).
Source sample proportions are not wall-time fractions. PM sampling is 1.5 us;
no precise phase-duration claim is made from those samples.
