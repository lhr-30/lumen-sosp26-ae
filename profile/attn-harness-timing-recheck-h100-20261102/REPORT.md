# Timing repeat: background GPU activity confirmed

This run repeats the frozen five-way H100 B16/T26 mixer comparison with 100
calls per graph and 20 randomized timing rounds. System GPU queue 3774 exited 0.
No source changes were made; the harness additionally records per-sample wall
windows and `nvidia-smi pmon` process activity.

The measured window 08:52:22.162–08:52:24.199 UTC overlapped a resident Triton
Python backend at 54%, 49% and 43% SM activity. This process was present before our
CUDA process and remained active during the queue reservation. Consequently the
results must not be used to rank the final harness implementation against the
manual implementation or to claim precise speedups.

| Implementation | Median μs | p10–p90 μs |
|---|---:|---:|
| Harness round0 | 104.785 | 89.671–125.380 |
| Harness round1 | 94.631 | 79.422–124.199 |
| Harness round2 | 76.190 | 59.212–96.058 |
| Prior manual round4 | 77.771 | 58.349–95.309 |
| CuTe | 44.095 | 35.400–59.875 |

[Raw timing](analysis/timing.json), [activity analysis](analysis/background_activity.json),
and [process sampling](logs/process_activity.log) are retained. The full
experiment, source identities, correctness and sanitizer results are in the
[parent report](../attn-harness-h100-20261102/REPORT.md).
