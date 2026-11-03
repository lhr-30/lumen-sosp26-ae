"""Plot the measured PM timelines; these are sampled replay traces."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
series = [
    ('SMs active', 'TPC.TriageCompute.sm__cycles_active.avg.per_cycle_elapsed', 100),
    ('Tensor active', 'TPC.TriageCompute.sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed', 1),
    ('DRAM throughput', 'FBSP.TriageCompute.dramc__throughput.avg.pct_of_peak_sustained_elapsed', 1),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex='row', sharey=True)
for row, stage in enumerate(['qkv', 'oproj']):
    for col, impl in enumerate(['baseline', 'candidate']):
        ax = axes[row, col]
        data = json.loads((HERE / f'pm_{impl}_{stage}.json').read_text())
        for label, key, scale in series:
            points = data[key]['samples']
            ax.plot([t for t, _ in points], [v * scale for _, v in points],
                    marker='.', markersize=3, label=label)
        ax.set_title(f'{impl} / {stage}')
        ax.set_xlabel('Replay time (microseconds)')
        ax.set_ylabel('Percent')
        ax.set_ylim(0, 105)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
fig.suptitle('H100 BF16 GEMM: round 2 vs swapped operands + two consumer warpgroups (1.5 us samples)')
fig.tight_layout()
fig.savefig(HERE / 'pm_utilization.png', dpi=180)
