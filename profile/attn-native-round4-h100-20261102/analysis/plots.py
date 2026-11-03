"""Plot sampled replay timelines; the candidate includes Q norm and RoPE."""
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
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, impl, title in zip(axes, ['baseline', 'candidate'],
        ['Round 3: attention + gate', 'Round 4: Q norm/RoPE + attention + gate'], strict=True):
    data = json.loads((HERE / f'pm_{impl}_attention.json').read_text())
    for label, key, scale in series:
        points = data[key]['samples']
        ax.plot([t for t, _ in points], [v * scale for _, v in points],
                marker='.', markersize=3, label=label)
    ax.set_title(title)
    ax.set_xlabel('Replay time (microseconds)')
    ax.set_ylabel('Percent')
    ax.set_ylim(0, 105)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle('H100 B16/T26: sampled NCU replay (1.5 us samples; different fused work)')
fig.tight_layout()
fig.savefig(HERE / 'pm_utilization.png', dpi=180)
