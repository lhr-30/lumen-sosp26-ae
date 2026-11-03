"""Standalone figures from measured CUDA-graph and NCU results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
timing=json.loads((HERE/'timing_both.json').read_text())['timing_us']
metrics=json.loads((HERE/'comparison.json').read_text())
groups=[('QKV','qkv'),('Unpack','unpack'),('K norm/RoPE','knorm'),
        ('Q norm + attention/gate','attention'),('Output projection','oproj'),('Residual','residual')]
ave=[];cute=[]
for name,key in groups:
    a=timing['ave/'+key]['median']
    if key=='attention':a+=timing['ave/qnorm']['median']
    ave.append(a);cute.append(timing.get('cute/'+key,{'median':0})['median'])
fig,axes=plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={'width_ratios':[1.6,1]})
y=np.arange(len(groups));ax=axes[0]
ax.barh(y-.18,ave,.34,label='Generated AveLang',color='#d27331')
ax.barh(y+.18,cute,.34,label='Original CuTe',color='#337ab7')
ax.set_yticks(y,[x[0] for x in groups]);ax.invert_yaxis();ax.set_xlabel('Standalone stage time (microseconds)')
ax.set_title('CUDA graphs: 100 calls x 20 samples');ax.legend(fontsize=9)
for ys,vs in [(y-.18,ave),(y+.18,cute)]:
    for pos,value in zip(ys,vs):ax.text(value+.3,pos,f'{value:.1f}',va='center',fontsize=8)
ax.set_xlim(0,max(ave)*1.15)
ax=axes[1];y=np.arange(2)
key='sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'
av=[metrics['ave_'+t]['metrics'][key] for t in ('qkv','oproj')]
cv=[metrics['cute_'+t]['metrics'][key] for t in ('qkv','oproj')]
ax.bar(y-.18,av,.34,color='#d27331');ax.bar(y+.18,cv,.34,color='#337ab7')
ax.set_xticks(y,['QKV','Output projection']);ax.set_ylim(0,100)
ax.set_ylabel('Tensor pipe active (% of elapsed cycles)');ax.set_title('Nsight Compute, base clock')
for xs,vs in [(y-.18,av),(y+.18,cv)]:
    for x,value in zip(xs,vs):ax.text(x,value+1,f'{value:.1f}%',ha='center',fontsize=9)
fig.suptitle('H100 NVL: B=16, T=26, H=8, KV=2, D=256',fontweight='bold')
fig.tight_layout();fig.savefig(HERE/'comparison.png',dpi=180);fig.savefig(HERE/'comparison.svg');plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(11,6),sharey=True)
colors={'long_scoreboard':'#d27331','barrier':'#337ab7','wait':'#22805d'}
for ax,(impl,stage) in zip(axes.flat,[('ave','qkv'),('cute','qkv'),('ave','oproj'),('cute','oproj')]):
    pm=json.loads((HERE/f'pm_{impl}_{stage}.json').read_text())
    for label,color in colors.items():
        series=pm['pmsampling:smsp__warps_issue_stalled_'+label+'.avg']['samples']
        if not series:continue
        data=np.asarray(series); peak=data[:,1].max()
        if peak:ax.plot(data[:,0],data[:,1]/peak,label=label,color=color,marker='.',markersize=3)
    ax.set_title(impl.upper()+' / '+stage);ax.set_xlabel('Time from replay start (microseconds)');ax.set_ylabel('Fraction of this curve\'s peak')
    ax.legend(fontsize=8);ax.set_ylim(0,1.08);ax.grid(alpha=.2)
fig.suptitle('PM stall timelines (1.5 us sampling; each metric may use a different replay)')
fig.tight_layout();fig.savefig(HERE/'pm_timelines.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(11,6),sharey=True)
series_names=[('SMs active','TPC.TriageCompute.sm__cycles_active.avg.per_cycle_elapsed',100),
              ('Tensor active','TPC.TriageCompute.sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed',1),
              ('DRAM throughput','FBSP.TriageCompute.dramc__throughput.avg.pct_of_peak_sustained_elapsed',1)]
for ax,(impl,stage) in zip(axes.flat,[('ave','qkv'),('cute','qkv'),('ave','oproj'),('cute','oproj')]):
    pm=json.loads((HERE/f'pm_{impl}_{stage}.json').read_text())
    for label,key,scale in series_names:
        series=pm[key]['samples']
        if series:
            data=np.asarray(series);ax.plot(data[:,0],data[:,1]*scale,label=label,marker='.',markersize=3)
    ax.set_title(impl.upper()+' / '+stage);ax.set_xlabel('Time from replay start (microseconds)')
    ax.set_ylabel('Percent');ax.set_ylim(0,105);ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('SM90 PM utilization timelines (1.5 us sampling)')
fig.tight_layout();fig.savefig(HERE/'pm_utilization.png',dpi=180);plt.close(fig)
