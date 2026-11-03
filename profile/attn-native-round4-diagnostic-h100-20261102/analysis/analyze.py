"""Extract SM90 metrics, actual SASS, source stalls and PM timelines via ncu_report."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, '/opt/nvidia/nsight-compute/2026.2.1/extras/python')
import ncu_report

TAGS = {'baseline': ['qkv', 'oproj'], 'candidate': ['qkv', 'oproj']}
KEYS = '''gpu__time_duration.sum launch__grid_size launch__block_size
launch__registers_per_thread launch__shared_mem_per_block launch__waves_per_multiprocessor
launch__occupancy_limit_blocks launch__occupancy_limit_registers launch__occupancy_limit_shared_mem launch__occupancy_limit_warps
sm__maximum_warps_per_active_cycle_pct sm__warps_active.avg.pct_of_peak_sustained_active
sm__throughput.avg.pct_of_peak_sustained_elapsed sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed sm__cycles_active.avg sm__cycles_active.min sm__cycles_active.max
dram__bytes_read.sum dram__bytes_write.sum dram__bytes_read.sum.pct_of_peak_sustained_elapsed dram__bytes_write.sum.pct_of_peak_sustained_elapsed
l1tex__t_sector_hit_rate.pct lts__t_sector_hit_rate.pct
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio
smsp__inst_issued.sum smsp__inst_executed.sum
smsp__sass_inst_executed_op_global_ld.sum smsp__sass_inst_executed_op_global_st.sum
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
l1tex__t_requests_pipe_lsu_mem_global_op_st.sum l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
sass__inst_executed_register_spilling_mem_local_op_read sass__inst_executed_register_spilling_mem_local_op_write'''.split()


def at(metric, index):
    try:
        return metric.as_double(index)
    except RuntimeError:
        return metric.as_uint64(index)


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str) + '\n')


def extract_pm(action):
    names = set(action.metric_names())
    starts = sorted(n for n in names if n.startswith('profiler__timestamp_workload_start_'))
    windows = []
    for name in starts:
        endname = name.replace('start_', 'end_')
        if endname in names:
            s = action[name].correlation_ids().as_uint64(0)
            e = action[endname].correlation_ids().as_uint64(0)
            windows.append((s,e))
    result = {}
    for name in sorted(n for n in names if n.startswith('pmsampling:') or '.TriageCompute.' in n):
        metric = action[name]
        if not metric.has_correlation_ids():
            continue
        ids = metric.correlation_ids()
        series = [(ids.as_uint64(i),at(metric,i)) for i in range(metric.num_instances())]
        if not windows:
            continue
        start,end = max(windows,key=lambda se:sum(v>0 for t,v in series if se[0]<=t<=se[1]))
        points = [((t-start)/1000,v) for t,v in series if start<=t<=end]
        result[name] = {'duration_us':(end-start)/1000,'samples':points,
                        'unit':metric.unit(),
                        'sampling_interval_ns':action['profiler__pmsampler_interval_time'].value()}
    return result


def extract_source(action):
    names = set(action.metric_names())
    per_pc = defaultdict(dict)
    summaries = {}
    total = action['smsp__pcsamp_sample_count'].value()
    for name in sorted(names):
        if not name.startswith('smsp__pcsamp_warps_issue_stalled_') or name.endswith('_not_issued'):
            continue
        metric = action[name]
        label = name.removeprefix('smsp__pcsamp_warps_issue_stalled_')
        summaries[label] = {'samples':metric.value(), 'pct_samples':metric.value()*100/total if total else None}
        if metric.has_correlation_ids():
            ids = metric.correlation_ids()
            for i in range(metric.num_instances()):
                value = at(metric,i)
                if value:
                    per_pc[ids.as_uint64(i)][label] = value
    rows = []
    for pc,stalls in per_pc.items():
        si = action.source_info(pc)
        rows.append({'pc':hex(pc),'file':si.file_name() if si else None,'line':si.line() if si else None,
                     'sass':action.sass_by_pc(pc),'ptx':action.ptx_by_pc(pc),
                     'samples':sum(stalls.values()),'stalls':stalls})
    rows.sort(key=lambda r:-r['samples'])
    return {'sample_count':total,'stall_distribution':summaries,'hotspots':rows}


def extract_sass_csv(path, tags, outdir, impl):
    if not path.exists():
        return
    groups=[]
    for row in csv.reader(path.open()):
        if row and row[0]=='Kernel Name':
            groups.append({'name':row[1],'instructions':[]})
        elif row and row[0].startswith('0x'):
            groups[-1]['instructions'].append((int(row[0],16),row[1]))
    assert len(groups)==len(tags)
    for tag,group in zip(tags,groups,strict=True):
        counts=Counter()
        lines=[]
        origin=group['instructions'][0][0]
        for pc,inst in group['instructions']:
            tokens=inst.split()
            opcode=tokens[1] if tokens[0].startswith('@') else tokens[0]
            counts[opcode]+=1
            lines.append(f'{pc-origin:06x}  {inst}')
        (outdir/f'{impl}_{tag}.sass').write_text('\n'.join(lines)+'\n')
        dump(outdir/f'{impl}_{tag}_static_opcodes.json',counts)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--stage',default='all')
    args=parser.parse_args(); out=args.run_dir/'analysis'
    combined={}; warnings=[]
    for impl in ('baseline','candidate'):
        validation=json.loads((args.run_dir/'analysis'/f'full_{impl}_validation.json').read_text())
        tags=[tag for name,tag in validation['profiled_stages'] if name==impl]
        paths={kind:args.run_dir/'reports'/f'{kind}_{impl}.ncu-rep' for kind in ('full','source')}
        if not paths['full'].exists():
            continue
        full=ncu_report.load_report(str(paths['full'])).range_by_idx(0)
        source=ncu_report.load_report(str(paths['source'])).range_by_idx(0)
        assert full.num_actions()==source.num_actions()==len(tags)
        for i,tag in enumerate(tags):
            a=full.action_by_idx(i); s=source.action_by_idx(i); names=set(a.metric_names()); prefix=impl+'_'+tag
            metrics={}
            for name in sorted(names):
                metric=a[name]
                try:
                    metrics[name]={'value':metric.value(),'unit':metric.unit()}
                except RuntimeError as error:
                    metrics[name]={'error':str(error)}
            dump(out/f'metrics_all_{prefix}.json',metrics)
            key={name:metrics.get(name,{}).get('value') for name in KEYS}
            for name in names:
                if name.startswith('smsp__average_warps_issue_stalled_') and name.endswith('.ratio'):
                    key[name]=metrics[name].get('value')
            for name in ('l1tex__t_sector_hit_rate.pct','lts__t_sector_hit_rate.pct'):
                value=key[name]
                if value is not None and not 0<=value<=100:
                    warnings.append(f'{prefix}: {name}={value}; uncontrolled-cache replay ratio, do not interpret as physical hit rate')
            dynamic={}
            for name in ('sass__inst_executed_per_opcode','sass__inst_executed_per_opcode_with_modifier_all'):
                m=a[name]; ids=m.correlation_ids()
                dynamic[name]={ids.as_string(j):at(m,j) for j in range(m.num_instances())}
            dump(out/f'opcodes_{prefix}.json',dynamic)
            src=extract_source(s); dump(out/f'stalls_{prefix}.json',src)
            dump(out/f'rules_{prefix}.json',a.rule_results_as_dicts())
            dump(out/f'pm_{prefix}.json',extract_pm(a))
            dump(out/f'metrics_key_{prefix}.json',key)
            combined[prefix]={'kernel':a.name(),'metrics':key,'stall_distribution':src['stall_distribution'],
                              'dynamic_opcodes':dynamic['sass__inst_executed_per_opcode_with_modifier_all']}
        extract_sass_csv(out/f'sass_{impl}.csv',tags,out,impl)
    dump(out/'comparison.json',combined); dump(out/'measurement_warnings.json',warnings)
    text=['impl/stage  NCU us  grid block regs smem  tensor% occ% DRAMread% storeB/sector']
    for name,row in combined.items():
        m=row['metrics']; text.append(f"{name:16s} {m['gpu__time_duration.sum']/1000:7.3f} {m['launch__grid_size']:4g} {m['launch__block_size']:3g} {m['launch__registers_per_thread']:3g} {m['launch__shared_mem_per_block']:6g} {m['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed']:7.2f} {m['sm__warps_active.avg.pct_of_peak_sustained_active']:6.2f} {m['dram__bytes_read.sum.pct_of_peak_sustained_elapsed']:7.2f} {m['smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio']:6.2f}")
    (out/'comparison.txt').write_text('\n'.join(text)+'\n')
    print('\n'.join(text)); print('Measurement warnings:',len(warnings))


if __name__=='__main__':
    main()
