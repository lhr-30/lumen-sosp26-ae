"""Profile the exact B16/T26 mixer implementations with setup outside capture."""
import argparse
import importlib
import json
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path[:0] = [str(HERE/'sources/ave'), str(HERE/'sources/cute')]
os.environ['CUTE_DSL_LINEINFO'] = '1'
os.environ['CUTE_DSL_KEEP'] = 'ptx,cubin'
os.environ['CUTE_DSL_DUMP_DIR'] = str(HERE/'generated/cute')

import torch
from evaluate import compare, capture
from input_model import reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--impl', choices=('ave','cute','both'), default='both')
    parser.add_argument('--mode', choices=('check','timing','profile'), default='check')
    parser.add_argument('--stage', default='all')
    parser.add_argument('--output')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    baseline = importlib.import_module('bench')
    cute_kernels, inputs, cute_outputs = baseline.build()
    implementations = {'cute': (cute_kernels, cute_outputs)}
    if args.impl in ('ave','both'):
        from native_lineinfo import install
        install(HERE/'generated/ave')
        candidate = importlib.import_module('native_candidate')
        implementations['ave'] = candidate.build(*inputs)
    selected = list(implementations) if args.impl == 'both' else [args.impl]
    expected = reference(*inputs)
    checks = {}
    for impl in selected:
        kernels, outputs = implementations[impl]
        for _ in range(3):
            for fn in kernels.values():
                fn()
        torch.cuda.synchronize()
        checks[impl] = compare(outputs, expected, impl)
    tags = {'ave':['qkv','unpack','qnorm','knorm','attention','oproj','residual'],
            'cute':['qkv','knorm','attention','oproj','residual']}
    ordered = [(impl,tag,fn) for impl in selected
               for tag,fn in zip(tags[impl], implementations[impl][0].values(), strict=True)]
    output = {'checks':checks,'stages':[(i,t) for i,t,_ in ordered],
              'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'torch':torch.__version__,
              'shape':{'B':16,'T':26,'H':8,'KV':2,'D':256,'hidden':2048},
              'line_info':{'cute':'CUTE_DSL_LINEINFO=1',
                           'ave':'unchanged PTX instructions + .loc/.file, CUDA driver CU_JIT_GENERATE_LINE_INFO'}}
    if args.mode == 'timing':
        graphs = {}
        for impl in selected:
            kernels = implementations[impl][0]
            def chain(kernels=kernels):
                for fn in kernels.values():
                    fn()
            graphs[impl+'/chain'] = capture(chain, 100)
        for impl,tag,fn in ordered:
            graphs[impl+'/'+tag] = capture(fn, 100)
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        values = {name:[] for name in graphs}
        start,end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        for sample in range(20):
            for name,graph in graphs.items():
                start.record(); graph.replay(); end.record(); end.synchronize()
                values[name].append(start.elapsed_time(end)*10)
        output['timing_us']={name:{'median':statistics.median(v),
                                  'p10':statistics.quantiles(v,n=10)[0],
                                  'p90':statistics.quantiles(v,n=10)[-1]}
                             for name,v in values.items()}
        for name,value in output['timing_us'].items():
            print(name,value,flush=True)
    if args.mode == 'profile':
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for impl,tag,fn in ordered:
            if args.stage not in ('all',tag):
                continue
            torch.cuda.nvtx.range_push(impl+'/'+tag)
            fn()
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    path = Path(args.output) if args.output else RUN/'analysis'/f'{args.mode}_{args.impl}.json'
    path.write_text(json.dumps(output,indent=2)+'\n')
    print('PASS',path,flush=True)


if __name__ == '__main__':
    main()
