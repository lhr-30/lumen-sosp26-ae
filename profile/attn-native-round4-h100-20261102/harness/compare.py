"""Round-3 / round-4 / CuTe timing with frozen sources and unchanged reference."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import random
import statistics
import sys

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path[:0] = [str(HERE / 'sources/ave'), str(HERE / 'sources/cute')]
import torch
from evaluate import capture, compare, make_inputs
from input_model import reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['check', 'timing', 'profile'], default='check')
    parser.add_argument('--impl', choices=['baseline', 'candidate', 'cute', 'all'], default='all')
    parser.add_argument('--stage', choices=['all', 'middle', 'attention', 'knorm'], default='all')
    parser.add_argument('--candidate-module', default='round4_async_candidate')
    parser.add_argument('--extra-candidate', help='one frozen alternate candidate, reported as shared')
    parser.add_argument('--extra-mma-module', help='frozen scalar-preload MMA control, reported as mma_raw')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--output')
    args = parser.parse_args()
    assert args.iterations >= 1 and args.samples >= 2
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    from native_lineinfo import install
    install(HERE / 'generated/ave')
    implementations, modules = {}, {}
    tags = {'cute': ['qkv', 'knorm', 'attention', 'oproj', 'residual'],
            'baseline': ['qkv', 'unpack', 'qnorm', 'knorm', 'attention', 'oproj', 'residual']}
    if args.impl in ('cute', 'all'):
        cute = importlib.import_module('bench')
        kernels, inputs, outputs = cute.build()
        implementations['cute'] = kernels, outputs
    else:
        inputs = make_inputs(0)
    candidate_modules = [('baseline', 'baseline_candidate'), ('candidate', args.candidate_module)]
    if args.extra_candidate:
        assert args.impl == 'all', '--extra-candidate requires --impl all'
        candidate_modules.append(('shared', args.extra_candidate))
    if args.extra_mma_module:
        assert args.impl == 'all', '--extra-mma-module requires --impl all'
        candidate_modules.append(('mma_raw', args.extra_mma_module))
    for name, module in candidate_modules:
        if args.impl in (name, 'all'):
            modules[name] = importlib.import_module(module)
            implementations[name] = modules[name].build(*inputs)
            if hasattr(modules[name], 'PROFILE_STAGE_TAGS'):
                tags[name] = modules[name].PROFILE_STAGE_TAGS
    report = {
        'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda, 'torch': torch.__version__,
        'shape': {'B': 16, 'T': 26, 'H': 8, 'KV': 2, 'D': 256, 'hidden': 2048},
        'source_sha256': {name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                          for name, m in modules.items()},
        'checks': {}, 'stage_tags': {name: tags[name] for name in implementations},
    }
    expected = reference(*inputs)
    for name, (kernels, outputs) in implementations.items():
        for _ in range(3):
            for fn in kernels.values():
                fn()
        torch.cuda.synchronize()
        report['checks'][name] = compare(outputs, expected, name)
    if args.impl == 'all':
        report['native_chain_bitwise_equal'] = all(
            torch.equal(a, b) for a, b in zip(implementations['baseline'][1],
                                             implementations['candidate'][1], strict=True))
    if args.extra_mma_module:
        report['async_vs_mma_bitwise_equal'] = all(
            torch.equal(a, b) for a, b in zip(implementations['mma_raw'][1],
                                             implementations['candidate'][1], strict=True))
        assert report['async_vs_mma_bitwise_equal'], 'Preload-only change altered output bits'
    ordered = [(name, tag, fn) for name, (kernels, _) in implementations.items()
               for tag, fn in zip(tags[name], kernels.values(), strict=True)]
    report['profiled_stages'] = [(name, tag) for name, tag, _ in ordered
        if args.stage in ('all', tag) or (args.stage == 'middle' and tag not in ('qkv', 'oproj', 'residual'))]
    if args.mode == 'timing':
        graphs = {}
        for name, (kernels, _) in implementations.items():
            def chain(kernels=kernels):
                for fn in kernels.values():
                    fn()
            middle_functions = [fn for impl, tag, fn in ordered
                                if impl == name and tag not in ('qkv', 'oproj', 'residual')]
            def middle(fns=middle_functions):
                for fn in fns:
                    fn()
            graphs[name + '/chain'] = capture(chain, args.iterations)
            graphs[name + '/middle'] = capture(middle, args.iterations)
        for name, tag, fn in ordered:
            graphs[name + '/' + tag] = capture(fn, args.iterations)
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        values = {name: [] for name in graphs}
        rng = random.Random(0)
        for _ in range(args.samples):
            order = list(graphs)
            rng.shuffle(order)
            for name in order:
                start.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                values[name].append(start.elapsed_time(end) * 1000 / args.iterations)
        report['timing_method'] = {'iterations_per_graph': args.iterations, 'samples': args.samples,
                                   'randomized_order_seed': 0}
        report['timing_us'] = {
            name: {'median': statistics.median(v), 'p10': statistics.quantiles(v, n=10)[0],
                   'p90': statistics.quantiles(v, n=10)[-1], 'samples': v}
            for name, v in values.items()}
        for name, row in report['timing_us'].items():
            print(name, {k: v for k, v in row.items() if k != 'samples'}, flush=True)
    if args.mode == 'profile':
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for name, tag, fn in ordered:
            if (name, tag) in report['profiled_stages']:
                torch.cuda.nvtx.range_push(name + '/' + tag)
                fn()
                torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    path = Path(args.output) if args.output else RUN / 'analysis' / f'{args.mode}_{args.impl}.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS', path, flush=True)


if __name__ == '__main__':
    main()
