"""Frozen baseline / agent epilogue / CuTe comparison on the same inputs."""
import argparse
import hashlib
import importlib
import json
import random
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path[:0] = [str(HERE / 'sources/ave'), str(HERE / 'sources/cute')]

import torch
from evaluate import capture, compare, make_inputs
from input_model import reference


def projection_check(baseline, candidate, inputs):
    """Check every output, especially the final 32 rows and the OOB guard."""
    result = {}
    x, _, weights, _ = inputs
    for n, key in [(5120, 'qkv'), (2048, 'o')]:
        expected = torch.empty((416, n), dtype=x.dtype, device=x.device)
        storage = torch.full((448 * n,), 23, dtype=x.dtype, device=x.device)
        actual = storage[:416 * n].view(416, n)
        actual.fill_(float('nan'))
        for module, output in [(baseline, expected), (candidate, actual)]:
            module.linear_native[lambda: ((n // 128, 7, 1), (160, 1, 1))](
                x.view(416, 2048), weights[key], output, n, num_warps=5)
        torch.cuda.synchronize()
        assert torch.isfinite(actual).all().item(), f'N={n}: unwritten/nonfinite output'
        assert torch.equal(actual, expected), f'N={n}: projection differs from baseline'
        assert (storage[416 * n:] == 23).all().item(), f'N={n}: OOB output modified'
        result[str(n)] = {'bitwise_equal': True, 'tail_rows_checked': 32,
                          'guard_elements_unchanged': 32 * n}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['check', 'timing', 'profile'], default='check')
    parser.add_argument('--impl', choices=['baseline', 'epilogue', 'cute', 'all'], default='all')
    parser.add_argument('--stage', choices=['all', 'gemms', 'qkv', 'oproj'], default='all')
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
    implementations = {}
    modules = {}
    if args.impl in ('cute', 'all'):
        cute = importlib.import_module('bench')
        kernels, inputs, outputs = cute.build()
        implementations['cute'] = kernels, outputs
    else:
        inputs = make_inputs(0)
    for name, module in [('baseline', 'baseline_candidate'), ('epilogue', 'round2_candidate')]:
        if args.impl in (name, 'all'):
            modules[name] = importlib.import_module(module)
            implementations[name] = modules[name].build(*inputs)
    report = {
        'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
        'torch': torch.__version__,
        'shape': {'B': 16, 'T': 26, 'H': 8, 'KV': 2, 'D': 256, 'hidden': 2048},
        'source_sha256': {name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                          for name, m in modules.items()},
        'checks': {},
    }
    expected = reference(*inputs)
    for name, (kernels, outputs) in implementations.items():
        for _ in range(3):
            for fn in kernels.values():
                fn()
        torch.cuda.synchronize()
        report['checks'][name] = compare(outputs, expected, name)
    if args.impl == 'all':
        report['projection_checks'] = projection_check(modules['baseline'], modules['epilogue'], inputs)
        report['native_chain_bitwise_equal'] = all(
            torch.equal(a, b) for a, b in zip(implementations['baseline'][1],
                                             implementations['epilogue'][1], strict=True))
        assert report['native_chain_bitwise_equal']
    tags = {'cute': ['qkv', 'knorm', 'attention', 'oproj', 'residual'],
            'baseline': ['qkv', 'unpack', 'qnorm', 'knorm', 'attention', 'oproj', 'residual'],
            'epilogue': ['qkv', 'unpack', 'qnorm', 'knorm', 'attention', 'oproj', 'residual']}
    ordered = [(name, tag, fn) for name, (kernels, _) in implementations.items()
               for tag, fn in zip(tags[name], kernels.values(), strict=True)]
    if args.mode == 'timing':
        graphs = {}
        for name, (kernels, _) in implementations.items():
            def chain(kernels=kernels):
                for fn in kernels.values():
                    fn()
            graphs[name + '/chain'] = capture(chain, args.iterations)
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
        report['timing_method'] = {'iterations_per_graph': args.iterations,
                                   'samples': args.samples, 'randomized_order_seed': 0}
        report['timing_us'] = {
            name: {'median': statistics.median(v),
                   'p10': statistics.quantiles(v, n=10)[0],
                   'p90': statistics.quantiles(v, n=10)[-1], 'samples': v}
            for name, v in values.items()}
        for name, row in report['timing_us'].items():
            print(name, {k: v for k, v in row.items() if k != 'samples'}, flush=True)
    if args.mode == 'profile':
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for name, tag, fn in ordered:
            if args.stage in ('all', tag) or (args.stage == 'gemms' and tag in ('qkv', 'oproj')):
                torch.cuda.nvtx.range_push(name + '/' + tag)
                fn()
                torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    path = Path(args.output) if args.output else RUN / 'analysis' / f'{args.mode}_{args.impl}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS', path, flush=True)


if __name__ == '__main__':
    main()
