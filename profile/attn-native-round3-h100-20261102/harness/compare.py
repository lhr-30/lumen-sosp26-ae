"""Frozen baseline / agent schedule / CuTe comparison on the same inputs."""
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
            if hasattr(module, 'bind_projection'):
                module.bind_projection(x.view(416, 2048), weights[key], output, n)()
            else:
                module.linear_native[lambda: ((n // 128, 7, 1), (160, 1, 1))](
                    x.view(416, 2048), weights[key], output, n, num_warps=5)
        torch.cuda.synchronize()
        assert torch.isfinite(actual).all().item(), f'N={n}: unwritten/nonfinite output'
        assert (storage[416 * n:] == 23).all().item(), f'N={n}: OOB output modified'
        reference = (x.view(416, 2048).float() @ weights[key].float().T).bfloat16()
        av, rv = actual.double().flatten(), reference.double().flatten()
        relative_l2 = ((av - rv).norm() / rv.norm()).item()
        cosine = torch.nn.functional.cosine_similarity(av, rv, dim=0).item()
        assert cosine >= 0.9999 and relative_l2 <= 0.02, (n, cosine, relative_l2)
        result[str(n)] = {'bitwise_equal_to_baseline': torch.equal(actual, expected),
                          'reference_cosine': cosine, 'reference_relative_l2': relative_l2,
                          'tail_rows_checked': 32, 'guard_elements_unchanged': 32 * n}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['check', 'timing', 'profile'], default='check')
    parser.add_argument('--impl', choices=['baseline', 'candidate', 'cute', 'all'], default='all')
    parser.add_argument('--stage', choices=['all', 'gemms', 'qkv', 'oproj'], default='all')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--compare-o', action='store_true', help='also time the alternate dual-WG OProj chain')
    parser.add_argument('--candidate-module', default='round3_candidate')
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
    for name, module in [('baseline', 'baseline_candidate'), ('candidate', args.candidate_module)]:
        if args.impl in (name, 'all'):
            modules[name] = importlib.import_module(module)
            implementations[name] = modules[name].build(*inputs)
    if args.compare_o:
        assert args.impl == 'all', '--compare-o requires --impl all'
        module = modules['candidate']
        saved = module.DEFAULT_PROJECTION_CONFIGS[2048]
        try:
            module.DEFAULT_PROJECTION_CONFIGS[2048] = 'o_t64_c128_w1'
            implementations['candidate_dual_o'] = module.build(*inputs)
        finally:
            module.DEFAULT_PROJECTION_CONFIGS[2048] = saved
    report = {
        'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
        'torch': torch.__version__,
        'shape': {'B': 16, 'T': 26, 'H': 8, 'KV': 2, 'D': 256, 'hidden': 2048},
        'source_sha256': {name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                          for name, m in modules.items()},
        'checks': {},
    }
    if 'candidate' in modules:
        report['candidate_configs'] = dict(modules['candidate'].DEFAULT_PROJECTION_CONFIGS)
    if args.compare_o:
        report['candidate_dual_o_configs'] = {5120: 'qkv_t160_c128_w1', 2048: 'o_t64_c128_w1'}
    expected = reference(*inputs)
    for name, (kernels, outputs) in implementations.items():
        for _ in range(3):
            for fn in kernels.values():
                fn()
        torch.cuda.synchronize()
        report['checks'][name] = compare(outputs, expected, name)
    if args.impl == 'all':
        report['projection_checks'] = projection_check(modules['baseline'], modules['candidate'], inputs)
        report['native_chain_bitwise_equal'] = all(
            torch.equal(a, b) for a, b in zip(implementations['baseline'][1],
                                             implementations['candidate'][1], strict=True))
    tags = {'cute': ['qkv', 'knorm', 'attention', 'oproj', 'residual'],
            'baseline': ['qkv', 'unpack', 'qnorm', 'knorm', 'attention', 'oproj', 'residual'],
            'candidate': ['qkv', 'unpack', 'qnorm', 'knorm', 'attention', 'oproj', 'residual']}
    tags['candidate_dual_o'] = tags['candidate']
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
