"""Bounded, correctness-gated GEMM configuration comparison on H100."""
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
sys.path[:0] = [str(HERE / 'sources/ave')]
import torch
from evaluate import capture, make_inputs


def check(actual, reference, storage):
    assert torch.isfinite(actual).all().item(), 'Unwritten/nonfinite output'
    assert (storage[actual.numel():] == 23).all().item(), 'Output guard modified'
    av, rv = actual.double().flatten(), reference.double().flatten()
    relative_l2 = ((av - rv).norm() / rv.norm()).item()
    cosine = torch.nn.functional.cosine_similarity(av, rv, dim=0).item()
    assert cosine >= 0.9999 and relative_l2 <= 0.02, (cosine, relative_l2)
    return {'cosine': cosine, 'relative_l2': relative_l2,
            'max_abs': (av - rv).abs().max().item(), 'guard_unchanged': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=HERE / 'cases.json')
    parser.add_argument('--candidate-module', default='round3_sweep_candidate')
    parser.add_argument('--output', type=Path, default=RUN / 'analysis/sweep.json')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--samples', type=int, default=20)
    args = parser.parse_args()
    assert args.iterations >= 1 and args.samples >= 2
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    base = importlib.import_module('baseline_candidate')
    candidate = importlib.import_module(args.candidate_module)
    cases = json.loads(args.cases.read_text())
    inputs = make_inputs(0)
    x, _, weights, _ = inputs
    functions, outputs, guards, references = {}, {}, {}, {}
    for n, key in [(5120, 'qkv'), (2048, 'o')]:
        weight = weights[key]
        references[n] = (x.view(416, 2048).float() @ weight.float().T).bfloat16()
        relevant = [{'name': f'baseline_{n}', 'N': n, 'config': None, 'baseline': True}]
        relevant += [case for case in cases if case['N'] == n]
        for case in relevant:
            name = case['name']
            storage = torch.full((512 * n,), 23, dtype=x.dtype, device=x.device)
            output = storage[:416 * n].view(416, n)
            output.fill_(float('nan'))
            guards[name], outputs[name] = storage, output
            if case.get('baseline'):
                def run(n=n, output=output, weight=weight):
                    base.linear_native[lambda: ((n // 128, 7, 1), (160, 1, 1))](
                        x.view(416, 2048), weight, output, n, num_warps=5)
            else:
                run = candidate.bind_projection(x.view(416, 2048), weight, output, n,
                                                config=case['config'])
            functions[name] = (n, run)
    report = {'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
              'torch': torch.__version__, 'shape': {'M': 416, 'K': 2048, 'N': [5120, 2048]},
              'cases': cases, 'checks': {},
              'candidate_sha256': hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
              'baseline_sha256': hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()}
    for name, (n, fn) in functions.items():
        fn()
        torch.cuda.synchronize()
        report['checks'][name] = check(outputs[name], references[n], guards[name])
        report['checks'][name]['bitwise_equal_to_baseline'] = torch.equal(
            outputs[name], outputs[f'baseline_{n}'])
        print('CHECK', name, report['checks'][name], flush=True)
    graphs = {name: capture(fn, args.iterations) for name, (_, fn) in functions.items()}
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
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS', args.output, flush=True)


if __name__ == '__main__':
    main()
