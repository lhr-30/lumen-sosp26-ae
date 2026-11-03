"""Independent correctness and CUDA-graph timing for the agent's AveLang mixer."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

from input_model import B, T, H, KV, D, HIDDEN, PACKED, reference, rope_tables


def make_inputs(seed):
    # Same allocation order/distributions as the authoritative bench.build().
    torch.manual_seed(seed)
    bf16 = dict(dtype=torch.bfloat16, device='cuda')
    w = {'qkv': torch.randn(PACKED, HIDDEN, **bf16) * 0.02,
         'o': torch.randn(HIDDEN, HIDDEN, **bf16) * 0.02,
         'q_norm': torch.randn(D, **bf16) * 0.1,
         'k_norm': torch.randn(D, **bf16) * 0.1}
    x = torch.randn(B, T, HIDDEN, **bf16)
    hidden = torch.randn(B, T, HIDDEN, **bf16)
    return x, hidden, w, rope_tables()


def copy_inputs(dst, src):
    dst[0].copy_(src[0])
    dst[1].copy_(src[1])
    for name in dst[2]:
        dst[2][name].copy_(src[2][name])
    dst[3].copy_(src[3])


def compare(actual, expected, label):
    result = {}
    for name, a, e in zip(('attended', 'output'), actual, expected, strict=True):
        assert a.shape == e.shape and a.dtype == e.dtype, (name, a.shape, a.dtype)
        assert torch.isfinite(a).all().item(), f'{label}/{name}: nonfinite output'
        av, ev = a.double().flatten(), e.double().flatten()
        error = av - ev
        en = ev.norm().item()
        cosine = F.cosine_similarity(av, ev, dim=0).item() if en else None
        relative_l2 = error.norm().item() / en if en else None
        row = {'cosine': cosine, 'relative_l2': relative_l2,
               'max_abs': error.abs().max().item(), 'mean_abs': error.abs().mean().item()}
        result[name] = row
        print(f'{label}/{name}: {row}', flush=True)
        if en:
            assert cosine >= 0.9999, f'{label}/{name}: cosine {cosine} < 0.9999'
            assert relative_l2 <= 0.02, f'{label}/{name}: relative L2 {relative_l2} > 0.02'
        else:
            assert row['max_abs'] == 0, f'{label}/{name}: expected all zeros'
    return result


def capture(fn, iterations):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            for _ in range(iterations):
                fn()
    torch.cuda.current_stream().wait_stream(side)
    return graph


def validate(inputs, kernels, outputs, smoke):
    def chain():
        for fn in kernels.values():
            fn()

    records = []
    expected = reference(*inputs)
    chain()
    torch.cuda.synchronize()
    records.append({'case': 'seed0', 'metrics': compare(outputs, expected, 'seed0')})
    if smoke:
        return records

    # Captured calls must read fresh values, including changed weights/tables.
    graph = capture(chain, 1)
    for seed in (17, 123):
        fresh = make_inputs(seed)
        if seed == 123:
            fresh[3].copy_(fresh[3].flip(0))
        copy_inputs(inputs, fresh)
        expected = reference(*fresh)
        graph.replay()
        torch.cuda.synchronize()
        label = f'graph_seed{seed}'
        records.append({'case': label, 'metrics': compare(outputs, expected, label)})

    # Modifying future tokens in one batch must not affect earlier tokens or
    # other batches. The full independent reference also verifies the suffix.
    before = tuple(o.clone() for o in outputs)
    inputs[0][0, 13:].mul_(-2)
    expected = reference(*inputs)
    graph.replay()
    torch.cuda.synchronize()
    for a, old in zip(outputs, before, strict=True):
        assert torch.equal(a[0, :13], old[0, :13]), 'causal prefix changed'
        assert torch.equal(a[1:], old[1:]), 'another batch changed'
    records.append({'case': 'causal_batch_isolation',
                    'metrics': compare(outputs, expected, 'causal_batch_isolation')})

    inputs[0].zero_()
    expected = reference(*inputs)
    graph.replay()
    torch.cuda.synchronize()
    records.append({'case': 'zero_input', 'metrics': compare(outputs, expected, 'zero_input')})
    return records


def benchmark(kernels, iterations, samples):
    def chain():
        for fn in kernels.values():
            fn()

    graphs = {name: capture(fn, iterations) for name, fn in {**kernels, 'chain': chain}.items()}
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    values = {name: [] for name in graphs}
    for _ in range(samples):
        for name, graph in graphs.items():
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            values[name].append(start.elapsed_time(end) * 1000 / iterations)
    result = {}
    for name, data in values.items():
        deciles = statistics.quantiles(data, n=10)
        result[name] = {'median_us': statistics.median(data), 'p10_us': deciles[0], 'p90_us': deciles[-1]}
        print(f'{name}: {result[name]}', flush=True)
    return {'iterations': iterations, 'samples': samples, 'kernels': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('test', 'bench'), default='test')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--output')
    parser.add_argument('--candidate', default='candidate', help='module exposing build(x, hidden, w, tables)')
    parser.add_argument('--invariant-validation', choices=('enabled', 'unavailable'), default='enabled')
    args = parser.parse_args()
    if args.iterations < 1 or args.samples < 2:
        parser.error('iterations must be >=1 and samples >=2')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    candidate = importlib.import_module(args.candidate)
    inputs = make_inputs(0)
    started = time.monotonic()
    kernels, outputs = candidate.build(*inputs)
    props = torch.cuda.get_device_properties(0)
    report = {'gpu': props.name, 'sm_count': props.multi_processor_count,
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'shape': {'B': B, 'T': T, 'H': H, 'KV': KV, 'D': D, 'hidden': HIDDEN},
              'candidate_sha256': hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
              'invariant_validation': args.invariant_validation == 'enabled',
              'invariant_validation_status': args.invariant_validation, 'stages': list(kernels)}
    report['correctness'] = validate(inputs, kernels, outputs, args.smoke)
    if args.mode == 'bench':
        copy_inputs(inputs, make_inputs(0))
        for fn in kernels.values():
            fn()
        torch.cuda.synchronize()
        report['benchmark'] = benchmark(kernels, args.iterations, args.samples)
    report['wall_seconds'] = time.monotonic() - started
    path = Path(args.output or Path(os.environ.get('ATTN_RESULTS', '.')) / f'{args.mode}.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: {path}', flush=True)


if __name__ == '__main__':
    main()
