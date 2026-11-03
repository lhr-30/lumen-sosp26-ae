"""Independent matched graph timing of frozen native-harness round outputs."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / 'sources'), str(HERE / 'sources/cute')]
from evaluate import capture, compare, make_inputs
from input_model import reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modules', nargs='+', required=True)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    cute = importlib.import_module('bench')
    kernels, inputs, outputs = cute.build()
    implementations = {'cute': (kernels, outputs)}
    hashes = {}
    for name in args.modules:
        module = importlib.import_module(name)
        hashes[name] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        implementations[name] = module.build(*inputs)
    expected = reference(*inputs)
    checks = {}
    graphs = {}
    for name, (kernels, outputs) in implementations.items():
        def chain(kernels=kernels):
            for fn in kernels.values():
                fn()
        chain()
        torch.cuda.synchronize()
        checks[name] = compare(outputs, expected, name)
        graphs[name + '/chain'] = capture(chain, args.iterations)
        middle_names = ('norm_rope', 'attention_gate')
        if name == 'manual':
            middle_names = ('k_norm_rope_packed', 'q_norm_rope_attn_gate')
        if all(key in kernels for key in middle_names):
            middle_functions = [kernels[key] for key in middle_names]
            def middle(functions=middle_functions):
                for fn in functions:
                    fn()
            graphs[name + '/middle'] = capture(middle, args.iterations)
        for stage, fn in kernels.items():
            graphs[name + '/' + stage] = capture(fn, args.iterations)
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    values = {key: [] for key in graphs}
    rng = random.Random(0)
    measurement_start = time.time()
    for _ in range(args.samples):
        order = list(graphs)
        rng.shuffle(order)
        for key in order:
            start.record()
            graphs[key].replay()
            end.record()
            end.synchronize()
            values[key].append(start.elapsed_time(end) * 1000 / args.iterations)
    times = {}
    for key, samples in values.items():
        deciles = statistics.quantiles(samples, n=10)
        times[key] = {'median': statistics.median(samples), 'p10': deciles[0],
                      'p90': deciles[-1], 'samples': samples}
        print(key, times[key]['median'], flush=True)
    args.output.write_text(json.dumps({
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'shape': {'B': 16, 'T': 26, 'H': 8, 'KV': 2, 'D': 256, 'hidden': 2048},
        'method': {'iterations_per_graph': args.iterations, 'samples': args.samples,
                   'randomized_order_seed': 0},
        'measurement_window_unix': [measurement_start, time.time()],
        'source_sha256': hashes, 'checks': checks, 'timing_us': times,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
