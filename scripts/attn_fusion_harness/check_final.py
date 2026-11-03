"""Independent unchanged full-layer checks plus allocation guards for final DSL."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'sources'))
from evaluate import make_inputs, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    candidate = importlib.import_module(args.module)
    inputs = make_inputs(0)
    original_empty, original_empty_like = torch.empty, torch.empty_like
    guarded = []

    def wrap(tensor):
        if not tensor.is_cuda or not tensor.is_contiguous():
            return tensor
        count = tensor.numel()
        # 1024 bytes of leading BF16 guard retains TMA's 128B alignment.
        storage = torch.full((512 + count + 4096,), 23, dtype=tensor.dtype, device=tensor.device)
        view = storage[512:512 + count].view(tensor.shape)
        if view.is_floating_point():
            view.fill_(float('nan'))
        guarded.append((storage, count, tuple(tensor.shape)))
        return view

    try:
        torch.empty = lambda *a, **k: wrap(original_empty(*a, **k))
        torch.empty_like = lambda *a, **k: wrap(original_empty_like(*a, **k))
        kernels, outputs = candidate.build(*inputs)
    finally:
        torch.empty, torch.empty_like = original_empty, original_empty_like
    checks = validate(inputs, kernels, outputs, False)
    torch.cuda.synchronize()
    for storage, count, shape in guarded:
        assert torch.all(storage[:512] == 23).item(), ('leading guard modified', shape)
        assert torch.all(storage[512 + count:] == 23).item(), ('trailing guard modified', shape)
    result = {'module': args.module, 'candidate_sha256': hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
              'checks': checks, 'guarded_allocations': [shape for _, _, shape in guarded],
              'guards_unchanged': True, 'gpu': torch.cuda.get_device_name(0),
              'stages': list(kernels), 'invariant_validation': 'unavailable'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Independent full-layer and guard checks PASS', flush=True)


if __name__ == '__main__':
    main()
