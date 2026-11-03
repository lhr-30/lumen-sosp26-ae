"""Independent packed-to-attended tests, including graph updates and layout cases."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(HERE / 'sources/ave'))
import torch
import torch.nn.functional as F
from evaluate import capture, make_inputs
from middle_reference import middle_reference


def check(actual, expected, storage, label):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all().item(), label + ': unwritten or nonfinite output'
    assert (storage[:512] == 23).all().item(), label + ': leading guard modified'
    assert (storage[512 + actual.numel():] == 23).all().item(), label + ': trailing guard modified'
    av, ev = actual.double().flatten(), expected.double().flatten()
    error, norm = av - ev, ev.norm().item()
    result = {'max_abs': error.abs().max().item(), 'guards_unchanged': True}
    if norm:
        result['cosine'] = F.cosine_similarity(av, ev, dim=0).item()
        result['relative_l2'] = error.norm().item() / norm
        assert result['cosine'] >= 0.9999, (label, result)
        assert result['relative_l2'] <= 0.02, (label, result)
    else:
        assert result['max_abs'] == 0, (label, result)
    print(label, result, flush=True)
    return {'case': label, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-module', default='round4_async_candidate')
    parser.add_argument('--config')
    parser.add_argument('--output', type=Path, default=RUN / 'analysis/middle.json')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    candidate = importlib.import_module(args.candidate_module)
    x, _, weights, tables = make_inputs(0)
    packed = (x.view(416, 2048).float() @ weights['qkv'].float().T).bfloat16()
    count = 16 * 26 * 2048
    storage = torch.full((512 + count + 4096,), 23, dtype=x.dtype, device=x.device)
    attended = storage[512:512 + count].view(16, 26, 2048)
    attended.fill_(float('nan'))
    kernels = candidate.bind_middle(packed, weights['q_norm'], weights['k_norm'], tables,
                                    attended, config=args.config)

    def chain():
        for fn in kernels.values():
            fn()

    def expected():
        return middle_reference(packed.view(16, 26, 5120), weights, tables)

    records = []
    chain()
    torch.cuda.synchronize()
    records.append(check(attended, expected(), storage, 'seed0'))
    graph = capture(chain, 1)
    for seed in [17, 123]:
        fresh_x, _, fresh_w, fresh_tables = make_inputs(seed)
        packed.copy_((fresh_x.view(416, 2048).float() @ fresh_w['qkv'].float().T).bfloat16())
        weights['q_norm'].copy_(fresh_w['q_norm'])
        weights['k_norm'].copy_(fresh_w['k_norm'])
        tables.copy_(fresh_tables.flip(0) if seed == 123 else fresh_tables)
        graph.replay()
        torch.cuda.synchronize()
        records.append(check(attended, expected(), storage, f'graph_seed{seed}'))
    previous = attended.clone()
    packed.view(16, 26, 5120)[0, 13:].mul_(-2)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(attended[0, :13], previous[0, :13]), 'Future tokens affect causal prefix'
    assert torch.equal(attended[1:], previous[1:]), 'Cross-batch contamination'
    records.append(check(attended, expected(), storage, 'causal_batch_isolation'))

    # Q=K=0 gives uniform causal attention. Distinct KV values and interleaved
    # gate-head constants exercise GQA/head/gate indexing and the final query.
    packed.zero_()
    weights['q_norm'].zero_()
    weights['k_norm'].zero_()
    matrix = packed.view(16, 26, 5120)
    for head in range(8):
        matrix[..., head * 512 + 256:head * 512 + 512].fill_((head - 3) * .5)
    batch = torch.arange(1, 17, device=x.device, dtype=torch.float32)[:, None, None]
    position = torch.arange(1, 27, device=x.device, dtype=torch.float32)[None, :, None]
    values = (batch * position / 256).bfloat16()
    matrix[..., 4608:4864].copy_(values)
    matrix[..., 4864:5120].copy_(-values)
    graph.replay()
    torch.cuda.synchronize()
    records.append(check(attended, expected(), storage, 'gqa_gate_layout_uniform_scores'))

    packed.zero_()
    graph.replay()
    torch.cuda.synchronize()
    records.append(check(attended, expected(), storage, 'zero_packed'))
    report = {'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
              'candidate_sha256': hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
              'config': args.config or candidate.DEFAULT_ATTENTION_CONFIG,
              'stages': list(kernels), 'checks': records,
              'reference': 'Unchanged original reference AST slice; see middle_reference_provenance.json'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS', args.output, flush=True)


if __name__ == '__main__':
    main()
