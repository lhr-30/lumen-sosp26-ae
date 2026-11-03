"""Latency of the five kernels of one Qwen3.5 full-attention mixer in the final kg-20261020 graph.

batch=16, tokens=26 (M=416), hidden=2048, 8 Q heads / 2 KV heads, head_dim=256, rotary_dim=64.
Each kernel is captured alone into a CUDA graph of --iterations back-to-back calls;
the chain graph runs all five in model order.
"""
import argparse
import math
import statistics

import torch
import torch.nn.functional as F
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda

from kernels.oproj import PreparedGemmNT as OProjGemm
from kernels.qkv import PreparedGemmNT as QkvGemm
from k_norm_rope import PackedKNormRoPE
from qnorm_attention import PrefetchedQNormAttention

B, T, H, KV, D, HIDDEN, ROTARY, THETA, EPS = 16, 26, 8, 2, 256, 2048, 64, 1e7, 1e-6
M, PACKED = B * T, H * 2 * D + 2 * KV * D
# Selected configuration of candidate.PackedQNormRoPEAttentionGate.cute.v4.
ATTENTION_CONFIG = {'norm_threads': 16, 'gate_prefetch': True, 'norm_vector': 2,
                    'query_rows': 16, 'warps': 4, 'q_staging': 'register', 'q_stride': 296,
                    'norm_parallel': True, 'q_load_order': 'after_kv',
                    'table_prefetch': True, 'norm_tree': False}


def rope_tables():
    """BF16 cos/sin for positions 0..25 and the first rotary_dim/2 frequencies, shape [26,32,2]."""
    index = torch.arange(ROTARY // 2, device='cuda', dtype=torch.float32)
    angles = torch.arange(T, device='cuda', dtype=torch.float32)[:, None] * torch.exp(
        -math.log(THETA) * (2.0 * index / ROTARY))[None, :]
    return torch.stack((angles.cos(), angles.sin()), dim=-1).bfloat16().contiguous()


def stream():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def cute_call(kernel, tensors):
    args = tuple(from_dlpack(t, assumed_align=16) for t in tensors)
    compiled = cute.compile(kernel, *args, stream())
    return lambda: compiled(*args, stream())


def build():
    torch.manual_seed(0)
    bf16 = dict(dtype=torch.bfloat16, device='cuda')
    w = {'qkv': torch.randn(PACKED, HIDDEN, **bf16) * 0.02, 'o': torch.randn(HIDDEN, HIDDEN, **bf16) * 0.02,
         'q_norm': torch.randn(D, **bf16) * 0.1, 'k_norm': torch.randn(D, **bf16) * 0.1}
    x = torch.randn(B, T, HIDDEN, **bf16)            # input_norm output
    hidden = torch.randn(B, T, HIDDEN, **bf16)       # residual stream
    packed = torch.empty(B, T, PACKED, **bf16)
    key = torch.empty(B, T, KV, D, **bf16)
    attended = torch.empty(B, T, H * D, **bf16)
    projected = torch.empty(B, T, HIDDEN, **bf16)
    output = torch.empty(B, T, HIDDEN, **bf16)
    q_scale, k_scale = (1.0 + w[n].float() for n in ('q_norm', 'k_norm'))
    tables = rope_tables()

    qg = packed[..., :H * 2 * D].view(B, T, H, 2 * D)
    value = packed[..., H * 2 * D + KV * D:].view(B, T, KV, D).transpose(1, 2)
    qkv_operands = (x.view(M, HIDDEN), w['qkv'], packed.view(M, PACKED))
    qkv_gemm = QkvGemm(*qkv_operands).bind(*qkv_operands)
    k_norm = cute_call(PackedKNormRoPE(4, 1),
                       (packed[..., H * 2 * D:H * 2 * D + KV * D].view(B, T, KV, D), k_scale, tables, key))
    attention = cute_call(PrefetchedQNormAttention(**ATTENTION_CONFIG),
                          (qg[..., :D], key.transpose(1, 2), value, qg[..., D:], q_scale, tables, attended))
    o_operands = (attended.view(M, H * D), w['o'], projected.view(M, HIDDEN))
    o_gemm = OProjGemm(*o_operands).bind(*o_operands)
    residual = lambda: torch.add(hidden, projected, out=output)  # noqa: E731
    kernels = {'qkv_proj (CuTe 416x5120x2048)': qkv_gemm,
               'k_norm_rope (CuTe)': k_norm,
               'q_norm_rope_attention_gate (CuTe)': attention,
               'o_proj (CuTe 416x2048x2048)': o_gemm,
               'residual_add (torch.add)': residual}
    return kernels, (x, hidden, w, tables), (attended, output)


def reference(x, hidden, w, tables):
    """FP32 math with the kernels' BF16 rounding points."""
    packed = (x.float().view(M, HIDDEN) @ w['qkv'].float().T).bfloat16().view(B, T, PACKED)
    qg = packed[..., :H * 2 * D].view(B, T, H, 2 * D)
    q, gate = qg[..., :D], qg[..., D:]
    k = packed[..., H * 2 * D:H * 2 * D + KV * D].view(B, T, KV, D)
    v = packed[..., H * 2 * D + KV * D:].view(B, T, KV, D)
    cos, sin = (tables[None, :, None, :, i].float() for i in (0, 1))

    def norm_rope(t, weight):
        t = t.float()
        t = (t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + weight.float())).bfloat16().float()
        a, b = t[..., :ROTARY // 2], t[..., ROTARY // 2:ROTARY]
        return torch.cat((a * cos - b * sin, b * cos + a * sin, t[..., ROTARY:]), -1).bfloat16()

    q, k = norm_rope(q, w['q_norm']), norm_rope(k, w['k_norm'])
    heads = lambda t: t.transpose(1, 2).float()  # noqa: E731
    attn = F.scaled_dot_product_attention(heads(q), heads(k).repeat_interleave(H // KV, 1),
                                          heads(v).repeat_interleave(H // KV, 1), is_causal=True, scale=D ** -0.5)
    attn = attn.transpose(1, 2).bfloat16().float()
    attended = (attn * torch.sigmoid(gate.float()).bfloat16().float()).bfloat16().reshape(B, T, H * D)
    projected = (attended.float().view(M, H * D) @ w['o'].float().T).bfloat16().view(B, T, HIDDEN)
    return attended, hidden + projected


def check(actual, expected):
    for name, a, e in zip(('attended', 'output'), actual, expected):
        a, e = a.float().flatten(), e.float().flatten()
        cosine = F.cosine_similarity(a.double(), e.double(), dim=0).item()
        print(f'check {name}: cosine={cosine:.7f} max_abs={(a - e).abs().max().item():.4g}')
        if cosine < 0.9999:
            raise AssertionError(f'{name} cosine {cosine} < 0.9999')


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=100, help='calls per graph replay')
    parser.add_argument('--samples', type=int, default=30, help='replays per kernel')
    parser.add_argument('--profile', action='store_true', help='launch each kernel once and exit (for ncu)')
    args = parser.parse_args()
    kernels, inputs, outputs = build()
    for fn in kernels.values():
        fn()
    torch.cuda.synchronize()
    if args.profile:
        return
    check(outputs, reference(*inputs))

    def chain():
        for fn in kernels.values():
            fn()

    graphs = {name: capture(fn, args.iterations) for name, fn in {**kernels, 'chain (all 5)': chain}.items()}
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = {name: [] for name in graphs}
    for _ in range(args.samples):  # interleave kernels so clock drift hits all of them equally
        for name, graph in graphs.items():
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / args.iterations)

    props = torch.cuda.get_device_properties(0)
    print(f'\nGPU: {props.name}, {props.multi_processor_count} SM; '
          f'{args.iterations} calls/replay x {args.samples} replays; median [p10, p90] us/call')
    total = 0.0
    for name, values in samples.items():
        median = statistics.median(values)
        deciles = statistics.quantiles(values, n=10)
        if name != 'chain (all 5)':
            total += median
        else:
            print(f'{"sum of 5":42s} {total:8.3f}')
        print(f'{name:42s} {median:8.3f}  [{deciles[0]:.3f}, {deciles[-1]:.3f}]')


if __name__ == '__main__':
    main()
