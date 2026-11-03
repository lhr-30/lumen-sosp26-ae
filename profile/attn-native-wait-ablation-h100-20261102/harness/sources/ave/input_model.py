"""Authoritative reference extracted from model-tunner/attn_fusion/bench.py.

Only imports, fixed constants, rope_tables and reference are retained.
No generated kernel or CuTe implementation is included.
"""

import math
import torch
import torch.nn.functional as F

B, T, H, KV, D, HIDDEN, ROTARY, THETA, EPS = 16, 26, 8, 2, 256, 2048, 64, 1e7, 1e-6

M, PACKED = B * T, H * 2 * D + 2 * KV * D

def rope_tables():
    """BF16 cos/sin for positions 0..25 and the first rotary_dim/2 frequencies, shape [26,32,2]."""
    index = torch.arange(ROTARY // 2, device='cuda', dtype=torch.float32)
    angles = torch.arange(T, device='cuda', dtype=torch.float32)[:, None] * torch.exp(
        -math.log(THETA) * (2.0 * index / ROTARY))[None, :]
    return torch.stack((angles.cos(), angles.sin()), dim=-1).bfloat16().contiguous()

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
