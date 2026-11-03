"""Packed-to-attended slice of the unchanged authoritative reference AST."""
from input_model import *

def middle_reference(packed, w, tables):
    qg = packed[..., :H * 2 * D].view(B, T, H, 2 * D)
    q, gate = (qg[..., :D], qg[..., D:])
    k = packed[..., H * 2 * D:H * 2 * D + KV * D].view(B, T, KV, D)
    v = packed[..., H * 2 * D + KV * D:].view(B, T, KV, D)
    cos, sin = (tables[None, :, None, :, i].float() for i in (0, 1))

    def norm_rope(t, weight):
        t = t.float()
        t = (t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + weight.float())).bfloat16().float()
        a, b = (t[..., :ROTARY // 2], t[..., ROTARY // 2:ROTARY])
        return torch.cat((a * cos - b * sin, b * cos + a * sin, t[..., ROTARY:]), -1).bfloat16()
    q, k = (norm_rope(q, w['q_norm']), norm_rope(k, w['k_norm']))
    heads = lambda t: t.transpose(1, 2).float()
    attn = F.scaled_dot_product_attention(heads(q), heads(k).repeat_interleave(H // KV, 1), heads(v).repeat_interleave(H // KV, 1), is_causal=True, scale=D ** (-0.5))
    attn = attn.transpose(1, 2).bfloat16().float()
    attended = (attn * torch.sigmoid(gate.float()).bfloat16().float()).bfloat16().reshape(B, T, H * D)
    return attended
