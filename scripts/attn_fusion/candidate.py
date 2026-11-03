"""Description-generated AveLang implementation of the fixed attention mixer.

No runtime tensor arithmetic runs on the host. WGMMA is used only for the two
linear layers; the small causal attention uses FP32 scalar math and shuffles.
"""
from __future__ import annotations

from collections import OrderedDict

import avelang
import avelang.language as S
from avelang.language import invariant


@avelang.jit
def rsqrt_positive(x: S.f32) -> S.f32:
    # The fixed compiler emits an unresolved libdevice call for S.sqrt.
    # Four Newton steps after the standard FP32 exponent seed reach FP32
    # precision for finite positive normal values, including epsilon=1e-6.
    bits = S.bitcast(x, S.i32)
    y = S.bitcast(1597463007 - (bits >> 1), S.f32)
    halfx = x * S.convert(0.5, S.f32)
    for iteration in S.range(4):
        y = y * (S.convert(1.5, S.f32) - halfx * y * y)
    return y


@avelang.jit
def exp_nonpositive(x: S.f32) -> S.f32:
    # Range reduction x = n*ln(2) + r with |r| <= ln(2)/2, followed by a
    # degree-seven Taylor polynomial. This is expressed entirely in the DSL.
    bounded = x
    if x < S.convert(-80, S.f32):
        bounded = S.convert(-80, S.f32)
    n = S.convert(bounded * S.convert(1.4426950408889634, S.f32) - S.convert(0.5, S.f32), S.i32)
    r = bounded - S.convert(n, S.f32) * S.convert(0.6931471805599453, S.f32)
    p = S.convert(0.0001984126984126984, S.f32)
    p = p * r + S.convert(0.001388888888888889, S.f32)
    p = p * r + S.convert(0.008333333333333333, S.f32)
    p = p * r + S.convert(0.041666666666666664, S.f32)
    p = p * r + S.convert(0.16666666666666666, S.f32)
    p = p * r + S.convert(0.5, S.f32)
    p = p * r + S.convert(1, S.f32)
    p = p * r + S.convert(1, S.f32)
    scale = S.bitcast((n + 127) << 23, S.f32)
    value = p * scale
    if x < S.convert(-80, S.f32):
        value = S.convert(0, S.f32)
    return value


@avelang.jit
def linear_bf16(
    A: S.Tensor((416, 2048), S.bf16),
    W: S.Tensor((N, 2048), S.bf16),
    C: S.Tensor((416, N), S.bf16),
    N: S.constexpr,
):
    tid = S.thread_id(0)
    warp = tid // 32
    lane = tid % 32
    row0 = S.block_id(1) * 64
    col0 = S.block_id(0) * 32
    a = S.make_shared((8, 2, 8, 8), S.bf16, 16)
    b0 = S.make_shared((1, 2, 8, 8), S.bf16, 16)
    b1 = S.make_shared((1, 2, 8, 8), S.bf16, 16)
    b2 = S.make_shared((1, 2, 8, 8), S.bf16, 16)
    b3 = S.make_shared((1, 2, 8, 8), S.bf16, 16)
    ready = S.make_shared((1,), S.i64, 8)
    acc0 = S.make_local((4,), S.f32)
    acc1 = S.make_local((4,), S.f32)
    acc2 = S.make_local((4,), S.f32)
    acc3 = S.make_local((4,), S.f32)
    for i in S.range(4):
        acc0[i] = S.convert(0, S.f32)
        acc1[i] = S.convert(0, S.f32)
        acc2[i] = S.convert(0, S.f32)
        acc3[i] = S.convert(0, S.f32)
    if tid == 0:
        S.nvvm.mbarrier_init(ready, 128)
    S.syncthreads()
    for kt in S.range(128):
        for j in S.range(8):
            z = tid + j * 128
            r = z // 16
            k = z % 16
            av = S.convert(0, S.bf16)
            if row0 + r < 416:
                av = A[row0 + r, kt * 16 + k]
            a[r // 8, k // 8, r % 8, k % 8] = av
        n = tid // 16
        k = tid % 16
        b0[0, k // 8, n, k % 8] = W[col0 + 0 + n, kt * 16 + k]
        b1[0, k // 8, n, k % 8] = W[col0 + 8 + n, kt * 16 + k]
        b2[0, k // 8, n, k % 8] = W[col0 + 16 + n, kt * 16 + k]
        b3[0, k // 8, n, k % 8] = W[col0 + 24 + n, kt * 16 + k]
        S.nvvm.fence_proxy_async_shared()
        S.nvvm.mbarrier_arrive(ready)
        S.nvvm.mbarrier_wait_parity(ready, kt % 2)
        acc0 = S.nvvm.wgmma_m64n8k16_bf16_f32(a, b0, acc0)
        acc1 = S.nvvm.wgmma_m64n8k16_bf16_f32(a, b1, acc1)
        acc2 = S.nvvm.wgmma_m64n8k16_bf16_f32(a, b2, acc2)
        acc3 = S.nvvm.wgmma_m64n8k16_bf16_f32(a, b3, acc3)
        S.syncthreads()
    if tid == 0:
        S.nvvm.mbarrier_inval(ready)
    r = row0 + warp * 16 + lane // 4
    c = col0 + 0 + (lane % 4) * 2
    if r < 416:
        C[r, c] = S.convert(acc0[0], S.bf16)
        C[r, c + 1] = S.convert(acc0[1], S.bf16)
    if r + 8 < 416:
        C[r + 8, c] = S.convert(acc0[2], S.bf16)
        C[r + 8, c + 1] = S.convert(acc0[3], S.bf16)

    c = col0 + 8 + (lane % 4) * 2
    if r < 416:
        C[r, c] = S.convert(acc1[0], S.bf16)
        C[r, c + 1] = S.convert(acc1[1], S.bf16)
    if r + 8 < 416:
        C[r + 8, c] = S.convert(acc1[2], S.bf16)
        C[r + 8, c + 1] = S.convert(acc1[3], S.bf16)

    c = col0 + 16 + (lane % 4) * 2
    if r < 416:
        C[r, c] = S.convert(acc2[0], S.bf16)
        C[r, c + 1] = S.convert(acc2[1], S.bf16)
    if r + 8 < 416:
        C[r + 8, c] = S.convert(acc2[2], S.bf16)
        C[r + 8, c + 1] = S.convert(acc2[3], S.bf16)

    c = col0 + 24 + (lane % 4) * 2
    if r < 416:
        C[r, c] = S.convert(acc3[0], S.bf16)
        C[r, c + 1] = S.convert(acc3[1], S.bf16)
    if r + 8 < 416:
        C[r + 8, c] = S.convert(acc3[2], S.bf16)
        C[r + 8, c + 1] = S.convert(acc3[3], S.bf16)


@avelang.jit
def unpack(
    Packed: S.Tensor((416, 5120), S.bf16),
    Q: S.Tensor((416, 8, 256), S.bf16),
    Gate: S.Tensor((416, 8, 256), S.bf16),
    K: S.Tensor((416, 2, 256), S.bf16),
    V: S.Tensor((416, 2, 256), S.bf16),
    GATE_SHIFT: S.constexpr,
):
    # Output tags specify the packed source coordinate required at each output
    # coordinate. Assertions are erased before execution; no uninitialized
    # output value is consumed by the runtime computation.
    invariant.tag(Packed, lambda m, c: (m, c), "PackedCoordinate")
    invariant.tag(Q, lambda m, h, d: (m, h * 512 + d), "QuerySource")
    invariant.tag(Gate, lambda m, h, d: (m, h * 512 + 256 + d), "GateSource")
    invariant.tag(K, lambda m, h, d: (m, 4096 + h * 256 + d), "KeySource")
    invariant.tag(V, lambda m, h, d: (m, 4608 + h * 256 + d), "ValueSource")
    m = S.block_id(0)
    h = S.block_id(1)
    d = S.thread_id(0)
    q = Packed[m, h * 512 + d]
    gate = Packed[m, h * 512 + 256 + d + GATE_SHIFT]
    invariant.assert_tag_eq(q, Q[m, h, d])
    invariant.assert_tag_eq(gate, Gate[m, h, d])
    Q[m, h, d] = q
    Gate[m, h, d] = gate
    if h < 2:
        key = Packed[m, 4096 + h * 256 + d]
        value = Packed[m, 4608 + h * 256 + d]
        invariant.assert_tag_eq(key, K[m, h, d])
        invariant.assert_tag_eq(value, V[m, h, d])
        K[m, h, d] = key
        V[m, h, d] = value


@avelang.jit
def norm_rope(
    X: S.Tensor((416, HEADS, 256), S.bf16),
    Weight: S.Tensor((256,), S.bf16),
    Tables: S.Tensor((26, 32, 2), S.bf16),
    Y: S.Tensor((416, HEADS, 256), S.bf16),
    HEADS: S.constexpr,
    PAIR_SHIFT: S.constexpr,
):
    # A RoPE pair has the same token, head and frequency. The 32-channel
    # split-half offset is deliberately absent from the frequency coordinate.
    invariant.tag(X, lambda m, h, d: (m, h, d % 32), "RopePairCoordinate")
    m = S.block_id(0)
    h = S.block_id(1)
    lane = S.thread_id(0)
    raw = S.make_shared((256,), S.bf16)
    norm = S.make_shared((256,), S.bf16)
    ss = S.convert(0, S.f32)
    for j in S.range(8):
        d = lane + j * 32
        raw[d] = X[m, h, d]
        f = S.convert(X[m, h, d], S.f32)
        ss = ss + f * f
    S.syncthreads()
    ss = ss + S.shuffle_xor(ss, 16, 32)
    ss = ss + S.shuffle_xor(ss, 8, 32)
    ss = ss + S.shuffle_xor(ss, 4, 32)
    ss = ss + S.shuffle_xor(ss, 2, 32)
    ss = ss + S.shuffle_xor(ss, 1, 32)
    inv = rsqrt_positive(ss * S.convert(0.00390625, S.f32) + S.convert(0.000001, S.f32))
    for j in S.range(8):
        d = lane + j * 32
        v = S.convert(raw[d], S.f32)
        gain = S.convert(1, S.f32) + S.convert(Weight[d], S.f32)
        norm[d] = S.convert(v * inv * gain, S.bf16)
    S.syncthreads()
    # Indices used to fetch the two normalized channels are independently
    # checked against the source-coordinate contract before doing arithmetic.
    lo = lane
    hi = lane + 32 + PAIR_SHIFT
    invariant.assert_tag_eq(X[m, h, lo], X[m, h, hi])
    a = S.convert(norm[lo], S.f32)
    b = S.convert(norm[hi], S.f32)
    co = S.convert(Tables[m % 26, lane, 0], S.f32)
    si = S.convert(Tables[m % 26, lane, 1], S.f32)
    Y[m, h, lane] = S.convert(a * co - b * si, S.bf16)
    Y[m, h, lane + 32] = S.convert(b * co + a * si, S.bf16)
    for j in S.range(2, 8):
        d = lane + j * 32
        Y[m, h, d] = norm[d]


@avelang.jit
def causal_gqa_gate(
    Q: S.Tensor((16, 26, 2, 4, 256), S.bf16),
    K: S.Tensor((16, 26, 2, 256), S.bf16),
    V: S.Tensor((16, 26, 2, 256), S.bf16),
    Gate: S.Tensor((16, 26, 2, 4, 256), S.bf16),
    Out: S.Tensor((16, 26, 2, 4, 256), S.bf16),
    KV_SHIFT: S.constexpr,
):
    # Reduction pairs must agree on batch, KV group and channel. Time is
    # intentionally not in this tag: each query attends multiple key positions.
    invariant.tag(Q, lambda b, t, g, h, d: (b, g, d), "QueryDotCoordinate")
    invariant.tag(K, lambda b, t, h, d: (b, h, d), "KeyDotCoordinate")
    invariant.tag(V, lambda b, t, h, d: (b, h, d), "ValueCoordinate")
    qpos = S.block_id(0)
    batch = S.block_id(2) // 4
    h = S.block_id(2) % 4
    group = S.block_id(1)
    lane = S.thread_id(0)
    kv = group + KV_SHIFT
    invariant.assert_tag_eq(Q[batch, qpos, group, h, lane], K[batch, 0, kv, lane])
    acc0 = S.convert(0, S.f32)
    acc1 = S.convert(0, S.f32)
    acc2 = S.convert(0, S.f32)
    acc3 = S.convert(0, S.f32)
    acc4 = S.convert(0, S.f32)
    acc5 = S.convert(0, S.f32)
    acc6 = S.convert(0, S.f32)
    acc7 = S.convert(0, S.f32)
    maximum = S.convert(-1.0e30, S.f32)
    denom = S.convert(0, S.f32)
    for kpos in S.range(qpos + 1):
        score = S.convert(0, S.f32)
        for j in S.range(8):
            d = lane + 32 * j
            q = Q[batch, qpos, group, h, d]
            k = K[batch, kpos, kv, d]
            invariant.assert_tag_eq(q, k)
            score = score + S.convert(q, S.f32) * S.convert(k, S.f32)
        score = score + S.shuffle_xor(score, 16, 32)
        score = score + S.shuffle_xor(score, 8, 32)
        score = score + S.shuffle_xor(score, 4, 32)
        score = score + S.shuffle_xor(score, 2, 32)
        score = score + S.shuffle_xor(score, 1, 32)
        score = score * S.convert(0.0625, S.f32)
        newmax = maximum
        if score > maximum:
            newmax = score
        alpha = exp_nonpositive(maximum - newmax)
        beta = exp_nonpositive(score - newmax)
        denom = denom * alpha + beta
        d = lane + 0
        value = V[batch, kpos, kv, d]
        invariant.assert_tag_eq(Q[batch, qpos, group, h, d], value)
        acc0 = acc0 * alpha + beta * S.convert(value, S.f32)
        d = lane + 32
        value = V[batch, kpos, kv, d]
        invariant.assert_tag_eq(Q[batch, qpos, group, h, d], value)
        acc1 = acc1 * alpha + beta * S.convert(value, S.f32)
        d = lane + 64
        value = V[batch, kpos, kv, d]
        invariant.assert_tag_eq(Q[batch, qpos, group, h, d], value)
        acc2 = acc2 * alpha + beta * S.convert(value, S.f32)
        d = lane + 96
        value = V[batch, kpos, kv, d]
        invariant.assert_tag_eq(Q[batch, qpos, group, h, d], value)
        acc3 = acc3 * alpha + beta * S.convert(value, S.f32)
        d = lane + 128
        value = V[batch, kpos, kv, d]
        invariant.assert_tag_eq(Q[batch, qpos, group, h, d], value)
        acc4 = acc4 * alpha + beta * S.convert(value, S.f32)
        d = lane + 160
        value = V[batch, kpos, kv, d]
        invariant.assert_tag_eq(Q[batch, qpos, group, h, d], value)
        acc5 = acc5 * alpha + beta * S.convert(value, S.f32)
        d = lane + 192
        value = V[batch, kpos, kv, d]
        invariant.assert_tag_eq(Q[batch, qpos, group, h, d], value)
        acc6 = acc6 * alpha + beta * S.convert(value, S.f32)
        d = lane + 224
        value = V[batch, kpos, kv, d]
        invariant.assert_tag_eq(Q[batch, qpos, group, h, d], value)
        acc7 = acc7 * alpha + beta * S.convert(value, S.f32)
        maximum = newmax
    d = lane + 0
    att = S.convert(acc0 / denom, S.bf16)
    g = S.convert(Gate[batch, qpos, group, h, d], S.f32)
    negg = -g
    if g < S.convert(0, S.f32):
        negg = g
    e = exp_nonpositive(negg)
    sig = S.convert(1, S.f32) / (S.convert(1, S.f32) + e)
    if g < S.convert(0, S.f32):
        sig = e * sig
    sg = S.convert(sig, S.bf16)
    Out[batch, qpos, group, h, d] = S.convert(S.convert(att, S.f32) * S.convert(sg, S.f32), S.bf16)

    d = lane + 32
    att = S.convert(acc1 / denom, S.bf16)
    g = S.convert(Gate[batch, qpos, group, h, d], S.f32)
    negg = -g
    if g < S.convert(0, S.f32):
        negg = g
    e = exp_nonpositive(negg)
    sig = S.convert(1, S.f32) / (S.convert(1, S.f32) + e)
    if g < S.convert(0, S.f32):
        sig = e * sig
    sg = S.convert(sig, S.bf16)
    Out[batch, qpos, group, h, d] = S.convert(S.convert(att, S.f32) * S.convert(sg, S.f32), S.bf16)

    d = lane + 64
    att = S.convert(acc2 / denom, S.bf16)
    g = S.convert(Gate[batch, qpos, group, h, d], S.f32)
    negg = -g
    if g < S.convert(0, S.f32):
        negg = g
    e = exp_nonpositive(negg)
    sig = S.convert(1, S.f32) / (S.convert(1, S.f32) + e)
    if g < S.convert(0, S.f32):
        sig = e * sig
    sg = S.convert(sig, S.bf16)
    Out[batch, qpos, group, h, d] = S.convert(S.convert(att, S.f32) * S.convert(sg, S.f32), S.bf16)

    d = lane + 96
    att = S.convert(acc3 / denom, S.bf16)
    g = S.convert(Gate[batch, qpos, group, h, d], S.f32)
    negg = -g
    if g < S.convert(0, S.f32):
        negg = g
    e = exp_nonpositive(negg)
    sig = S.convert(1, S.f32) / (S.convert(1, S.f32) + e)
    if g < S.convert(0, S.f32):
        sig = e * sig
    sg = S.convert(sig, S.bf16)
    Out[batch, qpos, group, h, d] = S.convert(S.convert(att, S.f32) * S.convert(sg, S.f32), S.bf16)

    d = lane + 128
    att = S.convert(acc4 / denom, S.bf16)
    g = S.convert(Gate[batch, qpos, group, h, d], S.f32)
    negg = -g
    if g < S.convert(0, S.f32):
        negg = g
    e = exp_nonpositive(negg)
    sig = S.convert(1, S.f32) / (S.convert(1, S.f32) + e)
    if g < S.convert(0, S.f32):
        sig = e * sig
    sg = S.convert(sig, S.bf16)
    Out[batch, qpos, group, h, d] = S.convert(S.convert(att, S.f32) * S.convert(sg, S.f32), S.bf16)

    d = lane + 160
    att = S.convert(acc5 / denom, S.bf16)
    g = S.convert(Gate[batch, qpos, group, h, d], S.f32)
    negg = -g
    if g < S.convert(0, S.f32):
        negg = g
    e = exp_nonpositive(negg)
    sig = S.convert(1, S.f32) / (S.convert(1, S.f32) + e)
    if g < S.convert(0, S.f32):
        sig = e * sig
    sg = S.convert(sig, S.bf16)
    Out[batch, qpos, group, h, d] = S.convert(S.convert(att, S.f32) * S.convert(sg, S.f32), S.bf16)

    d = lane + 192
    att = S.convert(acc6 / denom, S.bf16)
    g = S.convert(Gate[batch, qpos, group, h, d], S.f32)
    negg = -g
    if g < S.convert(0, S.f32):
        negg = g
    e = exp_nonpositive(negg)
    sig = S.convert(1, S.f32) / (S.convert(1, S.f32) + e)
    if g < S.convert(0, S.f32):
        sig = e * sig
    sg = S.convert(sig, S.bf16)
    Out[batch, qpos, group, h, d] = S.convert(S.convert(att, S.f32) * S.convert(sg, S.f32), S.bf16)

    d = lane + 224
    att = S.convert(acc7 / denom, S.bf16)
    g = S.convert(Gate[batch, qpos, group, h, d], S.f32)
    negg = -g
    if g < S.convert(0, S.f32):
        negg = g
    e = exp_nonpositive(negg)
    sig = S.convert(1, S.f32) / (S.convert(1, S.f32) + e)
    if g < S.convert(0, S.f32):
        sig = e * sig
    sg = S.convert(sig, S.bf16)
    Out[batch, qpos, group, h, d] = S.convert(S.convert(att, S.f32) * S.convert(sg, S.f32), S.bf16)


@avelang.jit
def add_residual(
    Hidden: S.Tensor((416, 2048), S.bf16),
    Projected: S.Tensor((416, 2048), S.bf16),
    Out: S.Tensor((416, 2048), S.bf16),
):
    invariant.tag(Hidden, lambda m, d: (m, d), "ResidualCoordinate")
    invariant.tag(Projected, lambda m, d: (m, d), "ProjectionCoordinate")
    m = S.block_id(0)
    d = S.block_id(1) * 256 + S.thread_id(0)
    h = Hidden[m, d]
    p = Projected[m, d]
    invariant.assert_tag_eq(h, p)
    Out[m, d] = S.convert(S.convert(h, S.f32) + S.convert(p, S.f32), S.bf16)


def build(x, hidden, w, tables):
    """Allocate tensors and bind launch callables; execute no numerical work."""
    import torch

    assert tuple(x.shape) == (16, 26, 2048)
    assert tuple(hidden.shape) == tuple(x.shape)
    assert tuple(tables.shape) == (26, 32, 2)
    assert all(t.is_contiguous() and t.dtype == torch.bfloat16 and t.device == x.device
               for t in [x, hidden, tables, *w.values()])
    packed = torch.empty((416, 5120), device=x.device, dtype=torch.bfloat16)
    qraw = torch.empty((416, 8, 256), device=x.device, dtype=torch.bfloat16)
    gate = torch.empty_like(qraw)
    q = torch.empty_like(qraw)
    kraw = torch.empty((416, 2, 256), device=x.device, dtype=torch.bfloat16)
    v = torch.empty_like(kraw)
    k = torch.empty_like(kraw)
    attended = torch.empty_like(x)
    projected = torch.empty_like(x)
    output = torch.empty_like(x)
    opts = {"validate_invariants": True}
    kernels = OrderedDict()
    kernels["qkv_projection"] = lambda: linear_bf16[lambda: ((160, 7, 1), (128, 1, 1))](
        x.view(416, 2048), w["qkv"], packed, 5120, num_warps=4, **opts)
    kernels["unpack_q_gate_k_v"] = lambda: unpack[lambda: ((416, 8, 1), (256, 1, 1))](
        packed, qraw, gate, kraw, v, 0, num_warps=8, **opts)
    kernels["q_norm_rope"] = lambda: norm_rope[lambda: ((416, 8, 1), (32, 1, 1))](
        qraw, w["q_norm"], tables, q, 8, 0, num_warps=1, **opts)
    kernels["k_norm_rope"] = lambda: norm_rope[lambda: ((416, 2, 1), (32, 1, 1))](
        kraw, w["k_norm"], tables, k, 2, 0, num_warps=1, **opts)
    kernels["causal_gqa_gate"] = lambda: causal_gqa_gate[lambda: ((26, 2, 64), (32, 1, 1))](
        q.view(16, 26, 2, 4, 256), k.view(16, 26, 2, 256),
        v.view(16, 26, 2, 256), gate.view(16, 26, 2, 4, 256),
        attended.view(16, 26, 2, 4, 256), 0, num_warps=1, **opts)
    kernels["output_projection"] = lambda: linear_bf16[lambda: ((64, 7, 1), (128, 1, 1))](
        attended.view(416, 2048), w["o"], projected.view(416, 2048), 2048, num_warps=4, **opts)
    kernels["residual"] = lambda: add_residual[lambda: ((416, 8, 1), (256, 1, 1))](
        hidden.view(416, 2048), projected.view(416, 2048), output.view(416, 2048), num_warps=8, **opts)
    return kernels, (attended, output)
