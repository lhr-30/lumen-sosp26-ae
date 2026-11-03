"""Native-Hopper AveLang lowering of the same fixed single-layer mixer.

Projection staging follows the newly supplied repo's TMA/WGMMA primitives;
attention remains FP32 with the reference's explicit BF16 rounding points.
This native branch does not provide the artifact invariant-validation API.
"""
from __future__ import annotations

from collections import OrderedDict

import avelang
import avelang.language as S

SWIZZLE_128B = S.WGMMA_SWIZZLE_128B
INVARIANT_VALIDATION_AVAILABLE = False


@avelang.jit
def linear_native(
    A: S.Tensor((416, 2048), S.bf16),
    W: S.Tensor((N, 2048), S.bf16),
    C: S.Tensor((416, N), S.bf16),
    N: S.constexpr,
):
    """64x128x64 tiles; three-stage TMA producer and one WGMMA consumer."""
    tid = S.thread_id(0)
    row0 = S.block_id(1) * 64
    col0 = S.block_id(0) * 128
    # Stage arenas are multiples of 128 bytes and preserve TMA swizzle phase.
    full = S.nvvm.mbarrier_create(3)
    empty = S.nvvm.mbarrier_create(3)
    b_shared = S.make_shared((3, 8192), S.bf16, 128)
    a_shared = S.make_shared((3, 4096), S.bf16, 128)
    a_tensor = S.view(A, S.bf16,
                     S.make_layout((416, 32, 64), (2048, 64, 1)))
    b_tensor = S.view(W, S.bf16,
                     S.make_layout((N, 32, 64), (2048, 64, 1)))
    a_tma = S.nvvm.make_tma_descriptor(
        a_tensor, S.make_layout((64, 1, 64), (64, 4096, 1)), SWIZZLE_128B)
    b_tma = S.nvvm.make_tma_descriptor(
        b_tensor, S.make_layout((128, 1, 64), (64, 8192, 1)), SWIZZLE_128B)
    for stage in S.range(3, unroll=True):
        S.nvvm.mbarrier_init(full, stage, count=1, predicate=tid == 0)
        S.nvvm.mbarrier_init(empty, stage, count=1, predicate=tid == 0)
    S.nvvm.fence_mbarrier_init_release_cluster()
    S.syncthreads()

    if tid < 128:
        warp = tid // 32
        lane = tid % 32
        accumulators = S.make_local((1, 64), S.f32)
        for element in S.range(64, unroll=True):
            accumulators[0, element] = S.convert(0, S.f32)
        for kt in S.range(32):
            stage = kt % 3
            phase = (kt // 3) % 2
            S.nvvm.mbarrier_try_wait_parity(full, phase, 10000000, stage)
            a_tile = S.view(
                S.subview(a_shared, (stage, 0), (1, 4096), (1, 1)),
                S.bf16, S.make_layout((64, 64), (64, 1)))
            b_tile = S.view(
                S.subview(b_shared, (stage, 0), (1, 8192), (1, 1)),
                S.bf16, S.make_layout((128, 64), (64, 1)))
            da = S.nvvm.make_wgmma_descriptor_bits(
                a_tile, SWIZZLE_128B, 0, 0, 0, 16, 1024)
            db = S.nvvm.make_wgmma_descriptor_bits(
                b_tile, SWIZZLE_128B, 0, 0, 0, 16, 1024)
            acc = accumulators[0]
            S.nvvm.wgmma_fence_aligned()
            acc = S.nvvm.wgmma_m64n128k16_f32_bf16_bf16(da, db, acc, 1)
            acc = S.nvvm.wgmma_m64n128k16_f32_bf16_bf16(da + 2, db + 2, acc, 1)
            acc = S.nvvm.wgmma_m64n128k16_f32_bf16_bf16(da + 4, db + 4, acc, 1)
            acc = S.nvvm.wgmma_m64n128k16_f32_bf16_bf16(da + 6, db + 6, acc, 1)
            S.nvvm.wgmma_group_sync_aligned()
            S.nvvm.wgmma_wait_group_sync(0)
            S.nvvm.wgmma_fence_aligned()
            for element in S.range(64, unroll=True):
                accumulators[0, element] = acc[element]
            if tid == 0:
                S.nvvm.mbarrier_arrive(empty, stage)
        # The tensor map zero-fills the final 32 invalid A rows. Only valid
        # output rows are stored; both fixed N dimensions are multiples of128.
        row = row0 + warp * 16 + lane // 4
        for group in S.range(16, unroll=True):
            col = col0 + group * 8 + (lane % 4) * 2
            if row < 416:
                C[row, col] = S.convert(accumulators[0, 4 * group], S.bf16)
                C[row, col + 1] = S.convert(accumulators[0, 4 * group + 1], S.bf16)
            if row + 8 < 416:
                C[row + 8, col] = S.convert(accumulators[0, 4 * group + 2], S.bf16)
                C[row + 8, col + 1] = S.convert(accumulators[0, 4 * group + 3], S.bf16)
    else:
        elected = S.nvvm.elect_sync()
        for kt in S.range(32):
            stage = kt % 3
            phase = 1 ^ ((kt // 3) % 2)
            S.nvvm.mbarrier_try_wait_parity(empty, phase, 10000000, stage)
            S.nvvm.mbarrier_arrive_expect_tx(full, 24576, stage, elected)
            a_dst = S.view(
                S.subview(a_shared, (stage, 0), (1, 4096), (1, 1)),
                S.bf16, S.make_layout((64, 1, 64), (64, 4096, 1)))
            b_dst = S.view(
                S.subview(b_shared, (stage, 0), (1, 8192), (1, 1)),
                S.bf16, S.make_layout((128, 1, 64), (64, 8192, 1)))
            S.nvvm.tma_load(a_dst, a_tma, (0, kt, row0), full,
                            mbar_id=stage, predicate=elected, expect_tx=False)
            S.nvvm.tma_load(b_dst, b_tma, (0, kt, col0), full,
                            mbar_id=stage, predicate=elected, expect_tx=False)
    S.syncthreads()


@avelang.jit
def unpack(
    Packed: S.Tensor((416, 5120), S.bf16),
    Q: S.Tensor((416, 8, 256), S.bf16),
    Gate: S.Tensor((416, 8, 256), S.bf16),
    K: S.Tensor((416, 2, 256), S.bf16),
    V: S.Tensor((416, 2, 256), S.bf16),
):
    m = S.block_id(0)
    h = S.block_id(1)
    d = S.thread_id(0)
    Q[m, h, d] = Packed[m, h * 512 + d]
    Gate[m, h, d] = Packed[m, h * 512 + 256 + d]
    if h < 2:
        K[m, h, d] = Packed[m, 4096 + h * 256 + d]
        V[m, h, d] = Packed[m, 4608 + h * 256 + d]


@avelang.jit
def norm_rope(
    X: S.Tensor((416, HEADS, 256), S.bf16),
    Weight: S.Tensor((256,), S.bf16),
    Tables: S.Tensor((26, 32, 2), S.bf16),
    Y: S.Tensor((416, HEADS, 256), S.bf16),
    HEADS: S.constexpr,
):
    m = S.block_id(0)
    h = S.block_id(1)
    lane = S.thread_id(0)
    norm = S.make_shared((256,), S.bf16, 16)
    ss = S.convert(0, S.f32)
    for j in S.range(8, unroll=True):
        d = lane + j * 32
        f = S.convert(X[m, h, d], S.f32)
        ss = ss + f * f
    ss = ss + S.shuffle_xor(ss, 16, 32)
    ss = ss + S.shuffle_xor(ss, 8, 32)
    ss = ss + S.shuffle_xor(ss, 4, 32)
    ss = ss + S.shuffle_xor(ss, 2, 32)
    ss = ss + S.shuffle_xor(ss, 1, 32)
    inv = S.nvvm.fast_rsqrt(ss * S.convert(0.00390625, S.f32) + S.convert(0.000001, S.f32))
    for j in S.range(8, unroll=True):
        d = lane + j * 32
        v = S.convert(X[m, h, d], S.f32)
        gain = S.convert(1, S.f32) + S.convert(Weight[d], S.f32)
        norm[d] = S.convert(v * inv * gain, S.bf16)
    S.syncthreads()
    a = S.convert(norm[lane], S.f32)
    b = S.convert(norm[lane + 32], S.f32)
    co = S.convert(Tables[m % 26, lane, 0], S.f32)
    si = S.convert(Tables[m % 26, lane, 1], S.f32)
    Y[m, h, lane] = S.convert(a * co - b * si, S.bf16)
    Y[m, h, lane + 32] = S.convert(b * co + a * si, S.bf16)
    for j in S.range(2, 8, unroll=True):
        d = lane + j * 32
        Y[m, h, d] = norm[d]


@avelang.jit
def gate_store(
    value: S.f32,
    gate: S.bf16,
) -> S.bf16:
    att = S.convert(value, S.bf16)
    g = S.convert(gate, S.f32)
    negg = -g
    if g < S.convert(0, S.f32):
        negg = g
    e = S.nvvm.fast_exp2(negg * S.convert(1.4426950408889634, S.f32))
    sig = S.nvvm.fast_rcp(S.convert(1, S.f32) + e)
    if g < S.convert(0, S.f32):
        sig = sig * e
    rounded_sig = S.convert(sig, S.bf16)
    return S.convert(S.convert(att, S.f32) * S.convert(rounded_sig, S.f32), S.bf16)


@avelang.jit
def causal_gqa_gate(
    Q: S.Tensor((16, 26, 2, 4, 256), S.bf16),
    K: S.Tensor((16, 26, 2, 256), S.bf16),
    V: S.Tensor((16, 26, 2, 256), S.bf16),
    Gate: S.Tensor((16, 26, 2, 4, 256), S.bf16),
    Out: S.Tensor((16, 26, 2, 4, 256), S.bf16),
):
    qpos = S.block_id(0)
    batch = S.block_id(2) // 4
    h = S.block_id(2) % 4
    group = S.block_id(1)
    lane = S.thread_id(0)
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
        for j in S.range(8, unroll=True):
            d = lane + 32 * j
            q = S.convert(Q[batch, qpos, group, h, d], S.f32)
            k = S.convert(K[batch, kpos, group, d], S.f32)
            score = S.nvvm.fast_fma(q, k, score)
        score = score + S.shuffle_xor(score, 16, 32)
        score = score + S.shuffle_xor(score, 8, 32)
        score = score + S.shuffle_xor(score, 4, 32)
        score = score + S.shuffle_xor(score, 2, 32)
        score = score + S.shuffle_xor(score, 1, 32)
        score = score * S.convert(0.0625, S.f32)
        newmax = S.nvvm.fast_fmax(maximum, score)
        alpha = S.nvvm.fast_exp2((maximum - newmax) * S.convert(1.4426950408889634, S.f32))
        beta = S.nvvm.fast_exp2((score - newmax) * S.convert(1.4426950408889634, S.f32))
        denom = denom * alpha + beta
        acc0 = acc0 * alpha + beta * S.convert(V[batch, kpos, group, lane], S.f32)
        acc1 = acc1 * alpha + beta * S.convert(V[batch, kpos, group, lane + 32], S.f32)
        acc2 = acc2 * alpha + beta * S.convert(V[batch, kpos, group, lane + 64], S.f32)
        acc3 = acc3 * alpha + beta * S.convert(V[batch, kpos, group, lane + 96], S.f32)
        acc4 = acc4 * alpha + beta * S.convert(V[batch, kpos, group, lane + 128], S.f32)
        acc5 = acc5 * alpha + beta * S.convert(V[batch, kpos, group, lane + 160], S.f32)
        acc6 = acc6 * alpha + beta * S.convert(V[batch, kpos, group, lane + 192], S.f32)
        acc7 = acc7 * alpha + beta * S.convert(V[batch, kpos, group, lane + 224], S.f32)
        maximum = newmax
    inv_denom = S.nvvm.fast_rcp(denom)
    Out[batch, qpos, group, h, lane] = gate_store(acc0 * inv_denom, Gate[batch, qpos, group, h, lane])
    Out[batch, qpos, group, h, lane + 32] = gate_store(acc1 * inv_denom, Gate[batch, qpos, group, h, lane + 32])
    Out[batch, qpos, group, h, lane + 64] = gate_store(acc2 * inv_denom, Gate[batch, qpos, group, h, lane + 64])
    Out[batch, qpos, group, h, lane + 96] = gate_store(acc3 * inv_denom, Gate[batch, qpos, group, h, lane + 96])
    Out[batch, qpos, group, h, lane + 128] = gate_store(acc4 * inv_denom, Gate[batch, qpos, group, h, lane + 128])
    Out[batch, qpos, group, h, lane + 160] = gate_store(acc5 * inv_denom, Gate[batch, qpos, group, h, lane + 160])
    Out[batch, qpos, group, h, lane + 192] = gate_store(acc6 * inv_denom, Gate[batch, qpos, group, h, lane + 192])
    Out[batch, qpos, group, h, lane + 224] = gate_store(acc7 * inv_denom, Gate[batch, qpos, group, h, lane + 224])


@avelang.jit
def add_residual(
    Hidden: S.Tensor((416, 2048), S.bf16),
    Projected: S.Tensor((416, 2048), S.bf16),
    Out: S.Tensor((416, 2048), S.bf16),
):
    m = S.block_id(0)
    d = S.block_id(1) * 256 + S.thread_id(0)
    Out[m, d] = S.convert(S.convert(Hidden[m, d], S.f32) + S.convert(Projected[m, d], S.f32), S.bf16)


def build(x, hidden, w, tables):
    """Allocate and bind seven launches; no host numerical tensor computation."""
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
    kernels = OrderedDict()
    kernels['qkv_projection'] = lambda: linear_native[lambda: ((40, 7, 1), (160, 1, 1))](
        x.view(416, 2048), w['qkv'], packed, 5120, num_warps=5)
    kernels['unpack_q_gate_k_v'] = lambda: unpack[lambda: ((416, 8, 1), (256, 1, 1))](
        packed, qraw, gate, kraw, v, num_warps=8)
    kernels['q_norm_rope'] = lambda: norm_rope[lambda: ((416, 8, 1), (32, 1, 1))](
        qraw, w['q_norm'], tables, q, 8, num_warps=1)
    kernels['k_norm_rope'] = lambda: norm_rope[lambda: ((416, 2, 1), (32, 1, 1))](
        kraw, w['k_norm'], tables, k, 2, num_warps=1)
    kernels['causal_gqa_gate'] = lambda: causal_gqa_gate[lambda: ((26, 2, 64), (32, 1, 1))](
        q.view(16, 26, 2, 4, 256), k.view(16, 26, 2, 256),
        v.view(16, 26, 2, 256), gate.view(16, 26, 2, 4, 256),
        attended.view(16, 26, 2, 4, 256), num_warps=1)
    kernels['output_projection'] = lambda: linear_native[lambda: ((16, 7, 1), (160, 1, 1))](
        attended.view(416, 2048), w['o'], projected.view(416, 2048), 2048, num_warps=5)
    kernels['residual'] = lambda: add_residual[lambda: ((416, 8, 1), (256, 1, 1))](
        hidden.view(416, 2048), projected.view(416, 2048), output.view(416, 2048), num_warps=8)
    return kernels, (attended, output)
