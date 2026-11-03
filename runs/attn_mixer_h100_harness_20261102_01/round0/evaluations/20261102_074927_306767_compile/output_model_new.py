"""Native AveLang attention mixer for B16/T26/H8/KV2/D256 on Hopper.

All arithmetic is in the returned GPU launches.  The two matrix products use
BF16 WGMMA operands and FP32 accumulators; attention uses FP32 online softmax
and FP32 value accumulation.  This native branch has no artifact invariant
validation API.  OFFLINE_COMPILE_CASES covers every launched specialization.
"""

from collections import OrderedDict
import faulthandler

faulthandler.enable()

import torch
import avelang
import avelang.language as S


@avelang.jit
def _linear(
    a_ptr: S.Pointer(S.bf16),
    w_ptr: S.Pointer(S.bf16),
    h_ptr: S.Pointer(S.bf16),
    o_ptr: S.Pointer(S.bf16),
    N: S.constexpr,
    RESIDUAL: S.constexpr,
):
    """64x64 nonpersistent WGMMA, with a four-stage TMA producer."""
    tid = S.thread_id(0)
    warp = tid // 32
    lane = tid % 32
    row_start = S.block_id(1) * 64
    col_start = S.block_id(0) * 64

    full = S.nvvm.mbarrier_create(4)
    empty = S.nvvm.mbarrier_create(4)
    b_shared = S.make_shared((4, 4096), S.bf16, 128)
    a_shared = S.make_shared((4, 4096), S.bf16, 128)

    a_tensor = S.make_tensor(
        a_ptr, S.bf16, S.make_layout((416, 32, 64), (2048, 64, 1))
    )
    b_tensor = S.make_tensor(
        w_ptr, S.bf16, S.make_layout((N, 32, 64), (2048, 64, 1))
    )
    out = S.make_tensor(
        o_ptr, S.bf16, S.make_layout((416, N), (N, 1))
    )
    hidden = S.make_tensor(
        h_ptr, S.bf16, S.make_layout((416, 2048), (2048, 1))
    )
    a_map = S.nvvm.make_tma_descriptor(
        a_tensor, S.make_layout((64, 1, 64), (64, 4096, 1)),
        S.WGMMA_SWIZZLE_128B,
    )
    b_map = S.nvvm.make_tma_descriptor(
        b_tensor, S.make_layout((64, 1, 64), (64, 4096, 1)),
        S.WGMMA_SWIZZLE_128B,
    )

    for stage in S.range(4, unroll=True):
        S.nvvm.mbarrier_init(full, stage, count=1, predicate=tid == 0)
        S.nvvm.mbarrier_init(empty, stage, count=1, predicate=tid == 0)
    S.nvvm.fence_mbarrier_init_release_cluster()
    S.syncthreads()

    if tid < 128:
        accumulators = S.full((1, 32), 0.0, S.f32)
        phase = 0
        for kt in S.range(32):
            stage = kt % 4
            S.nvvm.mbarrier_try_wait_parity(full, phase, 10000000, stage)
            a_matrix = S.view(
                S.subview(a_shared, (stage, 0), (1, 4096), (1, 1)),
                S.bf16, S.make_layout((64, 64), (64, 1)),
            )
            b_matrix = S.view(
                S.subview(b_shared, (stage, 0), (1, 4096), (1, 1)),
                S.bf16, S.make_layout((64, 64), (64, 1)),
            )
            da = S.nvvm.make_wgmma_descriptor_bits(
                a_matrix, S.WGMMA_SWIZZLE_128B, 0, 0, 0, 16, 1024
            )
            db = S.nvvm.make_wgmma_descriptor_bits(
                b_matrix, S.WGMMA_SWIZZLE_128B, 0, 0, 0, 16, 1024
            )
            acc = accumulators[0]
            S.nvvm.wgmma_fence_aligned()
            acc = S.nvvm.wgmma_m64n64k16_f32_bf16_bf16(da, db, acc, 1)
            acc = S.nvvm.wgmma_m64n64k16_f32_bf16_bf16(da + 2, db + 2, acc, 1)
            acc = S.nvvm.wgmma_m64n64k16_f32_bf16_bf16(da + 4, db + 4, acc, 1)
            acc = S.nvvm.wgmma_m64n64k16_f32_bf16_bf16(da + 6, db + 6, acc, 1)
            S.nvvm.wgmma_group_sync_aligned()
            S.nvvm.wgmma_wait_group_sync(0)
            S.nvvm.wgmma_fence_aligned()
            for i in S.range(32, unroll=True):
                accumulators[0, i] = acc[i]
            if tid == 0:
                S.nvvm.mbarrier_arrive(empty, stage)
            if stage == 3:
                phase = phase ^ 1

        row0 = row_start + warp * 16 + lane // 4
        row1 = row0 + 8
        for group in S.range(8, unroll=True):
            col = col_start + group * 8 + (lane % 4) * 2
            v0 = S.convert(S.convert(accumulators[0, group * 4], S.bf16), S.f32)
            v1 = S.convert(S.convert(accumulators[0, group * 4 + 1], S.bf16), S.f32)
            v2 = S.convert(S.convert(accumulators[0, group * 4 + 2], S.bf16), S.f32)
            v3 = S.convert(S.convert(accumulators[0, group * 4 + 3], S.bf16), S.f32)
            if row0 < 416:
                if RESIDUAL == 1:
                    v0 = v0 + S.convert(hidden[row0, col], S.f32)
                    v1 = v1 + S.convert(hidden[row0, col + 1], S.f32)
                out[row0, col] = S.convert(v0, S.bf16)
                out[row0, col + 1] = S.convert(v1, S.bf16)
            if row1 < 416:
                if RESIDUAL == 1:
                    v2 = v2 + S.convert(hidden[row1, col], S.f32)
                    v3 = v3 + S.convert(hidden[row1, col + 1], S.f32)
                out[row1, col] = S.convert(v2, S.bf16)
                out[row1, col + 1] = S.convert(v3, S.bf16)
    else:
        elected = S.nvvm.elect_sync()
        phase = 1
        for kt in S.range(32):
            stage = kt % 4
            S.nvvm.mbarrier_try_wait_parity(empty, phase, 10000000, stage)
            S.nvvm.mbarrier_arrive_expect_tx(full, 16384, stage, elected)
            a_dst = S.view(
                S.subview(a_shared, (stage, 0), (1, 4096), (1, 1)),
                S.bf16, S.make_layout((64, 1, 64), (64, 4096, 1)),
            )
            b_dst = S.view(
                S.subview(b_shared, (stage, 0), (1, 4096), (1, 1)),
                S.bf16, S.make_layout((64, 1, 64), (64, 4096, 1)),
            )
            S.nvvm.tma_load(
                a_dst, a_map, (0, kt, row_start), full,
                mbar_id=stage, predicate=elected, expect_tx=False,
            )
            S.nvvm.tma_load(
                b_dst, b_map, (0, kt, col_start), full,
                mbar_id=stage, predicate=elected, expect_tx=False,
            )
            if stage == 3:
                phase = phase ^ 1
    S.syncthreads()


@avelang.jit
def _norm_rope(
    packed: S.Tensor((416, 5120), S.bf16),
    q_weight: S.Tensor((256,), S.bf16),
    k_weight: S.Tensor((256,), S.bf16),
    tables: S.Tensor((26, 32, 2), S.bf16),
    q: S.Tensor((416, 8, 256), S.bf16),
    k: S.Tensor((416, 2, 256), S.bf16),
):
    lane = S.thread_id(0) % 32
    work = S.block_id(0) * 4 + S.thread_id(0) // 32
    token = work // 10
    head = work % 10
    offset = head * 512
    if head >= 8:
        offset = 4096 + (head - 8) * 256

    values = S.make_local((8,), S.f32)
    squared = S.convert(0.0, S.f32)
    for i in S.range(8, unroll=True):
        v = S.convert(packed[token, offset + lane + i * 32], S.f32)
        values[i] = v
        squared = S.fma(v, v, squared)
    for shift in S.range(5, unroll=True):
        squared = squared + S.shuffle_xor(squared, 1 << shift, 32)
    scale = S.nvvm.fast_rsqrt(squared * 0.00390625 + 0.000001)
    for i in S.range(8, unroll=True):
        d = lane + i * 32
        weight = S.convert(q_weight[d], S.f32)
        if head >= 8:
            weight = S.convert(k_weight[d], S.f32)
        values[i] = S.convert(
            S.convert(values[i] * scale * (1.0 + weight), S.bf16), S.f32
        )
    a = values[0]
    b = values[1]
    cos = S.convert(tables[token % 26, lane, 0], S.f32)
    sin = S.convert(tables[token % 26, lane, 1], S.f32)
    values[0] = a * cos - b * sin
    values[1] = b * cos + a * sin
    for i in S.range(8, unroll=True):
        d = lane + i * 32
        if head < 8:
            q[token, head, d] = S.convert(values[i], S.bf16)
        else:
            k[token, head - 8, d] = S.convert(values[i], S.bf16)


@avelang.jit
def _attention_gate(
    packed: S.Tensor((416, 5120), S.bf16),
    q: S.Tensor((416, 8, 256), S.bf16),
    k: S.Tensor((416, 2, 256), S.bf16),
    attended: S.Tensor((416, 8, 256), S.bf16),
):
    lane = S.thread_id(0) % 32
    work = S.block_id(0) * 4 + S.thread_id(0) // 32
    token = work // 8
    head = work % 8
    kv_head = head // 4
    batch_start = (token // 26) * 26
    position = token % 26
    query = S.make_local((8,), S.f32)
    result = S.full((8,), 0.0, S.f32)
    for i in S.range(8, unroll=True):
        query[i] = S.convert(q[token, head, lane + i * 32], S.f32)
    maximum = S.convert(-3.4028234663852886e38, S.f32)
    denominator = S.convert(0.0, S.f32)
    for key_pos in S.range(position + 1):
        key_token = batch_start + key_pos
        dot = S.convert(0.0, S.f32)
        for i in S.range(8, unroll=True):
            key = S.convert(k[key_token, kv_head, lane + i * 32], S.f32)
            dot = S.fma(query[i], key, dot)
        for shift in S.range(5, unroll=True):
            dot = dot + S.shuffle_xor(dot, 1 << shift, 32)
        score = dot * 0.0625
        new_max = S.nvvm.fast_fmax(maximum, score)
        alpha = S.nvvm.fast_exp2((maximum - new_max) * 1.4426950408889634)
        probability = S.nvvm.fast_exp2((score - new_max) * 1.4426950408889634)
        denominator = S.fma(denominator, alpha, probability)
        for i in S.range(8, unroll=True):
            v = S.convert(packed[key_token, 4608 + kv_head * 256 + lane + i * 32], S.f32)
            result[i] = S.fma(v, probability, result[i] * alpha)
        maximum = new_max
    inverse = S.nvvm.fast_rcp(denominator)
    for i in S.range(8, unroll=True):
        d = lane + i * 32
        attention = S.convert(S.convert(result[i] * inverse, S.bf16), S.f32)
        gate = S.convert(packed[token, head * 512 + 256 + d], S.f32)
        sigmoid = S.nvvm.fast_rcp(1.0 + S.nvvm.fast_exp2(-gate * 1.4426950408889634))
        sigmoid = S.convert(S.convert(sigmoid, S.bf16), S.f32)
        attended[token, head, d] = S.convert(attention * sigmoid, S.bf16)


OFFLINE_COMPILE_CASES = [
    ("norm_rope", _norm_rope, {}, 4),
    ("attention_gate", _attention_gate, {}, 4),
    ("qkv", _linear, {"N": 5120, "RESIDUAL": 0}, 5),
    ("projection_residual", _linear, {"N": 2048, "RESIDUAL": 1}, 5),
]


def build(x, hidden, w, tables):
    packed = torch.empty((416, 5120), dtype=torch.bfloat16, device=x.device)
    q = torch.empty((416, 8, 256), dtype=torch.bfloat16, device=x.device)
    k = torch.empty((416, 2, 256), dtype=torch.bfloat16, device=x.device)
    attended = torch.empty((16, 26, 2048), dtype=torch.bfloat16, device=x.device)
    output = torch.empty_like(hidden)
    x_view = x.view(416, 2048)
    hidden_view = hidden.view(416, 2048)
    attended_view = attended.view(416, 8, 256)
    output_view = output.view(416, 2048)

    def qkv():
        _linear[lambda: ((80, 7, 1), (160, 1, 1))](
            x_view, w["qkv"], hidden_view, packed,
            N=5120, RESIDUAL=0, num_warps=5,
        )

    def norm_rope():
        _norm_rope[lambda: ((1040, 1, 1), (128, 1, 1))](
            packed, w["q_norm"], w["k_norm"], tables, q, k, num_warps=4,
        )

    def attention_gate():
        _attention_gate[lambda: ((832, 1, 1), (128, 1, 1))](
            packed, q, k, attended_view, num_warps=4,
        )

    def projection_residual():
        _linear[lambda: ((32, 7, 1), (160, 1, 1))](
            attended, w["o"], hidden_view, output_view,
            N=2048, RESIDUAL=1, num_warps=5,
        )

    return OrderedDict([
        ("qkv", qkv),
        ("norm_rope", norm_rope),
        ("attention_gate", attention_gate),
        ("projection_residual", projection_residual),
    ]), (attended, output)
