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
def linear_round2(
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
    # Store only valid rows, including the final 32-row output tile.
    c_tma = S.nvvm.make_tma_descriptor(
        C, S.make_layout((64, 64), (64, 1)), SWIZZLE_128B)
    c_tail_tma = S.nvvm.make_tma_descriptor(
        C, S.make_layout((32, 64), (64, 1)), SWIZZLE_128B)
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
        # All 32 input transactions have completed their full-barrier waits,
        # and the last WGMMA group has completed. There are no later producer
        # stores. Join all consumers before reusing B stage 0 as output scratch.
        S.nvvm.named_barrier_sync(1, 128)
        for group in S.range(8, unroll=True):
            packed = S.full((1, 4), 0, S.i32)
            for matrix in S.range(4, unroll=True):
                i = matrix & 1
                w = 2 * group + matrix // 2
                packed[0, matrix] = S.nvvm.floatx2_to_bf16x2(
                    accumulators[0, 4 * w + 2 * i],
                    accumulators[0, 4 * w + 2 * i + 1])

            # WGMMA register fragments -> two contiguous 64x64 boxes, each
            # using TMA's 128-byte swizzle. The 8192 BF16 elements exactly fit
            # b_shared stage 0; stages 1/2 and the barrier storage are untouched.
            address_matrix = lane // 8
            address_i = address_matrix & 1
            address_w = 2 * group + address_matrix // 2
            box_n = address_w // 8
            local_address_w = address_w % 8
            local_row = warp * 16 + address_i * 8 + lane % 8
            logical_offset = box_n * 4096 + local_row * 64 + local_address_w * 8
            byte_offset = logical_offset * 2
            swizzled_offset = (byte_offset ^ ((byte_offset & 0x380) >> 3)) // 2
            destination = S.subview(b_shared, (0, swizzled_offset), (8, 8), (1, 1))
            S.nvvm.stmatrix_m8n8_x4_b16(destination, packed[0])

        # Publish every writer's generic shared stores to the async proxy,
        # then join the consumer warpgroup before its elected TMA issuer runs.
        S.nvvm.fence_proxy_async_shared_cta()
        S.nvvm.named_barrier_sync(1, 128)
        if tid == 0:
            for box_n in S.range(2, unroll=True):
                if row0 < 384:
                    box_raw = S.subview(b_shared, (0, box_n * 4096), (1, 4096), (1, 1))
                    box = S.view(box_raw, S.bf16, S.make_layout((64, 64), (64, 1)))
                    S.nvvm.tma_store(box, c_tma, (col0 + box_n * 64, row0), predicate=tid == 0)
                else:
                    tail_raw = S.subview(b_shared, (0, box_n * 4096), (1, 2048), (1, 1))
                    tail = S.view(tail_raw, S.bf16, S.make_layout((32, 64), (64, 1)))
                    S.nvvm.tma_store(tail, c_tail_tma, (col0 + box_n * 64, row0), predicate=tid == 0)
            S.nvvm.cp_async_bulk_commit_group()
            # Full completion (not just shared-read completion) before exit.
            S.nvvm.cp_async_bulk_wait_group(0)
        S.nvvm.named_barrier_sync(1, 128)
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
def linear_swap64x128(
    A: S.Tensor((416, 2048), S.bf16),
    W: S.Tensor((N, 2048), S.bf16),
    C: S.Tensor((416, N), S.bf16),
    N: S.constexpr,
    WAIT_GROUP: S.constexpr,
):
    # W[channel,K] x A[token,K]^T; transpose only in the output epilogue.
    tid = S.thread_id(0)
    wg = tid // 128
    lane128 = tid % 128
    warp = lane128 // 32
    lane = lane128 % 32
    token0 = S.block_id(1) * 64
    channel0 = S.block_id(0) * 128
    full = S.nvvm.mbarrier_create(4)
    empty = S.nvvm.mbarrier_create(4)
    token_shared = S.make_shared((4, 4096), S.bf16, 128)
    weight_shared = S.make_shared((4, 8192), S.bf16, 128)
    # No extra epilogue allocation. Both consumers must finish before this
    # arena can change ownership from WGMMA input to transposed output.
    output_shared = S.view(weight_shared, S.bf16,
                           S.make_layout((1, 32768), (32768, 1)))
    token_tensor = S.view(A, S.bf16, S.make_layout((416, 32, 64), (2048, 64, 1)))
    weight_tensor = S.view(W, S.bf16, S.make_layout((N, 32, 64), (2048, 64, 1)))
    token_tma = S.nvvm.make_tma_descriptor(
        token_tensor, S.make_layout((64, 1, 64), (64, 4096, 1)), SWIZZLE_128B)
    weight_tma = S.nvvm.make_tma_descriptor(
        weight_tensor, S.make_layout((128, 1, 64), (64, 8192, 1)), SWIZZLE_128B)
    output_tma = S.nvvm.make_tma_descriptor(
        C, S.make_layout((64, 64), (64, 1)), SWIZZLE_128B)
    tail_tma = S.nvvm.make_tma_descriptor(
        C, S.make_layout((32, 64), (64, 1)), SWIZZLE_128B)
    for stage in S.range(4, unroll=True):
        S.nvvm.mbarrier_init(full, stage, count=1, predicate=tid == 0)
        S.nvvm.mbarrier_init(empty, stage, count=2, predicate=tid == 0)
    S.nvvm.fence_mbarrier_init_release_cluster()
    S.syncthreads()

    if tid < 256:
        accumulators = S.make_local((1, 32), S.f32)
        for element in S.range(32, unroll=True):
            accumulators[0, element] = S.convert(0, S.f32)
        for kt in S.range(32):
            stage = kt % 4
            phase = (kt // 4) % 2
            S.nvvm.mbarrier_try_wait_parity(full, phase, 10000000, stage)
            weights = S.view(
                S.subview(weight_shared, (stage, wg * 4096), (1, 4096), (1, 1)),
                S.bf16, S.make_layout((64, 64), (64, 1)))
            tokens = S.view(
                S.subview(token_shared, (stage, 0), (1, 4096), (1, 1)),
                S.bf16, S.make_layout((64, 64), (64, 1)))
            da = S.nvvm.make_wgmma_descriptor_bits(weights, SWIZZLE_128B, 0, 0, 0, 16, 1024)
            db = S.nvvm.make_wgmma_descriptor_bits(tokens, SWIZZLE_128B, 0, 0, 0, 16, 1024)
            acc = accumulators[0]
            S.nvvm.wgmma_fence_aligned()
            acc = S.nvvm.wgmma_m64n64k16_f32_bf16_bf16(da, db, acc, 1)
            acc = S.nvvm.wgmma_m64n64k16_f32_bf16_bf16(da + 2, db + 2, acc, 1)
            acc = S.nvvm.wgmma_m64n64k16_f32_bf16_bf16(da + 4, db + 4, acc, 1)
            acc = S.nvvm.wgmma_m64n64k16_f32_bf16_bf16(da + 6, db + 6, acc, 1)
            S.nvvm.wgmma_group_sync_aligned()
            S.nvvm.wgmma_wait_group_sync(WAIT_GROUP)
            for element in S.range(32, unroll=True):
                accumulators[0, element] = acc[element]
            if WAIT_GROUP == 0:
                if lane128 == 0:
                    S.nvvm.mbarrier_arrive(empty, stage)
            else:
                if kt > 0 and lane128 == 0:
                    S.nvvm.mbarrier_arrive(empty, (kt - 1) % 4)
        S.nvvm.wgmma_wait_group_sync(0)
        S.nvvm.wgmma_fence_aligned()
        if WAIT_GROUP == 1 and lane128 == 0:
            S.nvvm.mbarrier_arrive(empty, 3)

        # All consumer WGs finish every stage before any of them aliases the
        # shared weight arena. The producer has issued exactly 32 input tiles.
        S.nvvm.named_barrier_sync(1, 256)
        for token_box in S.range(2, unroll=True):
            for local_group in S.range(2, unroll=True):
                packed = S.full((1, 4), 0, S.i32)
                source_group = 2 * token_box + local_group
                for matrix in S.range(4, unroll=True):
                    i = matrix & 1
                    source_w = 2 * source_group + matrix // 2
                    packed[0, matrix] = S.nvvm.floatx2_to_bf16x2(
                        accumulators[0, 4 * source_w + 2 * i],
                        accumulators[0, 4 * source_w + 2 * i + 1])
                address_matrix = lane // 8
                address_i = address_matrix & 1
                local_address_w = 2 * local_group + address_matrix // 2
                local_token = token_box * 32 + local_address_w * 8 + lane % 8
                local_channel = warp * 16 + address_i * 8
                logical_byte_offset = (local_token * 64 + local_channel) * 2
                swizzled_byte_offset = logical_byte_offset ^ ((logical_byte_offset & 0x380) >> 3)
                destination = S.subview(
                    output_shared, (0, wg * 4096 + swizzled_byte_offset // 2),
                    (8, 8), (1, 1))
                S.nvvm.stmatrix_m8n8_x4_b16_trans(destination, packed[0])
        S.nvvm.fence_proxy_async_shared_cta()
        S.nvvm.named_barrier_sync(2 + wg, 128)
        if lane128 == 0:
            if token0 <= 352:
                out_raw = S.subview(output_shared, (0, wg * 4096), (1, 4096), (1, 1))
                out_tile = S.view(out_raw, S.bf16, S.make_layout((64, 64), (64, 1)))
                S.nvvm.tma_store(out_tile, output_tma, (channel0 + wg * 64, token0), predicate=lane128 == 0)
            else:
                tail_raw = S.subview(output_shared, (0, wg * 4096), (1, 2048), (1, 1))
                tail_tile = S.view(tail_raw, S.bf16, S.make_layout((32, 64), (64, 1)))
                S.nvvm.tma_store(tail_tile, tail_tma, (channel0 + wg * 64, token0), predicate=lane128 == 0)
            S.nvvm.cp_async_bulk_commit_group()
            S.nvvm.cp_async_bulk_wait_group(0)
        S.nvvm.named_barrier_sync(2 + wg, 128)
    else:
        elected = S.nvvm.elect_sync()
        for kt in S.range(32):
            stage = kt % 4
            phase = 1 ^ ((kt // 4) % 2)
            S.nvvm.mbarrier_try_wait_parity(empty, phase, 10000000, stage)
            S.nvvm.mbarrier_arrive_expect_tx(full, 24576, stage, elected)
            weights_raw = S.subview(weight_shared, (stage, 0), (1, 8192), (1, 1))
            weights_dst = S.view(weights_raw, S.bf16,
                                 S.make_layout((128, 1, 64), (64, 8192, 1)))
            tokens_raw = S.subview(token_shared, (stage, 0), (1, 4096), (1, 1))
            tokens_dst = S.view(tokens_raw, S.bf16,
                                S.make_layout((64, 1, 64), (64, 4096, 1)))
            S.nvvm.tma_load(weights_dst, weight_tma, (0, kt, channel0), full,
                            mbar_id=stage, predicate=elected, expect_tx=False)
            S.nvvm.tma_load(tokens_dst, token_tma, (0, kt, token0), full,
                            mbar_id=stage, predicate=elected, expect_tx=False)
    S.syncthreads()


@avelang.jit
def linear_swap160x128(
    A: S.Tensor((416, 2048), S.bf16),
    W: S.Tensor((N, 2048), S.bf16),
    C: S.Tensor((416, N), S.bf16),
    N: S.constexpr,
    WAIT_GROUP: S.constexpr,
):
    # W[channel,K] x A[token,K]^T; transpose only in the output epilogue.
    tid = S.thread_id(0)
    wg = tid // 128
    lane128 = tid % 128
    warp = lane128 // 32
    lane = lane128 % 32
    token0 = S.block_id(1) * 160
    channel0 = S.block_id(0) * 128
    full = S.nvvm.mbarrier_create(4)
    empty = S.nvvm.mbarrier_create(4)
    token_shared = S.make_shared((4, 10240), S.bf16, 128)
    weight_shared = S.make_shared((4, 8192), S.bf16, 128)
    # No extra epilogue allocation. Both consumers must finish before this
    # arena can change ownership from WGMMA input to transposed output.
    output_shared = S.view(weight_shared, S.bf16,
                           S.make_layout((1, 32768), (32768, 1)))
    token_tensor = S.view(A, S.bf16, S.make_layout((416, 32, 64), (2048, 64, 1)))
    weight_tensor = S.view(W, S.bf16, S.make_layout((N, 32, 64), (2048, 64, 1)))
    token_tma = S.nvvm.make_tma_descriptor(
        token_tensor, S.make_layout((160, 1, 64), (64, 10240, 1)), SWIZZLE_128B)
    weight_tma = S.nvvm.make_tma_descriptor(
        weight_tensor, S.make_layout((128, 1, 64), (64, 8192, 1)), SWIZZLE_128B)
    output_tma = S.nvvm.make_tma_descriptor(
        C, S.make_layout((160, 64), (64, 1)), SWIZZLE_128B)
    tail_tma = S.nvvm.make_tma_descriptor(
        C, S.make_layout((96, 64), (64, 1)), SWIZZLE_128B)
    for stage in S.range(4, unroll=True):
        S.nvvm.mbarrier_init(full, stage, count=1, predicate=tid == 0)
        S.nvvm.mbarrier_init(empty, stage, count=2, predicate=tid == 0)
    S.nvvm.fence_mbarrier_init_release_cluster()
    S.syncthreads()

    if tid < 256:
        accumulators = S.make_local((1, 80), S.f32)
        for element in S.range(80, unroll=True):
            accumulators[0, element] = S.convert(0, S.f32)
        for kt in S.range(32):
            stage = kt % 4
            phase = (kt // 4) % 2
            S.nvvm.mbarrier_try_wait_parity(full, phase, 10000000, stage)
            weights = S.view(
                S.subview(weight_shared, (stage, wg * 4096), (1, 4096), (1, 1)),
                S.bf16, S.make_layout((64, 64), (64, 1)))
            tokens = S.view(
                S.subview(token_shared, (stage, 0), (1, 10240), (1, 1)),
                S.bf16, S.make_layout((160, 64), (64, 1)))
            da = S.nvvm.make_wgmma_descriptor_bits(weights, SWIZZLE_128B, 0, 0, 0, 16, 1024)
            db = S.nvvm.make_wgmma_descriptor_bits(tokens, SWIZZLE_128B, 0, 0, 0, 16, 1024)
            acc = accumulators[0]
            S.nvvm.wgmma_fence_aligned()
            acc = S.nvvm.wgmma_m64n160k16_f32_bf16_bf16(da, db, acc, 1)
            acc = S.nvvm.wgmma_m64n160k16_f32_bf16_bf16(da + 2, db + 2, acc, 1)
            acc = S.nvvm.wgmma_m64n160k16_f32_bf16_bf16(da + 4, db + 4, acc, 1)
            acc = S.nvvm.wgmma_m64n160k16_f32_bf16_bf16(da + 6, db + 6, acc, 1)
            S.nvvm.wgmma_group_sync_aligned()
            S.nvvm.wgmma_wait_group_sync(WAIT_GROUP)
            for element in S.range(80, unroll=True):
                accumulators[0, element] = acc[element]
            if WAIT_GROUP == 0:
                if lane128 == 0:
                    S.nvvm.mbarrier_arrive(empty, stage)
            else:
                if kt > 0 and lane128 == 0:
                    S.nvvm.mbarrier_arrive(empty, (kt - 1) % 4)
        S.nvvm.wgmma_wait_group_sync(0)
        S.nvvm.wgmma_fence_aligned()
        if WAIT_GROUP == 1 and lane128 == 0:
            S.nvvm.mbarrier_arrive(empty, 3)

        # All consumer WGs finish every stage before any of them aliases the
        # shared weight arena. The producer has issued exactly 32 input tiles.
        S.nvvm.named_barrier_sync(1, 256)
        for token_box in S.range(5, unroll=True):
            for local_group in S.range(2, unroll=True):
                packed = S.full((1, 4), 0, S.i32)
                source_group = 2 * token_box + local_group
                for matrix in S.range(4, unroll=True):
                    i = matrix & 1
                    source_w = 2 * source_group + matrix // 2
                    packed[0, matrix] = S.nvvm.floatx2_to_bf16x2(
                        accumulators[0, 4 * source_w + 2 * i],
                        accumulators[0, 4 * source_w + 2 * i + 1])
                address_matrix = lane // 8
                address_i = address_matrix & 1
                local_address_w = 2 * local_group + address_matrix // 2
                local_token = token_box * 32 + local_address_w * 8 + lane % 8
                local_channel = warp * 16 + address_i * 8
                logical_byte_offset = (local_token * 64 + local_channel) * 2
                swizzled_byte_offset = logical_byte_offset ^ ((logical_byte_offset & 0x380) >> 3)
                destination = S.subview(
                    output_shared, (0, wg * 10240 + swizzled_byte_offset // 2),
                    (8, 8), (1, 1))
                S.nvvm.stmatrix_m8n8_x4_b16_trans(destination, packed[0])
        S.nvvm.fence_proxy_async_shared_cta()
        S.nvvm.named_barrier_sync(2 + wg, 128)
        if lane128 == 0:
            if token0 <= 256:
                out_raw = S.subview(output_shared, (0, wg * 10240), (1, 10240), (1, 1))
                out_tile = S.view(out_raw, S.bf16, S.make_layout((160, 64), (64, 1)))
                S.nvvm.tma_store(out_tile, output_tma, (channel0 + wg * 64, token0), predicate=lane128 == 0)
            else:
                tail_raw = S.subview(output_shared, (0, wg * 10240), (1, 6144), (1, 1))
                tail_tile = S.view(tail_raw, S.bf16, S.make_layout((96, 64), (64, 1)))
                S.nvvm.tma_store(tail_tile, tail_tma, (channel0 + wg * 64, token0), predicate=lane128 == 0)
            S.nvvm.cp_async_bulk_commit_group()
            S.nvvm.cp_async_bulk_wait_group(0)
        S.nvvm.named_barrier_sync(2 + wg, 128)
    else:
        elected = S.nvvm.elect_sync()
        for kt in S.range(32):
            stage = kt % 4
            phase = 1 ^ ((kt // 4) % 2)
            S.nvvm.mbarrier_try_wait_parity(empty, phase, 10000000, stage)
            S.nvvm.mbarrier_arrive_expect_tx(full, 36864, stage, elected)
            weights_raw = S.subview(weight_shared, (stage, 0), (1, 8192), (1, 1))
            weights_dst = S.view(weights_raw, S.bf16,
                                 S.make_layout((128, 1, 64), (64, 8192, 1)))
            tokens_raw = S.subview(token_shared, (stage, 0), (1, 10240), (1, 1))
            tokens_dst = S.view(tokens_raw, S.bf16,
                                S.make_layout((160, 1, 64), (64, 10240, 1)))
            S.nvvm.tma_load(weights_dst, weight_tma, (0, kt, channel0), full,
                            mbar_id=stage, predicate=elected, expect_tx=False)
            S.nvvm.tma_load(tokens_dst, token_tma, (0, kt, token0), full,
                            mbar_id=stage, predicate=elected, expect_tx=False)
    S.syncthreads()


@avelang.jit
def linear_swap128x64(
    A: S.Tensor((416, 2048), S.bf16),
    W: S.Tensor((N, 2048), S.bf16),
    C: S.Tensor((416, N), S.bf16),
    N: S.constexpr,
    WAIT_GROUP: S.constexpr,
):
    # W[channel,K] x A[token,K]^T; transpose only in the output epilogue.
    tid = S.thread_id(0)
    wg = tid // 128
    lane128 = tid % 128
    warp = lane128 // 32
    lane = lane128 % 32
    token0 = S.block_id(1) * 128
    channel0 = S.block_id(0) * 64
    full = S.nvvm.mbarrier_create(4)
    empty = S.nvvm.mbarrier_create(4)
    token_shared = S.make_shared((4, 8192), S.bf16, 128)
    weight_shared = S.make_shared((4, 4096), S.bf16, 128)
    # No extra epilogue allocation. Both consumers must finish before this
    # arena can change ownership from WGMMA input to transposed output.
    output_shared = S.view(weight_shared, S.bf16,
                           S.make_layout((1, 16384), (16384, 1)))
    token_tensor = S.view(A, S.bf16, S.make_layout((416, 32, 64), (2048, 64, 1)))
    weight_tensor = S.view(W, S.bf16, S.make_layout((N, 32, 64), (2048, 64, 1)))
    token_tma = S.nvvm.make_tma_descriptor(
        token_tensor, S.make_layout((128, 1, 64), (64, 8192, 1)), SWIZZLE_128B)
    weight_tma = S.nvvm.make_tma_descriptor(
        weight_tensor, S.make_layout((64, 1, 64), (64, 4096, 1)), SWIZZLE_128B)
    output_tma = S.nvvm.make_tma_descriptor(
        C, S.make_layout((128, 64), (64, 1)), SWIZZLE_128B)
    tail_tma = S.nvvm.make_tma_descriptor(
        C, S.make_layout((32, 64), (64, 1)), SWIZZLE_128B)
    for stage in S.range(4, unroll=True):
        S.nvvm.mbarrier_init(full, stage, count=1, predicate=tid == 0)
        S.nvvm.mbarrier_init(empty, stage, count=1, predicate=tid == 0)
    S.nvvm.fence_mbarrier_init_release_cluster()
    S.syncthreads()

    if tid < 128:
        accumulators = S.make_local((1, 64), S.f32)
        for element in S.range(64, unroll=True):
            accumulators[0, element] = S.convert(0, S.f32)
        for kt in S.range(32):
            stage = kt % 4
            phase = (kt // 4) % 2
            S.nvvm.mbarrier_try_wait_parity(full, phase, 10000000, stage)
            weights = S.view(
                S.subview(weight_shared, (stage, wg * 4096), (1, 4096), (1, 1)),
                S.bf16, S.make_layout((64, 64), (64, 1)))
            tokens = S.view(
                S.subview(token_shared, (stage, 0), (1, 8192), (1, 1)),
                S.bf16, S.make_layout((128, 64), (64, 1)))
            da = S.nvvm.make_wgmma_descriptor_bits(weights, SWIZZLE_128B, 0, 0, 0, 16, 1024)
            db = S.nvvm.make_wgmma_descriptor_bits(tokens, SWIZZLE_128B, 0, 0, 0, 16, 1024)
            acc = accumulators[0]
            S.nvvm.wgmma_fence_aligned()
            acc = S.nvvm.wgmma_m64n128k16_f32_bf16_bf16(da, db, acc, 1)
            acc = S.nvvm.wgmma_m64n128k16_f32_bf16_bf16(da + 2, db + 2, acc, 1)
            acc = S.nvvm.wgmma_m64n128k16_f32_bf16_bf16(da + 4, db + 4, acc, 1)
            acc = S.nvvm.wgmma_m64n128k16_f32_bf16_bf16(da + 6, db + 6, acc, 1)
            S.nvvm.wgmma_group_sync_aligned()
            S.nvvm.wgmma_wait_group_sync(WAIT_GROUP)
            for element in S.range(64, unroll=True):
                accumulators[0, element] = acc[element]
            if WAIT_GROUP == 0:
                if lane128 == 0:
                    S.nvvm.mbarrier_arrive(empty, stage)
            else:
                if kt > 0 and lane128 == 0:
                    S.nvvm.mbarrier_arrive(empty, (kt - 1) % 4)
        S.nvvm.wgmma_wait_group_sync(0)
        S.nvvm.wgmma_fence_aligned()
        if WAIT_GROUP == 1 and lane128 == 0:
            S.nvvm.mbarrier_arrive(empty, 3)

        # All consumer WGs finish every stage before any of them aliases the
        # shared weight arena. The producer has issued exactly 32 input tiles.
        S.nvvm.named_barrier_sync(1, 128)
        for token_box in S.range(4, unroll=True):
            for local_group in S.range(2, unroll=True):
                packed = S.full((1, 4), 0, S.i32)
                source_group = 2 * token_box + local_group
                for matrix in S.range(4, unroll=True):
                    i = matrix & 1
                    source_w = 2 * source_group + matrix // 2
                    packed[0, matrix] = S.nvvm.floatx2_to_bf16x2(
                        accumulators[0, 4 * source_w + 2 * i],
                        accumulators[0, 4 * source_w + 2 * i + 1])
                address_matrix = lane // 8
                address_i = address_matrix & 1
                local_address_w = 2 * local_group + address_matrix // 2
                local_token = token_box * 32 + local_address_w * 8 + lane % 8
                local_channel = warp * 16 + address_i * 8
                logical_byte_offset = (local_token * 64 + local_channel) * 2
                swizzled_byte_offset = logical_byte_offset ^ ((logical_byte_offset & 0x380) >> 3)
                destination = S.subview(
                    output_shared, (0, wg * 8192 + swizzled_byte_offset // 2),
                    (8, 8), (1, 1))
                S.nvvm.stmatrix_m8n8_x4_b16_trans(destination, packed[0])
        S.nvvm.fence_proxy_async_shared_cta()
        S.nvvm.named_barrier_sync(2 + wg, 128)
        if lane128 == 0:
            if token0 <= 288:
                out_raw = S.subview(output_shared, (0, wg * 8192), (1, 8192), (1, 1))
                out_tile = S.view(out_raw, S.bf16, S.make_layout((128, 64), (64, 1)))
                S.nvvm.tma_store(out_tile, output_tma, (channel0 + wg * 64, token0), predicate=lane128 == 0)
            else:
                tail_raw = S.subview(output_shared, (0, wg * 8192), (1, 2048), (1, 1))
                tail_tile = S.view(tail_raw, S.bf16, S.make_layout((32, 64), (64, 1)))
                S.nvvm.tma_store(tail_tile, tail_tma, (channel0 + wg * 64, token0), predicate=lane128 == 0)
            S.nvvm.cp_async_bulk_commit_group()
            S.nvvm.cp_async_bulk_wait_group(0)
        S.nvvm.named_barrier_sync(2 + wg, 128)
    else:
        elected = S.nvvm.elect_sync()
        for kt in S.range(32):
            stage = kt % 4
            phase = 1 ^ ((kt // 4) % 2)
            S.nvvm.mbarrier_try_wait_parity(empty, phase, 10000000, stage)
            S.nvvm.mbarrier_arrive_expect_tx(full, 24576, stage, elected)
            weights_raw = S.subview(weight_shared, (stage, 0), (1, 4096), (1, 1))
            weights_dst = S.view(weights_raw, S.bf16,
                                 S.make_layout((64, 1, 64), (64, 4096, 1)))
            tokens_raw = S.subview(token_shared, (stage, 0), (1, 8192), (1, 1))
            tokens_dst = S.view(tokens_raw, S.bf16,
                                S.make_layout((128, 1, 64), (64, 8192, 1)))
            S.nvvm.tma_load(weights_dst, weight_tma, (0, kt, channel0), full,
                            mbar_id=stage, predicate=elected, expect_tx=False)
            S.nvvm.tma_load(tokens_dst, token_tma, (0, kt, token0), full,
                            mbar_id=stage, predicate=elected, expect_tx=False)
    S.syncthreads()


@avelang.jit
def linear_swap96x64(
    A: S.Tensor((416, 2048), S.bf16),
    W: S.Tensor((N, 2048), S.bf16),
    C: S.Tensor((416, N), S.bf16),
    N: S.constexpr,
    WAIT_GROUP: S.constexpr,
):
    # W[channel,K] x A[token,K]^T; transpose only in the output epilogue.
    tid = S.thread_id(0)
    wg = tid // 128
    lane128 = tid % 128
    warp = lane128 // 32
    lane = lane128 % 32
    token0 = S.block_id(1) * 96
    channel0 = S.block_id(0) * 64
    full = S.nvvm.mbarrier_create(4)
    empty = S.nvvm.mbarrier_create(4)
    token_shared = S.make_shared((4, 6144), S.bf16, 128)
    weight_shared = S.make_shared((4, 4096), S.bf16, 128)
    # No extra epilogue allocation. Both consumers must finish before this
    # arena can change ownership from WGMMA input to transposed output.
    output_shared = S.view(weight_shared, S.bf16,
                           S.make_layout((1, 16384), (16384, 1)))
    token_tensor = S.view(A, S.bf16, S.make_layout((416, 32, 64), (2048, 64, 1)))
    weight_tensor = S.view(W, S.bf16, S.make_layout((N, 32, 64), (2048, 64, 1)))
    token_tma = S.nvvm.make_tma_descriptor(
        token_tensor, S.make_layout((96, 1, 64), (64, 6144, 1)), SWIZZLE_128B)
    weight_tma = S.nvvm.make_tma_descriptor(
        weight_tensor, S.make_layout((64, 1, 64), (64, 4096, 1)), SWIZZLE_128B)
    output_tma = S.nvvm.make_tma_descriptor(
        C, S.make_layout((96, 64), (64, 1)), SWIZZLE_128B)
    tail_tma = S.nvvm.make_tma_descriptor(
        C, S.make_layout((32, 64), (64, 1)), SWIZZLE_128B)
    for stage in S.range(4, unroll=True):
        S.nvvm.mbarrier_init(full, stage, count=1, predicate=tid == 0)
        S.nvvm.mbarrier_init(empty, stage, count=1, predicate=tid == 0)
    S.nvvm.fence_mbarrier_init_release_cluster()
    S.syncthreads()

    if tid < 128:
        accumulators = S.make_local((1, 48), S.f32)
        for element in S.range(48, unroll=True):
            accumulators[0, element] = S.convert(0, S.f32)
        for kt in S.range(32):
            stage = kt % 4
            phase = (kt // 4) % 2
            S.nvvm.mbarrier_try_wait_parity(full, phase, 10000000, stage)
            weights = S.view(
                S.subview(weight_shared, (stage, wg * 4096), (1, 4096), (1, 1)),
                S.bf16, S.make_layout((64, 64), (64, 1)))
            tokens = S.view(
                S.subview(token_shared, (stage, 0), (1, 6144), (1, 1)),
                S.bf16, S.make_layout((96, 64), (64, 1)))
            da = S.nvvm.make_wgmma_descriptor_bits(weights, SWIZZLE_128B, 0, 0, 0, 16, 1024)
            db = S.nvvm.make_wgmma_descriptor_bits(tokens, SWIZZLE_128B, 0, 0, 0, 16, 1024)
            acc = accumulators[0]
            S.nvvm.wgmma_fence_aligned()
            acc = S.nvvm.wgmma_m64n96k16_f32_bf16_bf16(da, db, acc, 1)
            acc = S.nvvm.wgmma_m64n96k16_f32_bf16_bf16(da + 2, db + 2, acc, 1)
            acc = S.nvvm.wgmma_m64n96k16_f32_bf16_bf16(da + 4, db + 4, acc, 1)
            acc = S.nvvm.wgmma_m64n96k16_f32_bf16_bf16(da + 6, db + 6, acc, 1)
            S.nvvm.wgmma_group_sync_aligned()
            S.nvvm.wgmma_wait_group_sync(WAIT_GROUP)
            for element in S.range(48, unroll=True):
                accumulators[0, element] = acc[element]
            if WAIT_GROUP == 0:
                if lane128 == 0:
                    S.nvvm.mbarrier_arrive(empty, stage)
            else:
                if kt > 0 and lane128 == 0:
                    S.nvvm.mbarrier_arrive(empty, (kt - 1) % 4)
        S.nvvm.wgmma_wait_group_sync(0)
        S.nvvm.wgmma_fence_aligned()
        if WAIT_GROUP == 1 and lane128 == 0:
            S.nvvm.mbarrier_arrive(empty, 3)

        # All consumer WGs finish every stage before any of them aliases the
        # shared weight arena. The producer has issued exactly 32 input tiles.
        S.nvvm.named_barrier_sync(1, 128)
        for token_box in S.range(3, unroll=True):
            for local_group in S.range(2, unroll=True):
                packed = S.full((1, 4), 0, S.i32)
                source_group = 2 * token_box + local_group
                for matrix in S.range(4, unroll=True):
                    i = matrix & 1
                    source_w = 2 * source_group + matrix // 2
                    packed[0, matrix] = S.nvvm.floatx2_to_bf16x2(
                        accumulators[0, 4 * source_w + 2 * i],
                        accumulators[0, 4 * source_w + 2 * i + 1])
                address_matrix = lane // 8
                address_i = address_matrix & 1
                local_address_w = 2 * local_group + address_matrix // 2
                local_token = token_box * 32 + local_address_w * 8 + lane % 8
                local_channel = warp * 16 + address_i * 8
                logical_byte_offset = (local_token * 64 + local_channel) * 2
                swizzled_byte_offset = logical_byte_offset ^ ((logical_byte_offset & 0x380) >> 3)
                destination = S.subview(
                    output_shared, (0, wg * 6144 + swizzled_byte_offset // 2),
                    (8, 8), (1, 1))
                S.nvvm.stmatrix_m8n8_x4_b16_trans(destination, packed[0])
        S.nvvm.fence_proxy_async_shared_cta()
        S.nvvm.named_barrier_sync(2 + wg, 128)
        if lane128 == 0:
            if token0 <= 320:
                out_raw = S.subview(output_shared, (0, wg * 6144), (1, 6144), (1, 1))
                out_tile = S.view(out_raw, S.bf16, S.make_layout((96, 64), (64, 1)))
                S.nvvm.tma_store(out_tile, output_tma, (channel0 + wg * 64, token0), predicate=lane128 == 0)
            else:
                tail_raw = S.subview(output_shared, (0, wg * 6144), (1, 2048), (1, 1))
                tail_tile = S.view(tail_raw, S.bf16, S.make_layout((32, 64), (64, 1)))
                S.nvvm.tma_store(tail_tile, tail_tma, (channel0 + wg * 64, token0), predicate=lane128 == 0)
            S.nvvm.cp_async_bulk_commit_group()
            S.nvvm.cp_async_bulk_wait_group(0)
        S.nvvm.named_barrier_sync(2 + wg, 128)
    else:
        elected = S.nvvm.elect_sync()
        for kt in S.range(32):
            stage = kt % 4
            phase = 1 ^ ((kt // 4) % 2)
            S.nvvm.mbarrier_try_wait_parity(empty, phase, 10000000, stage)
            S.nvvm.mbarrier_arrive_expect_tx(full, 20480, stage, elected)
            weights_raw = S.subview(weight_shared, (stage, 0), (1, 4096), (1, 1))
            weights_dst = S.view(weights_raw, S.bf16,
                                 S.make_layout((64, 1, 64), (64, 4096, 1)))
            tokens_raw = S.subview(token_shared, (stage, 0), (1, 6144), (1, 1))
            tokens_dst = S.view(tokens_raw, S.bf16,
                                S.make_layout((96, 1, 64), (64, 6144, 1)))
            S.nvvm.tma_load(weights_dst, weight_tma, (0, kt, channel0), full,
                            mbar_id=stage, predicate=elected, expect_tx=False)
            S.nvvm.tma_load(tokens_dst, token_tma, (0, kt, token0), full,
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



@avelang.jit
def k_norm_rope_packed(
    Packed: S.Tensor((416, 5120), S.bf16),
    Weight: S.Tensor((256,), S.bf16),
    Tables: S.Tensor((26, 32, 2), S.bf16),
    Y: S.Tensor((416, 2, 256), S.bf16),
):
    m = S.block_id(0)
    h = S.block_id(1)
    lane = S.thread_id(0)
    norm = S.make_shared((256,), S.bf16, 16)
    ss = S.convert(0, S.f32)
    for j in S.range(8, unroll=True):
        d = lane + j * 32
        f = S.convert(Packed[m, 4096 + h * 256 + d], S.f32)
        ss = ss + f * f
    ss = ss + S.shuffle_xor(ss, 16, 32)
    ss = ss + S.shuffle_xor(ss, 8, 32)
    ss = ss + S.shuffle_xor(ss, 4, 32)
    ss = ss + S.shuffle_xor(ss, 2, 32)
    ss = ss + S.shuffle_xor(ss, 1, 32)
    inv = S.nvvm.fast_rsqrt(ss * S.convert(0.00390625, S.f32) + S.convert(0.000001, S.f32))
    for j in S.range(8, unroll=True):
        d = lane + j * 32
        v = S.convert(Packed[m, 4096 + h * 256 + d], S.f32)
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
def fused_qnorm_attention_shared_q4(
    Packed: S.Tensor((416, 5120), S.bf16),
    QWeight: S.Tensor((256,), S.bf16),
    Tables: S.Tensor((26, 32, 2), S.bf16),
    K: S.Tensor((16, 26, 2, 256), S.bf16),
    Out: S.Tensor((416, 8, 256), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % 32
    warp = tid // 32
    first_query = S.block_id(0) * 4
    head = S.block_id(1)
    batch = S.block_id(2)
    kv_head = head // 4
    key_count = first_query + 4
    if key_count > 26:
        key_count = 26
    keys = S.make_shared((26, 256), S.bf16, 16)
    values = S.make_shared((26, 256), S.bf16, 16)
    # Two 128-thread transfers cover one contiguous 256-channel head. Load
    # only the causal prefix needed by this query tile; all four warps join.
    for load in S.range(key_count * 2):
        linear = tid + load * 128
        key = linear // 256
        channel = linear % 256
        keys[key, channel] = K[batch, key, kv_head, channel]
        values[key, channel] = Packed[batch * 26 + key, 4608 + kv_head * 256 + channel]
    S.syncthreads()

    query = first_query + warp
    if query < 26:
        token = batch * 26 + query
        q0 = S.convert(Packed[token, head * 512 + lane + 0], S.f32)
        q1 = S.convert(Packed[token, head * 512 + lane + 32], S.f32)
        q2 = S.convert(Packed[token, head * 512 + lane + 64], S.f32)
        q3 = S.convert(Packed[token, head * 512 + lane + 96], S.f32)
        q4 = S.convert(Packed[token, head * 512 + lane + 128], S.f32)
        q5 = S.convert(Packed[token, head * 512 + lane + 160], S.f32)
        q6 = S.convert(Packed[token, head * 512 + lane + 192], S.f32)
        q7 = S.convert(Packed[token, head * 512 + lane + 224], S.f32)
        ss = S.convert(0, S.f32)
        ss = ss + q0 * q0
        ss = ss + q1 * q1
        ss = ss + q2 * q2
        ss = ss + q3 * q3
        ss = ss + q4 * q4
        ss = ss + q5 * q5
        ss = ss + q6 * q6
        ss = ss + q7 * q7
        ss = ss + S.shuffle_xor(ss, 16, 32)
        ss = ss + S.shuffle_xor(ss, 8, 32)
        ss = ss + S.shuffle_xor(ss, 4, 32)
        ss = ss + S.shuffle_xor(ss, 2, 32)
        ss = ss + S.shuffle_xor(ss, 1, 32)
        inv = S.nvvm.fast_rsqrt(ss * S.convert(0.00390625, S.f32) + S.convert(0.000001, S.f32))
        gain0 = S.convert(1, S.f32) + S.convert(QWeight[lane + 0], S.f32)
        q0 = S.convert(S.convert(q0 * inv * gain0, S.bf16), S.f32)
        gain1 = S.convert(1, S.f32) + S.convert(QWeight[lane + 32], S.f32)
        q1 = S.convert(S.convert(q1 * inv * gain1, S.bf16), S.f32)
        gain2 = S.convert(1, S.f32) + S.convert(QWeight[lane + 64], S.f32)
        q2 = S.convert(S.convert(q2 * inv * gain2, S.bf16), S.f32)
        gain3 = S.convert(1, S.f32) + S.convert(QWeight[lane + 96], S.f32)
        q3 = S.convert(S.convert(q3 * inv * gain3, S.bf16), S.f32)
        gain4 = S.convert(1, S.f32) + S.convert(QWeight[lane + 128], S.f32)
        q4 = S.convert(S.convert(q4 * inv * gain4, S.bf16), S.f32)
        gain5 = S.convert(1, S.f32) + S.convert(QWeight[lane + 160], S.f32)
        q5 = S.convert(S.convert(q5 * inv * gain5, S.bf16), S.f32)
        gain6 = S.convert(1, S.f32) + S.convert(QWeight[lane + 192], S.f32)
        q6 = S.convert(S.convert(q6 * inv * gain6, S.bf16), S.f32)
        gain7 = S.convert(1, S.f32) + S.convert(QWeight[lane + 224], S.f32)
        q7 = S.convert(S.convert(q7 * inv * gain7, S.bf16), S.f32)
        # Each lane owns its own split-half RoPE pair. BF16 normalization
        # happens before rotation; rotation is independently rounded to BF16.
        co = S.convert(Tables[query, lane, 0], S.f32)
        si = S.convert(Tables[query, lane, 1], S.f32)
        rope0 = q0 * co - q1 * si
        rope1 = q1 * co + q0 * si
        q0 = S.convert(S.convert(rope0, S.bf16), S.f32)
        q1 = S.convert(S.convert(rope1, S.bf16), S.f32)
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
        for key in S.range(query + 1):
            score = S.convert(0, S.f32)
            score = S.nvvm.fast_fma(q0, S.convert(keys[key, lane + 0], S.f32), score)
            score = S.nvvm.fast_fma(q1, S.convert(keys[key, lane + 32], S.f32), score)
            score = S.nvvm.fast_fma(q2, S.convert(keys[key, lane + 64], S.f32), score)
            score = S.nvvm.fast_fma(q3, S.convert(keys[key, lane + 96], S.f32), score)
            score = S.nvvm.fast_fma(q4, S.convert(keys[key, lane + 128], S.f32), score)
            score = S.nvvm.fast_fma(q5, S.convert(keys[key, lane + 160], S.f32), score)
            score = S.nvvm.fast_fma(q6, S.convert(keys[key, lane + 192], S.f32), score)
            score = S.nvvm.fast_fma(q7, S.convert(keys[key, lane + 224], S.f32), score)
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
            acc0 = acc0 * alpha + beta * S.convert(values[key, lane + 0], S.f32)
            acc1 = acc1 * alpha + beta * S.convert(values[key, lane + 32], S.f32)
            acc2 = acc2 * alpha + beta * S.convert(values[key, lane + 64], S.f32)
            acc3 = acc3 * alpha + beta * S.convert(values[key, lane + 96], S.f32)
            acc4 = acc4 * alpha + beta * S.convert(values[key, lane + 128], S.f32)
            acc5 = acc5 * alpha + beta * S.convert(values[key, lane + 160], S.f32)
            acc6 = acc6 * alpha + beta * S.convert(values[key, lane + 192], S.f32)
            acc7 = acc7 * alpha + beta * S.convert(values[key, lane + 224], S.f32)
            maximum = newmax
        inv_denom = S.nvvm.fast_rcp(denom)
        Out[token, head, lane + 0] = gate_store(acc0 * inv_denom, Packed[token, head * 512 + 256 + lane + 0])
        Out[token, head, lane + 32] = gate_store(acc1 * inv_denom, Packed[token, head * 512 + 256 + lane + 32])
        Out[token, head, lane + 64] = gate_store(acc2 * inv_denom, Packed[token, head * 512 + 256 + lane + 64])
        Out[token, head, lane + 96] = gate_store(acc3 * inv_denom, Packed[token, head * 512 + 256 + lane + 96])
        Out[token, head, lane + 128] = gate_store(acc4 * inv_denom, Packed[token, head * 512 + 256 + lane + 128])
        Out[token, head, lane + 160] = gate_store(acc5 * inv_denom, Packed[token, head * 512 + 256 + lane + 160])
        Out[token, head, lane + 192] = gate_store(acc6 * inv_denom, Packed[token, head * 512 + 256 + lane + 192])
        Out[token, head, lane + 224] = gate_store(acc7 * inv_denom, Packed[token, head * 512 + 256 + lane + 224])



@avelang.jit
def fused_qnorm_attention_mma_q8_async(
    Packed: S.Tensor((416, 5120), S.bf16),
    QWeight: S.Tensor((256,), S.bf16),
    Tables: S.Tensor((26, 32, 2), S.bf16),
    K: S.Tensor((16, 26, 2, 256), S.bf16),
    Out: S.Tensor((416, 8, 256), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % 32
    warp = tid // 32
    first_query = S.block_id(0) * 8
    head = S.block_id(1)
    batch = S.block_id(2)
    kv_head = head // 4
    key_count = first_query + 8
    if key_count > 26:
        key_count = 26
    queries = S.make_shared((16, 256), S.bf16, 128)
    keys = S.make_shared((32, 256), S.bf16, 128)
    values = S.make_shared((32, 256), S.bf16, 16)
    # Padding the score row helps the warp-fragment shared-store bank pattern.
    scores = S.make_shared((16, 40), S.f32, 16)

    # Copy aligned groups of eight BF16 values without routing K/V through
    # registers. The current native cp.async intrinsic has no src-size operand,
    # so invalid whole rows use explicit zero stores and never form global loads.
    for load in S.range(8):
        linear = (tid + load * 128) * 8
        key = linear // 256
        channel = linear % 256
        if key < key_count:
            # XOR only permutes complete 16-byte segments within each row.
            key_dst = (key * 256 + (channel ^ ((key % 8) * 8))) * 2
            key_src = (((batch * 26 + key) * 2 + kv_head) * 256 + channel) * 2
            value_dst = (key * 256 + channel) * 2
            value_src = ((batch * 26 + key) * 5120 + 4608 + kv_head * 256 + channel) * 2
            S.nvvm.cp_async_ca_shared_global(keys, K, key_dst, key_src, 16)
            S.nvvm.cp_async_ca_shared_global(values, Packed, value_dst, value_src, 16)
        else:
            for element in S.range(8, unroll=True):
                keys[key, (channel + element) ^ ((key % 8) * 8)] = S.convert(0, S.bf16)
                values[key, channel + element] = S.convert(0, S.bf16)
    S.nvvm.cp_async_commit_group()

    # Normalize Q while the committed K/V copies are in flight. Each warp
    # normalizes two real queries and initializes two padded rows.
    # No thread exits or skips a CTA barrier while staging the 16-row MMA tile.
    for slot in S.range(4, unroll=True):
        local_query = warp + slot * 4
        query = first_query + local_query
        q0 = S.convert(0, S.f32)
        q1 = S.convert(0, S.f32)
        q2 = S.convert(0, S.f32)
        q3 = S.convert(0, S.f32)
        q4 = S.convert(0, S.f32)
        q5 = S.convert(0, S.f32)
        q6 = S.convert(0, S.f32)
        q7 = S.convert(0, S.f32)
        if local_query < 8 and query < 26:
            token = batch * 26 + query
            q0 = S.convert(Packed[token, head * 512 + lane + 0], S.f32)
            q1 = S.convert(Packed[token, head * 512 + lane + 32], S.f32)
            q2 = S.convert(Packed[token, head * 512 + lane + 64], S.f32)
            q3 = S.convert(Packed[token, head * 512 + lane + 96], S.f32)
            q4 = S.convert(Packed[token, head * 512 + lane + 128], S.f32)
            q5 = S.convert(Packed[token, head * 512 + lane + 160], S.f32)
            q6 = S.convert(Packed[token, head * 512 + lane + 192], S.f32)
            q7 = S.convert(Packed[token, head * 512 + lane + 224], S.f32)
            ss = S.convert(0, S.f32)
            ss = ss + q0 * q0
            ss = ss + q1 * q1
            ss = ss + q2 * q2
            ss = ss + q3 * q3
            ss = ss + q4 * q4
            ss = ss + q5 * q5
            ss = ss + q6 * q6
            ss = ss + q7 * q7
            ss = ss + S.shuffle_xor(ss, 16, 32)
            ss = ss + S.shuffle_xor(ss, 8, 32)
            ss = ss + S.shuffle_xor(ss, 4, 32)
            ss = ss + S.shuffle_xor(ss, 2, 32)
            ss = ss + S.shuffle_xor(ss, 1, 32)
            inv = S.nvvm.fast_rsqrt(ss * S.convert(0.00390625, S.f32) + S.convert(0.000001, S.f32))
            gain0 = S.convert(1, S.f32) + S.convert(QWeight[lane + 0], S.f32)
            q0 = S.convert(S.convert(q0 * inv * gain0, S.bf16), S.f32)
            gain1 = S.convert(1, S.f32) + S.convert(QWeight[lane + 32], S.f32)
            q1 = S.convert(S.convert(q1 * inv * gain1, S.bf16), S.f32)
            gain2 = S.convert(1, S.f32) + S.convert(QWeight[lane + 64], S.f32)
            q2 = S.convert(S.convert(q2 * inv * gain2, S.bf16), S.f32)
            gain3 = S.convert(1, S.f32) + S.convert(QWeight[lane + 96], S.f32)
            q3 = S.convert(S.convert(q3 * inv * gain3, S.bf16), S.f32)
            gain4 = S.convert(1, S.f32) + S.convert(QWeight[lane + 128], S.f32)
            q4 = S.convert(S.convert(q4 * inv * gain4, S.bf16), S.f32)
            gain5 = S.convert(1, S.f32) + S.convert(QWeight[lane + 160], S.f32)
            q5 = S.convert(S.convert(q5 * inv * gain5, S.bf16), S.f32)
            gain6 = S.convert(1, S.f32) + S.convert(QWeight[lane + 192], S.f32)
            q6 = S.convert(S.convert(q6 * inv * gain6, S.bf16), S.f32)
            gain7 = S.convert(1, S.f32) + S.convert(QWeight[lane + 224], S.f32)
            q7 = S.convert(S.convert(q7 * inv * gain7, S.bf16), S.f32)
            co = S.convert(Tables[query, lane, 0], S.f32)
            si = S.convert(Tables[query, lane, 1], S.f32)
            rope0 = q0 * co - q1 * si
            rope1 = q1 * co + q0 * si
            q0 = S.convert(S.convert(rope0, S.bf16), S.f32)
            q1 = S.convert(S.convert(rope1, S.bf16), S.f32)
        queries[local_query, (lane + 0) ^ ((local_query % 8) * 8)] = S.convert(q0, S.bf16)
        queries[local_query, (lane + 32) ^ ((local_query % 8) * 8)] = S.convert(q1, S.bf16)
        queries[local_query, (lane + 64) ^ ((local_query % 8) * 8)] = S.convert(q2, S.bf16)
        queries[local_query, (lane + 96) ^ ((local_query % 8) * 8)] = S.convert(q3, S.bf16)
        queries[local_query, (lane + 128) ^ ((local_query % 8) * 8)] = S.convert(q4, S.bf16)
        queries[local_query, (lane + 160) ^ ((local_query % 8) * 8)] = S.convert(q5, S.bf16)
        queries[local_query, (lane + 192) ^ ((local_query % 8) * 8)] = S.convert(q6, S.bf16)
        queries[local_query, (lane + 224) ^ ((local_query % 8) * 8)] = S.convert(q7, S.bf16)
    S.nvvm.cp_async_wait_group(0)
    S.syncthreads()

    # Four warps compute the four eight-key blocks of a 16x32 FP32 score tile.
    # Q and K are the already-rounded BF16 operands. No probability appears in
    # MMA: the later softmax and all PV accumulation remain FP32 SIMT.
    matrix_acc = S.make_local((1, 4), S.f32)
    for element in S.range(4, unroll=True):
        matrix_acc[0, element] = S.convert(0, S.f32)
    for kt in S.range(16):
        qrow = lane % 16
        qcol = (kt * 16 + (lane // 16) * 8) ^ ((qrow % 8) * 8)
        krow = warp * 8 + lane % 8
        kcol = (kt * 16 + ((lane // 8) % 2) * 8) ^ ((krow % 8) * 8)
        q_tile = S.subview(queries, (qrow, qcol), (8, 8), (1, 1))
        k_tile = S.subview(keys, (krow, kcol), (8, 8), (1, 1))
        q_frag = S.nvvm.ldmatrix_m8n8_x4_b16(q_tile)
        k_frag = S.nvvm.ldmatrix_m8n8_x2_b16(k_tile)
        frag = matrix_acc[0]
        frag = S.nvvm.mma_16x8x16_bf16_f32(q_frag, k_frag, frag)
        for element in S.range(4, unroll=True):
            matrix_acc[0, element] = frag[element]
    row = lane // 4
    col = warp * 8 + (lane % 4) * 2
    scores[row, col] = matrix_acc[0, 0]
    scores[row, col + 1] = matrix_acc[0, 1]
    scores[row + 8, col] = matrix_acc[0, 2]
    scores[row + 8, col + 1] = matrix_acc[0, 3]
    S.syncthreads()

    # Both CTA barriers precede query validity branches. Each warp now owns
    # two queries; lanes own the32 scores/probabilities for a stable softmax.
    for slot in S.range(2, unroll=True):
        local_query = warp + slot * 4
        query = first_query + local_query
        if query < 26:
            token = batch * 26 + query
            score = scores[local_query, lane] * S.convert(0.0625, S.f32)
            if lane > query or lane >= 26:
                score = S.convert(-1.0e30, S.f32)
            maximum = score
            maximum = S.nvvm.fast_fmax(maximum, S.shuffle_xor(maximum, 16, 32))
            maximum = S.nvvm.fast_fmax(maximum, S.shuffle_xor(maximum, 8, 32))
            maximum = S.nvvm.fast_fmax(maximum, S.shuffle_xor(maximum, 4, 32))
            maximum = S.nvvm.fast_fmax(maximum, S.shuffle_xor(maximum, 2, 32))
            maximum = S.nvvm.fast_fmax(maximum, S.shuffle_xor(maximum, 1, 32))
            probability = S.nvvm.fast_exp2((score - maximum) * S.convert(1.4426950408889634, S.f32))
            if lane > query or lane >= 26:
                probability = S.convert(0, S.f32)
            denom = probability
            denom = denom + S.shuffle_xor(denom, 16, 32)
            denom = denom + S.shuffle_xor(denom, 8, 32)
            denom = denom + S.shuffle_xor(denom, 4, 32)
            denom = denom + S.shuffle_xor(denom, 2, 32)
            denom = denom + S.shuffle_xor(denom, 1, 32)
            probability = probability * S.nvvm.fast_rcp(denom)
            acc0 = S.convert(0, S.f32)
            acc1 = S.convert(0, S.f32)
            acc2 = S.convert(0, S.f32)
            acc3 = S.convert(0, S.f32)
            acc4 = S.convert(0, S.f32)
            acc5 = S.convert(0, S.f32)
            acc6 = S.convert(0, S.f32)
            acc7 = S.convert(0, S.f32)
            for key in S.range(query + 1):
                p = S.shuffle(probability, key, 32)
                acc0 = S.nvvm.fma(p, S.convert(values[key, lane + 0], S.f32), acc0)
                acc1 = S.nvvm.fma(p, S.convert(values[key, lane + 32], S.f32), acc1)
                acc2 = S.nvvm.fma(p, S.convert(values[key, lane + 64], S.f32), acc2)
                acc3 = S.nvvm.fma(p, S.convert(values[key, lane + 96], S.f32), acc3)
                acc4 = S.nvvm.fma(p, S.convert(values[key, lane + 128], S.f32), acc4)
                acc5 = S.nvvm.fma(p, S.convert(values[key, lane + 160], S.f32), acc5)
                acc6 = S.nvvm.fma(p, S.convert(values[key, lane + 192], S.f32), acc6)
                acc7 = S.nvvm.fma(p, S.convert(values[key, lane + 224], S.f32), acc7)
            Out[token, head, lane + 0] = gate_store(acc0, Packed[token, head * 512 + 256 + lane + 0])
            Out[token, head, lane + 32] = gate_store(acc1, Packed[token, head * 512 + 256 + lane + 32])
            Out[token, head, lane + 64] = gate_store(acc2, Packed[token, head * 512 + 256 + lane + 64])
            Out[token, head, lane + 96] = gate_store(acc3, Packed[token, head * 512 + 256 + lane + 96])
            Out[token, head, lane + 128] = gate_store(acc4, Packed[token, head * 512 + 256 + lane + 128])
            Out[token, head, lane + 160] = gate_store(acc5, Packed[token, head * 512 + 256 + lane + 160])
            Out[token, head, lane + 192] = gate_store(acc6, Packed[token, head * 512 + 256 + lane + 192])
            Out[token, head, lane + 224] = gate_store(acc7, Packed[token, head * 512 + 256 + lane + 224])


def bind_middle(packed, qnorm_weight, knorm_weight, tables, attended, *, config=None):
    """Allocate K scratch and return two ordered zero-argument launches."""
    import torch
    name = DEFAULT_ATTENTION_CONFIG if config is None else config
    if name not in ATTENTION_CONFIGS:
        raise ValueError('unknown attention configuration')
    spec = ATTENTION_CONFIGS[name]
    normalized_k = torch.empty((16, 26, 2, 256), dtype=torch.bfloat16, device=packed.device)
    flat_packed = packed.view(416, 5120)
    out = attended.view(416, 8, 256)
    kernels = OrderedDict()
    kernels['k_norm_rope_packed'] = lambda: k_norm_rope_packed[lambda: ((416, 2, 1), (32, 1, 1))](
        flat_packed, knorm_weight, tables, normalized_k.view(416, 2, 256), num_warps=1)
    kernel = spec['kernel']
    kernels['q_norm_rope_attn_gate'] = lambda: kernel[lambda: (spec['grid'], (spec['threads'], 1, 1))](
        flat_packed, qnorm_weight, tables, normalized_k, out, num_warps=spec['num_warps'])
    return kernels


def build(x, hidden, w, tables):
    """Five AveLang launches; host allocation/view only."""
    import torch
    assert tuple(x.shape) == (16, 26, 2048)
    assert tuple(hidden.shape) == tuple(x.shape)
    assert tuple(tables.shape) == (26, 32, 2)
    assert all(t.is_contiguous() and t.dtype == torch.bfloat16 and t.device == x.device
               for t in [x, hidden, tables, *w.values()])
    packed = torch.empty((416, 5120), device=x.device, dtype=torch.bfloat16)
    attended = torch.empty_like(x)
    projected = torch.empty_like(x)
    output = torch.empty_like(x)
    kernels = OrderedDict()
    kernels['qkv_projection'] = bind_projection(x.view(416, 2048), w['qkv'], packed, 5120)
    kernels.update(bind_middle(packed, w['q_norm'], w['k_norm'], tables, attended))
    kernels['output_projection'] = bind_projection(
        attended.view(416, 2048), w['o'], projected.view(416, 2048), 2048)
    kernels['residual'] = lambda: add_residual[lambda: ((416, 8, 1), (256, 1, 1))](
        hidden.view(416, 2048), projected.view(416, 2048), output.view(416, 2048), num_warps=8)
    return kernels, (attended, output)


# Initial four candidates, one final OProj test, and the accepted OProj baseline.
PROJECTION_CONFIGS = {
    'qkv_t160_c128_w1': {'kernel': linear_swap160x128, 'N': 5120, 'tokens': 160, 'channels': 128,
                        'consumer_warpgroups': 2, 'stages': 4, 'wait_group': 1,
                        'grid': (40, 3, 1), 'threads': 288, 'num_warps': 9},
    'qkv_t160_c128_w0': {'kernel': linear_swap160x128, 'N': 5120, 'tokens': 160, 'channels': 128,
                        'consumer_warpgroups': 2, 'stages': 4, 'wait_group': 0,
                        'grid': (40, 3, 1), 'threads': 288, 'num_warps': 9},
    'o_t128_c64_w1': {'kernel': linear_swap128x64, 'N': 2048, 'tokens': 128, 'channels': 64,
                    'consumer_warpgroups': 1, 'stages': 4, 'wait_group': 1,
                    'grid': (32, 4, 1), 'threads': 160, 'num_warps': 5},
    'o_t96_c64_w1': {'kernel': linear_swap96x64, 'N': 2048, 'tokens': 96, 'channels': 64,
                   'consumer_warpgroups': 1, 'stages': 4, 'wait_group': 1,
                   'grid': (32, 5, 1), 'threads': 160, 'num_warps': 5},
    'o_t64_c128_w1': {'kernel': linear_swap64x128, 'N': 2048, 'tokens': 64, 'channels': 128,
                    'consumer_warpgroups': 2, 'stages': 4, 'wait_group': 1,
                    'grid': (16, 7, 1), 'threads': 288, 'num_warps': 9},
    'o_round2': {'kernel': linear_round2, 'N': 2048, 'tokens': 64, 'channels': 128,
                'consumer_warpgroups': 1, 'stages': 3, 'wait_group': 0,
                'grid': (16, 7, 1), 'threads': 160, 'num_warps': 5,
                'argument_style': 'round2', 'wgmma_n': 128, 'transpose': False},
}
DEFAULT_PROJECTION_CONFIGS = {5120: 'qkv_t160_c128_w1', 2048: 'o_t64_c128_w1'}
# Compatibility for CPU frontend diagnosis; numerical harnesses use bind_projection.
linear_native = linear_swap160x128


def bind_projection(A, W, C, N, *, config=None):
    """Bind one native launch; config is one key of PROJECTION_CONFIGS."""
    name = DEFAULT_PROJECTION_CONFIGS[N] if config is None else config
    if not isinstance(name, str) or name not in PROJECTION_CONFIGS:
        raise ValueError('config must be a named PROJECTION_CONFIGS entry')
    spec = PROJECTION_CONFIGS[name]
    if N != spec['N']:
        raise ValueError('configuration output width mismatch')
    if tuple(A.shape) != (416, 2048) or tuple(W.shape) != (N, 2048) or tuple(C.shape) != (416, N):
        raise ValueError('projection tensors must use the fixed contiguous shapes')
    kernel = spec['kernel']
    dimensions = (spec['grid'], (spec['threads'], 1, 1))
    if spec.get('argument_style') == 'round2':
        return lambda: kernel[lambda: dimensions](A, W, C, N, num_warps=spec['num_warps'])
    return lambda: kernel[lambda: dimensions](A, W, C, N, spec['wait_group'], num_warps=spec['num_warps'])


OFFLINE_COMPILE_CASES = [
    (name, spec['kernel'],
     {'N': spec['N']} if spec.get('argument_style') == 'round2'
     else {'N': spec['N'], 'WAIT_GROUP': spec['wait_group']}, spec['num_warps'])
    for name, spec in PROJECTION_CONFIGS.items()
] + [
    ('k_norm_rope_packed', k_norm_rope_packed, {}, 1),
    ('attention_shared_q4', fused_qnorm_attention_shared_q4, {}, 4),
    ('attention_mma_q8_async', fused_qnorm_attention_mma_q8_async, {}, 4),
    ('residual', add_residual, {}, 8),
]


PROFILE_STAGE_TAGS = ['qkv', 'knorm', 'attention', 'oproj', 'residual']
ATTENTION_CONFIGS = {
    'shared_q4': {'kernel': fused_qnorm_attention_shared_q4, 'query_tile': 4,
                  'grid': (7, 8, 16), 'threads': 128, 'num_warps': 4,
                  'qk': 'FP32 SIMT', 'pv': 'FP32', 'shared_kv': True},
    'mma_q8_async': {'kernel': fused_qnorm_attention_mma_q8_async, 'query_tile': 8,
               'grid': (4, 8, 16), 'threads': 128, 'num_warps': 4,
               'qk': 'BF16 MMA with FP32 accumulator', 'pv': 'FP32', 'shared_kv': True,
               'kv_preload': '16B cp.async.ca'},
}
DEFAULT_ATTENTION_CONFIG = 'mma_q8_async'
