import torch

import avelang
import avelang.language as S


BATCH = 16
SEQUENCE = 2048
QUERY_HEADS = 8
HEAD_DIM = 128
KEY_BLOCK = 128
KEY_TILES = SEQUENCE // KEY_BLOCK
SUPPORTED_SEQUENCES = (1024, 2048, 4096, 8192, 16384)
QUERY_BLOCK = 128
PACKED_QUERY_TOKENS = QUERY_BLOCK // QUERY_HEADS
CONSUMER_ROWS = 64
WARPGROUP_THREADS = 128
CONSUMER_WARPGROUPS = 2
THREADS = 384
PERSISTENT_BLOCKS = 128
SOFTMAX_SCALE_LOG2 = 0.12751743082459868
NEG_INFINITY = -3.4028234663852886e38
SWIZZLE_128B = S.WGMMA_SWIZZLE_128B


@avelang.jit
def _pack_bf16(first: S.f32, second: S.f32) -> S.i32:
    return S.nvvm.floatx2_to_bf16x2(first, second)


@avelang.jit
def _group_max(value: S.f32) -> S.f32:
    value = S.nvvm.fast_fmax(value, S.shuffle_xor(value, 1, 32))
    value = S.nvvm.fast_fmax(value, S.shuffle_xor(value, 2, 32))
    return value


@avelang.jit
def _group_sum(value: S.f32) -> S.f32:
    value = value + S.shuffle_xor(value, 1, 32)
    value = value + S.shuffle_xor(value, 2, 32)
    return value


@avelang.jit
def _attention_mqa_1024_kernel(
    query: S.Tensor((16, 1024, 8, 128), S.bf16),
    key: S.Tensor((16, 1024, 1, 128), S.bf16),
    value: S.Tensor((16, 1024, 1, 128), S.bf16),
    output: S.Tensor((16, 1024, 8, 128), S.bf16),
):
    tid = S.thread_id(0)
    warpgroup = tid // WARPGROUP_THREADS
    warpgroup_thread = tid % WARPGROUP_THREADS

    query_barrier = S.nvvm.mbarrier_create(2)
    key_barrier = S.nvvm.mbarrier_create(2)
    value_barrier = S.nvvm.mbarrier_create(2)

    value_shared = S.make_shared((2, KEY_BLOCK * HEAD_DIM), S.bf16, 128)
    key_shared = S.make_shared((2, KEY_BLOCK * HEAD_DIM), S.bf16, 128)
    output_shared = S.make_shared(
        (CONSUMER_WARPGROUPS, CONSUMER_ROWS * HEAD_DIM), S.bf16, 128
    )
    query_shared = S.make_shared(
        (2, CONSUMER_WARPGROUPS, CONSUMER_ROWS * HEAD_DIM), S.bf16, 128
    )

    # Reinterpret the contiguous PyTorch tensors exactly like the CUDA tensor
    # maps. TMA coordinates are ordered from the fastest-changing dimension.
    query_map_tensor = S.view(
        query,
        S.bf16,
        S.make_layout(
            (2, BATCH * 1024 * QUERY_HEADS, 64),
            (64, HEAD_DIM, 1),
        ),
    )
    output_map_tensor = S.view(
        output,
        S.bf16,
        S.make_layout(
            (2, BATCH * 1024 * QUERY_HEADS, 64),
            (64, HEAD_DIM, 1),
        ),
    )
    key_map_tensor = S.view(
        key,
        S.bf16,
        S.make_layout(
            (2, BATCH * 1024, 64),
            (64, HEAD_DIM, 1),
        ),
    )
    value_map_tensor = S.view(
        value,
        S.bf16,
        S.make_layout(
            (BATCH * (1024 // 8), 2, 8, 64),
            (8 * HEAD_DIM, 64, HEAD_DIM, 1),
        ),
    )

    query_layout = S.make_layout((2, 64, 64), (4096, 64, 1))
    key_layout = S.make_layout((2, 128, 64), (8192, 64, 1))
    value_layout = S.make_layout((16, 2, 8, 64), (1024, 512, 64, 1))
    query_desc = S.nvvm.make_tma_descriptor(
        query_map_tensor, query_layout, SWIZZLE_128B
    )
    output_desc = S.nvvm.make_tma_descriptor(
        output_map_tensor, query_layout, SWIZZLE_128B
    )
    key_desc = S.nvvm.make_tma_descriptor(key_map_tensor, key_layout, SWIZZLE_128B)
    value_desc = S.nvvm.make_tma_descriptor(
        value_map_tensor, value_layout, SWIZZLE_128B
    )

    for stage in S.range(2, unroll=True):
        S.nvvm.mbarrier_init(query_barrier, stage, count=1, predicate=tid == 0)
        S.nvvm.mbarrier_init(key_barrier, stage, count=1, predicate=tid == 0)
        S.nvvm.mbarrier_init(value_barrier, stage, count=1, predicate=tid == 0)
    S.syncthreads()

    elected = S.nvvm.elect_sync()
    if warpgroup == 0:
        S.nvvm.setmaxnreg_dec(40)
        if warpgroup_thread < 32:
            work_iteration = 0
            for work in S.range(
                S.block_id(0),
                BATCH * (1024 // PACKED_QUERY_TOKENS),
                S.grid_dim(0),
            ):
                query_buffer = work_iteration & 1
                if work_iteration >= 2:
                    S.nvvm.named_barrier_sync(6 + query_buffer, 288)
                batch = work // (1024 // PACKED_QUERY_TOKENS)
                query_start = (
                    work - batch * (1024 // PACKED_QUERY_TOKENS)
                ) * PACKED_QUERY_TOKENS
                S.nvvm.mbarrier_arrive_expect_tx(
                    query_barrier,
                    QUERY_BLOCK * HEAD_DIM * 2,
                    query_buffer,
                    elected,
                )
                for consumer in S.range(CONSUMER_WARPGROUPS, unroll=True):
                    query_destination_raw = S.subview(
                        query_shared,
                        (query_buffer, consumer, 0),
                        (1, 1, CONSUMER_ROWS * HEAD_DIM),
                        (1, 1, 1),
                    )
                    query_destination = S.view(
                        query_destination_raw,
                        S.bf16,
                        S.make_layout((2, 64, 64), (4096, 64, 1)),
                    )
                    S.nvvm.tma_load(
                        query_destination,
                        query_desc,
                        (
                            0,
                            (batch * 1024 + query_start + consumer * 8)
                            * QUERY_HEADS,
                            0,
                        ),
                        query_barrier,
                        mbar_id=query_buffer,
                        predicate=elected,
                        expect_tx=False,
                    )

                # Prime the two-stage KV pipeline separately.  Keeping the
                # first transaction out of the steady-state loop mirrors the
                # CUDA specialization and gives ptxas the same TMA schedule.
                stage = work_iteration * 8
                buffer = stage & 1
                if stage >= 2:
                    S.nvvm.named_barrier_sync(4 + buffer, 288)
                S.nvvm.mbarrier_arrive_expect_tx(
                    key_barrier,
                    KEY_BLOCK * HEAD_DIM * 2,
                    buffer,
                    elected,
                )
                first_key_destination_raw = S.subview(
                    key_shared,
                    (buffer, 0),
                    (1, KEY_BLOCK * HEAD_DIM),
                    (1, 1),
                )
                first_key_destination = S.view(
                    first_key_destination_raw,
                    S.bf16,
                    S.make_layout((2, 128, 64), (8192, 64, 1)),
                )
                S.nvvm.tma_load(
                    first_key_destination,
                    key_desc,
                    (0, batch * 1024, 0),
                    key_barrier,
                    mbar_id=buffer,
                    predicate=elected,
                    expect_tx=False,
                )
                S.nvvm.mbarrier_arrive_expect_tx(
                    value_barrier,
                    KEY_BLOCK * HEAD_DIM * 2,
                    buffer,
                    elected,
                )
                first_value_destination_raw = S.subview(
                    value_shared,
                    (buffer, 0),
                    (1, KEY_BLOCK * HEAD_DIM),
                    (1, 1),
                )
                first_value_destination = S.view(
                    first_value_destination_raw,
                    S.bf16,
                    S.make_layout((16, 2, 8, 64), (1024, 512, 64, 1)),
                )
                S.nvvm.tma_load(
                    first_value_destination,
                    value_desc,
                    (0, 0, 0, batch * (1024 // 8)),
                    value_barrier,
                    mbar_id=buffer,
                    predicate=elected,
                    expect_tx=False,
                )

                for key_tile in S.range(1, 8):
                    stage = work_iteration * 8 + key_tile
                    buffer = stage & 1
                    if stage >= 2:
                        S.nvvm.named_barrier_sync(4 + buffer, 288)
                    S.nvvm.mbarrier_arrive_expect_tx(
                        key_barrier,
                        KEY_BLOCK * HEAD_DIM * 2,
                        buffer,
                        elected,
                    )
                    key_destination_raw = S.subview(
                        key_shared,
                        (buffer, 0),
                        (1, KEY_BLOCK * HEAD_DIM),
                        (1, 1),
                    )
                    key_destination = S.view(
                        key_destination_raw,
                        S.bf16,
                        S.make_layout((2, 128, 64), (8192, 64, 1)),
                    )
                    S.nvvm.tma_load(
                        key_destination,
                        key_desc,
                        (0, batch * 1024 + key_tile * KEY_BLOCK, 0),
                        key_barrier,
                        mbar_id=buffer,
                        predicate=elected,
                        expect_tx=False,
                    )
                    S.nvvm.mbarrier_arrive_expect_tx(
                        value_barrier,
                        KEY_BLOCK * HEAD_DIM * 2,
                        buffer,
                        elected,
                    )
                    value_destination_raw = S.subview(
                        value_shared,
                        (buffer, 0),
                        (1, KEY_BLOCK * HEAD_DIM),
                        (1, 1),
                    )
                    value_destination = S.view(
                        value_destination_raw,
                        S.bf16,
                        S.make_layout((16, 2, 8, 64), (1024, 512, 64, 1)),
                    )
                    S.nvvm.tma_load(
                        value_destination,
                        value_desc,
                        (
                            0,
                            0,
                            0,
                            key_tile * (KEY_BLOCK // 8) + batch * (1024 // 8),
                        ),
                        value_barrier,
                        mbar_id=buffer,
                        predicate=elected,
                        expect_tx=False,
                    )
                work_iteration = work_iteration + 1
        return

    S.nvvm.setmaxnreg_inc(232)
    consumer = warpgroup - 1
    lane = warpgroup_thread & 31
    warp = warpgroup_thread >> 5
    work_iteration = 0
    for work in S.range(
        S.block_id(0), BATCH * (1024 // PACKED_QUERY_TOKENS), S.grid_dim(0)
    ):
        batch = work // (1024 // PACKED_QUERY_TOKENS)
        query_start = (
            work - batch * (1024 // PACKED_QUERY_TOKENS)
        ) * PACKED_QUERY_TOKENS
        query_buffer = work_iteration & 1
        S.nvvm.mbarrier_try_wait_parity(
            query_barrier, (work_iteration >> 1) & 1, 10000000, query_buffer
        )
        output_accumulator = S.full((1, 64), 0.0, S.f32)
        max_first = S.convert(NEG_INFINITY, S.f32)
        max_second = S.convert(NEG_INFINITY, S.f32)
        sum_first = S.convert(0.0, S.f32)
        sum_second = S.convert(0.0, S.f32)
        softmax_scale_log2 = S.convert(SOFTMAX_SCALE_LOG2, S.f32)
        scores = S.full((1, 64), NEG_INFINITY, S.f32)
        probabilities = S.full((8, 8), 0.0, S.f32)
        query_matrix = S.subview(
            query_shared,
            (query_buffer, consumer, 0),
            (1, CONSUMER_ROWS, HEAD_DIM),
            (1, HEAD_DIM, 1),
        )

        for key_tile in S.range(8):
            stage = work_iteration * 8 + key_tile
            buffer = stage & 1
            S.nvvm.mbarrier_try_wait_parity(
                key_barrier, (stage >> 1) & 1, 10000000, buffer
            )
            key_matrix = S.subview(
                key_shared,
                (buffer, 0),
                (KEY_BLOCK, HEAD_DIM),
                (HEAD_DIM, 1),
            )
            query_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
                query_matrix, SWIZZLE_128B, 0, 0, 0, 16, 1024
            )
            key_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
                key_matrix, SWIZZLE_128B, 0, 0, 0, 16, 1024
            )
            S.nvvm.wgmma_fence_aligned()
            qk_result = S.nvvm.wgmma_m64n128k128_f32_bf16_bf16_ss(
                query_matrix_desc, key_matrix_desc
            )
            S.nvvm.wgmma_group_sync_aligned()
            S.nvvm.wgmma_wait_group_sync(0)
            for result_element in S.range(64, unroll=True):
                scores[0, result_element] = qk_result[result_element]
            if key_tile + 1 == 8:
                S.nvvm.named_barrier_arrive(6 + query_buffer, 288)

            tile_max_first = S.convert(NEG_INFINITY, S.f32)
            tile_max_second = S.convert(NEG_INFINITY, S.f32)
            for tile in S.range(8, unroll=True):
                tile_max_first = S.nvvm.fast_fmax(
                    tile_max_first,
                    S.nvvm.fast_fmax(
                        S.nvvm.fast_fmax(scores[0, tile * 8], scores[0, tile * 8 + 1]),
                        S.nvvm.fast_fmax(scores[0, tile * 8 + 4], scores[0, tile * 8 + 5]),
                    ),
                )
                tile_max_second = S.nvvm.fast_fmax(
                    tile_max_second,
                    S.nvvm.fast_fmax(
                        S.nvvm.fast_fmax(scores[0, tile * 8 + 2], scores[0, tile * 8 + 3]),
                        S.nvvm.fast_fmax(scores[0, tile * 8 + 6], scores[0, tile * 8 + 7]),
                    ),
                )
            tile_max_first = _group_max(tile_max_first * softmax_scale_log2)
            tile_max_second = _group_max(tile_max_second * softmax_scale_log2)
            new_max_first = S.nvvm.fast_fmax(max_first, tile_max_first)
            new_max_second = S.nvvm.fast_fmax(max_second, tile_max_second)
            old_scale_first = S.nvvm.fast_exp2(max_first - new_max_first)
            old_scale_second = S.nvvm.fast_exp2(max_second - new_max_second)
            tile_sum_first = S.convert(0.0, S.f32)
            tile_sum_second = S.convert(0.0, S.f32)

            for tile in S.range(8, unroll=True):
                probabilities[tile, 0] = S.nvvm.fast_exp2(
                    S.nvvm.fast_fma(scores[0, tile * 8], softmax_scale_log2, -new_max_first)
                )
                probabilities[tile, 1] = S.nvvm.fast_exp2(
                    S.nvvm.fast_fma(
                        scores[0, tile * 8 + 1],
                        softmax_scale_log2,
                        -new_max_first,
                    )
                )
                probabilities[tile, 4] = S.nvvm.fast_exp2(
                    S.nvvm.fast_fma(
                        scores[0, tile * 8 + 4],
                        softmax_scale_log2,
                        -new_max_first,
                    )
                )
                probabilities[tile, 5] = S.nvvm.fast_exp2(
                    S.nvvm.fast_fma(
                        scores[0, tile * 8 + 5],
                        softmax_scale_log2,
                        -new_max_first,
                    )
                )
                tile_sum_first = (
                    tile_sum_first
                    + probabilities[tile, 0]
                    + probabilities[tile, 1]
                    + probabilities[tile, 4]
                    + probabilities[tile, 5]
                )
            for tile in S.range(8, unroll=True):
                probabilities[tile, 2] = S.nvvm.fast_exp2(
                    S.nvvm.fast_fma(
                        scores[0, tile * 8 + 2],
                        softmax_scale_log2,
                        -new_max_second,
                    )
                )
                probabilities[tile, 3] = S.nvvm.fast_exp2(
                    S.nvvm.fast_fma(
                        scores[0, tile * 8 + 3],
                        softmax_scale_log2,
                        -new_max_second,
                    )
                )
                probabilities[tile, 6] = S.nvvm.fast_exp2(
                    S.nvvm.fast_fma(
                        scores[0, tile * 8 + 6],
                        softmax_scale_log2,
                        -new_max_second,
                    )
                )
                probabilities[tile, 7] = S.nvvm.fast_exp2(
                    S.nvvm.fast_fma(
                        scores[0, tile * 8 + 7],
                        softmax_scale_log2,
                        -new_max_second,
                    )
                )
                tile_sum_second = (
                    tile_sum_second
                    + probabilities[tile, 2]
                    + probabilities[tile, 3]
                    + probabilities[tile, 6]
                    + probabilities[tile, 7]
                )
            tile_sum_first = _group_sum(tile_sum_first)
            tile_sum_second = _group_sum(tile_sum_second)
            sum_first = S.nvvm.fast_fma(sum_first, old_scale_first, tile_sum_first)
            sum_second = S.nvvm.fast_fma(sum_second, old_scale_second, tile_sum_second)
            max_first = new_max_first
            max_second = new_max_second
            for tile in S.range(8, unroll=True):
                output_accumulator[0, tile * 8] = (
                    output_accumulator[0, tile * 8] * old_scale_first
                )
                output_accumulator[0, tile * 8 + 1] = (
                    output_accumulator[0, tile * 8 + 1] * old_scale_first
                )
                output_accumulator[0, tile * 8 + 4] = (
                    output_accumulator[0, tile * 8 + 4] * old_scale_first
                )
                output_accumulator[0, tile * 8 + 5] = (
                    output_accumulator[0, tile * 8 + 5] * old_scale_first
                )
                output_accumulator[0, tile * 8 + 2] = (
                    output_accumulator[0, tile * 8 + 2] * old_scale_second
                )
                output_accumulator[0, tile * 8 + 3] = (
                    output_accumulator[0, tile * 8 + 3] * old_scale_second
                )
                output_accumulator[0, tile * 8 + 6] = (
                    output_accumulator[0, tile * 8 + 6] * old_scale_second
                )
                output_accumulator[0, tile * 8 + 7] = (
                    output_accumulator[0, tile * 8 + 7] * old_scale_second
                )

            S.nvvm.mbarrier_try_wait_parity(
                value_barrier, (stage >> 1) & 1, 10000000, buffer
            )
            value_matrix = S.subview(
                value_shared,
                (buffer, 0),
                (8, 2 * 8 * 64),
                (2 * 8 * 64, 1),
            )
            value_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
                value_matrix, SWIZZLE_128B, 0, 0, 0, 1024, 2048
            )
            S.nvvm.wgmma_fence_aligned()
            packed = S.full((1, 4), 0, S.i32)
            packed[0, 0] = _pack_bf16(probabilities[0, 0], probabilities[0, 1])
            packed[0, 1] = _pack_bf16(probabilities[0, 2], probabilities[0, 3])
            packed[0, 2] = _pack_bf16(probabilities[0, 4], probabilities[0, 5])
            packed[0, 3] = _pack_bf16(probabilities[0, 6], probabilities[0, 7])
            packed0 = packed[0]
            packed[0, 0] = _pack_bf16(probabilities[1, 0], probabilities[1, 1])
            packed[0, 1] = _pack_bf16(probabilities[1, 2], probabilities[1, 3])
            packed[0, 2] = _pack_bf16(probabilities[1, 4], probabilities[1, 5])
            packed[0, 3] = _pack_bf16(probabilities[1, 6], probabilities[1, 7])
            packed1 = packed[0]
            packed[0, 0] = _pack_bf16(probabilities[2, 0], probabilities[2, 1])
            packed[0, 1] = _pack_bf16(probabilities[2, 2], probabilities[2, 3])
            packed[0, 2] = _pack_bf16(probabilities[2, 4], probabilities[2, 5])
            packed[0, 3] = _pack_bf16(probabilities[2, 6], probabilities[2, 7])
            packed2 = packed[0]
            packed[0, 0] = _pack_bf16(probabilities[3, 0], probabilities[3, 1])
            packed[0, 1] = _pack_bf16(probabilities[3, 2], probabilities[3, 3])
            packed[0, 2] = _pack_bf16(probabilities[3, 4], probabilities[3, 5])
            packed[0, 3] = _pack_bf16(probabilities[3, 6], probabilities[3, 7])
            packed3 = packed[0]
            packed[0, 0] = _pack_bf16(probabilities[4, 0], probabilities[4, 1])
            packed[0, 1] = _pack_bf16(probabilities[4, 2], probabilities[4, 3])
            packed[0, 2] = _pack_bf16(probabilities[4, 4], probabilities[4, 5])
            packed[0, 3] = _pack_bf16(probabilities[4, 6], probabilities[4, 7])
            packed4 = packed[0]
            packed[0, 0] = _pack_bf16(probabilities[5, 0], probabilities[5, 1])
            packed[0, 1] = _pack_bf16(probabilities[5, 2], probabilities[5, 3])
            packed[0, 2] = _pack_bf16(probabilities[5, 4], probabilities[5, 5])
            packed[0, 3] = _pack_bf16(probabilities[5, 6], probabilities[5, 7])
            packed5 = packed[0]
            packed[0, 0] = _pack_bf16(probabilities[6, 0], probabilities[6, 1])
            packed[0, 1] = _pack_bf16(probabilities[6, 2], probabilities[6, 3])
            packed[0, 2] = _pack_bf16(probabilities[6, 4], probabilities[6, 5])
            packed[0, 3] = _pack_bf16(probabilities[6, 6], probabilities[6, 7])
            packed6 = packed[0]
            packed[0, 0] = _pack_bf16(probabilities[7, 0], probabilities[7, 1])
            packed[0, 1] = _pack_bf16(probabilities[7, 2], probabilities[7, 3])
            packed[0, 2] = _pack_bf16(probabilities[7, 4], probabilities[7, 5])
            packed[0, 3] = _pack_bf16(probabilities[7, 6], probabilities[7, 7])
            packed7 = packed[0]
            pv_result = S.nvvm.wgmma_m64n128k128_f32_bf16_bf16_rs(
                packed0,
                packed1,
                packed2,
                packed3,
                packed4,
                packed5,
                packed6,
                packed7,
                value_matrix_desc,
                output_accumulator[0],
            )
            S.nvvm.wgmma_group_sync_aligned()
            S.nvvm.wgmma_wait_group_sync(0)
            for result_element in S.range(64, unroll=True):
                output_accumulator[0, result_element] = pv_result[result_element]

            S.nvvm.named_barrier_arrive(4 + buffer, 288)

        inverse_first = S.nvvm.fast_rcp(sum_first)
        inverse_second = S.nvvm.fast_rcp(sum_second)
        for tile in S.range(8, unroll=True):
            row = warp * 16 + (lane & 15)
            column = tile * 16 + (lane >> 4) * 8
            linear = row * 64 + (column & 63)
            if column >= 64:
                linear = linear + CONSUMER_ROWS * 64
            offset = linear ^ ((linear & 0x1C0) >> 3)
            destination = S.subview(output_shared, (consumer, offset), (8, 8), (1, 1))
            packed = S.full((1, 4), 0, S.i32)
            packed[0, 0] = _pack_bf16(
                output_accumulator[0, tile * 8] * inverse_first,
                output_accumulator[0, tile * 8 + 1] * inverse_first,
            )
            packed[0, 1] = _pack_bf16(
                output_accumulator[0, tile * 8 + 2] * inverse_second,
                output_accumulator[0, tile * 8 + 3] * inverse_second,
            )
            packed[0, 2] = _pack_bf16(
                output_accumulator[0, tile * 8 + 4] * inverse_first,
                output_accumulator[0, tile * 8 + 5] * inverse_first,
            )
            packed[0, 3] = _pack_bf16(
                output_accumulator[0, tile * 8 + 6] * inverse_second,
                output_accumulator[0, tile * 8 + 7] * inverse_second,
            )
            S.nvvm.stmatrix_m8n8_x4_b16(destination, packed[0])
        S.nvvm.fence_proxy_async_shared_cta()
        S.nvvm.named_barrier_sync(1 + consumer, WARPGROUP_THREADS)
        if warpgroup_thread == 0:
            if warpgroup == 1:
                output_source_raw = S.subview(
                    output_shared,
                    (0, 0),
                    (1, CONSUMER_ROWS * HEAD_DIM),
                    (1, 1),
                )
                output_source = S.view(
                    output_source_raw,
                    S.bf16,
                    S.make_layout((2, 64, 64), (4096, 64, 1)),
                )
                S.nvvm.tma_store(
                    output_source,
                    output_desc,
                    (0, (batch * 1024 + query_start) * QUERY_HEADS, 0),
                    predicate=warpgroup_thread == 0,
                )
                S.nvvm.cp_async_bulk_commit_group()
                S.nvvm.cp_async_bulk_wait_group(0, read=True)
            else:
                output_source_raw = S.subview(
                    output_shared,
                    (1, 0),
                    (1, CONSUMER_ROWS * HEAD_DIM),
                    (1, 1),
                )
                output_source = S.view(
                    output_source_raw,
                    S.bf16,
                    S.make_layout((2, 64, 64), (4096, 64, 1)),
                )
                S.nvvm.tma_store(
                    output_source,
                    output_desc,
                    (0, (batch * 1024 + query_start + 8) * QUERY_HEADS, 0),
                    predicate=warpgroup_thread == 0,
                )
                S.nvvm.cp_async_bulk_commit_group()
                S.nvvm.cp_async_bulk_wait_group(0, read=False)
        work_iteration = work_iteration + 1


@avelang.jit
def _attention_mqa_2048_kernel(
    query: S.Tensor((16, 2048, 8, 128), S.bf16),
    key: S.Tensor((16, 2048, 1, 128), S.bf16),
    value: S.Tensor((16, 2048, 1, 128), S.bf16),
    output: S.Tensor((16, 2048, 8, 128), S.bf16),
    query_length: S.i32,
):
    tid = S.thread_id(0)
    warpgroup = tid // WARPGROUP_THREADS
    warpgroup_thread = tid % WARPGROUP_THREADS
    query_blocks = query_length // PACKED_QUERY_TOKENS
    total_work = query_blocks * BATCH

    # Ave's module builder emits shared globals in reverse declaration order.
    # Keep the resulting arena identical to CUDA's SharedStorage: query,
    # output, key, value, then barriers. Besides making WGMMA descriptors
    # directly comparable, this keeps every TMA destination sector in range.
    query_barrier = S.nvvm.mbarrier_create()
    key_barrier = S.nvvm.mbarrier_create(3)
    value_barrier = S.nvvm.mbarrier_create(2)
    query_empty_barrier = S.nvvm.mbarrier_create()
    key_empty_barrier = S.nvvm.mbarrier_create(3)
    value_empty_barrier = S.nvvm.mbarrier_create(2)
    output_empty_barrier = S.nvvm.mbarrier_create(2)

    value_shared = S.make_shared((2, KEY_BLOCK * HEAD_DIM), S.bf16, 128)
    key_shared = S.make_shared((3, KEY_BLOCK * HEAD_DIM), S.bf16, 128)
    output_shared = S.make_shared(
        (CONSUMER_WARPGROUPS, CONSUMER_ROWS * HEAD_DIM), S.bf16, 128
    )
    query_shared = S.make_shared(
        (CONSUMER_WARPGROUPS, CONSUMER_ROWS * HEAD_DIM), S.bf16, 128
    )

    # Reinterpret the contiguous PyTorch tensors exactly like the CUDA tensor
    # maps. TMA coordinates are ordered from the fastest-changing dimension.
    query_map_tensor = S.view(
        query,
        S.bf16,
        S.make_layout(
            (2, BATCH * 2048 * QUERY_HEADS, 64),
            (64, HEAD_DIM, 1),
        ),
    )
    output_map_tensor = S.view(
        output,
        S.bf16,
        S.make_layout(
            (2, BATCH * 2048 * QUERY_HEADS, 64),
            (64, HEAD_DIM, 1),
        ),
    )
    key_map_tensor = S.view(
        key,
        S.bf16,
        S.make_layout(
            (2, BATCH * 2048, 64),
            (64, HEAD_DIM, 1),
        ),
    )
    value_map_tensor = S.view(
        value,
        S.bf16,
        S.make_layout(
            (BATCH * (2048 // 8), 2, 8, 64),
            (8 * HEAD_DIM, 64, HEAD_DIM, 1),
        ),
    )

    query_layout = S.make_layout((2, 64, 64), (4096, 64, 1))
    key_layout = S.make_layout((2, 128, 64), (8192, 64, 1))
    value_layout = S.make_layout((16, 2, 8, 64), (1024, 512, 64, 1))
    query_desc = S.nvvm.make_tma_descriptor(
        query_map_tensor, query_layout, SWIZZLE_128B
    )
    output_desc = S.nvvm.make_tma_descriptor(
        output_map_tensor, query_layout, SWIZZLE_128B
    )
    key_desc = S.nvvm.make_tma_descriptor(key_map_tensor, key_layout, SWIZZLE_128B)
    value_desc = S.nvvm.make_tma_descriptor(
        value_map_tensor, value_layout, SWIZZLE_128B
    )

    S.nvvm.mbarrier_init(query_barrier, 0, count=1, predicate=tid == 0)
    S.nvvm.mbarrier_init(query_empty_barrier, 0, count=256, predicate=tid == 0)
    for stage in S.range(3, unroll=True):
        S.nvvm.mbarrier_init(key_barrier, stage, count=1, predicate=tid == 0)
        S.nvvm.mbarrier_init(key_empty_barrier, stage, count=256, predicate=tid == 0)
    for stage in S.range(2, unroll=True):
        S.nvvm.mbarrier_init(value_barrier, stage, count=1, predicate=tid == 0)
        S.nvvm.mbarrier_init(value_empty_barrier, stage, count=256, predicate=tid == 0)
        S.nvvm.mbarrier_init(output_empty_barrier, stage, count=1, predicate=tid == 0)
    S.syncthreads()

    if warpgroup == 0:
        S.nvvm.setmaxnreg_dec(40)
        elected = S.nvvm.elect_sync()
        if warpgroup_thread < 32:
            if elected:
                work_iteration = 0
                for work in S.range(
                    S.block_id(0),
                    total_work,
                    S.grid_dim(0),
                ):
                    if work_iteration != 0:
                        S.nvvm.mbarrier_try_wait_parity(
                            query_empty_barrier,
                            (work_iteration - 1) & 1,
                            10000000,
                            0,
                        )
                    batch = work // query_blocks
                    query_start = (
                        work - batch * query_blocks
                    ) * PACKED_QUERY_TOKENS
                    S.nvvm.mbarrier_arrive_expect_tx(
                        query_barrier,
                        QUERY_BLOCK * HEAD_DIM * 2,
                        0,
                        elected,
                    )
                    for consumer in S.range(CONSUMER_WARPGROUPS, unroll=True):
                        query_destination_raw = S.subview(
                            query_shared,
                            (consumer, 0),
                            (1, CONSUMER_ROWS * HEAD_DIM),
                            (1, 1),
                        )
                        query_destination = S.view(
                            query_destination_raw,
                            S.bf16,
                            S.make_layout((2, 64, 64), (4096, 64, 1)),
                        )
                        S.nvvm.tma_load(
                            query_destination,
                            query_desc,
                            (
                                0,
                                (batch * 2048 + query_start + consumer * 8)
                                * QUERY_HEADS,
                                0,
                            ),
                            query_barrier,
                            mbar_id=0,
                            predicate=elected,
                            expect_tx=False,
                        )

                    for key_tile in S.range(16):
                        stage = work_iteration * 16 + key_tile
                        key_buffer = stage % 3
                        value_buffer = stage & 1
                        if stage >= 3:
                            S.nvvm.mbarrier_try_wait_parity(
                                key_empty_barrier,
                                ((stage // 3) - 1) & 1,
                                10000000,
                                key_buffer,
                            )
                        key_destination_raw = S.subview(
                            key_shared,
                            (key_buffer, 0),
                            (1, KEY_BLOCK * HEAD_DIM),
                            (1, 1),
                        )
                        key_destination = S.view(
                            key_destination_raw,
                            S.bf16,
                            S.make_layout((2, 128, 64), (8192, 64, 1)),
                        )
                        S.nvvm.tma_load(
                            key_destination,
                            key_desc,
                            (0, batch * 2048 + key_tile * KEY_BLOCK, 0),
                            key_barrier,
                            mbar_id=key_buffer,
                            predicate=elected,
                        )
                        if stage >= 2:
                            S.nvvm.mbarrier_try_wait_parity(
                                value_empty_barrier,
                                ((stage >> 1) - 1) & 1,
                                10000000,
                                value_buffer,
                            )
                        value_destination_raw = S.subview(
                            value_shared,
                            (value_buffer, 0),
                            (1, KEY_BLOCK * HEAD_DIM),
                            (1, 1),
                        )
                        value_destination = S.view(
                            value_destination_raw,
                            S.bf16,
                            S.make_layout((16, 2, 8, 64), (1024, 512, 64, 1)),
                        )
                        S.nvvm.tma_load(
                            value_destination,
                            value_desc,
                            (
                                0,
                                0,
                                0,
                                key_tile * (KEY_BLOCK // 8) + batch * (2048 // 8),
                            ),
                            value_barrier,
                            mbar_id=value_buffer,
                            predicate=elected,
                        )
                    work_iteration = work_iteration + 1
        return

    S.nvvm.setmaxnreg_inc(232)
    consumer = warpgroup - 1
    lane = warpgroup_thread & 31
    warp = warpgroup_thread >> 5
    work_iteration = 0
    for work in S.range(
        S.block_id(0), total_work, S.grid_dim(0)
    ):
        batch = work // query_blocks
        query_start = (
            work - batch * query_blocks
        ) * PACKED_QUERY_TOKENS
        S.nvvm.mbarrier_try_wait_parity(query_barrier, work_iteration & 1, 10000000, 0)
        output_accumulator = S.full((1, 64), 0.0, S.f32)
        max_first = S.convert(NEG_INFINITY, S.f32)
        max_second = S.convert(NEG_INFINITY, S.f32)
        sum_first = S.convert(0.0, S.f32)
        sum_second = S.convert(0.0, S.f32)
        softmax_scale_log2 = S.convert(SOFTMAX_SCALE_LOG2, S.f32)
        # Use a distinct initializer so the mutable score and output buffers
        # remain separate through common-subexpression elimination.
        scores = S.full((1, 64), NEG_INFINITY, S.f32)
        probabilities = S.full((8, 8), 0.0, S.f32)

        stage_base = work_iteration * 16
        first_key_buffer = stage_base % 3
        S.nvvm.mbarrier_try_wait_parity(
            key_barrier, (stage_base // 3) & 1, 10000000, first_key_buffer
        )

        query_matrix = S.subview(
            query_shared,
            (consumer, 0),
            (CONSUMER_ROWS, HEAD_DIM),
            (HEAD_DIM, 1),
        )
        key_matrix = S.subview(
            key_shared,
            (first_key_buffer, 0),
            (KEY_BLOCK, HEAD_DIM),
            (HEAD_DIM, 1),
        )
        query_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
            query_matrix, SWIZZLE_128B, 0, 0, 0, 16, 1024
        )
        key_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
            key_matrix, SWIZZLE_128B, 0, 0, 0, 16, 1024
        )
        S.nvvm.wgmma_fence_aligned()
        qk_result = S.nvvm.wgmma_m64n128k128_f32_bf16_bf16_ss(
            query_matrix_desc, key_matrix_desc
        )
        S.nvvm.wgmma_group_sync_aligned()
        S.nvvm.wgmma_wait_group_sync(0)
        for result_element in S.range(64, unroll=True):
            scores[0, result_element] = qk_result[result_element]

        for key_tile in S.range(15):
            stage = stage_base + key_tile
            key_buffer = stage % 3
            value_buffer = stage & 1
            S.nvvm.mbarrier_arrive(key_empty_barrier, key_buffer)

            tile_max_first = S.convert(NEG_INFINITY, S.f32)
            tile_max_second = S.convert(NEG_INFINITY, S.f32)
            for tile in S.range(8, unroll=True):
                tile_max_first = S.nvvm.fast_fmax(
                    tile_max_first,
                    S.nvvm.fast_fmax(
                        S.nvvm.fast_fmax(scores[0, tile * 8], scores[0, tile * 8 + 1]),
                        S.nvvm.fast_fmax(scores[0, tile * 8 + 4], scores[0, tile * 8 + 5]),
                    ),
                )
                tile_max_second = S.nvvm.fast_fmax(
                    tile_max_second,
                    S.nvvm.fast_fmax(
                        S.nvvm.fast_fmax(scores[0, tile * 8 + 2], scores[0, tile * 8 + 3]),
                        S.nvvm.fast_fmax(scores[0, tile * 8 + 6], scores[0, tile * 8 + 7]),
                    ),
                )
            tile_max_first = _group_max(tile_max_first * softmax_scale_log2)
            tile_max_second = _group_max(tile_max_second * softmax_scale_log2)
            new_max_first = S.nvvm.fast_fmax(max_first, tile_max_first)
            new_max_second = S.nvvm.fast_fmax(max_second, tile_max_second)
            old_scale_first = S.nvvm.fast_exp2(max_first - new_max_first)
            old_scale_second = S.nvvm.fast_exp2(max_second - new_max_second)
            tile_sum_first = S.convert(0.0, S.f32)
            tile_sum_second = S.convert(0.0, S.f32)

            for tile in S.range(8, unroll=True):
                probabilities[tile, 0] = S.nvvm.fast_exp2(
                    S.nvvm.fast_fma(scores[0, tile * 8], softmax_scale_log2, -new_max_first)
                )
                probabilities[tile, 1] = S.nvvm.fast_exp2(
                    S.nvvm.fast_fma(
                        scores[0, tile * 8 + 1],
                        softmax_scale_log2,
                        -new_max_first,
                    )
                )
                probabilities[tile, 4] = S.nvvm.fast_exp2(
                    S.nvvm.fast_fma(
                        scores[0, tile * 8 + 4],
                        softmax_scale_log2,
                        -new_max_first,
                    )
                )
                probabilities[tile, 5] = S.nvvm.fast_exp2(
                    S.nvvm.fast_fma(
                        scores[0, tile * 8 + 5],
                        softmax_scale_log2,
                        -new_max_first,
                    )
                )
                tile_sum_first = (
                    tile_sum_first
                    + probabilities[tile, 0]
                    + probabilities[tile, 1]
                    + probabilities[tile, 4]
                    + probabilities[tile, 5]
                )
            for tile in S.range(8, unroll=True):
                probabilities[tile, 2] = S.nvvm.fast_exp2(
                    S.nvvm.fast_fma(
                        scores[0, tile * 8 + 2],
                        softmax_scale_log2,
                        -new_max_second,
                    )
                )
                probabilities[tile, 3] = S.nvvm.fast_exp2(
                    S.nvvm.fast_fma(
                        scores[0, tile * 8 + 3],
                        softmax_scale_log2,
                        -new_max_second,
                    )
                )
                probabilities[tile, 6] = S.nvvm.fast_exp2(
                    S.nvvm.fast_fma(
                        scores[0, tile * 8 + 6],
                        softmax_scale_log2,
                        -new_max_second,
                    )
                )
                probabilities[tile, 7] = S.nvvm.fast_exp2(
                    S.nvvm.fast_fma(
                        scores[0, tile * 8 + 7],
                        softmax_scale_log2,
                        -new_max_second,
                    )
                )
                tile_sum_second = (
                    tile_sum_second
                    + probabilities[tile, 2]
                    + probabilities[tile, 3]
                    + probabilities[tile, 6]
                    + probabilities[tile, 7]
                )
            tile_sum_first = _group_sum(tile_sum_first)
            tile_sum_second = _group_sum(tile_sum_second)
            sum_first = S.nvvm.fast_fma(sum_first, old_scale_first, tile_sum_first)
            sum_second = S.nvvm.fast_fma(sum_second, old_scale_second, tile_sum_second)
            max_first = new_max_first
            max_second = new_max_second
            for tile in S.range(8, unroll=True):
                output_accumulator[0, tile * 8] = (
                    output_accumulator[0, tile * 8] * old_scale_first
                )
                output_accumulator[0, tile * 8 + 1] = (
                    output_accumulator[0, tile * 8 + 1] * old_scale_first
                )
                output_accumulator[0, tile * 8 + 4] = (
                    output_accumulator[0, tile * 8 + 4] * old_scale_first
                )
                output_accumulator[0, tile * 8 + 5] = (
                    output_accumulator[0, tile * 8 + 5] * old_scale_first
                )
                output_accumulator[0, tile * 8 + 2] = (
                    output_accumulator[0, tile * 8 + 2] * old_scale_second
                )
                output_accumulator[0, tile * 8 + 3] = (
                    output_accumulator[0, tile * 8 + 3] * old_scale_second
                )
                output_accumulator[0, tile * 8 + 6] = (
                    output_accumulator[0, tile * 8 + 6] * old_scale_second
                )
                output_accumulator[0, tile * 8 + 7] = (
                    output_accumulator[0, tile * 8 + 7] * old_scale_second
                )

            S.nvvm.mbarrier_try_wait_parity(
                value_barrier, (stage >> 1) & 1, 10000000, value_buffer
            )
            value_matrix = S.subview(
                value_shared,
                (value_buffer, 0),
                (8, 2 * 8 * 64),
                (2 * 8 * 64, 1),
            )
            value_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
                value_matrix, SWIZZLE_128B, 0, 0, 0, 1024, 2048
            )
            S.nvvm.wgmma_fence_aligned()
            packed = S.full((1, 4), 0, S.i32)
            packed[0, 0] = _pack_bf16(probabilities[0, 0], probabilities[0, 1])
            packed[0, 1] = _pack_bf16(probabilities[0, 2], probabilities[0, 3])
            packed[0, 2] = _pack_bf16(probabilities[0, 4], probabilities[0, 5])
            packed[0, 3] = _pack_bf16(probabilities[0, 6], probabilities[0, 7])
            packed0 = packed[0]
            packed[0, 0] = _pack_bf16(probabilities[1, 0], probabilities[1, 1])
            packed[0, 1] = _pack_bf16(probabilities[1, 2], probabilities[1, 3])
            packed[0, 2] = _pack_bf16(probabilities[1, 4], probabilities[1, 5])
            packed[0, 3] = _pack_bf16(probabilities[1, 6], probabilities[1, 7])
            packed1 = packed[0]
            packed[0, 0] = _pack_bf16(probabilities[2, 0], probabilities[2, 1])
            packed[0, 1] = _pack_bf16(probabilities[2, 2], probabilities[2, 3])
            packed[0, 2] = _pack_bf16(probabilities[2, 4], probabilities[2, 5])
            packed[0, 3] = _pack_bf16(probabilities[2, 6], probabilities[2, 7])
            packed2 = packed[0]
            packed[0, 0] = _pack_bf16(probabilities[3, 0], probabilities[3, 1])
            packed[0, 1] = _pack_bf16(probabilities[3, 2], probabilities[3, 3])
            packed[0, 2] = _pack_bf16(probabilities[3, 4], probabilities[3, 5])
            packed[0, 3] = _pack_bf16(probabilities[3, 6], probabilities[3, 7])
            packed3 = packed[0]
            packed[0, 0] = _pack_bf16(probabilities[4, 0], probabilities[4, 1])
            packed[0, 1] = _pack_bf16(probabilities[4, 2], probabilities[4, 3])
            packed[0, 2] = _pack_bf16(probabilities[4, 4], probabilities[4, 5])
            packed[0, 3] = _pack_bf16(probabilities[4, 6], probabilities[4, 7])
            packed4 = packed[0]
            packed[0, 0] = _pack_bf16(probabilities[5, 0], probabilities[5, 1])
            packed[0, 1] = _pack_bf16(probabilities[5, 2], probabilities[5, 3])
            packed[0, 2] = _pack_bf16(probabilities[5, 4], probabilities[5, 5])
            packed[0, 3] = _pack_bf16(probabilities[5, 6], probabilities[5, 7])
            packed5 = packed[0]
            packed[0, 0] = _pack_bf16(probabilities[6, 0], probabilities[6, 1])
            packed[0, 1] = _pack_bf16(probabilities[6, 2], probabilities[6, 3])
            packed[0, 2] = _pack_bf16(probabilities[6, 4], probabilities[6, 5])
            packed[0, 3] = _pack_bf16(probabilities[6, 6], probabilities[6, 7])
            packed6 = packed[0]
            packed[0, 0] = _pack_bf16(probabilities[7, 0], probabilities[7, 1])
            packed[0, 1] = _pack_bf16(probabilities[7, 2], probabilities[7, 3])
            packed[0, 2] = _pack_bf16(probabilities[7, 4], probabilities[7, 5])
            packed[0, 3] = _pack_bf16(probabilities[7, 6], probabilities[7, 7])
            packed7 = packed[0]
            pv_result = S.nvvm.wgmma_m64n128k128_f32_bf16_bf16_rs(
                packed0,
                packed1,
                packed2,
                packed3,
                packed4,
                packed5,
                packed6,
                packed7,
                value_matrix_desc,
                output_accumulator[0],
            )
            S.nvvm.wgmma_group_sync_aligned()

            next_stage = stage + 1
            next_key_buffer = next_stage % 3
            S.nvvm.mbarrier_try_wait_parity(
                key_barrier,
                (next_stage // 3) & 1,
                10000000,
                next_key_buffer,
            )
            next_key_matrix = S.subview(
                key_shared,
                (next_key_buffer, 0),
                (KEY_BLOCK, HEAD_DIM),
                (HEAD_DIM, 1),
            )
            key_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
                next_key_matrix, SWIZZLE_128B, 0, 0, 0, 16, 1024
            )
            query_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
                query_matrix, SWIZZLE_128B, 0, 0, 0, 16, 1024
            )
            S.nvvm.wgmma_fence_aligned()
            next_qk_result = S.nvvm.wgmma_m64n128k128_f32_bf16_bf16_ss(
                query_matrix_desc, key_matrix_desc
            )
            S.nvvm.wgmma_group_sync_aligned()
            S.nvvm.wgmma_wait_group_sync(1)
            for result_element in S.range(64, unroll=True):
                output_accumulator[0, result_element] = pv_result[result_element]
            S.nvvm.mbarrier_arrive(value_empty_barrier, value_buffer)
            S.nvvm.wgmma_wait_group_sync(0)
            for result_element in S.range(64, unroll=True):
                scores[0, result_element] = next_qk_result[result_element]

        key_tile = 15
        stage = stage_base + key_tile
        key_buffer = stage % 3
        value_buffer = stage & 1
        S.nvvm.mbarrier_arrive(key_empty_barrier, key_buffer)

        tile_max_first = S.convert(NEG_INFINITY, S.f32)
        tile_max_second = S.convert(NEG_INFINITY, S.f32)
        for tile in S.range(8, unroll=True):
            tile_max_first = S.nvvm.fast_fmax(
                tile_max_first,
                S.nvvm.fast_fmax(
                    S.nvvm.fast_fmax(scores[0, tile * 8], scores[0, tile * 8 + 1]),
                    S.nvvm.fast_fmax(scores[0, tile * 8 + 4], scores[0, tile * 8 + 5]),
                ),
            )
            tile_max_second = S.nvvm.fast_fmax(
                tile_max_second,
                S.nvvm.fast_fmax(
                    S.nvvm.fast_fmax(scores[0, tile * 8 + 2], scores[0, tile * 8 + 3]),
                    S.nvvm.fast_fmax(scores[0, tile * 8 + 6], scores[0, tile * 8 + 7]),
                ),
            )
        tile_max_first = _group_max(tile_max_first * softmax_scale_log2)
        tile_max_second = _group_max(tile_max_second * softmax_scale_log2)
        new_max_first = S.nvvm.fast_fmax(max_first, tile_max_first)
        new_max_second = S.nvvm.fast_fmax(max_second, tile_max_second)
        old_scale_first = S.nvvm.fast_exp2(max_first - new_max_first)
        old_scale_second = S.nvvm.fast_exp2(max_second - new_max_second)
        tile_sum_first = S.convert(0.0, S.f32)
        tile_sum_second = S.convert(0.0, S.f32)

        for tile in S.range(8, unroll=True):
            probabilities[tile, 0] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(scores[0, tile * 8], softmax_scale_log2, -new_max_first)
            )
            probabilities[tile, 1] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[0, tile * 8 + 1],
                    softmax_scale_log2,
                    -new_max_first,
                )
            )
            probabilities[tile, 4] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[0, tile * 8 + 4],
                    softmax_scale_log2,
                    -new_max_first,
                )
            )
            probabilities[tile, 5] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[0, tile * 8 + 5],
                    softmax_scale_log2,
                    -new_max_first,
                )
            )
            tile_sum_first = (
                tile_sum_first
                + probabilities[tile, 0]
                + probabilities[tile, 1]
                + probabilities[tile, 4]
                + probabilities[tile, 5]
            )
        for tile in S.range(8, unroll=True):
            probabilities[tile, 2] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[0, tile * 8 + 2],
                    softmax_scale_log2,
                    -new_max_second,
                )
            )
            probabilities[tile, 3] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[0, tile * 8 + 3],
                    softmax_scale_log2,
                    -new_max_second,
                )
            )
            probabilities[tile, 6] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[0, tile * 8 + 6],
                    softmax_scale_log2,
                    -new_max_second,
                )
            )
            probabilities[tile, 7] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[0, tile * 8 + 7],
                    softmax_scale_log2,
                    -new_max_second,
                )
            )
            tile_sum_second = (
                tile_sum_second
                + probabilities[tile, 2]
                + probabilities[tile, 3]
                + probabilities[tile, 6]
                + probabilities[tile, 7]
            )
        tile_sum_first = _group_sum(tile_sum_first)
        tile_sum_second = _group_sum(tile_sum_second)
        sum_first = S.nvvm.fast_fma(sum_first, old_scale_first, tile_sum_first)
        sum_second = S.nvvm.fast_fma(sum_second, old_scale_second, tile_sum_second)
        max_first = new_max_first
        max_second = new_max_second
        for tile in S.range(8, unroll=True):
            output_accumulator[0, tile * 8] = (
                output_accumulator[0, tile * 8] * old_scale_first
            )
            output_accumulator[0, tile * 8 + 1] = (
                output_accumulator[0, tile * 8 + 1] * old_scale_first
            )
            output_accumulator[0, tile * 8 + 4] = (
                output_accumulator[0, tile * 8 + 4] * old_scale_first
            )
            output_accumulator[0, tile * 8 + 5] = (
                output_accumulator[0, tile * 8 + 5] * old_scale_first
            )
            output_accumulator[0, tile * 8 + 2] = (
                output_accumulator[0, tile * 8 + 2] * old_scale_second
            )
            output_accumulator[0, tile * 8 + 3] = (
                output_accumulator[0, tile * 8 + 3] * old_scale_second
            )
            output_accumulator[0, tile * 8 + 6] = (
                output_accumulator[0, tile * 8 + 6] * old_scale_second
            )
            output_accumulator[0, tile * 8 + 7] = (
                output_accumulator[0, tile * 8 + 7] * old_scale_second
            )

        S.nvvm.mbarrier_try_wait_parity(
            value_barrier, (stage >> 1) & 1, 10000000, value_buffer
        )
        value_matrix = S.subview(
            value_shared,
            (value_buffer, 0),
            (8, 2 * 8 * 64),
            (2 * 8 * 64, 1),
        )
        value_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
            value_matrix, SWIZZLE_128B, 0, 0, 0, 1024, 2048
        )
        S.nvvm.wgmma_fence_aligned()
        last_packed = S.full((1, 4), 0, S.i32)
        last_packed[0, 0] = _pack_bf16(probabilities[0, 0], probabilities[0, 1])
        last_packed[0, 1] = _pack_bf16(probabilities[0, 2], probabilities[0, 3])
        last_packed[0, 2] = _pack_bf16(probabilities[0, 4], probabilities[0, 5])
        last_packed[0, 3] = _pack_bf16(probabilities[0, 6], probabilities[0, 7])
        last_packed0 = last_packed[0]
        last_packed[0, 0] = _pack_bf16(probabilities[1, 0], probabilities[1, 1])
        last_packed[0, 1] = _pack_bf16(probabilities[1, 2], probabilities[1, 3])
        last_packed[0, 2] = _pack_bf16(probabilities[1, 4], probabilities[1, 5])
        last_packed[0, 3] = _pack_bf16(probabilities[1, 6], probabilities[1, 7])
        last_packed1 = last_packed[0]
        last_packed[0, 0] = _pack_bf16(probabilities[2, 0], probabilities[2, 1])
        last_packed[0, 1] = _pack_bf16(probabilities[2, 2], probabilities[2, 3])
        last_packed[0, 2] = _pack_bf16(probabilities[2, 4], probabilities[2, 5])
        last_packed[0, 3] = _pack_bf16(probabilities[2, 6], probabilities[2, 7])
        last_packed2 = last_packed[0]
        last_packed[0, 0] = _pack_bf16(probabilities[3, 0], probabilities[3, 1])
        last_packed[0, 1] = _pack_bf16(probabilities[3, 2], probabilities[3, 3])
        last_packed[0, 2] = _pack_bf16(probabilities[3, 4], probabilities[3, 5])
        last_packed[0, 3] = _pack_bf16(probabilities[3, 6], probabilities[3, 7])
        last_packed3 = last_packed[0]
        last_packed[0, 0] = _pack_bf16(probabilities[4, 0], probabilities[4, 1])
        last_packed[0, 1] = _pack_bf16(probabilities[4, 2], probabilities[4, 3])
        last_packed[0, 2] = _pack_bf16(probabilities[4, 4], probabilities[4, 5])
        last_packed[0, 3] = _pack_bf16(probabilities[4, 6], probabilities[4, 7])
        last_packed4 = last_packed[0]
        last_packed[0, 0] = _pack_bf16(probabilities[5, 0], probabilities[5, 1])
        last_packed[0, 1] = _pack_bf16(probabilities[5, 2], probabilities[5, 3])
        last_packed[0, 2] = _pack_bf16(probabilities[5, 4], probabilities[5, 5])
        last_packed[0, 3] = _pack_bf16(probabilities[5, 6], probabilities[5, 7])
        last_packed5 = last_packed[0]
        last_packed[0, 0] = _pack_bf16(probabilities[6, 0], probabilities[6, 1])
        last_packed[0, 1] = _pack_bf16(probabilities[6, 2], probabilities[6, 3])
        last_packed[0, 2] = _pack_bf16(probabilities[6, 4], probabilities[6, 5])
        last_packed[0, 3] = _pack_bf16(probabilities[6, 6], probabilities[6, 7])
        last_packed6 = last_packed[0]
        last_packed[0, 0] = _pack_bf16(probabilities[7, 0], probabilities[7, 1])
        last_packed[0, 1] = _pack_bf16(probabilities[7, 2], probabilities[7, 3])
        last_packed[0, 2] = _pack_bf16(probabilities[7, 4], probabilities[7, 5])
        last_packed[0, 3] = _pack_bf16(probabilities[7, 6], probabilities[7, 7])
        last_packed7 = last_packed[0]
        pv_result = S.nvvm.wgmma_m64n128k128_f32_bf16_bf16_rs(
            last_packed0,
            last_packed1,
            last_packed2,
            last_packed3,
            last_packed4,
            last_packed5,
            last_packed6,
            last_packed7,
            value_matrix_desc,
            output_accumulator[0],
        )
        S.nvvm.wgmma_group_sync_aligned()

        S.nvvm.wgmma_wait_group_sync(0)
        S.nvvm.mbarrier_arrive(query_empty_barrier, 0)
        for result_element in S.range(64, unroll=True):
            output_accumulator[0, result_element] = pv_result[result_element]
        S.nvvm.mbarrier_arrive(value_empty_barrier, value_buffer)

        if work_iteration != 0:
            S.nvvm.mbarrier_try_wait_parity(
                output_empty_barrier,
                (work_iteration - 1) & 1,
                10000000,
                consumer,
            )
        inverse_first = S.nvvm.fast_rcp(sum_first)
        inverse_second = S.nvvm.fast_rcp(sum_second)
        for tile in S.range(8, unroll=True):
            row = warp * 16 + (lane & 15)
            column = tile * 16 + (lane >> 4) * 8
            linear = row * 64 + (column & 63)
            if column >= 64:
                linear = linear + CONSUMER_ROWS * 64
            offset = linear ^ ((linear & 0x1C0) >> 3)
            destination = S.subview(output_shared, (consumer, offset), (8, 8), (1, 1))
            packed = S.full((1, 4), 0, S.i32)
            packed[0, 0] = _pack_bf16(
                output_accumulator[0, tile * 8] * inverse_first,
                output_accumulator[0, tile * 8 + 1] * inverse_first,
            )
            packed[0, 1] = _pack_bf16(
                output_accumulator[0, tile * 8 + 2] * inverse_second,
                output_accumulator[0, tile * 8 + 3] * inverse_second,
            )
            packed[0, 2] = _pack_bf16(
                output_accumulator[0, tile * 8 + 4] * inverse_first,
                output_accumulator[0, tile * 8 + 5] * inverse_first,
            )
            packed[0, 3] = _pack_bf16(
                output_accumulator[0, tile * 8 + 6] * inverse_second,
                output_accumulator[0, tile * 8 + 7] * inverse_second,
            )
            S.nvvm.stmatrix_m8n8_x4_b16(destination, packed[0])
        S.nvvm.fence_proxy_async_shared_cta()
        S.nvvm.named_barrier_sync(1 + consumer, WARPGROUP_THREADS)
        if warpgroup_thread == 0:
            output_source_raw = S.subview(
                output_shared,
                (consumer, 0),
                (1, CONSUMER_ROWS * HEAD_DIM),
                (1, 1),
            )
            output_source = S.view(
                output_source_raw,
                S.bf16,
                S.make_layout((2, 64, 64), (4096, 64, 1)),
            )
            S.nvvm.tma_store(
                output_source,
                output_desc,
                (
                    0,
                    (batch * 2048 + query_start + consumer * 8) * QUERY_HEADS,
                    0,
                ),
                predicate=warpgroup_thread == 0,
            )
            S.nvvm.cp_async_bulk_commit_group()
            S.nvvm.cp_async_bulk_wait_group(0, read=True)
            S.nvvm.mbarrier_arrive(output_empty_barrier, consumer)
        work_iteration = work_iteration + 1



@avelang.jit
def _attention_mqa_4096_kernel(
    query: S.Tensor((16, 4096, 8, 128), S.bf16),
    key: S.Tensor((16, 4096, 1, 128), S.bf16),
    value: S.Tensor((16, 4096, 1, 128), S.bf16),
    output: S.Tensor((16, 4096, 8, 128), S.bf16),
):
    tid = S.thread_id(0)
    warpgroup = tid // WARPGROUP_THREADS
    warpgroup_thread = tid % WARPGROUP_THREADS
    query_blocks = 4096 // PACKED_QUERY_TOKENS
    key_tiles = 4096 // 128

    # Ave's module builder emits shared globals in reverse declaration order.
    # Keep the resulting arena identical to CUDA's SharedStorage: query,
    # output, key, value, then barriers. Besides making WGMMA descriptors
    # directly comparable, this keeps every TMA destination sector in range.
    query_barrier = S.nvvm.mbarrier_create()
    key_barrier = S.nvvm.mbarrier_create(3)
    value_barrier = S.nvvm.mbarrier_create(2)
    key_empty_barrier = S.nvvm.mbarrier_create(3)
    value_empty_barrier = S.nvvm.mbarrier_create(2)

    value_shared = S.make_shared((2, 128 * HEAD_DIM), S.bf16, 128)
    key_shared = S.make_shared((3, 128 * HEAD_DIM), S.bf16, 128)
    query_shared = S.make_shared(
        (CONSUMER_WARPGROUPS, CONSUMER_ROWS * HEAD_DIM), S.bf16, 128
    )

    # Reinterpret the contiguous PyTorch tensors exactly like the CUDA tensor
    # maps. TMA coordinates are ordered from the fastest-changing dimension.
    query_map_tensor = S.view(
        query,
        S.bf16,
        S.make_layout(
            (2, BATCH * 4096 * QUERY_HEADS, 64),
            (64, HEAD_DIM, 1),
        ),
    )
    output_map_tensor = S.view(
        output,
        S.bf16,
        S.make_layout(
            (2, BATCH * 4096 * QUERY_HEADS, 64),
            (64, HEAD_DIM, 1),
        ),
    )
    key_map_tensor = S.view(
        key,
        S.bf16,
        S.make_layout(
            (2, BATCH * 4096, 64),
            (64, HEAD_DIM, 1),
        ),
    )
    value_map_tensor = S.view(
        value,
        S.bf16,
        S.make_layout(
            (BATCH * (4096 // 8), 2, 8, 64),
            (8 * HEAD_DIM, 64, HEAD_DIM, 1),
        ),
    )

    query_layout = S.make_layout((2, 64, 64), (4096, 64, 1))
    key_layout = S.make_layout((2, 128, 64), (8192, 64, 1))
    value_layout = S.make_layout((16, 2, 8, 64), (1024, 512, 64, 1))
    query_desc = S.nvvm.make_tma_descriptor(
        query_map_tensor, query_layout, SWIZZLE_128B
    )
    output_desc = S.nvvm.make_tma_descriptor(
        output_map_tensor, query_layout, SWIZZLE_128B
    )
    key_desc = S.nvvm.make_tma_descriptor(key_map_tensor, key_layout, SWIZZLE_128B)
    value_desc = S.nvvm.make_tma_descriptor(
        value_map_tensor, value_layout, SWIZZLE_128B
    )

    if tid == 0:
        S.nvvm.mbarrier_init(query_barrier, 0, count=1)
        for stage in S.range(3, unroll=True):
            S.nvvm.mbarrier_init(key_barrier, stage, count=1)
            S.nvvm.mbarrier_init(key_empty_barrier, stage, count=256)
        for stage in S.range(2, unroll=True):
            S.nvvm.mbarrier_init(value_barrier, stage, count=1)
            S.nvvm.mbarrier_init(value_empty_barrier, stage, count=256)
    S.syncthreads()

    if warpgroup == 0:
        S.nvvm.setmaxnreg_dec(40)
        elected = S.nvvm.elect_sync()
        if warpgroup_thread < 32:
            if elected:
                work = S.block_id(0)
                batch = work // query_blocks
                query_start = (
                    work - batch * query_blocks
                ) * PACKED_QUERY_TOKENS
                S.nvvm.mbarrier_arrive_expect_tx(
                    query_barrier,
                    QUERY_BLOCK * HEAD_DIM * 2,
                    0,
                    elected,
                )
                for consumer in S.range(CONSUMER_WARPGROUPS, unroll=True):
                    query_destination_raw = S.subview(
                        query_shared,
                        (consumer, 0),
                        (1, CONSUMER_ROWS * HEAD_DIM),
                        (1, 1),
                    )
                    query_destination = S.view(
                        query_destination_raw,
                        S.bf16,
                        S.make_layout((2, 64, 64), (4096, 64, 1)),
                    )
                    S.nvvm.tma_load(
                        query_destination,
                        query_desc,
                        (
                            0,
                            (batch * 4096 + query_start + consumer * 8)
                            * QUERY_HEADS,
                            0,
                        ),
                        query_barrier,
                        mbar_id=0,
                        predicate=elected,
                        expect_tx=False,
                    )

                for key_tile in S.range(2, unroll=True):
                    stage = key_tile
                    key_buffer = stage % 3
                    value_buffer = stage & 1
                    if stage >= 3:
                        S.nvvm.mbarrier_try_wait_parity(
                            key_empty_barrier,
                            ((stage // 3) - 1) & 1,
                            10000000,
                            key_buffer,
                        )
                    key_destination_raw = S.subview(
                        key_shared,
                        (key_buffer, 0),
                        (1, 128 * HEAD_DIM),
                        (1, 1),
                    )
                    key_destination = S.view(
                        key_destination_raw,
                        S.bf16,
                        S.make_layout((2, 128, 64), (8192, 64, 1)),
                    )
                    S.nvvm.tma_load(
                        key_destination,
                        key_desc,
                        (0, batch * 4096 + key_tile * 128, 0),
                        key_barrier,
                        mbar_id=key_buffer,
                        predicate=elected,
                    )
                    if stage >= 2:
                        S.nvvm.mbarrier_try_wait_parity(
                            value_empty_barrier,
                            ((stage >> 1) - 1) & 1,
                            10000000,
                            value_buffer,
                        )
                    value_destination_raw = S.subview(
                        value_shared,
                        (value_buffer, 0),
                        (1, 128 * HEAD_DIM),
                        (1, 1),
                    )
                    value_destination = S.view(
                        value_destination_raw,
                        S.bf16,
                        S.make_layout((16, 2, 8, 64), (1024, 512, 64, 1)),
                    )
                    S.nvvm.tma_load(
                        value_destination,
                        value_desc,
                        (
                            0,
                            0,
                            0,
                            key_tile * (128 // 8) + batch * (4096 // 8),
                        ),
                        value_barrier,
                        mbar_id=value_buffer,
                        predicate=elected,
                    )
                for key_tile in S.range(2, 4096 // 128, unroll=True):
                    stage = key_tile
                    key_buffer = stage % 3
                    value_buffer = stage & 1
                    if stage >= 3:
                        S.nvvm.mbarrier_try_wait_parity(
                            key_empty_barrier,
                            ((stage // 3) - 1) & 1,
                            10000000,
                            key_buffer,
                        )
                    key_destination_raw = S.subview(
                        key_shared,
                        (key_buffer, 0),
                        (1, 128 * HEAD_DIM),
                        (1, 1),
                    )
                    key_destination = S.view(
                        key_destination_raw,
                        S.bf16,
                        S.make_layout((2, 128, 64), (8192, 64, 1)),
                    )
                    S.nvvm.tma_load(
                        key_destination,
                        key_desc,
                        (0, batch * 4096 + key_tile * 128, 0),
                        key_barrier,
                        mbar_id=key_buffer,
                        predicate=elected,
                    )
                    if stage >= 2:
                        S.nvvm.mbarrier_try_wait_parity(
                            value_empty_barrier,
                            ((stage >> 1) - 1) & 1,
                            10000000,
                            value_buffer,
                        )
                    value_destination_raw = S.subview(
                        value_shared,
                        (value_buffer, 0),
                        (1, 128 * HEAD_DIM),
                        (1, 1),
                    )
                    value_destination = S.view(
                        value_destination_raw,
                        S.bf16,
                        S.make_layout((16, 2, 8, 64), (1024, 512, 64, 1)),
                    )
                    S.nvvm.tma_load(
                        value_destination,
                        value_desc,
                        (
                            0,
                            0,
                            0,
                            key_tile * (128 // 8) + batch * (4096 // 8),
                        ),
                        value_barrier,
                        mbar_id=value_buffer,
                        predicate=elected,
                    )
        return

    S.nvvm.setmaxnreg_inc(232)
    consumer = warpgroup - 1
    lane = warpgroup_thread & 31
    warp = warpgroup_thread >> 5
    work = S.block_id(0)
    batch = work // query_blocks
    query_start = (
        work - batch * query_blocks
    ) * PACKED_QUERY_TOKENS
    S.nvvm.mbarrier_try_wait_parity(query_barrier, 0, 10000000, 0)
    output_accumulator = S.full((1, 64), 0.0, S.f32)
    max_first = S.convert(NEG_INFINITY, S.f32)
    max_second = S.convert(NEG_INFINITY, S.f32)
    sum_first = S.convert(0.0, S.f32)
    sum_second = S.convert(0.0, S.f32)
    softmax_scale_log2 = S.convert(SOFTMAX_SCALE_LOG2, S.f32)
    # Use a distinct initializer so the mutable score and output buffers
    # remain separate through common-subexpression elimination.
    scores = S.full((64,), NEG_INFINITY, S.f32)

    stage_base = 0
    first_key_buffer = stage_base & 1
    S.nvvm.mbarrier_try_wait_parity(
        key_barrier, (stage_base >> 1) & 1, 10000000, first_key_buffer
    )

    query_matrix = S.subview(
        query_shared,
        (consumer, 0),
        (CONSUMER_ROWS, HEAD_DIM),
        (HEAD_DIM, 1),
    )
    key_matrix = S.subview(
        key_shared,
        (first_key_buffer, 0),
        (128, HEAD_DIM),
        (HEAD_DIM, 1),
    )
    query_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
        query_matrix, SWIZZLE_128B, 0, 0, 0, 16, 1024
    )
    key_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
        key_matrix, SWIZZLE_128B, 0, 0, 0, 16, 1024
    )
    S.nvvm.wgmma_fence_aligned()
    qk_result = S.nvvm.wgmma_m64n128k128_f32_bf16_bf16_ss(
        query_matrix_desc, key_matrix_desc
    )
    S.nvvm.wgmma_group_sync_aligned()
    S.nvvm.wgmma_wait_group_sync(0)
    for result_element in S.range(64, unroll=True):
        scores[result_element] = qk_result[result_element]
    for key_tile in S.range(key_tiles - 1):
        stage = stage_base + key_tile
        key_buffer = stage % 3
        value_buffer = stage & 1
        S.nvvm.mbarrier_arrive(key_empty_barrier, key_buffer)
        tile_max_first = S.convert(NEG_INFINITY, S.f32)
        tile_max_second = S.convert(NEG_INFINITY, S.f32)
        for tile in S.range(8, unroll=True):
            tile_max_first = S.nvvm.fast_fmax(
                tile_max_first,
                S.nvvm.fast_fmax(
                    S.nvvm.fast_fmax(scores[tile * 8], scores[tile * 8 + 1]),
                    S.nvvm.fast_fmax(scores[tile * 8 + 4], scores[tile * 8 + 5]),
                ),
            )
            tile_max_second = S.nvvm.fast_fmax(
                tile_max_second,
                S.nvvm.fast_fmax(
                    S.nvvm.fast_fmax(scores[tile * 8 + 2], scores[tile * 8 + 3]),
                    S.nvvm.fast_fmax(scores[tile * 8 + 6], scores[tile * 8 + 7]),
                ),
            )
        tile_max_first = _group_max(tile_max_first * softmax_scale_log2)
        tile_max_second = _group_max(tile_max_second * softmax_scale_log2)
        new_max_first = S.nvvm.fast_fmax(max_first, tile_max_first)
        new_max_second = S.nvvm.fast_fmax(max_second, tile_max_second)
        old_scale_first = S.nvvm.fast_exp2(max_first - new_max_first)
        old_scale_second = S.nvvm.fast_exp2(max_second - new_max_second)
        tile_sum_first = S.convert(0.0, S.f32)
        tile_sum_second = S.convert(0.0, S.f32)

        for tile in S.range(8, unroll=True):
            scores[tile * 8 + 0] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(scores[tile * 8], softmax_scale_log2, -new_max_first)
            )
            scores[tile * 8 + 1] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[tile * 8 + 1],
                    softmax_scale_log2,
                    -new_max_first,
                )
            )
            scores[tile * 8 + 4] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[tile * 8 + 4],
                    softmax_scale_log2,
                    -new_max_first,
                )
            )
            scores[tile * 8 + 5] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[tile * 8 + 5],
                    softmax_scale_log2,
                    -new_max_first,
                )
            )
            tile_sum_first = (
                tile_sum_first
                + scores[tile * 8 + 0]
                + scores[tile * 8 + 1]
                + scores[tile * 8 + 4]
                + scores[tile * 8 + 5]
            )
        for tile in S.range(8, unroll=True):
            scores[tile * 8 + 2] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[tile * 8 + 2],
                    softmax_scale_log2,
                    -new_max_second,
                )
            )
            scores[tile * 8 + 3] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[tile * 8 + 3],
                    softmax_scale_log2,
                    -new_max_second,
                )
            )
            scores[tile * 8 + 6] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[tile * 8 + 6],
                    softmax_scale_log2,
                    -new_max_second,
                )
            )
            scores[tile * 8 + 7] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[tile * 8 + 7],
                    softmax_scale_log2,
                    -new_max_second,
                )
            )
            tile_sum_second = (
                tile_sum_second
                + scores[tile * 8 + 2]
                + scores[tile * 8 + 3]
                + scores[tile * 8 + 6]
                + scores[tile * 8 + 7]
            )
        tile_sum_first = _group_sum(tile_sum_first)
        tile_sum_second = _group_sum(tile_sum_second)
        sum_first = S.nvvm.fast_fma(sum_first, old_scale_first, tile_sum_first)
        sum_second = S.nvvm.fast_fma(sum_second, old_scale_second, tile_sum_second)
        max_first = new_max_first
        max_second = new_max_second

        # Start the next-K wait before rescaling the output accumulator.  The
        # independent multiplies hide the barrier latency, matching CUDA's
        # scheduled one-shot pipeline.
        next_stage = stage + 1
        next_key_buffer = next_stage % 3
        S.nvvm.mbarrier_try_wait_parity(
            key_barrier,
            (next_stage // 3) & 1,
            10000000,
            next_key_buffer,
        )
        for tile in S.range(8, unroll=True):
            output_accumulator[0, tile * 8] = (
                output_accumulator[0, tile * 8] * old_scale_first
            )
            output_accumulator[0, tile * 8 + 1] = (
                output_accumulator[0, tile * 8 + 1] * old_scale_first
            )
            output_accumulator[0, tile * 8 + 4] = (
                output_accumulator[0, tile * 8 + 4] * old_scale_first
            )
            output_accumulator[0, tile * 8 + 5] = (
                output_accumulator[0, tile * 8 + 5] * old_scale_first
            )
            output_accumulator[0, tile * 8 + 2] = (
                output_accumulator[0, tile * 8 + 2] * old_scale_second
            )
            output_accumulator[0, tile * 8 + 3] = (
                output_accumulator[0, tile * 8 + 3] * old_scale_second
            )
            output_accumulator[0, tile * 8 + 6] = (
                output_accumulator[0, tile * 8 + 6] * old_scale_second
            )
            output_accumulator[0, tile * 8 + 7] = (
                output_accumulator[0, tile * 8 + 7] * old_scale_second
            )

        S.nvvm.mbarrier_try_wait_parity(
            value_barrier, (stage >> 1) & 1, 10000000, value_buffer
        )
        value_matrix = S.subview(
            value_shared,
            (value_buffer, 0),
            (8, 2 * 8 * 64),
            (2 * 8 * 64, 1),
        )
        value_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
            value_matrix, SWIZZLE_128B, 0, 0, 0, 1024, 2048
        )
        S.nvvm.wgmma_fence_aligned()
        packed = S.full((1, 4), 0, S.i32)
        packed[0, 0] = _pack_bf16(scores[0], scores[1])
        packed[0, 1] = _pack_bf16(scores[2], scores[3])
        packed[0, 2] = _pack_bf16(scores[4], scores[5])
        packed[0, 3] = _pack_bf16(scores[6], scores[7])
        packed0 = packed[0]
        packed[0, 0] = _pack_bf16(scores[8], scores[9])
        packed[0, 1] = _pack_bf16(scores[10], scores[11])
        packed[0, 2] = _pack_bf16(scores[12], scores[13])
        packed[0, 3] = _pack_bf16(scores[14], scores[15])
        packed1 = packed[0]
        packed[0, 0] = _pack_bf16(scores[16], scores[17])
        packed[0, 1] = _pack_bf16(scores[18], scores[19])
        packed[0, 2] = _pack_bf16(scores[20], scores[21])
        packed[0, 3] = _pack_bf16(scores[22], scores[23])
        packed2 = packed[0]
        packed[0, 0] = _pack_bf16(scores[24], scores[25])
        packed[0, 1] = _pack_bf16(scores[26], scores[27])
        packed[0, 2] = _pack_bf16(scores[28], scores[29])
        packed[0, 3] = _pack_bf16(scores[30], scores[31])
        packed3 = packed[0]
        packed[0, 0] = _pack_bf16(scores[32], scores[33])
        packed[0, 1] = _pack_bf16(scores[34], scores[35])
        packed[0, 2] = _pack_bf16(scores[36], scores[37])
        packed[0, 3] = _pack_bf16(scores[38], scores[39])
        packed4 = packed[0]
        packed[0, 0] = _pack_bf16(scores[40], scores[41])
        packed[0, 1] = _pack_bf16(scores[42], scores[43])
        packed[0, 2] = _pack_bf16(scores[44], scores[45])
        packed[0, 3] = _pack_bf16(scores[46], scores[47])
        packed5 = packed[0]
        packed[0, 0] = _pack_bf16(scores[48], scores[49])
        packed[0, 1] = _pack_bf16(scores[50], scores[51])
        packed[0, 2] = _pack_bf16(scores[52], scores[53])
        packed[0, 3] = _pack_bf16(scores[54], scores[55])
        packed6 = packed[0]
        packed[0, 0] = _pack_bf16(scores[56], scores[57])
        packed[0, 1] = _pack_bf16(scores[58], scores[59])
        packed[0, 2] = _pack_bf16(scores[60], scores[61])
        packed[0, 3] = _pack_bf16(scores[62], scores[63])
        packed7 = packed[0]
        pv_result = S.nvvm.wgmma_m64n128k128_f32_bf16_bf16_rs(
            packed0,
            packed1,
            packed2,
            packed3,
            packed4,
            packed5,
            packed6,
            packed7,
            value_matrix_desc,
            output_accumulator[0],
        )
        S.nvvm.wgmma_group_sync_aligned()

        next_stage = stage + 1
        next_key_buffer = next_stage % 3
        next_key_matrix = S.subview(
            key_shared,
            (next_key_buffer, 0),
            (128, HEAD_DIM),
            (HEAD_DIM, 1),
        )
        key_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
            next_key_matrix, SWIZZLE_128B, 0, 0, 0, 16, 1024
        )
        query_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
            query_matrix, SWIZZLE_128B, 0, 0, 0, 16, 1024
        )
        S.nvvm.wgmma_fence_aligned()
        next_scores = S.nvvm.wgmma_m64n128k128_f32_bf16_bf16_ss(
            query_matrix_desc, key_matrix_desc
        )
        S.nvvm.wgmma_group_sync_aligned()
        S.nvvm.wgmma_wait_group_sync(1)
        for result_element in S.range(64, unroll=True):
            output_accumulator[0, result_element] = pv_result[result_element]
        S.nvvm.mbarrier_arrive(value_empty_barrier, value_buffer)
        S.nvvm.wgmma_wait_group_sync(0)
        for result_element in S.range(64, unroll=True):
            scores[result_element] = next_scores[result_element]

    key_tile = key_tiles - 1
    stage = stage_base + key_tile
    key_buffer = stage % 3
    value_buffer = stage & 1
    S.nvvm.mbarrier_arrive(key_empty_barrier, key_buffer)

    tile_max_first = S.convert(NEG_INFINITY, S.f32)
    tile_max_second = S.convert(NEG_INFINITY, S.f32)
    for tile in S.range(8, unroll=True):
        tile_max_first = S.nvvm.fast_fmax(
            tile_max_first,
            S.nvvm.fast_fmax(
                S.nvvm.fast_fmax(scores[tile * 8], scores[tile * 8 + 1]),
                S.nvvm.fast_fmax(scores[tile * 8 + 4], scores[tile * 8 + 5]),
            ),
        )
        tile_max_second = S.nvvm.fast_fmax(
            tile_max_second,
            S.nvvm.fast_fmax(
                S.nvvm.fast_fmax(scores[tile * 8 + 2], scores[tile * 8 + 3]),
                S.nvvm.fast_fmax(scores[tile * 8 + 6], scores[tile * 8 + 7]),
            ),
        )
    tile_max_first = _group_max(tile_max_first * softmax_scale_log2)
    tile_max_second = _group_max(tile_max_second * softmax_scale_log2)
    new_max_first = S.nvvm.fast_fmax(max_first, tile_max_first)
    new_max_second = S.nvvm.fast_fmax(max_second, tile_max_second)
    old_scale_first = S.nvvm.fast_exp2(max_first - new_max_first)
    old_scale_second = S.nvvm.fast_exp2(max_second - new_max_second)
    tile_sum_first = S.convert(0.0, S.f32)
    tile_sum_second = S.convert(0.0, S.f32)

    for tile in S.range(8, unroll=True):
        scores[tile * 8 + 0] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(scores[tile * 8], softmax_scale_log2, -new_max_first)
        )
        scores[tile * 8 + 1] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(
                scores[tile * 8 + 1],
                softmax_scale_log2,
                -new_max_first,
            )
        )
        scores[tile * 8 + 4] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(
                scores[tile * 8 + 4],
                softmax_scale_log2,
                -new_max_first,
            )
        )
        scores[tile * 8 + 5] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(
                scores[tile * 8 + 5],
                softmax_scale_log2,
                -new_max_first,
            )
        )
        tile_sum_first = (
            tile_sum_first
            + scores[tile * 8 + 0]
            + scores[tile * 8 + 1]
            + scores[tile * 8 + 4]
            + scores[tile * 8 + 5]
        )
    for tile in S.range(8, unroll=True):
        scores[tile * 8 + 2] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(
                scores[tile * 8 + 2],
                softmax_scale_log2,
                -new_max_second,
            )
        )
        scores[tile * 8 + 3] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(
                scores[tile * 8 + 3],
                softmax_scale_log2,
                -new_max_second,
            )
        )
        scores[tile * 8 + 6] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(
                scores[tile * 8 + 6],
                softmax_scale_log2,
                -new_max_second,
            )
        )
        scores[tile * 8 + 7] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(
                scores[tile * 8 + 7],
                softmax_scale_log2,
                -new_max_second,
            )
        )
        tile_sum_second = (
            tile_sum_second
            + scores[tile * 8 + 2]
            + scores[tile * 8 + 3]
            + scores[tile * 8 + 6]
            + scores[tile * 8 + 7]
        )
    tile_sum_first = _group_sum(tile_sum_first)
    tile_sum_second = _group_sum(tile_sum_second)
    sum_first = S.nvvm.fast_fma(sum_first, old_scale_first, tile_sum_first)
    sum_second = S.nvvm.fast_fma(sum_second, old_scale_second, tile_sum_second)
    max_first = new_max_first
    max_second = new_max_second
    for tile in S.range(8, unroll=True):
        output_accumulator[0, tile * 8] = (
            output_accumulator[0, tile * 8] * old_scale_first
        )
        output_accumulator[0, tile * 8 + 1] = (
            output_accumulator[0, tile * 8 + 1] * old_scale_first
        )
        output_accumulator[0, tile * 8 + 4] = (
            output_accumulator[0, tile * 8 + 4] * old_scale_first
        )
        output_accumulator[0, tile * 8 + 5] = (
            output_accumulator[0, tile * 8 + 5] * old_scale_first
        )
        output_accumulator[0, tile * 8 + 2] = (
            output_accumulator[0, tile * 8 + 2] * old_scale_second
        )
        output_accumulator[0, tile * 8 + 3] = (
            output_accumulator[0, tile * 8 + 3] * old_scale_second
        )
        output_accumulator[0, tile * 8 + 6] = (
            output_accumulator[0, tile * 8 + 6] * old_scale_second
        )
        output_accumulator[0, tile * 8 + 7] = (
            output_accumulator[0, tile * 8 + 7] * old_scale_second
        )

    S.nvvm.mbarrier_try_wait_parity(
        value_barrier, (stage >> 1) & 1, 10000000, value_buffer
    )
    value_matrix = S.subview(
        value_shared,
        (value_buffer, 0),
        (8, 2 * 8 * 64),
        (2 * 8 * 64, 1),
    )
    value_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
        value_matrix, SWIZZLE_128B, 0, 0, 0, 1024, 2048
    )
    S.nvvm.wgmma_fence_aligned()
    last_packed = S.full((1, 4), 0, S.i32)
    last_packed[0, 0] = _pack_bf16(scores[0], scores[1])
    last_packed[0, 1] = _pack_bf16(scores[2], scores[3])
    last_packed[0, 2] = _pack_bf16(scores[4], scores[5])
    last_packed[0, 3] = _pack_bf16(scores[6], scores[7])
    last_packed0 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[8], scores[9])
    last_packed[0, 1] = _pack_bf16(scores[10], scores[11])
    last_packed[0, 2] = _pack_bf16(scores[12], scores[13])
    last_packed[0, 3] = _pack_bf16(scores[14], scores[15])
    last_packed1 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[16], scores[17])
    last_packed[0, 1] = _pack_bf16(scores[18], scores[19])
    last_packed[0, 2] = _pack_bf16(scores[20], scores[21])
    last_packed[0, 3] = _pack_bf16(scores[22], scores[23])
    last_packed2 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[24], scores[25])
    last_packed[0, 1] = _pack_bf16(scores[26], scores[27])
    last_packed[0, 2] = _pack_bf16(scores[28], scores[29])
    last_packed[0, 3] = _pack_bf16(scores[30], scores[31])
    last_packed3 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[32], scores[33])
    last_packed[0, 1] = _pack_bf16(scores[34], scores[35])
    last_packed[0, 2] = _pack_bf16(scores[36], scores[37])
    last_packed[0, 3] = _pack_bf16(scores[38], scores[39])
    last_packed4 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[40], scores[41])
    last_packed[0, 1] = _pack_bf16(scores[42], scores[43])
    last_packed[0, 2] = _pack_bf16(scores[44], scores[45])
    last_packed[0, 3] = _pack_bf16(scores[46], scores[47])
    last_packed5 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[48], scores[49])
    last_packed[0, 1] = _pack_bf16(scores[50], scores[51])
    last_packed[0, 2] = _pack_bf16(scores[52], scores[53])
    last_packed[0, 3] = _pack_bf16(scores[54], scores[55])
    last_packed6 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[56], scores[57])
    last_packed[0, 1] = _pack_bf16(scores[58], scores[59])
    last_packed[0, 2] = _pack_bf16(scores[60], scores[61])
    last_packed[0, 3] = _pack_bf16(scores[62], scores[63])
    last_packed7 = last_packed[0]
    pv_result = S.nvvm.wgmma_m64n128k128_f32_bf16_bf16_rs(
        last_packed0,
        last_packed1,
        last_packed2,
        last_packed3,
        last_packed4,
        last_packed5,
        last_packed6,
        last_packed7,
        value_matrix_desc,
        output_accumulator[0],
    )
    S.nvvm.wgmma_group_sync_aligned()

    S.nvvm.wgmma_wait_group_sync(0)
    for result_element in S.range(64, unroll=True):
        output_accumulator[0, result_element] = pv_result[result_element]
    S.nvvm.mbarrier_arrive(value_empty_barrier, value_buffer)

    inverse_first = S.nvvm.fast_rcp(sum_first)
    inverse_second = S.nvvm.fast_rcp(sum_second)
    for tile in S.range(8, unroll=True):
        row = warp * 16 + (lane & 15)
        column = tile * 16 + (lane >> 4) * 8
        linear = row * 64 + (column & 63)
        if column >= 64:
            linear = linear + CONSUMER_ROWS * 64
        offset = linear ^ ((linear & 0x1C0) >> 3)
        destination = S.subview(query_shared, (consumer, offset), (8, 8), (1, 1))
        packed = S.full((1, 4), 0, S.i32)
        packed[0, 0] = _pack_bf16(
            output_accumulator[0, tile * 8] * inverse_first,
            output_accumulator[0, tile * 8 + 1] * inverse_first,
        )
        packed[0, 1] = _pack_bf16(
            output_accumulator[0, tile * 8 + 2] * inverse_second,
            output_accumulator[0, tile * 8 + 3] * inverse_second,
        )
        packed[0, 2] = _pack_bf16(
            output_accumulator[0, tile * 8 + 4] * inverse_first,
            output_accumulator[0, tile * 8 + 5] * inverse_first,
        )
        packed[0, 3] = _pack_bf16(
            output_accumulator[0, tile * 8 + 6] * inverse_second,
            output_accumulator[0, tile * 8 + 7] * inverse_second,
        )
        S.nvvm.stmatrix_m8n8_x4_b16(destination, packed[0])
    S.nvvm.fence_proxy_async_shared_cta()
    S.nvvm.named_barrier_sync(1 + consumer, WARPGROUP_THREADS)
    if warpgroup_thread == 0:
        output_source_raw = S.subview(
            query_shared,
            (consumer, 0),
            (1, CONSUMER_ROWS * HEAD_DIM),
            (1, 1),
        )
        output_source = S.view(
            output_source_raw,
            S.bf16,
            S.make_layout((2, 64, 64), (4096, 64, 1)),
        )
        S.nvvm.tma_store(
            output_source,
            output_desc,
            (
                0,
                (batch * 4096 + query_start + consumer * 8) * QUERY_HEADS,
                0,
            ),
            predicate=warpgroup_thread == 0,
        )
        S.nvvm.cp_async_bulk_commit_group()
        S.nvvm.cp_async_bulk_wait_group(0, read=True)



@avelang.jit
def _attention_mqa_8192_kernel(
    query: S.Tensor((16, 8192, 8, 128), S.bf16),
    key: S.Tensor((16, 8192, 1, 128), S.bf16),
    value: S.Tensor((16, 8192, 1, 128), S.bf16),
    output: S.Tensor((16, 8192, 8, 128), S.bf16),
    query_length: S.i32,
):
    tid = S.thread_id(0)
    warpgroup = tid // WARPGROUP_THREADS
    warpgroup_thread = tid % WARPGROUP_THREADS
    query_blocks = (
        query_length + PACKED_QUERY_TOKENS - 1
    ) // PACKED_QUERY_TOKENS

    # Ave's module builder emits shared globals in reverse declaration order.
    # Keep the resulting arena identical to CUDA's SharedStorage: query,
    # output, key, value, then barriers. Besides making WGMMA descriptors
    # directly comparable, this keeps every TMA destination sector in range.
    query_barrier = S.nvvm.mbarrier_create()
    key_barrier = S.nvvm.mbarrier_create(2)
    value_barrier = S.nvvm.mbarrier_create(2)
    empty_barrier = S.nvvm.mbarrier_create(2)

    value_shared = S.make_shared((2, 192 * HEAD_DIM), S.bf16, 128)
    key_shared = S.make_shared((2, 192 * HEAD_DIM), S.bf16, 128)
    query_shared = S.make_shared(
        (CONSUMER_WARPGROUPS, CONSUMER_ROWS * HEAD_DIM), S.bf16, 128
    )

    # Reinterpret the contiguous PyTorch tensors exactly like the CUDA tensor
    # maps. TMA coordinates are ordered from the fastest-changing dimension.
    query_map_tensor = S.view(
        query,
        S.bf16,
        S.make_layout(
            (2, BATCH * 8192 * QUERY_HEADS, 64),
            (64, HEAD_DIM, 1),
        ),
    )
    output_map_tensor = S.view(
        output,
        S.bf16,
        S.make_layout(
            (2, BATCH * 8192 * QUERY_HEADS, 64),
            (64, HEAD_DIM, 1),
        ),
    )
    key_map_tensor = S.view(
        key,
        S.bf16,
        S.make_layout(
            (2, BATCH * 8192, 64),
            (64, HEAD_DIM, 1),
        ),
    )
    value_map_tensor = S.view(
        value,
        S.bf16,
        S.make_layout(
            (BATCH * (8192 // 8), 2, 8, 64),
            (8 * HEAD_DIM, 64, HEAD_DIM, 1),
        ),
    )

    query_layout = S.make_layout((2, 64, 64), (4096, 64, 1))
    key_layout = S.make_layout((2, 192, 64), (12288, 64, 1))
    value_layout = S.make_layout((24, 2, 8, 64), (1024, 512, 64, 1))
    query_desc = S.nvvm.make_tma_descriptor(
        query_map_tensor, query_layout, SWIZZLE_128B
    )
    output_desc = S.nvvm.make_tma_descriptor(
        output_map_tensor, query_layout, SWIZZLE_128B
    )
    key_desc = S.nvvm.make_tma_descriptor(key_map_tensor, key_layout, SWIZZLE_128B)
    value_desc = S.nvvm.make_tma_descriptor(
        value_map_tensor, value_layout, SWIZZLE_128B
    )

    S.nvvm.mbarrier_init(query_barrier, 0, count=1, predicate=tid == 0)
    for stage in S.range(2, unroll=True):
        S.nvvm.mbarrier_init(key_barrier, stage, count=1, predicate=tid == 0)
        S.nvvm.mbarrier_init(empty_barrier, stage, count=256, predicate=tid == 0)
    for stage in S.range(2, unroll=True):
        S.nvvm.mbarrier_init(value_barrier, stage, count=1, predicate=tid == 0)
    S.syncthreads()

    if warpgroup == 0:
        S.nvvm.setmaxnreg_dec(24)
        elected = S.nvvm.elect_sync()
        if warpgroup_thread < 32:
            if elected:
                work = S.block_id(0)
                batch = work // query_blocks
                query_start = (
                    work - batch * query_blocks
                ) * PACKED_QUERY_TOKENS
                S.nvvm.mbarrier_arrive_expect_tx(
                    query_barrier,
                    QUERY_BLOCK * HEAD_DIM * 2,
                    0,
                    elected,
                )
                for consumer in S.range(CONSUMER_WARPGROUPS, unroll=True):
                    query_destination_raw = S.subview(
                        query_shared,
                        (consumer, 0),
                        (1, CONSUMER_ROWS * HEAD_DIM),
                        (1, 1),
                    )
                    query_destination = S.view(
                        query_destination_raw,
                        S.bf16,
                        S.make_layout((2, 64, 64), (4096, 64, 1)),
                    )
                    S.nvvm.tma_load(
                        query_destination,
                        query_desc,
                        (
                            0,
                            (batch * 8192 + query_start + consumer * 8)
                            * QUERY_HEADS,
                            0,
                        ),
                        query_barrier,
                        mbar_id=0,
                        predicate=elected,
                        expect_tx=False,
                    )

                for key_tile in S.range(43):
                    stage = key_tile
                    key_buffer = stage & 1
                    value_buffer = stage & 1
                    if stage >= 2:
                        S.nvvm.mbarrier_try_wait_parity(
                            empty_barrier,
                            ((stage >> 1) - 1) & 1,
                            10000000,
                            key_buffer,
                        )
                    key_destination_raw = S.subview(
                        key_shared,
                        (key_buffer, 0),
                        (1, 192 * HEAD_DIM),
                        (1, 1),
                    )
                    key_destination = S.view(
                        key_destination_raw,
                        S.bf16,
                        S.make_layout((2, 192, 64), (12288, 64, 1)),
                    )
                    S.nvvm.tma_load(
                        key_destination,
                        key_desc,
                        (0, batch * 8192 + key_tile * 192, 0),
                        key_barrier,
                        mbar_id=key_buffer,
                        predicate=elected,
                    )
                    value_destination_raw = S.subview(
                        value_shared,
                        (value_buffer, 0),
                        (1, 192 * HEAD_DIM),
                        (1, 1),
                    )
                    value_destination = S.view(
                        value_destination_raw,
                        S.bf16,
                        S.make_layout((24, 2, 8, 64), (1024, 512, 64, 1)),
                    )
                    S.nvvm.tma_load(
                        value_destination,
                        value_desc,
                        (
                            0,
                            0,
                            0,
                            batch * (8192 // 8) + key_tile * (192 // 8),
                        ),
                        value_barrier,
                        mbar_id=value_buffer,
                        predicate=elected,
                    )
        return

    S.nvvm.setmaxnreg_inc(240)
    consumer = warpgroup - 1
    lane = warpgroup_thread & 31
    warp = warpgroup_thread >> 5
    work = S.block_id(0)
    batch = work // query_blocks
    query_start = (
        work - batch * query_blocks
    ) * PACKED_QUERY_TOKENS
    S.nvvm.mbarrier_try_wait_parity(query_barrier, 0, 10000000, 0)
    output_accumulator = S.full((1, 64), 0.0, S.f32)
    max_first = S.convert(NEG_INFINITY, S.f32)
    max_second = S.convert(NEG_INFINITY, S.f32)
    sum_first = S.convert(0.0, S.f32)
    sum_second = S.convert(0.0, S.f32)
    softmax_scale_log2 = S.convert(SOFTMAX_SCALE_LOG2, S.f32)
    # Use a distinct initializer so the mutable score and output buffers
    # remain separate through common-subexpression elimination.
    scores = S.full((96,), NEG_INFINITY, S.f32)

    stage_base = 0
    first_key_buffer = stage_base & 1
    S.nvvm.mbarrier_try_wait_parity(
        key_barrier, (stage_base >> 1) & 1, 10000000, first_key_buffer
    )

    query_matrix = S.subview(
        query_shared,
        (consumer, 0),
        (CONSUMER_ROWS, HEAD_DIM),
        (HEAD_DIM, 1),
    )
    key_matrix = S.subview(
        key_shared,
        (first_key_buffer, 0),
        (192, HEAD_DIM),
        (HEAD_DIM, 1),
    )
    query_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
        query_matrix, SWIZZLE_128B, 0, 0, 0, 16, 1024
    )
    key_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
        key_matrix, SWIZZLE_128B, 0, 0, 0, 16, 1024
    )
    S.nvvm.wgmma_fence_aligned()
    qk_result = S.nvvm.wgmma_m64n192k128_f32_bf16_bf16_ss(
        query_matrix_desc, key_matrix_desc
    )
    S.nvvm.wgmma_group_sync_aligned()
    S.nvvm.wgmma_wait_group_sync(0)
    for result_element in S.range(96, unroll=True):
        scores[result_element] = qk_result[result_element]
    for key_tile in S.range(42):
        stage = stage_base + key_tile
        key_buffer = stage & 1
        value_buffer = stage & 1
        tile_max_first = S.convert(NEG_INFINITY, S.f32)
        tile_max_second = S.convert(NEG_INFINITY, S.f32)
        for tile in S.range(12, unroll=True):
            tile_max_first = S.nvvm.fast_fmax(
                tile_max_first,
                S.nvvm.fast_fmax(
                    S.nvvm.fast_fmax(scores[tile * 8], scores[tile * 8 + 1]),
                    S.nvvm.fast_fmax(scores[tile * 8 + 4], scores[tile * 8 + 5]),
                ),
            )
            tile_max_second = S.nvvm.fast_fmax(
                tile_max_second,
                S.nvvm.fast_fmax(
                    S.nvvm.fast_fmax(scores[tile * 8 + 2], scores[tile * 8 + 3]),
                    S.nvvm.fast_fmax(scores[tile * 8 + 6], scores[tile * 8 + 7]),
                ),
            )
        tile_max_first = _group_max(tile_max_first * softmax_scale_log2)
        tile_max_second = _group_max(tile_max_second * softmax_scale_log2)
        new_max_first = S.nvvm.fast_fmax(max_first, tile_max_first)
        new_max_second = S.nvvm.fast_fmax(max_second, tile_max_second)
        old_scale_first = S.nvvm.fast_exp2(max_first - new_max_first)
        old_scale_second = S.nvvm.fast_exp2(max_second - new_max_second)
        tile_sum_first = S.convert(0.0, S.f32)
        tile_sum_second = S.convert(0.0, S.f32)

        for tile in S.range(12, unroll=True):
            scores[tile * 8 + 0] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(scores[tile * 8], softmax_scale_log2, -new_max_first)
            )
            scores[tile * 8 + 1] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[tile * 8 + 1],
                    softmax_scale_log2,
                    -new_max_first,
                )
            )
            scores[tile * 8 + 4] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[tile * 8 + 4],
                    softmax_scale_log2,
                    -new_max_first,
                )
            )
            scores[tile * 8 + 5] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[tile * 8 + 5],
                    softmax_scale_log2,
                    -new_max_first,
                )
            )
            tile_sum_first = (
                tile_sum_first
                + scores[tile * 8 + 0]
                + scores[tile * 8 + 1]
                + scores[tile * 8 + 4]
                + scores[tile * 8 + 5]
            )
        for tile in S.range(12, unroll=True):
            scores[tile * 8 + 2] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[tile * 8 + 2],
                    softmax_scale_log2,
                    -new_max_second,
                )
            )
            scores[tile * 8 + 3] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[tile * 8 + 3],
                    softmax_scale_log2,
                    -new_max_second,
                )
            )
            scores[tile * 8 + 6] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[tile * 8 + 6],
                    softmax_scale_log2,
                    -new_max_second,
                )
            )
            scores[tile * 8 + 7] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[tile * 8 + 7],
                    softmax_scale_log2,
                    -new_max_second,
                )
            )
            tile_sum_second = (
                tile_sum_second
                + scores[tile * 8 + 2]
                + scores[tile * 8 + 3]
                + scores[tile * 8 + 6]
                + scores[tile * 8 + 7]
            )
        tile_sum_first = _group_sum(tile_sum_first)
        tile_sum_second = _group_sum(tile_sum_second)
        sum_first = S.nvvm.fast_fma(sum_first, old_scale_first, tile_sum_first)
        sum_second = S.nvvm.fast_fma(sum_second, old_scale_second, tile_sum_second)
        max_first = new_max_first
        max_second = new_max_second
        for tile in S.range(8, unroll=True):
            output_accumulator[0, tile * 8] = (
                output_accumulator[0, tile * 8] * old_scale_first
            )
            output_accumulator[0, tile * 8 + 1] = (
                output_accumulator[0, tile * 8 + 1] * old_scale_first
            )
            output_accumulator[0, tile * 8 + 4] = (
                output_accumulator[0, tile * 8 + 4] * old_scale_first
            )
            output_accumulator[0, tile * 8 + 5] = (
                output_accumulator[0, tile * 8 + 5] * old_scale_first
            )
            output_accumulator[0, tile * 8 + 2] = (
                output_accumulator[0, tile * 8 + 2] * old_scale_second
            )
            output_accumulator[0, tile * 8 + 3] = (
                output_accumulator[0, tile * 8 + 3] * old_scale_second
            )
            output_accumulator[0, tile * 8 + 6] = (
                output_accumulator[0, tile * 8 + 6] * old_scale_second
            )
            output_accumulator[0, tile * 8 + 7] = (
                output_accumulator[0, tile * 8 + 7] * old_scale_second
            )

        S.nvvm.mbarrier_try_wait_parity(
            value_barrier, (stage >> 1) & 1, 10000000, value_buffer
        )
        value_matrix = S.subview(
            value_shared,
            (value_buffer, 0),
            (12, 2 * 8 * 64),
            (2 * 8 * 64, 1),
        )
        value_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
            value_matrix, SWIZZLE_128B, 0, 0, 0, 1024, 2048
        )
        S.nvvm.wgmma_fence_aligned()
        packed = S.full((1, 4), 0, S.i32)
        packed[0, 0] = _pack_bf16(scores[0], scores[1])
        packed[0, 1] = _pack_bf16(scores[2], scores[3])
        packed[0, 2] = _pack_bf16(scores[4], scores[5])
        packed[0, 3] = _pack_bf16(scores[6], scores[7])
        packed0 = packed[0]
        packed[0, 0] = _pack_bf16(scores[8], scores[9])
        packed[0, 1] = _pack_bf16(scores[10], scores[11])
        packed[0, 2] = _pack_bf16(scores[12], scores[13])
        packed[0, 3] = _pack_bf16(scores[14], scores[15])
        packed1 = packed[0]
        packed[0, 0] = _pack_bf16(scores[16], scores[17])
        packed[0, 1] = _pack_bf16(scores[18], scores[19])
        packed[0, 2] = _pack_bf16(scores[20], scores[21])
        packed[0, 3] = _pack_bf16(scores[22], scores[23])
        packed2 = packed[0]
        packed[0, 0] = _pack_bf16(scores[24], scores[25])
        packed[0, 1] = _pack_bf16(scores[26], scores[27])
        packed[0, 2] = _pack_bf16(scores[28], scores[29])
        packed[0, 3] = _pack_bf16(scores[30], scores[31])
        packed3 = packed[0]
        packed[0, 0] = _pack_bf16(scores[32], scores[33])
        packed[0, 1] = _pack_bf16(scores[34], scores[35])
        packed[0, 2] = _pack_bf16(scores[36], scores[37])
        packed[0, 3] = _pack_bf16(scores[38], scores[39])
        packed4 = packed[0]
        packed[0, 0] = _pack_bf16(scores[40], scores[41])
        packed[0, 1] = _pack_bf16(scores[42], scores[43])
        packed[0, 2] = _pack_bf16(scores[44], scores[45])
        packed[0, 3] = _pack_bf16(scores[46], scores[47])
        packed5 = packed[0]
        packed[0, 0] = _pack_bf16(scores[48], scores[49])
        packed[0, 1] = _pack_bf16(scores[50], scores[51])
        packed[0, 2] = _pack_bf16(scores[52], scores[53])
        packed[0, 3] = _pack_bf16(scores[54], scores[55])
        packed6 = packed[0]
        packed[0, 0] = _pack_bf16(scores[56], scores[57])
        packed[0, 1] = _pack_bf16(scores[58], scores[59])
        packed[0, 2] = _pack_bf16(scores[60], scores[61])
        packed[0, 3] = _pack_bf16(scores[62], scores[63])
        packed7 = packed[0]
        packed[0, 0] = _pack_bf16(scores[64], scores[65])
        packed[0, 1] = _pack_bf16(scores[66], scores[67])
        packed[0, 2] = _pack_bf16(scores[68], scores[69])
        packed[0, 3] = _pack_bf16(scores[70], scores[71])
        packed8 = packed[0]
        packed[0, 0] = _pack_bf16(scores[72], scores[73])
        packed[0, 1] = _pack_bf16(scores[74], scores[75])
        packed[0, 2] = _pack_bf16(scores[76], scores[77])
        packed[0, 3] = _pack_bf16(scores[78], scores[79])
        packed9 = packed[0]
        packed[0, 0] = _pack_bf16(scores[80], scores[81])
        packed[0, 1] = _pack_bf16(scores[82], scores[83])
        packed[0, 2] = _pack_bf16(scores[84], scores[85])
        packed[0, 3] = _pack_bf16(scores[86], scores[87])
        packed10 = packed[0]
        packed[0, 0] = _pack_bf16(scores[88], scores[89])
        packed[0, 1] = _pack_bf16(scores[90], scores[91])
        packed[0, 2] = _pack_bf16(scores[92], scores[93])
        packed[0, 3] = _pack_bf16(scores[94], scores[95])
        packed11 = packed[0]
        pv_result = S.nvvm.wgmma_m64n128k192_f32_bf16_bf16_rs(
            packed0,
            packed1,
            packed2,
            packed3,
            packed4,
            packed5,
            packed6,
            packed7,
            packed8,
            packed9,
            packed10,
            packed11,
            value_matrix_desc,
            output_accumulator[0],
        )
        S.nvvm.wgmma_group_sync_aligned()

        next_stage = stage + 1
        next_key_buffer = next_stage & 1
        S.nvvm.mbarrier_try_wait_parity(
            key_barrier,
            (next_stage >> 1) & 1,
            10000000,
            next_key_buffer,
        )
        next_key_matrix = S.subview(
            key_shared,
            (next_key_buffer, 0),
            (192, HEAD_DIM),
            (HEAD_DIM, 1),
        )
        key_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
            next_key_matrix, SWIZZLE_128B, 0, 0, 0, 16, 1024
        )
        query_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
            query_matrix, SWIZZLE_128B, 0, 0, 0, 16, 1024
        )
        S.nvvm.wgmma_fence_aligned()
        next_scores = S.nvvm.wgmma_m64n192k128_f32_bf16_bf16_ss(
            query_matrix_desc, key_matrix_desc
        )
        S.nvvm.wgmma_group_sync_aligned()
        S.nvvm.wgmma_wait_group_sync(1)
        for result_element in S.range(64, unroll=True):
            output_accumulator[0, result_element] = pv_result[result_element]
        S.nvvm.mbarrier_arrive(empty_barrier, value_buffer)
        S.nvvm.wgmma_wait_group_sync(0)
        for result_element in S.range(96, unroll=True):
            scores[result_element] = next_scores[result_element]

    key_tile = 42
    stage = stage_base + key_tile
    key_buffer = stage & 1
    value_buffer = stage & 1
    for tile in S.range(8, 12, unroll=True):
        for element in S.range(8, unroll=True):
            scores[tile * 8 + element] = S.convert(NEG_INFINITY, S.f32)

    tile_max_first = S.convert(NEG_INFINITY, S.f32)
    tile_max_second = S.convert(NEG_INFINITY, S.f32)
    for tile in S.range(12, unroll=True):
        tile_max_first = S.nvvm.fast_fmax(
            tile_max_first,
            S.nvvm.fast_fmax(
                S.nvvm.fast_fmax(scores[tile * 8], scores[tile * 8 + 1]),
                S.nvvm.fast_fmax(scores[tile * 8 + 4], scores[tile * 8 + 5]),
            ),
        )
        tile_max_second = S.nvvm.fast_fmax(
            tile_max_second,
            S.nvvm.fast_fmax(
                S.nvvm.fast_fmax(scores[tile * 8 + 2], scores[tile * 8 + 3]),
                S.nvvm.fast_fmax(scores[tile * 8 + 6], scores[tile * 8 + 7]),
            ),
        )
    tile_max_first = _group_max(tile_max_first * softmax_scale_log2)
    tile_max_second = _group_max(tile_max_second * softmax_scale_log2)
    new_max_first = S.nvvm.fast_fmax(max_first, tile_max_first)
    new_max_second = S.nvvm.fast_fmax(max_second, tile_max_second)
    old_scale_first = S.nvvm.fast_exp2(max_first - new_max_first)
    old_scale_second = S.nvvm.fast_exp2(max_second - new_max_second)
    tile_sum_first = S.convert(0.0, S.f32)
    tile_sum_second = S.convert(0.0, S.f32)

    for tile in S.range(12, unroll=True):
        scores[tile * 8 + 0] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(scores[tile * 8], softmax_scale_log2, -new_max_first)
        )
        scores[tile * 8 + 1] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(
                scores[tile * 8 + 1],
                softmax_scale_log2,
                -new_max_first,
            )
        )
        scores[tile * 8 + 4] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(
                scores[tile * 8 + 4],
                softmax_scale_log2,
                -new_max_first,
            )
        )
        scores[tile * 8 + 5] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(
                scores[tile * 8 + 5],
                softmax_scale_log2,
                -new_max_first,
            )
        )
        tile_sum_first = (
            tile_sum_first
            + scores[tile * 8 + 0]
            + scores[tile * 8 + 1]
            + scores[tile * 8 + 4]
            + scores[tile * 8 + 5]
        )
    for tile in S.range(12, unroll=True):
        scores[tile * 8 + 2] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(
                scores[tile * 8 + 2],
                softmax_scale_log2,
                -new_max_second,
            )
        )
        scores[tile * 8 + 3] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(
                scores[tile * 8 + 3],
                softmax_scale_log2,
                -new_max_second,
            )
        )
        scores[tile * 8 + 6] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(
                scores[tile * 8 + 6],
                softmax_scale_log2,
                -new_max_second,
            )
        )
        scores[tile * 8 + 7] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(
                scores[tile * 8 + 7],
                softmax_scale_log2,
                -new_max_second,
            )
        )
        tile_sum_second = (
            tile_sum_second
            + scores[tile * 8 + 2]
            + scores[tile * 8 + 3]
            + scores[tile * 8 + 6]
            + scores[tile * 8 + 7]
        )
    tile_sum_first = _group_sum(tile_sum_first)
    tile_sum_second = _group_sum(tile_sum_second)
    sum_first = S.nvvm.fast_fma(sum_first, old_scale_first, tile_sum_first)
    sum_second = S.nvvm.fast_fma(sum_second, old_scale_second, tile_sum_second)
    max_first = new_max_first
    max_second = new_max_second
    for tile in S.range(8, unroll=True):
        output_accumulator[0, tile * 8] = (
            output_accumulator[0, tile * 8] * old_scale_first
        )
        output_accumulator[0, tile * 8 + 1] = (
            output_accumulator[0, tile * 8 + 1] * old_scale_first
        )
        output_accumulator[0, tile * 8 + 4] = (
            output_accumulator[0, tile * 8 + 4] * old_scale_first
        )
        output_accumulator[0, tile * 8 + 5] = (
            output_accumulator[0, tile * 8 + 5] * old_scale_first
        )
        output_accumulator[0, tile * 8 + 2] = (
            output_accumulator[0, tile * 8 + 2] * old_scale_second
        )
        output_accumulator[0, tile * 8 + 3] = (
            output_accumulator[0, tile * 8 + 3] * old_scale_second
        )
        output_accumulator[0, tile * 8 + 6] = (
            output_accumulator[0, tile * 8 + 6] * old_scale_second
        )
        output_accumulator[0, tile * 8 + 7] = (
            output_accumulator[0, tile * 8 + 7] * old_scale_second
        )

    S.nvvm.mbarrier_try_wait_parity(
        value_barrier, (stage >> 1) & 1, 10000000, value_buffer
    )
    value_matrix = S.subview(
        value_shared,
        (value_buffer, 0),
        (12, 2 * 8 * 64),
        (2 * 8 * 64, 1),
    )
    value_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
        value_matrix, SWIZZLE_128B, 0, 0, 0, 1024, 2048
    )
    S.nvvm.wgmma_fence_aligned()
    last_packed = S.full((1, 4), 0, S.i32)
    last_packed[0, 0] = _pack_bf16(scores[0], scores[1])
    last_packed[0, 1] = _pack_bf16(scores[2], scores[3])
    last_packed[0, 2] = _pack_bf16(scores[4], scores[5])
    last_packed[0, 3] = _pack_bf16(scores[6], scores[7])
    last_packed0 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[8], scores[9])
    last_packed[0, 1] = _pack_bf16(scores[10], scores[11])
    last_packed[0, 2] = _pack_bf16(scores[12], scores[13])
    last_packed[0, 3] = _pack_bf16(scores[14], scores[15])
    last_packed1 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[16], scores[17])
    last_packed[0, 1] = _pack_bf16(scores[18], scores[19])
    last_packed[0, 2] = _pack_bf16(scores[20], scores[21])
    last_packed[0, 3] = _pack_bf16(scores[22], scores[23])
    last_packed2 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[24], scores[25])
    last_packed[0, 1] = _pack_bf16(scores[26], scores[27])
    last_packed[0, 2] = _pack_bf16(scores[28], scores[29])
    last_packed[0, 3] = _pack_bf16(scores[30], scores[31])
    last_packed3 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[32], scores[33])
    last_packed[0, 1] = _pack_bf16(scores[34], scores[35])
    last_packed[0, 2] = _pack_bf16(scores[36], scores[37])
    last_packed[0, 3] = _pack_bf16(scores[38], scores[39])
    last_packed4 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[40], scores[41])
    last_packed[0, 1] = _pack_bf16(scores[42], scores[43])
    last_packed[0, 2] = _pack_bf16(scores[44], scores[45])
    last_packed[0, 3] = _pack_bf16(scores[46], scores[47])
    last_packed5 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[48], scores[49])
    last_packed[0, 1] = _pack_bf16(scores[50], scores[51])
    last_packed[0, 2] = _pack_bf16(scores[52], scores[53])
    last_packed[0, 3] = _pack_bf16(scores[54], scores[55])
    last_packed6 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[56], scores[57])
    last_packed[0, 1] = _pack_bf16(scores[58], scores[59])
    last_packed[0, 2] = _pack_bf16(scores[60], scores[61])
    last_packed[0, 3] = _pack_bf16(scores[62], scores[63])
    last_packed7 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[64], scores[65])
    last_packed[0, 1] = _pack_bf16(scores[66], scores[67])
    last_packed[0, 2] = _pack_bf16(scores[68], scores[69])
    last_packed[0, 3] = _pack_bf16(scores[70], scores[71])
    last_packed8 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[72], scores[73])
    last_packed[0, 1] = _pack_bf16(scores[74], scores[75])
    last_packed[0, 2] = _pack_bf16(scores[76], scores[77])
    last_packed[0, 3] = _pack_bf16(scores[78], scores[79])
    last_packed9 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[80], scores[81])
    last_packed[0, 1] = _pack_bf16(scores[82], scores[83])
    last_packed[0, 2] = _pack_bf16(scores[84], scores[85])
    last_packed[0, 3] = _pack_bf16(scores[86], scores[87])
    last_packed10 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[88], scores[89])
    last_packed[0, 1] = _pack_bf16(scores[90], scores[91])
    last_packed[0, 2] = _pack_bf16(scores[92], scores[93])
    last_packed[0, 3] = _pack_bf16(scores[94], scores[95])
    last_packed11 = last_packed[0]
    pv_result = S.nvvm.wgmma_m64n128k192_f32_bf16_bf16_rs(
        last_packed0,
        last_packed1,
        last_packed2,
        last_packed3,
        last_packed4,
        last_packed5,
        last_packed6,
        last_packed7,
        last_packed8,
        last_packed9,
        last_packed10,
        last_packed11,
        value_matrix_desc,
        output_accumulator[0],
    )
    S.nvvm.wgmma_group_sync_aligned()

    S.nvvm.wgmma_wait_group_sync(0)
    for result_element in S.range(64, unroll=True):
        output_accumulator[0, result_element] = pv_result[result_element]
    S.nvvm.mbarrier_arrive(empty_barrier, value_buffer)

    inverse_first = S.nvvm.fast_rcp(sum_first)
    inverse_second = S.nvvm.fast_rcp(sum_second)
    for tile in S.range(8, unroll=True):
        row = warp * 16 + (lane & 15)
        column = tile * 16 + (lane >> 4) * 8
        linear = row * 64 + (column & 63)
        if column >= 64:
            linear = linear + CONSUMER_ROWS * 64
        offset = linear ^ ((linear & 0x1C0) >> 3)
        destination = S.subview(query_shared, (consumer, offset), (8, 8), (1, 1))
        packed = S.full((1, 4), 0, S.i32)
        packed[0, 0] = _pack_bf16(
            output_accumulator[0, tile * 8] * inverse_first,
            output_accumulator[0, tile * 8 + 1] * inverse_first,
        )
        packed[0, 1] = _pack_bf16(
            output_accumulator[0, tile * 8 + 2] * inverse_second,
            output_accumulator[0, tile * 8 + 3] * inverse_second,
        )
        packed[0, 2] = _pack_bf16(
            output_accumulator[0, tile * 8 + 4] * inverse_first,
            output_accumulator[0, tile * 8 + 5] * inverse_first,
        )
        packed[0, 3] = _pack_bf16(
            output_accumulator[0, tile * 8 + 6] * inverse_second,
            output_accumulator[0, tile * 8 + 7] * inverse_second,
        )
        S.nvvm.stmatrix_m8n8_x4_b16(destination, packed[0])
    S.nvvm.fence_proxy_async_shared_cta()
    S.nvvm.named_barrier_sync(1 + consumer, WARPGROUP_THREADS)
    if warpgroup_thread == 0:
        output_source_raw = S.subview(
            query_shared,
            (consumer, 0),
            (1, CONSUMER_ROWS * HEAD_DIM),
            (1, 1),
        )
        output_source = S.view(
            output_source_raw,
            S.bf16,
            S.make_layout((2, 64, 64), (4096, 64, 1)),
        )
        S.nvvm.tma_store(
            output_source,
            output_desc,
            (
                0,
                (batch * 8192 + query_start + consumer * 8) * QUERY_HEADS,
                0,
            ),
            predicate=warpgroup_thread == 0,
        )
        S.nvvm.cp_async_bulk_commit_group()
        S.nvvm.cp_async_bulk_wait_group(0, read=True)



@avelang.jit
def _attention_mqa_16384_kernel(
    query: S.Tensor((16, 16384, 8, 128), S.bf16),
    key: S.Tensor((16, 16384, 1, 128), S.bf16),
    value: S.Tensor((16, 16384, 1, 128), S.bf16),
    output: S.Tensor((16, 16384, 8, 128), S.bf16),
):
    tid = S.thread_id(0)
    warpgroup = tid // WARPGROUP_THREADS
    warpgroup_thread = tid % WARPGROUP_THREADS

    # Ave's module builder emits shared globals in reverse declaration order.
    # Keep the resulting arena identical to CUDA's SharedStorage: query,
    # output, key, value, then barriers. Besides making WGMMA descriptors
    # directly comparable, this keeps every TMA destination sector in range.
    query_barrier = S.nvvm.mbarrier_create()
    key_barrier = S.nvvm.mbarrier_create(2)
    value_barrier = S.nvvm.mbarrier_create(2)
    empty_barrier = S.nvvm.mbarrier_create(2)

    value_shared = S.make_shared((2, 176 * HEAD_DIM), S.bf16, 128)
    key_shared = S.make_shared((2, 176 * HEAD_DIM), S.bf16, 128)
    query_shared = S.make_shared(
        (CONSUMER_WARPGROUPS, CONSUMER_ROWS * HEAD_DIM), S.bf16, 128
    )

    query_map_tensor = S.view(
        query,
        S.bf16,
        S.make_layout(
            (2, BATCH * 16384, QUERY_HEADS, 64),
            (64, QUERY_HEADS * HEAD_DIM, HEAD_DIM, 1),
        ),
    )
    output_map_tensor = S.view(
        output,
        S.bf16,
        S.make_layout(
            (2, BATCH * 16384, QUERY_HEADS, 64),
            (64, QUERY_HEADS * HEAD_DIM, HEAD_DIM, 1),
        ),
    )
    key_map_tensor = S.view(
        key,
        S.bf16,
        S.make_layout(
            (2, BATCH * 16384, 64),
            (64, HEAD_DIM, 1),
        ),
    )
    value_map_tensor = S.view(
        value,
        S.bf16,
        S.make_layout(
            (BATCH * (16384 // 8), 2, 8, 64),
            (8 * HEAD_DIM, 64, HEAD_DIM, 1),
        ),
    )

    query_layout = S.make_layout((2, 8, 8, 64), (4096, 512, 64, 1))
    key_layout = S.make_layout((2, 176, 64), (11264, 64, 1))
    value_layout = S.make_layout((22, 2, 8, 64), (1024, 512, 64, 1))
    query_desc = S.nvvm.make_tma_descriptor(
        query_map_tensor, query_layout, SWIZZLE_128B
    )
    output_desc = S.nvvm.make_tma_descriptor(
        output_map_tensor, query_layout, SWIZZLE_128B
    )
    key_desc = S.nvvm.make_tma_descriptor(key_map_tensor, key_layout, SWIZZLE_128B)
    value_desc = S.nvvm.make_tma_descriptor(
        value_map_tensor, value_layout, SWIZZLE_128B
    )

    S.nvvm.mbarrier_init(query_barrier, 0, count=1, predicate=tid == 0)
    for stage in S.range(2, unroll=True):
        S.nvvm.mbarrier_init(key_barrier, stage, count=1, predicate=tid == 0)
        S.nvvm.mbarrier_init(empty_barrier, stage, count=256, predicate=tid == 0)
    for stage in S.range(2, unroll=True):
        S.nvvm.mbarrier_init(value_barrier, stage, count=1, predicate=tid == 0)
    S.syncthreads()

    if warpgroup == 0:
        S.nvvm.setmaxnreg_dec(24)
        elected = S.nvvm.elect_sync()
        if warpgroup_thread < 32:
            if elected:
                work = S.block_id(0)
                batch = work // (16384 // PACKED_QUERY_TOKENS)
                query_start = (
                    work - batch * (16384 // PACKED_QUERY_TOKENS)
                ) * PACKED_QUERY_TOKENS
                S.nvvm.mbarrier_arrive_expect_tx(
                    query_barrier,
                    QUERY_BLOCK * HEAD_DIM * 2,
                    0,
                    elected,
                )
                for consumer in S.range(CONSUMER_WARPGROUPS, unroll=True):
                    query_destination_raw = S.subview(
                        query_shared,
                        (consumer, 0),
                        (1, CONSUMER_ROWS * HEAD_DIM),
                        (1, 1),
                    )
                    query_destination = S.view(
                        query_destination_raw,
                        S.bf16,
                        S.make_layout((2, 8, 8, 64), (4096, 512, 64, 1)),
                    )
                    S.nvvm.tma_load(
                        query_destination,
                        query_desc,
                        (
                            0,
                            0,
                            batch * 16384 + query_start + consumer * 8,
                            0,
                        ),
                        query_barrier,
                        mbar_id=0,
                        predicate=elected,
                        expect_tx=False,
                    )

                for key_tile in S.range(94):
                    stage = key_tile
                    key_buffer = stage & 1
                    value_buffer = stage & 1
                    if stage >= 2:
                        S.nvvm.mbarrier_try_wait_parity(
                            empty_barrier,
                            ((stage >> 1) - 1) & 1,
                            10000000,
                            key_buffer,
                        )
                    key_destination_raw = S.subview(
                        key_shared,
                        (key_buffer, 0),
                        (1, 176 * HEAD_DIM),
                        (1, 1),
                    )
                    key_destination = S.view(
                        key_destination_raw,
                        S.bf16,
                        S.make_layout((2, 176, 64), (11264, 64, 1)),
                    )
                    S.nvvm.tma_load(
                        key_destination,
                        key_desc,
                        (0, batch * 16384 + key_tile * 176, 0),
                        key_barrier,
                        mbar_id=key_buffer,
                        predicate=elected,
                    )
                    value_destination_raw = S.subview(
                        value_shared,
                        (value_buffer, 0),
                        (1, 176 * HEAD_DIM),
                        (1, 1),
                    )
                    value_destination = S.view(
                        value_destination_raw,
                        S.bf16,
                        S.make_layout((22, 2, 8, 64), (1024, 512, 64, 1)),
                    )
                    S.nvvm.tma_load(
                        value_destination,
                        value_desc,
                        (
                            0,
                            0,
                            0,
                            batch * (16384 // 8) + key_tile * (176 // 8),
                        ),
                        value_barrier,
                        mbar_id=value_buffer,
                        predicate=elected,
                    )
        return

    S.nvvm.setmaxnreg_inc(240)
    consumer = warpgroup - 1
    lane = warpgroup_thread & 31
    warp = warpgroup_thread >> 5
    work = S.block_id(0)
    batch = work // (16384 // PACKED_QUERY_TOKENS)
    query_start = (
        work - batch * (16384 // PACKED_QUERY_TOKENS)
    ) * PACKED_QUERY_TOKENS
    S.nvvm.mbarrier_try_wait_parity(query_barrier, 0, 10000000, 0)
    output_accumulator = S.full((1, 64), 0.0, S.f32)
    max_first = S.convert(NEG_INFINITY, S.f32)
    max_second = S.convert(NEG_INFINITY, S.f32)
    sum_first = S.convert(0.0, S.f32)
    sum_second = S.convert(0.0, S.f32)
    softmax_scale_log2 = S.convert(SOFTMAX_SCALE_LOG2, S.f32)
    # Use a distinct initializer so the mutable score and output buffers
    # remain separate through common-subexpression elimination.
    scores = S.full((88,), NEG_INFINITY, S.f32)

    stage_base = 0
    first_key_buffer = stage_base & 1
    S.nvvm.mbarrier_try_wait_parity(
        key_barrier, (stage_base >> 1) & 1, 10000000, first_key_buffer
    )

    query_matrix = S.subview(
        query_shared,
        (consumer, 0),
        (CONSUMER_ROWS, HEAD_DIM),
        (HEAD_DIM, 1),
    )
    key_matrix = S.subview(
        key_shared,
        (first_key_buffer, 0),
        (176, HEAD_DIM),
        (HEAD_DIM, 1),
    )
    query_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
        query_matrix, SWIZZLE_128B, 0, 0, 0, 16, 1024
    )
    key_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
        key_matrix, SWIZZLE_128B, 0, 0, 0, 16, 1024
    )
    S.nvvm.wgmma_fence_aligned()
    qk_result = S.nvvm.wgmma_m64n176k128_f32_bf16_bf16_ss(
        query_matrix_desc, key_matrix_desc
    )
    S.nvvm.wgmma_group_sync_aligned()
    S.nvvm.wgmma_wait_group_sync(0)
    for result_element in S.range(88, unroll=True):
        scores[result_element] = qk_result[result_element]
    for key_tile in S.range(93):
        stage = stage_base + key_tile
        key_buffer = stage & 1
        value_buffer = stage & 1
        tile_max_first = S.convert(NEG_INFINITY, S.f32)
        tile_max_second = S.convert(NEG_INFINITY, S.f32)
        for tile in S.range(11, unroll=True):
            tile_max_first = S.nvvm.fast_fmax(
                tile_max_first,
                S.nvvm.fast_fmax(
                    S.nvvm.fast_fmax(scores[tile * 8], scores[tile * 8 + 1]),
                    S.nvvm.fast_fmax(scores[tile * 8 + 4], scores[tile * 8 + 5]),
                ),
            )
            tile_max_second = S.nvvm.fast_fmax(
                tile_max_second,
                S.nvvm.fast_fmax(
                    S.nvvm.fast_fmax(scores[tile * 8 + 2], scores[tile * 8 + 3]),
                    S.nvvm.fast_fmax(scores[tile * 8 + 6], scores[tile * 8 + 7]),
                ),
            )
        tile_max_first = _group_max(tile_max_first * softmax_scale_log2)
        tile_max_second = _group_max(tile_max_second * softmax_scale_log2)
        new_max_first = S.nvvm.fast_fmax(max_first, tile_max_first)
        new_max_second = S.nvvm.fast_fmax(max_second, tile_max_second)
        old_scale_first = S.nvvm.fast_exp2(max_first - new_max_first)
        old_scale_second = S.nvvm.fast_exp2(max_second - new_max_second)
        tile_sum_first = S.convert(0.0, S.f32)
        tile_sum_second = S.convert(0.0, S.f32)

        for tile in S.range(11, unroll=True):
            scores[tile * 8 + 0] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(scores[tile * 8], softmax_scale_log2, -new_max_first)
            )
            scores[tile * 8 + 1] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[tile * 8 + 1],
                    softmax_scale_log2,
                    -new_max_first,
                )
            )
            scores[tile * 8 + 4] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[tile * 8 + 4],
                    softmax_scale_log2,
                    -new_max_first,
                )
            )
            scores[tile * 8 + 5] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[tile * 8 + 5],
                    softmax_scale_log2,
                    -new_max_first,
                )
            )
            tile_sum_first = (
                tile_sum_first
                + scores[tile * 8 + 0]
                + scores[tile * 8 + 1]
                + scores[tile * 8 + 4]
                + scores[tile * 8 + 5]
            )
        for tile in S.range(11, unroll=True):
            scores[tile * 8 + 2] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[tile * 8 + 2],
                    softmax_scale_log2,
                    -new_max_second,
                )
            )
            scores[tile * 8 + 3] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[tile * 8 + 3],
                    softmax_scale_log2,
                    -new_max_second,
                )
            )
            scores[tile * 8 + 6] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[tile * 8 + 6],
                    softmax_scale_log2,
                    -new_max_second,
                )
            )
            scores[tile * 8 + 7] = S.nvvm.fast_exp2(
                S.nvvm.fast_fma(
                    scores[tile * 8 + 7],
                    softmax_scale_log2,
                    -new_max_second,
                )
            )
            tile_sum_second = (
                tile_sum_second
                + scores[tile * 8 + 2]
                + scores[tile * 8 + 3]
                + scores[tile * 8 + 6]
                + scores[tile * 8 + 7]
            )
        tile_sum_first = _group_sum(tile_sum_first)
        tile_sum_second = _group_sum(tile_sum_second)
        sum_first = S.nvvm.fast_fma(sum_first, old_scale_first, tile_sum_first)
        sum_second = S.nvvm.fast_fma(sum_second, old_scale_second, tile_sum_second)
        max_first = new_max_first
        max_second = new_max_second
        for tile in S.range(8, unroll=True):
            output_accumulator[0, tile * 8] = (
                output_accumulator[0, tile * 8] * old_scale_first
            )
            output_accumulator[0, tile * 8 + 1] = (
                output_accumulator[0, tile * 8 + 1] * old_scale_first
            )
            output_accumulator[0, tile * 8 + 4] = (
                output_accumulator[0, tile * 8 + 4] * old_scale_first
            )
            output_accumulator[0, tile * 8 + 5] = (
                output_accumulator[0, tile * 8 + 5] * old_scale_first
            )
            output_accumulator[0, tile * 8 + 2] = (
                output_accumulator[0, tile * 8 + 2] * old_scale_second
            )
            output_accumulator[0, tile * 8 + 3] = (
                output_accumulator[0, tile * 8 + 3] * old_scale_second
            )
            output_accumulator[0, tile * 8 + 6] = (
                output_accumulator[0, tile * 8 + 6] * old_scale_second
            )
            output_accumulator[0, tile * 8 + 7] = (
                output_accumulator[0, tile * 8 + 7] * old_scale_second
            )

        S.nvvm.mbarrier_try_wait_parity(
            value_barrier, (stage >> 1) & 1, 10000000, value_buffer
        )
        value_matrix = S.subview(
            value_shared,
            (value_buffer, 0),
            (11, 2 * 8 * 64),
            (2 * 8 * 64, 1),
        )
        value_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
            value_matrix, SWIZZLE_128B, 0, 0, 0, 1024, 2048
        )
        S.nvvm.wgmma_fence_aligned()
        packed = S.full((1, 4), 0, S.i32)
        packed[0, 0] = _pack_bf16(scores[0], scores[1])
        packed[0, 1] = _pack_bf16(scores[2], scores[3])
        packed[0, 2] = _pack_bf16(scores[4], scores[5])
        packed[0, 3] = _pack_bf16(scores[6], scores[7])
        packed0 = packed[0]
        packed[0, 0] = _pack_bf16(scores[8], scores[9])
        packed[0, 1] = _pack_bf16(scores[10], scores[11])
        packed[0, 2] = _pack_bf16(scores[12], scores[13])
        packed[0, 3] = _pack_bf16(scores[14], scores[15])
        packed1 = packed[0]
        packed[0, 0] = _pack_bf16(scores[16], scores[17])
        packed[0, 1] = _pack_bf16(scores[18], scores[19])
        packed[0, 2] = _pack_bf16(scores[20], scores[21])
        packed[0, 3] = _pack_bf16(scores[22], scores[23])
        packed2 = packed[0]
        packed[0, 0] = _pack_bf16(scores[24], scores[25])
        packed[0, 1] = _pack_bf16(scores[26], scores[27])
        packed[0, 2] = _pack_bf16(scores[28], scores[29])
        packed[0, 3] = _pack_bf16(scores[30], scores[31])
        packed3 = packed[0]
        packed[0, 0] = _pack_bf16(scores[32], scores[33])
        packed[0, 1] = _pack_bf16(scores[34], scores[35])
        packed[0, 2] = _pack_bf16(scores[36], scores[37])
        packed[0, 3] = _pack_bf16(scores[38], scores[39])
        packed4 = packed[0]
        packed[0, 0] = _pack_bf16(scores[40], scores[41])
        packed[0, 1] = _pack_bf16(scores[42], scores[43])
        packed[0, 2] = _pack_bf16(scores[44], scores[45])
        packed[0, 3] = _pack_bf16(scores[46], scores[47])
        packed5 = packed[0]
        packed[0, 0] = _pack_bf16(scores[48], scores[49])
        packed[0, 1] = _pack_bf16(scores[50], scores[51])
        packed[0, 2] = _pack_bf16(scores[52], scores[53])
        packed[0, 3] = _pack_bf16(scores[54], scores[55])
        packed6 = packed[0]
        packed[0, 0] = _pack_bf16(scores[56], scores[57])
        packed[0, 1] = _pack_bf16(scores[58], scores[59])
        packed[0, 2] = _pack_bf16(scores[60], scores[61])
        packed[0, 3] = _pack_bf16(scores[62], scores[63])
        packed7 = packed[0]
        packed[0, 0] = _pack_bf16(scores[64], scores[65])
        packed[0, 1] = _pack_bf16(scores[66], scores[67])
        packed[0, 2] = _pack_bf16(scores[68], scores[69])
        packed[0, 3] = _pack_bf16(scores[70], scores[71])
        packed8 = packed[0]
        packed[0, 0] = _pack_bf16(scores[72], scores[73])
        packed[0, 1] = _pack_bf16(scores[74], scores[75])
        packed[0, 2] = _pack_bf16(scores[76], scores[77])
        packed[0, 3] = _pack_bf16(scores[78], scores[79])
        packed9 = packed[0]
        packed[0, 0] = _pack_bf16(scores[80], scores[81])
        packed[0, 1] = _pack_bf16(scores[82], scores[83])
        packed[0, 2] = _pack_bf16(scores[84], scores[85])
        packed[0, 3] = _pack_bf16(scores[86], scores[87])
        packed10 = packed[0]
        pv_result = S.nvvm.wgmma_m64n128k176_f32_bf16_bf16_rs(
            packed0,
            packed1,
            packed2,
            packed3,
            packed4,
            packed5,
            packed6,
            packed7,
            packed8,
            packed9,
            packed10,
            value_matrix_desc,
            output_accumulator[0],
        )
        S.nvvm.wgmma_group_sync_aligned()

        next_stage = stage + 1
        next_key_buffer = next_stage & 1
        S.nvvm.mbarrier_try_wait_parity(
            key_barrier,
            (next_stage >> 1) & 1,
            10000000,
            next_key_buffer,
        )
        next_key_matrix = S.subview(
            key_shared,
            (next_key_buffer, 0),
            (176, HEAD_DIM),
            (HEAD_DIM, 1),
        )
        key_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
            next_key_matrix, SWIZZLE_128B, 0, 0, 0, 16, 1024
        )
        query_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
            query_matrix, SWIZZLE_128B, 0, 0, 0, 16, 1024
        )
        S.nvvm.wgmma_fence_aligned()
        next_scores = S.nvvm.wgmma_m64n176k128_f32_bf16_bf16_ss(
            query_matrix_desc, key_matrix_desc
        )
        S.nvvm.wgmma_group_sync_aligned()
        S.nvvm.wgmma_wait_group_sync(1)
        for result_element in S.range(64, unroll=True):
            output_accumulator[0, result_element] = pv_result[result_element]
        S.nvvm.mbarrier_arrive(empty_barrier, value_buffer)
        S.nvvm.wgmma_wait_group_sync(0)
        for result_element in S.range(88, unroll=True):
            scores[result_element] = next_scores[result_element]

    key_tile = 93
    stage = stage_base + key_tile
    key_buffer = stage & 1
    value_buffer = stage & 1
    for tile in S.range(1, 11, unroll=True):
        for element in S.range(8, unroll=True):
            scores[tile * 8 + element] = S.convert(NEG_INFINITY, S.f32)

    tile_max_first = S.convert(NEG_INFINITY, S.f32)
    tile_max_second = S.convert(NEG_INFINITY, S.f32)
    for tile in S.range(11, unroll=True):
        tile_max_first = S.nvvm.fast_fmax(
            tile_max_first,
            S.nvvm.fast_fmax(
                S.nvvm.fast_fmax(scores[tile * 8], scores[tile * 8 + 1]),
                S.nvvm.fast_fmax(scores[tile * 8 + 4], scores[tile * 8 + 5]),
            ),
        )
        tile_max_second = S.nvvm.fast_fmax(
            tile_max_second,
            S.nvvm.fast_fmax(
                S.nvvm.fast_fmax(scores[tile * 8 + 2], scores[tile * 8 + 3]),
                S.nvvm.fast_fmax(scores[tile * 8 + 6], scores[tile * 8 + 7]),
            ),
        )
    tile_max_first = _group_max(tile_max_first * softmax_scale_log2)
    tile_max_second = _group_max(tile_max_second * softmax_scale_log2)
    new_max_first = S.nvvm.fast_fmax(max_first, tile_max_first)
    new_max_second = S.nvvm.fast_fmax(max_second, tile_max_second)
    old_scale_first = S.nvvm.fast_exp2(max_first - new_max_first)
    old_scale_second = S.nvvm.fast_exp2(max_second - new_max_second)
    tile_sum_first = S.convert(0.0, S.f32)
    tile_sum_second = S.convert(0.0, S.f32)

    for tile in S.range(11, unroll=True):
        scores[tile * 8 + 0] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(scores[tile * 8], softmax_scale_log2, -new_max_first)
        )
        scores[tile * 8 + 1] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(
                scores[tile * 8 + 1],
                softmax_scale_log2,
                -new_max_first,
            )
        )
        scores[tile * 8 + 4] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(
                scores[tile * 8 + 4],
                softmax_scale_log2,
                -new_max_first,
            )
        )
        scores[tile * 8 + 5] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(
                scores[tile * 8 + 5],
                softmax_scale_log2,
                -new_max_first,
            )
        )
        tile_sum_first = (
            tile_sum_first
            + scores[tile * 8 + 0]
            + scores[tile * 8 + 1]
            + scores[tile * 8 + 4]
            + scores[tile * 8 + 5]
        )
    for tile in S.range(11, unroll=True):
        scores[tile * 8 + 2] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(
                scores[tile * 8 + 2],
                softmax_scale_log2,
                -new_max_second,
            )
        )
        scores[tile * 8 + 3] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(
                scores[tile * 8 + 3],
                softmax_scale_log2,
                -new_max_second,
            )
        )
        scores[tile * 8 + 6] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(
                scores[tile * 8 + 6],
                softmax_scale_log2,
                -new_max_second,
            )
        )
        scores[tile * 8 + 7] = S.nvvm.fast_exp2(
            S.nvvm.fast_fma(
                scores[tile * 8 + 7],
                softmax_scale_log2,
                -new_max_second,
            )
        )
        tile_sum_second = (
            tile_sum_second
            + scores[tile * 8 + 2]
            + scores[tile * 8 + 3]
            + scores[tile * 8 + 6]
            + scores[tile * 8 + 7]
        )
    tile_sum_first = _group_sum(tile_sum_first)
    tile_sum_second = _group_sum(tile_sum_second)
    sum_first = S.nvvm.fast_fma(sum_first, old_scale_first, tile_sum_first)
    sum_second = S.nvvm.fast_fma(sum_second, old_scale_second, tile_sum_second)
    max_first = new_max_first
    max_second = new_max_second
    for tile in S.range(8, unroll=True):
        output_accumulator[0, tile * 8] = (
            output_accumulator[0, tile * 8] * old_scale_first
        )
        output_accumulator[0, tile * 8 + 1] = (
            output_accumulator[0, tile * 8 + 1] * old_scale_first
        )
        output_accumulator[0, tile * 8 + 4] = (
            output_accumulator[0, tile * 8 + 4] * old_scale_first
        )
        output_accumulator[0, tile * 8 + 5] = (
            output_accumulator[0, tile * 8 + 5] * old_scale_first
        )
        output_accumulator[0, tile * 8 + 2] = (
            output_accumulator[0, tile * 8 + 2] * old_scale_second
        )
        output_accumulator[0, tile * 8 + 3] = (
            output_accumulator[0, tile * 8 + 3] * old_scale_second
        )
        output_accumulator[0, tile * 8 + 6] = (
            output_accumulator[0, tile * 8 + 6] * old_scale_second
        )
        output_accumulator[0, tile * 8 + 7] = (
            output_accumulator[0, tile * 8 + 7] * old_scale_second
        )

    S.nvvm.mbarrier_try_wait_parity(
        value_barrier, (stage >> 1) & 1, 10000000, value_buffer
    )
    value_matrix = S.subview(
        value_shared,
        (value_buffer, 0),
        (11, 2 * 8 * 64),
        (2 * 8 * 64, 1),
    )
    value_matrix_desc = S.nvvm.make_wgmma_descriptor_bits(
        value_matrix, SWIZZLE_128B, 0, 0, 0, 1024, 2048
    )
    S.nvvm.wgmma_fence_aligned()
    last_packed = S.full((1, 4), 0, S.i32)
    last_packed[0, 0] = _pack_bf16(scores[0], scores[1])
    last_packed[0, 1] = _pack_bf16(scores[2], scores[3])
    last_packed[0, 2] = _pack_bf16(scores[4], scores[5])
    last_packed[0, 3] = _pack_bf16(scores[6], scores[7])
    last_packed0 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[8], scores[9])
    last_packed[0, 1] = _pack_bf16(scores[10], scores[11])
    last_packed[0, 2] = _pack_bf16(scores[12], scores[13])
    last_packed[0, 3] = _pack_bf16(scores[14], scores[15])
    last_packed1 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[16], scores[17])
    last_packed[0, 1] = _pack_bf16(scores[18], scores[19])
    last_packed[0, 2] = _pack_bf16(scores[20], scores[21])
    last_packed[0, 3] = _pack_bf16(scores[22], scores[23])
    last_packed2 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[24], scores[25])
    last_packed[0, 1] = _pack_bf16(scores[26], scores[27])
    last_packed[0, 2] = _pack_bf16(scores[28], scores[29])
    last_packed[0, 3] = _pack_bf16(scores[30], scores[31])
    last_packed3 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[32], scores[33])
    last_packed[0, 1] = _pack_bf16(scores[34], scores[35])
    last_packed[0, 2] = _pack_bf16(scores[36], scores[37])
    last_packed[0, 3] = _pack_bf16(scores[38], scores[39])
    last_packed4 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[40], scores[41])
    last_packed[0, 1] = _pack_bf16(scores[42], scores[43])
    last_packed[0, 2] = _pack_bf16(scores[44], scores[45])
    last_packed[0, 3] = _pack_bf16(scores[46], scores[47])
    last_packed5 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[48], scores[49])
    last_packed[0, 1] = _pack_bf16(scores[50], scores[51])
    last_packed[0, 2] = _pack_bf16(scores[52], scores[53])
    last_packed[0, 3] = _pack_bf16(scores[54], scores[55])
    last_packed6 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[56], scores[57])
    last_packed[0, 1] = _pack_bf16(scores[58], scores[59])
    last_packed[0, 2] = _pack_bf16(scores[60], scores[61])
    last_packed[0, 3] = _pack_bf16(scores[62], scores[63])
    last_packed7 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[64], scores[65])
    last_packed[0, 1] = _pack_bf16(scores[66], scores[67])
    last_packed[0, 2] = _pack_bf16(scores[68], scores[69])
    last_packed[0, 3] = _pack_bf16(scores[70], scores[71])
    last_packed8 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[72], scores[73])
    last_packed[0, 1] = _pack_bf16(scores[74], scores[75])
    last_packed[0, 2] = _pack_bf16(scores[76], scores[77])
    last_packed[0, 3] = _pack_bf16(scores[78], scores[79])
    last_packed9 = last_packed[0]
    last_packed[0, 0] = _pack_bf16(scores[80], scores[81])
    last_packed[0, 1] = _pack_bf16(scores[82], scores[83])
    last_packed[0, 2] = _pack_bf16(scores[84], scores[85])
    last_packed[0, 3] = _pack_bf16(scores[86], scores[87])
    last_packed10 = last_packed[0]
    pv_result = S.nvvm.wgmma_m64n128k176_f32_bf16_bf16_rs(
        last_packed0,
        last_packed1,
        last_packed2,
        last_packed3,
        last_packed4,
        last_packed5,
        last_packed6,
        last_packed7,
        last_packed8,
        last_packed9,
        last_packed10,
        value_matrix_desc,
        output_accumulator[0],
    )
    S.nvvm.wgmma_group_sync_aligned()

    S.nvvm.wgmma_wait_group_sync(0)
    for result_element in S.range(64, unroll=True):
        output_accumulator[0, result_element] = pv_result[result_element]
    S.nvvm.mbarrier_arrive(empty_barrier, value_buffer)

    inverse_first = S.nvvm.fast_rcp(sum_first)
    inverse_second = S.nvvm.fast_rcp(sum_second)
    for tile in S.range(8, unroll=True):
        row = warp * 16 + (lane & 15)
        column = tile * 16 + (lane >> 4) * 8
        linear = row * 64 + (column & 63)
        if column >= 64:
            linear = linear + CONSUMER_ROWS * 64
        offset = linear ^ ((linear & 0x1C0) >> 3)
        destination = S.subview(query_shared, (consumer, offset), (8, 8), (1, 1))
        packed = S.full((1, 4), 0, S.i32)
        packed[0, 0] = _pack_bf16(
            output_accumulator[0, tile * 8] * inverse_first,
            output_accumulator[0, tile * 8 + 1] * inverse_first,
        )
        packed[0, 1] = _pack_bf16(
            output_accumulator[0, tile * 8 + 2] * inverse_second,
            output_accumulator[0, tile * 8 + 3] * inverse_second,
        )
        packed[0, 2] = _pack_bf16(
            output_accumulator[0, tile * 8 + 4] * inverse_first,
            output_accumulator[0, tile * 8 + 5] * inverse_first,
        )
        packed[0, 3] = _pack_bf16(
            output_accumulator[0, tile * 8 + 6] * inverse_second,
            output_accumulator[0, tile * 8 + 7] * inverse_second,
        )
        S.nvvm.stmatrix_m8n8_x4_b16(destination, packed[0])
    S.nvvm.fence_proxy_async_shared_cta()
    S.nvvm.named_barrier_sync(1 + consumer, WARPGROUP_THREADS)
    if warpgroup_thread == 0:
        output_source_raw = S.subview(
            query_shared,
            (consumer, 0),
            (1, CONSUMER_ROWS * HEAD_DIM),
            (1, 1),
        )
        output_source = S.view(
            output_source_raw,
            S.bf16,
            S.make_layout((2, 8, 8, 64), (4096, 512, 64, 1)),
        )
        S.nvvm.tma_store(
            output_source,
            output_desc,
            (
                0,
                0,
                batch * 16384 + query_start + consumer * 8,
                0,
            ),
            predicate=warpgroup_thread == 0,
        )
        S.nvvm.cp_async_bulk_commit_group()
        S.nvvm.cp_async_bulk_wait_group(0, read=True)



_ATTENTION_KERNELS = {
    1024: _attention_mqa_1024_kernel,
    2048: _attention_mqa_2048_kernel,
    4096: _attention_mqa_4096_kernel,
    8192: _attention_mqa_8192_kernel,
    16384: _attention_mqa_16384_kernel,
}

_ATTENTION_BLOCKS = {
    1024: PERSISTENT_BLOCKS,
    2048: PERSISTENT_BLOCKS,
    # One CTA owns each packed 16-token tile for the long-sequence path.
    4096: BATCH * (4096 // PACKED_QUERY_TOKENS),
    8192: BATCH * (8192 // PACKED_QUERY_TOKENS),
    16384: BATCH * (16384 // PACKED_QUERY_TOKENS),
}


def flash_attention_mqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if query.ndim != 4:
        raise ValueError("query must have rank 4")
    sequence = query.shape[1]
    if sequence not in SUPPORTED_SEQUENCES:
        raise ValueError(
            f"sequence length must be one of {SUPPORTED_SEQUENCES}, got {sequence}"
        )
    expected_query_shape = (BATCH, sequence, QUERY_HEADS, HEAD_DIM)
    expected_kv_shape = (BATCH, sequence, 1, HEAD_DIM)
    for name, tensor, shape in (
        ("query", query, expected_query_shape),
        ("key", key, expected_kv_shape),
        ("value", value, expected_kv_shape),
    ):
        if tensor.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must use torch.bfloat16")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be on CUDA")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if key.device != query.device or value.device != query.device:
        raise ValueError("query, key, and value must be on the same device")

    if output is None:
        output = torch.empty_like(query)
    elif output.shape != expected_query_shape:
        raise ValueError(f"output must have shape {expected_query_shape}")
    elif output.dtype != torch.bfloat16 or output.device != query.device:
        raise ValueError("output must match query dtype and device")
    elif not output.is_contiguous():
        raise ValueError("output must be contiguous")

    kernel = _ATTENTION_KERNELS[sequence]
    launch = kernel[
        lambda: ((_ATTENTION_BLOCKS[sequence], 1, 1), (THREADS, 1, 1))
    ]
    if sequence in (2048, 8192):
        launch(
            query,
            key,
            value,
            output,
            sequence,
            num_warps=12,
            fast_math=True,
        )
    else:
        launch(query, key, value, output, num_warps=12, fast_math=True)
    return output
