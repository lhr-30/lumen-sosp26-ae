import torch

import avelang
import avelang.language as S


SUPPORTED_SIZES = (1024, 2048, 4096, 8192, 16384)

TILE_M = 128
TILE_N = 256
TILE_K = 64
PIPELINE_STAGES = 4
CLUSTER_M = 2
WARPGROUP_THREADS = 128
THREADS = 384
PERSISTENT_CLUSTERS = 66
SWIZZLE_128B = S.WGMMA_SWIZZLE_128B


@avelang.jit
def _pack_bf16(first: S.f32, second: S.f32) -> S.i32:
    return S.nvvm.floatx2_to_bf16x2(first, second)


@avelang.jit
def _gemm_bf16_64x128_kernel(
    a: S.Tensor((1024, 1024), S.bf16),
    b: S.Tensor((1024, 1024), S.bf16),
    c: S.Tensor((1024, 1024), S.bf16),
):
    tid = S.thread_id(0)
    lane128 = tid & 127
    warp = lane128 // 32
    lane = lane128 & 31
    cluster_rank = S.nvvm.cluster_block_rank()

    full_barrier = S.nvvm.mbarrier_create(8)
    empty_barrier = S.nvvm.mbarrier_create(8)
    output_shared = S.make_shared((1, 64 * 128), S.bf16, 128)
    b_shared = S.make_shared((8, 128 * 64), S.bf16, 128)
    a_shared = S.make_shared((8, 64 * 64), S.bf16, 128)
    output_shared_vectors = S.view(
        output_shared,
        S.u32,
        S.make_layout((1024, 4), (4, 1)),
    )

    a_tensor = S.view(
        a,
        S.bf16,
        S.make_layout((16, 1024, 64), (64, 1024, 1)),
    )
    b_tensor = S.view(
        b,
        S.bf16,
        S.make_layout((16, 1024, 64), (64, 1024, 1)),
    )
    operand_layout = S.make_layout((1, 64, 64), (4096, 64, 1))
    a_tma = S.nvvm.make_tma_descriptor(a_tensor, operand_layout, SWIZZLE_128B)
    b_tma = S.nvvm.make_tma_descriptor(b_tensor, operand_layout, SWIZZLE_128B)

    a_matrix_base = S.view(
        S.subview(a_shared, (0, 0), (1, 4096), (1, 1)),
        S.bf16,
        S.make_layout((64, 64), (64, 1)),
    )
    b_matrix0_base = S.view(
        S.subview(b_shared, (0, 0), (1, 4096), (1, 1)),
        S.bf16,
        S.make_layout((64, 64), (64, 1)),
    )
    b_matrix1_base = S.view(
        S.subview(b_shared, (0, 4096), (1, 4096), (1, 1)),
        S.bf16,
        S.make_layout((64, 64), (64, 1)),
    )
    desc_a_base = S.nvvm.make_wgmma_descriptor_bits(
        a_matrix_base, SWIZZLE_128B, 0, 0, 0, 8192, 1024
    )
    desc_b0_base = S.nvvm.make_wgmma_descriptor_bits(
        b_matrix0_base, SWIZZLE_128B, 0, 0, 0, 8192, 1024
    )
    desc_b1_base = S.nvvm.make_wgmma_descriptor_bits(
        b_matrix1_base, SWIZZLE_128B, 0, 0, 0, 8192, 1024
    )

    for work in S.range(S.block_id(1), 64, S.grid_dim(1)):
        for stage in S.range(8, unroll=True):
            S.nvvm.mbarrier_init(
                full_barrier, stage, count=1, predicate=tid == 0
            )
            S.nvvm.mbarrier_init(
                empty_barrier, stage, count=2, predicate=tid == 0
            )
        S.nvvm.fence_mbarrier_init_release_cluster()
        S.nvvm.cluster_arrive_relaxed()
        S.nvvm.cluster_wait()

        cluster_m = work // 8
        cluster_n = work & 7
        cta_m = cluster_m * 128 + cluster_rank * 64
        cta_n = cluster_n * 128

        if tid < 128:
            accumulators = S.make_local((2, 32), S.f32)
            for tile in S.range(2, unroll=True):
                for element in S.range(32, unroll=True):
                    accumulators[tile, element] = S.convert(0.0, S.f32)
            full_phase = 0
            for k_tile in S.range(16):
                stage = k_tile & 7
                S.nvvm.mbarrier_try_wait_parity(
                    full_barrier, full_phase, 10000000, stage
                )
                desc_a = desc_a_base + stage * 512
                desc_b0 = desc_b0_base + stage * 1024
                desc_b1 = desc_b1_base + stage * 1024
                accumulator0 = accumulators[0]
                accumulator1 = accumulators[1]
                S.nvvm.wgmma_fence_aligned()
                accumulator0 = S.nvvm.wgmma_m64n64k16_f32_bf16_bf16(
                    desc_b0,
                    desc_a,
                    accumulator0,
                    S.convert(k_tile != 0, S.i32),
                )
                accumulator0 = S.nvvm.wgmma_m64n64k16_f32_bf16_bf16(
                    desc_b0 + 2, desc_a + 2, accumulator0, 1
                )
                accumulator0 = S.nvvm.wgmma_m64n64k16_f32_bf16_bf16(
                    desc_b0 + 4, desc_a + 4, accumulator0, 1
                )
                accumulator0 = S.nvvm.wgmma_m64n64k16_f32_bf16_bf16(
                    desc_b0 + 6, desc_a + 6, accumulator0, 1
                )
                accumulator1 = S.nvvm.wgmma_m64n64k16_f32_bf16_bf16(
                    desc_b1,
                    desc_a,
                    accumulator1,
                    S.convert(k_tile != 0, S.i32),
                )
                accumulator1 = S.nvvm.wgmma_m64n64k16_f32_bf16_bf16(
                    desc_b1 + 2, desc_a + 2, accumulator1, 1
                )
                accumulator1 = S.nvvm.wgmma_m64n64k16_f32_bf16_bf16(
                    desc_b1 + 4, desc_a + 4, accumulator1, 1
                )
                accumulator1 = S.nvvm.wgmma_m64n64k16_f32_bf16_bf16(
                    desc_b1 + 6, desc_a + 6, accumulator1, 1
                )
                S.nvvm.wgmma_group_sync_aligned()
                S.nvvm.wgmma_wait_group_sync(1)
                for element in S.range(32, unroll=True):
                    accumulators[0, element] = accumulator0[element]
                    accumulators[1, element] = accumulator1[element]
                if k_tile != 0:
                    previous_stage = (k_tile - 1) & 7
                    if lane128 < 16 and (lane128 & 7) == 0:
                        S.nvvm.mbarrier_arrive_cluster(
                            empty_barrier, previous_stage, lane128 // 8
                        )
                if stage == 7:
                    full_phase = full_phase ^ 1

            S.nvvm.wgmma_wait_group_sync(0)
            S.nvvm.wgmma_fence_aligned()
            if lane128 < 16 and (lane128 & 7) == 0:
                S.nvvm.mbarrier_arrive_cluster(
                    empty_barrier, 7, lane128 // 8
                )
            for half_n in S.range(2, unroll=True):
                for box_m in S.range(2, unroll=True):
                    for local_group in S.range(2, unroll=True):
                        packed = S.full((1, 4), 0, S.i32)
                        source_group = 2 * box_m + local_group
                        for matrix in S.range(4, unroll=True):
                            i = matrix & 1
                            source_w = 2 * source_group + matrix // 2
                            packed[0, matrix] = _pack_bf16(
                                accumulators[
                                    half_n, 4 * source_w + 2 * i
                                ],
                                accumulators[
                                    half_n, 4 * source_w + 2 * i + 1
                                ],
                            )
                        address_matrix = lane // 8
                        address_i = address_matrix & 1
                        local_address_w = (
                            2 * local_group + address_matrix // 2
                        )
                        logical_byte_offset = (
                            (local_address_w * 8 + lane % 8) * 64
                            + warp * 16
                            + address_i * 8
                        ) * 2
                        swizzled_byte_offset = (
                            logical_byte_offset
                            ^ ((logical_byte_offset & 0x380) >> 3)
                        )
                        box_offset = (half_n * 2 + box_m) * 2048
                        destination = S.subview(
                            output_shared,
                            (0, box_offset + swizzled_byte_offset // 2),
                            (8, 8),
                            (1, 1),
                        )
                        S.nvvm.stmatrix_m8n8_x4_b16_trans(
                            destination, packed[0]
                        )
            S.nvvm.named_barrier_sync(1, 128)
            row0 = warp * 8 + lane // 8
            for half_n in S.range(2, unroll=True):
                for box_m in S.range(2, unroll=True):
                    box_byte_offset = (half_n * 2 + box_m) * 4096
                    row0_byte_offset = (row0 * 64 + (lane % 8) * 8) * 2
                    row1_byte_offset = (
                        (row0 + 4) * 64 + (lane % 8) * 8
                    ) * 2
                    shared_row0 = (
                        box_byte_offset
                        + (
                            row0_byte_offset
                            ^ ((row0_byte_offset & 0x380) >> 3)
                        )
                    ) // 16
                    shared_row1 = (
                        box_byte_offset
                        + (
                            row1_byte_offset
                            ^ ((row1_byte_offset & 0x380) >> 3)
                        )
                    ) // 16
                    output_row0 = cta_m + box_m * 32 + row0
                    output_row1 = output_row0 + 4
                    output_column = (
                        cta_n + half_n * 64 + (lane % 8) * 8
                    )
                    S.nvvm.store_global_v4_u32(
                        c,
                        (output_row0 * 1024 + output_column) * 2,
                        output_shared_vectors[shared_row0],
                    )
                    S.nvvm.store_global_v4_u32(
                        c,
                        (output_row1 * 1024 + output_column) * 2,
                        output_shared_vectors[shared_row1],
                    )

        else:
            elected = S.nvvm.elect_sync()
            empty_phase = 1
            for k_tile in S.range(16):
                stage = k_tile & 7
                S.nvvm.mbarrier_try_wait_parity(
                    empty_barrier, empty_phase, 10000000, stage
                )
                S.nvvm.mbarrier_arrive_expect_tx(
                    full_barrier, (64 + 128) * 64 * 2,
                    stage, elected,
                )
                a_raw = S.subview(a_shared, (stage, 0), (1, 4096), (1, 1))
                a_destination = S.view(
                    a_raw,
                    S.bf16,
                    S.make_layout((1, 64, 64), (4096, 64, 1)),
                )
                S.nvvm.tma_load(
                    a_destination, a_tma, (0, cta_m, k_tile),
                    full_barrier, mbar_id=stage, predicate=elected,
                    expect_tx=False,
                )
                b_raw = S.subview(
                    b_shared,
                    (stage, cluster_rank * 4096),
                    (1, 4096),
                    (1, 1),
                )
                b_destination = S.view(
                    b_raw,
                    S.bf16,
                    S.make_layout((1, 64, 64), (4096, 64, 1)),
                )
                S.nvvm.tma_load(
                    b_destination, b_tma,
                    (0, cta_n + cluster_rank * 64, k_tile),
                    full_barrier, mbar_id=stage, predicate=elected,
                    expect_tx=False, multicast_mask=3,
                )
                if stage == 7:
                    empty_phase = empty_phase ^ 1

            for stage in S.range(8, unroll=True):
                S.nvvm.mbarrier_try_wait_parity(
                    empty_barrier, empty_phase, 10000000, stage
                )
        S.syncthreads()


@avelang.jit
def _gemm_bf16_128x256_kernel(
    a_ptr: S.Pointer(S.bf16),
    b_ptr: S.Pointer(S.bf16),
    c_ptr: S.Pointer(S.bf16),
    size: S.constexpr,
):
    tid = S.thread_id(0)
    warpgroup = tid // WARPGROUP_THREADS
    warpgroup_thread = tid % WARPGROUP_THREADS
    warp = warpgroup_thread // 32
    lane = warpgroup_thread % 32
    cluster_rank = S.nvvm.cluster_block_rank()

    # Ave emits shared globals in reverse declaration order
    # The particular order does not affect the tensor-map layouts,
    # but keeping barriers first leaves the large operand arena
    # naturally 128-byte aligned.
    full_barrier = S.nvvm.mbarrier_create(PIPELINE_STAGES)
    empty_barrier = S.nvvm.mbarrier_create(PIPELINE_STAGES)
    output_shared = S.make_shared((2, 64 * 128), S.bf16, 128)
    b_shared = S.make_shared(
        (PIPELINE_STAGES, TILE_N * TILE_K), S.bf16, 128
    )
    a_shared = S.make_shared(
        (PIPELINE_STAGES, TILE_M * TILE_K), S.bf16, 128
    )

    # Split K into 64-element coordinates
    a_tensor = S.make_tensor(
        a_ptr,
        S.bf16,
        S.make_layout((size, size // TILE_K, TILE_K),
                      (size, TILE_K, 1)),
    )
    b_tensor = S.make_tensor(
        b_ptr,
        S.bf16,
        S.make_layout((size, size // TILE_K, TILE_K),
                      (size, TILE_K, 1)),
    )
    c_tensor = S.make_tensor(
        c_ptr,
        S.bf16,
        S.make_layout((size, size), (size, 1)),
    )

    # Keep box layouts literal: tensor-map metadata is collected from the AST
    # before runtime constexpr substitution.
    a_tma = S.nvvm.make_tma_descriptor(
        a_tensor,
        S.make_layout((128, 1, 64), (64, 8192, 1)),
        SWIZZLE_128B,
    )
    b_tma = S.nvvm.make_tma_descriptor(
        b_tensor,
        S.make_layout((128, 1, 64), (64, 8192, 1)),
        SWIZZLE_128B,
    )
    c_tma = S.nvvm.make_tma_descriptor(
        c_tensor,
        S.make_layout((64, 64), (64, 1)),
        SWIZZLE_128B,
    )

    cluster_tiles_m = size // (TILE_M * CLUSTER_M)
    cluster_tiles_n = size // TILE_N
    total_cluster_tiles = cluster_tiles_m * cluster_tiles_n
    interior_tiles_m = 0
    interior_tiles_n = 0
    right_fringe_tiles = 0
    top_fringe_tiles = 0
    if size == 16384:
        interior_size = (size // 768) * 768
        interior_tiles_m = interior_size // (TILE_M * CLUSTER_M)
        interior_tiles_n = interior_size // TILE_N
        right_fringe_tiles = cluster_tiles_n - interior_tiles_n
        top_fringe_tiles = interior_tiles_m * right_fringe_tiles
        total_cluster_tiles = (
            top_fringe_tiles
            + (cluster_tiles_m - interior_tiles_m) * cluster_tiles_n
        )

    # Match the H200 reference worker ordering. Spreading the first persistent
    # wave across M and N avoids a long tail at the grid boundary.
    worker = S.block_id(1)
    if size >= 4096:
        if worker < 5:
            worker = worker
        elif worker < 11:
            worker = (worker - 4) * 8
        elif worker == 11:
            worker = 57
        elif worker < 36:
            group = (worker - 12) // 8
            position = (worker - 12) % 8
            if position == 0:
                worker = 5 + group
            elif position == 1:
                worker = 9 + group
            elif position == 7:
                worker = 58 + group
            else:
                worker = 17 + group + (position - 2) * 8
        elif worker < 64:
            group = (worker - 36) // 7
            position = (worker - 36) % 7
            if position < 6:
                worker = 12 + group + position * 8
            else:
                worker = 61 + group
        elif worker == 64:
            worker = 56
        else:
            worker = 65

    # Initialize once, then drain each stage's final consumer release below
    # before reusing these barriers for the next persistent output tile.
    for stage in S.range(PIPELINE_STAGES, unroll=True):
        S.nvvm.mbarrier_init(
            full_barrier, stage, count=1, predicate=tid == 0
        )
        S.nvvm.mbarrier_init(
            empty_barrier, stage, count=4, predicate=tid == 0
        )
    S.nvvm.fence_mbarrier_init_release_cluster()
    S.nvvm.cluster_arrive_relaxed()
    S.nvvm.cluster_wait()

    for work in S.range(
        worker, total_cluster_tiles, S.grid_dim(1)
    ):
        cluster_tile_m = work // cluster_tiles_n
        cluster_tile_n = work % cluster_tiles_n
        if size == 16384:
            if work < top_fringe_tiles:
                cluster_tile_m = work // right_fringe_tiles
                cluster_tile_n = (
                    interior_tiles_n + work % right_fringe_tiles
                )
            else:
                bottom_work = work - top_fringe_tiles
                cluster_tile_m = (
                    interior_tiles_m + bottom_work // cluster_tiles_n
                )
                cluster_tile_n = bottom_work % cluster_tiles_n
        cta_m = cluster_tile_m * (TILE_M * CLUSTER_M) + cluster_rank * TILE_M
        cta_n = cluster_tile_n * TILE_N

        if warpgroup < 2:
            S.nvvm.setmaxnreg_inc(168)
            accumulators = S.make_local((2, 64), S.f32)
            for tile in S.range(2, unroll=True):
                for element in S.range(64, unroll=True):
                    accumulators[tile, element] = S.convert(0.0, S.f32)
            full_phase = 0

            for k_tile in S.range(size // TILE_K):
                stage = k_tile % PIPELINE_STAGES
                S.nvvm.mbarrier_try_wait_parity(
                    full_barrier, full_phase, 10000000, stage
                )
                S.nvvm.fence_proxy_async_shared_cta()

                b_stage = S.subview(
                    b_shared,
                    (stage, warpgroup * 128 * TILE_K),
                    (1, 128 * TILE_K),
                    (1, 1),
                )
                a_matrix0 = S.view(
                    S.subview(
                        a_shared,
                        (stage, 0),
                        (1, 64 * TILE_K),
                        (1, 1),
                    ),
                    S.bf16,
                    S.make_layout((64, TILE_K), (TILE_K, 1)),
                )
                a_matrix1 = S.view(
                    S.subview(
                        a_shared,
                        (stage, 64 * TILE_K),
                        (1, 64 * TILE_K),
                        (1, 1),
                    ),
                    S.bf16,
                    S.make_layout((64, TILE_K), (TILE_K, 1)),
                )
                b_matrix = S.view(
                    b_stage,
                    S.bf16,
                    S.make_layout((128, TILE_K), (TILE_K, 1)),
                )
                desc_a0 = S.nvvm.make_wgmma_descriptor_bits(
                    a_matrix0, SWIZZLE_128B, 0, 0, 0, 16, 1024
                )
                desc_a1 = S.nvvm.make_wgmma_descriptor_bits(
                    a_matrix1, SWIZZLE_128B, 0, 0, 0, 16, 1024
                )
                desc_b = S.nvvm.make_wgmma_descriptor_bits(
                    b_matrix, SWIZZLE_128B, 0, 0, 0, 16, 1024
                )
                accumulator0 = accumulators[0]
                accumulator1 = accumulators[1]
                S.nvvm.wgmma_fence_aligned()
                accumulator0 = S.nvvm.wgmma_m64n128k16_f32_bf16_bf16(
                    desc_a0, desc_b, accumulator0, 1
                )
                desc_a0 = desc_a0 + 2
                desc_b = desc_b + 2
                accumulator0 = S.nvvm.wgmma_m64n128k16_f32_bf16_bf16(
                    desc_a0, desc_b, accumulator0, 1
                )
                desc_a0 = desc_a0 + 2
                desc_b = desc_b + 2
                accumulator0 = S.nvvm.wgmma_m64n128k16_f32_bf16_bf16(
                    desc_a0, desc_b, accumulator0, 1
                )
                desc_a0 = desc_a0 + 2
                desc_b = desc_b + 2
                accumulator0 = S.nvvm.wgmma_m64n128k16_f32_bf16_bf16(
                    desc_a0, desc_b, accumulator0, 1
                )
                S.nvvm.wgmma_fence_aligned()
                desc_b = desc_b - 6
                accumulator1 = S.nvvm.wgmma_m64n128k16_f32_bf16_bf16(
                    desc_a1, desc_b, accumulator1, 1
                )
                desc_a1 = desc_a1 + 2
                desc_b = desc_b + 2
                accumulator1 = S.nvvm.wgmma_m64n128k16_f32_bf16_bf16(
                    desc_a1, desc_b, accumulator1, 1
                )
                desc_a1 = desc_a1 + 2
                desc_b = desc_b + 2
                accumulator1 = S.nvvm.wgmma_m64n128k16_f32_bf16_bf16(
                    desc_a1, desc_b, accumulator1, 1
                )
                desc_a1 = desc_a1 + 2
                desc_b = desc_b + 2
                accumulator1 = S.nvvm.wgmma_m64n128k16_f32_bf16_bf16(
                    desc_a1, desc_b, accumulator1, 1
                )
                S.nvvm.wgmma_group_sync_aligned()
                if size == 4096:
                    # Fill the four-stage pipeline to hide the barrier latency
                    # that dominates this size.
                    S.nvvm.wgmma_wait_group_sync(4)
                else:
                    S.nvvm.wgmma_wait_group_sync(1)
                for element in S.range(64, unroll=True):
                    accumulators[0, element] = accumulator0[element]
                    accumulators[1, element] = accumulator1[element]

                if k_tile != 0:
                    previous_stage = (k_tile - 1) % PIPELINE_STAGES
                    if warpgroup_thread == 0:
                        S.nvvm.mbarrier_arrive_cluster(
                            empty_barrier, previous_stage, 0
                        )
                    if warpgroup_thread == 8:
                        S.nvvm.mbarrier_arrive_cluster(
                            empty_barrier, previous_stage, 1
                        )

                if stage == PIPELINE_STAGES - 1:
                    full_phase = full_phase ^ 1

            S.nvvm.wgmma_wait_group_sync(0)
            S.nvvm.wgmma_fence_aligned()
            final_stage = (size // TILE_K - 1) % PIPELINE_STAGES
            if warpgroup_thread == 0:
                S.nvvm.mbarrier_arrive_cluster(
                    empty_barrier, final_stage, 0
                )
            if warpgroup_thread == 8:
                S.nvvm.mbarrier_arrive_cluster(
                    empty_barrier, final_stage, 1
                )

            for local_m in S.range(2, unroll=True):
                for group in S.range(8, unroll=True):
                    packed = S.full((1, 4), 0, S.i32)
                    for matrix in S.range(4, unroll=True):
                        i = matrix & 1
                        w = 2 * group + matrix // 2
                        packed[0, matrix] = _pack_bf16(
                            accumulators[local_m, 4 * w + 2 * i],
                            accumulators[local_m, 4 * w + 2 * i + 1],
                        )

                    address_matrix = lane // 8
                    address_i = address_matrix & 1
                    address_w = 2 * group + address_matrix // 2
                    box_n = address_w // 8
                    local_address_w = address_w % 8
                    local_row = warp * 16 + address_i * 8 + lane % 8
                    logical_offset = (
                        box_n * 64 * 64
                        + local_row * 64
                        + local_address_w * 8
                    )
                    byte_offset = logical_offset * 2
                    swizzled_offset = (
                        byte_offset ^ ((byte_offset & 0x380) >> 3)
                    ) // 2
                    destination = S.subview(
                        output_shared,
                        (warpgroup, swizzled_offset),
                        (8, 8),
                        (1, 1),
                    )
                    S.nvvm.stmatrix_m8n8_x4_b16(destination, packed[0])

                S.nvvm.fence_proxy_async_shared_cta()
                S.nvvm.named_barrier_sync(1 + warpgroup, WARPGROUP_THREADS)
                if warpgroup_thread == 0:
                    for box_n in S.range(2, unroll=True):
                        output_box_raw = S.subview(
                            output_shared,
                            (warpgroup, box_n * 64 * 64),
                            (1, 64 * 64),
                            (1, 1),
                        )
                        output_box_tensor = S.view(
                            output_box_raw,
                            S.bf16,
                            S.make_layout((64, 64), (64, 1)),
                        )
                        S.nvvm.tma_store(
                            output_box_tensor,
                            c_tma,
                            (
                                cta_n + warpgroup * 128 + box_n * 64,
                                cta_m + local_m * 64,
                            ),
                            predicate=warpgroup_thread == 0,
                        )
                    S.nvvm.cp_async_bulk_commit_group()
                    S.nvvm.cp_async_bulk_wait_group(0, read=True)
                S.nvvm.named_barrier_sync(1 + warpgroup, WARPGROUP_THREADS)

        if warpgroup >= 2:
            S.nvvm.setmaxnreg_dec(24)
            if tid // 32 == 8:
                elected = S.nvvm.elect_sync()
                empty_phase = 1
                for k_tile in S.range(size // TILE_K):
                    stage = k_tile % PIPELINE_STAGES
                    S.nvvm.mbarrier_try_wait_parity(
                        empty_barrier, empty_phase, 10000000, stage
                    )
                    S.nvvm.mbarrier_arrive_expect_tx(
                        full_barrier,
                        (TILE_M + TILE_N) * TILE_K * 2,
                        stage,
                        elected,
                    )

                    a_destination_raw = S.subview(
                        a_shared,
                        (stage, 0),
                        (1, TILE_M * TILE_K),
                        (1, 1),
                    )
                    a_destination = S.view(
                        a_destination_raw,
                        S.bf16,
                        S.make_layout(
                            (TILE_M, 1, TILE_K),
                            (TILE_K, TILE_M * TILE_K, 1),
                        ),
                    )
                    S.nvvm.tma_load(
                        a_destination,
                        a_tma,
                        (0, k_tile, cta_m),
                        full_barrier,
                        mbar_id=stage,
                        predicate=elected,
                        expect_tx=False,
                    )

                    b_destination_raw = S.subview(
                        b_shared,
                        (stage, cluster_rank * 128 * TILE_K),
                        (1, 128 * TILE_K),
                        (1, 1),
                    )
                    b_destination = S.view(
                        b_destination_raw,
                        S.bf16,
                        S.make_layout(
                            (128, 1, TILE_K),
                            (TILE_K, 128 * TILE_K, 1),
                        ),
                    )
                    S.nvvm.tma_load(
                        b_destination,
                        b_tma,
                        (0, k_tile, cta_n + cluster_rank * 128),
                        full_barrier,
                        mbar_id=stage,
                        predicate=elected,
                        expect_tx=False,
                        multicast_mask=3,
                    )
                    if stage == PIPELINE_STAGES - 1:
                        empty_phase = empty_phase ^ 1

                # All four consumers must release the last use of every stage
                # before a persistent CTA starts its next output tile.
                if elected:
                    drain_stage = (size // TILE_K) % PIPELINE_STAGES
                    for _ in S.range(PIPELINE_STAGES, unroll=True):
                        S.nvvm.mbarrier_try_wait_parity(
                            empty_barrier,
                            empty_phase,
                            10000000,
                            drain_stage,
                        )
                        drain_stage += 1
                        if drain_stage == PIPELINE_STAGES:
                            drain_stage = 0
                            empty_phase = empty_phase ^ 1

        S.syncthreads()


@avelang.jit
def _gemm_bf16_192x192_kernel(
    a_ptr: S.Pointer(S.bf16),
    b_ptr: S.Pointer(S.bf16),
    c_ptr: S.Pointer(S.bf16),
    size: S.constexpr,
):
    tid = S.thread_id(0)
    warpgroup = tid // WARPGROUP_THREADS
    warpgroup_thread = tid % WARPGROUP_THREADS
    warp = warpgroup_thread // 32
    lane = warpgroup_thread % 32
    cluster_rank = S.nvvm.cluster_block_rank()

    full_barrier = S.nvvm.mbarrier_create(4)
    empty_barrier = S.nvvm.mbarrier_create(4)
    output_shared = S.make_shared((2, 64 * 96), S.bf16, 128)
    operand_shared = S.make_shared((4, (192 + 192) * 64), S.bf16, 128)

    a_tensor = S.make_tensor(
        a_ptr,
        S.bf16,
        S.make_layout((size, size), (size, 1)),
    )
    b_tensor = S.make_tensor(
        b_ptr,
        S.bf16,
        S.make_layout((size, size), (size, 1)),
    )
    c_tensor = S.make_tensor(
        c_ptr,
        S.bf16,
        S.make_layout((size, size), (size, 1)),
    )
    a_tma = S.nvvm.make_tma_descriptor(
        a_tensor,
        S.make_layout((192, 64), (64, 1)),
        SWIZZLE_128B,
    )
    b_tma = S.nvvm.make_tma_descriptor(
        b_tensor,
        S.make_layout((96, 64), (64, 1)),
        SWIZZLE_128B,
    )
    c_tma = S.nvvm.make_tma_descriptor(
        c_tensor,
        S.make_layout((48, 64), (64, 1)),
        SWIZZLE_128B,
    )

    a_matrix_base = S.view(
        S.subview(operand_shared, (0, 0), (1, 192 * 64), (1, 1)),
        S.bf16,
        S.make_layout((192, 64), (64, 1)),
    )
    b_matrix_base = S.view(
        S.subview(
            operand_shared, (0, 192 * 64), (1, 192 * 64), (1, 1)
        ),
        S.bf16,
        S.make_layout((192, 64), (64, 1)),
    )
    desc_a_base = S.nvvm.make_wgmma_descriptor_bits(
        a_matrix_base, SWIZZLE_128B, 0, 0, 0, 8192, 1024
    )
    desc_b_base = S.nvvm.make_wgmma_descriptor_bits(
        b_matrix_base, SWIZZLE_128B, 0, 0, 0, 8192, 1024
    )

    interior_size = (size // 768) * 768
    cluster_tiles_m = interior_size // 384
    cluster_tiles_n = interior_size // 192
    total_cluster_tiles = cluster_tiles_m * cluster_tiles_n

    worker = S.block_id(1)
    if worker < 5:
        worker = worker
    elif worker < 11:
        worker = (worker - 4) * 8
    elif worker == 11:
        worker = 57
    elif worker < 36:
        group = (worker - 12) // 8
        position = (worker - 12) % 8
        if position == 0:
            worker = 5 + group
        elif position == 1:
            worker = 9 + group
        elif position == 7:
            worker = 58 + group
        else:
            worker = 17 + group + (position - 2) * 8
    elif worker < 64:
        group = (worker - 36) // 7
        position = (worker - 36) % 7
        if position < 6:
            worker = 12 + group + position * 8
        else:
            worker = 61 + group
    elif worker == 64:
        worker = 56
    else:
        worker = 65

    for stage in S.range(4, unroll=True):
        S.nvvm.mbarrier_init(
            full_barrier, stage, count=1, predicate=tid == 0
        )
        S.nvvm.mbarrier_init(
            empty_barrier, stage, count=4, predicate=tid == 0
        )
    S.nvvm.fence_mbarrier_init_release_cluster()
    S.nvvm.cluster_arrive_relaxed()
    S.nvvm.cluster_wait()

    for work in S.range(worker, total_cluster_tiles, S.grid_dim(1)):
        # Match the updated CUDA kernel's eight-column output swizzle.
        full_n_groups = cluster_tiles_n // 8
        full_group_tiles = full_n_groups * cluster_tiles_m * 8
        cluster_tile_m = 0
        cluster_tile_n = 0
        if work < full_group_tiles:
            group_tiles = cluster_tiles_m * 8
            tile_group = work // group_tiles
            in_group = work % group_tiles
            cluster_tile_m = in_group // 8
            cluster_tile_n = tile_group * 8 + in_group % 8
        else:
            tail_n = cluster_tiles_n - full_n_groups * 8
            in_tail = work - full_group_tiles
            cluster_tile_m = in_tail // tail_n
            cluster_tile_n = full_n_groups * 8 + in_tail % tail_n

        cta_m = cluster_tile_m * 384 + cluster_rank * 192
        cta_n = cluster_tile_n * 192

        if warpgroup < 2:
            S.nvvm.setmaxnreg_inc(184)
            accumulators = S.make_local((3, 48), S.f32)
            for tile in S.range(3, unroll=True):
                for element in S.range(48, unroll=True):
                    accumulators[tile, element] = S.convert(0.0, S.f32)

            full_phase = 0
            for k_tile in S.range(size // 64):
                stage = k_tile % 4
                S.nvvm.mbarrier_try_wait_parity(
                    full_barrier, full_phase, 10000000, stage
                )

                for local_n in S.range(3, unroll=True):
                    # Swap the operands: physical M64xN96 accumulators become
                    # logical M96xN64 tiles after the transposed epilogue.
                    desc_first = (
                        desc_b_base + stage * 3072 + local_n * 512
                    )
                    desc_second = desc_a_base + stage * 3072 + warpgroup * 768
                    accumulator = accumulators[local_n]
                    S.nvvm.wgmma_fence_aligned()
                    accumulator = (
                        S.nvvm.wgmma_m64n96k16_f32_bf16_bf16(
                            desc_first,
                            desc_second,
                            accumulator,
                            S.convert(k_tile != 0, S.i32),
                        )
                    )
                    accumulator = (
                        S.nvvm.wgmma_m64n96k16_f32_bf16_bf16(
                            desc_first + 2, desc_second + 2, accumulator, 1
                        )
                    )
                    accumulator = (
                        S.nvvm.wgmma_m64n96k16_f32_bf16_bf16(
                            desc_first + 4, desc_second + 4, accumulator, 1
                        )
                    )
                    accumulator = (
                        S.nvvm.wgmma_m64n96k16_f32_bf16_bf16(
                            desc_first + 6, desc_second + 6, accumulator, 1
                        )
                    )
                    for element in S.range(48, unroll=True):
                        accumulators[local_n, element] = accumulator[element]

                    if local_n == 0:
                        S.nvvm.wgmma_wait_group_sync(0)
                        if k_tile != 0:
                            previous_stage = (k_tile - 1) % 4
                            if warpgroup_thread == 0:
                                S.nvvm.mbarrier_arrive_cluster(
                                    empty_barrier, previous_stage, 0
                                )
                            if warpgroup_thread == 8:
                                S.nvvm.mbarrier_arrive_cluster(
                                    empty_barrier, previous_stage, 1
                                )

                S.nvvm.wgmma_group_sync_aligned()
                if stage == 3:
                    full_phase = full_phase ^ 1

            S.nvvm.wgmma_wait_group_sync(0)
            S.nvvm.wgmma_fence_aligned()
            final_stage = (size // 64 - 1) % 4
            if warpgroup_thread == 0:
                S.nvvm.mbarrier_arrive_cluster(
                    empty_barrier, final_stage, 0
                )
            if warpgroup_thread == 8:
                S.nvvm.mbarrier_arrive_cluster(
                    empty_barrier, final_stage, 1
                )

            logical_tile_m = cta_m + warpgroup * 96
            for local_n in S.range(3, unroll=True):
                logical_tile_n = cta_n + local_n * 64
                full_tile = (
                    logical_tile_m + 96 <= size
                    and logical_tile_n + 64 <= size
                )
                if full_tile:
                    for box_m in S.range(2, unroll=True):
                        for local_group in S.range(3, unroll=True):
                            packed = S.full((1, 4), 0, S.i32)
                            source_group = 3 * box_m + local_group
                            for matrix in S.range(4, unroll=True):
                                i = matrix & 1
                                source_w = 2 * source_group + matrix // 2
                                packed[0, matrix] = _pack_bf16(
                                    accumulators[
                                        local_n, 4 * source_w + 2 * i
                                    ],
                                    accumulators[
                                        local_n, 4 * source_w + 2 * i + 1
                                    ],
                                )

                            address_matrix = lane // 8
                            address_i = address_matrix & 1
                            local_address_w = (
                                2 * local_group + address_matrix // 2
                            )
                            logical_offset = (
                                box_m * 48 * 64
                                + (local_address_w * 8 + lane % 8) * 64
                                + warp * 16
                                + address_i * 8
                            )
                            byte_offset = logical_offset * 2
                            swizzled_offset = (
                                byte_offset ^ ((byte_offset & 0x380) >> 3)
                            ) // 2
                            destination = S.subview(
                                output_shared,
                                (warpgroup, swizzled_offset),
                                (8, 8),
                                (1, 1),
                            )
                            S.nvvm.stmatrix_m8n8_x4_b16_trans(
                                destination, packed[0]
                            )

                        S.nvvm.fence_proxy_async_shared_cta()
                        S.nvvm.named_barrier_sync(
                            1 + warpgroup, WARPGROUP_THREADS
                        )
                        if warpgroup_thread == 0:
                            output_box_raw = S.subview(
                                output_shared,
                                (warpgroup, box_m * 48 * 64),
                                (1, 48 * 64),
                                (1, 1),
                            )
                            output_box = S.view(
                                output_box_raw,
                                S.bf16,
                                S.make_layout(
                                    (48, 64), (64, 1)
                                ),
                            )
                            S.nvvm.tma_store(
                                output_box,
                                c_tma,
                                (
                                    logical_tile_n,
                                    logical_tile_m + box_m * 48,
                                ),
                                predicate=warpgroup_thread == 0,
                            )
                            S.nvvm.cp_async_bulk_commit_group()
                            S.nvvm.cp_async_bulk_wait_group(1, read=True)
                        S.nvvm.named_barrier_sync(
                            1 + warpgroup, WARPGROUP_THREADS
                        )
            if warpgroup_thread == 0:
                S.nvvm.cp_async_bulk_wait_group(0, read=True)
            S.nvvm.wgmma_fence_aligned()

        if warpgroup >= 2:
            S.nvvm.setmaxnreg_dec(24)
            if tid // 32 == 8:
                elected = S.nvvm.elect_sync()
                empty_phase = 1
                for k_tile in S.range(size // 64):
                    stage = k_tile % 4
                    S.nvvm.mbarrier_try_wait_parity(
                        empty_barrier, empty_phase, 10000000, stage
                    )
                    S.nvvm.mbarrier_arrive_expect_tx(
                        full_barrier, (192 + 192) * 64 * 2,
                        stage, elected,
                    )

                    a_destination = S.view(
                        S.subview(
                            operand_shared,
                            (stage, 0),
                            (1, 192 * 64),
                            (1, 1),
                        ),
                        S.bf16,
                        S.make_layout((192, 64), (64, 1)),
                    )
                    S.nvvm.tma_load(
                        a_destination,
                        a_tma,
                        (k_tile * 64, cta_m),
                        full_barrier,
                        mbar_id=stage,
                        predicate=elected,
                        expect_tx=False,
                    )

                    b_offset = 192 * 64 + cluster_rank * 96 * 64
                    b_destination = S.view(
                        S.subview(
                            operand_shared,
                            (stage, b_offset),
                            (1, 96 * 64),
                            (1, 1),
                        ),
                        S.bf16,
                        S.make_layout((96, 64), (64, 1)),
                    )
                    S.nvvm.tma_load(
                        b_destination,
                        b_tma,
                        (k_tile * 64, cta_n + cluster_rank * 96),
                        full_barrier,
                        mbar_id=stage,
                        predicate=elected,
                        expect_tx=False,
                        multicast_mask=3,
                    )
                    if stage == 3:
                        empty_phase = empty_phase ^ 1

                # All four consumers must release the last use of every stage
                # before a persistent CTA starts its next output tile.
                if elected:
                    drain_stage = (size // 64) % 4
                    for _ in S.range(4, unroll=True):
                        S.nvvm.mbarrier_try_wait_parity(
                            empty_barrier,
                            empty_phase,
                            10000000,
                            drain_stage,
                        )
                        drain_stage += 1
                        if drain_stage == 4:
                            drain_stage = 0
                            empty_phase = empty_phase ^ 1

        S.syncthreads()


def gemm_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``a @ b.T`` for one of the supported square BF16 shapes."""

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("a must be a square rank-2 tensor")
    size = a.shape[0]
    if size not in SUPPORTED_SIZES:
        raise ValueError(f"size must be one of {SUPPORTED_SIZES}, got {size}")
    expected_shape = (size, size)
    for name, tensor in (("a", a), ("b", b)):
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must use torch.bfloat16")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be on CUDA")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if b.device != a.device:
        raise ValueError("a and b must be on the same device")

    if out is None:
        out = torch.empty_like(a)
    elif out.shape != expected_shape:
        raise ValueError(f"out must have shape {expected_shape}")
    elif out.dtype != torch.bfloat16 or out.device != a.device:
        raise ValueError("out must match a dtype and device")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")

    if size == 1024:
        _gemm_bf16_64x128_kernel[
            lambda: ((2, 64, 1), (160, 1, 1), (2, 1, 1))
        ](a, b, out, num_warps=5, prefer_l1=True)
        return out

    if size == 16384:
        interior_size = (size // 768) * 768
        cluster_tiles = (interior_size // 384) * (interior_size // 192)
        active_clusters = min(cluster_tiles, PERSISTENT_CLUSTERS)
        _gemm_bf16_192x192_kernel[
            lambda: (
                (CLUSTER_M, active_clusters, 1),
                (THREADS, 1, 1),
                (CLUSTER_M, 1, 1),
            )
        ](a, b, out, size, num_warps=12, prefer_l1=True)
        all_tiles_m = size // (TILE_M * CLUSTER_M)
        all_tiles_n = size // TILE_N
        interior_tiles_m = interior_size // (TILE_M * CLUSTER_M)
        interior_tiles_n = interior_size // TILE_N
        cluster_tiles = (
            interior_tiles_m * (all_tiles_n - interior_tiles_n)
            + (all_tiles_m - interior_tiles_m) * all_tiles_n
        )
    else:
        cluster_tiles = (size // (TILE_M * CLUSTER_M)) * (size // TILE_N)
    active_clusters = min(cluster_tiles, PERSISTENT_CLUSTERS)
    _gemm_bf16_128x256_kernel[
        lambda: (
            (CLUSTER_M, active_clusters, 1),
            (THREADS, 1, 1),
            (CLUSTER_M, 1, 1),
        )
    ](a, b, out, size, num_warps=12, prefer_l1=True)
    return out
