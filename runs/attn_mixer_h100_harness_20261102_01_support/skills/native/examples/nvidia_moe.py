import avelang
import avelang.language as S
import torch

SUPPORTED_SIZES = (1024, 2048, 4096, 8192, 16384)

TILE_M = 128
TILE_N = 192
TILE_K = 128
CLUSTER_M = 2
SCHEDULER_M_BLOCKS = 16
PIPELINE_STAGES = 4
WARPGROUP_THREADS = 128
MATH_WARPGROUPS = 2
THREADS = 384
PERSISTENT_CLUSTERS = 66
SWIZZLE_128B = S.WGMMA_SWIZZLE_128B
FP8_MAX = 448.0


@avelang.jit
def _pack_bf16(first: S.f32, second: S.f32) -> S.i32:
    return S.nvvm.floatx2_to_bf16x2(first, second)


@avelang.jit
def _silu_mul_quant_fp8_kernel(
    input_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.u8),
    scale_ptr: S.Pointer(S.f32),
    max_rows: S.constexpr,
    inter_dim: S.constexpr,
):
    tid = S.thread_id(0)
    local_row = tid // 16
    row_lane = tid % 16
    row = S.block_id(0) * 8 + local_row
    first_column = S.block_id(1) * 128 + row_lane * 8

    input_tensor = S.make_tensor(
        input_ptr,
        S.bf16,
        S.make_layout((max_rows, 2 * inter_dim), (2 * inter_dim, 1)),
    )
    output_tensor = S.make_tensor(
        output_ptr,
        S.u8,
        S.make_layout((max_rows, inter_dim), (inter_dim, 1)),
    )
    scale_tensor = S.make_tensor(
        scale_ptr,
        S.f32,
        S.make_layout((inter_dim // 128, max_rows), (max_rows, 1)),
    )
    values = S.make_local((8,), S.f32)
    maximum = S.convert(0.0, S.f32)
    for item in S.range(8, unroll=True):
        column = first_column + item
        gate = S.convert(input_tensor[row, column], S.f32)
        up = S.convert(input_tensor[row, inter_dim + column], S.f32)
        exp_neg_gate = S.nvvm.fast_exp2(
            -gate * S.convert(1.4426950408889634, S.f32)
        )
        value = gate / (S.convert(1.0, S.f32) + exp_neg_gate) * up
        values[item] = value
        absolute = S.nvvm.fast_fmax(value, -value)
        maximum = S.nvvm.fast_fmax(maximum, absolute)

    maximum = S.nvvm.fast_fmax(maximum, S.shuffle_xor(maximum, 8, 32))
    maximum = S.nvvm.fast_fmax(maximum, S.shuffle_xor(maximum, 4, 32))
    maximum = S.nvvm.fast_fmax(maximum, S.shuffle_xor(maximum, 2, 32))
    maximum = S.nvvm.fast_fmax(maximum, S.shuffle_xor(maximum, 1, 32))
    scale = maximum / S.convert(FP8_MAX, S.f32)
    if maximum == S.convert(0.0, S.f32):
        scale = S.convert(1.0, S.f32)
    if row_lane == 0:
        scale_tensor[S.block_id(1), row] = scale

    inverse_scale = S.convert(FP8_MAX, S.f32) / maximum
    if maximum == S.convert(0.0, S.f32):
        inverse_scale = S.convert(1.0, S.f32)
    for item in S.range(8, unroll=True):
        output_tensor[row, first_column + item] = S.nvvm.float_to_fp8(
            values[item] * inverse_scale
        )


@avelang.jit
def _moe_gemm_fp8_kernel(
    a_ptr: S.Pointer(S.u8),
    b_ptr: S.Pointer(S.u8),
    a_scale_ptr: S.Pointer(S.f32),
    b_scale_ptr: S.Pointer(S.f32),
    block_expert_ptr: S.Pointer(S.i32),
    out_ptr: S.Pointer(S.bf16),
    max_rows: S.constexpr,
    n: S.constexpr,
    k: S.constexpr,
    num_experts: S.constexpr,
):
    tid = S.thread_id(0)
    warpgroup = tid // WARPGROUP_THREADS
    warpgroup_thread = tid % WARPGROUP_THREADS
    warp = warpgroup_thread // 32
    lane = warpgroup_thread % 32
    cluster_rank = S.nvvm.cluster_block_rank()

    num_k_blocks = k // TILE_K
    num_scale_n_blocks = (n + TILE_K - 1) // TILE_K
    num_n_blocks = (n + TILE_N - 1) // TILE_N
    num_m_blocks = max_rows // TILE_M
    pairs_per_group = SCHEDULER_M_BLOCKS // CLUSTER_M
    tasks_per_group = pairs_per_group * num_n_blocks
    total_tasks = (num_m_blocks // CLUSTER_M) * num_n_blocks
    full_groups = num_m_blocks // SCHEDULER_M_BLOCKS
    full_region_tasks = full_groups * tasks_per_group
    tail_pairs = (
        num_m_blocks - full_groups * SCHEDULER_M_BLOCKS
    ) // CLUSTER_M
    if tail_pairs == 0:
        tail_pairs = 1

    # Shared declarations are intentionally ordered as in the CUDA kernel.
    # Avelang emits shared globals in reverse declaration order, leaving the
    # operand arena naturally aligned for 128-byte TMA swizzling.
    full_barrier = S.nvvm.mbarrier_create(PIPELINE_STAGES)
    empty_barrier = S.nvvm.mbarrier_create(PIPELINE_STAGES)
    b_scale_shared = S.make_shared((2, 128), S.f32, 128)
    output_shared = S.make_shared((1, TILE_M * TILE_N), S.bf16, 128)
    a_scale_shared = S.make_shared(
        (PIPELINE_STAGES, TILE_M), S.f32, 128
    )
    b_shared = S.make_shared(
        (PIPELINE_STAGES, TILE_N * TILE_K), S.u8, 128
    )
    a_shared = S.make_shared(
        (PIPELINE_STAGES, TILE_M * TILE_K), S.u8, 128
    )

    # Tensor coordinates are supplied innermost-first to TMA.  Splitting K
    # makes each K128 block a natural tensor-map coordinate.
    a_tensor = S.make_tensor(
        a_ptr,
        S.u8,
        S.make_layout(
            (max_rows, k // TILE_K, TILE_K), (k, TILE_K, 1)
        ),
    )
    b_tensor = S.make_tensor(
        b_ptr,
        S.u8,
        S.make_layout(
            (num_experts * n, k // TILE_K, TILE_K),
            (k, TILE_K, 1),
        ),
    )
    a_scale_tensor = S.make_tensor(
        a_scale_ptr,
        S.f32,
        S.make_layout((k // TILE_K, max_rows), (max_rows, 1)),
    )
    b_scale_tensor = S.make_tensor(
        b_scale_ptr,
        S.f32,
        S.make_layout(
            (num_experts * num_scale_n_blocks * num_k_blocks,), (1,)
        ),
    )
    block_expert_tensor = S.make_tensor(
        block_expert_ptr,
        S.i32,
        S.make_layout((max_rows // TILE_M,), (1,)),
    )
    out_tensor = S.make_tensor(
        out_ptr,
        S.bf16,
        S.make_layout((max_rows, n), (n, 1)),
    )

    a_tma = S.nvvm.make_tma_descriptor(
        a_tensor,
        S.make_layout((128, 1, 128), (128, 16384, 1)),
        SWIZZLE_128B,
    )
    b_tma = S.nvvm.make_tma_descriptor(
        b_tensor,
        S.make_layout((192, 1, 128), (128, 24576, 1)),
        SWIZZLE_128B,
    )
    a_scale_tma = S.nvvm.make_tma_descriptor(
        a_scale_tensor,
        S.make_layout((1, 128), (128, 1)),
        0,
    )
    out_tma = S.nvvm.make_tma_descriptor(
        out_tensor,
        S.make_layout((128, 64), (64, 1)),
        SWIZZLE_128B,
    )

    if tid // 32 == 8:
        elected = S.nvvm.elect_sync()
        if elected:
            S.nvvm.tma_prefetch_descriptor(a_tma)
            S.nvvm.tma_prefetch_descriptor(b_tma)
            S.nvvm.tma_prefetch_descriptor(a_scale_tma)
            S.nvvm.tma_prefetch_descriptor(out_tma)
    S.nvvm.syncwarp()

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
    S.nvvm.griddepcontrol_wait()

    worker = S.block_id(1)
    if warpgroup < MATH_WARPGROUPS:
        S.nvvm.setmaxnreg_inc(232)

        scale_row0 = warp * 16 + lane // 4
        scale_row1 = scale_row0 + 8

        for task in S.range(worker, total_tasks, S.grid_dim(1)):
            block_pair = 0
            n_block = 0
            if task < full_region_tasks:
                group = task // tasks_per_group
                in_group = task % tasks_per_group
                block_pair = (
                    group * pairs_per_group
                    + in_group % pairs_per_group
                )
                n_block = in_group // pairs_per_group
            else:
                in_tail = task - full_region_tasks
                block_pair = (
                    full_groups * pairs_per_group
                    + in_tail % tail_pairs
                )
                n_block = in_tail // tail_pairs
            block = block_pair * CLUSTER_M + cluster_rank
            expert = block_expert_tensor[block]
            row = block * TILE_M
            column = n_block * TILE_N

            first_scale_n = column // TILE_K
            for scale_index in S.range(tid, 2 * num_k_blocks, 256):
                scale_set = scale_index // num_k_blocks
                k_block = scale_index % num_k_blocks
                scale_n = first_scale_n + scale_set
                if scale_n >= num_scale_n_blocks:
                    scale_n = num_scale_n_blocks - 1
                global_scale = (
                    (expert * num_scale_n_blocks + scale_n) * num_k_blocks
                    + k_block
                )
                b_scale_shared[scale_set, k_block] = b_scale_tensor[
                    global_scale
                ]
            S.nvvm.named_barrier_sync(3, 256)

            final_accumulator = S.make_local((96,), S.f32)
            for element in S.range(96, unroll=True):
                final_accumulator[element] = S.convert(0.0, S.f32)

            full_phase = 0
            for k_block in S.range(num_k_blocks):
                stage = k_block % PIPELINE_STAGES
                S.nvvm.mbarrier_try_wait_parity(
                    full_barrier, full_phase, 10000000, stage
                )
                S.nvvm.fence_proxy_async_shared_cta()

                a_matrix = S.view(
                    S.subview(
                        a_shared,
                        (stage, warpgroup * 64 * TILE_K),
                        (1, 64 * TILE_K),
                        (1, 1),
                    ),
                    S.u8,
                    S.make_layout((64, TILE_K), (TILE_K, 1)),
                )
                b_matrix = S.view(
                    S.subview(
                        b_shared,
                        (stage, 0),
                        (1, TILE_N * TILE_K),
                        (1, 1),
                    ),
                    S.u8,
                    S.make_layout((TILE_N, TILE_K), (TILE_K, 1)),
                )
                desc_a = S.nvvm.make_wgmma_descriptor_bits(
                    a_matrix, SWIZZLE_128B, 0, 0, 0, 16, 1024
                )
                desc_b = S.nvvm.make_wgmma_descriptor_bits(
                    b_matrix, SWIZZLE_128B, 0, 0, 0, 16, 1024
                )

                scale_a0 = a_scale_shared[
                    stage, warpgroup * 64 + scale_row0
                ]
                scale_a1 = a_scale_shared[
                    stage, warpgroup * 64 + scale_row1
                ]
                scale_b0 = b_scale_shared[0, k_block]
                scale_b1 = b_scale_shared[1, k_block]
                combined_scale00 = scale_a0 * scale_b0
                combined_scale10 = scale_a1 * scale_b0
                combined_scale01 = scale_a0 * scale_b1
                combined_scale11 = scale_a1 * scale_b1

                partial = S.nvvm.wgmma_init_result(96)
                S.nvvm.wgmma_fence_aligned()
                partial = S.nvvm.wgmma_m64n192k32_f32_e4m3_e4m3(
                    desc_a, desc_b, partial, 0
                )
                partial = S.nvvm.wgmma_m64n192k32_f32_e4m3_e4m3(
                    desc_a + 2, desc_b + 2, partial, 1
                )
                partial = S.nvvm.wgmma_m64n192k32_f32_e4m3_e4m3(
                    desc_a + 4, desc_b + 4, partial, 1
                )
                partial = S.nvvm.wgmma_m64n192k32_f32_e4m3_e4m3(
                    desc_a + 6, desc_b + 6, partial, 1
                )
                S.nvvm.wgmma_group_sync_aligned()
                S.nvvm.wgmma_wait_group_sync(0)

                first_scale_groups = 16
                if n_block % 2 != 0:
                    first_scale_groups = 8
                for accumulator_group in S.range(24, unroll=True):
                    scale0 = combined_scale01
                    scale1 = combined_scale11
                    if accumulator_group < first_scale_groups:
                        scale0 = combined_scale00
                        scale1 = combined_scale10
                    final_accumulator[4 * accumulator_group] = (
                        final_accumulator[4 * accumulator_group]
                        + partial[4 * accumulator_group] * scale0
                    )
                    final_accumulator[4 * accumulator_group + 1] = (
                        final_accumulator[4 * accumulator_group + 1]
                        + partial[4 * accumulator_group + 1] * scale0
                    )
                    final_accumulator[4 * accumulator_group + 2] = (
                        final_accumulator[4 * accumulator_group + 2]
                        + partial[4 * accumulator_group + 2] * scale1
                    )
                    final_accumulator[4 * accumulator_group + 3] = (
                        final_accumulator[4 * accumulator_group + 3]
                        + partial[4 * accumulator_group + 3] * scale1
                    )

                if warpgroup_thread == 0:
                    S.nvvm.mbarrier_arrive_cluster(
                        empty_barrier, stage, 0
                    )
                if warpgroup_thread == 8:
                    S.nvvm.mbarrier_arrive_cluster(
                        empty_barrier, stage, 1
                    )
                if stage == PIPELINE_STAGES - 1:
                    full_phase = full_phase ^ 1

            # Convert the 128x192 CTA result to BF16 in the same swizzled
            # shared layout consumed by three 128x64 TMA stores.
            if tid < 3:
                S.nvvm.cp_async_bulk_wait_group(0, read=True)
            S.nvvm.named_barrier_sync(4, 256)
            for box_n in S.range(3, unroll=True):
                for local_group in S.range(4, unroll=True):
                    packed = S.full((1, 4), 0, S.i32)
                    group_index = box_n * 4 + local_group
                    for matrix in S.range(4, unroll=True):
                        i = matrix & 1
                        w = 2 * group_index + matrix // 2
                        packed[0, matrix] = _pack_bf16(
                            final_accumulator[4 * w + 2 * i],
                            final_accumulator[4 * w + 2 * i + 1],
                        )

                    address_matrix = lane // 8
                    address_i = address_matrix & 1
                    local_address_w = (
                        2 * local_group + address_matrix // 2
                    )
                    local_row = warp * 16 + address_i * 8 + lane % 8
                    logical_byte_offset = (
                        box_n * TILE_M * 64
                        + (warpgroup * 64 + local_row) * 64
                        + local_address_w * 8
                    ) * 2
                    swizzled_offset = (
                        logical_byte_offset
                        ^ ((logical_byte_offset & 0x380) >> 3)
                    ) // 2
                    destination = S.subview(
                        output_shared,
                        (0, swizzled_offset),
                        (8, 8),
                        (1, 1),
                    )
                    S.nvvm.stmatrix_m8n8_x4_b16(
                        destination, packed[0]
                    )

            S.nvvm.fence_proxy_async_shared_cta()
            S.nvvm.named_barrier_sync(4, 256)
            for box_n in S.range(3, unroll=True):
                output_box = S.view(
                    S.subview(
                        output_shared,
                        (0, box_n * TILE_M * 64),
                        (1, TILE_M * 64),
                        (1, 1),
                    ),
                    S.bf16,
                    S.make_layout(
                        (TILE_M, 64), (64, 1)
                    ),
                )
                S.nvvm.tma_store(
                    output_box,
                    out_tma,
                    (column + box_n * 64, row),
                    predicate=tid == box_n,
                )
            if tid < 3:
                S.nvvm.cp_async_bulk_commit_group()
            S.nvvm.named_barrier_sync(4, 256)

    if warpgroup >= MATH_WARPGROUPS:
        S.nvvm.setmaxnreg_dec(40)
        if tid // 32 == 8:
            elected = S.nvvm.elect_sync()
            empty_phase = 1
            for task in S.range(worker, total_tasks, S.grid_dim(1)):
                block_pair = 0
                n_block = 0
                if task < full_region_tasks:
                    group = task // tasks_per_group
                    in_group = task % tasks_per_group
                    block_pair = (
                        group * pairs_per_group
                        + in_group % pairs_per_group
                    )
                    n_block = in_group // pairs_per_group
                else:
                    in_tail = task - full_region_tasks
                    block_pair = (
                        full_groups * pairs_per_group
                        + in_tail % tail_pairs
                    )
                    n_block = in_tail // tail_pairs
                block = block_pair * CLUSTER_M + cluster_rank
                peer_block = block_pair * CLUSTER_M + (cluster_rank ^ 1)
                expert = block_expert_tensor[block]
                peer_expert = block_expert_tensor[peer_block]
                multicast_b = expert == peer_expert
                row = block * TILE_M
                column = n_block * TILE_N

                for k_block in S.range(num_k_blocks):
                    stage = k_block % PIPELINE_STAGES
                    S.nvvm.mbarrier_try_wait_parity(
                        empty_barrier, empty_phase, 10000000, stage
                    )
                    S.nvvm.mbarrier_arrive_expect_tx(
                        full_barrier,
                        (TILE_M + TILE_N) * TILE_K + TILE_M * 4,
                        stage,
                        elected,
                    )

                    a_destination = S.view(
                        S.subview(
                            a_shared,
                            (stage, 0),
                            (1, TILE_M * TILE_K),
                            (1, 1),
                        ),
                        S.u8,
                        S.make_layout(
                            (TILE_M, 1, TILE_K),
                            (TILE_K, TILE_M * TILE_K, 1),
                        ),
                    )
                    S.nvvm.tma_load(
                        a_destination,
                        a_tma,
                        (0, k_block, row),
                        full_barrier,
                        mbar_id=stage,
                        predicate=elected,
                        expect_tx=False,
                    )

                    scale_destination = S.view(
                        S.subview(
                            a_scale_shared,
                            (stage, 0),
                            (1, TILE_M),
                            (1, 1),
                        ),
                        S.f32,
                        S.make_layout((1, TILE_M), (TILE_M, 1)),
                    )
                    S.nvvm.tma_load(
                        scale_destination,
                        a_scale_tma,
                        (row, k_block),
                        full_barrier,
                        mbar_id=stage,
                        predicate=elected,
                        expect_tx=False,
                    )

                    b_destination = S.view(
                        S.subview(
                            b_shared,
                            (stage, 0),
                            (1, TILE_N * TILE_K),
                            (1, 1),
                        ),
                        S.u8,
                        S.make_layout(
                            (TILE_N, 1, TILE_K),
                            (TILE_K, TILE_N * TILE_K, 1),
                        ),
                    )
                    if multicast_b:
                        if cluster_rank == 0:
                            S.nvvm.tma_load(
                                b_destination,
                                b_tma,
                                (0, k_block, expert * n + column),
                                full_barrier,
                                mbar_id=stage,
                                predicate=elected,
                                expect_tx=False,
                                multicast_mask=3,
                            )
                    else:
                        S.nvvm.tma_load(
                            b_destination,
                            b_tma,
                            (0, k_block, expert * n + column),
                            full_barrier,
                            mbar_id=stage,
                            predicate=elected,
                            expect_tx=False,
                        )
                    if stage == PIPELINE_STAGES - 1:
                        empty_phase = empty_phase ^ 1

            for drain_stage in S.range(PIPELINE_STAGES, unroll=True):
                S.nvvm.mbarrier_try_wait_parity(
                    empty_barrier,
                    empty_phase,
                    10000000,
                    drain_stage,
                )

    if tid < 3:
        S.nvvm.cp_async_bulk_wait_group(0, read=True)
    S.syncthreads()
    S.nvvm.cluster_arrive_relaxed()
    S.nvvm.cluster_wait()


def moe_gemm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    block_expert_ids: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a grouped block-scaled FP8 GEMM.

    ``a`` has shape ``[M, K]`` and ``b`` has shape ``[E, N, K]``.
    ``a_scale`` is K-block-major with shape
    ``[K / 128, M]``; ``b_scale`` has shape ``[E, N / 128, K / 128]``.
    ``block_expert_ids[i]`` selects the expert for A rows
    ``[128 * i, 128 * (i + 1))``.
    """

    if a.ndim != 2:
        raise ValueError("a must be a rank-2 tensor")
    max_rows, k = a.shape
    if max_rows % (TILE_M * CLUSTER_M) != 0:
        raise ValueError("a rows must be divisible by 256")
    if k % TILE_K != 0:
        raise ValueError("a columns must be divisible by 128")
    if a.dtype != torch.float8_e4m3fn:
        raise ValueError("a must use torch.float8_e4m3fn")
    if a.device.type != "cuda" or not a.is_contiguous():
        raise ValueError("a must be a contiguous CUDA tensor")

    if b.ndim != 3 or b.shape[2] != k:
        raise ValueError(f"b must have shape [experts, n, {k}]")
    if b.dtype != torch.float8_e4m3fn:
        raise ValueError("b must use torch.float8_e4m3fn")
    if b.device != a.device or not b.is_contiguous():
        raise ValueError("b must be contiguous and on the same device as a")
    num_experts = b.shape[0]
    n = b.shape[1]
    if num_experts < 1:
        raise ValueError("b must contain at least one expert")
    if n % TILE_K != 0:
        raise ValueError("b rows must be divisible by 128")

    k_scale_blocks = k // TILE_K
    n_scale_blocks = n // TILE_K
    expected_a_scale = (k_scale_blocks, max_rows)
    expected_b_scale = (num_experts, n_scale_blocks, k_scale_blocks)
    if tuple(a_scale.shape) != expected_a_scale:
        raise ValueError(f"a_scale must have shape {expected_a_scale}")
    if tuple(b_scale.shape) != expected_b_scale:
        raise ValueError(f"b_scale must have shape {expected_b_scale}")
    for name, tensor in (("a_scale", a_scale), ("b_scale", b_scale)):
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must use torch.float32")
        if tensor.device != a.device or not tensor.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous and on the same device as a"
            )

    expected_experts = (max_rows // TILE_M,)
    if tuple(block_expert_ids.shape) != expected_experts:
        raise ValueError(
            f"block_expert_ids must have shape {expected_experts}"
        )
    if block_expert_ids.dtype != torch.int32:
        raise ValueError("block_expert_ids must use torch.int32")
    if block_expert_ids.device != a.device or not block_expert_ids.is_contiguous():
        raise ValueError(
            "block_expert_ids must be contiguous and on the same device as a"
        )

    if out is None:
        out = torch.empty((max_rows, n), dtype=torch.bfloat16, device=a.device)
    elif tuple(out.shape) != (max_rows, n):
        raise ValueError(f"out must have shape {(max_rows, n)}")
    elif out.dtype != torch.bfloat16 or out.device != a.device:
        raise ValueError("out must use BF16 and be on the same device as a")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")

    total_tasks = (max_rows // (TILE_M * CLUSTER_M)) * (
        (n + TILE_N - 1) // TILE_N
    )
    active_clusters = min(total_tasks, PERSISTENT_CLUSTERS)
    _moe_gemm_fp8_kernel[
        lambda: (
            (CLUSTER_M, active_clusters, 1),
            (THREADS, 1, 1),
            (CLUSTER_M, 1, 1),
        )
    ](
        a.view(torch.uint8),
        b.view(torch.uint8),
        a_scale,
        b_scale,
        block_expert_ids,
        out,
        max_rows,
        n,
        k,
        num_experts,
        num_warps=12,
        prefer_l1=False,
    )
    return out


def silu_mul_quant_fp8(
    input: torch.Tensor,
    output: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse SiLU, gate/up multiplication, and K128 E4M3 quantization."""

    if input.ndim != 2 or input.shape[1] % 256 != 0:
        raise ValueError("input must have shape [rows, 2 * inter_dim]")
    max_rows, doubled_inter_dim = input.shape
    inter_dim = doubled_inter_dim // 2
    if max_rows % 8 != 0:
        raise ValueError("input rows must be divisible by 8")
    if input.dtype != torch.bfloat16 or input.device.type != "cuda":
        raise ValueError("input must be a CUDA BF16 tensor")
    if not input.is_contiguous():
        raise ValueError("input must be contiguous")
    if tuple(output.shape) != (max_rows, inter_dim):
        raise ValueError(f"output must have shape {(max_rows, inter_dim)}")
    if output.dtype != torch.float8_e4m3fn or output.device != input.device:
        raise ValueError("output must be E4M3 on the input device")
    expected_scale_shape = (inter_dim // 128, max_rows)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(f"scale must have shape {expected_scale_shape}")
    if scale.dtype != torch.float32 or scale.device != input.device:
        raise ValueError("scale must be FP32 on the input device")
    if not output.is_contiguous() or not scale.is_contiguous():
        raise ValueError("output and scale must be contiguous")

    _silu_mul_quant_fp8_kernel[
        lambda: (
            (max_rows // 8, inter_dim // 128, 1),
            (128, 1, 1),
        )
    ](
        input,
        output.view(torch.uint8),
        scale,
        max_rows,
        inter_dim,
        num_warps=4,
    )
    return output, scale
