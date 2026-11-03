"""NVIDIA-specific language intrinsics."""


def mma_16x8x16_f16_f16(a, b, c):
    pass


def mma_16x8x16_bf16_f32(a, b, c):
    pass


def mma_16x8x8_f16_f32(a, b, c):
    pass


def shared_address(ptr):
    pass


def load_shared_v2_f32(ptr, offset_bytes):
    pass


def opaque_i32(value):
    pass


def ldmatrix_m8n8_x1_b16(ptr):
    pass


def ldmatrix_m8n8_x1_b16_trans(ptr):
    pass


def ldmatrix_m8n8_x2_b16(ptr):
    pass


def ldmatrix_m8n8_x2_b16_trans(ptr):
    pass


def ldmatrix_m8n8_x4_b16(ptr):
    pass


def ldmatrix_m8n8_x4_b16_trans(ptr):
    pass


def ldmatrix_m8n8_x4_b16_kda_factors(
    smem_base, raw_stage, matrix_row, matrix_col, warp
):
    pass


def stmatrix_m8n8_x1_b16(ptr, data):
    pass


def stmatrix_m8n8_x1_b16_trans(ptr, data):
    pass


def stmatrix_m8n8_x2_b16(ptr, data):
    pass


def stmatrix_m8n8_x2_b16_trans(ptr, data):
    pass


def stmatrix_m8n8_x4_b16(ptr, data):
    pass


def stmatrix_m8n8_x4_b16_trans(ptr, data):
    pass


def wgmma_fence_aligned():
    pass


def wgmma_group_sync_aligned():
    pass


def wgmma_wait_group_sync(group: int):
    pass


def setmaxnreg_inc(register_count: int):
    pass


def setmaxnreg_dec(register_count: int):
    pass


def elect_sync(membermask: int = 0xFFFFFFFF):
    pass


def named_barrier_sync(barrier_id: int, thread_count: int):
    pass


def named_barrier_arrive(barrier_id: int, thread_count: int):
    pass


def syncwarp(mask: int = 0xFFFFFFFF):
    pass


def fence_mbarrier_init_release_cluster():
    pass


def cluster_arrive_relaxed():
    pass


def cluster_wait():
    pass


def cluster_block_rank():
    pass


def griddepcontrol_wait():
    pass


def fence_proxy_async_shared_cta():
    pass


def make_wgmma_descriptor(
    tensor, swizzle_kind: int, l2promo_kind: int, oob_kind: int,
    interleave_kind: int
):
    pass


def make_wgmma_descriptor_bits(
    tensor, swizzle_kind: int, l2promo_kind: int, oob_kind: int,
    interleave_kind: int, leading_byte_offset=None, stride_byte_offset=None
):
    pass


def wgmma_init_result(size: int):
    pass


def wgmma_m64n64k16_f32_bf16_bf16(desc_a, desc_b, acc, scale_d: int):
    pass


def wgmma_m64n96k16_f32_bf16_bf16(desc_a, desc_b, acc, scale_d: int):
    pass


def wgmma_m64n128k16_f32_bf16_bf16(desc_a, desc_b, acc, scale_d: int):
    pass


def wgmma_m64n160k16_f32_bf16_bf16(desc_a, desc_b, acc, scale_d: int):
    pass


def wgmma_m64n176k16_f32_bf16_bf16(desc_a, desc_b, acc, scale_d: int):
    pass


def wgmma_m64n192k16_f32_bf16_bf16(desc_a, desc_b, acc, scale_d: int):
    pass


def wgmma_m64n16k16_f32_bf16_bf16_rs(a, desc_b, acc, scale_d: int):
    pass


def wgmma_m64n64k16_f32_bf16_bf16_rs(a, desc_b, acc, scale_d: int):
    pass


def wgmma_m64n128k16_f32_bf16_bf16_rs(a, desc_b, acc, scale_d: int):
    pass


def wgmma_m64n16k16_f32_bf16_bf16_rs_smem_x8(
    a,
    smem_base,
    stage,
    acc,
    scale_d: int,
    base_byte_offset: int,
    stage_byte_stride: int,
    leading_byte_offset: int,
    descriptor_step: int,
):
    pass


def wgmma_m64n16x2_m64n64x4_k16_f32_bf16_bf16_rs_branch(
    a0,
    a1,
    smem_base,
    stage,
    out_acc0,
    out_acc1,
    state_acc0,
    state_acc1,
    state_acc2,
    state_acc3,
    zero_start,
):
    pass


def wgmma_m64n128k128_f32_bf16_bf16_ss(desc_a, desc_b):
    pass


def wgmma_m64n176k128_f32_bf16_bf16_ss(desc_a, desc_b):
    pass


def wgmma_m64n192k128_f32_bf16_bf16_ss(desc_a, desc_b):
    pass


def wgmma_m64n128k128_f32_bf16_bf16_rs(a0, a1, a2, a3, a4, a5, a6, a7, desc_b, acc):
    pass


def wgmma_m64n128k176_f32_bf16_bf16_rs(
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, desc_b, acc
):
    pass


def wgmma_m64n128k192_f32_bf16_bf16_rs(
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, desc_b, acc
):
    pass


def wgmma_m64n192k32_f32_e4m3_e4m3(desc_a, desc_b, acc, scale_d: int):
    pass


def float_to_fp8(value):
    pass


def floatx2_to_fp8x2(value0, value1):
    pass


def floatx2_to_bf16x2(value0, value1):
    pass


def bf16x2_to_floatx2(value):
    pass


def bf16x2_add(lhs, rhs):
    pass


def bf16x2_sub(lhs, rhs):
    pass


def bf16x2_mul(lhs, rhs):
    pass


def bf16x2_fma(lhs, rhs, acc):
    pass


def fmax(value0, value1):
    pass


def fast_fmax(value0, value1):
    pass


def fast_exp2(value):
    pass


def fast_log2(value):
    pass


def fast_tanh(value):
    pass


def fast_rsqrt(value):
    pass


def fast_rcp(value):
    pass


def fma(a, b, c):
    pass


def fast_fma(a, b, c):
    pass


def wgmma_async(desc_a, desc_b, acc):
    pass


def wgmma_init_accumulator(m: int, n: int):
    pass


def wgmma_store(acc, dst):
    pass


def mbarrier_create(num_barriers: int = 1):
    pass


def mbarrier_init(barrier, mbar_id: int, count: int = 0,
                  predicate: int = True):
    pass


def mbarrier_try_wait_parity(barrier, parity: int, ticks: int, mbar_id: int):
    pass


def mbarrier_arrive(barrier, mbar_id: int):
    pass


def mbarrier_arrive_cluster(barrier, mbar_id: int, cta_id: int):
    pass


def mbarrier_test_wait(barrier, token, mbar_id: int):
    pass


def mbarrier_arrive_expect_tx(barrier, txcount: int, mbar_id: int,
                              predicate: int):
    pass


def cp_async_ca_shared_global(
    dst, src, dst_offset_bytes, src_offset_bytes, size_bytes: int
):
    pass


def cp_async_cg_shared_global(
    dst, src, dst_offset_bytes, src_offset_bytes, size_bytes: int
):
    pass


def cp_async_commit_group():
    pass


def cp_async_wait_group(n: int):
    pass


def cp_async_bulk_commit_group():
    pass


def cp_async_bulk_global_shared_cta(
    dst, src, size, dst_offset_bytes=0, src_offset_bytes=0,
    l2_cache_hint=None, byte_mask=None
):
    pass


def store_global_v4_u32(dst, dst_offset_bytes, value):
    pass


def drain_kda_output(dst, src, pending, tid, head, heads: int):
    pass


def cp_async_bulk_prefetch(src, size, src_offset_bytes=0,
                           l2_cache_hint=None):
    pass


def cp_async_bulk_shared_cluster_global(
    dst, src, mbar, size, dst_offset_bytes=0, src_offset_bytes=0,
    mbar_offset_bytes=0, multicast_mask=None, l2_cache_hint=None
):
    pass


def cp_async_bulk_shared_cluster_shared_cta(
    dst, src, mbar, size, dst_offset_bytes=0, src_offset_bytes=0,
    mbar_offset_bytes=0
):
    pass


def cp_async_bulk_tensor_global_shared_cta(
    desc, src, coords, src_offset_bytes=0, l2_cache_hint=None,
    predicate=None
):
    pass


def cp_async_bulk_tensor_prefetch(
    desc, coords, im2col_offsets=None, l2_cache_hint=None
):
    pass


def cp_async_bulk_tensor_reduce(
    desc, src, coords, red_kind: int, src_offset_bytes=0, l2_cache_hint=None
):
    pass


def cp_async_bulk_tensor_shared_cluster_global(
    dst, desc, coords, mbar, dst_offset_bytes=0, mbar_offset_bytes=0,
    im2col_offsets=None, multicast_mask=None, l2_cache_hint=None,
    predicate=None
):
    pass


def cp_async_bulk_wait_group(group: int, read: bool = False):
    pass


def make_tma_descriptor(tensor, smem_layout, swizzle_kind: int = 0):
    pass


def tma_fence(desc):
    pass


def tma_prefetch_descriptor(desc):
    pass


def tma_load(
    dst, desc, coords, barrier, mbar_id=0, predicate=True,
    multicast_mask=None, expect_tx=True
):
    pass


def tma_store(src, desc, coords, predicate=True):
    pass


def atomic_add(byte_offset, value, tensor):
    pass
