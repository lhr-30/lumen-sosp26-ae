"""attention qkv 416x5120x2048 (6 calls/forward): its own copy of gemm_peak/cute_gemm.py, tuned for this shape only.
Stream-K was slower (variants/qkv_streamk.py, see ROOT_CAUSE_ZH.md)."""
# Derived from NVIDIA CUTLASS's Hopper CuTe DSL GEMM example (via gemm_fusion/cute_fusion/worker_gemm.py).
"""Persistent BF16 GEMM D[M,N] = A[M,K] @ B[N,K]^T for SM90, same semantics as deep_gemm.bf16_gemm_nt.

Swap AB: each task computes a tile_n-channel x tile_m-token block of B @ A^T, so the small token
dimension M sits in the WGMMA N mode and is not padded to 64. FP32 accumulation, BF16 RN output.
Warp 0 is the TMA warp: it initializes the stage barriers and starts loading without waiting for the
rest of the CTA; warp-groups 1..tile_n//64 run WGMMA. The stage ring is shared across tasks so the
next task's loads overlap the current epilogue. Tasks run token-tile fastest so CTAs that share a
weight tile run at the same time. Epilogue: each WGMMA warp-group writes its 64-channel slice with
transposed stmatrix into a 128B-swizzled token-major smem tile and stores it with one TMA store; the
warp-groups do not wait for each other.
"""
from dataclasses import dataclass

import torch
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90
from cutlass import BFloat16, Float32, Int32
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cuda.bindings import driver as cuda


@dsl_user_op
def pack_bf16x2(lo, hi, *, loc=None, ip=None):
    """Both values rounded to bf16 by one cvt.rn.bf16x2.f32: lo in bits 0..15, hi in bits 16..31."""
    return Int32(llvm.inline_asm(T.i32(), [Float32(hi).ir_value(loc=loc, ip=ip), Float32(lo).ir_value(loc=loc, ip=ip)],
                                 "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f", has_side_effects=False,
                                 is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


def _asm_i32(asm, loc, ip):
    return Int32(llvm.inline_asm(T.i32(), [], asm, "=r", has_side_effects=True,
                                 is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def clock32(*, loc=None, ip=None):
    """SM cycle counter (32-bit, wraps); take differences modulo 2^32."""
    return _asm_i32("mov.u32 $0, %clock;", loc, ip)


@dsl_user_op
def smid(*, loc=None, ip=None):
    return _asm_i32("mov.u32 $0, %smid;", loc, ip)


@dsl_user_op
def globaltimer(*, loc=None, ip=None):
    """Nanoseconds, comparable across SMs."""
    return cutlass.Int64(llvm.inline_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", has_side_effects=True,
                                         is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


# Trace fields, one row per math warp-group (thread 0 of it) and a last row for the TMA warp (lane 0).
# Clocks are raw 32-bit %clock values; *_SUM fields are cycle sums. Math rows: ENTRY, SETUP, TASK_START, FIRST_READY,
# WAIT_SUM (consumer_wait), PENDING_SUM (wait_group), LOOP_END, DRAIN, STAGED, STAGED_BAR, STORES_ISSUED, TASK_END.
# TMA row: ENTRY, SETUP, TASK_START, PARTITIONED, WAIT_SUM (producer_acquire), FIRST_READY (first k-block issued),
# LOOP_END.
ENTRY, SETUP, TASK_START, FIRST_READY, WAIT_SUM, PENDING_SUM, LOOP_END, DRAIN, STAGED, STAGED_BAR, STORES_ISSUED, \
    TASK_END, GLOBAL_START, GLOBAL_END, SMID, PARTITIONED = range(16)   # PARTITIONED: TMA row, after tma_partition
TRACE_FIELDS = 16


@dataclass
class StridedTasks:
    """Default task source: CTA c runs tasks c, c + grid, ... ; a fused kernel passes its own task table.

    start/stop/step bound the loop position; task(position) is the linear task id (n_tile * m_tiles + m_tile).
    (The GEMM epilogue has its own `for i` loops, so the position has a distinct name.)
    The hooks run on the TMA warp before a task's loads, on each math warp-group before its epilogue,
    and on the storing thread after its TMA store was committed.
    """

    cta: Int32
    grid: cutlass.Constexpr
    tasks: cutlass.Constexpr

    def start(self):
        return Int32(self.cta)

    def stop(self):
        return self.tasks

    def step(self):
        return self.grid

    def task(self, i):
        return i

    def before_load(self, i):
        pass

    def before_epilogue(self, i):
        pass

    @cute.jit
    def after_store(self, i):
        cute.arch.cp_async_bulk_wait_group(0, read=True)    # smem must stay valid until the store read it


@dataclass
class PdlTasks(StridedTasks):
    """Strided tasks of a kernel launched with programmatic dependent launch: the TMA warp loads every stage's weight
    tile before griddepcontrol.wait (weights never depend on the previous kernel), the activations after it; the math
    warp-groups wait as well before their epilogue reads or writes global memory."""

    def activations_ready(self, p):
        cute.arch.griddepcontrol_wait()

    def before_epilogue(self, i):
        cute.arch.griddepcontrol_wait()


class GemmNT:
    """tile_n: output channels per task (B rows, multiple of 64); tile_m: tokens per task (A rows).

    Default 192x104 for 416x6144x2048: 128 tasks in one wave on 132 SMs, no token padding
    (gemm_peak/runs/s4-tiles-2, -3; 128x208 leaves 36 SMs idle).
    trace=1 compiles register-only clock stamps at phase edges, written once per task into trace[task, row, field];
    trace=2 also sums the mainloop waits per k-block (more intrusive).
    """

    def __init__(self, m, n, k, *, sms, tile_n=128, tile_m=144, block_k=64, stages=4, wait=1, trace=0,
                 barrier_base=1, multi_task=None, pdl=False, cluster_n=1):
        # Edge tiles may overhang M or N: TMA zero-fills the loads and clips the stores there.
        if k % block_k or tile_n % 64 or tile_m % 8 or tile_m > 256:
            raise ValueError('require k % block_k == 0, tile_n % 64 == 0, tile_m % 8 == 0, tile_m <= 256')
        if not 0 <= wait < stages:
            raise ValueError('require 0 <= wait < stages')
        self.m, self.n, self.k = m, n, k
        self.pdl = pdl                                   # standalone launch with programmatic dependent launch
        # cluster_n consecutive CTAs (the token tiles of one channel tile: tasks run token tile fastest) form a cluster
        # and share the weight tile, each loading 1 / cluster_n of it by TMA multicast.
        self.cluster_n = cluster_n
        if -(-m // tile_m) % cluster_n:
            raise ValueError('cluster_n must divide the token tile count')
        self.tile_n, self.tile_m, self.bk, self.stages, self.wait = tile_n, tile_m, block_k, stages, wait
        self.trace = trace
        # Named barriers (0 is sync_threads): init, then one per math warp-group's epilogue.
        self.init_barrier, self.epilogue_barrier = barrier_base, barrier_base + 1
        self.math_groups = tile_n // 64
        self.math_threads = 128 * self.math_groups
        self.threads = self.math_threads + 128
        self.m_tiles = -(-m // tile_m)
        self.tasks = -(-n // tile_n) * self.m_tiles
        self.grid = min(self.tasks, sms)
        # Some CTA runs several tasks and reuses its staging tile; a task table may decide this instead of the grid.
        self.multi_task = self.tasks > self.grid if multi_task is None else multi_task
        # A tile_m % 16 == 8 token tile is loaded as load_m rows (TMA needs it; the extra rows are the next tile's
        # tokens or zero fill) while WGMMA reads only tile_m; transposed stmatrix.x4 would overrun it, so use x2.
        self.load_m = -(-tile_m // 16) * 16
        self.matrices = 4 if tile_m == self.load_m else 2
        self.stage_elements = (tile_n + self.load_m) * block_k * stages
        self.staging_elements = tile_m * 64              # per warp-group: tile_m tokens x 64 channels, 128 B rows
        # One math warp-group and one task per CTA: when it reaches the epilogue every WGMMA has drained and the
        # producer has nothing left to load, so the staging tile can reuse the stage buffers (one more stage fits).
        self.staging_offset = 0 if self.math_groups == 1 and not self.multi_task else self.stage_elements
        self.pool_elements = max(self.stage_elements, self.staging_offset + self.math_groups * self.staging_elements)
        self.smem_bytes = 1024 + self.pool_elements * 2
        if self.smem_bytes > 232448:
            raise ValueError(f'shared memory {self.smem_bytes} B exceeds the H100 limit')

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, d: cute.Tensor, trace: cute.Tensor, stream: cuda.CUstream):
        prepared = self.prepare(a, b, d)
        self.kernel(*prepared, trace).launch(
            grid=(self.grid, 1, 1), block=(self.threads, 1, 1), stream=stream, use_pdl=self.pdl,
            **({"cluster": (self.cluster_n, 1, 1)} if self.cluster_n > 1 else {}))

    @cute.jit
    def prepare(self, a: cute.Tensor, b: cute.Tensor, d: cute.Tensor):
        major = cute.nvgpu.warpgroup.OperandMajorMode.K
        mma = sm90.make_trivial_tiled_mma(BFloat16, BFloat16, major, major, Float32,
                                          (self.math_groups, 1, 1), tiler_mn=(64, self.tile_m))
        atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90.get_smem_layout_atom(utils.LayoutEnum.ROW_MAJOR, BFloat16, self.bk), BFloat16)
        lw = cute.tile_to_shape(atom, (self.tile_n, self.bk, self.stages), order=(0, 1, 2))
        lx = cute.tile_to_shape(atom, (self.load_m, self.bk, self.stages), order=(0, 1, 2))
        load = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        if cutlass.const_expr(self.cluster_n > 1):
            tma_w, gw = cute.nvgpu.cpasync.make_tiled_tma_atom(
                cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(), b, cute.slice_(lw, (None, None, 0)),
                (self.tile_n, self.bk), num_multicast=self.cluster_n)
        else:
            tma_w, gw = cute.nvgpu.cpasync.make_tiled_tma_atom(load, b, cute.slice_(lw, (None, None, 0)), (self.tile_n, self.bk))
        tma_x, gx = cute.nvgpu.cpasync.make_tiled_tma_atom(load, a, cute.slice_(lx, (None, None, 0)), (self.load_m, self.bk))
        # Epilogue: one warp-group's accumulator slice and its (tokens, 64 channels) row-major D tile.
        mma_wg = sm90.make_trivial_tiled_mma(BFloat16, BFloat16, major, major, Float32, (1, 1, 1), tiler_mn=(64, self.tile_m))
        ld = cute.tile_to_shape(cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90.get_smem_layout_atom(utils.LayoutEnum.ROW_MAJOR, BFloat16, 64), BFloat16), (self.tile_m, 64), order=(0, 1))
        tma_d, gd = cute.nvgpu.cpasync.make_tiled_tma_atom(cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), d, ld, (self.tile_m, 64))
        return tma_w, gw, tma_x, gx, lw, lx, mma, tma_d, gd, ld, mma_wg

    @cute.kernel
    def kernel(self, tma_w: cute.CopyAtom, gw: cute.Tensor, tma_x: cute.CopyAtom, gx: cute.Tensor,
               lw: cute.ComposedLayout, lx: cute.ComposedLayout, mma: cute.TiledMma, tma_d: cute.CopyAtom,
               gd: cute.Tensor, ld: cute.ComposedLayout, mma_wg: cute.TiledMma, trace: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
            self.run(tid, cta, tma_w, gw, tma_x, gx, lw, lx, mma, tma_d, gd, ld, mma_wg, trace,
                     tasks=PdlTasks(cta, self.grid, self.tasks))
        else:
            self.run(tid, cta, tma_w, gw, tma_x, gx, lw, lx, mma, tma_d, gd, ld, mma_wg, trace)

    @cute.jit
    def run(self, tid: Int32, cta: Int32, tma_w: cute.CopyAtom, gw: cute.Tensor,
            tma_x: cute.CopyAtom, gx: cute.Tensor, lw: cute.ComposedLayout, lx: cute.ComposedLayout,
            mma: cute.TiledMma, tma_d: cute.CopyAtom, gd: cute.Tensor, ld: cute.ComposedLayout,
            mma_wg: cute.TiledMma, trace: cute.Tensor, tasks=None, smem_offset: cutlass.Constexpr = 0):
        source = StridedTasks(cta, self.grid, self.tasks) if cutlass.const_expr(tasks is None) else tasks
        entry, global_start = Int32(0), cutlass.Int64(0)
        if cutlass.const_expr(self.trace):
            entry, global_start = clock32(), globaltimer()
        warp = cute.arch.make_warp_uniform(tid // 32)
        group = cute.arch.make_warp_uniform(tid // 128)
        if warp == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_w)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_x)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_d)
        smem = utils.SmemAllocator()
        if cutlass.const_expr(smem_offset):
            smem.allocate(smem_offset)
        bars = smem.allocate_array(cutlass.Int64, self.stages * 2)
        pool = smem.allocate_tensor(BFloat16, cute.make_layout(self.pool_elements), byte_alignment=1024).iterator
        sw = cute.make_tensor(cute.recast_ptr(pool, lw.inner), lw.outer)
        sx = cute.make_tensor(cute.recast_ptr(pool + self.tile_n * self.bk * self.stages, lx.inner), lx.outer)
        sx_mma = sx
        if cutlass.const_expr(self.load_m != self.tile_m):
            rows = cute.make_layout((self.tile_m, self.bk, self.stages), stride=(1, self.load_m, self.load_m * self.bk))
            sx_mma = cute.make_tensor(sx.iterator, cute.composition(lx.outer, rows))
        cn = self.cluster_n
        rn, w_mask = 0, None
        if cutlass.const_expr(cn > 1):
            rn = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            w_mask = cute.make_layout_image_mask(cute.make_layout((1, cn, 1)), (0, rn, 0), mode=1)
        # A stage is free only after the consumers of every CTA its weight slice was multicast to have released it.
        pipe = pipeline.PipelineTmaAsync.create(barrier_storage=bars, num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.math_threads // 32 * cn),
            tx_count=(self.tile_n + self.load_m) * self.bk * 2,
            cta_layout_vmnk=cute.make_layout((1, 1, cn, 1)), defer_sync=True)
        if cutlass.const_expr(cn > 1):
            # Every barrier of the cluster must be initialized before any multicast or remote arrive.
            pipeline.pipeline_init_arrive(cluster_shape_mn=(1, cn), is_relaxed=True)
            pipeline.pipeline_init_wait(cluster_shape_mn=(1, cn))
        k_tiles = self.k // self.bk
        setup = Int32(0)
        # Warp 0 initialized the stage barriers (PipelineTmaAsync.create), so it fences and starts loading at once;
        # the math warps wait on a named barrier for the initialization instead of a CTA-wide sync.
        init_threads = 32 + self.math_threads
        if group == 0:
            if warp == 0:
                if cutlass.const_expr(cn == 1):
                    cute.arch.mbarrier_init_fence()
                    cute.arch.barrier_arrive(barrier_id=self.init_barrier, number_of_threads=init_threads)
                if cutlass.const_expr(self.trace):
                    setup = clock32()
                state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
                position = source.start()
                while position < source.stop():
                    task = source.task(position)
                    source.before_load(position)
                    task_start, wait_sum, first_ready = Int32(0), Int32(0), Int32(0)
                    if cutlass.const_expr(self.trace):
                        task_start = clock32()
                    tw = cute.local_tile(gw, (self.tile_n, self.bk), (task // self.m_tiles, None))
                    tx = cute.local_tile(cute.domain_offset(((task % self.m_tiles) * self.tile_m, 0), gx),
                                         (self.load_m, self.bk), (0, None))
                    ssw, tgw = cute.nvgpu.cpasync.tma_partition(tma_w, rn, cute.make_layout(cn),
                                                                cute.group_modes(sw, 0, 2), cute.group_modes(tw, 0, 2))
                    ssx, tgx = cute.nvgpu.cpasync.tma_partition(tma_x, 0, cute.make_layout(1),
                                                                cute.group_modes(sx, 0, 2), cute.group_modes(tx, 0, 2))
                    partitioned = Int32(0)
                    if cutlass.const_expr(self.trace):
                        partitioned = clock32()
                    first_kt = 0
                    if cutlass.const_expr(hasattr(source, "activations_ready")):
                        # Weights first: every stage gets its weight tile at once, the activations follow once ready.
                        assert not self.multi_task
                        first_kt = self.stages
                        filled = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
                        for kt in cutlass.range_constexpr(self.stages):
                            pipe.producer_acquire(state)
                            if cutlass.const_expr(cn > 1):
                                cute.copy(tma_w, tgw[None, kt], ssw[None, state.index],
                                          tma_bar_ptr=pipe.producer_get_barrier(state), mcast_mask=w_mask)
                            else:
                                cute.copy(tma_w, tgw[None, kt], ssw[None, state.index],
                                          tma_bar_ptr=pipe.producer_get_barrier(state))
                            state.advance()
                        source.activations_ready(position)
                        for kt in cutlass.range_constexpr(self.stages):
                            cute.copy(tma_x, tgx[None, kt], ssx[None, filled.index],
                                      tma_bar_ptr=pipe.producer_get_barrier(filled))
                            filled.advance()
                    for kt in cutlass.range(first_kt, k_tiles, unroll=1):
                        t0 = Int32(0)
                        if cutlass.const_expr(self.trace > 1):
                            t0 = clock32()
                        pipe.producer_acquire(state)
                        if cutlass.const_expr(self.trace > 1):
                            wait_sum += clock32() - t0
                        bar = pipe.producer_get_barrier(state)
                        if cutlass.const_expr(cn > 1):
                            cute.copy(tma_w, tgw[None, kt], ssw[None, state.index], tma_bar_ptr=bar, mcast_mask=w_mask)
                        else:
                            cute.copy(tma_w, tgw[None, kt], ssw[None, state.index], tma_bar_ptr=bar)
                        cute.copy(tma_x, tgx[None, kt], ssx[None, state.index], tma_bar_ptr=bar)
                        state.advance()
                        if cutlass.const_expr(self.trace):
                            if kt == 0:
                                first_ready = clock32()
                    if cutlass.const_expr(self.trace):
                        loop_end = clock32()
                        if tid % 32 == 0:
                            rec = trace[task, self.math_groups, None]
                            rec[ENTRY], rec[SETUP], rec[TASK_START] = cutlass.Int64(entry), cutlass.Int64(setup), cutlass.Int64(task_start)
                            rec[PARTITIONED] = cutlass.Int64(partitioned)
                            rec[WAIT_SUM], rec[FIRST_READY], rec[LOOP_END] = cutlass.Int64(wait_sum), cutlass.Int64(first_ready), cutlass.Int64(loop_end)
                            rec[GLOBAL_START], rec[GLOBAL_END], rec[SMID] = global_start, globaltimer(), cutlass.Int64(smid())
                    position += source.step()
                pipe.producer_tail(state)
        else:
            if cutlass.const_expr(cn == 1):
                cute.arch.barrier(barrier_id=self.init_barrier, number_of_threads=init_threads)
            if cutlass.const_expr(self.trace):
                setup = clock32()
            wg = group - 1
            thr = mma.get_slice(wg * 128)
            rw, rx = mma.make_fragment_A(thr.partition_A(sw)), mma.make_fragment_B(thr.partition_B(sx_mma))
            accum = cute.make_fragment(mma.partition_shape_C((self.tile_n, self.tile_m)), Float32)
            values = cute.make_rmem_tensor(accum.shape, BFloat16)
            packed = cute.recast_tensor(values, Int32)
            # This warp-group's D tile in (token, channel) order for the TMA store, and its (channel, token) view
            # in accumulator order for the transposed stmatrix.
            sd = cute.make_tensor(cute.recast_ptr(pool + self.staging_offset + cute.assume(wg * self.staging_elements, divby=512),
                                                  ld.inner), ld.outer)
            sd_mma = cute.make_tensor(sd.iterator, cute.select(ld.outer, mode=[1, 0]))
            store = cute.make_tiled_copy_C(cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(True, self.matrices), BFloat16), mma_wg)
            thread_store = store.get_slice(tid % 128)
            wg_barrier = self.epilogue_barrier + wg
            read = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            release = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            position = source.start()
            while position < source.stop():
                task = source.task(position)
                task_start, wait_sum, pending_sum, first_ready = Int32(0), Int32(0), Int32(0), Int32(0)
                if cutlass.const_expr(self.trace):
                    task_start = clock32()
                mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                for kt in cutlass.range(k_tiles, unroll=1):
                    t0 = Int32(0)
                    if cutlass.const_expr(self.trace > 1):
                        t0 = clock32()
                    pipe.consumer_wait(read)
                    if cutlass.const_expr(self.trace > 1):
                        t1 = clock32()
                        wait_sum += t1 - t0
                        if kt == 0:
                            first_ready = t1
                    elif cutlass.const_expr(self.trace):
                        if kt == 0:
                            first_ready = clock32()
                    cute.nvgpu.warpgroup.fence()
                    for inner in cutlass.range_constexpr(self.bk // 16):
                        cute.gemm(mma, accum, rw[None, None, inner, read.index], rx[None, None, inner, read.index], accum)
                        mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                    cute.nvgpu.warpgroup.commit_group()
                    if kt >= self.wait:
                        if cutlass.const_expr(self.trace > 1):
                            t2 = clock32()
                            cute.nvgpu.warpgroup.wait_group(self.wait)
                            pending_sum += clock32() - t2
                        else:
                            cute.nvgpu.warpgroup.wait_group(self.wait)
                        pipe.consumer_release(release)
                        release.advance()
                    read.advance()
                loop_end = Int32(0)
                if cutlass.const_expr(self.trace):
                    loop_end = clock32()
                cute.nvgpu.warpgroup.wait_group(0)
                drain = Int32(0)
                if cutlass.const_expr(self.trace):
                    drain = clock32()
                for _ in cutlass.range_constexpr(self.wait):
                    pipe.consumer_release(release)
                    release.advance()

                source.before_epilogue(position)
                # Scalar BF16 conversion triggers ptxas C7514 and serializes WGMMA; convert in pairs.
                for i in cutlass.range_constexpr(cute.size(accum) // 2):
                    packed[i] = pack_bf16x2(accum[2 * i], accum[2 * i + 1])
                cute.copy(store, thread_store.retile(values), thread_store.partition_D(sd_mma))
                staged = Int32(0)
                if cutlass.const_expr(self.trace):
                    staged = clock32()
                cute.arch.fence_proxy("async.shared", space="cta")      # stmatrix writes before the TMA store reads
                cute.arch.barrier(barrier_id=wg_barrier, number_of_threads=128)
                staged_bar, stores_issued = Int32(0), Int32(0)
                if cutlass.const_expr(self.trace):
                    staged_bar = clock32()
                if warp % 4 == 0:
                    gtile = cute.local_tile(gd, (self.tile_m, 64),
                                            (task % self.m_tiles, (task // self.m_tiles) * self.math_groups + wg))
                    ssd, tgd = cute.nvgpu.cpasync.tma_partition(tma_d, 0, cute.make_layout(1),
                                                                cute.group_modes(sd, 0, 2), cute.group_modes(gtile, 0, 2))
                    cute.copy(tma_d, ssd, tgd)
                    cute.arch.cp_async_bulk_commit_group()
                    if cutlass.const_expr(self.trace):
                        stores_issued = clock32()
                    source.after_store(position)
                if cutlass.const_expr(self.multi_task):
                    cute.arch.barrier(barrier_id=wg_barrier, number_of_threads=128)
                if cutlass.const_expr(self.trace):
                    task_end = clock32()
                    if tid % 128 == 0:
                        rec = trace[task, group - 1, None]
                        rec[ENTRY], rec[SETUP], rec[TASK_START] = cutlass.Int64(entry), cutlass.Int64(setup), cutlass.Int64(task_start)
                        rec[FIRST_READY], rec[WAIT_SUM], rec[PENDING_SUM] = cutlass.Int64(first_ready), cutlass.Int64(wait_sum), cutlass.Int64(pending_sum)
                        rec[LOOP_END], rec[DRAIN], rec[STAGED] = cutlass.Int64(loop_end), cutlass.Int64(drain), cutlass.Int64(staged)
                        rec[STAGED_BAR], rec[STORES_ISSUED], rec[TASK_END] = cutlass.Int64(staged_bar), cutlass.Int64(stores_issued), cutlass.Int64(task_end)
                        rec[GLOBAL_START], rec[GLOBAL_END], rec[SMID] = global_start, globaltimer(), cutlass.Int64(smid())
                position += source.step()


class PreparedGemmNT:
    """Compile once for fixed A[M,K], B[N,K], D[M,N]; `bind(a, b, d)` reuses the binary for other buffers."""

    def __init__(self, a, b, d, *, sms=None, **config):
        (m, k), (n, kb) = a.shape, b.shape
        if kb != k or d.shape != (m, n) or not all(t.is_contiguous() and t.dtype == torch.bfloat16 for t in (a, b, d)):
            raise ValueError('expect contiguous BF16 A[M,K], B[N,K], D[M,N]')
        if sms is None:
            sms = torch.cuda.get_device_properties(a.device).multi_processor_count
        self.kernel = GemmNT(m, n, k, sms=sms, **config)
        self.no_trace = torch.zeros((1, 1, TRACE_FIELDS), device=a.device, dtype=torch.int64)
        self.compiled = cute.compile(self.kernel, *self.operands(a, b, d, self.trace_buffer() if self.kernel.trace else None),
                                     cuda.CUstream(torch.cuda.current_stream().cuda_stream), options='--generate-line-info')

    def trace_buffer(self, launches=None):
        """int64 [tasks, math_groups + 1, TRACE_FIELDS], or [launches, ...] for one slice per launch."""
        shape = (self.kernel.tasks, self.kernel.math_groups + 1, TRACE_FIELDS)
        return torch.zeros(shape if launches is None else (launches, *shape), device='cuda', dtype=torch.int64)

    def operands(self, a, b, d, trace=None):
        return tuple(from_dlpack(t, assumed_align=16) for t in (a, b, d, self.no_trace if trace is None else trace))

    def bind(self, a, b, d, trace=None):
        """Returns a zero-argument launcher on the current stream; `trace` is a [tasks, rows, fields] buffer."""
        operands = self.operands(a, b, d, trace)
        return lambda: self.compiled(*operands, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
