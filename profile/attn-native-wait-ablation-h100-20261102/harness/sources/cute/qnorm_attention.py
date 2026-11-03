"""Q RMSNorm + RoPE + causal GQA attention + sigmoid gate (final graph: candidate.PackedQNormRoPEAttentionGate.cute.v4).

Copied from qwen35-2b-kg-20261020 candidate_code/packed_qk_attention_split_cute_v3.py.
"""
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32
from cutlass.cute.nvgpu import warp
from cutlass.utils import SmemAllocator
from cuda.bindings import driver as cuda


@cute.jit
def load_key_constants(k_scale: cute.Tensor, tables: cute.Tensor, warp_id: Int32, lane: Int32,
                       warps: cutlass.Constexpr):
    """K norm weights and the rotary entries of the K rows warp `warp_id` normalizes (rows >= 26 get 0)."""
    rows = 32 // warps
    weights = cute.make_rmem_tensor((8,), Float32)
    for j in cutlass.range_constexpr(8):
        weights[j] = k_scale[lane + j * 32]
    key_tables = cute.make_rmem_tensor((2, rows), Float32)
    key_tables.fill(0.0)
    for r in cutlass.range_constexpr(rows):
        if warp_id + r * warps < 26:
            key_tables[0, r] = Float32(tables[warp_id + r * warps, lane, 0])
            key_tables[1, r] = Float32(tables[warp_id + r * warps, lane, 1])
    return weights, key_tables


@cute.jit
def normalize_keys(sk: cute.Tensor, weights: cute.Tensor, key_tables: cute.Tensor, warp_id: Int32, lane: Int32,
                   warps: cutlass.Constexpr):
    """K RMSNorm + RoPE in place on a [32, 256] shared tile: warp w takes rows w, w + warps, ... together (rows
    26-31 are zero padding and stay 0). Per row, the arithmetic and summation order of PackedKNormRoPE (vector=1)."""
    rows = 32 // warps
    xk = cute.make_rmem_tensor((8, rows), Float32)
    for r in cutlass.range_constexpr(rows):
        for j in cutlass.range_constexpr(8):
            xk[j, r] = Float32(sk[warp_id + r * warps, lane + j * 32])
    total = cute.make_rmem_tensor((rows,), Float32)
    total.fill(0.0)
    for j in cutlass.range_constexpr(8):
        for r in cutlass.range_constexpr(rows):
            total[r] = total[r] + xk[j, r] * xk[j, r]
    for shift in cutlass.range_constexpr(5):
        for r in cutlass.range_constexpr(rows):
            total[r] = total[r] + cute.arch.shuffle_sync_bfly(total[r], 1 << shift)
    for r in cutlass.range_constexpr(rows):
        inverse = cute.math.rsqrt(total[r] * Float32(1.0 / 256) + Float32(1e-6), fastmath=True)
        for j in cutlass.range_constexpr(8):
            xk[j, r] = ((xk[j, r] * inverse) * weights[j]).to(BFloat16).to(Float32)
        left, right = xk[0, r], xk[1, r]
        xk[0, r] = left * key_tables[0, r] - right * key_tables[1, r]
        xk[1, r] = right * key_tables[0, r] + left * key_tables[1, r]
    for r in cutlass.range_constexpr(rows):
        for j in cutlass.range_constexpr(8):
            sk[warp_id + r * warps, lane + j * 32] = xk[j, r].to(BFloat16)


class PrefetchedQNormAttention:
    def __init__(self, norm_threads=16, gate_prefetch=True, norm_vector=2,
                 query_rows=16, warps=4, q_staging='register', q_stride=296,
                 norm_parallel=True, q_load_order='early', table_prefetch=False, norm_tree=False,
                 apply_gate=True):
        self.query_rows, self.warps, self.threads = query_rows, warps, warps * 32
        self.q_staging = q_staging
        self.norm_parallel, self.q_load_order = norm_parallel, q_load_order
        self.table_prefetch, self.norm_tree = table_prefetch, norm_tree
        self.rows_per_group = query_rows // (self.threads // norm_threads)
        self.q_stride = q_stride
        self.loads, self.epilogue = 'scalar', 'vector'
        self.kv_schedule, self.softmax_parallel = 'async', True
        self.norm_threads, self.gate_prefetch = norm_threads, gate_prefetch
        self.apply_gate = apply_gate
        self.norm_vector = norm_vector
        self.items = 256 // norm_threads
        self.table_items = (64 if norm_threads * norm_vector == 64 else 32) // norm_threads
        self.rotary_mask = sum(((1 << (norm_threads // 4)) - 1) << i
                               for i in range(0, 32, norm_threads))
        self.shifts = {8: 3, 16: 4, 32: 5}[norm_threads]

    @cute.jit
    def __call__(self, query: cute.Tensor, key: cute.Tensor, value: cute.Tensor,
                 gate: cute.Tensor, scale: cute.Tensor, tables: cute.Tensor, output: cute.Tensor, stream: cuda.CUstream):
        mma = cute.make_tiled_mma(warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            (self.query_rows // 16, self.warps // (self.query_rows // 16), 1))
        self.kernel(query, key, value, gate, scale, tables, output, mma).launch(
            grid=(8, 16, cute.ceil_div(26, self.query_rows)),
            block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, query: cute.Tensor, key: cute.Tensor, value: cute.Tensor,
               gate: cute.Tensor, scale: cute.Tensor, tables: cute.Tensor, output: cute.Tensor, mma: cute.TiledMma):
        tid, _, _ = cute.arch.thread_idx()
        head, batch, tile = cute.arch.block_idx()
        self.run(tid, head, batch, tile, query, key, value, gate, scale, tables, output, mma, 0)

    @cute.jit
    def synchronize(self, barrier_id: cutlass.Constexpr):
        if cutlass.const_expr(barrier_id == 0):
            cute.arch.sync_threads()
        else:
            cute.arch.barrier(barrier_id=barrier_id, number_of_threads=self.threads)

    @cute.jit
    def run(self, tid: Int32, head: Int32, batch: Int32, tile: Int32,
            query: cute.Tensor, key: cute.Tensor, value: cute.Tensor, gate: cute.Tensor,
            scale: cute.Tensor, tables: cute.Tensor, output: cute.Tensor, mma: cute.TiledMma,
            barrier_id: cutlass.Constexpr, smem_offset: cutlass.Constexpr = 0, gate_wait=None, k_scale=None,
            kv=None, marks=None):
        """gate_wait: optional object whose wait() returns once the gate rows are written (a fused kernel computes
        them beside attention); the gate is then loaded right before P @ V instead of with K / V.
        k_scale: K norm weights; `key` is then the raw K of packed ([B, T, KV, 256] view of its K columns), which
        this item normalizes and rotates in shared memory with the arithmetic of PackedKNormRoPE (vector=1).
        kv: optional object with a static smem byte `offset` of [32, 256] K (normalized) and V tiles that other warps
        of the CTA fill, and a wait() for them; the item then loads neither and keeps only Q / S / P.
        marks: optional probe whose mark(slot) records a clock at phase points 1-5 (see fused/attention_probe.py)."""
        lane, warp_id = tid % 32, tid // 32
        first_query = tile * self.query_rows
        allocator = SmemAllocator()
        if cutlass.const_expr(smem_offset):
            allocator.allocate(smem_offset)
        sq = allocator.allocate_tensor(BFloat16,
            cute.make_layout((self.query_rows, 256), stride=(self.q_stride, 1)), byte_alignment=16)
        kv_allocator = allocator
        if cutlass.const_expr(kv is not None):
            kv_allocator = SmemAllocator()
            if cutlass.const_expr(kv.offset):
                kv_allocator.allocate(kv.offset)
        sk = kv_allocator.allocate_tensor(BFloat16,
            cute.make_layout((32, 256), stride=(264, 1)), byte_alignment=16)
        sv = kv_allocator.allocate_tensor(BFloat16,
            cute.make_layout((32, 256), stride=(264, 1)), byte_alignment=16)
        ss = allocator.allocate_tensor(Float32,
            cute.make_layout((self.query_rows, 32), stride=(33, 1)), byte_alignment=16)
        sp = allocator.allocate_tensor(BFloat16,
            cute.make_layout((self.query_rows, 32), stride=(40, 1)), byte_alignment=16)
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
        async_atom = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(),
            BFloat16, num_bits_per_copy=128)
        norm_lane, norm_row = tid % self.norm_threads, tid // self.norm_threads
        norm_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16,
                                       num_bits_per_copy=16 * self.norm_vector)
        raw_q = cute.make_rmem_tensor((self.items, self.rows_per_group), BFloat16)
        raw_q.fill(0.0)
        if cutlass.const_expr(self.q_load_order == 'early'):
            for r in cutlass.range_constexpr(self.rows_per_group):
                row = norm_row + r * (self.threads // self.norm_threads)
                token = first_query + row
                if token < 26:
                    if cutlass.const_expr(self.norm_vector == 8):
                        offset = cute.assume(batch * 133120 + token * 5120 + head * 512 + norm_lane * self.items, divby=8)
                        gq = cute.make_tensor(query.iterator + offset, cute.make_layout((self.items,)))
                        cute.copy(atom, gq, raw_q[None, r])
                    elif cutlass.const_expr(self.norm_vector > 1):
                        for j in cutlass.range_constexpr(self.items // self.norm_vector):
                            off = cute.assume(batch * 133120 + token * 5120 + head * 512
                                + norm_lane * self.norm_vector + j * self.norm_threads * self.norm_vector,
                                divby=self.norm_vector)
                            nq = cute.make_tensor(query.iterator + off, cute.make_layout((self.norm_vector,)))
                            rnx = cute.make_rmem_tensor((self.norm_vector,), BFloat16)
                            cute.copy(norm_atom, nq, rnx)
                            for v in cutlass.range_constexpr(self.norm_vector):
                                raw_q[j * self.norm_vector + v, r] = rnx[v]
                    else:
                        for j in cutlass.range_constexpr(self.items):
                            raw_q[j, r] = query[batch, token, head, norm_lane + j * self.norm_threads]
        if cutlass.const_expr(kv is None):
            for j in cutlass.range_constexpr(32 * 256 // (self.threads * 8)):
                index = (tid + j * self.threads) * 8
                token, col = index // 256, index % 256
                shared_offset = cute.assume(token * 264 + col, divby=8)
                dk = cute.make_tensor(sk.iterator + shared_offset, cute.make_layout((8,)))
                dv = cute.make_tensor(sv.iterator + shared_offset, cute.make_layout((8,)))
                if token < 26:
                    offset = cute.assume(batch * 13312 + head // 4 * 256 + token * 512 + col, divby=8)
                    if cutlass.const_expr(k_scale is not None):
                        offset = cute.assume(batch * 133120 + head // 4 * 256 + token * 5120 + col, divby=8)
                    gk = cute.make_tensor(key.iterator + offset, cute.make_layout((8,)))
                    voffset = cute.assume(batch * 133120 + head // 4 * 256 + token * 5120 + col, divby=8)
                    gv = cute.make_tensor(value.iterator + voffset, cute.make_layout((8,)))
                    if cutlass.const_expr(self.kv_schedule == 'async'):
                        cute.copy(async_atom, gk, dk)
                        cute.copy(async_atom, gv, dv)
                    else:
                        rk = cute.make_rmem_tensor((8,), BFloat16)
                        rv = cute.make_rmem_tensor((8,), BFloat16)
                        cute.copy(atom, gk, rk)
                        cute.copy(atom, gv, rv)
                        cute.copy(atom, rk, dk)
                        cute.copy(atom, rv, dv)
                else:
                    zero = cute.make_rmem_tensor((8,), BFloat16)
                    zero.fill(0.0)
                    cute.copy(atom, zero, dk)
                    cute.copy(atom, zero, dv)
            if cutlass.const_expr(self.kv_schedule == 'async'):
                cute.arch.cp_async_commit_group()
        if cutlass.const_expr(k_scale is not None):
            key_weights, key_tables = load_key_constants(k_scale, tables, warp_id, lane, self.warps)
        if cutlass.const_expr(self.gate_prefetch and self.apply_gate and gate_wait is None):
            gate_values = cute.make_rmem_tensor(
                (8, self.query_rows * 256 // (self.threads * 8)), BFloat16)
            gate_values.fill(0.0)
            for j in cutlass.range_constexpr(self.query_rows * 256 // (self.threads * 8)):
                index = (tid + j * self.threads) * 8
                row, col = index // 256, index % 256
                token = first_query + row
                if token < 26:
                    goffset = cute.assume(batch * 133120 + token * 5120 + head * 512 + col, divby=8)
                    gg = cute.make_tensor(gate.iterator + goffset, cute.make_layout((8,)))
                    cute.copy(atom, gg, gate_values[None, j])
        if cutlass.const_expr(self.q_load_order == 'after_kv'):
            for r in cutlass.range_constexpr(self.rows_per_group):
                row = norm_row + r * (self.threads // self.norm_threads)
                token = first_query + row
                if token < 26:
                    if cutlass.const_expr(self.norm_vector == 8):
                        offset = cute.assume(batch * 133120 + token * 5120 + head * 512 + norm_lane * self.items, divby=8)
                        gq = cute.make_tensor(query.iterator + offset, cute.make_layout((self.items,)))
                        cute.copy(atom, gq, raw_q[None, r])
                    elif cutlass.const_expr(self.norm_vector > 1):
                        for j in cutlass.range_constexpr(self.items // self.norm_vector):
                            off = cute.assume(batch * 133120 + token * 5120 + head * 512
                                + norm_lane * self.norm_vector + j * self.norm_threads * self.norm_vector,
                                divby=self.norm_vector)
                            nq = cute.make_tensor(query.iterator + off, cute.make_layout((self.norm_vector,)))
                            rnx = cute.make_rmem_tensor((self.norm_vector,), BFloat16)
                            cute.copy(norm_atom, nq, rnx)
                            for v in cutlass.range_constexpr(self.norm_vector):
                                raw_q[j * self.norm_vector + v, r] = rnx[v]
                    else:
                        for j in cutlass.range_constexpr(self.items):
                            raw_q[j, r] = query[batch, token, head, norm_lane + j * self.norm_threads]
        if cutlass.const_expr(self.table_prefetch):
            table_values = cute.make_rmem_tensor((2, self.table_items, self.rows_per_group), BFloat16)
            table_values.fill(0.0)
            table_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=32)
            for r in cutlass.range_constexpr(self.rows_per_group):
                token = first_query + norm_row + r * (self.threads // self.norm_threads)
                for j in cutlass.range_constexpr(self.table_items):
                    table_values[0, j, r] = BFloat16(1.0)
                    if token < 26:
                        frequency = norm_lane * self.norm_vector + (j // self.norm_vector) * self.norm_threads * self.norm_vector + j % self.norm_vector
                        if cutlass.const_expr(self.norm_vector * self.norm_threads == 64):
                            frequency = norm_lane % (self.norm_threads // 2) * self.norm_vector + j
                        off = cute.assume(token * 64 + frequency * 2, divby=2)
                        src_table = cute.make_tensor(tables.iterator + off, cute.make_layout((2,)))
                        cute.copy(table_atom, src_table, table_values[None, j, r])
        weights = cute.make_rmem_tensor((self.items,), Float32)
        if cutlass.const_expr(self.norm_vector == 8):
            gw = cute.make_tensor(scale.iterator + norm_lane * self.items, cute.make_layout((self.items,)))
            cute.autovec_copy(gw, weights)
        elif cutlass.const_expr(self.norm_vector > 1):
            for j in cutlass.range_constexpr(self.items // self.norm_vector):
                wn = cute.make_tensor(scale.iterator + norm_lane * self.norm_vector
                    + j * self.norm_threads * self.norm_vector, cute.make_layout((self.norm_vector,)))
                rw = cute.make_rmem_tensor((self.norm_vector,), Float32)
                cute.autovec_copy(wn, rw)
                for v in cutlass.range_constexpr(self.norm_vector):
                    weights[j * self.norm_vector + v] = rw[v]
        else:
            for j in cutlass.range_constexpr(self.items):
                weights[j] = scale[norm_lane + j * self.norm_threads]
        frequency = Int32(0)
        cosine, sine = Float32(1.0), Float32(0.0)
        if cutlass.const_expr(self.norm_parallel):
            all_x = cute.make_rmem_tensor((self.items, self.rows_per_group), Float32)
            all_x.store(raw_q.load().to(Float32))
            totals = cute.make_rmem_tensor((self.rows_per_group,), Float32)
            inverses = cute.make_rmem_tensor((self.rows_per_group,), Float32)
            totals.fill(0.0)
            if cutlass.const_expr(self.norm_tree):
                squares = cute.make_rmem_tensor((self.items, self.rows_per_group), Float32)
                for j in cutlass.range_constexpr(self.items):
                    for r in cutlass.range_constexpr(self.rows_per_group):
                        squares[j, r] = all_x[j, r] * all_x[j, r]
                for level in cutlass.range_constexpr(8 - self.shifts):
                    for j in cutlass.range_constexpr(self.items // (2 << level)):
                        for r in cutlass.range_constexpr(self.rows_per_group):
                            squares[j, r] = squares[2 * j, r] + squares[2 * j + 1, r]
                for r in cutlass.range_constexpr(self.rows_per_group):
                    totals[r] = squares[0, r]
            else:
                for j in cutlass.range_constexpr(self.items):
                    for r in cutlass.range_constexpr(self.rows_per_group):
                        totals[r] = totals[r] + all_x[j, r] * all_x[j, r]
            for shift in cutlass.range_constexpr(self.shifts):
                for r in cutlass.range_constexpr(self.rows_per_group):
                    totals[r] = totals[r] + cute.arch.shuffle_sync_bfly(totals[r], 1 << shift)
            for r in cutlass.range_constexpr(self.rows_per_group):
                inverses[r] = cute.math.rsqrt(totals[r] * Float32(1.0 / 256) + Float32(1e-6), fastmath=True)
            for j in cutlass.range_constexpr(self.items):
                for r in cutlass.range_constexpr(self.rows_per_group):
                    all_x[j, r] = (all_x[j, r] * inverses[r] * weights[j]).to(BFloat16).to(Float32)
            for r in cutlass.range_constexpr(self.rows_per_group):
                row = norm_row + r * (self.threads // self.norm_threads)
                token = first_query + row
                x = cute.make_rmem_tensor((self.items,), Float32)
                rx = cute.make_rmem_tensor((self.items,), BFloat16)
                x.store(all_x[None, r].load())
                if cutlass.const_expr(self.norm_vector == 8):
                    if norm_lane < self.norm_threads // 4:
                        for j in cutlass.range_constexpr(self.items):
                            partner = cute.arch.shuffle_sync_bfly(x[j], self.norm_threads // 8, mask=self.rotary_mask)
                            frequency = norm_lane % (self.norm_threads // 8) * self.items + j
                            cosine, sine = Float32(1.0), Float32(0.0)
                            if token < 26:
                                cosine = Float32(tables[token, frequency, 0])
                                sine = Float32(tables[token, frequency, 1])
                            rotated = x[j] * cosine + partner * sine
                            if norm_lane < self.norm_threads // 8:
                                rotated = x[j] * cosine - partner * sine
                            x[j] = rotated
                    rx.store(x.load().to(BFloat16))
                    norm_soffset = cute.assume(row * self.q_stride + norm_lane * self.items, divby=8)
                    dq = cute.make_tensor(sq.iterator + norm_soffset, cute.make_layout((self.items,)))
                    cute.copy(atom, rx, dq)
                else:
                    if cutlass.const_expr(self.norm_vector * self.norm_threads == 64):
                        for j in cutlass.range_constexpr(self.norm_vector):
                            partner = cute.arch.shuffle_sync_bfly(x[j], self.norm_threads // 2)
                            frequency = norm_lane % (self.norm_threads // 2) * self.norm_vector + j
                            cosine, sine = Float32(1.0), Float32(0.0)
                            if cutlass.const_expr(self.table_prefetch):
                                cosine = Float32(table_values[0, j, r])
                                sine = Float32(table_values[1, j, r])
                            elif token < 26:
                                cosine = Float32(tables[token, frequency, 0])
                                sine = Float32(tables[token, frequency, 1])
                            rotated = x[j] * cosine + partner * sine
                            if norm_lane < self.norm_threads // 2:
                                rotated = x[j] * cosine - partner * sine
                            x[j] = rotated
                    else:
                        if token < 26:
                            for j in cutlass.range_constexpr(32 // self.norm_threads):
                                frequency = norm_lane * self.norm_vector + (j // self.norm_vector) * self.norm_threads * self.norm_vector + j % self.norm_vector
                                if cutlass.const_expr(self.table_prefetch):
                                    cosine = Float32(table_values[0, j, r])
                                    sine = Float32(table_values[1, j, r])
                                else:
                                    cosine = Float32(tables[token, frequency, 0])
                                    sine = Float32(tables[token, frequency, 1])
                                left, right = x[j], x[j + 32 // self.norm_threads]
                                x[j] = left * cosine - right * sine
                                x[j + 32 // self.norm_threads] = right * cosine + left * sine
                    if cutlass.const_expr(self.norm_vector > 1):
                        for j in cutlass.range_constexpr(self.items // self.norm_vector):
                            nout_offset = cute.assume(row * self.q_stride + norm_lane * self.norm_vector
                                + j * self.norm_threads * self.norm_vector, divby=self.norm_vector)
                            nout = cute.make_tensor(sq.iterator + nout_offset, cute.make_layout((self.norm_vector,)))
                            rout = cute.make_rmem_tensor((self.norm_vector,), BFloat16)
                            for v in cutlass.range_constexpr(self.norm_vector):
                                rout[v] = x[j * self.norm_vector + v].to(BFloat16)
                            cute.copy(norm_atom, rout, nout)
                    else:
                        for j in cutlass.range_constexpr(self.items):
                            sq[row, norm_lane + j * self.norm_threads] = x[j].to(BFloat16)
        else:
            for r in cutlass.range_constexpr(self.rows_per_group):
                row = norm_row + r * (self.threads // self.norm_threads)
                token = first_query + row
                x = cute.make_rmem_tensor((self.items,), Float32)
                rx = cute.make_rmem_tensor((self.items,), BFloat16)
                x.store(raw_q[None, r].load().to(Float32))
                total = Float32(0.0)
                for j in cutlass.range_constexpr(self.items):
                    total = total + x[j] * x[j]
                for shift in cutlass.range_constexpr(self.shifts):
                    total = total + cute.arch.shuffle_sync_bfly(total, 1 << shift)
                inverse = cute.math.rsqrt(total * Float32(1.0 / 256) + Float32(1e-6), fastmath=True)
                for j in cutlass.range_constexpr(self.items):
                    x[j] = (x[j] * inverse * weights[j]).to(BFloat16).to(Float32)
                if cutlass.const_expr(self.norm_vector == 8):
                    if norm_lane < self.norm_threads // 4:
                        for j in cutlass.range_constexpr(self.items):
                            partner = cute.arch.shuffle_sync_bfly(x[j], self.norm_threads // 8, mask=self.rotary_mask)
                            frequency = norm_lane % (self.norm_threads // 8) * self.items + j
                            cosine, sine = Float32(1.0), Float32(0.0)
                            if token < 26:
                                cosine = Float32(tables[token, frequency, 0])
                                sine = Float32(tables[token, frequency, 1])
                            rotated = x[j] * cosine + partner * sine
                            if norm_lane < self.norm_threads // 8:
                                rotated = x[j] * cosine - partner * sine
                            x[j] = rotated
                    rx.store(x.load().to(BFloat16))
                    norm_soffset = cute.assume(row * self.q_stride + norm_lane * self.items, divby=8)
                    dq = cute.make_tensor(sq.iterator + norm_soffset, cute.make_layout((self.items,)))
                    cute.copy(atom, rx, dq)
                else:
                    if cutlass.const_expr(self.norm_vector * self.norm_threads == 64):
                        for j in cutlass.range_constexpr(self.norm_vector):
                            partner = cute.arch.shuffle_sync_bfly(x[j], self.norm_threads // 2)
                            frequency = norm_lane % (self.norm_threads // 2) * self.norm_vector + j
                            cosine, sine = Float32(1.0), Float32(0.0)
                            if cutlass.const_expr(self.table_prefetch):
                                cosine = Float32(table_values[0, j, r])
                                sine = Float32(table_values[1, j, r])
                            elif token < 26:
                                cosine = Float32(tables[token, frequency, 0])
                                sine = Float32(tables[token, frequency, 1])
                            rotated = x[j] * cosine + partner * sine
                            if norm_lane < self.norm_threads // 2:
                                rotated = x[j] * cosine - partner * sine
                            x[j] = rotated
                    else:
                        if token < 26:
                            for j in cutlass.range_constexpr(32 // self.norm_threads):
                                frequency = norm_lane * self.norm_vector + (j // self.norm_vector) * self.norm_threads * self.norm_vector + j % self.norm_vector
                                if cutlass.const_expr(self.table_prefetch):
                                    cosine = Float32(table_values[0, j, r])
                                    sine = Float32(table_values[1, j, r])
                                else:
                                    cosine = Float32(tables[token, frequency, 0])
                                    sine = Float32(tables[token, frequency, 1])
                                left, right = x[j], x[j + 32 // self.norm_threads]
                                x[j] = left * cosine - right * sine
                                x[j + 32 // self.norm_threads] = right * cosine + left * sine
                    if cutlass.const_expr(self.norm_vector > 1):
                        for j in cutlass.range_constexpr(self.items // self.norm_vector):
                            nout_offset = cute.assume(row * self.q_stride + norm_lane * self.norm_vector
                                + j * self.norm_threads * self.norm_vector, divby=self.norm_vector)
                            nout = cute.make_tensor(sq.iterator + nout_offset, cute.make_layout((self.norm_vector,)))
                            rout = cute.make_rmem_tensor((self.norm_vector,), BFloat16)
                            for v in cutlass.range_constexpr(self.norm_vector):
                                rout[v] = x[j * self.norm_vector + v].to(BFloat16)
                            cute.copy(norm_atom, rout, nout)
                    else:
                        for j in cutlass.range_constexpr(self.items):
                            sq[row, norm_lane + j * self.norm_threads] = x[j].to(BFloat16)
        if cutlass.const_expr(marks is not None):
            marks.mark(1)                   # Q normalized into sq
        if cutlass.const_expr(kv is not None):
            kv.wait()                       # also orders this item's sq writes before the QK^T reads
        else:
            if cutlass.const_expr(self.kv_schedule == 'async'):
                cute.arch.cp_async_wait_group(0)
            self.synchronize(barrier_id)
        if cutlass.const_expr(k_scale is not None):
            normalize_keys(sk, key_weights, key_tables, warp_id, lane, self.warps)
            self.synchronize(barrier_id)

        if cutlass.const_expr(marks is not None):
            marks.mark(2)                   # K / V ready
        thr = mma.get_slice(tid)
        qa, kb, sc = thr.partition_A(sq), thr.partition_B(sk), thr.partition_C(ss)
        ra, rb = mma.make_fragment_A(qa), mma.make_fragment_B(kb)
        scores = cute.make_fragment_like(sc, Float32)
        scores.fill(0.0)
        if cutlass.const_expr(self.loads == 'matrix'):
            load = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), BFloat16)
            cq = cute.make_tiled_copy_A(load, mma)
            ck = cute.make_tiled_copy_B(load, mma)
            tq, tk = cq.get_slice(tid), ck.get_slice(tid)
            sqsrc, sksrc = tq.partition_S(sq), tk.partition_S(sk)
            rqdst, rkdst = tq.retile(ra), tk.retile(rb)
            for step in cutlass.range_constexpr(cute.size(ra, mode=[2])):
                cute.copy(cq, sqsrc[None, None, step], rqdst[None, None, step])
                cute.copy(ck, sksrc[None, None, step], rkdst[None, None, step])
                cute.gemm(mma, scores, ra[None, None, step], rb[None, None, step], scores)
        else:
            cute.autovec_copy(qa, ra)
            cute.autovec_copy(kb, rb)
            cute.gemm(mma, scores, ra, rb, scores)
        coords = thr.partition_C(cute.make_identity_tensor((self.query_rows, 32)))
        for i in cutlass.range_constexpr(cute.size(scores)):
            row, col = coords[i]
            score = Float32(float('-inf'))
            if col < 26 and col <= first_query + row:
                score = scores[i] * Float32(0.0625)
            sc[i] = score
        self.synchronize(barrier_id)
        if cutlass.const_expr(marks is not None):
            marks.mark(3)                   # scores in smem

        if cutlass.const_expr(self.softmax_parallel):
            row_scores = cute.make_rmem_tensor((self.query_rows // self.warps,), Float32)
            maxima = cute.make_rmem_tensor((self.query_rows // self.warps,), Float32)
            probabilities = cute.make_rmem_tensor((self.query_rows // self.warps,), Float32)
            denominators = cute.make_rmem_tensor((self.query_rows // self.warps,), Float32)
            for r in cutlass.range_constexpr(self.query_rows // self.warps):
                row_scores[r] = ss[warp_id + r * self.warps, lane]
                maxima[r] = row_scores[r]
            for shift in cutlass.range_constexpr(5):
                for r in cutlass.range_constexpr(self.query_rows // self.warps):
                    maxima[r] = cute.arch.fmax(maxima[r], cute.arch.shuffle_sync_bfly(maxima[r], 1 << shift))
            for r in cutlass.range_constexpr(self.query_rows // self.warps):
                probabilities[r] = cute.math.exp(row_scores[r] - maxima[r], fastmath=True)
                denominators[r] = probabilities[r]
            for shift in cutlass.range_constexpr(5):
                for r in cutlass.range_constexpr(self.query_rows // self.warps):
                    denominators[r] = denominators[r] + cute.arch.shuffle_sync_bfly(denominators[r], 1 << shift)
            for r in cutlass.range_constexpr(self.query_rows // self.warps):
                sp[warp_id + r * self.warps, lane] = (probabilities[r] / denominators[r]).to(BFloat16)
        else:
            for r in cutlass.range_constexpr(self.query_rows // self.warps):
                row = warp_id + r * self.warps
                score = ss[row, lane]
                maximum = score
                for shift in cutlass.range_constexpr(5):
                    maximum = cute.arch.fmax(maximum, cute.arch.shuffle_sync_bfly(maximum, 1 << shift))
                probability = cute.math.exp(score - maximum, fastmath=True)
                denominator = probability
                for shift in cutlass.range_constexpr(5):
                    denominator = denominator + cute.arch.shuffle_sync_bfly(denominator, 1 << shift)
                sp[row, lane] = (probability / denominator).to(BFloat16)
        self.synchronize(barrier_id)
        if cutlass.const_expr(marks is not None):
            marks.mark(4)                   # probabilities in smem

        if cutlass.const_expr(gate_wait is not None):
            gate_wait.wait()
            gate_values = cute.make_rmem_tensor(
                (8, self.query_rows * 256 // (self.threads * 8)), BFloat16)
            gate_values.fill(0.0)
            for j in cutlass.range_constexpr(self.query_rows * 256 // (self.threads * 8)):
                index = (tid + j * self.threads) * 8
                row, col = index // 256, index % 256
                token = first_query + row
                if token < 26:
                    goffset = cute.assume(batch * 133120 + token * 5120 + head * 512 + col, divby=8)
                    gg = cute.make_tensor(gate.iterator + goffset, cute.make_layout((8,)))
                    cute.copy(atom, gg, gate_values[None, j])

        vt = cute.make_tensor(sv.iterator, cute.make_layout((256, 32), stride=(1, 264)))
        pa, vb, oc = thr.partition_A(sp), thr.partition_B(vt), thr.partition_C(sq)
        rp, rv = mma.make_fragment_A(pa), mma.make_fragment_B(vb)
        result = cute.make_fragment_like(oc, Float32)
        result.fill(0.0)
        if cutlass.const_expr(self.loads == 'matrix'):
            p_load = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), BFloat16)
            v_load = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 4), BFloat16)
            cp, cv = cute.make_tiled_copy_A(p_load, mma), cute.make_tiled_copy_B(v_load, mma)
            tp, tv = cp.get_slice(tid), cv.get_slice(tid)
            cute.copy(cp, tp.partition_S(sp), tp.retile(rp))
            cute.copy(cv, tv.partition_S(vt), tv.retile(rv))
        else:
            cute.autovec_copy(pa, rp)
            cute.autovec_copy(vb, rv)
        cute.gemm(mma, result, rp, rv, result)
        if cutlass.const_expr(marks is not None):
            marks.mark(5)                   # P @ V in registers
        if cutlass.const_expr(self.epilogue == 'vector'):
            rounded = cute.make_fragment_like(oc, BFloat16)
            rounded.store(result.load().to(BFloat16))
            cute.autovec_copy(rounded, oc)
            self.synchronize(barrier_id)
            for j in cutlass.range_constexpr(self.query_rows * 256 // (self.threads * 8)):
                index = (tid + j * self.threads) * 8
                row, col = index // 256, index % 256
                token = first_query + row
                if token < 26:
                    soffset = cute.assume(row * self.q_stride + col, divby=8)
                    goffset = cute.assume(batch * 133120 + token * 5120 + head * 512 + col, divby=8)
                    ooffset = cute.assume(batch * 53248 + token * 2048 + head * 256 + col, divby=8)
                    sy = cute.make_tensor(sq.iterator + soffset, cute.make_layout((8,)))
                    gy = cute.make_tensor(output.iterator + ooffset, cute.make_layout((8,)))
                    ry = cute.make_rmem_tensor((8,), BFloat16)
                    cute.copy(atom, sy, ry)
                    if cutlass.const_expr(self.apply_gate):
                        gg = cute.make_tensor(gate.iterator + goffset, cute.make_layout((8,)))
                        rg = cute.make_rmem_tensor((8,), BFloat16)
                        if cutlass.const_expr(self.gate_prefetch):
                            cute.copy(atom, gate_values[None, j], rg)
                        else:
                            cute.copy(atom, gg, rg)
                        for i in cutlass.range_constexpr(8):
                            g = Float32(rg[i])
                            sigmoid = cute.arch.rcp_approx(Float32(1.0) + cute.math.exp(-g, fastmath=True))
                            sigmoid = sigmoid.to(BFloat16).to(Float32)
                            ry[i] = (Float32(ry[i]) * sigmoid).to(BFloat16)
                    cute.copy(atom, ry, gy)
        else:
            out_coords = thr.partition_C(cute.make_identity_tensor((self.query_rows, 256)))
            for i in cutlass.range_constexpr(cute.size(result)):
                row, col = out_coords[i]
                token = first_query + row
                if token < 26:
                    attended = result[i].to(BFloat16).to(Float32)
                    if cutlass.const_expr(self.apply_gate):
                        g = Float32(gate[batch, token, head, col])
                        sigmoid = cute.arch.rcp_approx(Float32(1.0) + cute.math.exp(-g, fastmath=True))
                        sigmoid = sigmoid.to(BFloat16).to(Float32)
                        attended = attended * sigmoid
                    output[batch, token, head * 256 + col] = attended.to(BFloat16)
