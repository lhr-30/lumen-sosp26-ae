"""K RMSNorm + RoPE over the packed projection (final graph: candidate.PackedKNormRoPE.cute.v2).

Copied from qwen35-2b-kg-20261020 candidate_code/packed_attention_region_cute_v1.py; used with warps=4, vector=1.
"""
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32
from cuda.bindings import driver as cuda


class PackedKNormRoPE:
    def __init__(self, warps=4, vector=1):
        self.warps, self.vector = warps, vector

    @cute.jit
    def __call__(self, key: cute.Tensor, scale: cute.Tensor, tables: cute.Tensor,
                 output: cute.Tensor, stream: cuda.CUstream):
        self.kernel(key, scale, tables, output).launch(
            grid=(832 // self.warps, 1, 1), block=(32 * self.warps, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, key: cute.Tensor, scale: cute.Tensor, tables: cute.Tensor,
               output: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        self.run(tid, bid, key, scale, tables, output)

    @cute.jit
    def run(self, tid: Int32, bid: Int32, key: cute.Tensor, scale: cute.Tensor,
            tables: cute.Tensor, output: cute.Tensor):
        row, lane = bid * self.warps + tid // 32, tid % 32
        batch, token, head = row // 52, row // 2 % 26, row % 2
        x = cute.make_rmem_tensor((8,), Float32)
        weights = cute.make_rmem_tensor((8,), Float32)
        if cutlass.const_expr(self.vector == 8):
            offset = cute.assume(batch * 133120 + token * 5120 + head * 256 + lane * 8, divby=8)
            gx = cute.make_tensor(key.iterator + offset, cute.make_layout((8,)))
            gw = cute.make_tensor(scale.iterator + lane * 8, cute.make_layout((8,)))
            rx = cute.make_rmem_tensor((8,), BFloat16)
            atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
            cute.copy(atom, gx, rx)
            cute.autovec_copy(gw, weights)
            x.store(rx.load().to(Float32))
        else:
            for j in cutlass.range_constexpr(8):
                x[j] = Float32(key[batch, token, head, lane + j * 32])
                weights[j] = scale[lane + j * 32]
        total = Float32(0.0)
        for j in cutlass.range_constexpr(8):
            total = total + x[j] * x[j]
        for shift in cutlass.range_constexpr(5):
            total = total + cute.arch.shuffle_sync_bfly(total, 1 << shift)
        inverse = cute.math.rsqrt(total * Float32(1.0 / 256) + Float32(1e-6), fastmath=True)
        for j in cutlass.range_constexpr(8):
            x[j] = ((x[j] * inverse) * weights[j]).to(BFloat16).to(Float32)
        if cutlass.const_expr(self.vector == 8):
            if lane < 8:
                for j in cutlass.range_constexpr(8):
                    partner = cute.arch.shuffle_sync_bfly(x[j], 4, mask=0xFF)
                    cosine = Float32(tables[token, lane % 4 * 8 + j, 0])
                    sine = Float32(tables[token, lane % 4 * 8 + j, 1])
                    rotated = x[j] * cosine + partner * sine
                    if lane < 4:
                        rotated = x[j] * cosine - partner * sine
                    x[j] = rotated
            ooffset = cute.assume(row * 256 + lane * 8, divby=8)
            gy = cute.make_tensor(output.iterator + ooffset, cute.make_layout((8,)))
            ry = cute.make_rmem_tensor((8,), BFloat16)
            ry.store(x.load().to(BFloat16))
            cute.copy(atom, ry, gy)
        else:
            cosine, sine = Float32(tables[token, lane, 0]), Float32(tables[token, lane, 1])
            left, right = x[0], x[1]
            x[0] = left * cosine - right * sine
            x[1] = right * cosine + left * sine
            for j in cutlass.range_constexpr(8):
                output[batch, token, head, lane + j * 32] = x[j].to(BFloat16)
