"""CPU compiler tests. GPU numerical checks are in validate.py."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import avelang
import avelang.language as S
from avelang.language import invariant
from avelang.backends.compiler import GPUTarget
from avelang.backends.nvidia.compiler import NvidiaCompiler
from avelang.compiler.compiler import ASTSource
from gemm import gemm_hopper


def compile_fn(fn, constants=None, chip="sm_90a", warps=4, validate=True):
    target = GPUTarget("nvptx64-nvidia-cuda", chip)
    src = ASTSource(fn, {}, constexprs={
        k: {"type": "i32", "value": v} for k, v in (constants or {}).items()})
    return NvidiaCompiler(target).compile(src, target, {
        "num_warps": warps, "validate_invariants": validate})


@avelang.jit
def barrier_copy(X: S.Tensor((128,), S.i32), Y: S.Tensor((128,), S.i32),
                 SHIFT: S.constexpr):
    invariant.tag(X, lambda i: (i,), "Element")
    tid = S.thread_id(0)
    data = S.make_shared((128,), S.i32)
    bar = S.make_shared((1,), S.i64, 8)
    if tid == 0:
        S.nvvm.mbarrier_init(bar, 128)
    S.syncthreads()
    data[tid] = X[tid]
    S.nvvm.mbarrier_arrive(bar)
    S.nvvm.mbarrier_wait_parity(bar, 0)
    value = data[tid + SHIFT]
    invariant.assert_tag_eq(value, X[tid])
    Y[tid] = value
    S.syncthreads()
    if tid == 0:
        S.nvvm.mbarrier_inval(bar)


@avelang.jit
def bad_global_barrier(X: S.Tensor((1,), S.i64)):
    S.nvvm.mbarrier_init(X, 1)


@avelang.jit
def bad_barrier_count(X: S.Tensor((1,), S.i32)):
    bar = S.make_shared((1,), S.i64, 8)
    S.nvvm.mbarrier_init(bar, 0)


@avelang.jit
def bad_parity(X: S.Tensor((1,), S.i32)):
    bar = S.make_shared((1,), S.i64, 8)
    S.nvvm.mbarrier_wait_parity(bar, 2)


@avelang.jit
def bad_wgmma_layout(X: S.Tensor((1,), S.f32)):
    a = S.make_shared((64, 16), S.bf16, 16)
    b = S.make_shared((1, 2, 8, 8), S.bf16, 16)
    c = S.make_local((4,), S.f32)
    c = S.nvvm.wgmma_m64n8k16_bf16_f32(a, b, c)
    X[0] = c[0]


@avelang.jit
def bad_alignment(X: S.Tensor((1,), S.i32)):
    bar = S.make_shared((1,), S.i64, 4)
    S.nvvm.mbarrier_init(bar, 1)


@avelang.jit
def opaque_wgmma_output(X: S.Tensor((128,), S.f32)):
    invariant.tag(X, lambda i: (i,), "Input")
    a = S.make_shared((8, 2, 8, 8), S.bf16, 16)
    b = S.make_shared((1, 2, 8, 8), S.bf16, 16)
    c = S.make_local((4,), S.f32)
    for i in S.range(4):
        c[i] = X[S.thread_id(0)]
    c = S.nvvm.wgmma_m64n8k16_bf16_f32(a, b, c)
    # An unsupported arithmetic tag transfer must fail closed, even though the
    # accumulator was initially tagged. This kernel is compiled only, not run.
    invariant.assert_tag_eq(c[0], X[S.thread_id(0)])
    X[S.thread_id(0)] = c[0]


class HopperCompilerTests(unittest.TestCase):
    def test_ptx_and_assembler(self):
        ptx = compile_fn(gemm_hopper, {"M": 65, "N": 13, "K": 33})
        for instruction in [b".target sm_90a", b"wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16",
                            b"wgmma.fence", b"wgmma.commit_group", b"wgmma.wait_group",
                            b"fence.proxy.async.shared::cta", b"mbarrier.init",
                            b"mbarrier.arrive", b"mbarrier.try_wait.parity", b"mbarrier.inval"]:
            self.assertIn(instruction, ptx)
        self.assertIn(b".shared .align 8", ptx)
        self.assertEqual(ptx.count(b".shared .align 16"), 2)
        ptxas = shutil.which("ptxas")
        self.assertIsNotNone(ptxas, "CUDA ptxas is required for offline ISA validation")
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "gemm.ptx"
            src.write_bytes(ptx)
            subprocess.run([ptxas, "-arch=sm_90a", str(src), "-o", str(Path(d) / "gemm.cubin")], check=True)
        if os.environ.get("HOPPER_RESULTS"):
            Path(os.environ["HOPPER_RESULTS"]).mkdir(parents=True, exist_ok=True)
            (Path(os.environ["HOPPER_RESULTS"]) / "gemm.ptx").write_bytes(ptx)

    def test_wrong_architecture_and_warp_count(self):
        for chip, warps in [("sm_80", 4), ("sm_90", 4), ("sm_90a", 1), ("sm_90a", -1)]:
            with self.subTest(chip=chip, warps=warps), self.assertRaises(RuntimeError):
                compile_fn(gemm_hopper, {"M": 64, "N": 8, "K": 16}, chip, warps)

    def test_bad_operands(self):
        for fn in [bad_global_barrier, bad_barrier_count, bad_parity, bad_wgmma_layout, bad_alignment]:
            with self.subTest(fn=fn.__name__), self.assertRaises(RuntimeError):
                compile_fn(fn)

    def test_wgmma_output_tags_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "unable to prove invariant tag equality"):
            compile_fn(opaque_wgmma_output)

    def test_invariant_across_barrier(self):
        compile_fn(barrier_copy, {"SHIFT": 0})
        with self.assertRaises(RuntimeError):
            compile_fn(barrier_copy, {"SHIFT": 1})


if __name__ == "__main__":
    unittest.main()
