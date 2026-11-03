"""Correctness-first Hopper GEMM: BF16 inputs, FP32 accumulation/output.

This is a small AveLang integration example, not a tuned cuBLAS competitor.
Each CTA owns one 64x8 output tile. Bounds checks zero-pad all three axes.
"""
from __future__ import annotations

import avelang
import avelang.language as S


@avelang.jit
def gemm_hopper(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.f32),
    M: S.constexpr,
    N: S.constexpr,
    K: S.constexpr,
):
    tid = S.thread_id(0)
    warp = tid // 32
    lane = tid % 32
    row0 = S.block_id(1) * 64
    col0 = S.block_id(0) * 8
    # [row group, K half, row in atom, K in atom], K-major / no swizzle.
    a = S.make_shared((8, 2, 8, 8), S.bf16, 16)
    b = S.make_shared((1, 2, 8, 8), S.bf16, 16)
    ready = S.make_shared((1,), S.i64, 8)
    acc = S.make_local((4,), S.f32)
    for i in S.range(4):
        acc[i] = S.convert(0, S.f32)
    if tid == 0:
        S.nvvm.mbarrier_init(ready, 128)
    S.syncthreads()

    for kt in S.range((K + 15) // 16):
        for j in S.range(8):
            x = tid + j * 128
            r = x // 16
            k = x % 16
            av = S.convert(0, S.bf16)
            if row0 + r < M and kt * 16 + k < K:
                av = A[row0 + r, kt * 16 + k]
            a[r // 8, k // 8, r % 8, k % 8] = av
        n = tid // 16
        k = tid % 16
        bv = S.convert(0, S.bf16)
        if col0 + n < N and kt * 16 + k < K:
            bv = B[kt * 16 + k, col0 + n]
        b[0, k // 8, n, k % 8] = bv

        # Every writer publishes its stores to the asynchronous proxy. The
        # counted barrier joins all 128 writers and is reused with alternating
        # phase parity. No TMA or host tensor-map descriptors are involved.
        S.nvvm.fence_proxy_async_shared()
        S.nvvm.mbarrier_arrive(ready)
        S.nvvm.mbarrier_wait_parity(ready, kt % 2)
        acc = S.nvvm.wgmma_m64n8k16_bf16_f32(a, b, acc)
        # Everyone must finish reading before any thread reuses shared storage.
        S.syncthreads()

    if tid == 0:
        S.nvvm.mbarrier_inval(ready)
    row = row0 + warp * 16 + lane // 4
    col = col0 + (lane % 4) * 2
    if row < M and col < N:
        C[row, col] = acc[0]
    if row < M and col + 1 < N:
        C[row, col + 1] = acc[1]
    if row + 8 < M and col < N:
        C[row + 8, col] = acc[2]
    if row + 8 < M and col + 1 < N:
        C[row + 8, col + 1] = acc[3]


def matmul(a, b):
    """Contiguous CUDA BF16 matrices -> FP32 matrix, on the input device."""
    import torch
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    if (a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16
            or not a.is_cuda or b.device != a.device):
        raise ValueError("inputs must be BF16 tensors on the same CUDA device")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("inputs must be contiguous")
    if torch.cuda.get_device_capability(a.device) != (9, 0):
        raise ValueError("this example requires a Hopper GPU (SM90)")
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), dtype=torch.float32, device=a.device)
    if m == 0 or n == 0:
        return c
    with torch.cuda.device(a.device):
        gemm_hopper[lambda: (((n + 7) // 8, (m + 63) // 64, 1), (128, 1, 1))](
            a, b, c, m, n, k, num_warps=4, validate_invariants=True)
    return c
