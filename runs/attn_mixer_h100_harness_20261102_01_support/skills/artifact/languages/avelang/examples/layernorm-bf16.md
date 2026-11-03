---
id: avelang-example-layernorm-bf16
title: "AveLang BF16 LayerNorm"
type: example
language: avelang
description: "Verified AveLang BF16 LayerNorm with tiled reduction and apply passes."
operators: [layernorm, normalization, reduction]
dtype: [bf16, f32]
hardware: [amd-gpu, mi300x]
---

# AveLang BF16 LayerNorm

Compiled, correctness-checked, and faster than the PyTorch reference on AMD Instinct MI300X (BF16).

Use:
1. Keep the three-pass structure and launch mapping.
2. Change only normalization math, indexing, and shape contract.
3. Use only APIs shown here or in the [AveLang language spec](../../avelang-language-spec.md).


**PyTorch reference:**
```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Layer Normalization.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return self.ln(x)

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [(features, dim1, dim2)]
```

**Verified AveLang kernel:**
```python
import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
TILE_SIZE: al.constexpr = 4096
MAX_TILES_PER_ROUND: al.constexpr = 256


@avelang.jit
def layernorm_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    N: al.i32,
    num_tiles: al.i32,
):
    """
    Phase 1: reduce each tile into (sum, sum_sq).
    Launch: grid = (num_tiles * batch_size, 1, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    batch_idx = bid // num_tiles
    tile_idx = bid - batch_idx * num_tiles

    if batch_idx < batch_size:
        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        base = batch_idx * N
        tile_start = tile_idx * TILE_SIZE
        tile_end = tile_start + TILE_SIZE
        if tile_end > N:
            tile_end = N

        layout_in = al.make_layout((batch_size * N,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        local_sum = al.convert(0.0, al.f32)
        local_sq = al.convert(0.0, al.f32)

        for i in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            idx = base + i
            val = al.convert(x[idx], al.f32)
            local_sum = local_sum + val
            local_sq = local_sq + val * val

        smem_sum[tid] = local_sum
        smem_sq[tid] = local_sq
        al.syncthreads()

        if tid < 128:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 128]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 64]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 32]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 16]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 8]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 4]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 2]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 1]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 1]

        if tid == 0:
            layout_ps = al.make_layout((batch_size, num_tiles), (num_tiles, 1))
            ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)
            psq = al.make_tensor(partial_sq_ptr, al.f32, layout_ps)
            ps[batch_idx, tile_idx] = smem_sum[0]
            psq[batch_idx, tile_idx] = smem_sq[0]


@avelang.jit
def layernorm_aggregate_kernel(
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    mean_out_ptr: al.Pointer(al.f32),
    rstd_out_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    num_tiles: al.i32,
    N: al.i32,
    eps: al.f32,
):
    """
    Phase 2: aggregate partial tile results into per-batch mean and rstd.
    Handles num_tiles > BLOCK_SIZE by iterating in chunks of MAX_TILES_PER_ROUND.
    Launch: grid = (batch_size, 1, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < batch_size:
        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        layout_ps = al.make_layout((batch_size, num_tiles), (num_tiles, 1))
        ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)
        psq = al.make_tensor(partial_sq_ptr, al.f32, layout_ps)

        total_sum = al.convert(0.0, al.f32)
        total_sq = al.convert(0.0, al.f32)

        # Process tiles in chunks of MAX_TILES_PER_ROUND
        chunk_start = al.convert(0, al.i32)
        for _ in al.range(0, 16):
            if chunk_start >= num_tiles:
                break

            local_sum = al.convert(0.0, al.f32)
            local_sq = al.convert(0.0, al.f32)

            tile_idx = chunk_start + tid
            if tile_idx < num_tiles:
                local_sum = ps[bid, tile_idx]
                local_sq = psq[bid, tile_idx]

            smem_sum[tid] = local_sum
            smem_sq[tid] = local_sq
            al.syncthreads()

            if tid < 128:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 128]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 128]
            al.syncthreads()
            if tid < 64:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 64]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 64]
            al.syncthreads()
            if tid < 32:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 32]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 32]
            al.syncthreads()
            if tid < 16:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 16]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 16]
            al.syncthreads()
            if tid < 8:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 8]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 8]
            al.syncthreads()
            if tid < 4:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 4]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 4]
            al.syncthreads()
            if tid < 2:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 2]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 2]
            al.syncthreads()
            if tid < 1:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 1]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 1]

            if tid == 0:
                total_sum = total_sum + smem_sum[0]
                total_sq = total_sq + smem_sq[0]

            chunk_start = chunk_start + MAX_TILES_PER_ROUND
            al.syncthreads()

        if tid == 0:
            N_f32 = al.convert(N, al.f32)
            mean = total_sum / N_f32
            var = total_sq / N_f32 - mean * mean
            rstd = al.convert(1.0, al.f32) / al.sqrt(var + eps)

            layout_out = al.make_layout((batch_size,), (1,))
            mo = al.make_tensor(mean_out_ptr, al.f32, layout_out)
            ro = al.make_tensor(rstd_out_ptr, al.f32, layout_out)
            mo[bid] = mean
            ro[bid] = rstd


@avelang.jit
def layernorm_apply_kernel(
    x_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    rstd_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    N: al.i32,
    num_tiles: al.i32,
):
    """
    Phase 3: apply normalization and affine transform.
    Launch: grid = (num_tiles * batch_size, 1, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    batch_idx = bid // num_tiles
    tile_idx = bid - batch_idx * num_tiles

    if batch_idx < batch_size:
        base = batch_idx * N
        tile_start = tile_idx * TILE_SIZE
        tile_end = tile_start + TILE_SIZE
        if tile_end > N:
            tile_end = N

        layout_in = al.make_layout((batch_size * N,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        layout_w = al.make_layout((N,), (1,))
        wt = al.make_tensor(weight_ptr, al.bf16, layout_w)
        bt = al.make_tensor(bias_ptr, al.bf16, layout_w)

        layout_out = al.make_layout((batch_size * N,), (1,))
        ot = al.make_tensor(out_ptr, al.bf16, layout_out)

        layout_mean = al.make_layout((batch_size,), (1,))
        mt = al.make_tensor(mean_ptr, al.f32, layout_mean)
        rt = al.make_tensor(rstd_ptr, al.f32, layout_mean)

        mean = mt[batch_idx]
        rstd = rt[batch_idx]

        for i in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            idx = base + i
            x_val = al.convert(x[idx], al.f32)
            w_val = al.convert(wt[i], al.f32)
            b_val = al.convert(bt[i], al.f32)

            normalized = (x_val - mean) * rstd
            result = normalized * w_val + b_val
            ot[idx] = al.convert(result, al.bf16)


def avelang_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Input tensor must be bfloat16"

    batch_size = x.shape[0]
    features = x.shape[1]
    dim1 = x.shape[2]
    dim2 = x.shape[3]
    N = features * dim1 * dim2

    x_contig = x.contiguous()
    w_bf16 = weight.contiguous().to(torch.bfloat16)
    b_bf16 = bias.contiguous().to(torch.bfloat16)

    eps = 1e-5
    num_tiles = (N + TILE_SIZE - 1) // TILE_SIZE

    # Phase 1: tile-level reduction
    partial_sum = torch.empty((batch_size, num_tiles), dtype=torch.float32, device=x.device)
    partial_sq = torch.empty((batch_size, num_tiles), dtype=torch.float32, device=x.device)

    layernorm_reduce_kernel[lambda: ((batch_size * num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contig, partial_sum, partial_sq, batch_size, N, num_tiles
    )

    # Phase 2: aggregate across tiles
    mean_out = torch.empty((batch_size,), dtype=torch.float32, device=x.device)
    rstd_out = torch.empty((batch_size,), dtype=torch.float32, device=x.device)

    layernorm_aggregate_kernel[lambda: ((batch_size, 1, 1), (BLOCK_SIZE, 1, 1))](
        partial_sum, partial_sq, mean_out, rstd_out, batch_size, num_tiles, N, eps
    )

    # Phase 3: apply normalization
    out = torch.empty_like(x_contig)

    layernorm_apply_kernel[lambda: ((batch_size * num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contig, w_bf16, b_bf16, out, mean_out, rstd_out, batch_size, N, num_tiles
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using AveLang DSL.
    """
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.ln.weight.data
        bias = self.ln.bias.data

        x_bf16 = x.to(dtype=torch.bfloat16, device=x.device).contiguous()

        result = avelang_layernorm(x_bf16, weight, bias)
        return result.to(x.dtype)
```
