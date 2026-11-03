---
layout: layouts/base.njk
title: "Tutorial 2: GEMM"
description: Use shared memory, layouts, and vectorized loads to build a tiled BF16 GEMM.
---

# Tutorial 2: GEMM

## Naive Tensor-Tile GEMM

This tutorial uses matrix multiplication to introduce the Ave features that matter for GPU kernels: tensor indexing, shared memory, layouts, and vectorized memory movement. The kernels compute `C = A @ B^T`, where `A` is stored as an `M x K` matrix and `B` is stored as an `N x K` matrix.

Start with a direct implementation. Each block computes one `16 x 16` tile of `C`, using a one-dimensional `thread_id(0)` range. Each thread maps its linear thread id to one tile row and one tile column, walks across `K`, converts BF16 operands to `f32`, and writes one output element.

```python
import avelang
import avelang.language as al

@avelang.jit
def gemm_naive_bf16(
    A: al.Tensor((128, 128), al.bf16),
    B: al.Tensor((128, 128), al.bf16),
    C: al.Tensor((128, 128), al.f32),
):
    K = 128

    tid = al.thread_id(0)
    local_row = tid >> 4
    local_col = tid & 15

    row = al.block_id(1) * 16 + local_row
    col = al.block_id(0) * 16 + local_col

    acc = al.convert(0.0, al.f32)

    for k in al.range(K):
        a = al.convert(A[row, k], al.f32)
        b = al.convert(B[col, k], al.f32)
        acc = acc + a * b

    C[row, col] = acc
```

`A`, `B`, and `C` are all two-dimensional `128 x 128` tensors here. Ave uses tensor and tile concepts similar to [CuTE](https://docs.nvidia.com/cutlass/latest/index.html): a tensor describes a multidimensional memory region with a shape and optional strides. In full form, a tensor type is written as `al.Tensor((s0, s1, ...), (t0, t1, ...), type)`, where `(s0, s1, ...)` is the shape and `(t0, t1, ...)` is the stride. Subscript coordinates `(c0, c1, ...)` map to a linear index as:

```
idx = \sum_{i=0}^n c_i t_i
```

For contiguous tensors, the strides can be inferred and are omitted for brevity. Since this kernel computes `A @ B^T`, the inner loop multiplies `A[row, k]` with `B[col, k]`.

Launch the naive kernel with one block per `16 x 16` output tile and 256 one-dimensional threads per block:

```python
import torch

M = 128
N = 128
K = 128

A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
B = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
C = torch.empty((M, N), dtype=torch.float32, device="cuda")

gemm_naive_bf16[lambda: ((8, 8, 1), (256, 1, 1))](A, B, C)
expected = A.to(torch.float32) @ B.to(torch.float32).T
torch.testing.assert_close(C.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2)
```

## Using Shared Memory

The naive kernel reloads the same `A` and `B` values from global memory for many output elements. A straightforward optimization is to have the whole thread block cooperatively stage tiles of `A` and `B` in shared memory. For `C = A @ B^T`, both source tiles are contiguous along `K`: `A[row, k + kk]` and `B[col, k + kk]`.

The shared `a_tile` is indexed as `(local_m, local_k)`. The shared `b_tile` is indexed as `(local_n, local_k)` because `B` is stored as `N x K`.

```python
@avelang.jit
def gemm_tiled_16(
    A: al.Tensor((128, 128), al.bf16),
    B: al.Tensor((128, 128), al.bf16),
    C: al.Tensor((128, 128), al.f32),
):
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16
    K_TILES = 8

    tid = al.thread_id(0)
    ty = tid >> 4
    tx = tid & 15

    row = al.block_id(1) * BLOCK_M + ty
    col = al.block_id(0) * BLOCK_N + tx

    a_tile = al.make_shared((16, 16), al.bf16)
    b_tile = al.make_shared((16, 16), al.bf16)

    acc = al.convert(0.0, al.f32)

    for kt in al.range(K_TILES):
        k = kt * BLOCK_K

        a_tile[ty, tx] = A[row, k + tx]
        b_tile[ty, tx] = B[al.block_id(0) * BLOCK_N + ty, k + tx]

        al.syncthreads()

        for kk in al.range(BLOCK_K):
            a = al.convert(a_tile[ty, kk], al.f32)
            b = al.convert(b_tile[tx, kk], al.f32)
            acc = acc + a * b

        al.syncthreads()

    C[row, col] = acc
```

The shared-memory kernel uses the same launch shape and reference result:

```python
gemm_tiled_16[lambda: ((8, 8, 1), (256, 1, 1))](A, B, C)
expected = A.to(torch.float32) @ B.to(torch.float32).T
torch.testing.assert_close(C.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2)
```

## Vectorized Memory Accesses

The shared-memory kernel still moves BF16 elements one at a time. To use global memory bandwidth more efficiently, load 2-byte BF16 elements as 16-byte vectors. In this `A @ B^T` layout, the contiguous dimension is `K` for both operands: rows of `A` are contiguous, and rows of the stored `B` matrix are contiguous as well. That makes the global-to-shared tile load the natural place to vectorize.

Ave supports this by changing the logical view of a tensor with `al.view()`. The next kernel views each `128`-element BF16 row as `16` groups of `4` `i32` words. Each group is `4 * 4 = 16` bytes, representing the same data as `8` BF16 values. Shared memory uses the same packed `i32` layout, so both the global load and the shared-memory store are vectorized.

For clarity, this version stages the full `K = 128` dimension at once. A `16 x 16` block has exactly `16` rows times `16` packed K-vectors, so every one-dimensional thread performs one vectorized tile load for `A` and one for `B`.

```python
@avelang.jit
def gemm_tiled_vec8_bf16(
    A: al.Tensor((128, 128), al.bf16),
    B: al.Tensor((128, 128), al.bf16),
    C: al.Tensor((128, 128), al.f32),
):
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 128

    tid = al.thread_id(0)
    ty = tid >> 4
    tx = tid & 15

    row = al.block_id(1) * BLOCK_M + ty
    col = al.block_id(0) * BLOCK_N + tx

    A_vec = al.view(A, al.Tensor((128, 16, 4), al.i32))
    B_vec = al.view(B, al.Tensor((128, 16, 4), al.i32))

    a_tile = al.make_shared((16, 16, 4), al.i32)
    b_tile = al.make_shared((16, 16, 4), al.i32)

    acc = al.convert(0.0, al.f32)

    load_row = tid >> 4
    load_vec = tid & 15

    a_global_row = al.block_id(1) * BLOCK_M + load_row
    b_global_row = al.block_id(0) * BLOCK_N + load_row

    a_tile[load_row, load_vec] = A_vec[a_global_row, load_vec]
    b_tile[load_row, load_vec] = B_vec[b_global_row, load_vec]

    al.syncthreads()

    for kk in al.range(BLOCK_K):
        k_vec = kk >> 3
        k_word = (kk & 7) >> 1
        k_half = kk & 1

        a_words = a_tile[ty, k_vec]
        b_words = b_tile[tx, k_vec]
        a_frag = al.view(a_words, al.Tensor((4, 2), al.bf16))
        b_frag = al.view(b_words, al.Tensor((4, 2), al.bf16))

        a = al.convert(a_frag[k_word, k_half], al.f32)
        b = al.convert(b_frag[k_word, k_half], al.f32)
        acc = acc + a * b

    C[row, col] = acc
```

The views do not copy or rearrange `A` or `B`. They reinterpret each contiguous BF16 row as packed `i32` words, exposing the `K` dimension as `(vector, word)`. The shared tiles use the same `(row, vector, word)` structure, so a vector loaded from global memory can be assigned directly into shared memory. During accumulation, `al.view()` reinterprets each four-word packed vector as a `(4, 2)` BF16 fragment so the scalar dot product can select the requested `K` lane.

The Ave compiler recognizes accesses to four packed `i32` words and lowers them to a single 16-byte vectorized load.

```python
gemm_tiled_vec8_bf16[lambda: ((8, 8, 1), (256, 1, 1))](A, B, C)
expected = A.to(torch.float32) @ B.to(torch.float32).T
torch.testing.assert_close(C.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2)
```
