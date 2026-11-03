---
layout: layouts/base.njk
title: Functions And Control Flow
description: JIT functions, launches, control flow, runtime shapes, and specialization in Ave.
---

# Functions And Control Flow

Ave programs are Python functions compiled with the Ave JIT. Programs usually import the top-level package as `avelang` and the language package as `al`:

```python
import avelang
import avelang.language as al
```

## JIT Functions

`@avelang.jit` marks a Python function as an Ave JIT function. A JIT function can be launched as a GPU kernel, and it can also be called as a helper from another JIT function.

```python
@avelang.jit
def axpy(
    a: al.i32,
    b: al.i32,
    n: al.i32,
    x: al.Tensor((32,), al.i32),
    y: al.Tensor((32,), al.i32),
):
    idx = al.block_id(0) * al.block_dim(0) + al.thread_id(0)
    if idx < n:
        y[idx] = a * x[idx] + b
```

Functions called from JIT code must also be decorated with `@avelang.jit`. JIT functions must be defined in a Python file so the compiler can recover stable source code.

## Arguments

Function arguments use Ave type annotations. Scalar annotations such as `al.i32` describe scalar values. `al.Tensor(shape, type)` describes tensor-like memory with a compile-time shape. `al.Pointer(type)` describes raw pointer arguments whose logical shape will be recovered inside the kernel. See [Builtin Types](../builtin-types/) for the type forms.

## Launch

Launch a decorated kernel by providing grid and block dimensions. The launch provider returns `((grid_x, grid_y, grid_z), (block_x, block_y, block_z))`.

```python
axpy[lambda: ((1, 1, 1), (256, 1, 1))](a, b, n, x, y)
```

Backend options are passed as keyword arguments:

```python
kernel[lambda: ((1, 1, 1), (256, 1, 1))](x, num_warps=8)
```

## GPU Coordinates

Ave exposes GPU coordinates through `al.block_id`, `al.thread_id`, `al.block_dim`, and `al.grid_dim`. The dimension argument must be a constant integer `0`, `1`, or `2`.

```python
idx = al.block_id(0) * al.block_dim(0) + al.thread_id(0)
```
## Conditionals

Ave uses Python `if` syntax for conditional execution. Conditions can depend on runtime values such as thread coordinates and problem sizes.

```python
if idx < n:
    out[idx] = value
```

Conditionals lower to GPU control flow.

## Loops

Use `al.range` for compiler-lowered loops.

```python
for k in al.range(128):
    acc = acc + x[k]
```

Loop bounds may use supported runtime expressions:

```python
for i in al.range((n + al.block_dim(0) - 1) // al.block_dim(0)):
    ...
```

Use `al.constexpr` for tile sizes, static shapes, and loop bounds that should be known during specialization. Specialization lets the compiler see static storage sizes and unroll-friendly bounds while still accepting runtime problem sizes.

## Runtime Shapes

Generic kernels combine runtime problem sizes with specialization-time constants. Runtime sizes are passed as scalar arguments, while tile sizes are often passed as `al.constexpr`.

```python
@avelang.jit
def matrix_kernel(
    A_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.f32),
    m: al.u32,
    k: al.u32,
    BLOCK_M: al.constexpr,
):
    ...
```

## Launch-Time Tuning

The current tuning workflow is explicit. Choose launch geometry, specialize tile sizes, and pass backend options at launch.

```python
for block in [64, 128, 256]:
    kernel[lambda block=block: ((1, 1, 1), (block, 1, 1))](
        x,
        n,
        BLOCK=block,
        num_warps=block // 32,
    )
```

Tunable inputs include grid and block dimensions, tile sizes passed as `al.constexpr`, layout shapes and strides, and backend options such as `num_warps`.

## Python Subset

Ave parses a supported subset of Python source. Supported patterns include local variables, assignments, tuple values for shapes and launches, calls to compiler-registered language functions, calls to other JIT functions, and selected Python builtins such as `range`, `len`, `int`, `float`, `min`, and `max`.
