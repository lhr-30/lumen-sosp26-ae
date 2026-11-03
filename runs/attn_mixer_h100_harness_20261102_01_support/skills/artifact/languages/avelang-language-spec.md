---
id: language-avelang-spec
title: "AveLang Language Spec"
type: language
description: "AveLang imports, JIT syntax, types, launches, memory APIs, and host wrappers."
---

AveLang syntax and API reference.

## Import requirement
Start files with:
```python
import avelang
import avelang.language as al
```

## Public surface map

### Runtime-visible Python exports

These names exist as normal Python objects and are safe in annotations or host code:

- `avelang.jit`: JIT-kernel decorator. It returns a launchable object, not a normal Python callable.
- `avelang.language` as `al`: the standard namespace alias used throughout the repo.
- `al.i8`, `al.i16`, `al.i32`, `al.i64`: signed integer scalar dtypes.
- `al.u1`: 1-bit unsigned integer / bool-like scalar dtype. Treat it as a real 1-bit type, not as `u8`.
- `al.u8`, `al.u16`, `al.u32`, `al.u64`: unsigned integer scalar dtypes.
- `al.f16`, `al.bf16`, `al.f32`, `al.f64`: floating-point scalar dtypes.
- `al.constexpr`:
  - in annotations, marks a compile-time kernel parameter
  - as `al.constexpr(value)`, wraps a Python global/nonlocal compile-time value
- `al.dynamic`: exported sentinel for dynamic shapes, but repo code prefers `al.Pointer + al.make_tensor` for runtime sizes and strides.
- `al.Tensor(shape, dtype)`: static row-major tensor type used in annotations and `al.view(..., al.Tensor(...))` casts.
- `al.Pointer(dtype)`: raw pointer type used for runtime-shaped buffers.
- `al.block_id(dim)`: block index along launched grid dimension `dim`.
- `al.block_dim(dim)`: block extent along launched block dimension `dim`.
- `al.thread_id(dim)`: thread index inside the current block/workgroup along dimension `dim`.
- `al.grid_dim(dim)`: total grid extent along dimension `dim`.
- `al.shuffle(value, offset, width)`: absolute-lane shuffle.
- `al.shuffle_up(value, offset, width)`: upward lane shuffle.
- `al.shuffle_down(value, offset, width)`: downward lane shuffle.
- `al.shuffle_xor(value, offset, width)`: XOR-lane shuffle.
- `al.min(lhs, rhs)`: integer minimum. Signed vs unsigned behavior comes from operand type info.
- `al.max(lhs, rhs)`: integer maximum. Signed vs unsigned behavior comes from operand type info.
- `al.abs(x)`: absolute value for scalar GPU floating-point values.
- `al.exp2(x)`: floating-point exponential builtin computing `2**x`.
- `al.exp(x)`: floating-point exponential builtin computing `e**x`. AMDGPU lowering maps `f32` to `llvm.amdgcn.exp`.
- `al.tanh(x)`: floating-point hyperbolic tangent builtin.
- `al.log(x)`: floating-point natural logarithm builtin.
- `al.log2(x)`: floating-point base-2 logarithm builtin.
- `al.erf(x)`: floating-point error-function builtin. Compiler-exposed, lightly tested — accuracy on gfx942 BF16 may be insufficient for GELU.
- `al.sqrt(x)`: floating-point square-root builtin.

### Compiler-recognized DSL symbols

These are valid inside `@avelang.jit` function bodies because the AST lowerer recognizes them. Many are not real Python functions on `avelang.language`, so never rely on them outside JIT code:

- `al.convert`
- `al.bitcast`
- `al.range`
- `al.syncthreads`
- `al.printf`
- `al.make_tensor`
- `al.make_local`
- `al.make_shared`
- `al.make_layout`
- `al.subview`
- `al.view`
- `al.full`
- `al.amdgpu.*`
- `al.nvvm.*`

### Tooling-only public APIs

`avelang.compiler.compile`, `avelang.compiler.make_backend`, `avelang.compiler.ASTSource`, and `avelang.compiler.CompiledKernel` are public, but they are testing/tooling APIs. Do not emit them in generated model code unless the task is explicitly about compiler internals.

## Required file and import shape

- Always start generated files with `import avelang` and `import avelang.language as al`.
- Missing `import avelang.language as al` causes launch-time failure; there is an explicit error test for this.
- `@avelang.jit` functions must live in a Python file on disk so the compiler can recover stable source code.
- The kernel body is parsed from Python AST. Body-only DSL names like `al.make_shared` are not executed as normal Python code.

## Kernel declaration rules

- Kernels are Python functions decorated with `@avelang.jit`.
- Kernel parameters may be:
  - Scalar dtypes, for example `x: al.i32`
  - Static tensors, for example `x: al.Tensor((128, 64), al.f16)`
  - Raw pointers, for example `x_ptr: al.Pointer(al.bf16)`
  - Compile-time parameters, for example `BLOCK_M: al.constexpr`
- Use `al.Tensor(...)` when the shape contract is known statically.
- Use `al.Pointer(dtype)` plus runtime shape/stride scalars and `al.make_tensor(...)` when sizes or strides are dynamic.
- Launched kernel entrypoints should not use explicit non-void return types. Write externally visible results into output buffers.
- `return` with no value is valid for early exit and is used in repo tests.
- Helper functions may also be `@avelang.jit` and can be called from kernels.
- Nested local helper functions inside a kernel are supported in tests.
- Verified helper returns include scalar tuples, a single local tensor, and tuples of local tensors. Launched top-level kernels should still write externally visible results into output buffers.

Example:

```python
import avelang
import avelang.language as al

@avelang.jit
def saxpy(x: al.Tensor((64,), al.f32), y: al.Tensor((64,), al.f32), a: al.f32):
    tid = al.thread_id(0)
    y[tid] = a * x[tid] + y[tid]
```

## Launch and specialization behavior

- Launch syntax is:

```python
kernel[lambda: ((grid_x, grid_y, grid_z), (block_x, block_y, block_z))](...)
```

- The launch lambda is called with no arguments. Unlike Triton meta-launch patterns, do not expect meta-parameters to be passed into the lambda.
- Always provide explicit 3-tuples for grid and block.
- The only backend launch option verified in this repo is `num_warps=...`.
- `device`, `device_type`, and `stream` kwargs are explicitly rejected as deprecated.
- Compilation is cached per device and specialization.
- `al.constexpr` parameters and captured global `al.constexpr(...)` values contribute to specialization and cache keys.
- Calling an `@avelang.jit` function like a normal Python function from host code raises an error. Use launch syntax.

## Type system and scalar semantics

- Exported scalar types: `i8/i16/i32/i64`, `u1/u8/u16/u32/u64`, `f16/bf16/f32/f64`.
- `u1` is the bool-like integer type. Recent lowering work keeps 1-bit values as 1-bit instead of silently widening them to 8-bit storage.
- `al.Tensor((shape...), dtype)` means a statically typed row-major memref-like tensor.
- Although some prose docs mention `al.Tensor(shape, stride, dtype)`, the current Python runtime type constructor and tests use the two-argument form. For explicit strides or dynamic layouts, use `al.make_layout(...)` with `al.make_tensor(...)` or `al.view(...)`.
- `al.Pointer(dtype)` is a raw pointer-like argument. It is normally wrapped immediately with `al.make_tensor`.
- `al.constexpr` in parameter annotations marks compile-time parameters.
- `al.constexpr(value)` wraps a Python global/nonlocal value as compile-time data. The JIT currently infers compile-time types from Python `bool`, `int`, and `float`.
- `al.dynamic` is exported as a sentinel, but there are no repo examples using dynamic `al.Tensor` annotations. Prefer `al.Pointer + al.make_tensor` for runtime shapes.
- `void` exists internally in `core.py`, and some prose docs mention `al.void`, but the current `avelang.language` package does not export it. Do not use `al.void` in generated kernels.
- Use `al.convert` for explicit promotion or demotion. Do not rely on implicit dtype conversion.
- Assignment and store sites are stricter than `al.convert`: implicit demotion is still rejected in general, including augmented assignment such as `dst += value`.
- One narrow exception now exists for stores: a floating-point constant may be implicitly narrowed to a smaller floating-point destination type when the value round-trips exactly with no precision loss. Do not rely on this for non-constant values or for precision-losing literals; use `al.convert(...)` instead.

## Execution model primitives

- `al.thread_id(axis)`, `al.block_id(axis)`, `al.block_dim(axis)`, `al.grid_dim(axis)` are the core launch-space queries.
- Use axis values `0`, `1`, or `2`.
- AveLang does not provide Triton-style implicit program indexing. Compute flattened or tiled coordinates yourself.
- `al.syncthreads()` is the block/workgroup barrier and takes no arguments.
- `al.printf(fmt, *args)` is compiler-exposed. Constraints from the implementation:
  - first argument must be a non-empty string literal
  - extra value arguments are optional
  - MLIR generation tests cover it, but it is still a debugging aid; avoid it in performance kernels

## Control flow and function-call rules

- `if` / `else` is supported.
- Integer conditions are coerced to boolean by comparing against zero.
- `for` loops must iterate over `al.range(...)`. Plain Python `range(...)` is not accepted.
- `al.range` supports the three Python-like forms:
  - `al.range(stop)`
  - `al.range(start, stop)`
  - `al.range(start, stop, step)`
- Loop targets must be simple names, not tuple unpacking or other complex targets.
- Python unary operators are supported with current compiler semantics:
  - `-x` for float scalars/vectors, integer scalars/vectors, and index values
  - `+x` as a no-op
  - `not x` via boolean coercion against zero
- The compiler has a `while` lowering path, but the repo has no end-to-end `while` examples. Do not generate `while` unless you intend to validate it directly.
- Top-level helper callees should be `@avelang.jit`. Calling arbitrary top-level Python helpers from kernels is not a repo-proven pattern.
- Tuple unpacking from helper returns is verified:

```python
id_m, id_n = tuple_add(x, y)
```

## Core primitive semantics

- `al.convert(value, dtype)`: explicit numeric conversion. Demotion is allowed because the user requested it explicitly.
- `al.bitcast(value, dtype)`: reinterpret bits without changing the bit pattern.
- `al.min(lhs, rhs)` / `al.max(lhs, rhs)`: integer extrema. Signedness comes from type information. Do not treat them as verified floating-point min/max helpers.
- `al.abs(x)`: floating-point absolute value.
- `al.tanh(x)`: floating-point only. Lowers to `math.tanh`; AMDGPU lowering maps `f32` to `__ocml_tanh_f32`.
- `al.exp2(x)`: floating-point only. This is `2**x`. AMDGPU lowering maps `f32` to `llvm.amdgcn.exp2`.
- `al.exp(x)`: floating-point only. This is `e**x`. AMDGPU lowering maps `f32` to `llvm.amdgcn.exp`.
- `al.log(x)`: floating-point only. This is the natural logarithm.
- `al.log2(x)`: floating-point only. This is the base-2 logarithm.
- `al.erf(x)`: floating-point only. AMDGPU lowering maps `f32` to `__ocml_erf_f32`.
- `al.sqrt(x)`: floating-point only.
- `al.shuffle(value, offset, width)`: lane shuffle by absolute lane id.
- `al.shuffle_up(value, offset, width)`: upward lane shuffle.
- `al.shuffle_down(value, offset, width)`: downward lane shuffle.
- `al.shuffle_xor(value, offset, width)`: XOR-lane shuffle.
- Shuffle constraints:
  - `value` must be an int, float, or 1D vector of int/float
  - `offset` and `width` must be integer or index values

## Memory, layout, and view system

AveLang is not just a flat pointer DSL. Layout/view composition is a first-class part of the language and is intentionally close to CuTE/PyCuTe.

- `al.make_shared(shape, dtype)`
  - allocates workgroup/shared memory
  - shape must be compile-time static
  - row-major strides are synthesized automatically
  - dynamic shared memory is not supported
  - optional third argument `alignment` is supported, for example `al.make_shared((8,), al.i32, 128)`; it must be a positive power-of-two byte constant
- `al.make_local(shape, dtype)`
  - allocates private/register memory
  - shape must be compile-time static
  - row-major strides are synthesized automatically
  - optional third argument `alignment` follows the same rules as `al.make_shared`
- `al.make_layout(dims, strides)`
  - creates a layout descriptor
  - `dims` and `strides` must be tuples of the same arity
  - nested tuples are supported and are used for swizzled/tiled layouts
- `al.make_tensor(ptr, dtype, layout)`
  - wraps a raw pointer or i8 memref pointer with explicit shape/stride semantics
  - the third argument must come from `al.make_layout(...)`
  - this is the standard way to give a `al.Pointer(...)` argument tensor semantics
- `al.subview(base, offsets, sizes, strides)`
  - all three metadata arguments must be tuples
  - tuple lengths must match the base memref rank
  - offsets/sizes/strides may be static or runtime expressions
  - dimensions whose `size == 1` are rank-reduced in the result
- `al.view(memref, dtype, layout)`
  - remaps an existing memref through an explicit layout
  - preserves subview offsets correctly
  - the first argument must be a memref
  - the third argument must come from `al.make_layout(...)`
- `al.view(value, al.Tensor(...))`
  - reinterpret-casts a memref, scalar, or vector into a new tensor/vector view
  - for vector inputs, source and target must have the same total bitwidth
  - target shape must be static for vector inputs
  - constants are not accepted as the first argument
- `al.full(shape, fill_value, dtype)`
  - creates a private tensor filled with one scalar
  - shape must be a static tuple
  - fill value is converted to the target scalar dtype if needed

Indexing rules:

- Use tuple-style indexing, for example `buf[i, j]`.
- Direct indexing works on tensor arguments, shared memory, local memory, subviews, and layout views.
- Nested `make_layout` shapes support logical multi-indexing and also the repo-tested “nested-linear” pattern where a linear index expands across nested dimensions.

Example pattern:

```python
shared_words = al.make_shared((16,), al.u32)
left_words = al.subview(shared_words, (0,), (8,), (1,))
layout = al.make_layout((2, 2), (2, 1))
left = al.view(left_words, al.u32, layout)
left[0, 0] = al.convert(11, al.u32)
```

## AMDGPU namespace: `al.amdgpu.*`

Verified or compiler-exposed AMDGPU intrinsics:

- MFMA matmul instructions:
  - `al.amdgpu.mfma_16x16x16_f16_f32`
  - `al.amdgpu.mfma_16x16x16_bf16_f32`
  - `al.amdgpu.mfma_f32_16x16x16_bf16`
  - `al.amdgpu.mfma_32x32x8_bf16_f32`
  - `al.amdgpu.mfma_f32_32x32x8_bf16`
- Buffer/resource helpers:
  - `al.amdgpu.make_rsrc(tensor, range_bytes)`
  - `al.amdgpu.raw_buffer_load_x1(rsrc, vindex, soffset, aux)`
  - `al.amdgpu.raw_buffer_load_x2(rsrc, vindex, soffset, aux)`
  - `al.amdgpu.raw_buffer_load_x4(rsrc, vindex, soffset, aux)`
  - `al.amdgpu.raw_buffer_load_x1_lds(rsrc, lds_ptr, size, vindex, soffset, offset, aux)`
  - `al.amdgpu.raw_buffer_store_x1(vdata, rsrc, vindex, soffset, aux)`
  - `al.amdgpu.raw_buffer_store_x2(vdata, rsrc, vindex, soffset, aux)`
  - `al.amdgpu.raw_buffer_store_x4(vdata, rsrc, vindex, soffset, aux)`
- Scalar/control helpers:
  - `al.amdgpu.perm(hi, lo, selector)`
  - `al.amdgpu.rcp(x)`
  - `al.amdgpu.s_waitcnt(vmcnt, expcnt, lgkmcnt)`
  - `al.amdgpu.sched_group_barrier(mask, size, group_id)`

Important AMDGPU constraints from the implementation:

- MFMA intrinsics take vector operands. Use only the public intrinsic signatures and validate behavior directly in the current round.
- `make_rsrc` expects a tensor/memref and an integer/index byte range in `[0, 2^32 - 1]`.
- `raw_buffer_load_x{1,2,4}` expects `rsrc` to be `vector<4xi32>`.
- `raw_buffer_store_x1` expects an integer scalar payload; `x2/x4` expect `vector<2xi32>` / `vector<4xi32>`.
- `raw_buffer_load_x1_lds` is compiler-exposed and covered by MLIR generation tests. It requires exactly seven arguments `(rsrc, lds_ptr, size, vindex, soffset, offset, aux)`, `rsrc` as `vector<4xi32>`, `lds_ptr` as a memref, integer/index operands for the remaining metadata, and compile-time `size=4` plus `aux=0`.
- `perm` expects three 32-bit integer arguments.
- `rcp` expects `f32`.
- `s_waitcnt` requires compile-time integers with ranges `vmcnt=[0,63]`, `expcnt=[0,7]`, `lgkmcnt=[0,15]`.
- `sched_group_barrier` requires compile-time non-negative integers representable as `u32`.

## NVVM namespace: `al.nvvm.*`

Verified or compiler-exposed NVVM intrinsics:

- MMA:
  - `al.nvvm.mma_16x8x16_f16_f16`
  - `al.nvvm.mma_16x8x8_f16_f32`
- Load-matrix families:
  - `al.nvvm.ldmatrix_m8n8_x{1,2,4}_b16`
  - `al.nvvm.ldmatrix_m8n8_x{1,2,4}_b16_trans`
- Store-matrix families:
  - `al.nvvm.stmatrix_m8n8_x{1,2,4}_b16`
  - `al.nvvm.stmatrix_m8n8_x{1,2,4}_b16_trans`
  - In the current Python shim and intrinsic registry, only the `m8n8` / `b16` family is exported. Do not generate `m16n16` or `_b8` wrappers.

Important NVVM constraints from the implementation:

- `ldmatrix_*` takes one memref argument in workgroup/shared memory.
- The memref must have shape `8x8` for the currently exported `m8n8` wrappers.
- Element types must match bit width:
  - `_b16`: `f16`, `bf16`, or `i16`
- `stmatrix_*` takes `(ptr, source)` where source is `i32` for `x1` and `vector<num x i32>` for `x2/x4`.
- `mma_*` operands must be vector types. Use only the public intrinsic signatures and validate any fragment layout directly in the current round.

## Host-wrapper rules for KernelBench-style generation

- `ModelNew.forward` is the semantic entrypoint. Preserve the exact math, shapes, and output dtype semantics of the original `Model.forward`.
- Make tensors contiguous before launching AveLang kernels.
- Keep device moves explicit. If the wrapper accepts CPU inputs, move them to the active GPU backend before launch and move outputs back only when needed.
- Prefer static `al.Tensor(...)` signatures when the benchmark’s shape contract is fixed.
- Prefer `al.Pointer + al.make_tensor` when runtime lengths/strides come from the model input.
- Pass runtime metadata such as lengths, strides, block counts, or GQA ratios as scalar arguments.
- Use `num_warps=...` only when the kernel structure clearly depends on it or when existing examples do.
- Preserve reference accumulation semantics. For BF16 kernels, FP32 accumulation is common in AMD/NVIDIA matmul paths, but the output dtype should still match the model contract unless the benchmark explicitly expects a different dtype.

## Known failure patterns

These patterns have been observed to cause compilation hangs, MLIR lowering failures, HIP crashes,
or silent numerical errors in KernelBench runs. Avoid all of them.

### 1. `al.Tensor` with non-constexpr shape → compilation hang / timeout

`al.Tensor((shape), dtype)` requires every element of `shape` to be a **compile-time constant**
(a Python integer literal, or a value wrapped in `al.constexpr`).
Using Python module-level variables or expressions that reference runtime values causes the
AveLang AST lowerer to hang indefinitely while trying to resolve the shape.

```python
# ❌ M, K are Python variables — compiler hangs
@avelang.jit
def kernel(A: al.Tensor((M, K), al.f32), B: al.Tensor((K, N), al.f32)):
    ...

# ❌ BATCH_SIZE, DIM are module-level constants but not al.constexpr — same hang
@avelang.jit
def kernel(x: al.Tensor((BATCH_SIZE, DIM), al.f32)):
    ...

# ✅ Correct: use al.Pointer + al.make_tensor for any runtime-variable shape
@avelang.jit
def kernel(x_ptr: al.Pointer(al.bf16), m: al.i32, k: al.i32):
    layout = al.make_layout((m, k), (k, 1))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    ...
```

**Rule**: In KernelBench, `get_inputs()` returns tensors of arbitrary runtime shape.
Use `al.Pointer + al.make_tensor` by default.
Reserve `al.Tensor(static_shape)` only for shapes that are genuinely fixed compile-time
constants smaller than ~65536 elements per dimension.

### 2. `al.Tensor` with very large static shapes → MLIR codegen explosion / timeout

Even when the shape is a valid integer literal, huge extents cause the MLIR code generator
to produce enormous code, exceeding the 180 s eval timeout.

```python
# ❌ 1.6B-element tensor — codegen hangs
x: al.Tensor((1610612736,), al.f32)

# ❌ 2B-element 2-D tensor
x: al.Tensor((32768, 65535), al.f32)
```

## Do not invent APIs

Stay inside the API surface above. Common hallucinations from Triton/TVM/CUDA ports that are not verified AveLang APIs here:

- `al.load`, `al.store`, `al.arange`, `al.program_id`, `al.atomic_add`
- `tl.*` APIs from Triton
- `T.*` / `@T.prim_func` APIs from TVM/TIR for user-facing AveLang Python kernels
- ad hoc backend namespaces or MFMA names not present in the compiler
