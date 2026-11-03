---
layout: layouts/base.njk
title: Debugging
description: Debug Ave kernels, compiler input, generated IR, and diagnostics.
---

# Debugging

Ave debugging usually starts with a small file-backed `@avelang.jit` kernel and the repository IR dump tools. Keep the repro close to the failing behavior and avoid import-time side effects; the dump tools execute the module to materialize JIT functions.

## First Checks

- Rebuild with `ninja -C build` after native changes.
- Run with `PYTHONPATH=build/python:python` from a source checkout.
- Start from `test/primitives` when isolating language behavior.
- Use `test/examples` for larger kernel regressions.

## Inspect Generated Code

```bash
export PYTHONPATH=build/python:python

python tools/dump_llvm_ir.py \
  test/examples/gemv/test_axpy.py \
  axpy \
  --target-triple nvptx64-nvidia-cuda \
  --target-chipset sm_80 \
  -O 2 \
  -o /tmp/axpy.ll

python tools/dump_assembly.py \
  test/examples/gemv/test_axpy.py \
  axpy \
  --target-triple nvptx64-nvidia-cuda \
  --target-chipset sm_80 \
  -O 2 \
  -o /tmp/axpy.s
```

For AMDGPU, put the ROCm LLVM tools on `PATH` when disassembling final code objects:

```bash
export PATH=/opt/rocm/llvm/bin:$PATH
```
