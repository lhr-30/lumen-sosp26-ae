---
id: kernel-gemm
title: "GEMM Kernels"
type: kernel
description: "Dense matrix multiplication kernel characteristics."
operators: [gemm, matmul, linear]
tags: [tiling, tensor-core, mfma, epilogue]
architectures: [gfx940, gfx942, gfx950]
---

# GEMM Kernels

GEMM computes `C = A x B` or a linear-layer equivalent. Good kernels tile M/N, stream K, reuse operands, and accumulate in a wider type.

## Characteristics

- Output work is naturally tiled over M and N; each CTA owns one or more output tiles.
- The K dimension is streamed through the CTA tile and should be sized to balance reuse, shared-memory capacity, and register pressure.
- Tensor-core or matrix-core instructions require fixed per-wave operand layouts and accumulator mappings.
- Large M/N/K shapes benefit from persistent scheduling, grouped tile ordering, and epilogue fusion.
- Boundary handling is required when dimensions are not multiples of the tile sizes.

## Applicability

Use for dense matmul, batched matmul, or linear layers. Fuse cheap elementwise epilogues that only need the current output.

Avoid a pure GEMM strategy when the operator is dominated by reductions, gathers/scatters, sparse indexing, or irregular control flow that prevents regular operand reuse.

## Related Techniques

- [MFMA BF16 GEMM tiling](../techniques/mfma-bf16-gemm-tiling.md)
- [Vectorized global loads](../techniques/vectorized-global-loads.md)
- [Epilogue fusion](../techniques/epilogue-fusion.md)
- [Tile scheduling](../techniques/tile-scheduling.md)
- [Software pipelining and multi-stage buffering](../techniques/pipeline-stages.md)
