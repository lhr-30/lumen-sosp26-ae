---
id: technique-mfma-bf16-gemm-tiling
title: "MFMA BF16 GEMM Tiling"
type: technique
description: "Build BF16 GEMM CTA tiles from MFMA wave tiles."
tags: [gemm, mfma, bf16, tiling]
architectures: [gfx940, gfx942, gfx950]
---

# MFMA BF16 GEMM Tiling

Build a CTA tile from fixed per-wave MFMA tiles. Keep tile shape, warp ownership, operand packing, and writeback consistent.

## Applicability

- Dense BF16 GEMM or linear-layer kernels.
- Shapes large enough to amortize shared-memory staging and CTA setup.
- K dimension divisible by the chosen instruction K slice, or explicitly guarded.
- Operators where the epilogue is elementwise and can be applied from accumulator registers.

## Pattern

- Choose a CTA tile such as 128x128 with a K step sized for operand reuse.
- Split the CTA into a small warp grid, commonly 2 x 2, and assign each warp one output subtile.
- Stage A and B tiles into shared memory before issuing matrix instructions.
- Keep the per-wave operand swizzle and accumulator mapping fixed.
- Accumulate in f32 when using BF16 inputs, then convert at store time.

## Tradeoffs

Larger tiles increase operand reuse but consume more shared memory and registers. More output tiles per warp improve arithmetic intensity but increase accumulator pressure. Smaller K stages reduce shared-memory footprint but increase synchronization and loop overhead.

## Related

- [GEMM kernels](../kernels/gemm.md)
- [MFMA hardware notes](../hardware/mfma.md)
