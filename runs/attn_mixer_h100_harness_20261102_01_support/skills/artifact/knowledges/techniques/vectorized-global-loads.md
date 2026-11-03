---
id: technique-vectorized-global-loads
title: "Vectorized Global Loads"
type: technique
description: "Load adjacent elements with wider memory operations."
tags: [memory, vectorization, coalescing, shared-memory]
architectures: [gfx940, gfx942, gfx950]
---

# Vectorized Global Loads

Load multiple adjacent elements per instruction to reduce load count and stage regular tiles efficiently.

## Applicability

- Contiguous or predictably strided memory regions.
- Element counts and byte offsets aligned for the selected vector width.
- Tiled kernels that stage operands into shared memory before compute.
- Memory-bound kernels where scalar load overhead is visible.

## Pattern

- Assign each thread one or more fixed-width vector loads.
- Compute vector offsets from tile coordinates rather than scalar element coordinates.
- Store the loaded vector into shared memory in the layout expected by the compute phase.
- Guard or peel boundary tiles when dimensions are not multiples of the vector width.

## Tradeoffs

Vectorized loads improve bandwidth utilization only when alignment and coalescing are preserved. They can make boundary handling more complex and may waste bandwidth on partially valid vectors if edge tiles are common.

## Related

- [GEMM kernels](../kernels/gemm.md)
- [Axis reduction kernels](../kernels/axis-reduction.md)
- [Buffer instructions](../hardware/buffer-instructions.md)
