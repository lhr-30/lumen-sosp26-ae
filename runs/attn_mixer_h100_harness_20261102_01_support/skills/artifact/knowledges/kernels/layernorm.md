---
id: kernel-layernorm
title: "LayerNorm Kernels"
type: kernel
description: "LayerNorm kernel characteristics."
operators: [layernorm, normalization]
tags: [normalization, reduction, affine, memory-bound]
architectures: [gfx940, gfx942, gfx950]
---

# LayerNorm Kernels

LayerNorm reduces each row for mean/variance, then applies normalization and optional affine scale/bias.

## Characteristics

- Each normalized row needs at least the sum and sum of squares, or an equivalent stable variance computation.
- The apply phase rereads input and writes every output element, so memory bandwidth is a primary constraint.
- Small normalized shapes can fit in one CTA and often use one-pass or two-pass CTA-local reductions.
- Large normalized shapes require tiling, partial aggregates, and a second aggregation pass before applying normalization.
- Intermediate mean and reciprocal standard deviation values are reused across all elements in the row.

## Applicability

Use when normalization is independent per row or batch item and affine parameters broadcast over the normalized extent.

Use a multi-pass strategy when the normalized extent is larger than a single CTA can reduce without excessive serial work, shared memory, or register pressure.

## Related Techniques

- [Shared-memory reduction tree](../techniques/shared-memory-reduction-tree.md)
- [Multi-pass normalization](../techniques/multi-pass-normalization.md)
- [Vectorized global loads](../techniques/vectorized-global-loads.md)
