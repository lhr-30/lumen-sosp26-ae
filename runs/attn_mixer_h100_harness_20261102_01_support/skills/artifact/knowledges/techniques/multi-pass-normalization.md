---
id: technique-multi-pass-normalization
title: "Multi-Pass Normalization"
type: technique
description: "Normalize large rows with partial, aggregate, and apply passes."
tags: [normalization, reduction, tiling, memory-bandwidth]
architectures: [gfx940, gfx942, gfx950]
---

# Multi-Pass Normalization

Split large normalization into partial reduction, aggregate reduction, and apply passes.

## Applicability

- LayerNorm, RMSNorm, variance, or standardization over large feature dimensions.
- Normalized extents larger than one CTA can reduce efficiently.
- Cases where intermediate statistics can be stored compactly per row and tile.

## Pattern

- Pass 1 computes partial aggregates for each row tile.
- Pass 2 combines partial aggregates into row-level statistics such as mean and reciprocal standard deviation.
- Pass 3 rereads the input, applies normalization and optional affine parameters, then writes the output.

## Tradeoffs

The design increases global memory traffic and requires temporary buffers, but it bounds per-CTA work and avoids long serial loops. It is usually best when the normalized extent is large enough that a single-pass CTA-local implementation would underutilize threads or exceed local storage limits.

## Related

- [LayerNorm kernels](../kernels/layernorm.md)
- [Shared-memory reduction tree](shared-memory-reduction-tree.md)
