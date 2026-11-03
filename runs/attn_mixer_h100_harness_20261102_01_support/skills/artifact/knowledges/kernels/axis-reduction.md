---
id: kernel-axis-reduction
title: "Axis Reduction Kernels"
type: kernel
description: "Reduction kernel characteristics for collapsing tensor axes."
operators: [sum, max, min, norm, softmax, argmax, logsumexp]
tags: [reduction, memory-bound, shared-memory]
architectures: [gfx940, gfx942, gfx950]
---

# Axis Reduction Kernels

Axis reductions collapse one or more dimensions. Simple reductions are usually memory-bound; long serial scans are latency-sensitive.

## Characteristics

- Output coordinates identify the non-reduced dimensions; each output element reads all values along the reduced axis.
- Contiguous reduced axes provide coalesced reads and simple loop structure.
- Non-contiguous reductions often need layout transforms, vectorized striding, or different thread ownership to avoid scattered memory access.
- Short reductions can map one output element per thread or small thread group.
- Long reductions usually need parallel accumulation and a tree reduction inside a CTA or across multiple passes.

## Applicability

Use for max, min, sum, mean, norm, argmax, softmax components, or logsumexp with known axes and regular output shape.

Prefer a multi-pass design when a single CTA cannot cover the reduced axis efficiently or when numerical stability requires multiple aggregate values.

## Related Techniques

- [Shared-memory reduction tree](../techniques/shared-memory-reduction-tree.md)
- [Vectorized global loads](../techniques/vectorized-global-loads.md)
- [Multi-pass normalization](../techniques/multi-pass-normalization.md)
