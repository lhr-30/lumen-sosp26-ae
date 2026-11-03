---
id: technique-epilogue-fusion
title: "Epilogue Fusion"
type: technique
description: "Fuse elementwise post-processing into kernel writeback."
tags: [fusion, gemm, elementwise, bandwidth]
architectures: [gfx940, gfx942, gfx950]
---

# Epilogue Fusion

Apply cheap elementwise work before the final global store to avoid extra launches and output rereads.

## Applicability

- GEMM, convolution, and linear-layer kernels with elementwise post-processing.
- Bias addition, scaling, activation, clamping, residual add, or dtype conversion.
- Operations that depend only on the current output element and small broadcast operands.

## Pattern

- Keep accumulators in the compute type until all fused operations are complete.
- Load broadcast operands such as bias near the writeback loop.
- Apply scalar arithmetic and activation in the output mapping used by the compute tile.
- Convert to the output dtype only at the final store.

## Tradeoffs

Fusion saves bandwidth and launch overhead, but it increases kernel specialization and can add register pressure. Avoid fusing operations that require global communication or large additional tensors unless the memory-access pattern remains efficient.

## Related

- [GEMM kernels](../kernels/gemm.md)
