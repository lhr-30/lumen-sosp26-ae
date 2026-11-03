---
id: technique-shared-memory-reduction-tree
title: "Shared-Memory Reduction Tree"
type: technique
description: "Combine per-thread partials through shared memory."
tags: [reduction, shared-memory, synchronization]
architectures: [gfx940, gfx942, gfx950]
---

# Shared-Memory Reduction Tree

Combine per-thread partials into CTA-level aggregates through shared memory.

## Applicability

- Reductions where each CTA owns a complete output element or partial aggregate.
- Associative operations such as sum, max, min, and sum of squares.
- Normalization kernels that need row-level statistics.
- Workloads where cross-thread communication inside a CTA is cheaper than serial accumulation.

## Pattern

- Each thread accumulates a local partial over a strided slice of input.
- Store local partials into shared memory.
- Repeatedly halve the active thread count and combine with a fixed offset.
- Synchronize between reduction levels when later reads depend on earlier writes.
- Let one thread write the final aggregate or partial aggregate.

## Tradeoffs

Tree reductions reduce serial work but add shared-memory traffic and barriers. For very small reductions, a serial per-thread loop can be simpler and faster. For very large reductions, write partial aggregates and use a second pass.

## Related

- [Axis reduction kernels](../kernels/axis-reduction.md)
- [LayerNorm kernels](../kernels/layernorm.md)
