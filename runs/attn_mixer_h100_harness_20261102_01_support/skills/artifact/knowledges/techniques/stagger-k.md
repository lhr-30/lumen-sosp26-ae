---
id: technique-stagger-k
title: "Stagger-K Scheduling"
architectures: [gfx940, gfx942, gfx950]
---

Stagger-K scheduling rotates the starting K tile of each GEMM CTA while preserving the same complete K reduction. Use it when adjacent CTAs begin at the same K offset and operand strides create memory-system hotspots. AMD's ROCm workload guidance calls out GEMM stride patterns, especially strides that are multiples of 512 bytes, as a source of unfavorable memory behavior: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html.

The technique changes the phase of global-memory accesses, not the GEMM tile math. Instead of every CTA loading K tiles in order `0, 1, 2, ...`, choose a per-CTA start tile from the CTA coordinates, then walk the K dimension circularly until all tiles have been consumed.

Make the policy configurable: a mask controls how many CTA-coordinate bits participate, a stride controls spacing between start tiles, and a mapping selects whether the phase comes from `gid_m`, `gid_n`, or another tile-order coordinate. A zero mask should disable the optimization.

```c++
unsigned stagger_data = use_n_for_phase ? gid_n : gid_m;

if constexpr (stagger_mask > 0) {
    k_now = (stagger_data & stagger_mask) * stagger_stride;
    if (k_now >= k_total) {
        k_now = 0;
    }

    uint64_t k_offset = k_now * kGroupK / kVecSize * sizeof(uint4);
    r_a.v.ptr += k_offset;
    r_b.v.ptr += k_offset;
}

uint64_t advance_k() {
    uint64_t go = kGroupK / kVecSize * sizeof(uint4);
    if constexpr (stagger_mask > 0) {
        if (++k_now == k_total) {
            go = kGroupK / kVecSize * sizeof(uint4) -
                 k / kVecSize * sizeof(uint4);
        }
    }
    return go;
}
```

Keep these invariants fixed:
  - Every CTA consumes every logical K tile exactly once.
  - The circular K order does not change output tile ownership.
  - Disable or clamp stagger when the computed start tile is outside `k_total`.
  - Do not change LDS layout, MFMA operand mapping, reduction shape, or writeback semantics.
  - Use compile-time config where possible so the no-stagger case compiles away.
