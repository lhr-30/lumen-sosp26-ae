---
id: technique-tile-scheduling
title: "Tile Scheduling Strategies"
architectures: [gfx940, gfx942, gfx950]
---

Tile scheduling determines the order in which output tiles of a GEMM or attention kernel are assigned to CTAs. The scheduling order affects L2 cache reuse, tail effects, and how evenly work is distributed across compute resources.

Prefer an ordering that keeps CTAs with shared operands close together in time. For GEMM-style kernels, column-major or N-major scheduling usually improves B operand reuse compared with row-major raster scheduling, because adjacent CTAs consume nearby columns before moving across the full M dimension.

Use grouped or zigzag N scheduling when the N dimension is large. Split N tiles into fixed-width groups, walk each group over M, then advance to the next N group. This keeps a bounded working set of B columns in L2 while still covering all M tiles. Handle the final partial group explicitly when `n_groups_` is not divisible by the group width.

On multi-XCC GPUs, apply XCC-aware remapping before the local tile-ordering policy. The concrete idea is to deinterleave each scheduler-sized wave of CTAs by XCC: infer the CTA's XCC slot from its position in the wave, then map that CTA into the contiguous logical tile range owned by that XCC. After remapping, convert the remapped id to `(gid_m, gid_n)` using the selected grouped or zigzag policy.

```c++
// Inputs:
//   raw_gid: original block/workgroup id
//   total:   m_groups_ * n_groups_
//   xccs:    number of XCCs
//   cus:     number of CUs visible to the kernel scheduler
//
// Assumption: scheduler order cycles XCC slots inside a CU-sized wave, so
// cu_slot % xccs gives the XCC slot and cu_slot / xccs gives the local slot
// within that XCC.
unsigned remap_xcc(unsigned raw_gid, unsigned total,
                   unsigned xccs, unsigned cus) {
    unsigned cu_wave = raw_gid / cus;
    unsigned cu_slot = raw_gid % cus;
    unsigned wave_base = cu_wave * cus;

    unsigned full_wave_end = (total / cus) * cus;
    unsigned wave_size = raw_gid < full_wave_end ? cus : total - full_wave_end;
    unsigned xcc_full = wave_size / xccs;
    unsigned xcc_tail = wave_size % xccs;

    unsigned xcc_id = cu_slot % xccs;
    unsigned local = cu_slot / xccs;
    unsigned xcc_size = xcc_full + (xcc_id < xcc_tail ? 1 : 0);
    unsigned prior_tail = xcc_id < xcc_tail ? xcc_id : xcc_tail;
    unsigned xcc_base = xcc_id * xcc_full + prior_tail;

    return local < xcc_size ? wave_base + xcc_base + local : raw_gid;
}
```

This turns launch order like `XCC0 tile0, XCC1 tile0, ... XCC0 tile1, XCC1 tile1` into logical tile order like `XCC0 tile0, XCC0 tile1, ... XCC1 tile0, XCC1 tile1`. Each XCC therefore works on a compact region of the logical tile stream instead of all XCCs competing over the same early scheduling region.

Keep these invariants fixed:
  - `gid_m` is always in `[0, m_groups_)` and `gid_n` is always in `[0, n_groups_)`.
  - Every logical tile is assigned exactly once.
  - Remainder waves and partial N groups are handled without dropping or duplicating CTAs.
  - Scheduling changes do not alter tile math, LDS layout, MFMA operand mapping, or output writeback semantics.

Avoid plain linear raster scheduling unless simplicity is more important than locality. A direct row-major mapping is easy to reason about, but it often gives poor L2 reuse for B because consecutive CTAs advance across M before nearby N tiles are consumed.
