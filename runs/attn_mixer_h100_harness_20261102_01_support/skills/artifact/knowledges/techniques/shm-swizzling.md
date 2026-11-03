---
id: technique-shm-swizzling
title: "Shared Memory Swizzling"
architectures: [gfx940, gfx942, gfx950]
---

LDS swizzling remaps a matrix tile in LDS so lanes in the same `ds_` instruction phase avoid different-address accesses to the same bank. Use it for MFMA operand reads when column-wise LDS access conflicts and padding is not enough.

AMD LDS banks are 4-byte wide. On gfx940/gfx942, bank index is `(byte_address / 4) % 32`. On gfx950, bank index is `(byte_address / 4) % 64`. Same-address accesses can broadcast; different addresses in the same bank conflict.

Check conflicts against the emitted LDS instruction, not the whole wave at once. `ds_read_b32` touches one bank per lane, `ds_read_b64` touches two banks, and `ds_read_b128` touches four banks. The active lane phases differ across instruction width and architecture.

When the LDS address is in vector units, keep the swizzle in vector units too:

```c++
unsigned base = tile_idx_m * kTile * kGroupK / kVecSize +
                tile_idx_k * kTile * kLayoutM / kVecSize +
                batch_id * kWarpSize;

unsigned logical = col * kMmaM + row;
unsigned xor_stride = col * kBatchStride + batch_id;
unsigned lds_index = base + (logical ^ xor_stride);
```

Do not scale `xor_stride` by bytes for this form. If `access_bytes = kVecSize`, then `byte_address = lds_index * access_bytes` and `bank_start = lds_index * (access_bytes / 4) % bank_count`. The same source-level swizzle is alignment-correct for 4B, 8B, and 16B accesses, but conflict behavior still changes with `ds_read_b32`, `ds_read_b64`, and `ds_read_b128` phases.

Keep these invariants fixed:
  - Use the same swizzle for LDS stores and loads.
  - Swizzle the intra-tile logical offset, not unrelated tile-base bits.
  - Preserve alignment required by the chosen `ds_` vector width.
  - Do not change MFMA operand order; only change where logical elements live in LDS.
