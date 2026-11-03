--
id: hw-mfma
title: "MFMA for matrix multiplicaton"
architectures: [gfx942, gfx950]
--

Use MFMA instructions `S.amdgpu.mfma_32x32x8_bf16_f32` to do matrix multiplications. The MFMA instruction computes 32x32x8 matmul cooperatively in a wave. Build larger tiles by issuing multiple MFMA instructions across K and output subtiles.
  
MFMA swizzle invariants:
  For A, `i in [0,32)`, `j in [0,8)`: `A(i, j) -> (lane_id = i + (j / 4) * 32, element = j % 4)`
  For B, `j in [0,8)`, `i in [0,32)`: `B(j, i) -> (lane_id = j + (i / 4) * 32, element = i % 4)`

Accumulator invariant for C: for each lane in [0, 64) and acc_idx in [0, 16):
  - col = tile_col_base + (lane % 32)
  - row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)

Do not invent a different unpacking. Do not use a naive acc_idx -> contiguous row mapping. Treat this mapping as fixed when writing results. The AMD matrix instruction calculator (https://github.com/ROCm/amd_matrix_instruction_calculator) also gives the mappings.

Stage A and B through LDS. Each thread loads operand fragments in 16-byte chunks. Treat each 16-byte fragment as `(4, S.u32)` and reinterpret it as `2 x (4, S.bf16)`. Feed both `(4, S.bf16)` halves into MFMA in natural order. The intended effect is a cooperative `32x32x16` accumulation from two natural MFMA steps. Do not add lane-dependent or K-dependent control flow to select halves.
  - The two `(4, S.bf16)` halves from one 16-byte LDS load collectively represent a swizzled `32x16` operand contribution with 4-column interleaving.
  - Consuming them in natural order must produce the same final C as a naive conceptual layout because operand pairings remain consistent under MFMA swizzle.

Scale the kernel from one wave to four waves without changing the MFMA per-wave invariant. Interpret the 4 warps as a 2 x 2 warp grid. Keep MFMA math identical per warp. Only add warp ownership offsets at operand fetch and output writeback.