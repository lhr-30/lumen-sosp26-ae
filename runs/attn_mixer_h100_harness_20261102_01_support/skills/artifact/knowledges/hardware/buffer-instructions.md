--
id: hw-buffer-instructions
title: "Buffer Instructions"
architectures: [gfx940, gfx942, gfx950]
--

Use buffer instructions `S.amdgpu.raw_buffer_load_x4`, `S.amdgpu.raw_buffer_load_x2`, `S.amdgpu.raw_buffer_store_x4` to vectorize the loads and stores and remove the explicit branches guarding OOB access. The range is in the units of bytes. When the range of the descriptors in the instructions are set, the loads return 0 for OOB elements, and the stores are discarded. The optimizations are safe when the computations and LDS accesses work with 0 values. Removing the branches in the loop is more beneficial compared to reducing extra computations and LDS access.

## Async LDS buffer loads

Use asynchronous LDS buffer loads to stage global memory directly into LDS for GEMM pipelines whose next compute stage consumes from LDS. Express this with `llvm.amdgcn.raw.buffer.load.lds`, which maps to MUBUF `buffer_load_*` with the `lds` modifier. This is the preferred form over inline assembly because it keeps the buffer descriptor and SGPR tuple constraints in the compiler intrinsic interface.

The intrinsic takes a buffer resource descriptor, an LDS destination pointer in address space 3, the transfer size in bytes, vector and scalar offsets, an immediate byte offset, and aux/cache/control bits. A typical wrapper should keep the size and immediate offset compile-time constants, as in `LoadLds<kAux, kSize, kOffset>(lds_ptr, voffset, soffset)`.

Gate the transfer size by architecture:
  - On gfx940 and gfx942, only 32-bit / 4-byte LDS async buffer loads are supported.
  - On gfx950, 4-byte, 8-byte, and 16-byte LDS async buffer loads are supported.

Do not assume a 16-byte async LDS load is portable across all supported CDNA targets. Use 4-byte async LDS loads for gfx940/gfx942, or select the wider form only when targeting gfx950.
