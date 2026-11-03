export default [
  {
    title: "Tutorial 1: AXPY",
    url: "tutorials/01-axpy/",
    description: "Introduce `@ave.jit`, typed tensors, thread indexing, and kernel launch geometry."
  },
  {
    title: "Tutorial 2: GEMM",
    url: "tutorials/02-gemm/",
    description: "Start with naive BF16 `A @ B^T`, then add shared-memory tiling and vectorized tile loads."
  },
  {
    title: "Tutorial 3: Intrinsics",
    url: "tutorials/03-intrinsics/",
    description: "Use AMDGPU MFMA, runtime pointer views, and raw-buffer loads for generic GEMM kernels."
  }
];
