#!/usr/bin/env bash
set -euo pipefail
# CPU-only. Run inside the Hopper development container.
cd /workspace/avelang
cmake -S . -B build-hopper -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/llvm/bin/clang++ \
  -DHUNTER_ENABLED=OFF -DAVE_LANG_BACKEND=cuda -DWITH_PYTHON=ON \
  -DLLVM_DIR=/opt/llvm/lib/cmake/llvm -DMLIR_DIR=/opt/llvm/lib/cmake/mlir \
  -DClang_DIR=/opt/llvm/lib/cmake/clang \
  -DAVE_LANG_PYTHON_LIBRARY_OUTPUT_DIRECTORY=/workspace/avelang/python
cmake --build build-hopper --target _avelang_bindings -j "${BUILD_JOBS:-16}"
