#!/usr/bin/env bash
set -euo pipefail
# Called by run.sh verify inside ONE host GPU reservation.
python /workspace/hopper/validate.py --output /results/test.json
# Regression: retain the compiler's existing warp-level NVIDIA MMA path.
python /workspace/avelang/test/examples/gemm/nvidia/test_gemm_nvvm_16x8x16.py
for tool in memcheck synccheck racecheck; do
  compute-sanitizer --tool "$tool" --error-exitcode 1 \
    --log-file "/results/$tool.log" \
    python /workspace/hopper/validate.py --smoke --output "/results/$tool.json"
  cat "/results/$tool.log"
done
