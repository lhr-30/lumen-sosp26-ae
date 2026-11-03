#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source_dir=${HOPPER_SOURCE:-$root/.repro/avelang-artifact}
results=${ATTN_RESULTS:-$root/.repro/attn-fusion-results}
image=${HOPPER_IMAGE:-lumen-hopper:dev}
mode=${1:-test}
if (($#)); then shift; fi
mkdir -p "$results"
docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then docker_cmd=(sudo -n docker); fi
common=(run --rm --pull never --network none
  -e PYTHONPATH=/workspace/avelang/python:/workspace/attn:/workspace/hopper
  -e ATTN_RESULTS=/results
  -v "$source_dir:/workspace/avelang:ro"
  -v "$root/scripts/attn_fusion:/workspace/attn:ro"
  -v "$root/scripts/hopper:/workspace/hopper:ro"
  -v "$results:/results" --entrypoint /bin/bash)
case "$mode" in
  cpu)
    exec "${docker_cmd[@]}" "${common[@]}" -e NVIDIA_VISIBLE_DEVICES=void "$image" \
      -c 'cd /workspace/attn; exec python compile_candidate.py "$@"' bash "$@" ;;
  test|bench)
    workload=(python /workspace/attn/evaluate.py --mode "$mode" "$@") ;;
  baseline)
    reference_dir=${ATTN_REFERENCE:-$root/../model-tunner/attn_fusion}
    common+=(-v "$reference_dir:/reference:ro")
    workload=(python /reference/bench.py "$@") ;;
  memcheck|synccheck|racecheck)
    workload=(compute-sanitizer --tool "$mode" --error-exitcode 1 --print-limit 10
      python /workspace/attn/evaluate.py --mode test --smoke "$@") ;;
  *) echo 'Usage: scripts/attn_fusion/run.sh {cpu|test|bench|baseline|memcheck|synccheck|racecheck} [options]' >&2; exit 2 ;;
esac
command=("${docker_cmd[@]}" "${common[@]}" --device nvidia.com/gpu=0 "$image"
  -c 'exec "$@"' bash "${workload[@]}")
if [[ -n "${GPU_QUEUE_ACTIVE:-}" ]]; then exec "${command[@]}"; fi
exec gpu-queue-system run --label "codex/lumen Ave attention $mode" --timeout 900 -- "${command[@]}"
