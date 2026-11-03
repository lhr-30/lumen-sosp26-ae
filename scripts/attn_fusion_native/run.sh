#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source_dir=${NATIVE_SOURCE:-$root/.repro/avelang-native}
results=${NATIVE_RESULTS:-$root/.repro/attn-fusion-native-results}
image=${HOPPER_IMAGE:-lumen-hopper:dev}
mode=${1:-test}
if (($#)); then shift; fi
mkdir -p "$results"
docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then docker_cmd=(sudo -n docker); fi
if [[ "$mode" == build ]]; then
  HOPPER_SOURCE="$source_dir" HOPPER_RESULTS="$results" exec "$root/scripts/hopper/run.sh" build
fi
common=(run --rm --pull never --network none
  -e PYTHONPATH=/workspace/avelang/python:/workspace/native:/workspace/attn
  -e ATTN_RESULTS=/results
  -v "$source_dir:/workspace/avelang:ro"
  -v "$root/scripts/attn_fusion_native:/workspace/native:ro"
  -v "$root/scripts/attn_fusion:/workspace/attn:ro"
  -v "$results:/results" --entrypoint /bin/bash)
evaluation=(python /workspace/attn/evaluate.py --candidate native_candidate --invariant-validation unavailable)
case "$mode" in
  cpu)
    exec "${docker_cmd[@]}" "${common[@]}" -e NVIDIA_VISIBLE_DEVICES=void "$image" \
      -c 'cd /workspace/native; exec python -X faulthandler compile_candidate.py "$@"' bash "$@" ;;
  test|bench)
    workload=("${evaluation[@]}" --mode "$mode" "$@") ;;
  baseline)
    reference_dir=${ATTN_REFERENCE:-$root/../model-tunner/attn_fusion}
    common+=(-v "$reference_dir:/reference:ro")
    workload=(python /reference/bench.py "$@") ;;
  memcheck|synccheck|racecheck)
    workload=(compute-sanitizer --tool "$mode" --error-exitcode 1 --print-limit 10
      "${evaluation[@]}" --mode test --smoke "$@") ;;
  *) echo 'Usage: scripts/attn_fusion_native/run.sh {build|cpu|test|bench|baseline|memcheck|synccheck|racecheck} [options]' >&2; exit 2 ;;
esac
command=("${docker_cmd[@]}" "${common[@]}" --device nvidia.com/gpu=0 "$image"
  -c 'exec "$@"' bash "${workload[@]}")
if [[ -n "${GPU_QUEUE_ACTIVE:-}" ]]; then exec "${command[@]}"; fi
exec gpu-queue-system run --label "codex/lumen native Hopper attention $mode" --timeout 900 -- "${command[@]}"
