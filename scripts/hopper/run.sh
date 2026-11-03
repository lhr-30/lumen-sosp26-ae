#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source_dir=${HOPPER_SOURCE:-$root/.repro/avelang-artifact}
results=${HOPPER_RESULTS:-$root/.repro/hopper-results}
image=${HOPPER_IMAGE:-lumen-hopper:dev}
mode=${1:-test}
mkdir -p "$results"
docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then docker_cmd=(sudo -n docker); fi
common=(run --rm --pull never --network none
  -e PYTHONPATH=/workspace/avelang/python:/workspace/hopper
  -e HOPPER_RESULTS=/results -e BUILD_JOBS="${BUILD_JOBS:-16}"
  -v "$source_dir:/workspace/avelang" -v "$root/scripts/hopper:/workspace/hopper:ro"
  -v "$results:/results" --entrypoint /bin/bash)
case "$mode" in
  build)
    exec "${docker_cmd[@]}" "${common[@]}" -e NVIDIA_VISIBLE_DEVICES=void "$image" /workspace/hopper/build.sh ;;
  cpu)
    exec "${docker_cmd[@]}" "${common[@]}" -e NVIDIA_VISIBLE_DEVICES=void "$image" \
      -c 'python /workspace/hopper/test_hopper.py -v' ;;
  verify)
    workload=(bash /workspace/hopper/verify_gpu.sh) ;;
  test|smoke)
    workload=(python /workspace/hopper/validate.py --output "/results/$mode.json")
    if [[ "$mode" == smoke ]]; then workload+=(--smoke); fi ;;
  memcheck|synccheck|racecheck)
    workload=(compute-sanitizer --tool "$mode" --error-exitcode 1
      python /workspace/hopper/validate.py --smoke --output "/results/$mode.json") ;;
  *) echo 'Usage: scripts/hopper/run.sh {build|cpu|verify|test|smoke|memcheck|synccheck|racecheck}' >&2; exit 2 ;;
esac
command=("${docker_cmd[@]}" "${common[@]}" --device nvidia.com/gpu=0 "$image" -c 'exec "$@"' bash "${workload[@]}")
# Reuse an enclosing system reservation; never nest queue wrappers.
if [[ -n "${GPU_QUEUE_ACTIVE:-}" ]]; then
  exec "${command[@]}"
fi
exec gpu-queue-system run --label "codex/lumen H100 $mode" --timeout 600 -- "${command[@]}"
