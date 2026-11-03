#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
run_dir=$(cd -- "$here/.." && pwd)
root=$(cd -- "$here/../../.." && pwd)
ncu_dir=/opt/nvidia/nsight-compute/2026.2.1
mode=${1:-timing}
if (($#)); then shift; fi
docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then docker_cmd=(sudo -n docker); fi
common=(run --rm --pull never --network none --cap-add SYS_ADMIN
  --device nvidia.com/gpu=0
  -e PYTHONPATH=/workspace/avelang/python
  -v "$root/.repro/avelang-native:/workspace/avelang:ro"
  -v "$run_dir:/profile" -v "$ncu_dir:$ncu_dir:ro"
  --entrypoint /bin/bash lumen-hopper:dev)
case "$mode" in
  timing|check)
    workload=(python /profile/harness/compare.py --mode "$mode" "$@") ;;
  full|source)
    impl=${1:?specify baseline or epilogue}
    shift
    options=(--set "$mode" --profile-from-start off --cache-control none --clock-control base
      --import-source yes --export "/profile/reports/${mode}_${impl}")
    if [[ "$mode" == full ]]; then
      options+=(--section PmSampling --section PmSampling_WarpStates)
    else
      options+=(--section SourceCounters)
    fi
    workload=("$ncu_dir/ncu" "${options[@]}" python /profile/harness/compare.py
      --mode profile --impl "$impl" --output "/profile/analysis/${mode}_${impl}_validation.json" "$@") ;;
  *) echo 'Usage: run.sh {check|timing|full|source} [baseline|epilogue] [harness args]' >&2; exit 2 ;;
esac
command=("${docker_cmd[@]}" "${common[@]}" -c 'exec "$@"' bash "${workload[@]}")
if [[ -n "${GPU_QUEUE_ACTIVE:-}" ]]; then exec "${command[@]}"; fi
exec gpu-queue-system run --label "codex/lumen H100 TMA epilogue $mode" --timeout 1200 -- "${command[@]}"
