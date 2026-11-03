#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
run_dir=$(cd -- "$here/.." && pwd)
root=$(cd -- "$here/../../.." && pwd)
mode=${1:-timing}
if (($#)); then shift; fi
if [[ -z "${GPU_QUEUE_ACTIVE:-}" ]]; then
  exec gpu-queue-system run --label "codex/lumen native harness independent $mode" \
    --timeout 900 -- "$0" "$mode" "$@"
fi
common=(run --rm --pull never --network none --device nvidia.com/gpu=0
  -e PYTHONPATH=/workspace/avelang/python:/profile/harness/sources
  -e PYTHONFAULTHANDLER=1
  -v "$root/.repro/avelang-native:/workspace/avelang:ro"
  -v "$run_dir:/profile" --entrypoint /bin/bash lumen-hopper:dev)
launch() { sudo -n docker "${common[@]}" -c 'exec "$@"' bash "$@"; }
timing() {
  nvidia-smi --query-gpu=name,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu \
    --format=csv > "$run_dir/logs/gpu_before_timing.csv"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
    > "$run_dir/logs/processes_before_timing.csv"
  nvidia-smi pmon -s u -d 1 -o DT > "$run_dir/logs/process_activity.log" &
  activity_pid=$!
  trap 'kill "$activity_pid" 2>/dev/null || true; wait "$activity_pid" 2>/dev/null || true' EXIT
  launch python /profile/harness/compare.py --modules round0 round1 round2 manual \
    --output /profile/analysis/timing.json "$@" > "$run_dir/logs/timing.log" 2>&1
  kill "$activity_pid" 2>/dev/null || true
  wait "$activity_pid" 2>/dev/null || true
  trap - EXIT
}
case "$mode" in
  final)
    launch python /profile/harness/check_final.py --module round2 \
      --output /profile/analysis/correctness_guarded.json > "$run_dir/logs/correctness_guarded.log" 2>&1
    for sanitizer in memcheck synccheck racecheck; do
      launch compute-sanitizer --tool "$sanitizer" --error-exitcode 1 --print-limit 10 \
        python /profile/harness/sources/evaluate.py --candidate round2 \
        --invariant-validation unavailable --mode test --smoke \
        --output "/profile/analysis/$sanitizer.json" > "$run_dir/logs/$sanitizer.log" 2>&1
    done
    timing "$@" ;;
  timing) timing "$@" ;;
  *) echo 'Usage: run.sh {final|timing} [comparison options]' >&2; exit 2 ;;
esac
