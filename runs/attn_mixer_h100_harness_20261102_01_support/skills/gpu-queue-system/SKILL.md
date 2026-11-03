---
name: gpu-queue-system
description: Coordinate this machine's single GPU across all Linux users and agents through one system-wide FIFO. Use before GPU correctness tests, benchmarks, warmups, autotuning, training, or ncu/nsys profiling (跨用户 GPU 排队、独占). CPU-only work does not need the queue.
---

# System GPU queue

Use the single host service for every GPU workload. All Linux users share its
FIFO; commands execute as the submitting user with their environment, working
directory, standard streams and GPU selection preserved. The service coordinates
leases and cancellation; it does not execute user commands.

## Run and wait

```bash
gpu-queue-system run --label "codex/project baseline" -- python3 benchmark.py
gpu-queue-system run --label "claude/project ncu" --timeout 600 -- ncu --set full -o profile python3 kernel.py
gpu-queue-system run --label "project compare" -- bash -lc 'python3 baseline.py && python3 candidate.py'
```

If the CLI is absent from PATH, use
`/usr/bin/python3 /usr/local/share/agent-skills/gpu-queue-system/scripts/gpu_queue.py`.
If the service is unavailable, stop GPU work and consult
[operations](references/operations.md); never fall back to a private queue.

- Wrap correctness tests, warmups, autotuning and profilers as well as benchmarks.
  Compilation that probes the GPU also needs a reservation.
- Keep CPU work outside the reservation. Queue one finite experiment or small
  comparison, with warmup and measurement in the same invocation.
- Wait using the coding tool's background session support and continue CPU work.
  Do not resubmit after a tool timeout. Waiting jobs log every 15 seconds.
- Put pipelines, redirections and sequences **inside** the wrapper. The service
  does not store output; write result files for long experiments.
- `--wait-timeout SECONDS` limits waiting; `--timeout SECONDS` limits runtime,
  including children. Both default to unlimited. Timeout exits 124, queue errors
  exit 125, cancellation exits 128 + signal; ordinary command exit codes propagate.
- Labels are public to all local users; keep secrets out of them. Command
  arguments, environment and working directory are not published in queue status.

## Inspect and cancel

```bash
gpu-queue-system status
gpu-queue-system status --json
gpu-queue-system cancel 17
```

Status shows tickets, labels, users/UIDs and supervisor PIDs. Ordinary users can
cancel only their own UID's jobs; root can cancel any job. Cancel only unnecessary
work belonging to the requesting user or a job they explicitly ask to cancel.
`cancel` is asynchronous: check status until the job disappears. SIGINT/SIGTERM
to the outer wrapper also request cancellation. Descendants receive TERM, then
KILL after three seconds when necessary; the reservation remains until exit.

## Invariants

- Always use `/run/gpu-queue-system/queue.sock`. The `--socket`, `--state-dir`
  and `serve --test-mode` options are for isolated CPU tests only.
- Do not nest wrappers. `GPU_QUEUE_ACTIVE` means the enclosing job already owns
  the GPU; run inner commands directly and sequentially. The old per-UID queue
  does not coordinate with this service. Follow the migration procedure in
  operations before switching existing sessions; do not run both queues side by side.
- Never delete `/var/lib/gpu-queue-system` or its lock files to bypass a wait.
  Lock identity must survive service restart. Only the service writes this state.
- Run foreground workloads. Do not daemonize, change UID, detach via `setsid`,
  send GPU work to an existing external service, or escape the supervisor's
  process lifetime. Forked children are tracked until exit.
- Killing the outer CLI does not cancel its independent supervisor. Inspect
  status and cancel the ticket. Service restart preserves durable reservations
  and inherited leases, including surviving same-session children of a dead supervisor.
- Waiting clients exit with an error if they encounter a service outage. Inspect
  status after recovery before resubmitting. Running jobs can finish while the
  service is down; wrapper cancellation and runtime limits still work.
- This is cooperative scheduling, not GPU device access enforcement. Unqueued
  processes can still interfere. Coordinate with their owners before trusting
  results; do not kill unrelated workloads. Queueing does not fix CUDA timing.

For system installation, per-user activation, migration and recovery, read
[references/operations.md](references/operations.md).
