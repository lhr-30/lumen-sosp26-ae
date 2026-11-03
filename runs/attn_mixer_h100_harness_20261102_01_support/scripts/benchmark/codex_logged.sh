#!/usr/bin/env bash
set -euo pipefail
codex "$@" > >(tee codex.stdout.log) 2> >(tee codex.stderr.log >&2)
