"""Run the repository's unchanged Lumen optimization runtime on the H100 mixer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'python'))
from lumen.harness.datasets.lumen.optimization_runtime import (
    OptimizationConfig, OptimizationSpec, run_optimization_sequence,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--timeout-seconds', type=float, default=1800)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 3:
        parser.error('--rounds must be between 1 and 3')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    run = (args.run_dir or REPO / 'runs' / ('attn_mixer_h100_harness_' + stamp)).resolve()
    if run.exists():
        raise FileExistsError(run)
    # This is a clean support root for the stock runtime's scripts/skills
    # allowlist. It contains no previous mixer implementation or tuning results.
    support = run.with_name(run.name + '_support')
    support.mkdir(parents=True)
    for name in ('scripts/benchmark', 'skills/native', 'prompts'):
        (support / name).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / 'pyproject.toml', support / 'pyproject.toml')
    compiler = REPO / '.repro/avelang-native'
    for rel, name in [('docs/content', 'docs'), ('python/avelang/language', 'language'),
                      ('python/avelang_kernels', 'examples'), ('lib/IR/Intrinsics', 'intrinsics')]:
        shutil.copytree(compiler / rel, support / 'skills/native' / name,
                        ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(REPO / 'skills', support / 'skills/artifact')
    queue_skill = Path('/usr/local/share/agent-skills/gpu-queue-system')
    (support / 'skills/gpu-queue-system').mkdir()
    shutil.copyfile(queue_skill / 'SKILL.md', support / 'skills/gpu-queue-system/SKILL.md')
    shutil.copytree(queue_skill / 'references', support / 'skills/gpu-queue-system/references')
    for name in ('bench_mixer.py', 'offline_compile.py', 'codex_logged.sh'):
        shutil.copyfile(Path(__file__).parent / name, support / 'scripts/benchmark' / name)
    (support / 'scripts/benchmark/codex_logged.sh').chmod(0o755)
    for name in ('input_model.py', 'evaluate.py'):
        shutil.copyfile(REPO / 'scripts/attn_fusion' / name, support / 'scripts/benchmark' / name)
    # Only the authoritative reference is the seed. No prior DSL is supplied.
    shutil.copyfile(REPO / 'scripts/attn_fusion/input_model.py', support / 'seed.py')
    environment = {'compiler': str(compiler), 'image': 'lumen-hopper:dev',
                   'codex': shutil.which('codex'), 'domain': 'attention_mixer',
                   'workload': {'B': 16, 'T': 26, 'H': 8, 'KV': 2, 'D': 256, 'hidden': 2048},
                   'compiler_commit': subprocess.check_output(
                       ['git', '-C', str(compiler), 'rev-parse', 'HEAD'], text=True).strip()}
    (support / 'scripts/benchmark/environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    common = (Path(__file__).parent / 'prompts/common.md').read_text()
    # Every round is predeclared; we do not inject manual optimization hints.
    prompts = []
    for index in range(args.rounds):
        text = common + '\n' + (Path(__file__).parent / f'prompts/round{index}.md').read_text()
        text = text.replace('{SUPPORT}', str(support))
        path = support / 'prompts' / f'round{index}.md'
        path.write_text(text)
        prompts.append(path)
    runtime = REPO / 'python/lumen/harness/datasets/lumen/optimization_runtime.py'
    backend = REPO / 'python/lumen/harness/backend/codex/runner.py'
    provenance = {'runtime': str(runtime), 'runtime_sha256': sha(runtime),
                  'backend': str(backend), 'backend_sha256': sha(backend),
                  'runtime_modified': False, 'seed_sha256': sha(support / 'seed.py'),
                  'prior_mixer_candidates_provided': False,
                  'customization': ['workload spec', 'H100 evaluator/queue adapter',
                                    'task-specific prompts and public native DSL references'],
                  'rounds': args.rounds, 'support_root': str(support)}
    (support / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    manifest = {str(path.relative_to(support)): sha(path)
                for path in support.rglob('*') if path.is_file()}
    (support / 'support_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    spec = OptimizationSpec(
        domain='attention_mixer', run_slug='attn_mixer_h100', default_kernel=support / 'seed.py',
        default_prompts=tuple(prompts), workloads=(26,), workload_key='seq_len',
        workload_flag='--seq-lens', benchmark_script='bench_mixer.py',
        adapter_source='''"""Candidate locator consumed by the H100 workload adapter."""
from pathlib import Path
KERNEL_PATH = Path(__file__).resolve().parents[4] / "output_model_new.py"
''',
    )
    config = OptimizationConfig(
        repo_root=support, kernel=spec.default_kernel, run_dir=run, gpu_id=0,
        codex_bin=support / 'scripts/benchmark/codex_logged.sh',
        timeout_seconds=args.timeout_seconds,
    )
    print('Original Lumen runtime:', runtime, flush=True)
    print('Run:', run, flush=True)
    rounds = run_optimization_sequence(config, spec, tuple(prompts))
    summary = []
    for workspace, result in rounds:
        evaluation = workspace.round_dir / 'eval_result.json'
        summary.append({'round': workspace.round_dir.name, 'ok': result.ok,
                        'status': result.status, 'error': result.error,
                        'candidate_sha256': sha(workspace.output_model_path),
                        'evaluation': json.loads(evaluation.read_text()) if evaluation.exists() else None})
    (run / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    assert sha(runtime) == provenance['runtime_sha256'] and sha(backend) == provenance['backend_sha256']
    print('Completed:', run, flush=True)
    return 0 if all(r['ok'] for r in summary) and len(summary) == args.rounds else 1


if __name__ == '__main__':
    raise SystemExit(main())
