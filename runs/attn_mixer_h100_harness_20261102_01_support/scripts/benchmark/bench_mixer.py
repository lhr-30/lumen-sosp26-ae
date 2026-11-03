"""Stock Lumen JSONL benchmark adapter, with queued H100 evaluation in Docker."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent


def worker(args):
    if args.mode != 'compile' and not os.environ.get('GPU_QUEUE_ACTIVE'):
        raise RuntimeError('GPU worker requires the system queue reservation')
    env = json.loads((HERE / 'environment.json').read_text())
    invocation = args.worker.resolve()
    container = 'lumen-attn-harness-' + uuid.uuid4().hex[:12]
    docker = ['sudo', '-n', 'docker']
    cmd = docker + ['run', '--rm', '--pull', 'never', '--network', 'none', '--name', container,
                    '-e', 'PYTHONFAULTHANDLER=1',
                    '-e', 'PYTHONPATH=/workspace/avelang/python:/experiment:/support',
                    '-v', env['compiler'] + ':/workspace/avelang:ro',
                    '-v', str(HERE) + ':/support:ro',
                    '-v', str(invocation) + ':/experiment', '--entrypoint', 'python']
    if args.mode == 'compile':
        cmd += ['-e', 'NVIDIA_VISIBLE_DEVICES=void', env['image'], '/support/offline_compile.py']
    else:
        cmd += ['--device', 'nvidia.com/gpu=0', env['image'], '/support/evaluate.py',
                '--candidate', 'output_model_new', '--invariant-validation', 'unavailable',
                '--mode', args.mode, '--iterations', str(args.iterations), '--samples', str(args.samples),
                '--output', '/experiment/result.json']
    log = invocation / 'output.log'
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=240)
        log.write_text(result.stdout)
        print(result.stdout, end='', flush=True)
        return result.returncode
    except subprocess.TimeoutExpired as error:
        value = error.stdout or b''
        text = value.decode(errors='replace') if isinstance(value, bytes) else value
        log.write_text(text + '\nEvaluation timed out after 240s\n')
        print(text, end='', flush=True)
        print('Evaluation timed out after 240s', file=sys.stderr, flush=True)
        return 124
    finally:
        # Also handles a hung candidate before releasing the queue reservation.
        subprocess.run(docker + ['rm', '-f', container], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['lumen'], default='lumen')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--check-correctness', action='store_true')
    parser.add_argument('--seq-lens', type=int, nargs='+', default=[26])
    parser.add_argument('--mode', choices=['compile', 'test', 'bench'], default='bench')
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    assert args.seq_lens == [26], 'Only the original B16/T26 workload is supported'
    candidate = args.candidate
    if candidate is None:
        if args.model_path is None:
            parser.error('--model-path or --candidate required')
        candidate = args.model_path.resolve().parents[4] / 'output_model_new.py'
    candidate = candidate.resolve()
    data = candidate.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    invocation = candidate.parent / 'evaluations' / (stamp + '_' + args.mode)
    invocation.mkdir(parents=True)
    (invocation / 'output_model_new.py').write_bytes(data)
    (invocation / 'invocation.json').write_text(json.dumps({
        'source': str(candidate), 'sha256': digest, 'mode': args.mode,
        'iterations': args.iterations, 'samples': args.samples}, indent=2) + '\n')
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(invocation),
               '--mode', args.mode, '--iterations', str(args.iterations), '--samples', str(args.samples)]
    if args.mode != 'compile' and not os.environ.get('GPU_QUEUE_ACTIVE'):
        command = ['gpu-queue-system', 'run', '--label', 'codex/lumen native harness mixer ' + args.mode,
                   '--timeout', '300', '--'] + command
    result = subprocess.run(command, check=False)
    if args.mode == 'compile':
        return result.returncode
    record = {'domain': 'attention_mixer', 'seq_len': 26, 'B': 16, 'H': 8, 'KV': 2,
              'D': 256, 'correctness': False, 'candidate_sha256': digest,
              'artifacts': str(invocation), 'invariant_validation': 'unavailable'}
    output = invocation / 'result.json'
    if result.returncode == 0 and output.is_file():
        payload = json.loads(output.read_text())
        assert payload['candidate_sha256'] == digest
        assert len(payload['correctness']) == 5
        record['correctness'] = True
        record['validation'] = payload['correctness']
        if args.mode == 'bench':
            record['latency_us'] = payload['benchmark']['kernels']['chain']['median_us']
            record['stages'] = payload['benchmark']['kernels']
    print(json.dumps(record, separators=(',', ':')), flush=True)
    return 0 if record['correctness'] else result.returncode or 1


if __name__ == '__main__':
    raise SystemExit(main())
