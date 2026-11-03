"""CPU-only compilation of the candidate's declared specializations."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

from avelang.backends.compiler import GPUTarget
from avelang.backends.nvidia.compiler import NvidiaCompiler
from avelang.compiler.compiler import ASTSource

sys.path.insert(0, '/experiment')
candidate = importlib.import_module('output_model_new')
cases = candidate.OFFLINE_COMPILE_CASES
assert cases, 'Declare all specialization cases in OFFLINE_COMPILE_CASES'
out = Path('/experiment/artifacts')
out.mkdir(exist_ok=True)
target = GPUTarget('nvptx64-nvidia-cuda', 'sm_90a')
records = []
for name, fn, constants, warps in cases:
    assert name.replace('_', '').isalnum()
    print('Compiling', name, flush=True)
    source = ASTSource(fn, {}, constexprs={
        k: {'type': 'i32', 'value': v} for k, v in constants.items()},
        global_constexprs=fn._collect_global_constexprs(constants.keys()))
    ptx = NvidiaCompiler(target).compile(source, target, {'num_warps': warps})
    path = out / (name + '.ptx')
    path.write_bytes(ptx)
    result = subprocess.run(['ptxas', '-arch=sm_90a', '-v', str(path), '-o',
                             str(out / (name + '.cubin'))], capture_output=True, text=True)
    print(name, result.stderr, flush=True)
    (out / (name + '.ptxas.txt')).write_text(result.stdout + result.stderr)
    result.check_returncode()
    assert '.extern .func' not in ptx.decode(), 'Unresolved device function: ' + name
    records.append({'name': name, 'ptx_sha256': hashlib.sha256(ptx).hexdigest(),
                    'ptxas': result.stderr, 'local_memory': '.local ' in ptx.decode()})
(out / 'compile.json').write_text(json.dumps(records, indent=2) + '\n')
print('CPU compile PASS', len(records), flush=True)
