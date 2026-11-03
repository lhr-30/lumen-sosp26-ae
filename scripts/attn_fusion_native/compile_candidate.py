"""Offline native-Hopper compilation: no GPU initialization or fake invariants."""
import argparse
import ast
import copy
import importlib
import json
import re
import sys
from pathlib import Path
import shutil
import subprocess
import time

from avelang.backends.compiler import GPUTarget
from avelang.backends.nvidia.compiler import NvidiaCompiler
from avelang.compiler.compiler import ASTSource


def compile_one(fn, constants, warps):
    target = GPUTarget('nvptx64-nvidia-cuda', 'sm_90a')
    source = ASTSource(fn, {}, constexprs={
        k: {'type': 'i32', 'value': v} for k, v in constants.items()},
        global_constexprs=fn._collect_global_constexprs(constants.keys()))
    return NvidiaCompiler(target).compile(source, target, {'num_warps': warps})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', default='native_candidate',
                        choices=['native_candidate', 'round1_candidate', 'round2_candidate', 'round3_candidate', 'round4_candidate', 'round4_mma_candidate', 'round4_async_candidate'])
    parser.add_argument('--only', help='compile one named kernel for CPU diagnosis')
    parser.add_argument('--frontend-until', type=int, help='debug projection frontend source-line prefix only')
    parser.add_argument('--gdb', action='store_true', help='run this compile under gdb')
    args = parser.parse_args()
    if args.gdb:
        debugger = shutil.which('gdb')
        if debugger is None:
            raise RuntimeError('gdb unavailable in the builder image')
        forwarded = [a for a in sys.argv[1:] if a != '--gdb']
        raise SystemExit(subprocess.call([debugger, '-batch', '-ex', 'run', '-ex', 'bt 40',
                                        '--args', sys.executable, '-X', 'faulthandler', __file__, *forwarded]))
    selected = importlib.import_module(args.module)
    linear_native = selected.linear_native
    unpack = selected.unpack
    norm_rope = selected.norm_rope
    causal_gqa_gate = selected.causal_gqa_gate
    add_residual = selected.add_residual
    out = Path('/results')
    if args.module != 'native_candidate':
        out = out / args.module
    out.mkdir(parents=True, exist_ok=True)
    if args.frontend_until is not None:
        import _avelang_bindings as bindings
        from avelang.compiler.code_generator import _build_import_module, _get_function_def, _serialize_constexprs
        fn = linear_native
        frontend_constants = {'N': 5120}
        declared_cases = getattr(selected, 'OFFLINE_COMPILE_CASES', None)
        if declared_cases:
            requested = args.only or selected.DEFAULT_PROJECTION_CONFIGS[5120]
            matches = [case for case in declared_cases if case[0] == requested]
            if not matches:
                raise ValueError('unknown frontend case: ' + requested)
            _, fn, frontend_constants, _ = matches[0]
        source = ASTSource(fn, {}, constexprs={
            k: {'type': 'i32', 'value': v} for k, v in frontend_constants.items()},
            global_constexprs=fn._collect_global_constexprs(frontend_constants.keys()))
        tree = copy.deepcopy(_get_function_def(fn.parse()))
        cutoff = args.frontend_until
        def trim(node):
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    retained = []
                    for item in value:
                        if isinstance(item, ast.stmt) and item.lineno > cutoff:
                            continue
                        if isinstance(item, ast.AST):
                            trim(item)
                        retained.append(item)
                    setattr(node, field, retained)
                elif isinstance(value, ast.AST):
                    trim(value)
            if hasattr(node, 'body') and isinstance(node.body, list) and not node.body:
                node.body = [ast.copy_location(ast.Pass(), node)]
        trim(tree)
        print('Frontend projection prefix through parsed line ' + str(cutoff), flush=True)
        generator = bindings.MLIRGenerator()
        generator.generate_from_python_ast(_build_import_module([fn]))
        generator.visit_function_def(tree, _serialize_constexprs(source), 'kernel')
        (out / ('frontend_' + str(cutoff) + '.mlir')).write_text(generator.get_mlir())
        print('Frontend prefix PASS', flush=True)
        return
    assembler = shutil.which('ptxas')
    if assembler is None:
        raise RuntimeError('ptxas is required for offline assembly validation')
    cases = [
        ('qkv_projection', linear_native, {'N': 5120}, 5),
        ('unpack_q_gate_k_v', unpack, {}, 8),
        ('q_norm_rope', norm_rope, {'HEADS': 8}, 1),
        ('k_norm_rope', norm_rope, {'HEADS': 2}, 1),
        ('causal_gqa_gate', causal_gqa_gate, {}, 1),
        ('output_projection', linear_native, {'N': 2048}, 5),
        ('residual', add_residual, {}, 8),
    ]
    cases = getattr(selected, 'OFFLINE_COMPILE_CASES', cases)
    projection_specs = getattr(selected, 'PROJECTION_CONFIGS', {})
    attention_specs = getattr(selected, 'ATTENTION_CONFIGS', {})
    if args.only:
        cases = [case for case in cases if case[0] == args.only]
        if not cases:
            raise ValueError('unknown kernel: ' + args.only)
    report = {'module': args.module, 'invariant_validation': 'unavailable in native branch', 'kernels': []}
    for name, fn, constants, warps in cases:
        start = time.monotonic()
        print('Compiling ' + name, flush=True)
        ptx = compile_one(fn, constants, warps)
        path = out / (name + '.ptx')
        path.write_bytes(ptx)
        asm = subprocess.run([assembler, '-arch=sm_90a', '-v', str(path), '-o',
                              str(out / (name + '.cubin'))], check=True, capture_output=True, text=True)
        (out / (name + '.ptxas.txt')).write_text(asm.stdout + asm.stderr)
        text = ptx.decode()
        is_projection = name in ('qkv_projection', 'output_projection') or name in projection_specs
        if is_projection:
            width = projection_specs[name].get('wgmma_n', projection_specs[name]['tokens']) if name in projection_specs else 128
            required = [f'wgmma.mma_async.sync.aligned.m64n{width}k16.f32.bf16.bf16',
                        'cp.async.bulk.tensor', 'mbarrier', 'wgmma.commit_group',
                        'wgmma.wait_group']
            if args.module in ('native_candidate', 'round2_candidate', 'round3_candidate', 'round4_candidate', 'round4_mma_candidate', 'round4_async_candidate'):
                spec = projection_specs.get(name)
                transposed = spec.get('transpose', spec.get('argument_style') != 'round2') if spec else False
                matrix_store = 'stmatrix.sync.aligned.m8n8.x4.trans.shared.b16' if transposed else 'stmatrix.sync.aligned.m8n8.x4.shared.b16'
                required += [matrix_store,
                             'cp.async.bulk.tensor.2d.global.shared::cta',
                             'cp.async.bulk.commit_group', 'cp.async.bulk.wait_group']
            for instruction in required:
                if instruction not in text:
                    raise AssertionError(name + ' missing ' + instruction)
        if args.module in ('native_candidate', 'round2_candidate', 'round3_candidate', 'round4_candidate', 'round4_mma_candidate', 'round4_async_candidate') and is_projection:
            if not re.search(r'cp\.async\.bulk\.wait_group\s+0;', text):
                raise AssertionError(name + ' missing full TMA-store completion wait')
            if 'st.global' in text:
                raise AssertionError(name + ' retained scalar global output stores')
        # Match the selected kernel's declared configuration, independently of
        # the Python module name, so promotion preserves the same PTX checks.
        attention_config, attention_spec = next(
            ((config, spec) for config, spec in attention_specs.items()
             if spec['kernel'] is fn), (None, {}))
        is_mma_attention = (
            attention_spec.get('qk') == 'BF16 MMA with FP32 accumulator'
            or name in ('attention_mma_q8', 'attention_mma_q8_async'))
        has_async_preload = (
            attention_spec.get('kv_preload') == '16B cp.async.ca'
            or name == 'attention_mma_q8_async')
        attention_checks = []
        if is_mma_attention:
            for instruction in ['mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32',
                                'ldmatrix.sync.aligned.m8n8.x4.shared.b16',
                                'ldmatrix.sync.aligned.m8n8.x2.shared.b16']:
                if instruction not in text:
                    raise AssertionError(name + ' missing ' + instruction)
            attention_checks.append('BF16 QK MMA and ldmatrix x4/x2')
        if is_mma_attention and attention_spec.get('pv', 'FP32') == 'FP32':
            for instruction in ['shfl.sync.idx.b32', 'fma.rn.f32']:
                if instruction not in text:
                    raise AssertionError(name + ' missing FP32 PV instruction ' + instruction)
            if re.search(r'\bfma\.[^\s;]*\.ftz\.[^\s;]*', text):
                raise AssertionError(name + ' retained FTZ FMA in the FP32 PV kernel')
            attention_checks.append('FP32 probability shuffle and non-FTZ PV FMA')
        if has_async_preload:
            if not re.search(r'cp\.async\.ca\.shared\.global\s+[^;]+,\s*16;', text):
                raise AssertionError(name + ' missing 16-byte asynchronous K/V copies')
            if 'cp.async.commit_group;' not in text:
                raise AssertionError(name + ' missing asynchronous-copy group commit')
            if not re.search(r'cp\.async\.wait_group\s+0;', text):
                raise AssertionError(name + ' missing complete asynchronous-copy wait')
            attention_checks.append('16-byte cp.async.ca with commit and wait_group 0')
        if '.extern .func' in text:
            raise AssertionError(name + ' has unresolved device-function dependency')
        result = {'name': name, 'seconds': time.monotonic() - start,
                  'ptx_bytes': len(ptx), 'ptxas': asm.stderr.strip(),
                  'has_ptx_local_memory': '.local ' in text}
        shared_match = re.search(r'__avelang_dynamic_shared_([0-9]+)', text)
        result['dynamic_shared_bytes'] = int(shared_match.group(1)) if shared_match else 0
        if name in projection_specs:
            result['projection_config'] = {k: v for k, v in projection_specs[name].items() if k != 'kernel'}
        if attention_config is not None:
            result['attention_config_name'] = attention_config
            result['attention_config'] = {k: v for k, v in attention_spec.items() if k != 'kernel'}
        if attention_checks:
            result['attention_ptx_checks'] = attention_checks
        report['kernels'].append(result)
        print(json.dumps(result), flush=True)
    (out / ('compile_' + args.only + '.json' if args.only else 'compile.json')).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
