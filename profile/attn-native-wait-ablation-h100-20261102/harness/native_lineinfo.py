"""Attach PTX line tables for profiling without changing GPU instructions."""
import hashlib
import json
import re
from pathlib import Path


def annotate(ptx, source_path):
    lines = ptx.splitlines()
    result = []
    entry = False
    body = False
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('.visible .entry'):
            entry = True
        if entry and stripped == '{':
            body = True
        if body and ';' in stripped and not stripped.startswith(('.', '//', '{', '}')):
            result.append(f'\t.loc 1 {number} 0')
        result.append(line)
        if stripped.startswith('.address_size'):
            result.append(f'.file 1 "{source_path}"')
    decorated = '\n'.join(result) + '\n'
    # This check concerns emitted instruction identity, not numerical results.
    restored = '\n'.join(line for line in decorated.splitlines()
                         if not line.strip().startswith(('.loc ', '.file '))) + '\n'
    assert restored == ptx
    return decorated


def install(directory):
    import avelang.backends.nvidia.driver as backend
    from avelang.runtime.driver import driver
    from avelang.runtime.build import compile_module_from_src

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(backend.__file__).with_name('driver.c').read_text()
    old = 'cuModuleLoadData(&mod, data)'
    assert source.count(old) == 1
    source = source.replace('    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(' + old + ');',
        '''    CUjit_option profile_options[] = {CU_JIT_GENERATE_LINE_INFO};
    void *profile_values[] = {(void *)(uintptr_t)1};
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        cuModuleLoadDataEx(&mod, data, 1, profile_options, profile_values));''')
    source = source.replace('"cuda_utils"', '"profile_cuda_utils"')
    source = source.replace('PyInit_cuda_utils', 'PyInit_profile_cuda_utils')
    (directory.parent.parent / 'build/profile_cuda_utils.c').write_text(source)
    module = compile_module_from_src(source, 'profile_cuda_utils',
        library_dirs=backend.library_dirs(), include_dirs=backend.include_dirs,
        libraries=backend.libraries)
    records = []

    def load(name, data, shared, device, prefer_l1=0):
        ptx = data.decode()
        original_ptx = ptx
        ptx, changed = re.subn(
            r"(mbarrier\.try_wait\.parity\.shared::cta\.b64 done, \[%r\d+\], %r\d+);",
            r"\1, 10000000;", ptx)
        assert changed == (2 if name == 'linear_native' else 0)
        # Diagnostic-only ABI-preserving wait hint; production sources unchanged.
        tag = name + '_' + hashlib.sha256(data).hexdigest()[:10]
        path = directory / (tag + '.ptx')
        path.write_text(ptx)
        (directory/(tag+'.original.ptx')).write_text(original_ptx)
        annotated = annotate(ptx, path)
        (directory / (tag + '.lineinfo.ptx')).write_text(annotated)
        result = module.load_binary(name, annotated.encode(), shared, device, prefer_l1)
        records.append({'name':name, 'wait_ticks_added':changed, 'ptx':str(path), 'shared_bytes':shared,
                        'registers':result[2], 'local_words':result[3], 'max_threads':result[4]})
        (directory / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
        return result

    driver.active.utils.load_binary = load
