"""Offline invariant validation and sm_90a assembly; never initializes CUDA."""
import json
from pathlib import Path
import shutil
import subprocess
import time

from avelang.backends.compiler import GPUTarget
from avelang.backends.nvidia.compiler import NvidiaCompiler
from avelang.compiler.compiler import ASTSource
from candidate import linear_bf16, unpack, norm_rope, causal_gqa_gate, add_residual


def compile_one(fn, constants, warps):
    target = GPUTarget("nvptx64-nvidia-cuda", "sm_90a")
    source = ASTSource(fn, {}, constexprs={
        k: {"type": "i32", "value": v} for k, v in constants.items()})
    return NvidiaCompiler(target).compile(source, target, {
        "num_warps": warps, "validate_invariants": True})


def main():
    out = Path("/results")
    out.mkdir(parents=True, exist_ok=True)
    cases = [
        ("qkv_projection", linear_bf16, {"N": 5120}, 4),
        ("unpack_q_gate_k_v", unpack, {"GATE_SHIFT": 0}, 8),
        ("q_norm_rope", norm_rope, {"HEADS": 8, "PAIR_SHIFT": 0}, 1),
        ("k_norm_rope", norm_rope, {"HEADS": 2, "PAIR_SHIFT": 0}, 1),
        ("causal_gqa_gate", causal_gqa_gate, {"KV_SHIFT": 0}, 1),
        ("output_projection", linear_bf16, {"N": 2048}, 4),
        ("residual", add_residual, {}, 8),
    ]
    report = {"validate_invariants": True, "kernels": [], "negative_invariants": []}
    assembler = shutil.which("ptxas")
    if not assembler:
        raise RuntimeError("ptxas is required for offline validation")
    for name, fn, constants, warps in cases:
        started = time.monotonic()
        ptx = compile_one(fn, constants, warps)
        path = out / (name + ".ptx")
        path.write_bytes(ptx)
        subprocess.run([assembler, "-arch=sm_90a", str(path), "-o", str(out / (name + ".cubin"))], check=True)
        result = {"name": name, "seconds": time.monotonic() - started, "ptx_bytes": len(ptx)}
        report["kernels"].append(result)
        print(json.dumps(result), flush=True)
    for name, fn, constants, warps in [
        ("wrong_gate_offset", unpack, {"GATE_SHIFT": 1}, 8),
        ("wrong_rope_pair", norm_rope, {"HEADS": 8, "PAIR_SHIFT": 1}, 1),
        ("wrong_gqa_group", causal_gqa_gate, {"KV_SHIFT": 1}, 1),
    ]:
        try:
            compile_one(fn, constants, warps)
        except RuntimeError as error:
            detail = str(error)
            if "invariant" not in detail.lower():
                raise
            result = {"name": name, "rejected": True, "diagnostic": detail}
            report["negative_invariants"].append(result)
            print(json.dumps(result), flush=True)
        else:
            raise AssertionError(name + " unexpectedly passed invariant validation")
    (out / "compile.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
