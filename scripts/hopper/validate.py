"""Run inside the host system GPU queue; write correctness evidence as JSON."""
import argparse
import json
from pathlib import Path
import time

import torch
from gemm import gemm_hopper, matmul


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("hopper-results.json"))
    parser.add_argument("--smoke", action="store_true", help="short sanitizer run")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    shapes = [(64, 8, 16), (65, 13, 33)] if args.smoke else [
        (64, 8, 16), (128, 32, 128), (65, 13, 33), (1, 1, 1),
        (17, 9, 7), (64, 8, 0), (64, 16, 17), (129, 25, 257), (256, 128, 256)]
    results = []
    for m, n, k in shapes:
        for seed in ([0] if args.smoke else [0, 17, 123]):
            torch.manual_seed(seed)
            a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
            b = torch.randn((k, n), dtype=torch.bfloat16, device="cuda")
            actual = matmul(a, b)
            torch.cuda.synchronize()
            # FP32 reference avoids BF16 output rounding hiding errors.
            expected = a.float() @ b.float()
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
            # Repeated calls exercise barrier phase reuse and independent CTAs.
            for _ in range(2):
                torch.testing.assert_close(matmul(a, b), expected, rtol=1e-4, atol=1e-4)
            err = (actual - expected).abs().max().item()
            results.append({"shape_mnk": [m, n, k], "seed": seed, "max_abs_error": err})
            print(f"PASS M={m} N={n} K={k} seed={seed} max_abs_error={err:g}", flush=True)
    # Invalid block dimensions must be rejected before CUDA launch.
    a = torch.ones((64, 16), dtype=torch.bfloat16, device="cuda")
    b = torch.ones((16, 8), dtype=torch.bfloat16, device="cuda")
    c = torch.empty((64, 8), dtype=torch.float32, device="cuda")
    try:
        gemm_hopper[lambda: ((1, 1, 1), (32, 1, 1))](a, b, c, 64, 8, 16, num_warps=4)
    except ValueError as exc:
        assert "block=(128, 1, 1)" in str(exc)
    else:
        raise AssertionError("invalid WGMMA launch was accepted")
    report = {"gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
              "torch": torch.__version__, "cuda": torch.version.cuda, "timestamp": time.time(),
              "rtol": 1e-4, "atol": 1e-4, "invalid_block_rejected": True, "cases": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {len(results)} cases to {args.output}", flush=True)


if __name__ == "__main__":
    main()
