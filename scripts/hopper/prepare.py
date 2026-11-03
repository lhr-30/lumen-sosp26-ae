#!/usr/bin/env python3
"""Fetch the pinned artifact compiler and apply the reviewable Hopper patches."""
import argparse
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def run(*args, cwd=None, check=True):
    return subprocess.run(args, cwd=cwd, check=check, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / ".repro/avelang-artifact")
    args = parser.parse_args()
    source = args.source.resolve()
    spec_dir = ROOT / "third_party/AveLang"
    spec = tomllib.loads((spec_dir / "source.toml").read_text())["repo"]
    if not source.exists():
        source.mkdir(parents=True)
        run("git", "init", str(source))
        run("git", "remote", "add", "origin", spec["upstream"], cwd=source)
        run("git", "fetch", "--depth=1", "origin", spec["base_commit"], cwd=source)
        run("git", "checkout", "--detach", "FETCH_HEAD", cwd=source)
    head = run("git", "rev-parse", "HEAD", cwd=source).stdout.strip()
    if head != spec["base_commit"]:
        raise SystemExit(f"Refusing to patch {source}: expected {spec['base_commit']}, got {head}")
    for name in spec["patches"]:
        patch = str(spec_dir / name)
        if run("git", "apply", "--reverse", "--check", patch, cwd=source, check=False).returncode == 0:
            print(f"Already applied: {name}")
            continue
        run("git", "apply", "--check", patch, cwd=source)
        run("git", "apply", patch, cwd=source)
        print(f"Applied: {name}")
    print(source)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.stderr or str(exc)) from exc
