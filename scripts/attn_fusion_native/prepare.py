"""Prepare the supplied native Hopper branch without editing its original checkout."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
COMMIT = '5f349c594ad510ad6b90d0e1ca310c0d47aa9ed1'
PATCH = ROOT / 'third_party/AveLang/native-build.patch'


def git(directory, *args, check=True):
    return subprocess.run(['git', '-C', str(directory), *args], check=check,
                          text=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT / 'avelang')
    parser.add_argument('--dest', type=Path, default=ROOT / '.repro/avelang-native')
    args = parser.parse_args()
    if not args.dest.exists():
        subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout',
                        str(args.repo.resolve()), str(args.dest)], check=True)
        git(args.dest, 'checkout', '--detach', COMMIT)
    actual = git(args.dest, 'rev-parse', 'HEAD').stdout.strip()
    if actual != COMMIT:
        raise RuntimeError(f'{args.dest} is at {actual}, expected {COMMIT}; existing checkout preserved')
    if git(args.dest, 'apply', '--reverse', '--check', str(PATCH), check=False).returncode == 0:
        print(f'Already prepared: {args.dest}')
        return
    result = git(args.dest, 'apply', '--check', str(PATCH), check=False)
    if result.returncode:
        raise RuntimeError('Build patch conflicts with existing changes; no files overwritten:\n' + result.stderr)
    git(args.dest, 'apply', str(PATCH))
    print(f'Prepared {COMMIT}: {args.dest}')


if __name__ == '__main__':
    main()
