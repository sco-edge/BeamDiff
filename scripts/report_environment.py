"""Print installed package and optional GPU metadata as JSON; do not run training."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import platform
import subprocess
import sys


PACKAGES = ('numpy', 'torch', 'tensorflow', 'tensorflow-cpu', 'keras',
            'scikit-learn', 'matplotlib', 'pytest')


def package_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_metadata():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {'available': False, 'reason': type(error).__name__}
    return {'available': True, 'devices': result.stdout.strip().splitlines()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true', help='also query NVIDIA GPU metadata')
    args = parser.parse_args()
    report = {
        'python': platform.python_version(),
        'os': platform.system(),
        'machine': platform.machine(),
        'packages': package_versions(),
        'note': 'Current environment only; not proof of the original experimental environment.',
    }
    if args.gpu:
        report['gpu'] = gpu_metadata()
    json.dump(report, sys.stdout, indent=2)
    print()


if __name__ == '__main__':
    main()
