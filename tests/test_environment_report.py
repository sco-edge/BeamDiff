import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from scripts.report_environment import gpu_metadata, package_versions


def test_report_is_valid_json_without_gpu_framework_imports():
    script = Path(__file__).resolve().parents[1] / 'scripts/report_environment.py'
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, check=True)
    report = json.loads(result.stdout)
    assert report['python']
    assert 'numpy' in report['packages']
    assert 'gpu' not in report
    assert 'Current environment only' in report['note']


def test_missing_gpu_tools_are_reported_not_fatal():
    with patch('scripts.report_environment.subprocess.run', side_effect=FileNotFoundError):
        assert gpu_metadata() == {'available': False, 'reason': 'FileNotFoundError'}


def test_package_report_uses_strings_or_null():
    assert all(isinstance(value, str) or value is None for value in package_versions().values())
