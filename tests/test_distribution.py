import subprocess
import sys
from pathlib import Path


def test_distribution_hygiene_script():
    root = Path(__file__).resolve().parents[1]
    validator = root / "scripts" / "validate_distribution.py"

    result = subprocess.run(
        [sys.executable, str(validator)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr