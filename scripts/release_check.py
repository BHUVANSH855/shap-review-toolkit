from __future__ import annotations

import compileall
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(label: str, args: list[str]) -> None:
    print(f"[{label}]", " ".join(args), flush=True)
    result = subprocess.run(args, cwd=ROOT, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    if not compileall.compile_dir(str(ROOT / "shap_review"), quiet=1):
        raise SystemExit("compileall failed")
    run(
        "import",
        [
            sys.executable,
            "-c",
            "import shap_review; import shap_review.fuzzing; import adapters",
        ],
    )
    run("cli", [sys.executable, "-m", "shap_review.cli", "--help"])
    run("evidence", [sys.executable, "scripts/validate_evidence.py"])
    run("source-hygiene", [sys.executable, "scripts/validate_distribution.py"])
    print("release source checks: PASS", flush=True)
    print(
        "full test gate: run `python -m pytest -q --disable-warnings` separately; "
        "the release archive must also pass `scripts/validate_distribution.py <archive.zip>`."
    )
