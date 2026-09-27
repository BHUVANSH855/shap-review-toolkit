from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def run_script(
    script: str,
    timeout: int = 60,
    python: str | None = None,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> dict:
    start = time.monotonic()
    executable = python or sys.executable
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        p = subprocess.run(
            [executable, script],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=merged_env,
            cwd=cwd,
        )
        return {
            "returncode": p.returncode,
            "passed": p.returncode == 0,
            "stdout": p.stdout[-10000:],
            "stderr": p.stderr[-10000:],
            "duration_seconds": round(time.monotonic() - start, 3),
            "python": executable,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "returncode": None,
            "passed": False,
            "timeout": True,
            "stdout": (e.stdout or "")[-10000:] if isinstance(e.stdout, str) else "",
            "stderr": (e.stderr or "")[-10000:] if isinstance(e.stderr, str) else "",
            "duration_seconds": round(time.monotonic() - start, 3),
            "python": executable,
        }
