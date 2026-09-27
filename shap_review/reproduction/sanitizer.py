from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

SANITIZER_PATTERNS = {
    "asan": re.compile(
        r"AddressSanitizer|heap-use-after-free|heap-buffer-overflow|stack-use-after",
        re.IGNORECASE,
    ),
    "ubsan": re.compile(r"UndefinedBehaviorSanitizer|runtime error:", re.IGNORECASE),
}


def sanitizer_environment(kind: str) -> dict[str, str]:
    env = os.environ.copy()
    kind = kind.lower()
    if kind == "asan":
        env.setdefault(
            "ASAN_OPTIONS", "detect_leaks=1:abort_on_error=1:halt_on_error=1"
        )
    elif kind == "ubsan":
        env.setdefault("UBSAN_OPTIONS", "halt_on_error=1:print_stacktrace=1")
    return env


def run_sanitized(
    script: str | Path,
    kind: str,
    python: str | None = None,
    timeout: int = 120,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> dict:
    executable = python or sys.executable
    merged = sanitizer_environment(kind)
    merged.update(env or {})
    try:
        p = subprocess.run(
            [executable, str(script)],
            cwd=str(cwd) if cwd else None,
            env=merged,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        combined = (p.stdout or "") + "\n" + (p.stderr or "")
        pattern = SANITIZER_PATTERNS.get(kind.lower())
        finding = bool(pattern and pattern.search(combined))
        verdict = (
            "SANITIZER_FINDING"
            if finding
            else (
                "SANITIZER_CLEAN" if p.returncode == 0 else "SANITIZER_EXECUTION_FAILED"
            )
        )
        return {
            "ok": p.returncode == 0 and not finding,
            "sanitizer": kind.lower(),
            "returncode": p.returncode,
            "finding": finding,
            "verdict": verdict,
            "instrumentation_assumed": True,
            "memory_safety_confirmation": False,
            "memory_safety_note": "A sanitizer finding is strong memory-safety evidence, not a formal proof.",
            "stdout": p.stdout[-30000:],
            "stderr": p.stderr[-30000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "sanitizer": kind.lower(),
            "timeout": True,
            "returncode": None,
            "finding": False,
            "verdict": "SANITIZER_EXECUTION_FAILED",
            "memory_safety_confirmation": False,
            "memory_safety_note": "Sanitizer execution did not establish memory safety.",
            "stdout": (exc.stdout or "")[-30000:]
            if isinstance(exc.stdout, str)
            else "",
            "stderr": (exc.stderr or "")[-30000:]
            if isinstance(exc.stderr, str)
            else "",
        }
