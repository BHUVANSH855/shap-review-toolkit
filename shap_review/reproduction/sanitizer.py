from __future__ import annotations

import os
import platform
import re
import shutil
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

# Shared library name fragments that indicate a sanitizer-instrumented runtime.
_ASAN_LIB_PATTERNS = re.compile(
    r"libasan|libclang_rt\.asan|asan_dynamic", re.IGNORECASE
)
_UBSAN_LIB_PATTERNS = re.compile(
    r"libubsan|libclang_rt\.ubsan", re.IGNORECASE
)


def _linked_libraries(executable: str) -> str:
    """Return a string of linked library names for *executable*.

    Uses ldd (Linux), otool -L (macOS), or dumpbin /dependents (Windows).
    Returns an empty string on failure — callers treat absence as unknown,
    not as clean.
    """
    system = platform.system()
    try:
        if system == "Linux" and shutil.which("ldd"):
            result = subprocess.run(
                ["ldd", executable],
                capture_output=True, text=True, timeout=10, check=False,
            )
            return result.stdout + result.stderr
        if system == "Darwin" and shutil.which("otool"):
            result = subprocess.run(
                ["otool", "-L", executable],
                capture_output=True, text=True, timeout=10, check=False,
            )
            return result.stdout + result.stderr
        if system == "Windows" and shutil.which("dumpbin"):
            result = subprocess.run(
                ["dumpbin", "/dependents", executable],
                capture_output=True, text=True, timeout=10, check=False,
            )
            return result.stdout + result.stderr
    except Exception:  # noqa: BLE001, S110
        pass  # ldd/otool/dumpbin unavailable or failed — caller treats empty string as unknown
    return ""


def detect_instrumentation(kind: str, executable: str | None = None) -> dict:
    """Detect whether *executable* is instrumented for *kind* (asan or ubsan).

    Returns a dict with keys:
      instrumented: bool | None   — True/False when determined; None when unknown
      method: str                 — how instrumentation was detected
      note: str                   — human-readable explanation
      executable: str             — path that was inspected
    """
    exe = executable or sys.executable
    kind = kind.lower()

    # Fast path: sanitizer env vars that instrumented binaries set at startup.
    if kind == "asan":
        lib_pattern = _ASAN_LIB_PATTERNS
        env_indicator = os.environ.get("ASAN_OPTIONS") or os.environ.get("ASAN_SYMBOLIZER_PATH")
    elif kind == "ubsan":
        lib_pattern = _UBSAN_LIB_PATTERNS
        env_indicator = os.environ.get("UBSAN_OPTIONS")
    else:
        return {
            "instrumented": None,
            "method": "unknown_sanitizer",
            "note": f"Unknown sanitizer kind {kind!r} — cannot detect instrumentation.",
            "executable": exe,
        }

    # Check linked libraries — most reliable signal.
    libs = _linked_libraries(exe)
    if libs:
        instrumented = bool(lib_pattern.search(libs))
        return {
            "instrumented": instrumented,
            "method": "linked_libraries",
            "note": (
                f"Sanitizer runtime library detected in linked libraries of {exe}."
                if instrumented
                else (
                    f"No {kind.upper()} runtime library found in linked libraries of {exe}. "
                    "SANITIZER_CLEAN results from this binary have no memory-safety meaning."
                )
            ),
            "executable": exe,
        }

    # Fallback: env var presence is a weak signal (user may have set it manually).
    if env_indicator:
        return {
            "instrumented": None,
            "method": "env_var_only",
            "note": (
                f"{kind.upper()} environment variable is set but linked-library check "
                "was unavailable. Instrumentation cannot be confirmed."
            ),
            "executable": exe,
        }

    # Cannot determine — ldd/otool/dumpbin unavailable and no env var.
    return {
        "instrumented": None,
        "method": "unavailable",
        "note": (
            f"Cannot determine whether {exe} is {kind.upper()}-instrumented. "
            "ldd/otool/dumpbin are not available and no sanitizer env var is set. "
            "Pass an explicitly instrumented Python binary via the python= argument."
        ),
        "executable": exe,
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
    kind_lower = kind.lower()

    # Check instrumentation before running — avoids misleading SANITIZER_CLEAN.
    instrumentation = detect_instrumentation(kind_lower, executable)
    instrumented = instrumentation["instrumented"]

    if instrumented is False:
        # Binary is confirmed NOT instrumented — refuse to run and report clearly.
        return {
            "ok": False,
            "sanitizer": kind_lower,
            "returncode": None,
            "finding": False,
            "verdict": "NOT_INSTRUMENTED",
            "instrumented": False,
            "instrumentation_detection": instrumentation,
            "memory_safety_confirmation": False,
            "memory_safety_note": (
                f"The Python binary at {executable!r} is not instrumented with "
                f"{kind.upper()}. Results from a non-instrumented binary have no "
                "memory-safety meaning. Pass a sanitizer-instrumented Python binary "
                "via the python= argument (e.g. a CPython built with "
                "--with-address-sanitizer)."
            ),
            "stdout": "",
            "stderr": "",
        }

    merged = sanitizer_environment(kind_lower)
    merged.update(env or {})

    try:
        p = subprocess.run(
            [executable, str(script)],
            cwd=str(cwd) if cwd else None,
            env=merged,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        combined = (p.stdout or "") + "\n" + (p.stderr or "")
        pattern = SANITIZER_PATTERNS.get(kind_lower)
        finding = bool(pattern and pattern.search(combined))

        if instrumented is None:
            # Instrumentation unknown — downgrade SANITIZER_CLEAN to INSTRUMENTATION_UNKNOWN.
            if finding:
                verdict = "SANITIZER_FINDING"
            elif p.returncode == 0:
                verdict = "INSTRUMENTATION_UNKNOWN"
            else:
                verdict = "SANITIZER_EXECUTION_FAILED"
        else:
            # instrumented is True — normal verdict.
            if finding:
                verdict = "SANITIZER_FINDING"
            elif p.returncode == 0:
                verdict = "SANITIZER_CLEAN"
            else:
                verdict = "SANITIZER_EXECUTION_FAILED"

        memory_safety_note = (
            "A sanitizer finding is strong memory-safety evidence, not a formal proof."
            if finding
            else (
                "SANITIZER_CLEAN confirmed on an instrumented binary."
                if instrumented is True
                else (
                    "Instrumentation could not be confirmed — SANITIZER_CLEAN verdict "
                    "has no memory-safety meaning without a verified instrumented binary."
                )
            )
        )

        return {
            "ok": p.returncode == 0 and not finding,
            "sanitizer": kind_lower,
            "returncode": p.returncode,
            "finding": finding,
            "verdict": verdict,
            "instrumented": instrumented,
            "instrumentation_detection": instrumentation,
            "memory_safety_confirmation": bool(instrumented and not finding),
            "memory_safety_note": memory_safety_note,
            "stdout": p.stdout[-30000:],
            "stderr": p.stderr[-30000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "sanitizer": kind_lower,
            "timeout": True,
            "returncode": None,
            "finding": False,
            "verdict": "SANITIZER_EXECUTION_FAILED",
            "instrumented": instrumented,
            "instrumentation_detection": instrumentation,
            "memory_safety_confirmation": False,
            "memory_safety_note": "Sanitizer execution did not complete within the timeout.",
            "stdout": (exc.stdout or "")[-30000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-30000:] if isinstance(exc.stderr, str) else "",
        }