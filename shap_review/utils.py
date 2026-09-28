from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def iter_source_files(root: Path, include_tests: bool = False) -> Iterable[Path]:
    """Iterate over analysable source files under *root*.

    Test directories, example directories, documentation directories, and build
    artefacts are excluded by default.  Pass ``include_tests=True`` to include
    them (useful when analysing test-coverage gaps rather than implementation
    correctness).

    Without this exclusion, running against the real SHAP repository would flag
    every test file that calls ``shap.TreeExplainer`` as a candidate, generating
    hundreds of low-quality signals and drowning real findings.

    IMPORTANT: exclusion is based on path components RELATIVE to root, not
    absolute path.  This means that if root itself is inside a directory named
    "tests/" (e.g. a test fixture), its files are still yielded correctly.
    """
    ignored_always = {".git", ".venv", "build", "dist", "node_modules", "__pycache__"}
    # Excluded by default: these directories contain *correct* SHAP usage (tests,
    # examples, docs) that would flood the candidate list with false positives.
    ignored_by_default = {
        "tests",
        "test",
        "testing",
        "examples",
        "example",
        "benchmarks",
        "benchmark",
        "notebooks",
        "notebook",
        "docs",
        "doc",
        "documentation",
    }
    ignored = ignored_always | (set() if include_tests else ignored_by_default)
    suffixes = {".py", ".pyx", ".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh"}
    root = Path(root).resolve()
    for p in root.rglob("*"):
        if not (p.is_file() and p.suffix in suffixes):
            continue
        # Only check relative parts (parts after root), not the absolute path.
        # This ensures that a root that lives inside a "tests/" ancestor is still
        # fully scanned — we only want to skip subdirectories of root itself.
        try:
            rel_parts = p.relative_to(root).parts
        except ValueError:
            rel_parts = p.parts
        if not any(part in ignored for part in rel_parts):
            yield p


def rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def fingerprint(*parts: str) -> str:
    raw = "\0".join(parts).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def symbol_at(text: str, line: int) -> str | None:
    lines = text.splitlines()
    start = max(0, line - 1)
    for i in range(start, max(-1, start - 40), -1):
        m = re.match(
            r"\s*(?:async\s+)?def\s+(\w+)|\s*class\s+(\w+)",
            lines[i] if i < len(lines) else "",
        )
        if m:
            return m.group(1) or m.group(2)
    return None
