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


def iter_source_files(root: Path) -> Iterable[Path]:
    ignored = {".git", ".venv", "build", "dist", "node_modules", "__pycache__"}
    suffixes = {".py", ".pyx", ".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh"}
    for p in root.rglob("*"):
        if (
            p.is_file()
            and p.suffix in suffixes
            and not any(part in ignored for part in p.parts)
        ):
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
