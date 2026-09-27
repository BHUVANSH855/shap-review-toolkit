from __future__ import annotations

from pathlib import Path

from shap_review.utils import read_text


def source_context(
    root: str | Path, file: str, line: int | None, window: int = 20
) -> dict:
    p = Path(root) / file
    text = read_text(p)
    lines = text.splitlines()
    idx = max(0, (line or 1) - 1)
    return {
        "file": file,
        "start": max(1, idx + 1 - window),
        "end": min(len(lines), idx + 1 + window),
        "lines": lines[max(0, idx - window) : idx + 1 + window],
    }
