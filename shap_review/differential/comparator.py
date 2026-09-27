from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Comparison:
    equal: bool
    reason: str
    max_abs_error: float | None = None


def compare_numeric(a, b, rtol=1e-5, atol=1e-8) -> Comparison:
    try:
        import numpy as np

        aa = np.asarray(a)
        bb = np.asarray(b)
        if aa.shape != bb.shape:
            return Comparison(False, f"shape {aa.shape} != {bb.shape}")
        err = float(np.max(np.abs(aa - bb))) if aa.size else 0.0
        return Comparison(
            bool(np.allclose(aa, bb, rtol=rtol, atol=atol, equal_nan=True)),
            "numeric comparison",
            err,
        )
    except (ImportError, TypeError, ValueError):
        return Comparison(a == b, "equality comparison")


def compare(
    a: Any,
    b: Any,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    path: str = "$",
    max_diffs: int = 20,
) -> dict:
    """Recursively compare JSON-compatible results from two executions."""
    diffs: list[dict] = []

    def walk(x: Any, y: Any, p: str) -> None:
        if len(diffs) >= max_diffs:
            return
        if (
            isinstance(x, (int, float))
            and isinstance(y, (int, float))
            and not isinstance(x, bool)
            and not isinstance(y, bool)
        ):
            c = compare_numeric(x, y, rtol, atol)
            if not c.equal:
                diffs.append(
                    {
                        "path": p,
                        "reason": c.reason,
                        "left": x,
                        "right": y,
                        "max_abs_error": c.max_abs_error,
                    }
                )
            return
        if type(x) is not type(y):
            diffs.append(
                {"path": p, "reason": f"type {type(x).__name__} != {type(y).__name__}"}
            )
            return
        if isinstance(x, dict):
            keys = sorted(set(x) | set(y))
            for k in keys:
                if k not in x or k not in y:
                    diffs.append({"path": f"{p}.{k}", "reason": "missing key"})
                else:
                    walk(x[k], y[k], f"{p}.{k}")
            return
        if isinstance(x, list):
            if len(x) != len(y):
                diffs.append({"path": p, "reason": f"length {len(x)} != {len(y)}"})
                return
            for i, (lx, ry) in enumerate(zip(x, y)):
                walk(lx, ry, f"{p}[{i}]")
            return
        if x != y:
            diffs.append({"path": p, "reason": "value mismatch", "left": x, "right": y})

    walk(a, b, path)
    return {"equal": not diffs, "differences": diffs, "difference_count": len(diffs)}
