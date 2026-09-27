from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class OracleResult:
    passed: bool
    message: str
    details: dict[str, Any]


def numeric_additivity(
    values, base_values, model_output, rtol=1e-2, atol=1e-6
) -> OracleResult:
    import numpy as np

    lhs = np.asarray(base_values) + np.asarray(values).sum(axis=-1)
    rhs = np.asarray(model_output)
    ok = np.allclose(lhs, rhs, rtol=rtol, atol=atol, equal_nan=False)
    err = np.max(np.abs(lhs - rhs)) if lhs.size else 0.0
    return OracleResult(
        bool(ok),
        "additivity holds" if ok else f"additivity mismatch; max_abs_error={err}",
        {"max_abs_error": float(err), "rtol": rtol, "atol": atol},
    )


def shape_equal(actual, expected) -> OracleResult:
    a = tuple(getattr(actual, "shape", actual))
    e = tuple(getattr(expected, "shape", expected))
    ok = a == e
    return OracleResult(
        ok,
        "shape matches" if ok else f"shape mismatch: {a} != {e}",
        {"actual": a, "expected": e},
    )
