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
    """Check SHAP additivity using the canonical semantic tensor path.

    The reconstruction is delegated to SHAPSemanticTensor so that feature
    reduction remains aware of multiclass and interaction tensor layouts.

    Invalid tolerance values are rejected explicitly rather than being passed
    through to NumPy, where the resulting error would be less specific.
    """
    import numpy as np

    from shap_review.contracts.tensor import SHAPSemanticTensor

    if rtol < 0 or atol < 0:
        return OracleResult(
            False,
            "additivity check failed: tolerances must be non-negative",
            {"rtol": rtol, "atol": atol},
        )

    try:
        tensor = SHAPSemanticTensor.from_values(values, base_values=base_values)
        reconstructed = np.asarray(tensor.reconstruction())
        rhs = np.asarray(model_output, dtype=float)

        ok = np.allclose(
            reconstructed,
            rhs,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        )
        err = float(np.max(np.abs(reconstructed - rhs))) if reconstructed.size else 0.0

        return OracleResult(
            bool(ok),
            "additivity holds"
            if ok
            else f"additivity mismatch; max_abs_error={err:.6g}",
            {
                "max_abs_error": err,
                "rtol": rtol,
                "atol": atol,
                "values_shape": list(np.asarray(values).shape),
                "reconstructed_shape": list(reconstructed.shape),
                "target_shape": list(rhs.shape),
                "feature_axis": tensor.axis_spec.feature_axis,
            },
        )
    except (ValueError, TypeError) as exc:
        return OracleResult(
            False,
            f"additivity check failed: {exc}",
            {"error": str(exc)},
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
