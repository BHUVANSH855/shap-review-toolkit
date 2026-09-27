from __future__ import annotations

from typing import Any

import numpy as np

from shap_review.contracts.tensor import (
    SHAPSemanticTensor,
    semantic_align,
)

from .policy import classify_field

SEMANTIC_KEYS = {
    "values",
    "base_values",
    "expected_value",
    "feature_names",
    "output_names",
    "model_output",
    "shape",
    "dtype",
    "data",
    "interaction_values",
    "error_std",
    "output_indexes",
}


def normalize_shap_result(value: Any) -> Any:
    """Normalize legacy lists and Explanation objects into a semantic JSON form."""
    if hasattr(value, "values") and not isinstance(value, (dict, list, tuple)):
        result = {"values": normalize_shap_result(value.values)}
        for name in (
            "base_values",
            "data",
            "feature_names",
            "output_names",
            "output_indexes",
        ):
            if hasattr(value, name):
                result[name] = normalize_shap_result(getattr(value, name))
        return result
    if isinstance(value, (list, tuple)):
        if value and all(hasattr(x, "shape") for x in value):
            try:
                arrays = [np.asarray(x) for x in value]
                if len({a.shape for a in arrays}) == 1:
                    return np.stack(arrays, axis=-1).tolist()
            except Exception:
                pass
        return [normalize_shap_result(v) for v in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, dict):
        return {
            str(k): normalize_shap_result(v)
            for k, v in value.items()
            if str(k) in SEMANTIC_KEYS or not str(k).startswith("_")
        }
    return value


def _as_array(value):
    try:
        return np.asarray(value, dtype=float)
    except Exception:
        return None


def _semantic_array(result, field="values", axis_spec=None):
    arr = _as_array(result.get(field) if isinstance(result, dict) else result)
    return arr


def evaluate_additivity(
    values,
    base_values,
    model_output,
    rtol=1e-5,
    atol=1e-8,
    *,
    interaction=False,
    axis_spec=None,
):
    vals = _as_array(values)
    base = _as_array(base_values)
    target = _as_array(model_output)
    if vals is None or base is None or target is None:
        return {"applicable": False, "reason": "non-numeric-or-missing-values"}
    try:
        tensor = SHAPSemanticTensor.from_values(
            vals, base_values=base, interaction=interaction, axis_spec=axis_spec
        )
        reconstructed = tensor.reconstruction()
        reconstructed, target, broadcast_details = semantic_align(
            reconstructed, target, axes=axis_spec, role="target"
        )
        diff = reconstructed - target
        err = float(np.nanmax(np.abs(diff))) if diff.size else 0.0
        return {
            "applicable": True,
            "passed": bool(
                np.allclose(reconstructed, target, rtol=rtol, atol=atol, equal_nan=True)
            ),
            "max_error": err,
            "rtol": rtol,
            "atol": atol,
            "broadcast": broadcast_details,
        }
    except ValueError as exc:
        return {"applicable": False, "reason": str(exc)}


def compare_shap_contract(left: Any, right: Any, rtol=1e-5, atol=1e-8, contract=None):
    a = normalize_shap_result(left)
    b = normalize_shap_result(right)
    if not isinstance(a, dict) or not isinstance(b, dict):
        return {"equal": a == b, "semantic_equal": a == b, "fields": {}}
    fields = sorted(set(a) | set(b))
    field_results = {}
    for field in fields:
        policy = classify_field(field)
        if field not in a or field not in b:
            field_results[field] = {
                "equal": False,
                "severity": policy,
                "reason": "missing-field",
            }
            continue
        aa, bb = _as_array(a[field]), _as_array(b[field])
        if aa is not None and bb is not None:
            # Base-value broadcasting and legacy output-axis normalization are semantic,
            # not structural differences.
            if aa.shape != bb.shape:
                try:
                    aa, bb, align_details = semantic_align(
                        aa,
                        bb,
                        axes=contract.axis_spec
                        if contract is not None and hasattr(contract, "axis_spec")
                        else None,
                        role=field,
                    )
                except ValueError as exc:
                    field_results[field] = {
                        "equal": False,
                        "severity": policy,
                        "shape_left": list(aa.shape),
                        "shape_right": list(bb.shape),
                        "reason": f"semantic-normalization-could-not-align-shapes: {exc}",
                    }
                    continue
            else:
                align_details = {
                    "broadcast_applied": False,
                    "source_shape": list(aa.shape),
                    "target_shape": list(bb.shape),
                    "semantic_axis": None,
                    "axes": [],
                    "reason": "exact-shape",
                    "role": field,
                }
            if aa.shape == bb.shape:
                equal = bool(np.allclose(aa, bb, rtol=rtol, atol=atol, equal_nan=True))
                field_results[field] = {
                    "equal": equal,
                    "severity": policy,
                    "shape": list(aa.shape),
                    "max_abs_delta": float(np.nanmax(np.abs(aa - bb)))
                    if aa.size
                    else 0.0,
                    "alignment": align_details,
                }
        else:
            equal = a[field] == b[field]
            field_results[field] = {"equal": equal, "severity": policy}
    critical = [
        v for v in field_results.values() if v.get("severity") == "correctness-critical"
    ]
    result = {
        "equal": all(x["equal"] for x in field_results.values()),
        "semantic_equal": all(x["equal"] for x in critical) if critical else True,
        "fields": field_results,
        "normalization": "legacy list outputs stacked on output axis; Explanation values/base_values canonicalized; baseline broadcasting is contract-aware",
    }
    if contract is not None:
        from shap_review.contracts.shap_contract import validate_contract

        result["contract_validation"] = {
            "left": validate_contract(
                contract,
                a.get("values"),
                a.get("base_values"),
                expected_value=a.get("expected_value"),
            )
            if "values" in a
            else None,
            "right": validate_contract(
                contract,
                b.get("values"),
                b.get("base_values"),
                expected_value=b.get("expected_value"),
            )
            if "values" in b
            else None,
        }
    return result
