from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .oracles import SHAPSemanticOracle
from .tensor import SHAPAxisSpec

# Dtype-adaptive tolerances.  float32 arithmetic can legitimately produce
# errors up to ~5e-4; using 1e-6 on float32 causes valid computations to fail
# the additivity and output-space oracles (false positives).
_DTYPE_TOLERANCE: dict[str, float] = {
    "float16": 1e-2,
    "float32": 5e-4,
    "float64": 1e-6,
    "float128": 1e-10,
}
_DEFAULT_TOLERANCE = 1e-6


def dtype_tolerance(dtype) -> float:
    """Return an appropriate numerical tolerance for *dtype*.

    Parameters
    ----------
    dtype:
        A numpy dtype object, dtype string, or ``None``.
        When ``None`` or unrecognised the default ``1e-6`` is returned.

    Returns
    -------
    float
        Tolerance suitable for both ``rtol`` and ``atol`` in numpy.allclose.
    """
    if dtype is None:
        return _DEFAULT_TOLERANCE
    name = getattr(dtype, "name", str(dtype))
    for key in _DTYPE_TOLERANCE:
        if key in name:
            return _DTYPE_TOLERANCE[key]
    return _DEFAULT_TOLERANCE



@dataclass(frozen=True)
class SHAPContract:
    explainer: str
    model_family: str
    model_output: str = "raw"
    feature_perturbation: str | None = None
    input_representation: str = "ndarray"
    output_representation: str = "Explanation"
    values_shape: tuple[int, ...] | None = None
    base_values_shape: tuple[int, ...] | None = None
    target_shape: tuple[int, ...] | None = None
    interaction: bool = False
    expected_value_semantics: str = "model-baseline"
    base_value_semantics: str = "scalar"
    target_semantics: str = "scalar"
    contribution_space: str = "raw"
    base_space: str = "raw"
    target_space: str = "raw"
    link_direction: str = "forward"
    additivity_required: bool = True
    output_space_required: bool = True
    expected_value_required: bool = False
    tolerance: float | None = None  # None → dtype-adaptive via dtype_tolerance()
    api_era: str = "modern"
    metadata: dict[str, Any] = field(default_factory=dict)
    axis_spec: SHAPAxisSpec | None = None


    def resolved_tolerance(self, dtype=None) -> float:
        """Return effective tolerance, applying dtype-adaptive defaults when tolerance is None."""
        if self.tolerance is not None:
            return self.tolerance
        return dtype_tolerance(dtype)

    def to_dict(self):
        d = asdict(self)
        d['tolerance_default_policy'] = 'dtype-adaptive when tolerance is None'
        return d


def validate_contract(
    contract: SHAPContract,
    values,
    base_values=None,
    target=None,
    *,
    expected_value=None,
    interaction_values=None,
    model=None,
    inputs=None,
):
    import dataclasses

    import numpy as np

    v = np.asarray(values)
    b = None if base_values is None else np.asarray(base_values)
    t = None if target is None else np.asarray(target)

    # Apply dtype-adaptive tolerance when contract.tolerance is None.
    # Capture original_tolerance BEFORE any dataclasses.replace so that
    # tolerance_source correctly reflects the original caller intent.
    original_tolerance = contract.tolerance
    inferred_dtype = getattr(v, "dtype", None)
    effective_tolerance = contract.resolved_tolerance(inferred_dtype)
    if original_tolerance is None and effective_tolerance != _DEFAULT_TOLERANCE:
        contract = dataclasses.replace(contract, tolerance=effective_tolerance)

    result = {
        "values_shape": list(v.shape),
        "base_values_shape": None if b is None else list(b.shape),
        "target_shape": None if t is None else list(t.shape),
        "contract": contract.to_dict(),
        "dtype": str(v.dtype),
        "effective_tolerance": effective_tolerance,
        "tolerance_source": (
            "caller_supplied"
            if original_tolerance is not None
            else "dtype_adaptive"
        ),
    }

    def shape_matches(actual, expected):
        if expected is None:
            return True
        if actual is None or len(actual) != len(expected):
            return False
        return all(e == -1 or a == e for a, e in zip(actual, expected))

    result["values_shape_match"] = shape_matches(v.shape, contract.values_shape)
    result["base_values_shape_match"] = shape_matches(
        None if b is None else b.shape, contract.base_values_shape
    )
    result["target_shape_match"] = shape_matches(
        None if t is None else t.shape, contract.target_shape
    )
    semantic = SHAPSemanticOracle().evaluate(
        contract=contract,
        values=v,
        base_values=b,
        model_output=t,
        expected_value=expected_value,
        interaction_values=interaction_values,
        target_shape=contract.target_shape,
        model=model,
        inputs=inputs,
    )
    result["semantic_oracle"] = semantic
    result["valid"] = (
        all(
            result[k]
            for k in (
                "values_shape_match",
                "base_values_shape_match",
                "target_shape_match",
            )
        )
        and semantic["status"] == "PASS"
    )
    return result


def compare_contracts(left: SHAPContract, right: SHAPContract):
    keys = (
        "explainer",
        "model_family",
        "model_output",
        "feature_perturbation",
        "input_representation",
        "output_representation",
        "values_shape",
        "base_values_shape",
        "target_shape",
        "interaction",
        "expected_value_semantics",
        "base_value_semantics",
        "target_semantics",
        "contribution_space",
        "base_space",
        "target_space",
        "link_direction",
        "api_era",
        "axis_spec",
    )
    mismatches = {
        k: (getattr(left, k), getattr(right, k))
        for k in keys
        if getattr(left, k) != getattr(right, k)
    }
    return {"equal": not mismatches, "mismatches": mismatches}
