from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np

from .tensor import SHAPAxisSpec, infer_axis_spec, reduce_contributions, semantic_align


class TargetSource(str, Enum):
    MODEL_API = "model_api"
    INDEPENDENT_FUNCTION = "independent_function"
    EXTERNAL_REFERENCE = "external_reference"
    USER_SUPPLIED = "user_supplied"
    UNKNOWN = "unknown"


class OracleIndependence(str, Enum):
    PROVEN = "PROVEN"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


def classify_target_provenance(
    source: str | None, explicit_independent: bool | None = None
) -> dict[str, Any]:
    """Classify target provenance without equating a separate model call with an independent oracle."""
    raw = (source or "unknown").lower()
    if raw in {"model.predict", "model.predict_proba", "model_api"}:
        source_kind = TargetSource.MODEL_API.value
        independence = (
            OracleIndependence.PARTIAL.value
            if explicit_independent is True
            else OracleIndependence.UNKNOWN.value
        )
    elif raw in {"independent_probability_fn", "independent_function"}:
        source_kind = TargetSource.INDEPENDENT_FUNCTION.value
        independence = (
            OracleIndependence.PROVEN.value
            if explicit_independent is True
            else OracleIndependence.PARTIAL.value
        )
    elif raw in {"supplied_external", "external_reference"}:
        source_kind = TargetSource.EXTERNAL_REFERENCE.value
        independence = (
            OracleIndependence.PROVEN.value
            if explicit_independent is True
            else OracleIndependence.UNKNOWN.value
        )
    elif raw in {"user_supplied"}:
        source_kind = TargetSource.USER_SUPPLIED.value
        independence = OracleIndependence.UNKNOWN.value
    else:
        source_kind = TargetSource.UNKNOWN.value
        independence = OracleIndependence.UNKNOWN.value
    return {
        "target_source": source_kind,
        "oracle_independence": independence,
        "explicit_independent": explicit_independent,
    }


class OracleStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


def validate_oracle_result(result):
    if result.applicable and result.passed is None:
        raise ValueError(f"{result.name}: applicable=True requires passed")
    if not result.applicable and result.passed is not None:
        raise ValueError(f"{result.name}: applicable=False requires passed=None")
    return result


@dataclass(frozen=True)
class OracleResult:
    name: str
    applicable: bool
    passed: bool | None
    reason: str
    observed: Any = None
    expected: Any = None
    details: dict[str, Any] | None = None
    requirement_source: str | None = None
    contract_required: bool = False
    policy_required: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class OraclePolicy:
    """Contract-driven applicability policy. Required unavailable checks are INCONCLUSIVE."""

    require_shape: bool = True
    require_additivity: bool = True
    require_output_space: bool = True
    require_expected_value: bool = False
    require_interaction: bool = False
    require_input_mutation: bool = False

    def required(self, name: str) -> bool:
        return {
            "ShapeOracle": self.require_shape,
            "AdditivityOracle": self.require_additivity,
            "OutputSpaceOracle": self.require_output_space,
            "ExpectedValueOracle": self.require_expected_value,
            "InteractionOracle": self.require_interaction,
            "InputMutationOracle": self.require_input_mutation,
        }.get(name, False)


class OutputSpaceOracle:
    """Independently validates SHAP reconstruction in the declared output space."""

    def check(
        self,
        *,
        contract,
        shap_values,
        base_values=None,
        model=None,
        inputs=None,
        model_output=None,
        probability_fn=None,
        link: Callable[[Any], Any] | None = None,
        expected_value=None,
        target_source=None,
        independent_target=None,
    ):
        try:
            vals = np.asarray(shap_values, dtype=float)
            base = None if base_values is None else np.asarray(base_values, dtype=float)
            if base is None:
                return OracleResult(
                    "OutputSpaceOracle", False, None, "base_values are required"
                )
            target = (
                None if model_output is None else np.asarray(model_output, dtype=float)
            )
            resolved_source = target_source or (
                "supplied_external" if target is not None else None
            )
            if target is None and model is not None and inputs is not None:
                mode = contract.model_output
                if mode == "probability":
                    if probability_fn is not None:
                        target = np.asarray(probability_fn(model, inputs), dtype=float)
                        resolved_source = "independent_probability_fn"
                    elif hasattr(model, "predict_proba"):
                        target = np.asarray(model.predict_proba(inputs), dtype=float)
                        resolved_source = "model.predict_proba"
                    else:
                        return OracleResult(
                            "OutputSpaceOracle",
                            False,
                            None,
                            "independent probability evaluator unavailable",
                        )
                elif mode == "raw" and hasattr(model, "predict"):
                    target = np.asarray(model.predict(inputs), dtype=float)
                    resolved_source = "model.predict"
                elif mode == "log_loss":
                    if not hasattr(model, "predict_proba"):
                        return OracleResult(
                            "OutputSpaceOracle",
                            False,
                            None,
                            "log_loss requires predict_proba",
                        )
                    probs = np.asarray(model.predict_proba(inputs), dtype=float)
                    labels = (
                        np.asarray(getattr(inputs, "y", None))
                        if hasattr(inputs, "y")
                        else None
                    )
                    return OracleResult(
                        "OutputSpaceOracle",
                        False,
                        None,
                        "log_loss requires explicit true labels",
                        details={"probability_shape": list(probs.shape)},
                    )
                else:
                    return OracleResult(
                        "OutputSpaceOracle",
                        False,
                        None,
                        f"independent evaluator unavailable for {mode}",
                    )
            if target is None:
                return OracleResult(
                    "OutputSpaceOracle", False, None, "target output is required"
                )
            provenance = classify_target_provenance(resolved_source, independent_target)
            # Independence is a provenance property, not a prerequisite for the
            # numerical reconstruction check. The oracle may compute a semantic
            # PASS/FAIL while explicitly reporting that the target is not proven
            # independent. Downstream evidence scoring must use that provenance.
            if independent_target is False:
                return OracleResult(
                    "OutputSpaceOracle",
                    False,
                    None,
                    "target provenance explicitly marked non-independent",
                    details=provenance,
                )
            contrib = _feature_sum(
                vals,
                interaction=contract.interaction,
                axes=getattr(contract, "axis_spec", None),
            )
            reconstructed = np.asarray(contrib) + base
            if link is not None:
                reconstructed = np.asarray(link(reconstructed), dtype=float)
            try:
                reconstructed, target, broadcast_details = _semantic_align(
                    reconstructed,
                    target,
                    role="target",
                    axes=getattr(contract, "axis_spec", None),
                )
            except ValueError as exc:
                return OracleResult(
                    "OutputSpaceOracle", True, False, f"output-shape mismatch: {exc}"
                )
            err = (
                float(np.nanmax(np.abs(reconstructed - target))) if target.size else 0.0
            )
            passed = bool(
                np.allclose(
                    reconstructed,
                    target,
                    rtol=contract.tolerance,
                    atol=contract.tolerance,
                    equal_nan=True,
                )
            )
            return OracleResult(
                "OutputSpaceOracle",
                True,
                passed,
                "SHAP reconstruction matches declared output space"
                if passed
                else "SHAP reconstruction does not match declared output space",
                observed=err,
                expected=contract.tolerance,
                details={
                    "model_output": contract.model_output,
                    "target_shape": list(target.shape),
                    "reconstructed_shape": list(reconstructed.shape),
                    "target_source": resolved_source,
                    "independent_target": provenance["oracle_independence"]
                    == OracleIndependence.PROVEN.value,
                    "oracle_independence": provenance["oracle_independence"],
                    "contribution_space": getattr(
                        contract, "contribution_space", "raw"
                    ),
                    "base_space": getattr(contract, "base_space", "raw"),
                    "target_space": getattr(
                        contract, "target_space", contract.model_output
                    ),
                    "link_direction": getattr(contract, "link_direction", "forward"),
                    "broadcast": broadcast_details,
                },
            )
        except Exception as exc:
            return OracleResult(
                "OutputSpaceOracle",
                False,
                None,
                f"oracle evaluation failed: {type(exc).__name__}: {exc}",
            )


class ExpectedValueOracle:
    """Contract-aware baseline check: scalar/vector expected values may repeat per sample."""

    def check(
        self, *, base_values, expected_value, tolerance=1e-6, semantics="model-baseline"
    ):
        if expected_value is None or base_values is None:
            return OracleResult(
                "ExpectedValueOracle",
                False,
                None,
                "expected value and base values are required",
            )
        b, e = (
            np.asarray(base_values, dtype=float),
            np.asarray(expected_value, dtype=float),
        )
        if semantics in {"per-sample", "dynamic-per-sample"}:
            if b.shape != e.shape:
                return OracleResult(
                    "ExpectedValueOracle",
                    True,
                    False,
                    "per-sample base values require matching expected values",
                    observed=list(b.shape),
                    expected=list(e.shape),
                )
            passed = bool(
                np.allclose(b, e, rtol=tolerance, atol=tolerance, equal_nan=True)
            )
        else:
            try:
                eb, bb, align = semantic_align(e, b, axes=None, role="expected_value")
            except ValueError as exc:
                return OracleResult(
                    "ExpectedValueOracle",
                    True,
                    False,
                    f"expected value is not semantically aligned: {exc}",
                )
            passed = bool(
                np.allclose(bb, eb, rtol=tolerance, atol=tolerance, equal_nan=True)
            )
        details = {"semantics": semantics, "broadcast_shape": list(b.shape)}
        if semantics not in {"per-sample", "dynamic-per-sample"}:
            details["alignment"] = align
        return OracleResult(
            "ExpectedValueOracle",
            True,
            passed,
            "baseline semantics satisfied"
            if passed
            else "base values differ from expected baseline",
            observed=b.tolist(),
            expected=e.tolist(),
            details=details,
        )


class ShapeOracle:
    def check(self, *, values, target_shape=None, expected_shape=None):
        v = np.asarray(values)
        expected = expected_shape or target_shape
        if expected is None:
            return OracleResult("ShapeOracle", False, None, "no expected shape")
        passed = len(v.shape) == len(expected) and all(
            e == -1 or a == e for a, e in zip(v.shape, expected)
        )
        return OracleResult(
            "ShapeOracle",
            True,
            passed,
            "shape matches contract" if passed else "shape mismatch",
            observed=list(v.shape),
            expected=list(expected),
        )


class InteractionOracle:
    def check(self, *, values, interaction_values, axes: SHAPAxisSpec | None = None):
        if interaction_values is None:
            return OracleResult(
                "InteractionOracle", False, None, "no interaction values supplied"
            )
        iv = np.asarray(interaction_values, dtype=float)
        try:
            spec = (axes or infer_axis_spec(iv, interaction=True)).normalize(iv.ndim)
        except ValueError as exc:
            return OracleResult("InteractionOracle", False, None, str(exc))
        pair = spec.interaction_feature_axes
        if pair is None:
            return OracleResult(
                "InteractionOracle",
                False,
                None,
                "interaction feature axes are required",
            )
        if iv.shape[pair[0]] != iv.shape[pair[1]]:
            return OracleResult(
                "InteractionOracle",
                True,
                False,
                "interaction feature axes have different sizes",
            )
        symmetric = bool(
            np.allclose(iv, np.swapaxes(iv, pair[0], pair[1]), equal_nan=True)
        )
        if values is None:
            # Symmetry alone is INCONCLUSIVE — a symmetric but semantically wrong
            # tensor would previously return passed=True here, creating a false
            # positive.  Without a reference SHAP-values tensor we cannot verify
            # reconstruction so we must return passed=None (INCONCLUSIVE).
            return OracleResult(
                "InteractionOracle",
                True,  # applicable=True (we can check symmetry)
                None,  # passed=None → INCONCLUSIVE, not a passing result
                "interaction symmetry "
                + ("satisfied" if symmetric else "violated")
                + "; reconstruction cannot be checked without reference values — result is INCONCLUSIVE",
                details={
                    "interaction_feature_axes": list(pair),
                    "output_axis": spec.output_axis,
                    "symmetry_checked": True,
                    "symmetry_passed": symmetric,
                    "reconstruction_checked": False,
                    "inconclusive_reason": "reference_values_required_for_reconstruction",
                },
            )
        direct = np.asarray(values, dtype=float)
        # If a separate SHAP-value tensor is supplied, row-wise interaction sums
        # must reconstruct it. When ``values`` is itself the interaction tensor,
        # symmetry remains the applicable invariant and reconstruction is skipped.
        if direct.ndim == iv.ndim - 1:
            try:
                reduced = iv.sum(axis=pair[1])
                reduced, target, align = semantic_align(
                    reduced, direct, axes=None, role="interaction_reconstruction"
                )
                reconstruction_ok = bool(np.allclose(reduced, target, equal_nan=True))
            except (ValueError, TypeError):
                reconstruction_ok = False
                align = {
                    "reason": "interaction reconstruction could not be semantically aligned"
                }
            checked = True
        else:
            reconstruction_ok = True
            align = {
                "reason": "interaction tensor supplied as values; reconstruction not applicable"
            }
            checked = False
        passed = symmetric and reconstruction_ok
        reason = (
            "interaction symmetry and reconstruction satisfied"
            if passed
            else ("interaction symmetry/reconstruction violation")
        )
        return OracleResult(
            "InteractionOracle",
            True,
            passed,
            reason,
            details={
                "interaction_feature_axes": list(pair),
                "output_axis": spec.output_axis,
                "symmetry_checked": True,
                "reconstruction_checked": checked,
                "reconstruction_passed": reconstruction_ok,
                "reconstruction_alignment": align,
            },
        )


class AdditivityOracle:
    def check(
        self,
        *,
        values,
        base_values,
        model_output,
        contract,
        link: Callable[[Any], Any] | None = None,
        required_override: bool = False,
    ):
        if not contract.additivity_required and not required_override:
            return OracleResult(
                "AdditivityOracle", False, None, "contract disables additivity"
            )
        if values is None:
            return OracleResult(
                "AdditivityOracle", False, None, "SHAP values are required"
            )
        if base_values is None:
            return OracleResult(
                "AdditivityOracle", False, None, "base_values are required"
            )
        if model_output is None:
            return OracleResult(
                "AdditivityOracle", False, None, "model output is required"
            )
        try:
            v = np.asarray(values, dtype=float)
            b = np.asarray(base_values, dtype=float)
            target = np.asarray(model_output, dtype=float)
            if v.ndim < 2:
                return OracleResult(
                    "AdditivityOracle", False, None, "values lack feature axis"
                )
            contrib = _feature_sum(
                v,
                interaction=contract.interaction,
                axes=getattr(contract, "axis_spec", None),
            )
            reconstructed = contrib + b
            if link is not None:
                reconstructed = np.asarray(link(reconstructed), dtype=float)
            try:
                reconstructed, target, broadcast_details = _semantic_align(
                    reconstructed,
                    target,
                    role="target",
                    axes=getattr(contract, "axis_spec", None),
                )
            except ValueError as exc:
                return OracleResult(
                    "AdditivityOracle",
                    True,
                    False,
                    f"reconstruction shape mismatch: {exc}",
                )
            diff = reconstructed - target
            max_error = (
                float(np.nanmax(np.abs(diff)))
                if diff.size and not np.all(np.isnan(diff))
                else 0.0
            )
            passed = bool(
                np.allclose(
                    reconstructed,
                    target,
                    rtol=contract.tolerance,
                    atol=contract.tolerance,
                    equal_nan=True,
                )
            )
            return OracleResult(
                "AdditivityOracle",
                True,
                passed,
                "additivity relation satisfied" if passed else "additivity violation",
                observed=max_error,
                expected=contract.tolerance,
                details={
                    "model_output": contract.model_output,
                    "contribution_shape": list(contrib.shape),
                    "broadcast": broadcast_details,
                },
            )
        except Exception as exc:
            return OracleResult(
                "AdditivityOracle",
                False,
                None,
                f"oracle evaluation failed: {type(exc).__name__}: {exc}",
            )


def _feature_sum(values: np.ndarray, *, interaction=False, axes=None):
    return reduce_contributions(values, interaction=interaction, axes=axes)


class InputMutationOracle:
    """Detect value and structural mutations without erasing representation semantics."""

    def check(self, before, after):
        try:
            a, b = np.asarray(before), np.asarray(after)
            value_changed = (
                True
                if a.shape != b.shape
                else not bool(np.array_equal(a, b, equal_nan=True))
            )
            details = _mutation_dimensions(before, after)
            details["value_changed"] = value_changed
            details["object_identity_changed"] = id(before) != id(after)
            mutated = any(
                details[k]
                for k in (
                    "value_changed",
                    "shape_changed",
                    "dtype_changed",
                    "strides_changed",
                    "writeability_changed",
                )
            )
            return OracleResult(
                "InputMutationOracle",
                True,
                not mutated,
                "input preserved" if not mutated else "input mutated",
                observed=details,
                expected={"all_changed_flags": False},
                details=details,
            )
        except Exception as exc:
            return OracleResult(
                "InputMutationOracle",
                False,
                None,
                f"comparison failed: {type(exc).__name__}: {exc}",
            )


ORACLE_REGISTRY = {}


def register_oracle(name, oracle):
    ORACLE_REGISTRY[name] = oracle


class SHAPSemanticOracle:
    def __init__(self, registry=None):
        self.registry = ORACLE_REGISTRY if registry is None else registry

    def evaluate(
        self,
        *,
        contract,
        values,
        base_values=None,
        model_output=None,
        expected_value=None,
        interaction_values=None,
        target_shape=None,
        model=None,
        inputs=None,
        policy=None,
        target_source=None,
        independent_target=None,
        mutation_before=None,
        mutation_after=None,
    ):
        policy = policy or OraclePolicy(
            require_shape=contract.values_shape is not None,
            require_additivity=contract.additivity_required,
            require_output_space=contract.output_space_required,
            require_expected_value=contract.expected_value_required,
            require_interaction=contract.interaction,
            require_input_mutation=mutation_before is not None
            or mutation_after is not None,
        )
        req = {
            "ShapeOracle": (contract.values_shape is not None, policy.require_shape),
            "ExpectedValueOracle": (
                contract.expected_value_required or expected_value is not None,
                policy.require_expected_value,
            ),
            "InteractionOracle": (contract.interaction, policy.require_interaction),
            "AdditivityOracle": (
                contract.additivity_required,
                policy.require_additivity,
            ),
            "OutputSpaceOracle": (
                contract.output_space_required,
                policy.require_output_space,
            ),
            "InputMutationOracle": (
                (mutation_before is not None or mutation_after is not None),
                policy.require_input_mutation,
            ),
        }
        args = {
            "ShapeOracle": dict(values=values, expected_shape=contract.values_shape),
            "ExpectedValueOracle": dict(
                base_values=base_values,
                expected_value=expected_value,
                tolerance=contract.tolerance,
                semantics=contract.expected_value_semantics,
            ),
            "InteractionOracle": dict(
                values=values,
                interaction_values=interaction_values,
                axes=getattr(contract, "axis_spec", None),
            ),
            "AdditivityOracle": dict(
                values=values,
                base_values=base_values,
                model_output=model_output,
                contract=contract,
                required_override=True,
            ),
            "OutputSpaceOracle": dict(
                contract=contract,
                shap_values=values,
                base_values=base_values,
                model_output=model_output,
                model=model,
                inputs=inputs,
                target_source=target_source,
                independent_target=independent_target,
            ),
            "InputMutationOracle": dict(before=mutation_before, after=mutation_after),
        }
        results = []
        for name, (contract_required, policy_required) in req.items():
            if not (contract_required or policy_required):
                continue
            oracle = self.registry.get(name)
            source = (
                "policy+contract"
                if policy_required and contract_required
                else ("policy" if policy_required else "contract")
            )
            if oracle is None:
                r = OracleResult(
                    name,
                    False,
                    None,
                    "oracle not registered",
                    requirement_source=source,
                    contract_required=contract_required,
                    policy_required=policy_required,
                )
            else:
                r = oracle.check(**args[name])
                r = OracleResult(
                    **{
                        **r.to_dict(),
                        "requirement_source": source,
                        "contract_required": contract_required,
                        "policy_required": policy_required,
                    }
                )
            validate_oracle_result(r)
            results.append(r)
        failed = [
            r
            for r in results
            if r.passed is False and (r.policy_required or r.contract_required)
        ]
        unknown = [
            r
            for r in results
            if r.passed is None and (r.policy_required or r.contract_required)
        ]
        status = (
            "FAIL"
            if failed
            else (
                "INCONCLUSIVE" if unknown else ("PASS" if results else "INCONCLUSIVE")
            )
        )
        return {
            "status": status,
            "passed": status == "PASS",
            "policy": asdict(policy),
            "required_checks": [r.name for r in results],
            "results": [r.to_dict() for r in results],
            "registry": sorted(self.registry),
        }


register_oracle("ShapeOracle", ShapeOracle())
register_oracle("ExpectedValueOracle", ExpectedValueOracle())
register_oracle("InteractionOracle", InteractionOracle())
register_oracle("AdditivityOracle", AdditivityOracle())
register_oracle("OutputSpaceOracle", OutputSpaceOracle())
register_oracle("InputMutationOracle", InputMutationOracle())


def _semantic_align(
    left: np.ndarray, right: np.ndarray, *, role: str, axes: SHAPAxisSpec | None = None
):
    return semantic_align(left, right, axes=axes, role=role)


def _mutation_dimensions(before, after):
    import numpy as np

    out = {}
    for name, obj in (("before", before), ("after", after)):
        try:
            arr = np.asarray(obj)
            out[name] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "strides": list(arr.strides),
                "writeable": bool(arr.flags.writeable),
                "id": id(obj),
            }
        except Exception:
            out[name] = {"type": type(obj).__name__, "id": id(obj)}
    return {
        "shape_changed": out["before"].get("shape") != out["after"].get("shape"),
        "dtype_changed": out["before"].get("dtype") != out["after"].get("dtype"),
        "strides_changed": out["before"].get("strides") != out["after"].get("strides"),
        "writeability_changed": out["before"].get("writeable")
        != out["after"].get("writeable"),
        "payload_identity_changed": out["before"].get("id") != out["after"].get("id"),
        "buffer_identity_changed": _buffer_identity(before) != _buffer_identity(after),
    }


def _buffer_identity(obj):
    try:
        arr = np.asarray(obj)
        return id(arr.base) if arr.base is not None else id(arr)
    except Exception:
        return None
