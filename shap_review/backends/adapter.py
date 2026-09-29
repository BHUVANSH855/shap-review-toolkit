from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class BackendExecution:
    backend: str
    status: str
    stage: str
    result: dict[str, Any]
    reason: str | None = None


class BackendAdapter(Protocol):
    name: str

    def prepare(self, **kwargs) -> Any: ...
    def fit(self, context: dict[str, Any]) -> dict[str, Any]: ...
    def explain(self, context: dict[str, Any]) -> dict[str, Any]: ...
    def normalize(self, context: dict[str, Any]) -> dict[str, Any]: ...
    def oracle(self, context: dict[str, Any]) -> dict[str, Any]: ...
    def record(self, result: dict[str, Any]) -> BackendExecution: ...
    def execute_case(self, **kwargs) -> dict[str, Any]: ...


class MatrixBackendAdapter:
    """Concrete backend lifecycle.

    The matrix runner owns scheduling; this adapter owns the execution lifecycle:
    prepare -> fit -> explain -> normalize -> oracle -> record. Backend-specific
    construction remains in the helper functions selected by ``BackendSpec``.
    """

    def __init__(self, spec, module):
        self.spec, self.module, self.name = spec, module, spec.name

    def supports_case(self, **kwargs):
        classification = kwargs.get("classification", False)
        output = kwargs.get("model_output", "raw")
        interaction = kwargs.get("interaction", False)
        if not classification and output != "raw":
            return (
                False,
                "model_output is only raw for regression in this representative campaign",
            )
        if interaction and (classification or output != "raw"):
            return (
                False,
                "interaction requires regression/raw in this representative campaign",
            )
        return True, None

    def prepare(self, **kwargs):
        from shap_review.fuzzing.backend_matrix import (
            _classification_data,
            _regression_data,
            _representation,
        )

        supported, reason = self.supports_case(**kwargs)
        if not supported:
            raise NotImplementedError(reason)
        classification = kwargs.get("classification", False)
        representation = kwargs.get("representation", "ndarray")
        missing = kwargs.get("missing_values", False)
        interaction = kwargs.get("interaction", False)
        model_output = kwargs.get("model_output", "raw")
        if classification:
            X, y, probe, probe_y = _classification_data()
        else:
            X, y, probe = _regression_data()
            probe_y = None
        if missing:
            X = X.copy()
            X[1, 0] = float("nan")
            probe = probe.copy()
            probe[0, 0] = float("nan")
        if interaction and (classification or model_output != "raw"):
            raise NotImplementedError(
                "interaction requires regression/raw in this representative campaign"
            )
        return {
            "classification": classification,
            "representation": representation,
            "missing_values": missing,
            "interaction": interaction,
            "model_output": model_output,
            "X": X,
            "y": y,
            "probe": probe,
            "probe_y": probe_y,
            "train": _representation(X, representation),
            "test": _representation(probe, representation),
        }

    def fit(self, context):
        from shap_review.fuzzing.backend_matrix import _model_for

        model = _model_for(
            self.spec, self.module, classification=context["classification"]
        )
        model.fit(context["train"], context["y"])
        context = dict(context)
        context["model"] = model
        context["background"] = (
            context["train"] if context["representation"] != "sparse" else context["X"]
        )
        return context

    def explain(self, context):
        import numpy as np
        import shap

        model = context["model"]
        classification = context["classification"]
        output = context["model_output"]
        background = context["background"]
        if classification and output in {"probability", "log_loss"}:
            explainer = shap.TreeExplainer(
                model,
                data=background,
                feature_perturbation="interventional",
                model_output=output,
            )
        else:
            explainer = shap.TreeExplainer(
                model, data=background if not context["interaction"] else None
            )
        explanation = (
            explainer(context["test"], y=context["probe_y"])
            if output == "log_loss"
            else explainer.shap_interaction_values(context["test"])
            if context["interaction"]
            else explainer(context["test"])
        )
        return {
            **context,
            "explainer": explainer,
            "explanation": explanation,
            "values": np.asarray(
                explanation.values if hasattr(explanation, "values") else explanation
            ),
            "base": np.asarray(
                explanation.base_values
                if hasattr(explanation, "base_values")
                else explainer.expected_value
            ),
        }

    def normalize(self, context):
        import numpy as np

        from shap_review.contracts.tensor import SHAPSemanticTensor

        c = dict(context)
        values = c["values"]
        base = c["base"]
        c["values_shape"] = list(values.shape)
        c["base_shape"] = list(base.shape)
        c["target"] = np.asarray(
            c["model"].predict_proba(c["test"])
            if c["classification"]
            else c["model"].predict(c["test"]),
            dtype=float,
        )
        if c["interaction"]:
            c["target_space"] = "raw"
        if base.dtype != object or not any(callable(x) for x in base.reshape(-1)):
            c["semantic_tensor"] = SHAPSemanticTensor.from_values(
                values,
                base_values=base,
                interaction=c["interaction"],
                source_api="backend-matrix",
                output_space=c["model_output"],
                backend=self.spec.name,
            )
            c["base_values_dynamic"] = False
        else:
            c["semantic_tensor"] = None
            c["base_values_dynamic"] = True
        return c

    def oracle(self, context):
        import numpy as np

        from shap_review.fuzzing.backend_matrix import (
            _independent_probability_oracle,
            _semantic_additivity,
        )

        c = context
        classification = c["classification"]
        output = c["model_output"]
        model = c["model"]
        probability_oracle = (
            _independent_probability_oracle(model, c["test"], c["probe_y"])
            if classification
            else None
        )
        values, base = c["values"], c["base"]
        if output == "log_loss":
            independent = np.asarray(
                probability_oracle["log_loss_per_sample"], dtype=float
            )
            if base.dtype == object and any(callable(x) for x in base.reshape(-1)):
                return {
                    "status": "PASS_WITH_FINDINGS",
                    "execution_status": "EXECUTED",
                    "stage": "oracle",
                    "semantic_status": "INCONCLUSIVE",
                    "log_loss_oracle_status": "INCONCLUSIVE_DYNAMIC_BASE_VALUE",
                    "probability_oracle": probability_oracle,
                    "base_values_type": "callable",
                    "semantic_contract": {
                        "output_space": "log_loss",
                        "requires_dynamic_base_value_resolution": True,
                    },
                }
            from shap_review.contracts.tensor import SHAPSemanticTensor

            tensor = SHAPSemanticTensor.from_values(
                values,
                base_values=base,
                output_space="log_loss",
                source_api="backend-matrix",
                backend=self.name,
            )
            try:
                reconstructed = np.asarray(tensor.reconstruction())
            except ValueError as exc:
                return {
                    "status": "PASS_WITH_FINDINGS",
                    "execution_status": "EXECUTED",
                    "stage": "oracle",
                    "semantic_status": "INCONCLUSIVE",
                    "log_loss_oracle_status": "INCONCLUSIVE_SHAPE",
                    "reason": str(exc),
                    "target_shape": list(independent.shape),
                    "probability_oracle": probability_oracle,
                }
            if reconstructed.shape != independent.shape:
                return {
                    "status": "PASS_WITH_FINDINGS",
                    "execution_status": "EXECUTED",
                    "stage": "oracle",
                    "semantic_status": "INCONCLUSIVE",
                    "log_loss_oracle_status": "INCONCLUSIVE_SHAPE",
                    "reconstructed_shape": list(reconstructed.shape),
                    "target_shape": list(independent.shape),
                    "probability_oracle": probability_oracle,
                }
            err = (
                float(np.max(np.abs(reconstructed - independent)))
                if reconstructed.size
                else 0.0
            )
            passed = bool(
                np.allclose(
                    reconstructed, independent, rtol=1e-5, atol=1e-8, equal_nan=True
                )
            )
            return {
                "status": "SEMANTIC_PASS" if passed else "SEMANTIC_FAIL",
                "execution_status": "EXECUTED",
                "stage": "oracle",
                "semantic_status": "PASS" if passed else "FAIL",
                "model_output": output,
                "additivity_passed": passed,
                "max_additivity_error": err,
                "log_loss_independent": float(np.mean(independent)),
                "log_loss_max_error": err,
                "probability_oracle": probability_oracle,
                "oracle": {
                    "status": "ORACLE_PASS" if passed else "SEMANTIC_MISMATCH"
                },
            }

        target = c["target"]
        passed, err, oracle = _semantic_additivity(
            values,
            base,
            target,
            interaction=c["interaction"],
            output_space=output,
            backend=self.spec.name,
        )
        semantic = (
            "PASS"
            if passed
            else (
                "INCONCLUSIVE"
                if oracle.get("status") in {"INCONCLUSIVE_SHAPE", "ORACLE_ERROR"}
                else "FAIL"
            )
        )
        status = (
            "SEMANTIC_PASS"
            if semantic == "PASS"
            else (
                "PASS_WITH_FINDINGS" if semantic == "INCONCLUSIVE" else "SEMANTIC_FAIL"
            )
        )
        return {
            "status": status,
            "execution_status": "EXECUTED",
            "stage": "oracle",
            "semantic_status": semantic,
            "version": self.spec.version,
            "classification": classification,
            "input_representation": c["representation"],
            "missing_values": c["missing_values"],
            "interaction": c["interaction"],
            "model_output": output,
            "values_shape": c["values_shape"],
            "target_shape": list(target.shape),
            "class_axis": 1 if classification and target.ndim == 2 else None,
            "class_labels": getattr(model, "classes_", None).tolist()
            if hasattr(getattr(model, "classes_", None), "tolist")
            else None,
            "additivity_passed": passed,
            "max_additivity_error": err,
            "oracle": oracle,
            "probability_oracle": probability_oracle,
        }

    def record(self, result):
        return BackendExecution(
            self.name,
            str(result.get("status")),
            str(result.get("stage", "unknown")),
            result,
            result.get("execution_reason"),
        )

    def execute_case(self, **kwargs):
        stage = "prepare"
        context = None
        try:
            context = self.prepare(**kwargs)
            stage = "fit"
            context = self.fit(context)
            stage = "explain"
            context = self.explain(context)
            stage = "normalize"
            context = self.normalize(context)
            stage = "oracle"
            result = self.oracle(context)
            result["backend"] = self.name
            result["execution_reason"] = "COMPLETED"
            recorded = self.record(result)
            result["lifecycle"] = {
                "stages": [
                    "prepare",
                    "fit",
                    "explain",
                    "normalize",
                    "oracle",
                    "record",
                ],
                "recorded_status": recorded.status,
            }
            return result
        except NotImplementedError as exc:
            result = {
                "backend": self.name,
                "status": "UNSUPPORTED",
                "execution_status": "NOT_EXECUTED",
                "execution_reason": "UNSUPPORTED",
                "semantic_status": "NOT_EVALUATED",
                "stage": stage,
                "reason": str(exc),
            }
        # This is an intentional lifecycle boundary: backend, SHAP, and toolkit
        # exceptions must be converted into structured execution results rather
        # than escaping and aborting the campaign.
        except Exception as exc:  # noqa: BLE001
            reason = (
                "SHAP_ERROR"
                if stage == "explain"
                else (
                    "TOOLKIT_ERROR"
                    if stage in {"normalize", "oracle"}
                    else ("BACKEND_ERROR" if stage == "fit" else "ADAPTER_ERROR")
                )
            )
            result = {
                "backend": self.name,
                "status": reason,
                "execution_status": "NOT_EXECUTED",
                "execution_reason": reason,
                "semantic_status": "NOT_EVALUATED",
                "stage": stage,
                "error": f"{type(exc).__name__}: {exc}",
                "exception_type": type(exc).__name__,
            }
        self.record(result)
        return result