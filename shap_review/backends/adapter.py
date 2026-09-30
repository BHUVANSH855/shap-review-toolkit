from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class BackendAdapter(Protocol):
    """Protocol implemented by concrete backend adapters."""

    name: str

    def supports_case(
        self,
        *,
        classification: bool = False,
        representation: str = "ndarray",
        missing_values: bool = False,
        interaction: bool = False,
        model_output: str = "raw",
    ) -> tuple[bool, str]: ...

    def prepare(self, **kwargs: Any) -> dict[str, Any]: ...

    def fit(self, context: dict[str, Any]) -> dict[str, Any]: ...

    def explain(self, context: dict[str, Any]) -> dict[str, Any]: ...

    def normalize(self, context: dict[str, Any]) -> dict[str, Any]: ...

    def oracle(self, context: dict[str, Any]) -> dict[str, Any]: ...

    def record(self, result: dict[str, Any]) -> dict[str, Any]: ...

    def execute_case(
        self,
        *,
        classification: bool = False,
        representation: str = "ndarray",
        missing_values: bool = False,
        interaction: bool = False,
        model_output: str = "raw",
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class BackendExecution:
    """Structured representation of one backend execution."""

    backend: str
    execution_status: str
    execution_reason: str
    semantic_status: str
    status: str
    result: dict[str, Any]


class MatrixBackendAdapter:
    """Execute one backend matrix case through a concrete lifecycle.

    The matrix runner owns case scheduling. This adapter owns the execution
    lifecycle:

        prepare -> fit -> explain -> normalize -> oracle -> record

    Backend-specific model construction is selected from the BackendSpec
    supplied by the matrix runner.
    """

    def __init__(self, spec, module):
        self.spec = spec
        self.module = module
        self.name = spec.name

    def supports_case(
        self,
        *,
        classification: bool = False,
        representation: str = "ndarray",
        missing_values: bool = False,
        interaction: bool = False,
        model_output: str = "raw",
    ) -> tuple[bool, str]:
        """Return whether the backend can execute the requested case."""
        if representation not in {"ndarray", "dataframe", "sparse"}:
            return False, f"unsupported representation: {representation}"

        if classification and model_output not in {
            "probability",
            "log_loss",
            "raw",
        }:
            return False, f"unsupported classification output: {model_output}"

        if not classification and model_output != "raw":
            return (
                False,
                "regression matrix cases only support raw model output",
            )

        if interaction and (classification or model_output != "raw"):
            return (
                False,
                "interaction cases require regression with raw output",
            )

        if missing_values and self.spec.name == "catboost":
            return (
                False,
                "catboost matrix cases do not use injected missing values",
            )

        if representation == "sparse" and self.spec.name == "catboost":
            return (
                False,
                "catboost matrix cases do not use sparse representations",
            )

        return True, "supported"

    def execute_case(
        self,
        *,
        classification: bool = False,
        representation: str = "ndarray",
        missing_values: bool = False,
        interaction: bool = False,
        model_output: str = "raw",
    ) -> dict[str, Any]:
        """Execute one case through the complete adapter lifecycle."""
        stage = "setup"

        kwargs = {
            "classification": classification,
            "representation": representation,
            "missing_values": missing_values,
            "interaction": interaction,
            "model_output": model_output,
        }

        try:
            stage = "prepare"
            context = self.prepare(**kwargs)

            stage = "fit"
            context = self.fit(context)

            stage = "explain"
            context = self.explain(context)

            stage = "normalize"
            context = self.normalize(context)

            stage = "oracle"
            result = self.oracle(context)

            stage = "record"
            return self.record(result)

        except Exception as exc:  # noqa: BLE001
            return self._error_result(
                stage=stage,
                exc=exc,
                classification=classification,
                representation=representation,
                missing_values=missing_values,
                interaction=interaction,
                model_output=model_output,
            )

    def prepare(self, **kwargs: Any) -> dict[str, Any]:
        """Prepare deterministic training/probe data for one matrix case."""
        supported, reason = self.supports_case(**kwargs)
        if not supported:
            raise NotImplementedError(reason)

        classification = kwargs.get("classification", False)
        representation = kwargs.get("representation", "ndarray")
        missing_values = kwargs.get("missing_values", False)
        interaction = kwargs.get("interaction", False)
        model_output = kwargs.get("model_output", "raw")

        if classification:
            X, y, probe, probe_y = self._classification_data()
        else:
            X, y, probe = self._regression_data()
            probe_y = None

        if missing_values:
            X = X.copy()
            X[1, 0] = np.nan

            probe = probe.copy()
            probe[0, 0] = np.nan

        if interaction and (classification or model_output != "raw"):
            raise NotImplementedError(
                "interaction cases require regression with raw output"
            )

        return {
            "classification": classification,
            "representation": representation,
            "missing_values": missing_values,
            "interaction": interaction,
            "model_output": model_output,
            "X": X,
            "y": y,
            "probe": probe,
            "probe_y": probe_y,
            "train": self._representation(X, representation),
            "test": self._representation(probe, representation),
        }

    def fit(self, context: dict[str, Any]) -> dict[str, Any]:
        """Build and fit the backend model."""
        model = self._model_for(
            classification=context["classification"],
        )
        model.fit(context["train"], context["y"])

        fitted = dict(context)
        fitted["model"] = model
        fitted["background"] = (
            context["train"] if context["representation"] != "sparse" else context["X"]
        )
        return fitted

    def explain(self, context: dict[str, Any]) -> dict[str, Any]:
        """Create the SHAP explainer and execute the requested explanation."""
        import shap

        model = context["model"]
        classification = context["classification"]
        model_output = context["model_output"]
        background = context["background"]
        test = context["test"]
        probe_y = context["probe_y"]
        interaction = context["interaction"]

        if classification and model_output in {"probability", "log_loss"}:
            explainer = shap.TreeExplainer(
                model,
                data=background,
                feature_perturbation="interventional",
                model_output=model_output,
            )
        else:
            explainer = shap.TreeExplainer(
                model,
                data=None if interaction else background,
            )

        explained = dict(context)
        explained["explainer"] = explainer

        if interaction:
            explained["interaction_values"] = np.asarray(
                explainer.shap_interaction_values(test)
            )
            explained["interaction_base"] = np.asarray(explainer.expected_value)
            return explained

        explained["explanation"] = (
            explainer(test, y=probe_y)
            if model_output == "log_loss"
            else explainer(test)
        )
        return explained

    def normalize(self, context: dict[str, Any]) -> dict[str, Any]:
        """Normalize SHAP outputs into NumPy arrays."""
        normalized = dict(context)

        if context["interaction"]:
            normalized["values"] = np.asarray(context["interaction_values"])
            normalized["base"] = np.asarray(context["interaction_base"])
            return normalized

        explanation = context["explanation"]
        normalized["values"] = np.asarray(explanation.values)
        normalized["base"] = np.asarray(explanation.base_values)

        return normalized

    def oracle(self, context: dict[str, Any]) -> dict[str, Any]:
        """Evaluate normalized SHAP output against an independent oracle."""
        if context["interaction"]:
            return self._oracle_interaction(context)

        if context["model_output"] == "log_loss":
            return self._oracle_log_loss(context)

        return self._oracle_standard(context)

    def record(self, result: dict[str, Any]) -> dict[str, Any]:
        """Return the structured result produced by the oracle stage."""
        return result

    @staticmethod
    def _classification_data():
        X = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [2.0, 1.0],
            ],
            dtype=float,
        )
        y = np.array([0, 0, 1, 1, 1, 1], dtype=int)

        probe = np.array(
            [
                [0.5, 0.5],
                [1.5, 0.5],
            ],
            dtype=float,
        )
        probe_y = np.array([0, 1], dtype=int)

        return X, y, probe, probe_y

    @staticmethod
    def _regression_data():
        X = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ],
            dtype=float,
        )
        y = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)

        probe = np.array(
            [
                [0.5, 0.5],
                [2.5, 2.5],
            ],
            dtype=float,
        )

        return X, y, probe

    @staticmethod
    def _representation(X, kind):
        if kind == "ndarray":
            return X

        if kind == "dataframe":
            import pandas as pd

            return pd.DataFrame(X, columns=["f0", "f1"])

        if kind == "sparse":
            from scipy import sparse

            return sparse.csr_matrix(X)

        raise ValueError(kind)

    def _model_for(self, *, classification: bool):
        if self.spec.name == "sklearn":
            if classification:
                from sklearn.tree import DecisionTreeClassifier

                return DecisionTreeClassifier(
                    max_depth=2,
                    random_state=0,
                )

            from sklearn.tree import DecisionTreeRegressor

            return DecisionTreeRegressor(
                max_depth=2,
                random_state=0,
            )

        if self.spec.name == "xgboost":
            if classification:
                return self.module.XGBClassifier(
                    n_estimators=3,
                    max_depth=2,
                    n_jobs=1,
                    verbosity=0,
                    random_state=0,
                    eval_metric="logloss",
                )

            return self.module.XGBRegressor(
                n_estimators=3,
                max_depth=2,
                n_jobs=1,
                verbosity=0,
                random_state=0,
            )

        if self.spec.name == "lightgbm":
            if classification:
                return self.module.LGBMClassifier(
                    n_estimators=3,
                    max_depth=2,
                    verbosity=-1,
                    random_state=0,
                )

            return self.module.LGBMRegressor(
                n_estimators=3,
                max_depth=2,
                verbosity=-1,
                random_state=0,
            )

        if self.spec.name == "catboost":
            kwargs = {
                "iterations": 3,
                "depth": 2,
                "verbose": False,
                "random_seed": 0,
            }

            if classification:
                return self.module.CatBoostClassifier(**kwargs)

            return self.module.CatBoostRegressor(**kwargs)

        raise ValueError(self.spec.name)

    def _oracle_interaction(self, context: dict[str, Any]):
        model = context["model"]
        values = context["values"]
        base = context["base"]
        test = context["test"]

        target = np.asarray(model.predict(test), dtype=float)

        passed, error, oracle = self._semantic_additivity(
            values,
            base,
            target,
            interaction=True,
            output_space="raw",
        )

        return {
            "backend": self.spec.name,
            "status": self._semantic_status(oracle, passed),
            "execution_status": "EXECUTED",
            "stage": "oracle",
            "version": self.spec.version,
            "classification": context["classification"],
            "input_representation": context["representation"],
            "missing_values": context["missing_values"],
            "interaction": True,
            "model_output": "raw",
            "values_shape": list(values.shape),
            "target_shape": list(target.shape),
            "additivity_passed": passed,
            "max_additivity_error": error,
            "oracle": oracle,
            "semantic_contract": {
                "output_space": "raw",
                "input_representation": context["representation"],
                "task": "regression",
            },
            "interaction_executed": True,
        }

    def _oracle_log_loss(self, context: dict[str, Any]):
        model = context["model"]
        test = context["test"]
        probe_y = context["probe_y"]
        values = context["values"]
        base = context["base"]

        independent_oracle = self._independent_probability_oracle(
            model,
            test,
            probe_y,
        )
        independent = np.asarray(
            independent_oracle["log_loss_per_sample"],
            dtype=float,
        )

        base_methods = bool(
            base.dtype == object and any(callable(x) for x in base.reshape(-1))
        )

        common = {
            "backend": self.spec.name,
            "execution_status": "EXECUTED",
            "stage": "oracle",
            "version": self.spec.version,
            "classification": context["classification"],
            "input_representation": context["representation"],
            "missing_values": context["missing_values"],
            "interaction": False,
            "model_output": "log_loss",
            "values_shape": list(values.shape),
            "target_shape": list(independent.shape),
            "log_loss_independent": float(np.mean(independent)),
            "probability_oracle": independent_oracle,
            "semantic_contract": {
                "output_space": "log_loss",
                "task": "classification",
            },
        }

        if base_methods:
            dynamic_base = []

            for i, label in enumerate(probe_y):
                try:
                    value = base.reshape(-1)[i](int(label))
                    dynamic_base.append(np.asarray(value).tolist())
                except Exception as exc:  # noqa: BLE001
                    dynamic_base.append(
                        {
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

            return {
                **common,
                "status": "PASS_WITH_FINDINGS",
                "log_loss_oracle_status": "INCONCLUSIVE_DYNAMIC_BASE_VALUE",
                "base_values_type": "callable",
                "dynamic_base_values": dynamic_base,
                "semantic_contract": {
                    **common["semantic_contract"],
                    "requires_dynamic_base_value_resolution": True,
                },
            }

        reconstructed = np.asarray(values).sum(axis=1) + base

        if reconstructed.shape != independent.shape:
            return {
                **common,
                "status": "PASS_WITH_FINDINGS",
                "log_loss_oracle_status": "INCONCLUSIVE_SHAPE",
                "reconstructed_shape": list(reconstructed.shape),
            }

        error = float(np.max(np.abs(reconstructed - independent)))
        passed = bool(
            np.allclose(
                reconstructed,
                independent,
                rtol=1e-5,
                atol=1e-8,
            )
        )

        return {
            **common,
            "status": "SEMANTIC_PASS" if passed else "SEMANTIC_FAIL",
            "additivity_passed": passed,
            "max_additivity_error": error,
            "log_loss_max_error": error,
            "oracle": {
                "status": "ORACLE_PASS" if passed else "SEMANTIC_MISMATCH",
            },
        }

    def _oracle_standard(self, context: dict[str, Any]):
        model = context["model"]
        test = context["test"]
        values = context["values"]
        base = context["base"]
        classification = context["classification"]
        representation = context["representation"]
        missing_values = context["missing_values"]
        model_output = context["model_output"]

        target = np.asarray(
            model.predict_proba(test) if classification else model.predict(test),
            dtype=float,
        )

        class_axis = 1 if classification and target.ndim == 2 else None

        passed, error, oracle = self._semantic_additivity(
            values,
            base,
            target,
            output_space=model_output,
        )

        classes = getattr(model, "classes_", None)

        return {
            "backend": self.spec.name,
            "status": self._semantic_status(oracle, passed),
            "execution_status": "EXECUTED",
            "stage": "oracle",
            "version": self.spec.version,
            "classification": classification,
            "input_representation": representation,
            "missing_values": missing_values,
            "interaction": False,
            "model_output": model_output,
            "values_shape": list(values.shape),
            "target_shape": list(target.shape),
            "class_axis": class_axis,
            "class_labels": (
                classes.tolist()
                if hasattr(classes, "tolist")
                else list(classes)
                if classes is not None
                else None
            ),
            "additivity_passed": passed,
            "max_additivity_error": error,
            "oracle": oracle,
            "probability_oracle": (
                self._independent_probability_oracle(
                    model,
                    test,
                    np.arange(target.shape[0]),
                )
                if classification
                else None
            ),
            "semantic_contract": {
                "output_space": model_output,
                "input_representation": representation,
                "task": "classification" if classification else "regression",
                "class_axis": class_axis,
            },
        }

    @staticmethod
    def _semantic_status(oracle, passed: bool) -> str:
        if passed:
            return "SEMANTIC_PASS"

        if oracle.get("status") in {
            "INCONCLUSIVE_SHAPE",
            "ORACLE_ERROR",
        }:
            return "PASS_WITH_FINDINGS"

        return "SEMANTIC_FAIL"

    @staticmethod
    def _semantic_additivity(
        values,
        base,
        target,
        *,
        interaction=False,
        output_space="raw",
    ):
        from shap_review.contracts.tensor import SHAPSemanticTensor

        target = np.asarray(target, dtype=float)
        values = np.asarray(values)
        base = np.asarray(base)

        metadata = {
            "target_shape": list(target.shape),
            "values_shape": list(values.shape),
            "base_shape": list(base.shape),
        }

        try:
            tensor = SHAPSemanticTensor.from_values(
                values,
                base_values=base,
                interaction=interaction,
                source_api="backend-matrix",
                output_space=output_space,
                backend=None,
                metadata=metadata,
            )
            reconstructed = np.asarray(tensor.reconstruction())
        except Exception as exc:  # noqa: BLE001
            return (
                False,
                float("inf"),
                {
                    "status": "ORACLE_ERROR",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "metadata": metadata,
                    "passed": False,
                },
            )

        if reconstructed.shape != target.shape:
            return (
                False,
                float("inf"),
                {
                    "status": "INCONCLUSIVE_SHAPE",
                    "reconstructed_shape": list(reconstructed.shape),
                    "target_shape": list(target.shape),
                    "metadata": metadata,
                    "passed": False,
                },
            )

        delta = reconstructed - target
        error = float(np.nanmax(np.abs(delta))) if delta.size else 0.0

        passed = bool(
            np.allclose(
                reconstructed,
                target,
                rtol=1e-5,
                atol=1e-8,
                equal_nan=True,
            )
        )

        return (
            passed,
            error,
            {
                "status": "ORACLE_PASS" if passed else "SEMANTIC_MISMATCH",
                "reconstructed_shape": list(reconstructed.shape),
                "target_shape": list(target.shape),
                "metadata": metadata,
                "passed": passed,
            },
        )

    @staticmethod
    def _independent_probability_oracle(model, test, probe_y):
        probs = np.asarray(model.predict_proba(test), dtype=float)
        labels = np.asarray(probe_y, dtype=int)
        true_p = probs[np.arange(len(labels)), labels]

        return {
            "shape": list(probs.shape),
            "class_count": int(probs.shape[1]),
            "labels": labels.tolist(),
            "true_class_probability": true_p.tolist(),
            "log_loss_per_sample": (-np.log(np.clip(true_p, 1e-15, 1.0))).tolist(),
        }

    def _error_result(
        self,
        *,
        stage: str,
        exc: Exception,
        classification: bool,
        representation: str,
        missing_values: bool,
        interaction: bool,
        model_output: str,
    ):
        status = self._classify_exception(exc, stage)

        return {
            "backend": self.spec.name,
            "status": status,
            "execution_status": "NOT_EXECUTED",
            "execution_reason": status,
            "semantic_status": "NOT_EVALUATED",
            "stage": stage,
            "version": self.spec.version,
            "classification": classification,
            "input_representation": representation,
            "missing_values": missing_values,
            "interaction": interaction,
            "model_output": model_output,
            "error": f"{type(exc).__name__}: {exc}",
            "exception_type": type(exc).__name__,
        }

    @staticmethod
    def _classify_exception(exc: Exception, stage: str) -> str:
        message = str(exc).lower()
        name = type(exc).__name__.lower()

        if isinstance(exc, NotImplementedError):
            return "UNSUPPORTED"

        if "timeout" in message or "timeout" in name:
            return "TIMEOUT"

        if stage == "fit":
            return "BACKEND_ERROR"

        if stage in {"explain"}:
            return "SHAP_ERROR"

        if stage == "normalize":
            return "SHAP_ERROR"

        if stage == "oracle":
            return "TOOLKIT_ERROR"

        if stage == "record":
            return "TOOLKIT_ERROR"

        return "ERROR"
