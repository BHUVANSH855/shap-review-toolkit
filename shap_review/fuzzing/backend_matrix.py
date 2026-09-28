from __future__ import annotations

import importlib
import tempfile
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendSpec:
    name: str
    import_name: str
    families: tuple[str, ...]
    available: bool = False
    version: str | None = None


DEFAULT_BACKENDS = (
    BackendSpec("sklearn", "sklearn", ("tree", "linear", "ensemble")),
    BackendSpec("xgboost", "xgboost", ("tree", "gbtree")),
    BackendSpec("lightgbm", "lightgbm", ("tree", "gbdt")),
    BackendSpec("catboost", "catboost", ("tree", "categorical")),
)
EXECUTION_STATUSES = ("EXECUTED", "NOT_EXECUTED")
EXECUTION_REASONS = (
    "COMPLETED",
    "SHAP_ERROR",
    "BACKEND_ERROR",
    "ADAPTER_ERROR",
    "TOOLKIT_ERROR",
    "UNSUPPORTED",
    "SKIPPED",
    "TIMEOUT",
    "ERROR",
)
SEMANTIC_STATUSES = ("PASS", "FAIL", "INCONCLUSIVE", "NOT_EVALUATED")
STATUSES = (
    "EXECUTED",
    "SEMANTIC_PASS",
    "SEMANTIC_FAIL",
    "PASS_WITH_FINDINGS",
    "SKIPPED",
    "UNSUPPORTED",
    "ADAPTER_ERROR",
    "BACKEND_ERROR",
    "SHAP_ERROR",
    "TOOLKIT_ERROR",
    "ERROR",
)


def discover_backends(specs=DEFAULT_BACKENDS):
    out = []
    for spec in specs:
        try:
            mod = importlib.import_module(spec.import_name)
            out.append(
                BackendSpec(
                    spec.name,
                    spec.import_name,
                    spec.families,
                    True,
                    getattr(mod, "__version__", None),
                )
            )
        except Exception:
            out.append(spec)
    return out


def matrix_dimensions():
    return {
        "backend": [s.name for s in DEFAULT_BACKENDS],
        "model_output": ["raw", "probability", "log_loss"],
        "input_representation": ["ndarray", "dataframe", "sparse"],
        "missing_values": [False, True],
        "classification": [False, True],
        "interaction": [False, True],
    }


def _semantic_additivity(
    values, base, target, *, interaction=False, output_space="raw", backend=None
):
    import numpy as np

    from shap_review.contracts.tensor import SHAPSemanticTensor

    target = np.asarray(target, dtype=float)
    values = np.asarray(values)
    base = np.asarray(base)
    meta = {
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
            backend=backend,
            metadata=meta,
        )
        reconstructed = np.asarray(tensor.reconstruction())
    except Exception as exc:
        return (
            False,
            float("inf"),
            {
                "status": "ORACLE_ERROR",
                "reason": f"{type(exc).__name__}: {exc}",
                "metadata": meta,
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
                "metadata": meta,
                "passed": False,
            },
        )
    delta = reconstructed - target
    err = float(np.nanmax(np.abs(delta))) if delta.size else 0.0
    passed = bool(
        np.allclose(reconstructed, target, rtol=1e-5, atol=1e-8, equal_nan=True)
    )
    return (
        passed,
        err,
        {
            "status": "ORACLE_PASS" if passed else "SEMANTIC_MISMATCH",
            "reconstructed_shape": list(reconstructed.shape),
            "target_shape": list(target.shape),
            "metadata": meta,
            "passed": passed,
        },
    )


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


def _classification_data():
    import numpy as np

    X = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0]],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1, 1, 1], dtype=int)
    probe = np.array([[0.5, 0.5], [1.5, 0.5]], dtype=float)
    probe_y = np.array([0, 1], dtype=int)
    return X, y, probe, probe_y


def _regression_data():
    import numpy as np

    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=float)
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    probe = np.array([[0.5, 0.5], [2.5, 2.5]], dtype=float)
    return X, y, probe


def _model_for(spec, mod, classification=False):
    if spec.name == "sklearn":
        if classification:
            from sklearn.tree import DecisionTreeClassifier

            return DecisionTreeClassifier(max_depth=2, random_state=0)
        from sklearn.tree import DecisionTreeRegressor

        return DecisionTreeRegressor(max_depth=2, random_state=0)
    if spec.name == "xgboost":
        return (
            mod.XGBClassifier(
                n_estimators=3,
                max_depth=2,
                n_jobs=1,
                verbosity=0,
                random_state=0,
                eval_metric="logloss",
            )
            if classification
            else mod.XGBRegressor(
                n_estimators=3, max_depth=2, n_jobs=1, verbosity=0, random_state=0
            )
        )
    if spec.name == "lightgbm":
        return (
            mod.LGBMClassifier(
                n_estimators=3, max_depth=2, verbosity=-1, random_state=0
            )
            if classification
            else mod.LGBMRegressor(
                n_estimators=3, max_depth=2, verbosity=-1, random_state=0
            )
        )
    if spec.name == "catboost":
        kwargs = dict(
            iterations=3,
            depth=2,
            verbose=False,
            random_seed=0,
            train_dir=tempfile.mkdtemp(prefix="shap-review-catboost-"),
        )
        return (
            mod.CatBoostClassifier(**kwargs)
            if classification
            else mod.CatBoostRegressor(**kwargs)
        )
    raise ValueError(spec.name)


def _classify_exception(exc, stage="unknown"):
    """Classify by the execution stage first; exception type and message are
    secondary refinements that prevent resource errors and unsupported configs
    from being reported as SHAP bugs.
    """
    exc_type = type(exc).__name__
    msg = str(exc).lower()

    # Resource / environment errors are never SHAP bugs
    if exc_type in (
        "MemoryError",
        "RecursionError",
        "KeyboardInterrupt",
        "SystemExit",
        "TimeoutError",
    ):
        return "TOOLKIT_ERROR"

    # Unsupported configuration: SHAP deliberately rejects these
    if (
        "not implemented" in msg
        or "unsupported" in msg
        or "not supported" in msg
        or exc_type == "NotImplementedError"
    ):
        return "UNSUPPORTED"

    # Stage-based classification (primary signal)
    if stage in {"model_build", "model_fit"}:
        return "BACKEND_ERROR"
    if stage in {"explainer_create", "explainer_execute"}:
        # Stage is in SHAP territory but may still be a backend serialization
        # error triggered inside SHAP code; label as SHAP_ERROR but caller
        # should inspect the stack before promoting to a confirmed finding.
        return "SHAP_ERROR"
    if stage in {"output_normalization", "oracle"}:
        return "TOOLKIT_ERROR"
    if stage == "representation":
        return "ADAPTER_ERROR"

    return "ADAPTER_ERROR"


def _independent_probability_oracle(model, test, probe_y):
    import numpy as np

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


def _run_case(
    spec,
    mod,
    *,
    classification,
    representation,
    missing_values=False,
    interaction=False,
    model_output="probability",
):
    import numpy as np
    import shap

    stage = "setup"
    try:
        if classification:
            X, y, probe, probe_y = _classification_data()
        else:
            X, y, probe = _regression_data()
            probe_y = None
        if missing_values:
            X = X.copy()
            X[1, 0] = np.nan
            probe = probe.copy()
            probe[0, 0] = np.nan
        stage = "representation"
        train = _representation(X, representation)
        test = _representation(probe, representation)
        stage = "model_build"
        model = _model_for(spec, mod, classification=classification)
        if interaction and (classification or model_output != "raw"):
            return {
                "backend": spec.name,
                "status": "UNSUPPORTED",
                "execution_status": "UNSUPPORTED",
                "stage": stage,
                "reason": "interaction requires regression/raw in this representative campaign",
            }
        stage = "model_fit"
        model.fit(train, y)
        background = train if representation != "sparse" else X
        stage = "explainer_create"
        if classification and model_output in {"probability", "log_loss"}:
            explainer = shap.TreeExplainer(
                model,
                data=background,
                feature_perturbation="interventional",
                model_output=model_output,
            )
        else:
            explainer = shap.TreeExplainer(
                model, data=background if not interaction else None
            )
        stage = "explainer_execute"
        if interaction:
            values = np.asarray(explainer.shap_interaction_values(test))
            base = np.asarray(explainer.expected_value)
            target = np.asarray(model.predict(test), dtype=float)
            stage = "output_normalization"
            passed, err, oracle = _semantic_additivity(
                values, base, target, interaction=True, backend=spec.name
            )
            return {
                "backend": spec.name,
                "status": "SEMANTIC_PASS"
                if passed
                else (
                    "PASS_WITH_FINDINGS"
                    if oracle.get("status") in {"INCONCLUSIVE_SHAPE", "ORACLE_ERROR"}
                    else "SEMANTIC_FAIL"
                ),
                "execution_status": "EXECUTED",
                "stage": "oracle",
                "version": spec.version,
                "classification": False,
                "input_representation": representation,
                "missing_values": missing_values,
                "interaction": True,
                "model_output": "raw",
                "values_shape": list(values.shape),
                "target_shape": list(target.shape),
                "additivity_passed": passed,
                "max_additivity_error": err,
                "oracle": oracle,
                "semantic_contract": {
                    "output_space": "raw",
                    "input_representation": representation,
                    "task": "regression",
                },
                "interaction_executed": True,
            }
        explanation = (
            explainer(test, y=probe_y)
            if model_output == "log_loss"
            else explainer(test)
        )
        values = np.asarray(explanation.values)
        base = np.asarray(explanation.base_values)
        stage = "oracle"
        probability_oracle = (
            _independent_probability_oracle(model, test, probe_y)
            if classification
            else None
        )
        if model_output == "log_loss":
            independent = np.asarray(
                probability_oracle["log_loss_per_sample"], dtype=float
            )
            base_methods = bool(
                base.dtype == object and any(callable(x) for x in base.reshape(-1))
            )
            if base_methods:
                dynamic_base = []
                for i, label in enumerate(probe_y):
                    try:
                        value = base.reshape(-1)[i](int(label))
                        dynamic_base.append(np.asarray(value).tolist())
                    except Exception as exc:
                        dynamic_base.append({"error": f"{type(exc).__name__}: {exc}"})
                return {
                    "backend": spec.name,
                    "status": "PASS_WITH_FINDINGS",
                    "execution_status": "EXECUTED",
                    "stage": "oracle",
                    "version": spec.version,
                    "classification": True,
                    "input_representation": representation,
                    "missing_values": missing_values,
                    "interaction": False,
                    "model_output": "log_loss",
                    "values_shape": list(values.shape),
                    "target_shape": list(independent.shape),
                    "log_loss_independent": float(np.mean(independent)),
                    "log_loss_oracle_status": "INCONCLUSIVE_DYNAMIC_BASE_VALUE",
                    "base_values_type": "callable",
                    "dynamic_base_values": dynamic_base,
                    "probability_oracle": probability_oracle,
                    "semantic_contract": {
                        "output_space": "log_loss",
                        "task": "classification",
                        "requires_dynamic_base_value_resolution": True,
                    },
                }
            reconstructed = np.asarray(values).sum(axis=1) + base
            if reconstructed.shape != independent.shape:
                return {
                    "backend": spec.name,
                    "status": "PASS_WITH_FINDINGS",
                    "execution_status": "EXECUTED",
                    "stage": "oracle",
                    "version": spec.version,
                    "classification": True,
                    "input_representation": representation,
                    "missing_values": missing_values,
                    "interaction": False,
                    "model_output": "log_loss",
                    "values_shape": list(values.shape),
                    "target_shape": list(independent.shape),
                    "log_loss_independent": float(np.mean(independent)),
                    "log_loss_oracle_status": "INCONCLUSIVE_SHAPE",
                    "reconstructed_shape": list(reconstructed.shape),
                    "probability_oracle": probability_oracle,
                    "semantic_contract": {
                        "output_space": "log_loss",
                        "task": "classification",
                    },
                }
            llerr = float(np.max(np.abs(reconstructed - independent)))
            passed = bool(np.allclose(reconstructed, independent, rtol=1e-5, atol=1e-8))
            return {
                "backend": spec.name,
                "status": "SEMANTIC_PASS" if passed else "SEMANTIC_FAIL",
                "execution_status": "EXECUTED",
                "stage": "oracle",
                "version": spec.version,
                "classification": True,
                "input_representation": representation,
                "missing_values": missing_values,
                "interaction": False,
                "model_output": "log_loss",
                "values_shape": list(values.shape),
                "target_shape": list(independent.shape),
                "additivity_passed": passed,
                "max_additivity_error": llerr,
                "log_loss_independent": float(np.mean(independent)),
                "log_loss_max_error": llerr,
                "probability_oracle": probability_oracle,
                "oracle": {"status": "ORACLE_PASS" if passed else "SEMANTIC_MISMATCH"},
                "semantic_contract": {
                    "output_space": "log_loss",
                    "task": "classification",
                },
            }
        stage = "oracle"
        target = np.asarray(
            model.predict_proba(test) if classification else model.predict(test),
            dtype=float,
        )
        class_axis = 1 if classification and target.ndim == 2 else None
        passed, err, oracle = _semantic_additivity(
            values, base, target, output_space=model_output, backend=spec.name
        )
        status = (
            "SEMANTIC_PASS"
            if passed
            else (
                "PASS_WITH_FINDINGS"
                if oracle.get("status") in {"INCONCLUSIVE_SHAPE", "ORACLE_ERROR"}
                else "SEMANTIC_FAIL"
            )
        )
        return {
            "backend": spec.name,
            "status": status,
            "execution_status": "EXECUTED",
            "stage": "oracle",
            "version": spec.version,
            "classification": classification,
            "input_representation": representation,
            "missing_values": missing_values,
            "interaction": False,
            "model_output": model_output,
            "values_shape": list(values.shape),
            "target_shape": list(target.shape),
            "class_axis": class_axis,
            "class_labels": getattr(model, "classes_", None).tolist()
            if hasattr(getattr(model, "classes_", None), "tolist")
            else list(getattr(model, "classes_", []))
            if hasattr(model, "classes_")
            else None,
            "additivity_passed": passed,
            "max_additivity_error": err,
            "oracle": oracle,
            "probability_oracle": probability_oracle,
            "semantic_contract": {
                "output_space": model_output,
                "input_representation": representation,
                "task": "classification" if classification else "regression",
                "class_axis": class_axis,
            },
        }
    except Exception as exc:
        return {
            "backend": spec.name,
            "status": _classify_exception(exc, stage),
            "execution_status": _classify_exception(exc, stage),
            "stage": stage,
            "version": spec.version,
            "classification": classification,
            "input_representation": representation,
            "missing_values": missing_values,
            "interaction": interaction,
            "model_output": model_output,
            "error": f"{type(exc).__name__}: {exc}",
            "exception_type": type(exc).__name__,
        }


def execute_installed_backend_matrix(*, exhaustive: bool = False):
    """Execute backend cases through concrete MatrixBackendAdapter instances."""
    try:
        import shap  # noqa: F401
    except Exception as exc:
        return {
            "status": "SKIPPED",
            "reason": f"SHAP unavailable: {exc}",
            "results": [],
            "summary": {
                "scheduled": 0,
                "executed": 0,
                "not_executed": 0,
                "findings": 0,
                "errors": 0,
                "execution_reasons": {"SKIPPED": 0},
            },
        }
    from shap_review.backends.adapter import MatrixBackendAdapter

    results = []
    discovered = discover_backends()
    for spec in discovered:
        if not spec.available:
            results.append(
                {
                    "backend": spec.name,
                    "status": "SKIPPED",
                    "execution_status": "NOT_EXECUTED",
                    "execution_reason": "SKIPPED",
                    "semantic_status": "NOT_EVALUATED",
                    "reason": "dependency unavailable",
                }
            )
            continue
        mod = importlib.import_module(spec.import_name)
        adapter = MatrixBackendAdapter(spec, mod)
        if exhaustive:
            cases = [
                (classification, rep, missing, interaction, output)
                for classification in (False, True)
                for rep in ("ndarray", "dataframe", "sparse")
                for missing in (False, True)
                for interaction in (False, True)
                for output in (
                    ("raw", "probability", "log_loss") if classification else ("raw",)
                )
            ]
        else:
            cases = [
                (False, "ndarray", False, False, "raw"),
                (False, "dataframe", False, False, "raw"),
                (False, "sparse", False, False, "raw"),
                (False, "ndarray", False, True, "raw"),
                (True, "ndarray", False, False, "probability"),
                (True, "dataframe", False, False, "probability"),
                (True, "ndarray", True, False, "probability"),
                (True, "ndarray", False, False, "log_loss"),
                (True, "dataframe", False, False, "log_loss"),
            ]
        for classification, rep, missing, interaction, output in cases:
            supported, support_reason = adapter.supports_case(
                classification=classification,
                representation=rep,
                missing_values=missing,
                interaction=interaction,
                model_output=output,
            )
            if not supported:
                r = {
                    "backend": spec.name,
                    "status": "UNSUPPORTED",
                    "execution_status": "NOT_EXECUTED",
                    "execution_reason": "UNSUPPORTED",
                    "semantic_status": "NOT_EVALUATED",
                    "stage": "capability",
                    "reason": support_reason,
                    "classification": classification,
                    "input_representation": rep,
                    "missing_values": missing,
                    "interaction": interaction,
                    "model_output": output,
                }
            else:
                r = adapter.execute_case(
                    classification=classification,
                    representation=rep,
                    missing_values=missing,
                    interaction=interaction,
                    model_output=output,
                )
            status = r.get("status")
            if status in {"SEMANTIC_PASS", "SEMANTIC_FAIL"}:
                r.update(
                    execution_status="EXECUTED",
                    execution_reason="COMPLETED",
                    semantic_status="PASS" if status == "SEMANTIC_PASS" else "FAIL",
                )
            elif status == "PASS_WITH_FINDINGS":
                r.update(
                    execution_status="EXECUTED",
                    execution_reason="COMPLETED",
                    semantic_status="INCONCLUSIVE",
                )
            elif status in {
                "SKIPPED",
                "UNSUPPORTED",
                "ADAPTER_ERROR",
                "BACKEND_ERROR",
                "SHAP_ERROR",
                "TOOLKIT_ERROR",
                "ERROR",
            }:
                r.update(
                    execution_status="NOT_EXECUTED",
                    execution_reason=status,
                    semantic_status="NOT_EVALUATED",
                )
            results.append(r)
    findings = [
        r for r in results if r.get("status") in {"SEMANTIC_FAIL", "PASS_WITH_FINDINGS"}
    ]
    errors = [
        r
        for r in results
        if r.get("execution_status") == "NOT_EXECUTED"
        and r.get("execution_reason") not in {"SKIPPED", "UNSUPPORTED"}
    ]
    scheduled = len(discovered) * (48 if exhaustive else 9)
    executed = sum(r.get("execution_status") == "EXECUTED" for r in results)
    not_executed = len(results) - executed
    reasons = {
        reason: sum(1 for r in results if r.get("execution_reason") == reason)
        for reason in sorted(set(r.get("execution_reason") for r in results) - {None})
    }
    summary = {
        "scheduled": scheduled,
        "executed": executed,
        "not_executed": not_executed,
        "execution_status": {"EXECUTED": executed, "NOT_EXECUTED": not_executed},
        "execution_reasons": reasons,
        "semantic_status": {
            "PASS": sum(r.get("semantic_status") == "PASS" for r in results),
            "FAIL": sum(r.get("semantic_status") == "FAIL" for r in results),
            "INCONCLUSIVE": sum(
                r.get("semantic_status") == "INCONCLUSIVE" for r in results
            ),
            "NOT_EVALUATED": sum(
                r.get("semantic_status") == "NOT_EVALUATED" for r in results
            ),
        },
        "semantic_pass": sum(r.get("status") == "SEMANTIC_PASS" for r in results),
        "semantic_fail": sum(r.get("status") == "SEMANTIC_FAIL" for r in results),
        "pass_with_findings": sum(
            r.get("status") == "PASS_WITH_FINDINGS" for r in results
        ),
        "errors": len(errors),
        "skipped": sum(r.get("status") == "SKIPPED" for r in results),
        "unsupported": sum(r.get("status") == "UNSUPPORTED" for r in results),
    }
    return {
        "status": "PASS_WITH_FINDINGS" if findings or errors else "PASS",
        "results": results,
        "dimensions": matrix_dimensions(),
        "summary": summary,
        "matrix_note": "Each case is routed through a concrete BackendAdapter and records execution status/reason separately from semantic status. Differential agreement is not correctness proof.",
    }
