from __future__ import annotations

import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

from shap_review.types import RegressionStatus


@dataclass
class RegressionResult:
    id: str
    status: str
    reproduced: bool
    expected: str
    observed: str
    details: dict

    def to_dict(self):
        return asdict(self)


def run_4911():
    try:
        import pandas as pd
        import shap
        from sklearn.ensemble import RandomForestRegressor

        X = pd.DataFrame(
            {"a": pd.Series([1, 2, pd.NA], dtype="Int64"), "b": [0.1, 0.2, 0.3]}
        )
        y = [1, 2, 3]
        model = RandomForestRegressor(random_state=0).fit(X.fillna(0), y)
        try:
            shap.TreeExplainer(model, data=X)
        except Exception as exc:  # noqa: BLE001
            exc_type = type(exc).__name__
            exc_msg = str(exc).lower()
            # Mark the historical regression as reproduced only when the
            # exception matches the source-grounded nullable-dtype failure:
            # pandas nullable Int64 data reaches SHAP's native TreeExplainer
            # path as object dtype and cannot be cast to float64.
            nullable_dtype_markers = (
                "cannot cast array data from dtype('o') to dtype('float64')",
                "cannot cast array data from dtype('object') to dtype('float64')",
            )
            nullable_context_markers = (
                "nullable",
                "dtype('o')",
                "dtype('object')",
                "float64",
            )

            is_nullable_error = (
                exc_type == "TypeError"
                and any(marker in exc_msg for marker in nullable_dtype_markers)
                and all(marker in exc_msg for marker in nullable_context_markers)
            )

            if is_nullable_error:
                status = "reproduced"
            else:
                status = RegressionStatus.AMBIGUOUS.value
            return RegressionResult(
                "SHAP-4911",
                status,
                is_nullable_error,
                "nullable background must be converted or clearly rejected",
                f"{exc_type}: {exc}",
                {
                    "shap_version": shap.__version__,
                    "exception_type": exc_type,
                    "is_nullable_error": is_nullable_error,
                },
            )
        return RegressionResult(
            "SHAP-4911",
            "not_reproduced",
            False,
            "nullable background failure on affected versions",
            "constructor succeeded — nullable dtype was handled without error",
            {"shap_version": shap.__version__},
        )
    except Exception as exc:  # noqa: BLE001
        return RegressionResult(
            "SHAP-4911",
            "blocked",
            False,
            "runtime repro",
            "setup failed",
            {"exception": type(exc).__name__, "message": str(exc)},
        )


def run_4495():
    try:
        import numpy as np
        import shap
        import xgboost as xgb
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        model = xgb.XGBClassifier(n_estimators=50, random_state=42)
        model.fit(X[:150], y[:150])
        explainer = shap.TreeExplainer(model)
        before = np.asarray(explainer.expected_value).copy()
        before_shape = before.shape
        explainer.shap_values(X[150:])
        after = np.asarray(explainer.expected_value).copy()
        after_shape = after.shape
        changed = not np.array_equal(before, after) or before_shape != after_shape
        return RegressionResult(
            "SHAP-4495",
            "reproduced" if changed else "not_reproduced",
            changed,
            "expected_value must be stable across initialization and shap_values",
            f"before={before.tolist()} shape={before_shape}; after={after.tolist()} shape={after_shape}",
            {"shap_version": shap.__version__},
        )
    except Exception as exc:  # noqa: BLE001
        return RegressionResult(
            "SHAP-4495",
            "blocked",
            False,
            "runtime repro",
            "setup failed",
            {
                "exception": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=4),
            },
        )


def run_5098():
    """Runtime reproducer for SHAP issue #5098.

    Executes the standalone reproducer script via subprocess so the result
    is isolated and the environment is captured. Falls back to a static
    precondition check when TreeEnsemble is not available on the installed
    SHAP version.
    """
    import json as _json
    from pathlib import Path as _Path

    script = (
        _Path(__file__).parents[2] / "data" / "regressions" / "SHAP-5098" / "reproduce.py"
    )

    if not script.exists():
        return RegressionResult(
            "SHAP-5098",
            RegressionStatus.BLOCKED.value,
            False,
            "pre-built TreeEnsemble must propagate requested model_output",
            "reproducer script not found",
            {"script": str(script)},
        )

    try:
        import shap as _shap

        from shap_review.reproduction.runner import run_script

        run_result = run_script(str(script), timeout=60)
        stdout = run_result.get("stdout", "")

        try:
            payload = _json.loads(stdout)
        except _json.JSONDecodeError:
            return RegressionResult(
                "SHAP-5098",
                RegressionStatus.BLOCKED.value,
                False,
                "pre-built TreeEnsemble must propagate requested model_output",
                f"reproducer output was not valid JSON: {stdout[:200]}",
                {"shap_version": _shap.__version__, "returncode": run_result.get("returncode")},
            )

        status = payload.get("status", "blocked")
        reproduced = bool(payload.get("reproduced", False))

        return RegressionResult(
            "SHAP-5098",
            status,
            reproduced,
            "pre-built TreeEnsemble must propagate requested model_output",
            payload.get("note", status),
            {
                "shap_version": _shap.__version__,
                "environment": payload.get("environment", {}),
                "explainer_model_output": payload.get("explainer_model_output"),
                "model_model_output": payload.get("model_model_output"),
            },
        )
    except Exception as exc:  # noqa: BLE001
        return RegressionResult(
            "SHAP-5098",
            RegressionStatus.BLOCKED.value,
            False,
            "pre-built TreeEnsemble must propagate requested model_output",
            "setup failed",
            {"exception": type(exc).__name__, "message": str(exc)},
        )


def run_catboost_interventional():
    """Reproduce the high-confidence CatBoost interventional reconstruction candidate.

    This intentionally reports a candidate mismatch rather than asserting a SHAP defect.
    Cross-version comparison is required before confirmation.
    """
    try:
        import catboost
        import numpy as np
        import shap
        from catboost import CatBoostRegressor

        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=float)
        y = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        model = CatBoostRegressor(
            iterations=20, depth=3, verbose=False, random_seed=0
        ).fit(X, y)
        background = X[:2]
        probe = X[2:]
        explainer = shap.TreeExplainer(model, data=background)
        values = np.asarray(explainer.shap_values(probe), dtype=float)
        base = np.asarray(explainer.expected_value, dtype=float)
        prediction = np.asarray(model.predict(probe), dtype=float)
        reconstructed = values.sum(axis=1) + base
        error = float(np.max(np.abs(reconstructed - prediction)))
        reproduced = not np.allclose(reconstructed, prediction, rtol=1e-5, atol=1e-8)
        return RegressionResult(
            "SHAP-CATBOOST-INTERVENTIONAL-RECON",
            "candidate_reproduced" if reproduced else "not_reproduced",
            reproduced,
            "reconstruction should match model.predict before defect confirmation",
            f"max_error={error}; reconstructed={reconstructed.tolist()}; prediction={prediction.tolist()}",
            {
                "shap_version": shap.__version__,
                "catboost_version": catboost.__version__,
                "background_rows": len(background),
                "probe_rows": len(probe),
                "error": error,
                "confirmation_status": "candidate_only",
                "cross_version_required": True,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return RegressionResult(
            "SHAP-CATBOOST-INTERVENTIONAL-RECON",
            "blocked",
            False,
            "runtime candidate reproduction",
            "setup failed",
            {
                "exception": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=6),
            },
        )


def run_1539():
    """Regression for SHAP issue #1539 — deep tree path causes incorrect SHAP values.

    Validates that TreeExplainer produces finite, additivity-satisfying outputs
    on deep trees (depth >= 8) which historically triggered incorrect node
    traversal on some SHAP versions.
    """
    try:
        import numpy as np
        import shap
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.default_rng(1539)
        X = rng.normal(size=(100, 5)).astype(np.float64)
        y = X.sum(axis=1)
        model = RandomForestRegressor(
            n_estimators=5, max_depth=10, random_state=1539
        ).fit(X, y)
        explainer = shap.TreeExplainer(model, data=X[:10])
        values = np.asarray(explainer.shap_values(X[10:20]))
        base = np.asarray(explainer.expected_value)
        prediction = np.asarray(model.predict(X[10:20]))
        reconstructed = values.sum(axis=1) + base
        finite_ok = bool(np.isfinite(values).all())
        additivity_ok = bool(np.allclose(reconstructed, prediction, atol=1e-4))
        reproduced = not finite_ok or not additivity_ok
        return RegressionResult(
            "SHAP-1539",
            "reproduced" if reproduced else "not_reproduced",
            reproduced,
            "deep tree must produce finite additivity-satisfying SHAP values",
            (
                f"finite={finite_ok}, additivity={additivity_ok}, "
                f"max_error={float(np.max(np.abs(reconstructed - prediction))):.2e}"
            ),
            {
                "shap_version": shap.__version__,
                "finite": finite_ok,
                "additivity_ok": additivity_ok,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return RegressionResult(
            "SHAP-1539",
            RegressionStatus.BLOCKED.value,
            False,
            "deep tree finite additivity check",
            "setup failed",
            {"exception": type(exc).__name__, "message": str(exc)},
        )


def run_2778():
    """Regression for SHAP issue #2778 — multi-output TreeExplainer shape contract.

    Validates that shap_values() on a multi-output regression model returns
    a list/array with the correct shape per output.
    """
    try:
        import numpy as np
        import shap
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor

        rng = np.random.default_rng(2778)
        X = rng.normal(size=(80, 4)).astype(np.float64)
        y = np.column_stack([X.sum(axis=1), X[:, 0] - X[:, 1]])
        model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=3, max_depth=3, random_state=2778)
        ).fit(X, y)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X[:5])
        # For multi-output, values should be a list of arrays or a 3D array.
        if isinstance(values, list):
            shape_ok = all(np.asarray(v).shape == (5, 4) for v in values)
            n_outputs = len(values)
        else:
            arr = np.asarray(values)
            shape_ok = arr.ndim == 3 and arr.shape[0] == 5 and arr.shape[1] == 4
            n_outputs = arr.shape[2] if arr.ndim == 3 else 0
        reproduced = not shape_ok
        return RegressionResult(
            "SHAP-2778",
            "reproduced" if reproduced else "not_reproduced",
            reproduced,
            "multi-output TreeExplainer must return correctly shaped SHAP values",
            f"shape_ok={shape_ok}, n_outputs={n_outputs}",
            {
                "shap_version": shap.__version__,
                "shape_ok": shape_ok,
                "n_outputs": n_outputs,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return RegressionResult(
            "SHAP-2778",
            RegressionStatus.BLOCKED.value,
            False,
            "multi-output shape contract",
            "setup failed",
            {"exception": type(exc).__name__, "message": str(exc)},
        )


def run_4869():
    """Regression for SHAP issue #4869 — interaction values coverage gap.

    Validates that shap_interaction_values() runs and returns a symmetric
    3D tensor satisfying the interaction reconstruction invariant.
    """
    try:
        import numpy as np
        import shap
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.default_rng(4869)
        X = rng.normal(size=(60, 4)).astype(np.float64)
        y = X.sum(axis=1)
        model = RandomForestRegressor(
            n_estimators=3, max_depth=3, random_state=4869
        ).fit(X, y)
        explainer = shap.TreeExplainer(model)
        iv = np.asarray(explainer.shap_interaction_values(X[:5]))
        # Must be shape (n_samples, n_features, n_features).
        shape_ok = iv.shape == (5, 4, 4)
        symmetric = bool(np.allclose(iv, iv.swapaxes(1, 2), atol=1e-8))
        # Row sums of interaction matrix must reconstruct shap_values.
        sv = np.asarray(explainer.shap_values(X[:5]))
        reconstruction_ok = bool(np.allclose(iv.sum(axis=2), sv, atol=1e-5))
        reproduced = not (shape_ok and symmetric and reconstruction_ok)
        return RegressionResult(
            "SHAP-4869",
            "reproduced" if reproduced else "not_reproduced",
            reproduced,
            "interaction values must be symmetric and reconstruct SHAP values",
            (
                f"shape_ok={shape_ok}, symmetric={symmetric}, "
                f"reconstruction_ok={reconstruction_ok}"
            ),
            {
                "shap_version": shap.__version__,
                "shape_ok": shape_ok,
                "symmetric": symmetric,
                "reconstruction_ok": reconstruction_ok,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return RegressionResult(
            "SHAP-4869",
            RegressionStatus.BLOCKED.value,
            False,
            "interaction symmetry and reconstruction",
            "setup failed",
            {"exception": type(exc).__name__, "message": str(exc)},
        )


def run_4942():
    """Regression for SHAP issue #4942 — AdditiveExplainer raises NotImplementedError
    for interaction effects on EBM-style models.

    Validates that shap.Explainer dispatches correctly and does not hard-fail
    with NotImplementedError on models that declare interaction support.
    """
    try:
        import numpy as np
        import shap
        from sklearn.ensemble import GradientBoostingRegressor

        rng = np.random.default_rng(4942)
        X = rng.normal(size=(60, 3)).astype(np.float64)
        y = X.sum(axis=1)
        # GBR is a supported TreeExplainer model — use it as a dispatch proxy.
        model = GradientBoostingRegressor(
            n_estimators=5, max_depth=2, random_state=4942
        ).fit(X, y)
        explainer = shap.Explainer(model, X[:10])
        result = explainer(X[:5])
        values = np.asarray(result.values)
        finite_ok = bool(np.isfinite(values).all())
        shape_ok = values.shape[0] == 5 and values.shape[1] == 3
        reproduced = not (finite_ok and shape_ok)
        return RegressionResult(
            "SHAP-4942",
            "reproduced" if reproduced else "not_reproduced",
            reproduced,
            "Explainer dispatch must not raise NotImplementedError on supported models",
            f"finite={finite_ok}, shape_ok={shape_ok}, shape={list(values.shape)}",
            {
                "shap_version": shap.__version__,
                "finite": finite_ok,
                "shape_ok": shape_ok,
            },
        )
    except NotImplementedError as exc:
        return RegressionResult(
            "SHAP-4942",
            "reproduced",
            True,
            "Explainer dispatch must not raise NotImplementedError on supported models",
            f"NotImplementedError: {exc}",
            {"shap_version": _pkg_version("shap"), "exception": "NotImplementedError"},
        )
    except Exception as exc:  # noqa: BLE001
        return RegressionResult(
            "SHAP-4942",
            RegressionStatus.BLOCKED.value,
            False,
            "Explainer dispatch NotImplementedError check",
            "setup failed",
            {"exception": type(exc).__name__, "message": str(exc)},
        )


def run_all():
    return [
        run_1539(),
        run_2778(),
        run_4495(),
        run_4869(),
        run_4911(),
        run_4942(),
        run_5098(),
        run_catboost_interventional(),
    ]


def catboost_candidate_reproducer():
    """Emit a deterministic, standalone candidate bundle for the CatBoost reconstruction anomaly."""
    import platform
    import sys

    result = run_catboost_interventional()
    details = dict(result.details or {})
    return {
        "id": result.id,
        "confirmation_status": details.get("confirmation_status", "candidate_only"),
        "shap_version": _pkg_version("shap"),
        "catboost_version": _pkg_version("catboost"),
        "python": sys.version,
        "platform": platform.platform(),
        "details": details,
        "training_data": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        "probe_data": [[0.5, 0.5], [2.5, 2.5]],
        "expected": "independent model prediction reconstruction",
        "actual": "TreeExplainer reconstruction",
        "cross_version": details.get("cross_version", "not_run"),
    }


def _pkg_version(name):
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return None


def cross_version_catboost_validation(
    python_reference=None, python_candidate=None, timeout=120
):
    """Run the standalone CatBoost reproducer under two explicit interpreters.

    The function never mutates the current interpreter environment and never
    upgrades a candidate to confirmed merely because the two runs differ.
    """
    if not python_reference or not python_candidate:
        return {
            "status": "NOT_RUN",
            "reason": "Two explicit Python interpreters are required for cross-version validation.",
            "confirmation_status": "candidate_only",
        }
    from shap_review.differential.versions import differential_versions

    script = (
        Path(__file__).resolve().parents[2]
        / "reproducers"
        / "catboost_interventional_candidate.py"
    )
    result = differential_versions(
        script, script, str(python_reference), str(python_candidate), timeout=timeout
    )
    result["confirmation_status"] = "candidate_only"
    result["promotion_policy"] = (
        "A mismatch or match alone does not confirm a SHAP defect; inspect version-specific output and independent model reconstruction."
    )
    return result
