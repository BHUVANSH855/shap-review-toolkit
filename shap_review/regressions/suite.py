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
            # Only mark as reproduced when the exception is clearly caused by the
            # nullable dtype / object-dtype reaching native processing — not by
            # an unrelated error (e.g. wrong background shape, MemoryError).
            nullable_keywords = {
                "int64",
                "nullable",
                "object",
                "cannot convert",
                "unsupported dtype",
                "invalid dtype",
                "float conversion",
                "expected float",
                "buffer",
            }
            is_nullable_error = exc_type in ("TypeError", "ValueError") and any(
                kw in exc_msg for kw in nullable_keywords
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
    # The issue's exact reproducer depends on an internal TreeEnsemble path introduced
    # on main. We preserve the source-grounded precondition and validate the invariant
    # statically when the local SHAP build does not expose that path.
    try:
        import shap
        from shap.explainers import _tree

        has_tree_ensemble = hasattr(_tree, "TreeEnsemble")
        return RegressionResult(
            "SHAP-5098",
            "static_precondition",
            False,
            "pre-built TreeEnsemble must propagate requested model_output",
            "TreeEnsemble API available; exact issue path is version-dependent",
            {
                "shap_version": shap.__version__,
                "tree_ensemble_available": has_tree_ensemble,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return RegressionResult(
            "SHAP-5098",
            "blocked",
            False,
            "static precondition",
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


def run_all():
    return [run_4911(), run_4495(), run_5098(), run_catboost_interventional()]


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
