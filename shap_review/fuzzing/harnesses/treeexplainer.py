from __future__ import annotations

import traceback

import numpy as np


def dependency_available() -> bool:
    try:
        import shap  # noqa: F401

        return True
    except Exception:
        return False


def _arrays(case):
    import pandas as pd

    rng = np.random.default_rng(case.get("seed", 0))
    dtype = np.float32 if case.get("dtype") == "float32" else np.float64
    n = max(12, case["n_samples"] * 4)
    f = case["n_features"]
    X = rng.normal(size=(n, f)).astype(dtype)
    if case.get("nan") and f:
        X[0, 0] = np.nan
    clean = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    y = (
        (clean.sum(axis=1) > 0).astype(int)
        if case.get("classification")
        else clean.sum(axis=1) + rng.normal(scale=0.01, size=n)
    )
    train = clean
    columns = [f"f{i}" for i in range(f)]
    if case.get("representation") == "dataframe":
        train = pd.DataFrame(train, columns=columns)
    if case.get("classification"):
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=case["n_trees"],
            max_depth=case["depth"],
            random_state=case.get("seed", 0),
        ).fit(train, y)
    else:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=case["n_trees"],
            max_depth=case["depth"],
            random_state=case.get("seed", 0),
        ).fit(train, y)
    sample = X[: max(1, min(case["n_samples"], len(X)))]
    background = clean[: min(8, len(X))]
    if case.get("representation") == "dataframe":
        sample = pd.DataFrame(sample, columns=columns)
    if case.get("background_representation") == "dataframe":
        background = pd.DataFrame(background, columns=columns)
    return model, sample, background


def _canonical_reconstruction(values, base, *, interaction=False, model_output="raw"):
    from shap_review.contracts.tensor import SHAPSemanticTensor, infer_axis_spec

    tensor = SHAPSemanticTensor.from_values(
        values,
        base_values=base,
        interaction=interaction,
        axis_spec=infer_axis_spec(np.asarray(values), interaction=interaction),
        output_space=model_output,
        source_api="TreeExplainer.shap_values",
    )
    return tensor.reconstruction(), tensor


def run_case(case: dict) -> dict:
    if not dependency_available():
        return {
            "executed": False,
            "reason": "SHAP/sklearn/pandas unavailable",
            "case": case,
        }
    try:
        import shap

        model, sample, background = _arrays(case)
        classification = case.get("classification", False)
        interaction = case.get("interaction", False)
        model_output = case.get("model_output", "raw")
        if interaction and (classification or model_output != "raw"):
            return {
                "executed": False,
                "skipped": True,
                "reason": "interaction contract requires regression/raw",
                "case": case,
            }
        kwargs = {}
        if classification and model_output == "probability":
            kwargs.update(
                model_output="probability",
                feature_perturbation="interventional",
                data=background,
            )
        else:
            # Interaction values require the tree-path-dependent contract in
            # current SHAP; passing an interventional background triggers a
            # documented unsupported path.
            if not interaction:
                kwargs["data"] = background
        explainer = shap.TreeExplainer(model, **kwargs)
        before = np.asarray(sample).copy()
        if interaction:
            values = explainer.shap_interaction_values(sample)
            interaction_executed = True
            base = np.asarray(explainer.expected_value)
            target = np.asarray(model.predict(sample))
        else:
            values = explainer.shap_values(sample, check_additivity=True)
            interaction_executed = False
            base = np.asarray(explainer.expected_value)
            target = np.asarray(
                model.predict_proba(sample) if classification else model.predict(sample)
            )
        if isinstance(values, list):
            values = np.stack([np.asarray(v) for v in values], axis=-1)
        arr = np.asarray(values)
        recon, tensor = _canonical_reconstruction(
            values, base, interaction=interaction, model_output=model_output
        )
        target_for_compare = target
        shape_compatible = recon.shape == target_for_compare.shape
        err = (
            float(np.max(np.abs(recon - target_for_compare)))
            if shape_compatible and recon.size
            else (0.0 if shape_compatible else float("inf"))
        )
        tol = 5e-4 if case.get("dtype") == "float32" else 1e-5
        return {
            "executed": True,
            "shap_version": getattr(shap, "__version__", ""),
            "shape": list(arr.shape),
            "prediction_shape": list(target_for_compare.shape),
            "input_unchanged": bool(
                np.array_equal(np.asarray(sample), before, equal_nan=True)
            ),
            "finite": bool(np.isfinite(arr).all()),
            "additivity_checked": True,
            "additivity_passed": bool(err <= tol),
            "interaction_executed": interaction_executed,
            "shape_compatible": shape_compatible,
            "max_additivity_error": err,
            "additivity_tolerance": tol,
            "reconstruction_shape": list(np.asarray(recon).shape),
            "target_shape": list(target_for_compare.shape),
            "axis_spec": tensor.axis_spec.__dict__,
            "case": case,
        }
    except Exception as exc:
        return {
            "executed": True,
            "failed": True,
            "exception": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=10),
            "case": case,
        }
