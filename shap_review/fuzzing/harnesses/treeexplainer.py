from __future__ import annotations

import traceback

import numpy as np


def dependency_available() -> bool:
    try:
        import shap  # noqa: F401

        return True
    except ImportError:
        return False


def _make_input(rng, n: int, f: int, dtype, variant: str):
    """Generate input data according to *variant*."""
    if variant == "zeros":
        X = np.zeros((n, f), dtype=dtype)
    elif variant == "zero_variance":
        X = np.ones((n, f), dtype=dtype) * rng.random()
    elif variant == "large":
        X = rng.normal(size=(n, f)).astype(dtype) * 1000.0
    elif variant == "int_as_float":
        X = rng.integers(-10, 10, size=(n, f)).astype(dtype)
    elif variant == "nan_row":
        X = rng.normal(size=(n, f)).astype(dtype)
        if n > 0:
            X[0, :] = np.nan
    elif variant == "mixed_nan":
        X = rng.normal(size=(n, f)).astype(dtype)
        if f > 0 and n > 0:
            mask = rng.random(size=(n, f)) < 0.1
            X[mask] = np.nan
    else:
        # normal — standard baseline
        X = rng.normal(size=(n, f)).astype(dtype)
    return X


def _arrays(case):
    import pandas as pd

    rng = np.random.default_rng(case.get("seed", 0))
    dtype = np.float32 if case.get("dtype") == "float32" else np.float64
    n = max(12, case["n_samples"] * 4)
    f = case["n_features"]
    variant = case.get("input_variant", "normal")
    backend = case.get("backend", "sklearn")

    X = _make_input(rng, n, f, dtype, variant)
    clean = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)

    y = (
        (clean.sum(axis=1) > 0).astype(int)
        if case.get("classification")
        else clean.sum(axis=1) + rng.normal(scale=0.01, size=n)
    )

    columns = [f"f{i}" for i in range(f)]
    train = clean

    if case.get("representation") == "dataframe":
        train = pd.DataFrame(train, columns=columns)

    model = _build_model(backend, case, train, y)

    sample = X[: max(1, min(case["n_samples"], len(X)))]
    background = clean[: min(8, len(X))]

    if case.get("nullable_dtype") and f > 0:
        # Reproduce issue #4911: nullable Int64 column in background DataFrame.
        bg_df = pd.DataFrame(background, columns=columns)
        bg_df[columns[0]] = pd.array(
            [int(v) if not np.isnan(v) else pd.NA for v in bg_df[columns[0]]],
            dtype="Int64",
        )
        background = bg_df
    elif case.get("background_representation") == "dataframe":
        background = pd.DataFrame(background, columns=columns)

    if case.get("representation") == "dataframe":
        sample = pd.DataFrame(sample, columns=columns)

    return model, sample, background


def _build_model(backend: str, case: dict, train, y):
    """Build a tree model for the requested backend."""
    classification = case.get("classification", False)
    n_trees = case["n_trees"]
    depth = case["depth"]
    seed = case.get("seed", 0)

    if backend == "xgboost":
        import xgboost as xgb

        params = {
            "n_estimators": n_trees,
            "max_depth": depth,
            "random_state": seed,
            "verbosity": 0,
        }
        if classification:
            model = xgb.XGBClassifier(
                **params, use_label_encoder=False, eval_metric="logloss"
            )
        else:
            model = xgb.XGBRegressor(**params)
        import pandas as pd

        train_arr = train.values if isinstance(train, pd.DataFrame) else train
        return model.fit(train_arr, y)

    if backend == "lightgbm":
        import lightgbm as lgb

        params = {
            "n_estimators": n_trees,
            "max_depth": depth,
            "random_state": seed,
            "verbosity": -1,
        }
        if classification:
            model = lgb.LGBMClassifier(**params)
        else:
            model = lgb.LGBMRegressor(**params)
        import pandas as pd

        train_arr = train.values if isinstance(train, pd.DataFrame) else train
        return model.fit(train_arr, y)

    if backend == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        params = {
            "iterations": n_trees,
            "depth": min(depth, 8),
            "random_seed": seed,
            "verbose": False,
        }
        import pandas as pd

        train_arr = train.values if isinstance(train, pd.DataFrame) else train
        if classification:
            return CatBoostClassifier(**params).fit(train_arr, y)
        return CatBoostRegressor(**params).fit(train_arr, y)

    # Default: sklearn RandomForest
    if classification:
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=n_trees, max_depth=depth, random_state=seed
        ).fit(train, y)
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=n_trees, max_depth=depth, random_state=seed
    ).fit(train, y)


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

    backend = case.get("backend", "sklearn")

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

        kwargs: dict = {}

        if classification and model_output == "probability":
            kwargs.update(
                model_output="probability",
                feature_perturbation="interventional",
                data=background,
            )
        else:
            if not interaction:
                kwargs["data"] = background

        explainer = shap.TreeExplainer(model, **kwargs)
        import pandas as pd

        before = (
            sample.copy()
            if isinstance(sample, pd.DataFrame)
            else np.asarray(sample).copy()
        )

        if interaction:
            values = explainer.shap_interaction_values(sample)
            interaction_executed = True
            base = np.asarray(explainer.expected_value)
            target = np.asarray(model.predict(sample))
        else:
            values = explainer.shap_values(sample, check_additivity=True)
            interaction_executed = False
            base = np.asarray(explainer.expected_value)
            if classification:
                if hasattr(model, "predict_proba"):
                    s = sample.values if isinstance(sample, pd.DataFrame) else sample
                    target = np.asarray(model.predict_proba(s))
                else:
                    target = np.asarray(model.predict(sample))
            else:
                target = np.asarray(model.predict(sample))

        if isinstance(values, list):
            values = np.stack([np.asarray(v) for v in values], axis=-1)

        arr = np.asarray(values)

        recon, tensor = _canonical_reconstruction(
            values,
            base,
            interaction=interaction,
            model_output=model_output,
        )

        target_for_compare = target
        shape_compatible = recon.shape == target_for_compare.shape

        err = (
            float(np.max(np.abs(recon - target_for_compare)))
            if shape_compatible and recon.size
            else (0.0 if shape_compatible else float("inf"))
        )

        tol = 5e-4 if case.get("dtype") == "float32" else 1e-5

        after = (
            sample.copy()
            if isinstance(sample, pd.DataFrame)
            else np.asarray(sample).copy()
        )
        if isinstance(before, pd.DataFrame):
            input_unchanged = before.equals(after)
        else:
            input_unchanged = bool(
                np.array_equal(np.asarray(before), np.asarray(after), equal_nan=True)
            )

        return {
            "executed": True,
            "backend": backend,
            "input_variant": case.get("input_variant", "normal"),
            "shap_version": getattr(shap, "__version__", ""),
            "shape": list(arr.shape),
            "prediction_shape": list(target_for_compare.shape),
            "input_unchanged": input_unchanged,
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
    except Exception as exc:  # noqa: BLE001
        return {
            "executed": True,
            "failed": True,
            "backend": backend,
            "input_variant": case.get("input_variant", "normal"),
            "exception": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=10),
            "case": case,
        }
