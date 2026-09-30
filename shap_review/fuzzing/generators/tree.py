from __future__ import annotations

import random


def _backend_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def _available_backends() -> list[str]:
    """Return backends installed in the current environment."""
    backends = ["sklearn"]
    for name in ("xgboost", "lightgbm", "catboost"):
        if _backend_available(name):
            backends.append(name)
    return backends


def generate_tree_case(rng: random.Random) -> dict:
    classification = rng.choice([False, True])
    model_output = "raw"
    if classification and rng.choice([False, True]):
        model_output = "probability"
    interaction = (not classification) and rng.choice([False, True, False])

    # Backend: prefer sklearn always; include others when available.
    available = _available_backends()
    backend = rng.choice(available)

    # Interaction values only supported for sklearn in current harness.
    if backend != "sklearn":
        interaction = False

    # Input variant: normal distribution is baseline; add boundary cases.
    input_variant = rng.choice(
        [
            "normal",  # standard normal — baseline
            "normal",  # weighted to appear more often
            "normal",
            "zeros",  # all-zero features — edge case for tree splits
            "zero_variance",  # single repeated value per feature
            "large",  # values scaled to ±1000 — overflow/precision check
            "int_as_float",  # integer values cast to float — dtype edge case
            "nan_row",  # entire first row is NaN
            "mixed_nan",  # random NaN scatter
        ]
    )

    # DataFrame nullable dtype variant (pandas Int64 — issue #4911).
    nullable_dtype = (
        rng.choice([False, False, False, True])
        and rng.choice(["ndarray", "dataframe"]) == "dataframe"
        and backend == "sklearn"
    )

    return {
        "n_features": rng.choice([1, 2, 3, 5, 10, 20]),
        "depth": rng.choice([1, 2, 3, 5, 8]),
        "n_trees": rng.choice([1, 2, 5, 10]),
        "n_samples": rng.choice([1, 2, 5, 10, 25]),
        "dtype": rng.choice(["float32", "float64", "float64", "float64"]),
        "nan": rng.choice([False, False, True]),
        "representation": rng.choice(["ndarray", "dataframe"]),
        "background_representation": rng.choice(["ndarray", "dataframe"]),
        "interaction": interaction,
        "model_output": model_output,
        "classification": classification,
        "backend": backend,
        "input_variant": input_variant,
        "nullable_dtype": nullable_dtype,
    }
