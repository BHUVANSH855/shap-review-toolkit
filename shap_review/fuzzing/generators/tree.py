from __future__ import annotations

import random


def generate_tree_case(rng: random.Random) -> dict:
    classification = rng.choice([False, True])
    model_output = "raw"
    if classification and rng.choice([False, True]):
        model_output = "probability"
    interaction = (not classification) and rng.choice([False, True])
    return {
        "n_features": rng.choice([1, 2, 3, 5, 10, 20]),
        "depth": rng.choice([1, 2, 3, 5, 8]),
        "n_trees": rng.choice([1, 2, 5, 10]),
        "n_samples": rng.choice([1, 2, 5, 10, 25]),
        "dtype": rng.choice(["float32", "float64"]),
        "nan": rng.choice([False, False, True]),
        "representation": rng.choice(["ndarray", "dataframe"]),
        "interaction": interaction,
        "model_output": model_output,
        "classification": classification,
        "background_representation": rng.choice(["ndarray", "dataframe"]),
    }
