from __future__ import annotations

import random


def generate_input_case(rng: random.Random) -> dict:
    """Generate only dimensions that the TreeExplainer harness executes."""
    return {
        "samples": rng.choice([1, 2, 5, 10, 25]),
        "features": rng.choice([1, 2, 3, 5, 10]),
        "dtype": rng.choice(["float32", "float64"]),
        "representation": rng.choice(["ndarray", "dataframe"]),
        "nan": rng.choice([False, False, True]),
    }
