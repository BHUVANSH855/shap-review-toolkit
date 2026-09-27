from __future__ import annotations


def minimize_case(case: dict) -> dict:
    out = dict(case)
    for key in ["n_trees", "depth", "n_features", "n_samples", "samples", "features"]:
        if key in out and isinstance(out[key], int):
            out[key] = max(1, min(out[key], 3))
    return out
