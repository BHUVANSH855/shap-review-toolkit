from __future__ import annotations


def invalid_shape(case: dict) -> bool:
    return case.get("samples", 1) == 0 or case.get("features", 1) == 0


def suspicious_tree_case(case: dict) -> bool:
    return case.get("depth", 0) >= 10 and case.get("n_features", 0) >= 50
