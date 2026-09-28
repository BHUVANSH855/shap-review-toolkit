from __future__ import annotations

from pathlib import Path

from shap_review.engine import ReviewEngine
from shap_review.fuzzing.engine import TreeExplainerFuzzer


def validate(root: str | Path, iterations: int = 10):
    root = Path(root)
    engine = ReviewEngine()
    discovery = engine.discover(root)
    candidates = engine.analyze(root)
    fuzz = TreeExplainerFuzzer(seed=0).run(iterations)
    return {"discovery": discovery, "candidates": len(candidates), "fuzz": fuzz}
