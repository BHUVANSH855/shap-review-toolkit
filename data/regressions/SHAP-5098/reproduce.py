"""Standalone reproducer for SHAP issue #5098.

Pre-built TreeEnsemble must propagate the requested model_output correctly.

Expected: explainer.model_output == "probability" after construction with
          model_output="probability" on a pre-built TreeEnsemble.
Observed: model_output may not propagate correctly on affected versions.
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
import sys

import numpy as np


def _versions() -> dict:
    out = {}
    for pkg in ("shap", "numpy", "scikit-learn"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


result: dict = {
    "issue": "SHAP-5098",
    "environment": {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _versions(),
    },
    "reproduced": False,
    "status": "blocked",
    "note": "",
}

try:
    from shap.explainers._tree import TreeEnsemble, TreeExplainer

    def make_tree_dict():
        return {
            "objective": "binary_crossentropy",
            "tree_output": "log_odds",
            "trees": [
                {
                    "children_left": np.array([1, -1, -1], dtype=np.int32),
                    "children_right": np.array([2, -1, -1], dtype=np.int32),
                    "children_default": np.array([1, -1, -1], dtype=np.int32),
                    "features": np.array([0, -2, -2], dtype=np.int32),
                    "thresholds": np.array([0.5, -2, -2], dtype=np.float64),
                    "values": np.array([[0.0], [0.2], [0.8]], dtype=np.float64),
                    "node_sample_weight": np.array([100.0, 40.0, 60.0]),
                }
            ],
        }

    prebuilt = TreeEnsemble(make_tree_dict())
    explainer = TreeExplainer(model=prebuilt, model_output="probability")
    actual_output = explainer.model_output
    model_output = getattr(explainer.model, "model_output", None)
    reproduced = actual_output != "probability"
    result.update({
        "reproduced": reproduced,
        "status": "reproduced" if reproduced else "not_reproduced",
        "explainer_model_output": actual_output,
        "model_model_output": model_output,
        "note": (
            "model_output was not propagated correctly"
            if reproduced
            else "model_output propagated correctly on this version"
        ),
    })
except ImportError as exc:
    result.update({
        "status": "blocked",
        "note": f"TreeEnsemble API unavailable on this SHAP version: {exc}",
    })
except Exception as exc:  # noqa: BLE001
    result.update({
        "status": "blocked",
        "note": f"{type(exc).__name__}: {exc}",
    })

print(json.dumps(result, indent=2))